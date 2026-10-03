# Architecture

The runtime is two modules in `src/qwen38/`:

| Module | Contents |
| --- | --- |
| `chat.py` | Token and cache logic shared by both entry points: canonical prompt rendering, exact-suffix computation, history reconciliation, cache offset checks, and `generate_tokens`, the single direct `generate_step` loop. It also contains `ExactSuffixChat` and the `qwen38-chat` CLI. |
| `protocol.py` | Pure OpenAI Chat Completions layer: request validation into structured messages, the Qwen3.8 tool-call parser, transcript identity, and streaming assembly. No MLX import. |
| `server.py` | `AgentSession` (session bookkeeping, exact reuse vs fresh fallback, cancellation) and a stdlib `http.server` backend with JSON and SSE responses. |

MLX and mlx-lm are imported lazily (`chat._require_mlx`), so the protocol and
token tests run without a Metal device.

## Direct generation, no wired-limit wrapper

Generation calls the pinned `mlx_lm.generate.generate_step` directly with a
persistent `prompt_cache`. The stock `stream_generate` wrapper is not used.
At the pinned revision it raises MLX's wired limit to the device's
recommended working set, about 12 GiB on an 18 GB Mac. That caused a
reproducible Metal OOM during prefill. The runtime never calls
`mx.set_wired_limit()`, and the system `iogpu.wired_limit_mb` must stay
untouched.

## Hybrid cache

Qwen3.8-27B has 64 layers: 48 linear-attention (GatedDelta) layers and 16
full-attention layers. `make_prompt_cache` returns:

- 48 `ArraysCache` objects holding recurrent conv/SSM state;
- 16 `KVCache` objects holding attention keys and values.

Recurrent state is a running summary, so it cannot be trimmed back to an
earlier position. The cache can therefore only move forward. If the next
prompt is not an exact extension of the tokens already in the cache, the only
correct action is to discard the cache and prefill from scratch.

## Exact-suffix persistent continuation

Each turn, the runtime:

1. renders the full logical transcript with the tokenizer's chat template.
   It checks that the string form and the tokenized form agree, which gives
   the canonical token stream;
2. checks that every cache layer reports the same offset, and that this
   offset equals the length of the token ledger (`cached_token_ids`);
3. requires the ledger to be an exact prefix of the canonical stream, then
   feeds only the remaining suffix to `generate_step`;
4. after generation, checks that the cache offset equals
   `len(canonical) + len(generated)` and commits the new ledger.

A same-boundary oracle validated this design. A fresh segmented run and the
persistent run produced identical state at every boundary: all 48 recurrent
states, all 16 KV states, the offsets, the first-token logits and the
generated tokens. A single fresh prefill over the whole transcript can
diverge in floating point from a segmented run. That difference is normal
for this hybrid model, not a cache bug. It means "equivalent" here means
equivalent to the same segmentation.

## Authoritative generated token IDs

Decoded assistant text does not always re-render to the token IDs the model
generated. Two cases were measured:

- **Template trimming.** The Qwen assistant template applies `|trim`. A
  turn cut off by `max_tokens` right after a newline token (`198`)
  re-renders with `<|im_end|>` at that position. This was the original HTTP
  409 bug.
- **Partial UTF-8.** A length cut inside a multi-byte character decodes to
  U+FFFD. For example, `[9008, 239, 102, 373]` re-encodes as
  `[9008, 239, 102, 5513]`.

For every assistant generation, the session stores the exact generated IDs,
the decoded text, the finish reason and the prompt-token boundary. Before
computing the suffix, `reconcile_canonical_history` swaps each rendered
assistant span for its stored IDs and keeps a single EOS boundary. Spans are
replaced in order and in place, so each recorded boundary is already a cache
position; no offset is carried between spans. Tool-call turns make this
matter: the template re-renders a call's JSON values with `tojson`, which
rarely has the same length as the model's own spelling. The API
still returns the decoded text. Exact token identity is used only internally.

## Server request handling (`AgentSession.complete`)

### Structured messages

`protocol.parse_chat_request` turns an OpenAI request into the structure the
Qwen3.8 chat template renders: text content parts joined into strings
(images and other parts are rejected with 400), assistant `tool_calls` with
`arguments` parsed from their JSON string into a mapping (the template
iterates `arguments|items` and fails on a string), and `tool_call_id` kept on
tool results. `developer` is accepted as the first-message `system` role. The
template is the only renderer; the server adds no tool syntax of its own.

### Native tool format

The template (see `tests/test_tool_session.py`) renders:

- `tools` into the system block as one `tojson` line each, in request order
  and key order, followed by the format instructions;
- assistant calls as `qwen3_coder` XML:
  `<tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n</function>\n</tool_call>`,
  strings verbatim and other values as JSON, several calls per turn allowed;
- consecutive tool results as one user turn of `<tool_response>` blocks,
  matched by order. Call IDs never reach the prompt; the API layer assigns
  them (`call_<24 hex>`).

`<tool_call>`/`</tool_call>` are single added tokens (248058/248059), not
special, so they survive `skip_special_tokens` decoding. With thinking off
the generation prompt ends in an empty `<think>\n\n</think>\n\n`, and the
template's default `preserve_thinking` re-renders that same empty block for
every assistant turn, so transcripts stay exact extensions.

### Generated tokens are the truth; the API message is a view

After generation the decoded text is parsed by `parse_assistant_text`: text
before the first `<tool_call>` becomes `content` (right-trimmed), each call is
parsed strictly and its values converted with the declared JSON Schema type
(a value that does not fit stays a string and is counted as an argument type
mismatch, so the client's validator reports it to the model). Malformed or
truncated tool syntax is never turned into a call: the whole text becomes
`content`, `tool_parse_error` is recorded, and `finish_reason` is `stop` or
`length`. A parsed call gives `finish_reason: "tool_calls"`. Without tools in
effect (`tool_choice: "none"` or no `tools`) nothing is parsed.

The session stores both the generated IDs and the API view it returned.
Clients replay that view differently (Pi re-serializes `arguments`, sends
`null` content), so `protocol.messages_equal` compares assistant messages by
trimmed content and by tool-call id, name and parsed arguments. Equality only
identifies which stored generation a replayed message stands for. The
rendered span is then replaced by the stored IDs (`reconcile_canonical_history`),
exactly as for plain text turns. Cache correctness never depends on
re-serializing the parsed calls.

### Exact reuse and fresh fallback

`_plan` decides without touching any state:

- **Exact reuse**: the effective tool set is identical (`tools_identity`, the
  exact rendered JSON in order), the request strictly extends the stored
  transcript under `messages_equal`, the cache offset equals the ledger, and
  the reconciled canonical stream starts with the cached tokens. The stored
  copies of already-served messages are rendered, so assistant spans come
  from the server's own view before being replaced by their IDs.
- **Fresh fallback**: any of these fails (`cache_fallback_reason` says which:
  `tool_set_changed`, `transcript_does_not_extend_session`,
  `token_prefix_mismatch`). Changed, reordered, added or removed tools, or a
  switch to `tool_choice: "none"`, always land here. A cache miss is never an
  HTTP error.

Only after rendering does the server check `--max-context-tokens` (default
16384, prompt plus completion): a prompt that does not fit gets a 400
`context_length_exceeded` with OpenAI's wording, which Pi recognizes and
answers by compacting. The completion budget is clamped to what fits. A
rejected request leaves the session untouched.

### Streaming, heartbeats and cancellation

With `stream: true` the server validates and renders first (errors are plain
HTTP 4xx), then sends `text/event-stream` headers and a role chunk *before*
prefill, so a long cold prefill cannot trip the client's header timeout. A
heartbeat thread writes `: keep-alive` comments after 10 s of silence. Content
deltas come from `StreamAssembler`, which holds back a trailing incomplete
UTF-8 sequence (U+FFFD), trailing whitespace, and anything from `<tool_call>`
on; the concatenated deltas equal the non-streaming `content` exactly. At the
end the server sends parsed tool calls (one chunk with index, id and name,
then argument fragments), a `finish_reason` chunk, a `usage` chunk when
`stream_options.include_usage` is set (with `prompt_tokens_details.cached_tokens`
= the reused prefix), and `data: [DONE]`. An error after the headers is an
SSE `{"error": ...}` event.

If a write fails (client disconnect), the next token callback stops
generation. Any exception after generation starts invalidates the session:
the recurrent cache already holds tokens the client never accepted and cannot
be trimmed, so the next request is a fresh prefill. A failure while writing
the final chunks also invalidates the already-committed turn, because the
client may not hold it.

### Errors and serialization

Invalid requests get 400. A cache-state inconsistency that survives the
fallback gets 409 and resets the session. Any other inference failure gets
500 and resets the session. One `threading.Lock` guards the session; at most
one session exists and one inference runs at a time. A configured
`--system-prompt` is prepended when the client omits a system message.

`--metrics-log PATH` appends one JSON line per request (cache mode, suffix,
TTFT, decode rate, tool calls, parse errors, HTTP status); `GET /metrics`
returns cumulative counters.

## Model build (`tools/build_g128.py`)

The checkpoint is built from the BF16 `Qwen/Qwen3.8-27B` snapshot at revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. The only other inputs are the
builder and the pinned mlx-lm and transformers code.

1. **Verify the source.** The six metadata files must match their Hub hashes
   at that revision. The index must list 18 shards. Exactly 851 BF16 text
   tensors must remain after dropping 333 `model.visual.*` and 15 `mtp.*`
   tensors. A tensor from any other family is an error.
2. **Derive the config.** `derive_text_config` is a pure function of the
   source `config.json` and `generation_config.json`. It keeps the nested
   `text_config`, which pinned mlx-lm loads through
   `mlx_lm.models.qwen3_5`, and applies the edits listed in
   [decisions](decisions.md#g128-is-built-from-the-bf16-source-alone).
3. **Check the plan against the model.** The pinned model class is
   instantiated from the derived config with lazy, unallocated parameters.
   The 851 mapped source keys must equal the model's parameter keys, with
   matching element counts. There must be 64 layers (48 linear), and 498
   modules must be quantizable.
4. **Convert one layer at a time.** Each chunk is loaded and passed through
   the pinned `sanitize`. A conv1d sentinel keeps the global norm-shift rule
   intact. The chunk is quantized (affine, 3 bits, group 128) and written to
   ~512 MiB shards. Peak MLX memory is 3.3 GiB.
5. **Write metadata.** `total_size` is summed from the written tensors.
   `total_parameters` comes from the pinned `get_total_parameters`, as
   `mlx_lm.utils.save_model` computes it. Tokenizer files are re-serialized
   through `TokenizersBackend`. `generation_config.json` is copied. The
   output is checked to tokenize like the source `tokenizer.json`.
6. **Verify identity.** The staging directory is hashed and compared with
   `tools/g128-reference-manifest.json` (size and SHA-256 for every file;
   extra visible files fail). Only a matching build is renamed to the final
   path. `--verify-output DIR` runs the same check standalone using only the
   standard library.

## Limitations

- No `stop` sequences, `n > 1`, `response_format` other than text,
  `tool_choice: "required"` or named-function forcing (all rejected with
  400: decoding is not constrained), images, batching or concurrency.
  `parallel_tool_calls` is accepted but not enforced.
- Cancellation is noticed at the next token or heartbeat; a prefill already
  running is not interrupted.
- Requests with a different session id replace the active session.
- Only the server has the fresh-fallback path; the CLI resets the session on
  any cache error.
