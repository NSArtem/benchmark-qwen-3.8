# Architecture

The runtime is two modules in `src/qwen38/`:

| Module | Contents |
| --- | --- |
| `chat.py` | Token and cache logic shared by both entry points: canonical prompt rendering, exact-suffix computation, history reconciliation, cache offset checks. It also contains `ExactSuffixChat` and the `qwen38-chat` CLI. |
| `server.py` | OpenAI-style message normalization, `AgentSession` (adds session bookkeeping and fallback to `chat.py`), and a stdlib `http.server` backend. |

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
assistant span for its stored IDs and keeps a single EOS boundary. The API
still returns the decoded text. Exact token identity is used only internally.

## Server request handling (`AgentSession.complete`)

- **Exact reuse.** The incoming messages extend the stored transcript, and
  the reconciled canonical stream starts with the cached tokens. Only the
  suffix is prefilled. This applies to turns that ended by EOS and turns that
  ended by `finish_reason=length` alike.
- **Fresh fallback.** The messages are valid but cannot be proven to extend
  the cache, for example after an edited history, a missing message, or a
  failed reconciliation. The cache is cleared and the whole canonical
  transcript is prefilled. A cache miss never causes an HTTP 409.
- **Errors.** Invalid requests get 400. A cache-state inconsistency that
  survives the fallback gets 409 and resets the session. Any other inference
  failure gets 500 and resets the session.

A configured server `--system-prompt` is prepended when the client omits a
system message.

One `threading.Lock` guards the session. At most one session exists, and at
most one inference runs at a time.

## Limitations

- No streaming, `stop` sequences, cancellation, batching or concurrency.
- Tool calls travel as text. Native OpenAI `tool_calls` are not produced.
- Requests with a different session id replace the active session.
- Generation logic is duplicated between `ExactSuffixChat.turn` and
  `AgentSession.complete`, and only the server has the fresh-fallback path.
