# qwen38

A single-user local runtime for **Qwen3.8-27B** on Apple Silicon with MLX. It
targets an 18 GB M3 Pro and provides:

- `qwen38-chat`, an interactive CLI with a persistent prompt cache;
- `qwen38-server`, a serialized OpenAI-compatible HTTP backend for coding
  agents (`/v1/chat/completions` with native tool calling and streaming),
  tested with the Pi coding agent.

Both reuse the model cache across turns: each turn feeds the model only the
new tokens. When the server cannot prove that a request extends its cached
history, it rebuilds the cache from the full transcript.

## Supported model

Only one checkpoint is supported: Qwen3.8-27B, text-only, affine **3-bit,
group size 128** (`g128`), as an MLX checkpoint (~10.18 GiB of weights).
Commands below assume it is at:

```
/Users/artem/models/Qwen3.8-27B-3bit-g128-MLX
```

To rebuild it, see [Rebuilding the model](#rebuilding-the-g128-model). The
reasons for this choice are in [docs/decisions.md](docs/decisions.md).

## Install

Requires macOS on Apple Silicon and [uv](https://docs.astral.sh/uv/). The
pins are exact: Python 3.14.7, `mlx==0.32.1`, and `mlx-lm` at commit
`254d153fdeb6f150edd4fc5a54f9828638481fa8`.

```sh
uv sync --locked
```

Do not run `uv lock --upgrade` or loosen the pins. The runtime has only been
validated against this exact stack.

## Chat CLI

```sh
uv run qwen38-chat --model /Users/artem/models/Qwen3.8-27B-3bit-g128-MLX
```

Type a message per line. `/reset` clears the transcript and cache, `/stats`
shows the cached context size, and `/quit` exits. Metrics for each turn go to
stderr. Useful flags:

- `--mode sample` sets temperature 0.7 and top-p 0.9. The default is greedy.
- `--max-tokens N` sets the per-turn generation cap. The default is 512.
- `--system-prompt TEXT`
- `--enable-thinking` turns on Qwen thinking mode, which is off by default.

## Agent server

```sh
uv run qwen38-server \
  --model /Users/artem/models/Qwen3.8-27B-3bit-g128-MLX \
  --host 127.0.0.1 --port 8000 --model-name qwen38-local
```

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/chat/completions` | OpenAI Chat Completions, JSON or SSE streaming (`stream: true`), with native tool calling. An optional `session_id` field (or `X-Qwen38-Session-Id` header) names the session. |
| `POST /v1/sessions/{id}/reset` | Drop that session's transcript and cache. |
| `GET /v1/models`, `GET /health` | Model id and readiness. |
| `GET /metrics` | Request, streaming, tool-call, parse-failure, cancellation and overflow counters, plus session cache counters. |

Supported request fields: `messages` (`system`/`developer` first,
`user`, `assistant` with `tool_calls`, `tool` with `tool_call_id`; text or
text-part content), `tools` (function tools), `tool_choice` (`auto` or
`none`), `stream`, `stream_options.include_usage`, `max_completion_tokens` or
`max_tokens`, `temperature`, `top_p`. Harmless extras (`store`, `metadata`,
`user`, `seed`, `parallel_tool_calls`) are ignored. Options the server cannot
honor are rejected with 400: images, `stop`, `n > 1`, non-text
`response_format`, `tool_choice: "required"` or a named function.

**Tool calling.** Tools are rendered by Qwen3.8's own chat template, and the
model's native `<tool_call><function=…><parameter=…>` output is parsed into
OpenAI `tool_calls` (`call_…` ids, JSON-string `arguments` typed by the
declared schema, several calls per turn) with `finish_reason: "tool_calls"`.
Malformed or truncated tool syntax is returned as plain text, never as a
call. Generated token IDs remain the cache truth; see
[architecture](docs/architecture.md#generated-tokens-are-the-truth-the-api-message-is-a-view).

**Streaming.** SSE chunks with `delta.content`, `delta.tool_calls` (index, id,
name, argument fragments), a `finish_reason` chunk, an optional usage chunk
(`prompt_tokens_details.cached_tokens` is the reused cache prefix) and
`data: [DONE]`. Headers go out before prefill and `: keep-alive` comments
every 10 s keep clients from timing out on long prefills. A client disconnect
stops generation and discards the session cache.

**Sessions and context.** One session and one inference run at a time; send
the full conversation on every request. A request that extends the stored
transcript with the same tools reuses the cache and prefills only the new
tokens. Anything else (edited history, changed tools, compaction) is rebuilt
by a fresh prefill, never an error. `--max-context-tokens` (default 16384)
caps prompt plus completion; a larger prompt gets a 400
`context_length_exceeded`. JSON responses and the final stream chunk carry
`x_qwen38_metrics` (TTFT, decode speed, cache mode, memory); `--metrics-log
PATH` writes one JSON line per request.

Check the protocol layer without loading the model:

```sh
uv run qwen38-server --self-test
```

## Using it with Pi

Tested with Pi 1.0.0 (`pi-coding-agent`, Homebrew). Pi needs only its
standard OpenAI-compatible provider, no extension. Copy
[`integrations/pi/models.example.json`](integrations/pi/models.example.json)
to `~/.pi/agent/models.json` (or merge its provider into an existing file) and
[`integrations/pi/settings.example.json`](integrations/pi/settings.example.json)
into `~/.pi/agent/settings.json`. The settings matter: Pi's default
compaction reserve (16384) and kept tail (20000) assume a large window and
would never compact a 16K one.

```sh
uv run qwen38-server --model /Users/artem/models/Qwen3.8-27B-3bit-g128-MLX --port 8000
# in the project you want to work on:
pi --provider qwen38-local --model qwen38-local
pi -p --provider qwen38-local --model qwen38-local "Run the tests and fix the failure"
```

To try it without touching your Pi configuration, point
`PI_CODING_AGENT_DIR` at a directory holding the two example files. Pi's
observed wire behavior is in [docs/pi-wire-contract.md](docs/pi-wire-contract.md).

## Hermes Agent (compatibility only)

Hermes Agent v0.21.5 (`a5e7df27`, official installer) works against the same
endpoint as a `custom` provider with no server changes specific to it:

```yaml
model:
  provider: custom
  base_url: http://127.0.0.1:8000/v1
  default: qwen38-local
  api_key: local
```

A bounded smoke passed: plain chat, a `read_file` call and its result,
and a small fix-the-test task (`search_files`, `read_file`, `terminal`,
`patch`), all streamed, with exact cache reuse between turns. It is **not a
supported setup on this Mac**: Hermes refuses to start below a 64K context
window, so the smoke had to declare `context_length: 65536` while the server
still enforces 16K. Hermes also sends side requests (title generation with
`response_format: json_schema`, which the server rejects and Hermes retries
without) that replace the single session's cache, so the next turn pays a
fresh prefill.

## Context size

Use **12K–16K tokens** of conversation context for routine work. At 16K, warm
turns start in about 1.2 s and decode at about 10.5 tok/s, with a peak of
about 15.2 GiB. 20K has worked once on a freshly booted, otherwise idle
machine, but it is not a daily setting. The first full prefill of a long
prompt runs at about 70–80 tok/s, so a 16K cold start takes about 4 minutes.
Follow-up turns only pay for their new tokens. See
[docs/performance.md](docs/performance.md).

## Tests

```sh
uv run python -m unittest discover -s tests
```

The tests load no model weights and need no Metal device. Protocol tests
(`test_tool_protocol.py`) need nothing. The tokenizer-based tests
(`test_tool_session.py`, `test_streaming.py`, `test_session_recovery.py`)
pin the Qwen3.8 template's tool format and drive the real session and HTTP/SSE
code with a scripted fake generator; they read `tokenizer.json` from the model
directory. Point them
elsewhere with `QWEN38_MODEL=/path/to/model`. They are skipped if no tokenizer
is found. Two model-builder tests read the BF16 source snapshot
(`QWEN38_SOURCE`, default: the Hugging Face cache) and are skipped without it.

## Integration benchmarks

`benchmarks/long_agent.py` starts the real server and drives one
coding-agent session with:

- tool requests and a long tool result;
- a wrong hypothesis and failing-test revisions;
- three forced `finish_reason=length` truncations and their continuations.

It loads the 27B model and takes about 5 minutes. Do not run it alongside
another heavy MLX process.

```sh
uv run python benchmarks/long_agent.py --output benchmarks/results/long-agent.json
```

It exits 0 when the classification is `PASS_LONG_SERIALIZED_AGENT`. The
reference run reached 10,589 logical tokens over 29 assistant turns, with no
HTTP errors.

`benchmarks/tool_smoke.py` is a short real-model check of native tool calls
over SSE: one `read_file` and one `run_command` round trip, a non-streaming
turn, a changed tool schema, and a two-call turn. It takes about 2 minutes.

```sh
uv run python benchmarks/tool_smoke.py --output benchmarks/results/tool-smoke.json
```

`benchmarks/pi_agent.py` runs Pi itself on seven fixture repositories in
`benchmarks/pi_fixtures/` (Pi owns the loop and its built-in tools; each
fixture has a hidden evaluator Pi never sees) and writes per-request and
per-task records to `benchmarks/results/pi-stage9.jsonl`. It needs `pi` on
`PATH` and uses a disposable Pi agent directory.

```sh
uv run python benchmarks/pi_agent.py --verify-fixtures   # no model: baselines must fail
uv run python benchmarks/pi_agent.py                     # real run, ~25 minutes
```

`tools/memory_watchdog.sh <pid> <log>` stops a benchmark whose host free
memory drops below 6% or whose swap grows by more than 3 GiB.

## Rebuilding the g128 model

The model can be rebuilt from the original BF16 `Qwen/Qwen3.8-27B` release
alone. No other MLX checkpoint is needed. The build is pinned to source
revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; the tool checks the
source metadata hashes and refuses any other revision.

Fetch the source (55.6 GB of weights plus six small metadata files):

```sh
uv run hf download Qwen/Qwen3.8-27B \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --include '*.safetensors' --include config.json \
  --include generation_config.json --include model.safetensors.index.json \
  --include chat_template.jinja --include tokenizer_config.json \
  --include tokenizer.json
```

Repeat `--include` for every pattern. If the patterns follow a single
`--include`, `hf` reads them as filenames and skips the weights.

The reproducibility chain is:

```
BF16 source (Qwen/Qwen3.8-27B @ 1d4bf0f…)  →  tools/build_g128.py  →  g128 checkpoint
                                                                      ↓
                                     tools/g128-reference-manifest.json (SHA-256)
```

The BF16 source is the asset to keep. The g128 checkpoint can always be
rebuilt from it, and the committed manifest proves that a rebuild is the
validated canonical checkpoint. No reference MLX checkpoint is needed.

Build:

```sh
SOURCE="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
OUTPUT="$HOME/models/Qwen3.8-27B-3bit-g128-MLX-rebuild"

uv run python tools/build_g128.py --source "$SOURCE" --self-test

uv run python tools/build_g128.py \
  --source "$SOURCE" \
  --output "$OUTPUT"
```

`--self-test` checks the source, the derived config, the model mapping and
the tokenizer files in a few seconds, without loading weights. The build
quantizes one layer at a time (peak MLX memory 3.3 GiB) and writes into
`<output>.partial`. It refuses to overwrite an existing output.

When conversion finishes, the build checks `<output>.partial` against the
committed manifest. It renames it to `<output>` only if every file matches
(`PASS: canonical g128 manifest matched`). On a mismatch it prints each
offending file, leaves the `.partial` directory for inspection and exits
nonzero. A non-default `--shard-bytes` cannot produce the canonical files,
so that build skips the check and says so.

### Verifying a checkpoint

```sh
uv run python tools/build_g128.py --verify-output "$OUTPUT"
```

```
PASS
canonical g128 manifest verified
25 files
10,951,147,660 bytes
SHA-256 all match
```

Verification uses only the standard library. It does not need the source,
MLX, the network or another checkpoint. Every listed file must have the
exact size and SHA-256. Any other visible file or directory fails the check.
Hidden entries such as `.DS_Store` are ignored. Each failure names its file,
for example:

```
FAIL model-00007-of-00019.safetensors: sha256 mismatch
  expected sha256: …
  actual sha256:   …
```

The exit status is 0 on PASS, 1 on any mismatch, and 2 if the manifest itself
is unreadable or uses an unsupported schema.

The manifest covers 25 files: 19 weight shards, the index, `config.json`,
`generation_config.json`, `tokenizer.json`, `tokenizer_config.json` and
`chat_template.jinja`. They total 10,951,147,660 bytes, and the tensor
payload is 10,930,715,648 bytes (1,847 tensors, 498 quantized modules). The
canonical `config.json` is the builder's output. The original production
checkpoint (built before the source-only builder) differs only by two
ignored MTP keys in `config.json`. The manifest records that file as a
legacy variant, so a plain `--verify-output` reports it as legacy, and
`--allow-legacy` accepts it. See [docs/decisions.md](docs/decisions.md).

## Limitations

- Single user, serialized. Requests with a different `session_id` replace the
  current session and its cache.
- No `stop` sequences, forced tool choice, constrained decoding or images.
- A client disconnect is noticed at the next token or heartbeat; a running
  prefill is not interrupted.
- No speculative decoding or MTP. Exact MTP was built and measured, but it is
  slower than plain decoding (see [docs/decisions.md](docs/decisions.md)).
- Never call `mx.set_wired_limit()` or change `iogpu.wired_limit_mb`. The
  runtime deliberately leaves MLX's wired-memory state untouched.

Further reading: [architecture](docs/architecture.md),
[decisions](docs/decisions.md), [performance](docs/performance.md).
The research that produced this runtime is preserved in git under tag
`research-stage8-complete`.
