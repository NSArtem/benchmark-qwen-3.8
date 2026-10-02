# qwen38

A single-user local runtime for **Qwen3.8-27B** on Apple Silicon with MLX. It
targets an 18 GB M3 Pro and provides:

- `qwen38-chat`, an interactive CLI with a persistent prompt cache;
- `qwen38-server`, a serialized OpenAI-compatible HTTP backend for coding
  agents (`/v1/chat/completions`).

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
  --host 127.0.0.1 --port 8000 --mode greedy --max-tokens 512
```

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/chat/completions` | Non-streaming chat completion. Accepts `system`, `user`, `assistant`, `tool` messages and an optional `session_id` field (or the `X-Qwen38-Session-Id` header). |
| `POST /v1/sessions/{id}/reset` | Drop that session's transcript and cache. |
| `GET /v1/models`, `GET /health` | Model id and readiness. |
| `GET /metrics` | Request count plus session counters: exact reuse, fresh fallback, conflicts, truncation continuations. |

One session and one inference run at a time. Sending the full conversation
on every request is the intended client behavior. Responses also include an
`x_qwen38_metrics` object with TTFT, decode speed, cache mode and memory.

Check the server without loading the model:

```sh
uv run qwen38-server --self-test
```

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

The tests load no model weights and need no Metal device. The tokenizer-based
session tests read `tokenizer.json` from the model directory. Point them
elsewhere with `QWEN38_MODEL=/path/to/model`. They are skipped if no tokenizer
is found. Two model-builder tests read the BF16 source snapshot
(`QWEN38_SOURCE`, default: the Hugging Face cache) and are skipped without it.

## Integration benchmark

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

Build:

```sh
SOURCE="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"

uv run python tools/build_g128.py --source "$SOURCE" --self-test

uv run python tools/build_g128.py \
  --source "$SOURCE" \
  --output "$HOME/models/Qwen3.8-27B-3bit-g128-MLX"
```

`--self-test` checks the source, the derived config, the model mapping and
the tokenizer files in a few seconds, without loading weights. The build
quantizes one layer at a time (peak MLX memory 3.3 GiB) and writes into
`<output>.partial`, renaming it only when complete. It refuses to overwrite
an existing output.

The output is 1,847 tensors (498 quantized modules) in 19 shards, totaling
10,930,715,648 bytes. A source-only rebuild on 2026-10-02 produced weight
shards, index and tokenizer files byte-identical to the production
checkpoint. `config.json` differed only by two dropped MTP keys (see
[docs/decisions.md](docs/decisions.md)).

## Limitations

- Single user, serialized. Requests with a different `session_id` replace the
  current session and its cache.
- No streaming, no `stop` sequences, and no cancellation.
- Tool calls are plain text. Incoming `tool_calls` are serialized into the
  assistant message content, and responses never contain structured
  `tool_calls`.
- No speculative decoding or MTP. Exact MTP was built and measured, but it is
  slower than plain decoding (see [docs/decisions.md](docs/decisions.md)).
- Never call `mx.set_wired_limit()` or change `iogpu.wired_limit_mb`. The
  runtime deliberately leaves MLX's wired-memory state untouched.

Further reading: [architecture](docs/architecture.md),
[decisions](docs/decisions.md), [performance](docs/performance.md).
The research that produced this runtime is preserved in git under tag
`research-stage8-complete`.
