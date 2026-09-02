# qwen38-chat

`qwen38_chat.py` is the production single-user Qwen3.8 MLX CLI. It keeps the
complete logical transcript, renders the canonical tokenizer chat template on
each turn, and sends only the exact suffix after the measured persistent-cache
offset to the pinned `generate_step` API.

The default is deterministic, non-thinking chat. `/reset` clears both the
logical transcript and cache; `/stats` and `/context` report the cached logical
token count; `/quit` exits. A generation exception or Ctrl-C resets the session
before accepting another message.

The serialized backend retains each assistant turn's exact generated token IDs
and finish reason internally. This makes a `finish_reason=length` turn a valid
continuation boundary even when decoded text is not token-lossless. Exact
known extensions reuse the cache; a valid transcript that cannot be proven to
extend it is safely fresh-prefilled. `/metrics` reports exact reuse, fallback,
conflict, and truncation-continuation counters.

Run it from the repository with the canonical environment:

```sh
source research/stage4/canonical_env.sh
uv run --project envs/mlx-0.32.1 --no-sync python research/runtime/qwen38_chat.py \
  --model /Users/artem/models/Qwen3.8-27B-3bit-g128-MLX
```

Use normal KV at 12K–16K for routine work. A clean-host 20K prompt completed
once but is an upper-envelope mode on an 18 GB Mac; 24K was not validated.

For practical sampling, add `--mode sample` (default temperature 0.7,
top-p 0.9). For KV headroom at long context, add `--kv-bits 8`; this is an
optional memory/speed tradeoff, not a replacement for the weight quant.

The runtime intentionally does not provide a multi-user or concurrent server.
The serialized OpenAI-compatible backend uses the same g128 model and accepts
system/user/assistant/tool text roles:

```sh
source research/stage4/canonical_env.sh
uv run --project envs/mlx-0.32.1 --no-sync python \
  research/runtime/qwen38_agent_server.py \
  --model /Users/artem/models/Qwen3.8-27B-3bit-g128-MLX \
  --host 127.0.0.1 --port 8000 --mode greedy --max-tokens 512
```

It is intentionally single-session and serialized; it does not batch or
provide concurrent inference. Tool-call payloads are normalized as text rather
than emitted as provider-specific structured tool calls. Reset with
`POST /v1/sessions/{id}/reset`. Stage 8 profiling found the recovered MTP
prototype exact but slower and more memory-intensive at useful contexts, so it
is not enabled by this production command; g128 greedy decoding remains the
default.

The regression tests are token/state-only and do not load the 27B model:

```sh
source research/stage4/canonical_env.sh
uv run --project envs/mlx-0.32.1 --no-sync python -m unittest \
  research.runtime.test_qwen38_chat
```
