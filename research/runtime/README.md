# qwen38-chat

`qwen38_chat.py` is the production single-user Qwen3.8 MLX CLI. It keeps the
complete logical transcript, renders the canonical tokenizer chat template on
each turn, and sends only the exact suffix after the measured persistent-cache
offset to the pinned `generate_step` API.

The default is deterministic, non-thinking chat. `/reset` clears both the
logical transcript and cache; `/stats` and `/context` report the cached logical
token count; `/quit` exits. A generation exception or Ctrl-C resets the session
before accepting another message.

Run it from the repository with the canonical environment:

```sh
source research/stage4/canonical_env.sh
uv run --project envs/mlx-0.32.1 --no-sync python research/runtime/qwen38_chat.py \
  --model /Users/artem/models/Qwen3.8-27B-mixed-3-4-control-g128
```

For practical sampling, add `--mode sample` (default temperature 0.7,
top-p 0.9). For KV headroom at long context, add `--kv-bits 8`; this is an
optional memory/speed tradeoff, not a replacement for the weight quant.

The runtime intentionally does not provide a multi-user or concurrent server.
The regression tests are token/state-only and do not load the 27B model:

```sh
source research/stage4/canonical_env.sh
uv run --project envs/mlx-0.32.1 --no-sync python -m unittest \
  research.runtime.test_qwen38_chat
```
