# Qwen3.8 Night-2 runtime commands

These commands use the canonical MLX 0.32.1 environment and the preserved
source-backed g128 checkpoint. The local runtime calls pinned `generate_step`
directly, leaves MLX wired state untouched, and never changes a system wired
memory setting.

## Inspect the runtime/model

```sh
uv run --project envs/mlx-0.32.1 --no-sync python research/runtime/qwen38_runtime.py \
  info \
  --model /Users/artem/models/Qwen3.8-27B-3bit-g128-MLX
```

## One-shot g128 generation

```sh
uv run --project envs/mlx-0.32.1 --no-sync python research/runtime/qwen38_runtime.py \
  generate \
  --model /Users/artem/models/Qwen3.8-27B-3bit-g128-MLX \
  --prompt-file qwen38-mtp-smoke/coding-prompt.txt \
  --max-tokens 256 \
  --temperature 0 \
  --top-p 1 \
  --prefill-step-size 2048
```

## Correctness-safe interactive chat

```sh
uv run --project envs/mlx-0.32.1 --no-sync python research/runtime/qwen38_runtime.py \
  chat-safe \
  --model /Users/artem/models/Qwen3.8-27B-3bit-g128-MLX \
  --max-tokens 256 \
  --temperature 0 \
  --top-p 1 \
  --prefill-step-size 2048
```

Enter one user query per line. Enter `/reset` to clear the logical transcript
or `/quit` to exit. `chat-safe` re-renders the full logical transcript and
creates a fresh hybrid cache for every turn. The `chat` subcommand is a safe
backward-compatible alias. Persistent incremental cache reuse is not exposed
as a trusted fast mode.

## Optional ~16K KV8 one-shot headroom variant

This is the measured safety variant, not the normal 4K/8K default and not a
multi-turn KV8 oracle-validated chat mode:

```sh
uv run --project envs/mlx-0.32.1 --no-sync python research/runtime/qwen38_runtime.py \
  generate \
  --model /Users/artem/models/Qwen3.8-27B-3bit-g128-MLX \
  --prompt-file qwen38-mtp-smoke/coding-prompt.txt \
  --max-tokens 128 \
  --temperature 0 \
  --top-p 1 \
  --prefill-step-size 2048 \
  --kv-bits 8 \
  --kv-group-size 64 \
  --quantized-kv-start 0
```

Measured at 16,727 prompt tokens: 14.771 GiB peak, 9.495 tok/s decode, and
coherent output. It saved 0.465 GiB peak versus normal g128 but cost about
10.8% decode throughput.
