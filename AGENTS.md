# Project instructions

This repository maintains a single-user Qwen3.8-27B MLX runtime: the
`qwen38-chat` CLI and the serialized OpenAI-compatible `qwen38-server`. See
`README.md` for usage and `docs/` for architecture, decisions, and
performance.

The research phase is closed. Its history is preserved in git under tag
`research-stage8-complete`. Do not restore research trees into HEAD. If you
need old evidence, read it from the tag
(`git show research-stage8-complete:<path>`).

## Hardware and model

- MacBook Pro, Apple M3 Pro, 18 GB unified memory. Memory residency is the
  binding constraint.
- The only supported model is Qwen3.8-27B affine 3-bit, group 128, at
  `/Users/artem/models/Qwen3.8-27B-3bit-g128-MLX` (~10.18 GiB). Never modify
  files under `/Users/artem/models/`.
- Practical context is 12K–16K tokens.

## Environment

- Install with `uv sync --locked` and run with `uv run ...` from the
  repository root.
- Pins are exact and validated: Python 3.14.7, `mlx==0.32.1`, and `mlx-lm` at
  `254d153fdeb6f150edd4fc5a54f9828638481fa8`. Do not upgrade, relock with
  `--upgrade`, or loosen these pins as a side effect of other work. Any
  runtime upgrade must be its own deliberate, re-validated change.
- Do not add dependencies the runtime, tests, benchmark, or build tool do
  not need.

## System safety

- Never modify `sudo sysctl iogpu.wired_limit_mb`.
- Never call `mx.set_wired_limit()`. Do not use mlx-lm's `stream_generate` /
  `wired_limit` wrapper; the runtime calls `generate_step` directly.
- Never run two heavy Qwen/MLX workloads at the same time.
- Stop immediately on Metal OOM, GPU timeout, WindowServer instability,
  major unresponsiveness, repeated failed allocations, or near-exhausted
  swap. Check memory and swap before heavy runs (`memory_pressure -Q`,
  `sysctl vm.swapusage`).

## Correctness invariants

- Persistent chat is exact-suffix. The cache may only ever receive the exact
  canonical suffix after the tokens it already holds. Recurrent
  `ArraysCache` state cannot be trimmed.
- Generated token IDs are authoritative. Decoded text is for API responses
  only.
- If cache reuse cannot be proven for a valid transcript, fresh-prefill. Do
  not return an error just because a cache optimization missed.
- Correctness comes before cache hit rate and before speed.

## Validation

- Unit and regression tests use no weights and no Metal:
  `uv run python -m unittest discover -s tests`.
- Protocol check without loading the model:
  `uv run qwen38-server --self-test`.
- The real-model integration benchmark is `benchmarks/long_agent.py`. It
  loads 27B; run it only on a healthy host and never concurrently with
  another heavy MLX process.
- Run a short real-model smoke only when a change touches generation or
  cache code.

## Out of scope unless explicitly requested

Static weight quantization or DWQ research, MTP or speculative decoding,
GGUF/llama.cpp, contexts of 20K and above, multi-user batching or concurrent
serving, and wired-memory tuning.

## Git

- Start with `git status` and `git log --oneline -20`. Preserve unrelated
  changes. Never `git reset --hard` or `git clean -fd`.
- Never commit model weights, Hugging Face caches, virtualenvs, or large raw
  traces. Write benchmark outputs to the git-ignored `benchmarks/results/`.
- Use focused, logical commits.
