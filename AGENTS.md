# Agent instructions

This repository holds a single-user Qwen3.8-27B MLX runtime (the
`qwen38-chat` CLI and the serialized OpenAI-compatible `qwen38-server`) and
the research behind it. See `README.md` for usage and `docs/` for the
architecture, the decisions, and the performance figures.

## Current stage: Stage 10 (quantization research)

**The active task is `TASK.md`. Read it fully before doing anything,
including its section 12 (amendment 1).** It
asks you to build measurement tools, build new quantized checkpoints, and
record numbers. Earlier stages declared weight-quantization research out of
scope. Stage 10 lifts that restriction, for the experiments listed in
`TASK.md` only.

## How to work

- Follow `TASK.md` literally and in order. Use the exact numbers it gives.
- You collect data; you do not draw conclusions. Never write "better",
  "worse", "recommend", or a verdict in any file. Write numbers, statuses,
  commands, and error messages.
- Do not add, remove, reorder, or "improve" experiments. Put new ideas in
  `research/stage10/IDEAS.md` and do not run them.
- If something is unclear or blocked, write it to
  `research/stage10/QUESTIONS.md`, take the option that changes the least,
  and move on. Stop only on the STOP rules in `TASK.md` §7.
- Write a unit test before using new code on real weights, where `TASK.md`
  asks for one. A failing test means you fix the code. You never weaken the
  test.
- Keep `research/stage10/PROGRESS.md` and `research/stage10/LOG.md` up to
  date after every step, so that work can resume after an interruption.
  When you start or resume, read PROGRESS.md and the end of LOG.md first.

## Hardware and paths

- MacBook Pro, Apple M3 Pro, 18 GB unified memory. Memory is the binding
  constraint.
- Production model (read-only):
  `/Users/artem/models/Qwen3.8-27B-3bit-g128-MLX` (~10.18 GiB). **Never
  create, modify, or delete anything under `/Users/artem/models/`.**
- BF16 source (read-only):
  `$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
  Do not modify or delete it.
- Stage 10 candidates: `/Users/artem/models-stage10/<EXP_ID>/`. Disk is not
  a constraint; keep everything.

## System safety (overrides everything else)

- Never change `iogpu.wired_limit_mb` (`sysctl`). Never call
  `mx.set_wired_limit()`. Never use mlx-lm's `stream_generate` or its
  `wired_limit` wrapper; call `generate_step` directly. Never set an
  environment variable that raises a wired limit (for example
  `ESCHA_MLX_WIRED_GB`).
- Never run two heavy MLX workloads at the same time. Heavy means anything
  that loads more than 2 GB of weights.
- Before every heavy run, check memory (`memory_pressure -Q`,
  `sysctl vm.swapusage`). Run every heavy job under
  `tools/memory_watchdog.sh <pid> <log>`.
- Stop immediately on Metal OOM, GPU timeout, WindowServer instability,
  major unresponsiveness, repeated failed allocations, or near-exhausted
  swap.
- Never kill or close the user's applications to free memory.

## Production runtime: frozen during Stage 10

- Do not modify `src/`, `tools/`, `tests/`, `docs/`, `benchmarks/*.py`,
  `README.md`, the root `pyproject.toml`, or the root `uv.lock`. Exception:
  you may *read* and *run* `tools/memory_watchdog.sh` and copy code from
  `tools/build_g128.py`.
- The production stack stays pinned (Python 3.14.7, `mlx==0.32.1`,
  `mlx-lm` at `254d153fdeb6f150edd4fc5a54f9828638481fa8`). Production
  checks: `uv run python -m unittest discover -s tests` and
  `uv run qwen38-server --self-test`, both from the repository root.
- Runtime invariants that any later production change must keep:
  - persistent chat is exact-suffix, and recurrent `ArraysCache` state is
    never trimmed;
  - generated token IDs are authoritative;
  - when cache reuse cannot be proven, fresh-prefill instead of erroring.

## Research environment

- All Stage 10 code lives in `research/stage10/`, a separate uv project
  with its own `pyproject.toml` and `uv.lock`. Run it with
  `cd research/stage10 && uv run ...`. There you use the newest mlx and
  mlx-lm releases (`TASK.md` §2). Pin them exactly and record them.
- The reference runtime for E14 lives in its own uv project,
  `research/stage10/escha/`.
- Unit tests: `cd research/stage10 && uv run python -m unittest discover -s tests`.
  They must not load 27B weights or need more than a few GB.
- Older research evidence may exist under the git tag
  `research-stage8-complete` on the local machine only (it is not on
  GitHub). Read it with `git show research-stage8-complete:<path>` if you
  need it. Do not restore it into HEAD.

## Data hygiene

- Never commit model weights, Hugging Face caches, virtualenvs, `.npy`
  arrays, or raw logs. Large outputs go to the git-ignored
  `benchmarks/results/stage10/`.
- What you commit: code, tests, `research/stage10/*.md`,
  `data_manifest.json`, and `results/<EXP_ID>/summary.json` and
  `gen_check.json`.
- The root `.gitignore` ignores any path named `fixtures`, `long_agent.py`,
  `pi_agent.py`, `pi_fixtures`, or `tool_smoke.py`, wherever it is. Do not
  give Stage 10 files or directories these names. After every commit, check
  `git status --ignored research/` for files that were silently ignored.

## Git

- Work on branch **`stage-10`**. Never commit to or push `main`. At the
  start of every session run `git checkout stage-10`, `git status`, and
  `git log --oneline -20`.
- Commit after each completed tool (Phase A item) and after each
  experiment, with focused messages, for example
  `stage10: add GPTQ builder` or `stage10: E07 results`.
- Before pushing: `git pull --rebase origin stage-10`, then
  `git push origin stage-10`. Never force-push, never `git reset --hard`, and
  never `git clean -fd`. Preserve changes you did not make.
