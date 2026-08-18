# Qwen3.8 MLX Investigation Plan

This ledger follows `TASK.md` on the canonical MLX 0.32.1 environment and the
existing affine 3-bit/group-64 checkpoint. Every cold configuration uses a
fresh process, direct `generate_step`, greedy decoding, `prefill_step_size=2048`,
and leaves MLX wired state at its default. Heavy workloads are serialized.

## Experiment IDs and gates

| ID | Phase | Question / action | Decision gate |
| --- | --- | --- | --- |
| `P0-RECOVER` | 0 | Recover git, artifacts, process state, and ledger | No competing heavy workload; preserve existing changes |
| `P1-G64-LADDER` | 1 | Fill useful cold g64 points around 1K/2K/4K/8K/12K; reuse existing 0.32.1 evidence | Stop at Metal OOM, severe pressure, or first repeated correctness failure |
| `P2-NOWIRED` | 2 | Add a small explicit no-wired generate/chat helper and short/moderate validation | Helper must not call `mx.set_wired_limit()` and must log the policy |
| `P3-WARM4K` / `P3-WARM8K` | 3 | Measure persistent prompt-cache multi-turn latency, correctness, and memory | Classify EXCELLENT/USEFUL/MARGINAL/UNUSABLE before continuing |
| `P4-QSIZE` | 4 | Inspect actual tensors/config and calculate g64/g128/2-bit/mixed candidates | No conversion or quality claim from size estimates |
| `P5-CONVERT-FEAS` | 5 | Identify source weights, disk budget, and bounded conversion path | Choose `CONVERSION_SAFE`, `STREAMING_CONVERSION_REQUIRED`, `BLOCKED_BY_SOURCE`, `BLOCKED_BY_DISK`, or `BLOCKED_BY_RUNTIME` |
| `P6-G128-CONVERT` | 6 | Produce affine 3-bit/group-128 only if Phase 5 permits | One failed implementation is recorded; no repeated identical OOM |
| `P7-G128-SMOKE` | 7 | Validate g128 short/768/4K with 256 tokens | Reject after one repeat if a representative failure reproduces |
| `P8-G128-LADDER` | 8 | Adaptive 8K/12K/16K then 20K/24K only with headroom | Stop at first unsafe point; 32K requires explicit evidence gate |
| `P9-KV8` | 9 | Inspect cache API and isolate g64+KV8, then only promising combinations | Do not run a large matrix; preserve correctness and memory comparisons |
| `P10-MIXED` | 10 | At most one optional second weight candidate if all prerequisites pass | Short/768/4K gate before any long-context run |
| `P11-COMPARE` | 11 | Compare only measured configurations | No guessed cells or recommendations without coherence/stability evidence |
| `P12-COMMANDS-REPORT` | 12 | Commit small runtime commands, final report, and ledger | Verify artifacts, final git status/log, then stop |

## Concrete command conventions

Canonical environment prefix:

```text
uv run --project envs/mlx-0.32.1 --no-sync ...
```

All benchmark commands will write under `research/` or a phase-specific
artifact directory, save raw output and token IDs, and append one JSON object
per experiment to `research/results.jsonl`. Existing artifact paths are cited
instead of rerun when their evidence satisfies the question.

## Current reuse map

- `P0-ALIGNMENT-REUSE`: `qwen38-mlx0321-alignment/results.json` and
  `results.md`; 18 fresh-process 0.32.1 direct runs at 512/513/640/641/768/769,
  all coherent and deterministic.
- `P0-OOM-REUSE`: `qwen38-mlx0321-oom/results.json` and `results.md`; direct
  default-wired pass versus forced recommended wired-limit OOM at 768.
- `P0-16K-POLICY`: the current `AGENTS.md` records a manual no-wired 16,767
  token coding prompt with coherent 256-token output, 63.348 prompt tok/s,
  4.118 decode tok/s, and 17.200 GB MLX peak. The raw run is not present as a
  separate artifact in this checkout and will not be duplicated solely for
  that reason.
- `qwen38-3bit-mlx-rescue/` and `qwen38-3bit-textonly/` contain historical
  0.32.0 results and are excluded from the canonical baseline.

## Safety decisions

- Never modify `iogpu.wired_limit_mb`, call `mx.set_wired_limit()`, or increase
  any wired-memory setting.
- Never run MLX 0.32.0, GGUF, two large workloads concurrently, or a repeated
  known-failing OOM configuration.
- Do not download source weights or convert until disk/source/conversion
  feasibility is measured.
- Treat 17+ GB MLX peak as a conservative danger boundary on this 18 GB Mac.
