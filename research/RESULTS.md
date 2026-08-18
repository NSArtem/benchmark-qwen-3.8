# Qwen3.8-27B MLX Research Results

This report is a running ledger. Values marked `reuse` come from saved raw
artifacts; values marked `policy-summary` come from the current project
instructions and have no separate raw run directory in this checkout.

## Phase 0 evidence inventory

| Experiment | Runtime | Prompt/context | Result | Performance / memory | Provenance |
| --- | --- | ---: | --- | --- | --- |
| `P0-ALIGNMENT-REUSE` | MLX 0.32.1, direct `generate_step`, default wired, g64 affine | 512/513/640/641/768/769 final tokens; 128 output | 18/18 COHERENT; 0 OOM; deterministic within length | Peaks: 12.188, 12.187, 12.325, 12.327, 12.461, 12.459 GiB respectively; after-first-token active+cache roughly 11.21–11.40 GiB | `qwen38-mlx0321-alignment/results.json`, commit `4418c08` |
| `P0-OOM-REUSE` | MLX 0.32.1, direct `generate_step`, g64 affine | 768 final tokens | Default wired PASS; forced recommended 12.000 GiB wired limit OOM | Default after-first-token peak 12.461 GiB; forced OOM direct snapshot active+cache 11.958 GiB | `qwen38-mlx0321-oom/results.json`, commit `664941d` |
| `P0-16K-POLICY` | MLX 0.32.1, normal generate with wired wrapper made no-op, g64 affine | 16,767 actual prompt tokens; 256 output | COHERENT; no OOM; understands Lantern task | 63.348 prompt tok/s; 4.118 decode tok/s; 17.200 GB MLX peak | `AGENTS.md` project-policy summary; raw run not present |

## Baseline context table (to be filled by Phase 1)

The 512–769 rows are correctness probes with a compact Lantern prompt family,
not interchangeable with the larger coding prompt. The 16,767 row is retained
without duplication because the task explicitly supplies it as existing
evidence. Missing cells are intentionally blank, not estimates.

| Target | Actual prompt tokens | Classification | TTFT s | Prompt tok/s | Decode tok/s | Generated tokens | Active GiB | Cache GiB | Active+cache GiB | MLX peak GiB | OS memory | Exception / stability |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ~1K | — | — | — | — | — | — | — | — | — | — | — | — |
| ~2K | — | — | — | — | — | — | — | — | — | — | — | — |
| ~4K | — | — | — | — | — | — | — | — | — | — | — | — |
| ~8K | — | — | — | — | — | — | — | — | — | — | — | — |
| ~12K | — | — | — | — | — | — | — | — | — | — | — | — |
| ~16K | 16,767 | COHERENT (policy-summary) |  — | 63.348 | 4.118 | 256 | — | — | — | 17.200 | no OOM reported; raw OS telemetry unavailable | Existing manual run; do not duplicate solely for evidence |

## Reuse conclusions

- The simple prefill alignment hypothesis is rejected by the 18/18 coherent
  result in the saved 0.32.1 experiment.
- The forced stock wired-limit policy is a distinct failure mechanism from the
  default-wired direct path; it must not be used by subsequent experiments.
- Older `qwen38-3bit-mlx-rescue/` and `qwen38-3bit-textonly/` reports used MLX
  0.32.0 and are not valid current-baseline rows.
