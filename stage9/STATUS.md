# Stage 9 working status (temporary; distilled into docs at the end)

## Phase 0 — baseline (2026-10-03)

- HEAD 7d32eb5, branch main. Uncommitted user changes: AGENTS.md (Stage 9
  instructions), untracked TASK.md, TRIMMING_REPORT.md — preserved.
- `uv run python -m unittest discover -s tests`: 71 tests OK (1 skipped).
- `uv run qwen38-server --self-test`: PROTOCOL_SELF_TEST_PASS.
- Host: 18 GB, 74% free, swap 387 MB used at start. BlueStacks (~3.6 GB) and
  Firefox (~4.5 GB) resident; the first model load evicted them to swap
  (+4.2 GB, free never < 15%). Subsequent loads: swap +1.4 GB max, free >= 24%.
- Pi: `/opt/homebrew/bin/pi` → Homebrew `pi-coding-agent` 1.0.0
  (`@earendil-works/pi-coding-agent` 1.0.0, released 2026-10-01).
- Hermes: not installed.

## Phase 1 — canonical runtime equivalence (closed)

`benchmarks/results/g128-rebuild/runtime_equivalence.py`, run sequentially
under a memory watchdog:

| Case | Production vs rebuild token IDs | TTFT prod/rebuild | Decode |
| --- | --- | --- | --- |
| coding (96) | identical | 2.33 / 1.45 s | 12.0 tok/s |
| reasoning (96) | identical | 0.96 / 0.96 s | 12.0 |
| prose (96) | identical | 0.59 / 0.60 s | 12.0 / 11.9 |
| structured (42, EOS) | identical | 0.60 / 0.60 s | 12.0 |
| two-turn turn 1 / turn 2 | identical | — | — |

Rebuild two-turn: all checks true; turn 2 prefilled a 29-token suffix of an
86-token prompt (exact reuse), recalled context, no replay. Load 3.8 s,
MLX peak 10.55 GiB for both.

Verdict: the canonical source-only rebuild is runtime-equivalent. Production
path remains `/Users/artem/models/Qwen3.8-27B-3bit-g128-MLX` (its config is the
manifest's accepted legacy variant). Both directories are kept; deleting
either is the user's decision.
