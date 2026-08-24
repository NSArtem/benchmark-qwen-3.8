# Stage 3 status

Last updated: 2026-08-24 — Phase 0 recovery and initialization.

## Current phase

Phase 0 complete; Phase 1 same-boundary cache oracle is next.

## Recovered baseline from raw artifacts

| item | measured fact |
|---|---|
| g64 reference payload | 11,771,132,928 bytes = 10.9627218 GiB; affine 3-bit/group-64 |
| g128 candidate payload | 10,930,715,648 bytes = 10.1800222 GiB; affine 3-bit/group-128; 1,847 tensors / 19 shards |
| g128 cold 4K | 4,096 prompt tokens; peak 13.3595572 GiB; TTFT 51.2942 s in P9 repeat 2; decode 11.6228 tok/s |
| g128 cold 8K | 8,191 prompt tokens; peak 13.9828330 GiB; TTFT median 105.9749 s across P9 repeats; decode median 11.2751 tok/s |
| g128 cold 12K | 12,288 prompt tokens; peak 14.6091910 GiB; TTFT 158.2382 s; decode 10.8301 tok/s |
| g128 cold 16K-ish | 16,727 prompt tokens; peak 15.2356711 GiB; TTFT 219.6105 s; decode 10.6409 tok/s |
| exact-suffix g128 4K persistent turn 2 | 38-token suffix; TTFT 0.979969 s; decode 11.6337 tok/s; coherent/no stale replay in the saved run |
| exact-suffix g128 4K fresh full-transcript turn 2 | 4,230-token prompt; TTFT 54.745441 s; decode 11.5998 tok/s; coherent |
| exact-suffix unresolved comparison | persistent/fresh outputs shared a 42-token prefix then diverged; all compact final state signatures differed; no same-boundary fresh-segmented control exists yet |
| prior g128 KV8 16K-ish | peak 14.7705832 GiB vs normal 15.2356711 GiB: -0.4650879 GiB; decode 9.4946 vs 10.6409 tok/s; coherent one-shot only |

All memory values above are binary GiB from saved byte counters. The earlier
decimal-GB observations remain labeled in their original artifacts.

## Environment and safety state

- Worktree was clean at recovery (`main`); current HEAD at recovery was
  `0992416` (`stage 3 prompts`).
- g64 path exists at the exact Hugging Face snapshot recorded in `AGENTS.md`.
- g128 path exists at `/Users/artem/models/Qwen3.8-27B-3bit-g128-MLX`.
- `envs/mlx-0.32.1` exists. Its `.venv` is bare, but the existing uv archive
  contains MLX 0.32.1, mlx-metal 0.32.1, and mlx-lm 0.31.3 from pinned git
  revision `254d153fdeb6f150edd4fc5a54f9828638481fa8`. A read-only archive
  path plus the required `uv run --project ... --no-sync` prefix successfully
  reported Python 3.14.7, MLX 0.32.1, GPU device, and the pinned mlx-lm
  source. No dependency upgrade or environment mutation was performed.
- No matching MLX/Qwen process was visible in the initial process check; all
  workloads launched in this run remain serialized.
- No wired-memory setting has been changed. Do not call `mx.set_wired_limit()`.

## Completed experiments / decisions

- Recovered and inspected Night-2 reports, raw suffix artifacts, context
  ladder, performance repeats, KV8 result, and NLL records.
- Preserved the Night-2 conclusion that the old 8K fragment protocol is
  invalid; its replay artifacts are not reused as correctness evidence.
- Reused the existing 4K exact-suffix result only as a pre-Phase-1 baseline.
- Did not rerun any completed Night-2 workload during Phase 0.

## Current best configuration

Before the Phase 1 gate, the strongest validated configuration is source-backed
g128 with direct `generate_step`, default MLX wired state, normal KV, and the
full-transcript correctness-safe mode. It is oracle-validated through the prior
approximately-16K safe run, but it re-prefills the whole transcript and is not
the desired warm-cache result. The practical warm-cache candidate remains
unvalidated pending Phase 1.

## Rejected configurations / hypotheses

- Historical 8K `P3-WARM8K` correctness label: rejected because turn 2 exactly
  replayed the first 64 tokens of turn 1.
- Old standalone user-fragment persistent protocol: invalid transcript
  serialization; do not scale it.
- Mixed 2-bit/6-bit g128, uniform 2-bit, DWQ, external drafts, MTP, and 32K
  chasing are out of scope for this stage.
- KV8 is not a default small-context mode; only its long-context safety value
  remains open.

## Current risks

- Same-boundary fresh-segmented vs true-persistent recurrent state has not been
  compared. A-vs-persistent divergence alone is not enough to reject exact
  suffix; B-vs-C is the Phase 1 gate.
- Long-context persistent scaling and endurance are unsafe to start until that
  gate is resolved.
- The archive-backed canonical launcher depends on protected external cache
  paths; each heavy command must retain the exact pinned versions and no-wired
  policy.

## Next action

Implement and run the Phase 1 g128 cache-boundary oracle. Save complete token
IDs/text and compact per-layer signatures at initial prefill, after turn 1,
after the exact suffix, after the first turn-2 token, and after turn 2. Append
machine-readable records and update this file before deciding whether Phase 2
or Phase 3 is allowed.

