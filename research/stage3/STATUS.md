# Stage 3 status

Last updated: 2026-08-24 — Phase 1 same-boundary oracle passed.

## Current phase

Phase 1 complete with `VALIDATED_FOR_SCALING`; Phase 2 is skipped because the
B-vs-C gate passed. Phase 3 exact-suffix scaling is next.

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
- Phase 1 attempt 1 stopped after model load while converting the first
  bfloat16 logprob vector for the new diagnostic (`ValueError: bfloat16 is not
  a valid PEP 3118 buffer format string`). This is recorded as a harness
  compatibility failure, not inference/cache evidence; the path is patched to
  cast logprobs to MLX float32 before host conversion.
- `P1-SAME-BOUNDARY-G128` rerun completed A/B/C in 210.5768 s with no
  exception or safety stop. B and C both generated the preserved turn-1 hash
  `2090a312...` and turn-2 hash `810fea0c...`; both protocol equations held:
  measured cached prefix 4,192 + exact suffix 38 = canonical turn-2 length
  4,230. The final turn-2 B/C state signature was `1b123c1d...` at offset
  4,294 in all 48 recurrent and 16 attention layers. The suffix boundary
  callback showed offset 4,231; this is the pinned `generate_step` callback
  after the first scheduled generation step, not a pure suffix-only snapshot.
- B-vs-C state signatures and offsets matched at every required boundary;
  turn-1/turn-2 token streams were identical; first-token logprob max and mean
  absolute differences were both 0.0 for both generation stages, with 8/8
  top-k overlap. Both turn-2 outputs were coherent with no replay. The Phase
  1 gate is `PASS` and exact-suffix persistent cache is validated for scaling
  under these same boundaries.
- A's fresh full-transcript turn-2 output was also coherent and non-replaying,
  but its token stream shared only the first 42 tokens with C. This is the
  expected one-shot-versus-segmented sensitivity observation and is not a
  B-vs-C failure.

## Current best configuration

The strongest validated warm-cache candidate is now source-backed g128 with
direct `generate_step`, default MLX wired state, normal KV, exact canonical
suffixes, and one persistent cache. At the 4K oracle boundary its turn-2
suffix was 38 tokens with 1.0268 s TTFT and 11.4549 tok/s decode in the new
run (the preserved earlier run measured 0.9800 s and 11.6337 tok/s). The
full-transcript correctness-safe mode remains the fallback for contexts not
yet scaled under this gate.

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

- Phase 1 validated only the approximately-4K exact-suffix conversation;
  long-context persistent scaling and endurance still need direct evidence.
- The 4K A-vs-C token divergence is a measured segmentation sensitivity; it is
  not a proven universal equivalence rule for all prompt lengths.
- The archive-backed canonical launcher depends on protected external cache
  paths; each heavy command must retain the exact pinned versions and no-wired
  policy.

## Next action

Re-read `AGENTS.md`, `TASK.md`, and this file, then run Phase 3 with g128,
normal KV, exact suffixes, and a bounded realistic multi-turn prompt at about
8K, 12K, and 16K (reusing the prior 4K evidence where appropriate). Save a
fresh full-transcript semantic oracle at 8K and the largest successful context.
