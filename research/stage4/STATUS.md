# Stage 4 Status

Last updated: 2026-08-25 — Phases 0–2 complete; Phase 3 in progress.

## Current phase

Phase 3: bounded quantization sensitivity mapping. Phases 0–2, including the
Quality Suite V2 validation gate and the focused early-EOS control, are complete.

## Completed experiments

Phase 0 recovery, the bounded converter self-test, Quality Suite V2, the
teacher-forced subset, and the early-EOS control are complete. Raw model runs
are under `research/stage4/artifacts/`; rubric-only rescoring is separated from
the preserved generation runs.

## Phase 1 — Quality Suite V2

- Frozen calibration split: 12 cases, SHA-256
  `e856f39622c48f29f913d65d55f18d4ff9fb9bca16b0b23736361347472b9558`.
- Frozen evaluation split: 28 cases, SHA-256
  `ee6e02f56479b16310a864503fc933f9f4928111739ccf5c06776a88918ef721`.
- Categories: coding 8, instruction 4, structured 4, reasoning 4, retrieval
  4, prose 4. Coding/reasoning budgets are 512/384 tokens; retrieval-01 is
  5,418 prompt tokens and was run from the local fixed ledger.
- Final objective score: g64 23/24; g128 23/24. Final semi-objective prose
  score: 4/4 for both. Both models' sole substantive objective failure is the
  unambiguous arithmetic/deduplication case `reasoning-03` (they answer 2
  instead of 3).
- Per-category objective results are identical: coding 8/8, instruction 4/4,
  structured 4/4, reasoning 3/4, retrieval 4/4; prose is 4/4 semi-objective.
- All 56 evaluation generations completed by EOS; no evaluation case tripped
  the short-EOS threshold. Evaluation-run mean decode was 11.075 tok/s g64
  versus 11.935 tok/s g128; maximum observed evaluation generation peak was
  14.143 GiB g64 versus 13.361 GiB g128.
- Teacher-forced V2 subset (six fixed reference answers, 167 target tokens):
  weighted mean NLL g64 1.4034, g128 1.3690 (delta −0.0344 nats); weighted
  top-1 g64 70.06%, g128 69.46%. This supplementary subset does not show a
  material g128 NLL penalty.
- Calibration failures are retained as calibration observations, not final
  evaluation claims: `cal-coding-02` is semantically underspecified and
  `cal-structured-02` distinguishes a JSON object from a one-element list.

## Phase 2 — Turn-17 early EOS

- g128 fresh one-shot: 5 tokens, `The smallest worker method`, EOS; cache
  accounting true; EOS rank 2, log-probability −3.25.
- g128 fresh segmented: 5 tokens, same text, EOS; accounting true; EOS rank 2,
  log-probability −3.125.
- g128 sequential persistent reproduction: all 16 prior saved generations
  matched; 4 tokens, `The smallest worker`, EOS; accounting true; EOS rank 2,
  log-probability −3.25.
- g64 fresh one-shot and fresh segmented both produced the same incomplete
  11-token response (`The smallest change is to modify `apps/worker`) and EOS;
  both had valid cache accounting, with EOS rank 4 and log-probability −4.75.
- The g64 sequential arm is retained as an invalid diagnostic because it was
  fed g128's saved prior generated streams; none of its 16 prior generations
  matched and its final accounting was false. It is not used as evidence.
- Decision: the Stage 3 event is a model/quantization generation-quality and
  segmentation-sensitive event, not persistent-cache corruption. Do not reopen
  the validated cache oracle.

## Validated facts from Stage 3

- Hardware: Apple M3 Pro, 18 GB unified memory, macOS 15.7.8.
- Canonical runtime: Python 3.14.7, MLX 0.32.1, pinned mlx-lm revision
  `254d153fdeb6f150edd4fc5a54f982863848fa8`.
- g64: affine 3-bit/group-64, payload 11,771,132,928 bytes = 10.9627218 GiB.
- g128: source-backed affine 3-bit/group-128, payload 10,930,715,648 bytes =
  10.1800222 GiB; 840,417,280 bytes / 0.7826996 GiB smaller than g64.
- g128 persistent normal-KV scaling: 8K peak 13.983 GiB, warm TTFT median
  about 1.071 s, decode 11.105 tok/s; 12K peak 14.609 GiB, 1.117 s,
  10.803 tok/s; 16K-class peak 15.236 GiB, 1.205 s, 10.409 tok/s.
- Exact-suffix B-vs-C persistence matched token-for-token and cache signatures;
  this is closed unless new contradictory evidence appears.
- Stage 3 endurance stopped at turn 17 after a coherent/accounted session:
  four generated tokens, `The smallest worker`, then EOS; no replay and no
  accounting failure, but incomplete output classified `BROKEN`.
- Stage 3 quality suite was 38 cases/model but used many 64-token caps and had
  at least one ambiguous reasoning expectation. It is not final evidence.
- Existing rejected candidates: uniform 2-bit and mixed_2_6; do not retest.
- KV8 passed as an optional long-context lever: about 0.465 GiB peak saving and
  about 9.4% warm-decode penalty at 16K-class.

## Recovered paths

- g64: `/Users/artem/.cache/huggingface/hub/models--lukaskremla--Qwen3.8-27B-3bit-MLX-TextOnly/snapshots/c98bba5926f51fec1c8d8737e577221673f524d7`
- g128: `/Users/artem/models/Qwen3.8-27B-3bit-g128-MLX`
- high-precision source: `/Users/artem/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (55,562,855,904-byte payload, 18 shards)
- bounded converter: `research/harness/convert_qwen38_textonly_g128.py`
- existing runtime to extend: `research/runtime/qwen38_runtime.py`
- canonical archive-backed launcher helper: `research/stage4/canonical_env.sh`
- converter self-test: PASS; 498 quantization keys, text-only architecture,
  global norm shift 2.0, wired-limit calls 0.

## Candidate models and sizes

No Stage 4 candidates built yet. Do not count the existing g64/g128 baselines
as new candidates.

## Rejected candidates

Stage 3: uniform 2-bit and mixed_2_6 were rejected for unacceptable quality /
pathological repetition. Stage 4 has not added any rejection.

## Current best configuration

Pending sensitivity/candidate work, g128 normal KV remains the practical
performance baseline for approximately 8K initial context; g64 remains the
quality/reference baseline. Quality Suite V2 currently shows no objective-rate
or teacher-forced penalty for g128, so the default role remains provisional.

## Active risks

- any new converter must remain layer-bounded on the 18 GB machine.
- the source snapshot is local and must not be re-downloaded or committed.
- the sensitivity proxy must be clearly labeled if it is weight-distortion
  rather than activation-aware; calibration ranking must not be overclaimed.

## Next action

Inspect pinned/upstream quantization predicates and the existing bounded
converter, then run a representative bounded per-tensor/layer sensitivity
pass using local calibration metadata only. Produce the sensitivity ranking,
budget curve, and a smoke-tested conversion plan before creating any new
candidate.
