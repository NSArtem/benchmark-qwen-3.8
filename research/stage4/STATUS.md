# Stage 4 Status

Last updated: 2026-08-25 — Phase 0 complete; Phase 1 in progress.

## Current phase

Phase 1: build and validate Quality Suite V2. Phase 0 recovery and scaffolding
are complete.

## Completed experiments

Phase 0 recovery and the bounded converter self-test are complete. Stage 3
evidence has been recovered from reports and raw artifacts.

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

Stage 3 practical baseline remains g128 normal KV for approximately 8K initial
context. g64 remains the quality/reference baseline. Final Stage 4 roles are
not assigned yet.

## Active risks

- g128 quality gap versus g64 is unresolved on a corrected suite.
- turn-17 early EOS needs one focused fresh/persistent and g64/g128 control.
- any new converter must remain layer-bounded on the 18 GB machine.
- the source snapshot is local and must not be re-downloaded or committed.

## Next action

Construct the local calibration/evaluation split, run g64 first, inspect every
objective failure, correct only objectively broken checks, then run g128 and
freeze the validated suite hashes.
