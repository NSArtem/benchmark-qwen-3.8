# Stage 4 Status

Last updated: 2026-08-25 — Phases 0–7 complete; Phase 8 in progress.

## Current phase

Phase 8: productionize the validated exact-suffix chat runtime. Phases 0–7 are
complete, including the Quality Suite V2 validation gate, focused early-EOS
control, bounded sensitivity map, two candidate conversions, the mini
tournament, corrected full finalist comparison, and mixed-3/4 8K/16K runtime
validation.

## Completed experiments

Phase 0 recovery, the bounded converter self-test, Quality Suite V2, the
teacher-forced subset, the early-EOS control, sensitivity mapping, candidate
conversion, and the mini tournament are complete. Raw model runs are under
`research/stage4/artifacts/`; rubric-only rescoring is separated from the
preserved generation runs.

## Phase 1 — Quality Suite V2

- Frozen calibration split: 12 cases, SHA-256
  `e856f39622c48f29f913d65d55f18d4ff9fb9bca16b0b23736361347472b9558`.
- Frozen evaluation split: 28 cases, final corrected SHA-256
  `f24e3983efbd898084852c3fde7b0f824604bc6a63349de5f32efeaa36ca3de0`.
  The pre-correction run hash was
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
- After the finalist run, `coding-03` was corrected to accept semantically
  equivalent `transaction boundary`/`atomic transaction` wording for the
  required single-transaction boundary. That exact case was rerun for g64,
  g128, and Candidate A; all three passed. The correction is preserved in the
  final hash and raw rerun artifacts.

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

## Phase 3 — Bounded sensitivity map

- Upstream/pinned Dynamic Quantization was inspected. Its estimator deep-copies
  the full teacher/model/layer set and was not run on this 27B/18 GB machine.
- The bounded calibration-activated proxy measured temporary affine 4-bit/g128
  module output distortion against the resident g128 module over all 12 frozen
  calibration cases. It is a promotion-ranking proxy, not a full-teacher KL or
  perplexity sensitivity.
- Complete map: 498 modules, 64 layers, 18 source shards; calibration SHA-256
  `e856f39622c48f29f913d65d55f18d4ff9fb9bca16b0b23736361347472b9558`.
- Map wall time was 617.7 seconds; maximum MLX peak was 13.807 GiB and maximum
  active+cache was 11.439 GiB. No memory guard or safety stop fired.
- Highest normalized proxy sensitivities were concentrated in selected late
  self-attention `o_proj`/`k_proj`, late linear-attention `out_proj` and
  `in_proj_a`, selected late MLP `up_proj`/`down_proj`, plus the embedding
  output path. Family means and full rankings are in
  `artifacts/sensitivity/sensitivity-summary.json`.
- The pinned mixed-3/4 predicate selected 41 modules and predicted 10.665 GiB;
  the budgeted sensitivity rank selected 135 modules and predicted 10.555 GiB
  at a 384 MiB extra-payload budget.

## Phase 4 — Candidate conversions

- Candidate A, `mixed-3-4-control-g128`,
  `/Users/artem/models/Qwen3.8-27B-mixed-3-4-control-g128`: 11,451,399,168
  bytes = 10.664947 GiB, effective 3.4061 bpw, 41×4-bit and 457×3-bit
  modules, 21 shards. Conversion took 27.2 seconds; max RSS was 3.10 GiB and
  MLX peak was 3.478 GiB.
- Candidate B, `sensitivity-3-4-gain-per-parameter`,
  `/Users/artem/models/Qwen3.8-27B-sensitivity-3-4-g128`: 11,333,362,688
  bytes = 10.555017 GiB, effective 3.3710 bpw, 135×4-bit and 363×3-bit
  modules, 21 shards. Conversion took 27.7 seconds; max RSS was 3.44 GiB and
  MLX peak was 3.478 GiB.
- No Candidate C (targeted 3/6) was justified: corrected V2 showed no g128
  quality gap requiring an outlier repair, and the mini gate already rejected
  the guided 3/4 candidate.
- One invalid diagnostic conversion briefly produced an all-3-bit copy because
  of the corrected manifest suffix check; it is preserved as
  `Qwen3.8-27B-sensitivity-3-4-g128-invalid-base` and is not a candidate.

## Phase 5 — Mini tournament

- Fixed subset: 8 evaluation cases spanning coding, instruction, structured,
  reasoning, retrieval, and prose. g64 and g128 each scored 6/7 objective and
  1/1 semi-objective; both missed only `reasoning-03`.
- Candidate A scored 7/7 objective and 1/1 semi-objective, with all cases ending
  in EOS, and advanced to the full finalist comparison.
- Candidate B scored 6/7 objective and 0/1 semi-objective: it repeated the
  known arithmetic miss and changed the supplied west-room fact to east. It is
  rejected without a full suite run.

## Phase 6 — Full finalist comparison

- Final corrected evaluation hash:
  `f24e3983efbd898084852c3fde7b0f824604bc6a63349de5f32efeaa36ca3de0`.
- g64: 23/24 objective, 4/4 semi-objective; only `reasoning-03` failed.
- g128: 23/24 objective, 4/4 semi-objective; only `reasoning-03` failed.
- Candidate A mixed-3/4: 24/24 objective, 4/4 semi-objective. It passed the
  corrected `coding-03` and also answered `reasoning-03` correctly. All 28
  cases had no repetition; 27 ended by EOS and coding-07 reached its length
  cap after satisfying the objective checks.
- Full-suite maximum generation peaks were g64 14.143 GiB, g128 13.361 GiB,
  and Candidate A 13.846 GiB. Mean evaluation decode was 11.075, 11.935,
  and 11.362 tok/s respectively.
- Teacher-forced six-case subset: g64 weighted NLL 1.4034/top-1 70.06%, g128
  1.3690/69.46%, Candidate A 1.3855/70.06%. Candidate A's NLL delta versus
  g64 was −0.0180 nats/token and its top-1 delta was 0.00 percentage points.
- Role decision entering runtime validation: Candidate A is the DEFAULT
  finalist; g128 is the PERFORMANCE/LONG-CONTEXT mode; g64 remains the
  conservative quality/reference mode rather than a measured winner.

## Phase 7 — Persistent runtime validation

- Selected default: Candidate A mixed-3/4 group-128 at
  `/Users/artem/models/Qwen3.8-27B-mixed-3-4-control-g128`.
- 8K exact-suffix persistent run: 5/5 turns PASS. The actual initial prompt
  was 8,191 tokens. Cold TTFT was 106.862 s; warm TTFT mean was 0.989 s
  (0.712–1.088 s); warm decode mean was 10.743 tok/s. Peak was 14.468 GiB
  and maximum active+cache was 11.399 GiB.
- 16K exact-suffix persistent run: 5/5 turns PASS. The actual initial prompt
  was 16,727 tokens. Cold TTFT was 225.670 s; warm TTFT mean was 1.174 s
  (0.846–1.296 s); warm decode mean was 5.719 tok/s. Peak was 15.721 GiB
  and maximum active+cache was 11.904 GiB, below the 16.2 GiB unattended
  safety stop.
- All ten turns were classified COHERENT. Every turn satisfied exact logical
  prefix/suffix accounting and cache-after-length checks; no stale replay,
  exception, or safety stop occurred. Follow-up suffixes were only 33–39
  tokens. Raw per-turn prompts, suffixes, generated tokens, cache signatures,
  outputs, and persistent summaries are under
  `research/stage4/artifacts/runtime-validation/mixed34/`.
- The 16K mixed-3/4 decode mean is materially below the earlier same-harness
  g128 normal-KV baseline (~10.409 tok/s), while 8K is ~10.743 tok/s. This is
  a measured long-context performance cost of the selected mixed candidate,
  not a correctness failure; g128 remains the performance/long-context role.

## Phase 8 — Production runtime

- In progress. The next action is to implement and test the user-facing
  `research/runtime/qwen38_chat.py` exact-suffix CLI, then document its command
  and safe reset/error behavior.

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

- g64 reference: 11,771,132,928 bytes = 10.962722 GiB, effective 3.5012 bpw.
- g128 baseline: 10,930,715,648 bytes = 10.180022 GiB, effective 3.2513 bpw.
- Candidate A: 11,451,399,168 bytes = 10.664947 GiB, effective 3.4061 bpw.
- Candidate B: 11,333,362,688 bytes = 10.555017 GiB, effective 3.3710 bpw.

## Rejected candidates

Stage 3: uniform 2-bit and mixed_2_6 were rejected for unacceptable quality /
pathological repetition. Stage 4 Candidate B was rejected at the mini gate for
the prose fact-preservation failure and no objective improvement. The invalid
all-3-bit manifest diagnostic is excluded from candidate counts. No targeted
3/6 candidate was built.

## Current best configuration

Candidate A mixed-3/4 g128 is the provisional DEFAULT after the corrected full
suite: 24/24 objective and 4/4 semi-objective. g128 normal KV remains the
PERFORMANCE/LONG-CONTEXT mode at 10.180 GiB and the fastest measured decode;
g64 remains the conservative QUALITY/reference mode. Persistent validation is
complete; runtime productionization is still outstanding.

## Active risks

- any new converter must remain layer-bounded on the 18 GB machine.
- the source snapshot is local and must not be re-downloaded or committed.
- the sensitivity proxy must be clearly labeled if it is weight-distortion
  rather than activation-aware; calibration ranking must not be overclaimed.
- The production CLI and regression tests are not yet complete.
- production CLI regression coverage and the learned-quant feasibility memo
  remain outstanding.

## Next action

Productionize and test `qwen38-chat`, then record the bounded learned-quant
Stage 5 feasibility recommendation.
