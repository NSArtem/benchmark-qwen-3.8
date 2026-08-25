# Stage 4 Results

## Phase 0 — recovery

### Measured facts

Stage 3 established a correct exact-suffix persistent protocol for the hybrid
Qwen3.8 cache. The g128 baseline is 10.1800222 GiB versus 10.9627218 GiB for
g64. At 8K/12K/16K-class contexts, g128 measured 13.983/14.609/15.236 GiB
peak and approximately 11.105/10.803/10.409 tok/s warm decode.

The Stage 3 endurance artifact shows turn 17 generated exactly four tokens,
`The smallest worker`, followed by EOS. The logical cache equations still held
and no replay was detected, so this is an unresolved generation-quality event,
not evidence of cache corruption.

### Inference

Quality Suite V2 must separate objective checks, rubric judgments, pairwise
judgment, and teacher-forced metrics. The old 64-token coding cap cannot be used
to make the final g64/g128 quality decision.

### Future hypothesis

A corrected suite may narrow the apparent g128 coding deficit, but the prior
teacher-forced NLL gap and observed arithmetic error make a quality difference
plausible. This will be tested, not assumed.

The local high-precision source is present at snapshot
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (55,562,855,904 bytes across 18
shards). The existing bounded converter self-test passed in the pinned MLX
0.32.1 environment: 498 quantization keys, text-only model schema, and zero
wired-limit calls. The archive-backed runtime helper only exposes existing
cached packages and the matching MLX 0.32.1 Metal library; it does not sync or
upgrade the canonical environment.

## Phase 1 — Quality Suite V2

The corrected suite has 12 calibration cases and 28 evaluation cases across
coding, instruction following, structured output, arithmetic/reasoning,
retrieval, and prose continuity. The frozen hashes are:

| split | cases | SHA-256 |
|---|---:|---|
| calibration | 12 | `e856f39622c48f29f913d65d55f18d4ff9fb9bca16b0b23736361347472b9558` |
| evaluation | 28 | `ee6e02f56479b16310a864503fc933f9f4928111739ccf5c06776a88918ef721` |

The evaluation result is g64 23/24 objective and 4/4 semi-objective versus
g128 23/24 and 4/4. Both models fail only `reasoning-03`, where the correct
distinct event count is 3 and both answer 2. The final raw generations and the
rubric-only final rescoring are under
`research/stage4/artifacts/quality-v2/{g64,g128}`.

Teacher-forced V2 on six fixed reference answers (167 target tokens) produced
weighted mean NLL 1.4034 nats for g64 and 1.3690 for g128, delta −0.0344 nats;
weighted top-1 agreement was 70.06% and 69.46%. This is supplementary evidence
and is not a universal quality claim.

## Phase 2 — Turn-17 early-EOS control

The focused control used the exact saved 10,439-token turn-17 transcript and
three arms per model where valid: fresh full transcript, fresh segmented
prefix/suffix, and sequential persistent reproduction. g128 produced EOS after
5/5/4 tokens respectively; the persistent reproduction matched all 16 saved
prior generations and both cache equations. g64 fresh one-shot and segmented
both produced an 11-token incomplete answer and EOS with valid accounting.

For g128, EOS was rank 2 on the first generation step in all valid arms, with
log-probability −3.25 (one-shot/persistent) or −3.125 (segmented). For g64 it
was rank 4 with −4.75 in both valid fresh arms. The g64 sequential arm is
marked invalid because it used g128's prior answer stream and none of its 16
prior generations matched. The evidence closes the cache-corruption question:
this is a generation-quality/segmentation-sensitive event, not stale replay or
cache-accounting failure.

## Phase 3 — Bounded quantization sensitivity

### Measured facts

The pinned Dynamic Quantization implementation was inspected locally and
against the current upstream source. Its sensitivity estimator deep-copies the
full model, a quantized copy, and the quantizable layer set, so it was not run
against the 27B model on this 18 GB machine. The bounded replacement kept the
g128 model resident, loaded source tensors one converter chunk at a time, and
measured a temporary affine 4-bit/group-128 output against the resident
3-bit/group-128 module on all 12 frozen calibration cases.

The complete map contains 498 module records across all 64 layers and 18
source shards. It took 617.7 seconds, peaked at 13.807 GiB MLX memory, and
reached 11.439 GiB active+cache. The raw map is
`research/stage4/artifacts/sensitivity/sensitivity-map.json`; the summarized
ranking and manifests are in
`research/stage4/artifacts/sensitivity/sensitivity-summary.json`.

The largest normalized proxy values were not uniform by architecture: selected
late regular-attention `o_proj`/`k_proj`, late linear-attention `out_proj` and
`in_proj_a`, and selected late MLP `up_proj`/`down_proj` ranked highly. The
embedding path had a high relative output delta but a large byte cost; `lm_head`
also had a measurable signal. These are module-output distortion rankings, not
claims of end-to-end teacher-loss sensitivity.

### Inference

The ranking supports spending extra bits selectively rather than globally.
At a 384 MiB extra-payload budget, the gain-per-parameter greedy selection
promoted 135 modules and predicted a 10.555 GiB model. The pinned mixed
predicate promoted 41 modules and predicted 10.665 GiB. These manifests were
used to build the two allowed Stage 4 candidates.

## Phase 4 — Candidate conversions

### Measured facts

| candidate | payload bytes | size GiB | effective bpw | high/low modules | shards | conversion time | max RSS | MLX peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mixed 3/4 control g128 | 11,451,399,168 | 10.664947 | 3.4061 | 41/457 | 21 | 27.2 s | 3.10 GiB | 3.478 GiB |
| sensitivity 3/4 g128 | 11,333,362,688 | 10.555017 | 3.3710 | 135/363 | 21 | 27.7 s | 3.44 GiB | 3.478 GiB |

Both conversions used the existing bounded layer-oriented converter, the local
source snapshot, and no wired-memory change. An invalid all-3-bit diagnostic
from the first manifest suffix check is preserved under the explicit
`*-invalid-base` path and is not counted as a candidate. No 3/6 candidate was
created because V2 did not show a g128 quality gap requiring outlier repair.

## Phase 5 — Mini tournament

### Measured facts

The fixed eight-case tournament covered coding, instruction following,
structured output, reasoning, retrieval, and prose. g64 and g128 each scored
6/7 objective and 1/1 semi-objective, missing only the known `reasoning-03`
arithmetic case. The mixed 3/4 control scored 7/7 objective and 1/1
semi-objective, with all cases ending by EOS, and was advanced to the full
suite. The sensitivity 3/4 candidate scored 6/7 objective and 0/1
semi-objective: it repeated the arithmetic miss and changed the supplied west
room to east in `prose-01`. It was rejected before full evaluation.

Raw mini-tournament outputs, token streams, checks, and memory records are in
`research/stage4/artifacts/mini-tournament/`.

### Inference

The mixed control is the only new candidate with enough corrected-suite
evidence to justify a full comparison. Its mini result is a gate result, not a
claim that it is globally better than g64 or g128; Phase 6 will make that
classification on the frozen 28-case evaluation.

## Phase 6 — Full finalist comparison

### Measured facts

The finalist run initially exposed one objective rubric false negative in
`coding-03`: Candidate A used “single, atomic transaction boundary,” which is
the requested fix but did not contain the rubric's contiguous “single
transaction” phrase. The rubric was corrected to accept `transaction boundary`
and `atomic transaction`, and `coding-03` was rerun for all three models. All
three passed the corrected case. The final evaluation hash is
`f24e3983efbd898084852c3fde7b0f824604bc6a63349de5f32efeaa36ca3de0`.

| model | objective | semi-objective | failures | mean decode | max eval peak |
|---|---:|---:|---|---:|---:|
| g64 | 23/24 | 4/4 | reasoning-03 | 11.075 tok/s | 14.143 GiB |
| g128 | 23/24 | 4/4 | reasoning-03 | 11.935 tok/s | 13.361 GiB |
| mixed 3/4 control | 24/24 | 4/4 | none | 11.362 tok/s | 13.846 GiB |

All three models had zero repetition detections. g64 and g128 ended all 28
cases by EOS. Candidate A ended 27 by EOS and reached its 512-token cap on
coding-07 after satisfying every objective check; it was not marked wrong for
the cap.

Teacher-forced metrics on the same six reference answers and 167 target tokens:

| model | weighted mean NLL | delta vs g64 | weighted top-1 | delta vs g64 |
|---|---:|---:|---:|---:|
| g64 | 1.4034 | 0.0000 | 70.06% | 0.00 pp |
| g128 | 1.3690 | −0.0344 | 69.46% | −0.60 pp |
| mixed 3/4 control | 1.3855 | −0.0180 | 70.06% | 0.00 pp |

### Inference

On this corrected suite, g128 is not materially worse than g64 on objective
quality: it is tied on objective rate and slightly better on this teacher NLL
subset, with a 7.8% higher evaluation decode mean and 0.783 GiB less payload.
Candidate A is the strongest measured practical compromise in this study: it
passed every corrected objective case, retained g64's teacher top-1 rate, and
cost 0.485 GiB over g128 while remaining 0.298 GiB smaller than g64.

Provisional roles are Candidate A as DEFAULT, g64 as conservative QUALITY /
reference mode, and g128 normal-KV as PERFORMANCE/LONG-CONTEXT mode. The role
decision remains subject to persistent 8K/16K validation.

## Phase 7 — Persistent runtime validation

### Measured facts

Candidate A mixed-3/4 group-128 was run with the validated exact-canonical-
suffix protocol in fresh processes at both target contexts. Each run used five
turns, deterministic generation, 128 initial output tokens, 64 follow-up
output tokens, normal KV, and a 16.2 GiB unattended stop.

| context | turns | actual initial prompt | cold TTFT | warm TTFT mean (range) | warm decode mean | peak | active+cache max | result |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 8K | 5/5 | 8,191 | 106.862 s | 0.989 s (0.712–1.088) | 10.743 tok/s | 14.468 GiB | 11.399 GiB | PASS |
| 16K | 5/5 | 16,727 | 225.670 s | 1.174 s (0.846–1.296) | 5.719 tok/s | 15.721 GiB | 11.904 GiB | PASS |

All ten turns were COHERENT. All logical prefix/suffix equations and
cache-after-length checks held; no stale replay, exception, or safety stop was
observed. Follow-up suffixes were 33–39 tokens. The raw per-turn records,
generated outputs, exact-suffix token files, cache signatures, and summaries
are in `research/stage4/artifacts/runtime-validation/mixed34/`.

### Inference

The selected default is runtime-correct and practical for 8K and 16K
conversations. Warm TTFT remains near one second at both contexts. Its 16K
decode is materially slower than the earlier same-harness g128 normal-KV
baseline (~10.409 tok/s), so g128 retains the PERFORMANCE/LONG-CONTEXT role;
this is a measured performance tradeoff of Candidate A, not a cache failure.

## Phase 8 — Production qwen38-chat

### Measured facts

`research/runtime/qwen38_chat.py` now provides a single-user production CLI
around the validated direct MLX API. It keeps the full logical transcript,
renders the canonical chat template every turn, checks the measured cache
offset, and sends only the exact suffix. `/reset`, `/stats`, `/context`, and
`/quit` are implemented; greedy non-thinking chat is the default, sampling is
available with `--mode sample`, and `--kv-bits 8` is optional. Generation
exceptions and Ctrl-C reset the session before another turn. No wired-limit
setting is called.

The six no-model token/state regression tests pass in the canonical environment.
They cover later-turn BOS exclusion, exact suffix calculation, cached prefix
accounting, stale-prefix rejection, short-answer replay heuristic behavior,
EOS-boundary normalization, and reset clearing. A two-turn real-model smoke
also passed: exact suffixes were 18 and 20 tokens, the final logical context
was 42 tokens, both responses had `replay=false`, and the measured peaks were
10.931 and 10.944 GiB. The raw capture is
`research/stage4/artifacts/runtime-validation/qwen38-chat-smoke/cli-smoke.txt`.

### Inference

The runtime is ready for normal single-user local chat/coding-agent use. It is
not an OpenAI-compatible endpoint and intentionally does not implement
multi-user scheduling or shared prefix caches. The first smoke caught and
corrected a false positive on a short repeated “OK” answer; exact cache offset
and transcript accounting remain the stronger correctness checks.
