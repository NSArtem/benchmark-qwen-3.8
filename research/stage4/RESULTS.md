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
