# Qwen3.8 Stage 4 Final Report

Date: 2026-08-25  
Hardware: Apple M3 Pro, 18 GB unified memory, macOS 15.7.8  
Canonical runtime: Python 3.14.7, MLX 0.32.1, mlx-lm 0.31.3, pinned revision
`254d153fdeb6f150edd4fc5a54f9828638481fa8`

## Executive decision

### MEASURED FACT

- `g128` is not materially worse than `g64` on the corrected objective suite:
  both score 23/24 and fail the same unambiguous arithmetic case.
- The mixed 3/4 control is the strongest measured quality/size compromise:
  24/24 objective, 4/4 semi-objective, 10.665 GiB, and g64-matching teacher
  top-1 accuracy.
- The mixed 3/4 control passed five-turn exact-suffix persistent validation at
  both 8K and 16K-class contexts with no replay or accounting failure.
- The mixed candidate has a real 16K decode cost: 5.719 tok/s versus the
  previously measured g128 normal-KV baseline of about 10.409 tok/s. Its 8K
  decode was 10.743 tok/s. This performance difference is recorded rather than
  hidden by the quality result.

### INFERENCE — recommended roles

| role | model | reason |
|---|---|---|
| DEFAULT | `Qwen3.8-27B-mixed-3-4-control-g128` | Best corrected-suite quality signal while 0.298 GiB smaller than g64 and still safe at 16K. |
| QUALITY / REFERENCE | g64, group-64 | Conservative reference configuration; it was not a measured V2 quality winner, but remains useful when minimizing quantization risk. |
| PERFORMANCE / LONG-CONTEXT | g128, group-128, normal KV | Smallest and fastest measured weight quant; Stage 3 showed about 10.4 tok/s at 16K-class with 15.236 GiB peak. |

The DEFAULT choice prioritizes quality and correctness. Use g128 when long-
context decode throughput matters more than the mixed candidate's recovered
objective result.

## 1. What Stage 3 quality defects were corrected?

### MEASURED FACT

Quality Suite V2 corrected the following defects:

1. Coding and reasoning tasks received completion budgets up to 256–512 tokens
   instead of broadly using 64-token caps. A single intentionally short case
   retained a 64-token budget.
2. Ambiguous or wrong calibration expectations were separated from final
   evaluation. `cal-coding-02` and `cal-structured-02` remain calibration
   observations, not final quality claims.
3. Reasoning expectations were manually checked. The final `reasoning-03`
   arithmetic answer is 3; both baseline models answered 2, so that failure is
   retained as a model-quality result.
4. The finalist run exposed an objectively narrow `coding-03` rubric false
   negative: “single, atomic transaction boundary” expressed the requested
   invariant without the rubric's contiguous “single transaction” phrase. The
   rubric was corrected to accept `transaction boundary` and `atomic
   transaction`, and that case was rerun for all three finalists.
5. Objective checks, semi-objective prose checks, and teacher-forced metrics
   remain separate. Subjective judging was not used as ground truth.

## 2. Frozen Quality Suite V2

### MEASURED FACT

The evaluation split has 28 cases:

| category | cases |
|---|---:|
| coding | 8 |
| instruction | 4 |
| structured | 4 |
| reasoning | 4 |
| retrieval | 4 |
| prose / continuity | 4 |
| total | 28 |

The calibration split has 12 cases, two per category. The final evaluation
SHA-256 is
`f24e3983efbd898084852c3fde7b0f824604bc6a63349de5f32efeaa36ca3de0`.
The calibration SHA-256 is
`e856f39622c48f29f913d65d55f18d4ff9fb9bca16b0b23736361347472b9558`.
The pre-correction evaluation hash was
`ee6e02f56479b16310a864503fc933f9f4928111739ccf5c06776a88918ef721`.

Raw frozen inputs are `research/stage4/calibration/calibration_v2.jsonl` and
`research/stage4/evaluation/evaluation_v2.jsonl`. The teacher-forced subset is
six fixed reference answers and 167 target tokens.

## 3. Turn-17 early-EOS control

### MEASURED FACT

| arm | result |
|---|---|
| g128 fresh one-shot | 5 tokens, “The smallest worker method”, EOS; EOS rank 2, log-probability −3.25; accounting true |
| g128 fresh segmented | Same 5-token output; EOS rank 2, log-probability −3.125; accounting true |
| g128 persistent | 4 tokens, “The smallest worker”, EOS; EOS rank 2, log-probability −3.25; all 16 prior generations matched; accounting true |
| g64 fresh one-shot / segmented | Same incomplete 11-token response, valid accounting; EOS rank 4, log-probability −4.75 |
| g64 persistent diagnostic | Invalid arm: it was fed g128's prior generated streams, matched 0/16, and ended with false accounting; excluded |

### INFERENCE

The event is a model/quantization generation-quality and
segmentation-sensitive event, not evidence of persistent-cache corruption. The
validated B-vs-C cache oracle remains closed.

Raw evidence is under `research/stage4/artifacts/early-eos/`.

## 4. Final quality comparison

### MEASURED FACT

| model | objective | semi-objective | objective failure | mean decode | max eval peak | max active+cache |
|---|---:|---:|---|---:|---:|---:|
| g64 | 23/24 | 4/4 | `reasoning-03` | 11.075 tok/s | 14.143 GiB | 11.462 GiB |
| g128 | 23/24 | 4/4 | `reasoning-03` | 11.935 tok/s | 13.361 GiB | 10.726 GiB |
| mixed 3/4 control | 24/24 | 4/4 | none | 11.362 tok/s | 13.846 GiB | 11.211 GiB |

All 28 generations had zero repetition detections. g64 and g128 ended all
cases by EOS. The mixed candidate ended 27 by EOS and reached its 512-token
cap on `coding-07` after satisfying all objective checks.

### Teacher-forced metrics

| model | weighted NLL | delta vs g64 | weighted top-1 | delta vs g64 |
|---|---:|---:|---:|---:|
| g64 | 1.4034 | 0.0000 | 70.06% | 0.00 pp |
| g128 | 1.3690 | −0.0344 nats/token | 69.46% | −0.60 pp |
| mixed 3/4 control | 1.3855 | −0.0180 nats/token | 70.06% | 0.00 pp |

### INFERENCE

On this corrected suite, g128 is not materially worse than g64. Its objective
score is tied, its teacher-forced NLL is slightly lower, and its top-1 accuracy
is 0.60 percentage points lower on the small supplementary subset. The mixed
candidate improves the discrete objective result without showing a teacher
NLL/top-1 penalty against g64.

The raw comparison is
`research/stage4/artifacts/quality-v2/final-comparison.json`.

## 5. Sensitivity map

### MEASURED FACT

The bounded map kept g128 resident and loaded high-precision source tensors one
converter chunk at a time. For each module it compared temporary affine
4-bit/group-128 output with the resident 3-bit/group-128 output over the 12
calibration cases.

- 498 module records, 64 layers, 65 processed chunks, 18 source shards.
- Wall time: 617.661 s.
- Maximum MLX peak: 13.806894 GiB.
- Maximum active+cache: 11.438730 GiB.
- Map SHA-256:
  `463c1e3a58b00219217b3452f6ad4d28ca1341a65dbfffc6670c289573af6bbb`.
- This is an activation-weighted output-distortion promotion proxy, not a
  full-teacher KL, perplexity, or gradient sensitivity measurement.

Family-level mean relative output deltas were:

| family | module count | mean relative RMS delta | highest proxy path |
|---|---:|---:|---|
| embedding | 1 | 0.2719 | `model.embed_tokens` |
| MLP | 192 | 0.2106 | `layers.38.mlp.up_proj` by relative RMS; `layers.63.mlp.down_proj` by energy/parameter |
| self-attention | 64 | 0.1939 | `layers.43.self_attn.o_proj` by relative RMS |
| linear attention | 240 | 0.1897 | `layers.62.linear_attn.out_proj` by relative RMS; `layers.60.linear_attn.in_proj_a` by energy/parameter |
| lm_head | 1 | 0.1442 | `lm_head` |

The highest useful ranking concentration was in selected late regular-
attention `o_proj`/`k_proj`, late linear-attention `out_proj`/`in_proj_a`, and
late MLP `up_proj`/`down_proj`. The embedding has high relative distortion but
is expensive to promote.

Greedy proxy budget estimates, not built models, were:

| extra payload budget | promoted modules | predicted payload |
|---:|---:|---:|
| +100 MiB | 91 | 10.278 GiB |
| +200 MiB | 89 | 10.375 GiB |
| +400 MiB | 153 | 10.571 GiB |

The selected 384 MiB sensitivity manifest promoted 135 modules and predicted
10.555 GiB. The reusable map and ranking are in
`research/stage4/artifacts/sensitivity/`.

### INFERENCE

The map supports selective extra precision, but its ranking should not be
treated as equivalent to upstream Dynamic Quantization until a bounded
gradient/logit cross-check is performed.

## 6. Quant candidates

### MEASURED FACT

| candidate | recipe | 4-bit / 3-bit modules | payload | effective bpw | shards | conversion |
|---|---|---:|---:|---:|---:|---:|
| A: mixed 3/4 control g128 | pinned-style mixed predicate | 41 / 457 | 11,451,399,168 B = 10.665 GiB | 3.4061 | 21 | 27.238 s; MLX peak 3.478 GiB |
| B: sensitivity 3/4 g128 | 384 MiB proxy gain/parameter rank | 135 / 363 | 11,333,362,688 B = 10.555 GiB | 3.3710 | 21 | 27.660 s; MLX peak 3.478 GiB |

The conversion completion records show about 3.10 GiB max RSS for both
candidate processes; the Phase 4 machine-readable ledger retained a more
conservative 3.44 GiB maximum for Candidate B. No Candidate C targeted 3/6
model was built: the corrected suite showed no unresolved g128 quality gap and
Candidate B already failed the mini gate.

## 7. Rejected branches

### MEASURED FACT

- Candidate B scored 6/7 objective and 0/1 semi-objective in the mini
  tournament. It repeated the arithmetic miss and changed the supplied
  west-room fact to east. It was rejected without a full suite run.
- The invalid all-3-bit diagnostic output is preserved separately and is not a
  candidate.
- Uniform 2-bit and `mixed_2_6` remain Stage 3 rejections for unacceptable
  quality/pathological repetition.
- No generic uniform 4-bit, 32K, DSpark, MTP, or other quant-zoo branch was
  reopened.

## 8. Persistent runtime validation

### MEASURED FACT

Candidate A ran in fresh processes for five exact-suffix turns at each context.
Every turn was COHERENT, had exact logical prefix/suffix accounting and
cache-after-length equality, and had no replay, exception, or safety stop.

| context | actual initial prompt | cold TTFT | warm TTFT mean (range) | warm decode mean | peak | max active+cache |
|---|---:|---:|---:|---:|---:|---:|
| 8K | 8,191 tokens | 106.862 s | 0.989 s (0.712–1.088) | 10.743 tok/s | 14.468 GiB | 11.399 GiB |
| 16K-class | 16,727 tokens | 225.670 s | 1.174 s (0.846–1.296) | 5.719 tok/s | 15.721 GiB | 11.904 GiB |

Follow-up suffixes were 33–39 tokens. The 16K raw output records include five
cache signatures and five exact-suffix token files. Raw artifacts are under
`research/stage4/artifacts/runtime-validation/mixed34/`.

### Recommended commands

8K default, normal KV:

```sh
source research/stage4/canonical_env.sh
uv run --project envs/mlx-0.32.1 --no-sync python research/runtime/qwen38_chat.py \
  --model /Users/artem/models/Qwen3.8-27B-mixed-3-4-control-g128 \
  --max-tokens 512
```

16K-class default, normal KV:

```sh
source research/stage4/canonical_env.sh
uv run --project envs/mlx-0.32.1 --no-sync python research/runtime/qwen38_chat.py \
  --model /Users/artem/models/Qwen3.8-27B-mixed-3-4-control-g128 \
  --max-tokens 512
```

The same command is used because the runtime follows the logical context and
cache offset; no fixed 8K/16K mode switch is needed. `--kv-bits 8` is exposed
as an optional headroom mode, but Candidate A + KV8 was not separately
benchmarked in Stage 4. Stage 3 measured g128 normal-KV versus KV8 at 16K:
KV8 saved about 0.465 GiB and reduced warm decode by about 9.4% with nearly
unchanged TTFT and passing correctness.

## 9. Production runtime

### MEASURED FACT

`research/runtime/qwen38_chat.py` is productionized as a single-user,
single-conversation CLI. It supports:

- `--model PATH`;
- canonical non-thinking chat by default;
- deterministic greedy mode and `--mode sample`;
- `/reset`, `/quit`, `/stats`, and `/context`;
- `--kv-bits 8`;
- TTFT, suffix-token count, decode rate, peak, active+cache, finish reason,
  and replay diagnostic metrics;
- reset-on-error and reset-on-Ctrl-C safety behavior.

Six lightweight token/state regression tests pass without loading the 27B
model. A real two-turn smoke passed with 18-token and 20-token suffixes, 42
cached logical tokens, replay false on both turns, and 10.931/10.944 GiB
peaks. The documented command is in
`research/runtime/README.md`.

There is no OpenAI-compatible endpoint. This is intentional: concurrent
batching, shared prefix caches, multi-user rotation, and speculative cache
trimming are outside Stage 4.

## 10. Learned-quant feasibility and Stage 5

### MEASURED FACT

Static source inspection was completed with no learned-quant execution:

| method | Stage 4 result |
|---|---|
| Dynamic Quantization | Unsafe as pinned because of full-model/deep-copy sensitivity estimation and repeated full-model threshold search. |
| DWQ | Unsafe and non-reproducible as pinned: full teacher/student residency, remote default data, and a `mx.set_wired_limit` call. |
| AWQ | Potentially block-bounded, but no `qwen3_5` configuration and no Qwen3.8 hybrid block adapter. |
| GPTQ | Promising, but all-layer Hessian capture is not bounded enough; requires serial layer/block adaptation and local fixed token data. |

The pinned source SHA-256 values and official upstream source links are in
`research/stage4/learned-quant-feasibility.md` and
`research/stage4/artifacts/learned-quant/source-inspection.json`.

### INFERENCE — ranked Stage 5 recommendations

1. Implement bounded gradient/logit Dynamic Quant sensitivity on the existing
   local calibration split and compare its ranking with the Stage 4 proxy.
2. Build a serial GPTQ smoke for the highest-ranked modules with an explicit
   Hessian memory cap and no whole-model Hessian residency.
3. If a held-out quality gap remains, adapt AWQ to Qwen3.5 hybrid blocks. DWQ
   should wait for an offline teacher-target and layer/block-wise distillation
   design.

## 11. Highest-value next experiments

These are bounded follow-ups, not an authorization to continue an open-ended
quant sweep:

1. Run one apples-to-apples 16K mixed34 versus g128 throughput control in the
   current CLI/harness to isolate the mixed-kernel long-context slowdown.
2. Compare the bounded gradient/logit sensitivity ranking with the existing
   output-distortion map on held-out local calibration prompts, then build at
   most one candidate if the ranking materially changes.
3. Add a small held-out coding/reasoning suite with explicit 256–512-token
   budgets and a blind review subset to test whether Candidate A's 24/24 result
   persists outside the tuned Stage 4 evaluation set.

## 12. Final Pareto table

### MEASURED FACT

`n/m` means that Stage 4 did not run that model at that context; it is not an
estimate.

| model | size GiB | effective bpw | quality verdict | 8K peak | 16K peak | decode | warm TTFT | recommended role |
|---|---:|---:|---|---:|---:|---:|---:|---|
| g64 | 10.963 | 3.5012 | 23/24 objective, conservative reference | n/m | n/m | n/m | n/m | QUALITY / REFERENCE |
| g128 | 10.180 | 3.2513 | 23/24 objective; fastest baseline | 13.983 GiB | 15.236 GiB | 11.105 / 10.409 tok/s at 8K / 16K-class | ~1.071 / 1.205 s | PERFORMANCE / LONG-CONTEXT |
| mixed 3/4 control | 10.665 | 3.4061 | 24/24 objective; 4/4 semi; validated persistent | 14.468 GiB | 15.721 GiB | 10.743 / 5.719 tok/s at 8K / 16K-class | 0.989 / 1.174 s | DEFAULT |

The g128 numbers in this table are measured Stage 3 normal-KV persistent
scaling results; the mixed numbers are the Stage 4 five-turn validation. The
g64 long-context cells are intentionally not filled from inference.

## 13. Commit record

Logical Stage 4 commits:

- `71cf49f` — initialize Stage 4 state;
- `34bad71` — Quality Suite V2 and early-EOS controls;
- `757e0cc` — sensitivity map, bounded converter recipes, candidates, and mini tournament;
- `f03293b` — corrected full finalist quality comparison;
- `1f03d34` — mixed-3/4 8K/16K persistent validation;
- `648eb52` — production qwen38-chat runtime, tests, and documentation;
- the final Stage 4 documentation commit includes this report and the learned-
  quant feasibility assessment.

No model weights, high-precision source checkpoint, or unrelated
`archive_3.8_stage3.zip` file is committed.
