# Stage 5 Results

## Phase 0 — recovery

### Measured facts

The starting worktree was clean on `main`. Stage 4 artifacts were recovered
from `research/stage4/`. The local high-precision Qwen3.8-27B source is the
snapshot `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, recorded at 55,562,855,904
bytes across 18 shards. The bounded converter is
`research/harness/convert_qwen38_textonly_g128.py`; the existing validated
runtime is `research/runtime/qwen38_chat.py`.

Entering models are g64, g128, and the mixed34 control. Stage 4 measured V2
objective scores of 23/24, 23/24, and 24/24 respectively. Stage 4 mixed34
persistent runtime measured 10.743 tok/s at 8K-class and 5.719 tok/s at
16K-class, while the earlier g128 normal-KV baseline measured 11.105 and
10.409 tok/s at those approximate contexts.

### Inference

The mixed34 long-context cost is a high-priority reproducibility question, not
yet a localized memory-causality claim. Stage 5 must first interleave fresh
processes and correlate decode with context, active+cache, peak, RSS, and
system pressure.

## Phase 1 — mixed34/g128 decode-cliff localization (complete; safety-limited)

### Measured facts

The frozen Stage 5 cliff prompt family uses the Stage 4 coding-agent source
(`67dcccb037993b679c5593073bbc1875ad61541827a6fc3bb6720410b8096903`) and
the g128 tokenizer. It produces exact chat lengths 8,191, 10,240, 12,288,
14,336, and 16,727 tokens. The required first fresh 16K control,
`P1A-16K-G128-R1`, completed coherently with 192 generated tokens:

| model | prompt | TTFT | prefill tok/s | decode tok/s | peak | end active+cache | RSS max | finish |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| g128 | 16,727 | 242.118 s | 69.088 | 10.583 | 15.236 GiB | 11.403 GiB | 2.44 GiB | length |

Its raw OS-after snapshot reported 22% free memory and approximately 11.2 GiB
encrypted swap. The next interleaved mixed34 process loaded its 11,451,399,168
byte payload (10.666 GiB active) but was interrupted during prefill before a
first token after the system had accumulated approximately 14.4 GiB swap. It
is classified as a safety stop, not as a Metal OOM or quality result. No second
16K process was started concurrently.

### Safety inference

The requested six-run 16K interleave is not safe to repeat under the observed
system pressure. The gated isolated mixed34 retry completed after recovery and
reproduced the Stage 4 long-context result. A lower-context sweep will locate
the transition without repeatedly reproducing the same high-memory prefill
load. Stage 4's prior five-turn mixed34 data remains controlled supporting
evidence; the measurements establish correlation with context/residency, not
causality by themselves.

The isolated mixed34 retry was:

| model | prompt | TTFT | prefill tok/s | decode tok/s | peak | end active+cache | RSS max | finish |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| mixed34 | 16,727 | 240.786 s | 69.470 | 4.890 | 15.721 GiB | 11.873 GiB | 5.55 GiB | length |

The fresh 8K g128 point was 11.137 tok/s at 13.983 GiB peak. Its following
mixed34 process was intentionally interrupted after model load/setup when the
system fell to 32% free memory and ~13.7 GiB swap; it produced no tokens. The
remaining 10K/12K/14K sweep points were not run after the safety gate engaged.

### Phase 1 cliff table

| evidence | model | actual prompt | decode tok/s | peak GiB | result |
|---|---|---:|---:|---:|---|
| fresh Stage 5 | g128 | 8,191 | 11.137 | 13.983 | coherent |
| Stage 4 controlled | mixed34 | 8,191 | 10.743 mean | 14.468 | 5/5 persistent pass |
| Stage 4 controlled | g128 | 12,288 | 10.803 | 14.609 | prior baseline |
| Stage 4 controlled | g128 | 16,727 | 10.409 | 15.236 | prior baseline |
| fresh Stage 5 | g128 | 16,727 | 10.583 | 15.236 | coherent |
| Stage 4 controlled | mixed34 | 16,727 | 5.719 mean | 15.721 | 5/5 persistent pass |
| fresh Stage 5 | mixed34 | 16,727 | 4.890 | 15.721 | coherent |

### Phase 1 gate conclusion

The mixed34 long-context decode cliff is reproduced by an independent fresh
process: 4.890 tok/s versus 10.583 tok/s for fresh g128 at the same 16,727
tokens, and 5.719 tok/s in the earlier five-turn mixed34 control. The first
observed transition is bounded between the 8,191-token mixed34 result
(10.743 tok/s) and 16,727 tokens (4.890 tok/s); the exact onset between those
points is not measured because repeating the full prefill accumulated 12–14 GiB
swap and repeatedly crossed the system safety concern. The 16,727-token mixed34
peak is 15.721 GiB and its end active+cache is 11.873 GiB. These observations
establish a practical long-context penalty and a residency correlation, not a
causal proof that payload size alone causes the slowdown.

## Phase 2 — held-out baseline comparison (complete)

### Measured facts

The frozen suite was run once per model in a fresh serialized process. The raw
model records are `research/stage5/artifacts/heldout/{g64,g128,mixed34}/model.json`;
the suite hash was `6fe7bd7da6732e9057a75764292b18b29c3262a96eb39391b700f038f641851b`.
No result was used for calibration, sensitivity, allocation, or mini-selection.

| model | coding | instruction | structured | reasoning | retrieval | prose | objective | semi-objective | exceptions / early EOS / repetition |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| g64 | 5/6 | 4/4 | 4/4 | 2/4 | 3/3 | 2/3 | 18/21 | 2/3 | 0 / 0 / 0 |
| g128 | 5/6 | 4/4 | 4/4 | 1/4 | 3/3 | 3/3 | 17/21 | 3/3 | 0 / 0 / 0 |
| mixed34 | 5/6 | 4/4 | 4/4 | 2/4 | 3/3 | 2/3 | 18/21 | 2/3 | 0 / 0 / 0 |

All three models reproduced the six retrieval anchors and all instruction and
structured cases. The objective gap is concentrated in the deliberately hard
reasoning cases; g128 recovered every prose case but missed one additional
reasoning case. The held-out evidence therefore does not establish a quality
win for mixed34 over g128, while the Phase 1 runtime cliff remains a material
default-selection penalty.

### Phase 2 gate conclusion

The held-out suite is valid and provides genuinely independent final-selection
evidence. Mixed34 ties g64 and is one objective case above g128, but it does
not outperform g128 on the complete suite once semi-objective prose is included
(20/24 versus 20/24 when the two score families are reported together), and it
has a reproduced 16K decode cliff. Keep mixed34 as a quality alternate/control;
keep g128 as the practical default pending Stage 5 loss-guided candidates.

## Phase 3 — bounded end-to-end sensitivity (complete)

### Measured facts

The pinned learned-quant source was inspected at mlx-lm revision
`254d153fdeb6f150edd4fc5a54f9828638481fa8`. The exact file hashes and safety
findings are in `research/stage5/quant/upstream_inspection.json`. The pinned
Dynamic Quant flow deep-copies the full model and repeats that copy during
threshold search; the pinned DWQ flow keeps a full teacher/student path and
calls `mx.set_wired_limit`. Neither was run as-is.

The bounded replacement captured g64 logits for the 12-case Stage 4
calibration split in a separate process, then loaded g128 and promoted one
source-backed affine 4-bit/group-128 module at a time. No full source or full
teacher was resident. The teacher-logit payload is retained locally under
`research/stage5/quant/teacher_logits_g64_v1/` and is ignored by git.

| measurement | value |
|---|---:|
| calibration cases / target tokens | 12 / 477 |
| primary baseline KL to g64 teacher | 0.115868 |
| modules evaluated | 498/498 |
| sweep wall time | 4,579.2 s |
| maximum MLX peak | 14.480 GiB |
| maximum active+cache | 12.302 GiB |
| full ranking hash | `e9d561a11807810dc44a9a6e38080623f2b41207d03643515b0661c8a188dd6b` |

The primary score is final-logit teacher-KL recovery divided by bytes added;
teacher-argmax NLL recovery, top-1 agreement, and logit MSE are retained as
secondary signals. The top full-split slots were hybrid recurrent projections:

| rank | module | KL recovery / byte |
|---:|---|---:|
| 1 | layer 60 `linear_attn.in_proj_a` | 1.2313e-7 |
| 2 | layer 49 `linear_attn.in_proj_b` | 1.2210e-7 |
| 3 | layer 18 `linear_attn.in_proj_b` | 1.2174e-7 |
| 4 | layer 21 `linear_attn.in_proj_b` | 1.0835e-7 |
| 5 | layer 17 `linear_attn.in_proj_b` | 1.0491e-7 |

All top 25 were `linear_attn` modules. By family, the mean KL recovery was
0.000572 for `linear_attn`, 0.000203 for `self_attn`, and 0.000091 for `mlp`;
the embedding promotion was negative on this calibration signal.

### Stability and Stage 4 comparison

An all-layer two-case repeat (instruction plus reasoning) completed in 716.7 s
at 13.932 GiB peak. Its Spearman rank correlation with the 12-case ranking was
0.257; top-10 overlap was 3/10 and top-25 overlap was 8/25. This means the
module scores are deterministic within a run but sensitive to calibration
content. The full 12-case ranking is frozen for explicit allocation, with this
instability recorded as a limitation rather than hidden by threshold tuning.

The new ranking is not a restatement of Stage 4 RMS sensitivity: top-25 overlap
was 1/25 and all-module Spearman correlation was -0.121 against the old
relative-output-RMS order. The end-to-end result therefore materially replaces
the old ranking, but it is not yet sufficient to claim predictive validation on
an independently loaded candidate.

### Phase 3 gate conclusion

The bounded end-to-end KL/logit method is feasible and memory-safe under the
observed 18 GB constraints. It produces a concrete quality-recovery-per-byte
ranking with explicit deterministic ordering. Its low two-case stability and
the lack of a completed candidate-level calibration comparison require
caution: Q1/Q2/Q3 are research checkpoints, not quality finalists yet.

## Phase 4 — fixed-budget frontier (conversion complete; smoke gate safety-limited)

Three and only three new serious checkpoints were built from the frozen full
ranking. All use affine 3-bit/group-128 as the base and promote selected
modules to affine 4-bit/group-128:

| candidate | promoted modules | added payload | payload bytes | payload GiB | effective BPW | families |
|---|---:|---:|---:|---:|---:|---|
| Q1 | 112 | +150 MiB | 11,088,002,048 | 10.326507 | 3.2980 | 86 linear-attn, 24 self-attn, 2 MLP |
| Q2 | 137 | +300 MiB | 11,245,288,448 | 10.472991 | 3.3448 | 104 linear-attn, 25 self-attn, 8 MLP |
| Q3 | 161 | +450 MiB | 11,402,574,848 | 10.619475 | 3.3916 | 116 linear-attn, 30 self-attn, 15 MLP |

Conversion was bounded and recorded in `research/stage5/quant/Q{1,2,3}-conversion.jsonl`;
each conversion reported about 3.33 GiB converter peak and no wired-limit call.
The explicit manifest is `research/stage5/quant/frontier_manifests_v1.json`.

Q1's first short factual smoke loaded and generated a coherent answer at
11.728 tok/s, with 10.642 GiB peak. Its process ended at 26% free memory with
12.205 GiB encrypted swap used and only 1.107 GiB swap free. The next Q1
roughly-1K smoke was stopped before launch by the safety gate, so Q1 has not
completed the required multi-prompt smoke and Q2/Q3 have not been loaded.
No candidate was promoted to the tournament; no broad-3/4 versus sparse-3/5/6
experiment was attempted because no 5/6-bit branch was justified before the
memory gate.

### Phase 4 gate conclusion

The fixed-budget frontier exists on disk and obeys the candidate-count and
payload ladder limits, but the evidence gate is incomplete. Do not claim that
the KL ranking beats g128 or mixed34 until a safe fresh-process candidate-level
calibration and quality smoke can run. The current advanced roles remain g128
default and mixed34 quality alternate.
## Held-out suite definition (frozen)

The suite is generated by `research/stage5/harness/make_heldout_suite.py` and
frozen at `research/stage5/heldout/heldout_v1.jsonl`. It contains 24 manually
verified, local-only cases:

| category | cases | purpose |
|---|---:|---|
| coding/debugging | 6 | idempotency canonicalization, lease epochs, outbox delivery, reducer ordering, tenant predicates, SSE cursor race |
| instruction following | 4 | exact lines, bullet order, constrained terminology, exact sentinel |
| structured output | 4 | strict JSON keys/types/values |
| reasoning/arithmetic | 4 | queue arithmetic, monotonic versions, grouped totals, event de-duplication |
| context retrieval | 3 | new frozen ledger anchors at entries 014, 057, and 093 |
| prose/continuity | 3 | named facts, location continuity, sentence-shape rubrics |

The suite SHA-256 is
`6fe7bd7da6732e9057a75764292b18b29c3262a96eb39391b700f038f641851b`.
Six fixed reference answers are marked for supplementary teacher-forced NLL,
top-1, and optional logit metrics. The suite has no exact prompt or case-ID
overlap with Stage 4 calibration/evaluation and is not used for Stage 5
calibration, sensitivity, allocation, mini-tournament selection, or candidate
rejection. Baseline g64/g128/mixed34 runs are final-selection evidence only;
future candidates remain gated from this data until the final tournament.

## Phase 6 — bounded DWQ feasibility (complete; full-model run rejected)

### Measured facts

The pinned `dwq.py` implementation was inspected before any execution. It
keeps a full teacher/student path, uses full target files, and calls
`mx.set_wired_limit()` in its main flow, so it was not launched on the 18 GB
machine. A synthetic layer-sized replacement then optimized affine 3-bit
scales and biases against fixed teacher outputs through the MLX quantized matmul
path. It reduced local MSE from 0.881265 to 0.718006 in 40 steps, an 18.5%
reduction, with a maximum MLX allocation of only 13,600 bytes. The raw record
is `research/stage5/quant/dwq_bounded_dry_run.json` and the harness is
`research/stage5/harness/dwq_bounded_dry_run.py`.

The static minimum residency calculation is already disqualifying for the
naive whole-model path: the 55,562,855,904-byte high-precision source plus the
10,930,715,648-byte g128 student is 61.927 GiB before activations, optimizer
state, targets, or cache. Even g64 plus g128 is 21.143 GiB before those
overheads. The synthetic result therefore demonstrates that the local
scale/bias optimization is numerically workable, not that a 27B DWQ job is
safe.

### Phase 6 gate conclusion

No real 27B DWQ refinement was attempted. On this hardware, upstream whole-
model DWQ is rejected for safety and memory reasons. A future bounded DWQ
branch would require a genuinely streamed per-layer teacher/student design,
with measured residency and a new dry run; it is not worth opening during this
Stage 5 window after the candidate smoke gate already accumulated 12.2 GiB of
swap.

## Phases 5 and 7 — candidate/final tournaments (not run; safety gate)

The Q1 multi-prompt smoke gate did not complete: only its 37-token factual
smoke ran, while the next roughly-1K smoke was stopped before model load after
the system had 12.205 GiB encrypted swap used and 1.107 GiB free. Q2 and Q3
were therefore not loaded, and no new checkpoint was selected as a finalist.
The Stage 5 held-out suite remains reserved for final selection and was not
used to make this stop decision. The entering-model baselines remain the only
complete quality comparison: g128 remains the default, mixed34 remains a
quality alternate/control with a long-context penalty, and g64 remains the
reference.

## Phases 8–10 — activation, context, and MTP gates

### Measured facts

The pinned activation-quant path accepts only `nvfp4` and `mxfp8` weight modes
and replaces compatible weight modules with `QQLinear`. The selected g128
winner and all three Stage 5 frontier checkpoints use affine 3-bit/group-128,
so the current runtime rejects activation quantization before inference. No
format-changing activation experiment was opened under the no-quant-zoo rule.

The original high-precision source does contain an MTP block: 15 top-level
`mtp.*` BF16 tensors in shard 18, totaling 849,398,784 bytes (0.791 GiB).
However, the pinned Qwen3.5 runtime's sanitizer detects and removes all
`mtp.*` weights, and its text model has no native Qwen3.8 MTP head/path. The
validated text-only converter also intentionally excludes those top-level
keys. The exact static inspection is
`research/stage5/mtp/static_inspection.json`.

### Gate conclusions

Activation quantization is unsupported for the current affine winner and was
not run. MTP weights are present but incompatible with the pinned runtime;
conversion, speculative execution, and head fabrication were stopped. The
20K/24K context expansion was not launched after the 27B smoke runs caused
dangerous swap pressure; 32K remains closed. No 24K result exists.

## Phase 11 — serialized agent backend (protocol complete; model smoke deferred)

`research/runtime/qwen38_agent_server.py` now provides a single-user,
single-inference HTTP service with OpenAI-compatible
`/v1/chat/completions`, `/v1/models`, `/health`, `/metrics`, and session reset
routes. It uses the validated exact-suffix cache boundary, persistent
conversation state, deterministic or sampled generation, generic system/user/
assistant/tool text messages, metrics, and graceful reset-on-cache-error
behavior. It intentionally does not parse provider-specific tool-call objects;
those payloads are normalized as text.

The canonical protocol self-test and the existing runtime token tests pass (9
tests total) without importing a Metal device. A real multi-turn model-backed
agent smoke was not launched because the Q1 27B smoke had left only 1.107 GiB
of swap free. The backend command is therefore code-validated, not end-to-end
model-validated, and the limitation is recorded in
`research/stage5/artifacts/backend/protocol_self_test.json`.
