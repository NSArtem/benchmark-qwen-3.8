# Stage 7 Results

This ledger is a human-readable companion to `results.jsonl`. It separates
measured facts, inferences, and hypotheses and is updated after each gated
phase.

## Phase 0 — recovery and entry baseline

### MEASURED FACT

- Entry commit is `36fa569` (`stage 7`); pre-existing worktree changes are the
  untracked `archive_3.8_stage5.zip` and `archive_3.8_stage6.zip`, preserved.
- Stage 6 reports, raw JSONL, corrected heldout suite, bounded-DWQ artifacts,
  MTP tensor inventory, context evidence, and serialized-agent smoke were
  recovered.
- The canonical g128 path exists with 19 safetensor shards. A filesystem sum
  over those shards is 10,930,954,231 bytes; Stage 6's MLX materialized model
  payload is 10,930,715,648 bytes (10.180022 GiB).
- The source snapshot and `research/runtime/{qwen38_chat.py,qwen38_agent_server.py}`
  exist. A default-sandbox import reached MLX but could not access Metal; no
  model was loaded and this is not a model failure.
- Stage 6 established g128 as the static-weight winner, one-module bounded DWQ
  feasibility, unsupported MTP in the pinned text path, and valid serialized
  transport/cache behavior.

### INFERENCE

The repository and stable model/source paths are usable for Stage 7. Heavy
canonical runs require host-level Metal access, and will be serialized one at a
time with memory snapshots.

### HYPOTHESIS

A diverse, disjoint calibration set may make module selection/refinement more
robust across usage domains than the earlier narrow subset, but it may not
translate local loss gains into heldout generation wins.

## Phase 1 — diversity-aware calibration freeze

### MEASURED FACT

- Frozen train split: 16 deterministic local prompts, 2 each in coding,
  patch reasoning, structured output, supplied-context retrieval, general
  reasoning, instruction following, tool/agent transcript, and prose
  continuity. SHA-256:
  `3fedf6d50c6f202f189f0d585fae418e3475eb4458e5f352af509e4f9a58db98`.
- Frozen validation split: 8 disjoint prompts, one per same domain. SHA-256:
  `1193e48e77bee8ada73d27ab5a025eb7100c8072c03051bec8fe0fb6feaa91dd`.
- Final corrected heldout_v2 and Quality Suite V2 are explicitly excluded from
  optimization and have hashes recorded in `calibration/manifest.json`.

### INFERENCE

This is materially more diverse than the Stage 5 six-category calibration
because patch-reasoning and tool/agent transcripts are represented separately,
and the validation content is disjoint rather than a same-data repeat.

### HYPOTHESIS

The eight-domain split should reduce calibration-composition sensitivity, but
one validation example per domain is a small internal gate and cannot alone
establish heldout quality improvement.

### Next gate

Capture a separate g64 teacher-logit manifest for each split and compute
teacher/student baseline metrics before any DWQ optimization.

### Baseline measurement

#### MEASURED FACT

- Separate g64 teacher captures completed for all 16 train and 8 validation
  cases in 26.7 s and 12.2 s respectively, with no exceptions.
- g128-vs-g64 metrics over 776 train target positions were KL 0.156752,
  teacher-argmax NLL 0.285187, actual-token NLL 7.843152, top-1 agreement
  0.917526, and logit MSE 0.764543.
- Over 404 validation target positions they were KL 0.142997,
  teacher-argmax NLL 0.277587, actual-token NLL 7.081791, top-1 agreement
  0.918317, and logit MSE 0.757290.
- The g128 metrics process peaked at 10.922 GiB MLX memory. Teacher weights
  were not resident during the g128 metric pass.

#### INFERENCE

The frozen protocol has enough teacher headroom to test bounded refinement and
the added domains are present in both optimization and validation. The lower
validation KL than train is an observation about this small corpus, not a
generalization claim.

#### Next gate

Productionize the Stage 6 streamed module optimizer and reproduce layer 60 on
the new split before considering top-4 or broader refinement.

## Phase 2 — reusable bounded DWQ and reproduction

### MEASURED FACT

- The reusable harness is `research/stage7/harness/run_bounded_dwq.py`. It
  supports explicit module lists/ranking top-N, cumulative safetensor patch
  checkpoints, resume from an earlier patch, deterministic case order,
  bounded source loading, finite-gradient checks, conservative parameter
  clipping, and train/validation metrics split by domain.
- The known Stage 6 sensitive module was rerun on all 16 frozen Stage 7 train
  cases (792 captured activation rows) with LR `1e-6`, 40 steps, and clip
  fraction `0.25`. Local MSE reduced from 0.2657081783 to 0.2311711609
  (12.998%). The run was finite and peaked at 10.886557 GiB MLX memory in
  59.22 seconds.
- Stage 7 train end-to-end KL changed from 0.1567515774 to 0.1556147865;
  teacher-argmax NLL from 0.2851869464 to 0.2841425212; top-1 from
  0.9175257732 to 0.9188144329.
- Stage 7 validation KL changed from 0.1429969692 to 0.1422168810;
  teacher-argmax NLL from 0.2775873937 to 0.2776506384; top-1 from
  0.9183168317 to 0.9158415842. The validation KL improvement and top-1
  decrease are both retained; neither is hidden by a mean-only score.
- The cumulative one-module patch and raw result are under
  `research/stage7/artifacts/dwq/reproduce-layer60.*`.

### INFERENCE

The Stage 6 local-refinement direction reproduced under a broader,
eight-domain corpus and improved validation KL, supporting a cautious top-4
continuation. The validation top-1 tradeoff means the module-count gate must
remain multi-metric and can reject further scaling.

### HYPOTHESIS

The Stage 5 ranking may still identify useful modules, but independent
module-local improvements may interfere when composed. A cumulative resume
checkpoint is therefore safer than assuming additive gains.

### Next gate

Refine only ranks 2–4 on top of the saved layer-60 patch. Measure the complete
top-4 state on train/validation and inspect domain deltas before attempting
top-8.

## Phase 3 — top-4 incremental DWQ gate

### MEASURED FACT

- Top-4 resumed from the layer-60 patch and refined ranks 2–4:
  `layers.49.linear_attn.in_proj_b`, `layers.18.linear_attn.in_proj_b`, and
  `layers.21.linear_attn.in_proj_b`. Three local objectives reduced by
  16.20%, 14.48%, and 12.62% respectively; all gradients remained finite.
- Relative to the original g128 baseline, top-4 train KL changed
  0.1567515774 -> 0.1509779404 and teacher-argmax NLL
  0.2851869464 -> 0.2839706270. Validation KL changed
  0.1429969692 -> 0.1368660092 and NLL 0.2775873937 -> 0.2758729293.
- Top-4 train top-1 changed 0.9175257732 -> 0.9162371134 and validation
  top-1 0.9183168317 -> 0.9133663366. The KL/NLL gains and top-1 regression
  are both measured; no aggregate metric was substituted for the other.
- Peak MLX memory was 10.886573 GiB and wall time 85.47 s. The cumulative
  patch is `research/stage7/artifacts/dwq/top4.safetensors`; the base model
  directory and payload are unchanged.
- Eight disjoint validation-prompt greedy sanity cases ran for baseline and
  top-4. Both had no exceptions, identical finish categories per case, and
  coherent semantic content. The coding case was more concise in top-4; no
  structured, retrieval, reasoning, tool, or prose case lost its requested
  facts.

### INFERENCE

Top-4 passes the internal continuation gate on validation KL/NLL and practical
sanity, although its token-level top-1 decline requires caution. The result is
strong enough to test whether the next three modules continue the calibration
gain, not strong enough for heldout evaluation or production promotion.

### Next gate

Resume from top-4 and refine ranks 5–8. Compare against the original baseline
and top-4 state; do not inspect final heldout outputs yet.

## Phase 3 — top-8 gate and DWQ saturation

### MEASURED FACT

- Top-8 resumed from top-4 and refined ranks 5–8:
  `layers.17.linear_attn.in_proj_b`, `layers.2.linear_attn.in_proj_b`,
  `layers.42.linear_attn.in_proj_a`, and `layers.18.linear_attn.in_proj_a`.
  Local MSE reductions were 10.32%, 3.50%, 13.77%, and 7.86%; all optimizer
  traces were finite.
- Relative to top-4, train KL regressed 0.1509779404 -> 0.1560821364 and
  validation KL regressed 0.1368660092 -> 0.1422903157. Train
  teacher-argmax NLL regressed 0.2839706270 -> 0.2847247253 and validation
  NLL 0.2758729293 -> 0.2789507018. Validation top-1 recovered from
  0.9133663366 to 0.9183168317, but this did not prevent the KL/NLL
  regression.
- The top-8 process stayed finite, peaked at 10.886598 GiB, and took 101.08
  seconds. Its cumulative patch is retained as a rejected control; g128 and
  top-4 patches are unchanged.
- Because the continuation gate requires validation improvement or a
  plausibly flat result, no top-16 or top-32 refinement was launched.

### INFERENCE

The top-4 state is the current DWQ candidate and top-8 demonstrates non-
monotonic module interference under a sequential local objective. The
calibration gain cannot be treated as additive by rank, and top-1 alone is not
an adequate selection metric.

### HYPOTHESIS

The early ranked linear-attention projections may share recurrent-state
interactions that make independently optimized updates conflict when composed.
The rejected top-8 result is evidence for this hypothesis, not proof of the
mechanism.

### Next gate

Run the small non-heldout top-8 generation sanity check, then perform Phase 4
calibration robustness ablation only on the fixed top-4-sized module subset.

## Phase 3 — top-8 generation sanity

### MEASURED FACT

- The rejected top-8 patch was exercised on all eight frozen validation
  prompts. There were zero exceptions, and each case recorded a finish reason
  and generated-token count. Structured, retrieval, instruction, and prose
  cases terminated with EOS; the bounded coding, patch, reasoning, and tool
  cases reached the same 96-token generation cap used by the sanity harness.

### INFERENCE

The top-8 regression is not a runtime-integrity failure. Its rejection is due
to the measured validation KL/NLL regression under composition, not a crash or
obvious generation corruption.

## Phase 4 — diversity-aware calibration robustness

### MEASURED FACT

- The fixed top-4 module set was refined three ways, always starting from the
  unpatched g128 model and validating on all eight disjoint validation cases:
  the diverse 16-case corpus, a coding/tool-heavy eight-case subset, and a
  reasoning/retrieval/prose/instruction-heavy eight-case subset.
- Diverse top-4 validation metrics were KL `0.1368660092`,
  teacher-argmax NLL `0.2758729293`, actual-token NLL `7.0465717882`, and
  top-1 `0.9133663366`.
- Coding-heavy validation metrics were KL `0.1409067689`,
  teacher-argmax NLL `0.2775192461`, actual-token NLL `7.0662236733`, and
  top-1 `0.9183168317`.
- Reasoning-heavy validation metrics were KL `0.1420198095`,
  teacher-argmax NLL `0.2791354904`, actual-token NLL `7.0699190952`, and
  top-1 `0.9183168317`.
- The diverse, coding-heavy, and reasoning-heavy train selections had 776,
  417, and 359 target positions respectively. Each narrow run improved its
  own selected training loss, but neither matched the diverse run on
  validation teacher loss. All optimizers remained finite and the peak MLX
  memory stayed at about 10.887 GiB.
- The raw ablation records are
  `artifacts/dwq/ablation-coding-heavy.json` and
  `artifacts/dwq/ablation-reasoning-heavy.json`; the cumulative patches are
  ignored research artifacts and do not modify the canonical g128 directory.

### INFERENCE

On this fixed module subset, diversity-aware calibration is more robust than
either narrow domain mixture for the validation KL/NLL objective. The top-1
metric favors the narrow controls because it recovered the baseline value,
while the diverse candidate gives the stronger distributional match. This is
why final selection must retain both metrics and cannot be decided by top-1
alone.

### HYPOTHESIS

Including tool/agent, structured, coding, reasoning, retrieval, instruction,
and prose examples together may reduce calibration-composition bias in the
module updates. The evidence is limited to eight validation examples and does
not yet establish a final heldout or task-level gain.

### Gate decision

Phase 4 is complete. Keep the diverse top-4 patch as the sole DWQ candidate
for final heldout evaluation; do not promote either narrow ablation and do not
launch top-16/top-32.
