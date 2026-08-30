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

## Phase 5 — final DWQ selection gate

### MEASURED FACT

- Fresh final heldout_v2 evaluation used separate processes and no tuning after
  inspecting outputs. g128 scored 20/21 objective and 3/3 semi-objective;
  diverse top-4 scored 19/21 and 3/3. Both had zero harness exceptions.
- The top-4 loss was `h5-coding-06`. Manual inspection found a materially
  different, overlong reconciliation discussion that did not state the exact
  required client-side discard rule. The shared `h5-reasoning-04` repetition
  flag is a mechanical false positive: both outputs end with the correct
  answer `4`.
- Fresh Quality Suite V2 evaluation scored g128 at 23/24 objective and 4/4
  semi-objective, versus top-4 at 22/24 and 4/4. The additional top-4 loss was
  `coding-07`, whose output omitted the required idempotent retry condition.
- Teacher-forced evaluation on 167 reference tokens gave g128 weighted NLL
  `1.3690119760`, top-1 `0.6946107784`; top-4 gave NLL `1.3720059880`, top-1
  `0.6946107784`. Both processes completed without exceptions.
- Model-load active memory was 10.1806 GiB for both. The four-module patch
  changed live patch metadata/arrays by only about 3.5 KiB and did not change
  the static model directory or payload class.
- Raw evidence is under `research/stage7/artifacts/final/` and includes both
  token IDs and decoded outputs. These final suites were not used to retune
  learning rate, module count, or ranking.

### INFERENCE

Top-4's calibration KL/NLL improvement does not survive the final practical
quality gates. The correct production decision is to retain g128 as DEFAULT
and keep top-4 only as a research artifact. The same-payload constraint was
met, but no quality gain was measured.

### HYPOTHESIS

The top-4 local objective improves teacher distributional match on the small
calibration split while perturbing generation-critical recurrent dynamics in
specific coding contexts. This is consistent with the heldout coding loss but
does not establish a causal mechanism.

### Gate decision

DWQ scaling and promotion are stopped at four modules. No further DWQ tuning
or final-suite retuning is allowed in Stage 7.

## Phase 7 — MTP architecture gate

### MEASURED FACT

- The original source config has `mtp_num_hidden_layers=1` and
  `mtp_use_dedicated_embeddings=false`; all 15 `mtp.*` tensors are in source
  shard 18 and total 849,398,784 BF16 bytes.
- The tensors map exactly to `fc`, two pre-FC RMSNorms, one full-attention
  decoder layer, its q/k norms and projections, MLP, and final norm. No
  dedicated MTP embedding or LM head is present.
- A local `mlx_vlm==0.6.13` Qwen3.5 MTP reference was instantiated and loaded
  with all 15 source tensors after source-to-MLX norm conversion. The loaded
  MTP payload was exactly 849,398,784 bytes.
- The architecture derivation, position/bonus-token semantics, two-token
  proposal width, shared embedding/LM-head behavior, and hybrid KV plus
  recurrent rollback requirement are recorded in
  `research/stage7/mtp/ARCHITECTURE.md`.

### INFERENCE

The local reference plus source/config agreement is sufficient support for a
guarded canonical-runtime correctness prototype. A prototype may proceed,
but MTP remains experimental until exact greedy output identity and explicit
rejection rollback are demonstrated.

### HYPOTHESIS

The source-trained MTP layer may propose enough matching tokens to amortize
its extra full-attention forward on this g128 base. Acceptance and speed are
unknown; no performance claim is made here.

## Phase 8 — exact greedy MTP prototype

### MEASURED FACT

- The prototype is `research/stage7/harness/run_mtp_correctness.py`. It uses
  g128 as the target, the separately loaded 15-tensor BF16 source MTP block,
  and the pinned generator's split prefill schedule. It does not modify the
  base model directory.
- A first attempt that verified a hybrid recurrent batch in one call failed
  identity and exposed a recurrent transition mismatch. The corrected
  sequential target verification/replay path is retained in the source and
  makes the baseline one-token transition explicit.
- On coding, reasoning, prose, and structured short prompts, baseline and MTP
  greedy token sequences matched exactly. All four cases passed every observed
  explicit rejection rollback check, covering both KV and ArraysCache-backed
  target state. There were zero harness exceptions.
- The BF16 live payload is 849,398,784 bytes / 0.791 GiB. Active MLX memory was
  10.1806 GiB after base load and 10.9717 GiB after MTP load, before decode
  cache growth.
- Per-case results were: coding 30 proposed / 16 accepted (53.33%), reasoning
  24 / 19 (79.17%), prose 26 / 12 (46.15%), structured 22 / 19 (86.36%). The
  raw correctness artifact is
  `research/stage7/artifacts/mtp/correctness-bf16-v3/correctness.json`.

### INFERENCE

The corrected prototype establishes that the derived graph and source norm
conversion can preserve exact greedy output on short prompts, including
rejected speculative suffixes. This passes the Phase 8 correctness gate and
justifies the one-candidate 4-bit precision experiment.

### HYPOTHESIS

The high variation in acceptance by domain may make the extra BF16 MTP
residency and sequential prototype cost unattractive at long context. A
4-bit MTP block may retain enough proposal quality to be useful, but this must
be measured and exactness rechecked.

## Phase 9 — source-backed 4-bit MTP precision gate

### MEASURED FACT

- One and only one quantized MTP candidate was created from the source-backed
  architecture: affine 4-bit linear layers with group size 128. The candidate
  has 225,659,904 quantized parameter bytes (225,663,093 bytes on disk), versus
  849,398,784 BF16 source parameter bytes.
- The candidate was loaded into the same guarded prototype and passed exact
  greedy token identity on all four short prompts. All observed explicit
  rejection rollback checks also passed, with zero harness exceptions.
- Per-case proposal/acceptance was coding 26/18 (69.23%), reasoning 26/20
  (76.92%), prose 28/11 (39.29%), and structured 20/19 (95.00%). The total was
  68/100 accepted (68.00%).
- Active MLX memory was 10.1806 GiB after g128 load and 10.3908 GiB after the
  4-bit MTP load, an active overhead of 225,661,184 bytes / approximately
  0.210 GiB before decode-cache growth. The raw candidate, manifest, and
  correctness log are under `research/stage7/artifacts/mtp/`.

### INFERENCE

The source-backed 4-bit MTP weights preserve the prototype's exact greedy and
rollback behavior on the short gate while reducing live MTP overhead by about
0.58 GiB relative to BF16. This is sufficient to proceed to the required
performance/residency gate, not sufficient to promote MTP.

### HYPOTHESIS

Quantization may reduce proposal quality on some domains, but the short-gate
acceptance increase relative to the BF16 sample is not evidence of a general
speed advantage. Long-context verification and rollback cost remain unknown.

## Phase 10 — MTP performance and residency gate

### MEASURED FACT

- Headline comparisons used separate fresh canonical-runtime processes, the
  same g128 target, the same source-backed affine 4-bit/group-128 MTP model,
  64 generated-token budgets, and baseline token files for exactness checks.
  The MTP implementation was the sequential target-verification plus explicit
  snapshot/replay path; the batched target shortcut was excluded after its
  separate exactness probe failed.
- The short prompt was 1,023 tokens, the 4K prompt 4,096, the 8K prompt 8,191,
  and the 16K coding prompt 16,727 tokens. Results:

| context | baseline TTFT | baseline decode | MTP TTFT | MTP decode | MTP/base ratio | accepted/proposed | exact | MTP peak |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| short | 13.463 s | 11.898 tok/s | 13.580 s | 6.289 tok/s | 0.529x | 39/52 (75.00%) | yes | 12.287 GiB |
| 4K | 53.098 s | 11.647 tok/s | 54.638 s | 4.070 tok/s | 0.349x | 29/68 (42.65%) | yes | 13.621 GiB |
| 8K | 106.831 s | 11.147 tok/s | 112.819 s | 2.651 tok/s | 0.238x | 34/60 (56.67%) | yes | 15.456 GiB |
| 16K | 241.185 s | 10.628 tok/s | safety stop | — | — | — | not reached | — |

- Baseline peaks were 12.077 GiB, 13.360 GiB, 13.983 GiB, and 15.236 GiB at
  short/4K/8K/16K. MTP peaks were 12.287 GiB, 13.621 GiB, and 15.456 GiB at
  short/4K/8K; the 16K attempt did not complete. MTP active-plus-cache after
  generation was 11.398 GiB, 11.394 GiB, and 12.481 GiB at the first three
  contexts, versus 10.517 GiB, 10.645 GiB, and 11.390 GiB for baseline.
- The 16K MTP process was stopped after a host snapshot showed 28% free
  memory, 909 free pages, 544,794 pages occupied by the compressor, and
  swapouts rising from 28,056,329 to 29,694,221 between checks. The host
  recovered to 89% free after termination. No 16K output, acceptance, or
  exactness claim is made. The raw safety record is
  `research/stage7/artifacts/performance/mtp-4bit-g128-v1/16k-mtp-safety-stop.json`.
- The raw per-process logs, including stage-separated proposal, verification,
  replay, and target-prefill timings, are under
  `research/stage7/artifacts/performance/mtp-4bit-g128-v1/`. The separate
  batched-verification failure is under
  `research/stage7/artifacts/mtp/correctness-4bit-g128-batch/`.

### INFERENCE

The exact sequential MTP prototype is slower than direct g128 decoding at
short, 4K, and 8K contexts, with the deficit widening as context grows. Its
snapshot/replay implementation also adds substantial cache pressure, and the
16K attempt crossed the project safety gate. MTP is therefore rejected as a
production DEFAULT, OPTIONAL SHORT-CONTEXT mode, or ACCELERATED MODE in this
pinned runtime.

### HYPOTHESIS

The dominant loss is implementation cost rather than only proposal quality:
the sequential verifier performs multiple one-token target transitions per
round, and rejected rounds require recurrent plus attention state replay. A
future fused verifier/rollback implementation could change this conclusion,
but the failed batched exactness probe means that it cannot be assumed for
Qwen3.8 today.

### Gate decision

MTP is retained as an exact research prototype and rejected for production.
No MTP flag or MTP weights are integrated into the default chat or agent
runtime.
