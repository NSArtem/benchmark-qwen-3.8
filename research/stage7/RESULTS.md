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
