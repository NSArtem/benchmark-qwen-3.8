# Qwen3.8 MLX Investigation — Stage 7 Project Instructions

This repository is an empirical engineering project for making Qwen3.8-27B
a strong practical local model on an 18 GB Apple Silicon Mac.

Stage 7 is no longer a broad quantization search.

Stages 3-6 established a stable practical baseline:

- exact-suffix persistent chat is correct;
- g128 is the best static-weight Pareto point found so far;
- adding more static precision through g64, mixed34, RMS-guided, and KL-guided
  Q1 did not beat g128 overall;
- 16K is a practical long-context mode on g128;
- 20K works as an upper clean-host envelope;
- bounded DWQ produced a real local and end-to-end improvement on one sensitive
  module;
- Qwen3.8 source contains real MTP tensors, but there is no validated Qwen3.8
  MTP decode path in the pinned runtime;
- a serialized single-user agent backend exists and has passed a real multi-turn
  27B smoke.

Stage 7 should therefore spend its time on the two remaining high-upside
directions:

1. bounded, diversity-aware DWQ / teacher-guided refinement WITHOUT increasing
   the static model payload;
2. a correct Qwen3.8 MTP decode path that may improve decode throughput.

A third practical objective is:

3. validate the final runtime in real coding-agent/tool-use workloads rather
   than only synthetic chat prompts.

Do not reopen solved questions without contradictory evidence.

---

## Hardware

Primary machine:

- MacBook Pro
- Apple M3 Pro
- 18 GB unified memory

The project is constrained by unified-memory residency, not disk size alone.

Priority order:

1. correctness;
2. system stability;
3. practical task quality;
4. sustained 8K-16K decode throughput;
5. persistent warm-turn latency;
6. model/runtime memory;
7. cold-prefill performance.

A speed optimization that makes 16K unusable is not a general win.

A quality optimization that only improves calibration loss but not held-out
behavior is not a production win.

---

## Canonical Environment

Use:

    envs/mlx-0.32.1

Canonical command prefix:

    uv run --project envs/mlx-0.32.1 --no-sync ...

Expected:

    Python 3.14.7
    mlx == 0.32.1

Pinned mlx-lm revision:

    254d153fdeb6f150edd4fc5a54f9828638481fa8

Do NOT:

    uv sync
    uv lock
    uv update
    uvx

Do not silently mutate or upgrade the canonical runtime.

Current upstream source may be inspected for algorithms and architecture.

If a newer upstream implementation must be executed experimentally, use an
isolated checkout/environment and keep the canonical baseline unchanged.

Any Stage 7 production recommendation must still be reproducible from a clearly
identified environment/revision.

---

## System Safety

NEVER modify:

    sudo sysctl iogpu.wired_limit_mb

Never call:

    mx.set_wired_limit()

in the canonical runtime.

Do not change equivalent system GPU/wired-memory limits.

Never run two large Qwen/MLX jobs concurrently.

Hard stop conditions:

- Metal OOM;
- GPU timeout;
- WindowServer/UI instability;
- major machine unresponsiveness;
- repeated failed allocation;
- severe unexplained memory growth;
- near-exhausted swap where benchmark results are no longer trustworthy.

Do not deliberately reproduce a known unsafe state repeatedly.

Record a memory baseline before major benchmark phases.

---

## Current Static-Weight Winner

### g128

Path:

    /Users/artem/models/Qwen3.8-27B-3bit-g128-MLX

Quantization:

    affine
    3-bit
    group_size = 128

Payload:

    ~10.180 GiB

Current role:

    DEFAULT

Representative validated runtime:

    ~8K:
        warm TTFT ~1.1 s
        decode ~11.1 tok/s
        peak ~13.98 GiB

    ~12K:
        warm TTFT ~1.1 s
        decode ~10.8 tok/s
        peak ~14.61 GiB

    ~16K:
        warm TTFT ~1.2 s
        decode ~10.4-10.6 tok/s
        peak ~15.24 GiB

    ~20K clean-host probe:
        decode ~10.37 tok/s
        warm follow-up TTFT ~0.72 s in that specific run
        peak ~15.86 GiB
        usable but not the recommended everyday context

Recommended practical context entering Stage 7:

    12K-16K

20K is an upper envelope, not the default daily setting.

---

## Static Quantization Search Is Considered Saturated

Previous serious static weight variants include:

### g64
- ~10.963 GiB
- no convincing practical quality advantage over g128
- worse memory headroom

### mixed34
- ~10.665 GiB
- no convincing held-out quality advantage
- reproducible severe ~16K decode cliff

### Stage 4 RMS-guided mixed quant
- local module distortion ranking did not predict end-to-end model quality well
- rejected

### Stage 5 KL-guided Q1
- ~10.327 GiB
- coherent and operational
- worse corrected quality evidence than g128
- worse teacher-forced NLL/top-1
- slightly heavier and slightly slower
- rejected as default

### Q2/Q3
- larger KL-guided candidates existed
- Stage 6 correctly did not spend heavy evaluation time on them after Q1 failed
  to improve the Pareto frontier

Therefore Stage 7 must NOT create another generic mixed-bit frontier.

New static weight quantization is allowed only if a DWQ/MTP experiment requires
a specific support artifact.

---

## Corrected Quality Evidence

Stage 6 repaired Stage 5 held-out truncation/scoring defects.

The corrected suite increased generation budget for cases where a correct answer
had been cut before an obligatory final sentinel.

Important lesson:

> max-token truncation is not a clean semantic failure.

Always record:

    finish reason
    generated token count

Manually inspect surprising failures.

Entering Stage 7, g128 had the strongest combined practical evidence among the
tested static checkpoints.

Do not rewrite expected answers based on a candidate's output.

---

## KL Sensitivity Result

Stage 5 performed an expensive end-to-end 3-bit -> 4-bit module-promotion
sensitivity sweep over all 498 quantizable modules.

The Stage 6 reproducibility study separated two concepts correctly.

### Same-data numerical reproducibility

On the same calibration set:

    Spearman = 1.0
    Pearson = 1.0
    top-10 overlap = 10/10
    top-25 overlap = 25/25
    sign agreement = 100%
    score deltas = zero in the reported repeat

Conclusion:

    the implemented ranking is deterministic/reproducible on the same data.

### Cross-calibration robustness

On a different calibration subset:

    ranking correlation was much lower
    overlap degraded
    sign agreement degraded

Conclusion:

> calibration composition, not numerical nondeterminism, is now the major
> sensitivity issue.

This is central to Stage 7 DWQ work.

---

## Diversity-Aware Calibration Philosophy

Stage 7 must not train/refine against a narrow set of only arithmetic or only
coding prompts.

Build a compact but diverse CALIBRATION corpus separate from final held-out
evaluation.

Calibration should cover representative daily usage:

- code reasoning/debugging;
- code modification/patch planning;
- structured output;
- factual/context retrieval from supplied text;
- general reasoning;
- instruction following;
- tool/agent transcripts;
- prose/continuity.

Use deterministic local inputs.

Do not use live web text during optimization.

Do not use the final held-out evaluation set as DWQ training/calibration data.

Maintain explicit hashes.

A useful objective is not merely mean calibration loss. Track per-domain
improvement and detect regressions.

---

## DWQ Stage 6 Proof

Stage 6 implemented a bounded real refinement on one highly sensitive module.

Representative result:

    layer 60
    linear_attn.in_proj_a

With a small stable learning rate:

    local MSE:
        ~0.253556 -> ~0.218637
        ~13.8% reduction

    end-to-end teacher KL:
        ~0.091406 -> ~0.090705
        positive improvement

Memory:

    peak MLX ~10.68 GiB

The full teacher model was NOT resident simultaneously.

This proves:

> bounded teacher-guided refinement can work on this 18 GB machine.

It does NOT yet prove:

- whole-model DWQ quality improvement;
- optimal optimizer hyperparameters;
- how many modules should be refined;
- whether local MSE improvement reliably predicts held-out end-to-end gains.

Stage 7 must scale this carefully.

---

## DWQ Objective

The ideal Stage 7 quality result is:

> improve g128 quality while keeping the same static weight payload class.

Do not increase the model from ~10.18 GiB merely to call DWQ successful.

Primary student:

    g128

A DWQ-refined checkpoint may have tiny metadata differences but should remain
essentially in the same payload class.

Primary teacher target:

    whichever reference gives the most meaningful and memory-safe quality
    target, likely g64 or bounded high-precision source behavior

Use evidence, not assumption.

---

## DWQ Optimization Rules

Do not optimize every module immediately.

Use a staged set:

    top-N sensitive modules first

Potential progression:

    4 modules
    8 modules
    16 modules
    32 modules

Exact counts should be driven by results and time.

At each step evaluate:

    calibration teacher loss
    validation teacher loss
    small held-out task subset
    runtime/model integrity

Stop adding modules if:

    held-out metrics stop improving
    calibration improves but validation worsens
    optimizer becomes unstable
    runtime time exceeds useful remaining window

Do not assume monotonic gains.

---

## DWQ Overfitting Protection

Split optimization data:

    DWQ train calibration
    DWQ validation calibration
    final held-out evaluation

Do not use final held-out cases to tune:

    learning rate
    module count
    step count
    ranking selection

For each refinement stage record:

    train loss before/after
    validation loss before/after
    per-domain metrics
    final held-out only when a candidate passes internal gates

If train improves and validation degrades:

    classify overfit.

---

## DWQ Stability

Stage 6 observed that a much larger learning rate could diverge to infinity.

Therefore use conservative optimization.

Always:

- dry-run one module;
- watch finite loss/gradients;
- save intermediate artifacts safely;
- have a rollback path;
- validate output after each refinement group.

Do not overwrite the canonical g128 checkpoint.

Create a separate refined model path.

Do not continue training after NaN/Inf.

---

## MTP Source Status

Stage 6 confirmed real MTP tensors exist in the original Qwen3.8 source.

Observed:

    15 mtp.* tensors
    BF16
    ~849,398,784 bytes
    ~0.791 GiB

They represent substantial model structure, not merely a trivial small output
head.

The current text-only MLX conversion/runtime does not provide a validated
Qwen3.8 MTP decode path.

Stage 7 may invest serious effort in implementing one.

---

## MTP Rules

Do not guess the architecture from tensor names alone.

Before runtime implementation:

1. inspect original Qwen3.8 config;
2. inspect tensor names and shapes;
3. inspect official/reference model code if locally available or current
   upstream source;
4. inspect related Qwen3.x MTP implementations;
5. derive the exact computation graph;
6. derive acceptance/cache semantics.

Document the derivation.

If no trustworthy architecture/reference can be established:

    do not invent one.

---

## MTP Correctness

MTP/speculative decoding is only useful if the base model distribution is
preserved according to the chosen algorithm.

For greedy baseline:

If the implementation is intended to be exact speculative decoding:

    output tokens must match baseline greedy token-for-token.

If a different approximation is implemented:

    label it clearly;
    do not call it exact MTP acceleration.

Measure:

    proposal tokens
    accepted tokens
    acceptance rate
    fallback/rejection behavior

Always test token identity before speed.

---

## MTP Memory

Raw BF16 MTP payload ~0.791 GiB is large on an 18 GB machine.

Therefore investigate quantized MTP weights.

Estimate/test where supported:

    8-bit
    4-bit
    3-bit

The main base model should remain g128 unless DWQ winner replaces it.

Do not automatically keep BF16 MTP resident at 16K.

A good Stage 7 MTP result could be:

    4-bit MTP
    ~0.2 GiB-class payload
    substantial acceptance/speedup
    no 16K memory cliff

The exact numbers must be measured.

---

## MTP Evaluation Contexts

Order:

    tiny correctness prompt
    4K
    8K
    16K

Do not jump to 16K first.

At each context record:

    baseline decode
    MTP decode
    speedup
    acceptance
    active/cache
    peak
    RSS
    correctness

A 4K-only speedup is not enough for general DEFAULT.

Classify:

    PRACTICAL DEFAULT
    OPTIONAL SHORT-CONTEXT MODE
    EXPERIMENTAL
    UNSUPPORTED

---

## Agent Backend Status

A serialized single-user OpenAI-compatible backend exists.

Expected path:

    research/runtime/qwen38_agent_server.py

It has passed:

- protocol/runtime tests;
- a real 27B multi-turn serialized smoke;
- system/user/assistant/tool-role transport;
- persistent exact-suffix reuse;
- reset/session behavior.

Ordinary warm-turn TTFT was around sub-second to ~1 second in that specific
smoke, with longer tool-result suffixes taking somewhat longer.

Do not claim that native autonomous structured tool calling is validated unless
the model actually emitted/consumed the structured format correctly.

Stage 7 should test real task utility.

---

## Real Agent Evaluation

Synthetic "remember this token" chat is no longer sufficient.

Create a local coding-agent benchmark using small deterministic local fixtures.

Useful tasks:

- inspect a small repository;
- identify a bug;
- request a file;
- receive tool output;
- propose a patch;
- receive test failure;
- correct the patch;
- summarize final changes.

The model does NOT need permission to mutate the actual research repository for
the benchmark.

Prefer isolated toy/snapshot repositories.

Measure:

    task success
    tool-turn validity
    number of exchanges
    context growth
    warm TTFT
    decode
    tool payload handling
    recovery after incorrect hypothesis

Compare final g128/DWQ winner where useful.

Do not optimize DWQ directly against the final agent benchmark.

---

## Context Recommendation

Entering Stage 7:

    12K-16K = practical daily range
    20K = validated upper clean-host envelope

Stage 7 does not need another context-length campaign.

Re-test 16K only when a new DWQ or MTP mode changes residency/performance.

20K may be retested only for the final selected mode if it materially differs
from baseline.

Do not test 24K/32K merely for novelty.

---

## Existing Runtime Correctness

Exact-suffix persistent chat is solved infrastructure.

Do not re-open cache correctness without contradictory evidence.

Do not switch to generic server prefix caching or arbitrary cache trimming.

Keep:

    one serialized session
    exact canonical transcript
    exact suffix
    direct generation path
    default wired state

---

## Results and Project Memory

Maintain:

    research/stage7/PLAN.md
    research/stage7/STATUS.md
    research/stage7/RESULTS.md
    research/stage7/results.jsonl
    research/stage7/FINAL_REPORT.md

Before each major phase re-read:

    AGENTS.md
    TASK.md
    research/stage7/STATUS.md

STATUS.md must include:

    current phase
    system memory baseline
    current default checkpoint
    DWQ train/validation corpus hashes
    DWQ refined checkpoint status
    MTP architecture status
    MTP weight precision/status
    current runtime Pareto
    agent benchmark status
    rejected hypotheses
    next action

---

## Experiment Discipline

Each experiment must have:

1. question;
2. baseline;
3. controlled variables;
4. expected decision gate;
5. raw artifacts;
6. measured result;
7. conclusion.

Separate:

    MEASURED FACT
    INFERENCE
    HYPOTHESIS

Do not hide negative results.

Do not silently retune after seeing final held-out results.

---

## Machine-Readable Logging

Append meaningful runs to:

    research/stage7/results.jsonl

Useful fields include:

    timestamp
    phase
    experiment_id
    git_commit

    model_name
    model_path
    model_payload_bytes
    quant_recipe

    dwq_enabled
    dwq_module_paths
    dwq_steps
    dwq_lr
    dwq_train_loss_before
    dwq_train_loss_after
    dwq_validation_loss_before
    dwq_validation_loss_after

    mtp_enabled
    mtp_precision
    mtp_payload_bytes
    mtp_draft_tokens
    mtp_accepted_tokens
    mtp_acceptance_rate
    mtp_exact_match

    suite_name
    suite_hash
    case_id
    category
    objective_pass
    structured_valid
    finish_reason

    nll
    kl
    top1_agreement

    context_tokens
    suffix_tokens
    generated_tokens
    prompt_tps
    ttft_seconds
    decode_tps

    active_memory_bytes
    cache_memory_bytes
    active_plus_cache_bytes
    peak_memory_bytes
    rss_bytes
    swap_used_bytes

    agent_task_id
    agent_turns
    agent_tool_calls
    agent_task_success

    classification
    exception
    safety_stop_reason

Use null when unavailable.

Never invent values.

---

## Git Discipline

Start:

    git status
    git log --oneline -30

Preserve unrelated modifications.

Never:

    git reset --hard
    git clean -fd
    discard unrelated work

Do not commit:

    model weights
    HF cache
    virtualenvs
    giant temporary tensors
    large activation caches

Commit:

    DWQ implementation
    MTP implementation
    calibration manifests/hashes
    small evaluation fixtures
    runtime changes
    tests
    JSON/JSONL summaries
    reports

Use focused logical commits.

---

## Autonomy

Expected Stage 7 window:

    approximately 10-15 hours

Do not ask routine clarification questions.

If a branch is blocked:

    record the reason
    preserve useful evidence
    continue with another independent high-value phase

Do not spend the whole stage forcing MTP if architecture compatibility is not
established.

Do not spend the whole stage training DWQ if validation stops improving.

---

## Explicitly Out of Scope

Do NOT spend Stage 7 on:

- uniform 2-bit;
- mixed_2_6;
- another static mixed-bit frontier;
- Q1/Q2/Q3 resurrection;
- generic uniform 4-bit;
- MXFP4/NVFP4 sweep;
- DSpark;
- 24K/32K context records;
- generic prefix-cache trimming;
- multi-user batching;
- parallel model serving;
- system wired-memory tuning;
- training a new MTP head from scratch.

AWQ/GPTQ are also not priorities.

Only inspect them if both primary Stage 7 branches finish early and evidence
shows a concrete reason.

---

## Final Standard

Stage 7 should answer:

1. Can bounded DWQ improve g128 on validation and final held-out evidence
   without increasing model payload?

2. How many sensitive modules should be refined before returns saturate?

3. Does diversity-aware calibration improve robustness over the narrow Stage 5
   sensitivity setup?

4. Is the DWQ improvement real on coding/agent-style tasks?

5. What exact Qwen3.8 MTP architecture is represented by the 15 source tensors?

6. Can a correct MTP decode path be implemented?

7. Can MTP weights be quantized safely?

8. What is MTP acceptance and speedup at 4K/8K/16K?

9. Does MTP trigger a long-context residency cliff?

10. Does the real serialized coding-agent benchmark pass reliably?

11. What exact model/runtime should be used TODAY?

12. Is any future static-weight quant research still justified?
