# Stage 7 — Bounded DWQ, Qwen3.8 MTP, and Real Agent Validation

## Mission

Run an autonomous Stage 7 for approximately 10-15 hours.

This stage is deliberately narrower than Stages 4-6.

The static quantization frontier is considered largely saturated.

Primary objectives:

A. scale bounded DWQ from a one-module proof into a real diversity-aware
   g128 refinement experiment;

B. reverse-engineer and, if possible, implement a correct Qwen3.8 MTP decode
   path using the MTP tensors already present in the original source model;

C. evaluate the final runtime on realistic serialized coding-agent/tool-use
   workloads.

Critical path:

    recover state
    -> freeze Stage 7 calibration/evaluation protocol
    -> scalable bounded DWQ
    -> DWQ quality decision
    -> Qwen3.8 MTP architecture
    -> MTP correctness prototype
    -> MTP performance / quantized MTP
    -> real coding-agent benchmark
    -> production runtime decision
    -> final report

Use gates.

Do not begin another broad static quant search.

---

# Phase 0 — Recover Stage 6 and Establish Clean Baseline

Target: 20-30 minutes.

Start:

    git status
    git log --oneline -30

Read:

    AGENTS.md
    research/stage6/FINAL_REPORT.md
    research/stage6/STATUS.md
    research/stage6/RESULTS.md
    research/stage6/results.jsonl

Inspect raw artifacts for:

    g128 final quality
    corrected heldout_v2
    KL ranking reproducibility
    bounded one-module DWQ
    20K context
    MTP tensor inventory
    serialized agent smoke

Verify canonical g128 path:

    /Users/artem/models/Qwen3.8-27B-3bit-g128-MLX

Verify high-precision/source path.

Verify current runtime paths:

    qwen38_chat.py
    qwen38_agent_server.py

Record system baseline:

    memory_pressure
    vm_stat
    swap
    free disk
    heavy running processes

Create:

    research/stage7/PLAN.md
    research/stage7/STATUS.md
    research/stage7/RESULTS.md
    research/stage7/results.jsonl

Do not rerun Stage 6 headline benchmarks.

---

# Phase 1 — Build Diversity-Aware DWQ Calibration Protocol

Target: 45-60 minutes.

Stage 6 established that calibration composition materially changes sensitivity
ranking.

Build a fixed Stage 7 optimization protocol.

## Create three disjoint sets

### A. DWQ train calibration

Target:

    ~16-24 compact examples

Balanced across:

    coding/debugging
    patch reasoning
    instruction following
    structured output
    supplied-context retrieval
    general reasoning
    tool/agent transcript
    prose/continuity

### B. DWQ validation calibration

Target:

    ~8-12 examples

Same broad domain distribution, different prompts/content.

Used for:

    early stopping
    module-count selection
    learning-rate/step validation

### C. Final evaluation

Reuse:

    corrected heldout_v2
    Quality Suite V2

and optionally add a very small Stage 7 agent-specific final set.

Do not train/tune against final evaluation.

## Requirements

All examples:

    local
    deterministic
    reproducible
    hashed

Record:

    train hash
    validation hash
    final suite hashes

Do not scrape live web content.

## Teacher target

Run baseline teacher/student metrics on train and validation:

    teacher/reference
    g128

Record:

    KL
    NLL where appropriate
    top-1 agreement

This establishes headroom.

Commit.

Suggested:

    benchmark: define diversity-aware Qwen3.8 DWQ calibration

---

# Phase 2 — Productionize the Bounded DWQ Primitive

Target: 60-90 minutes.

Recover the Stage 6 one-module bounded DWQ implementation.

Do not rewrite it from scratch unless necessary.

Refactor into a reusable harness capable of:

    selected module list
    configurable learning rate
    configurable steps
    train calibration set
    validation set
    checkpoint/resume
    per-module metrics
    bounded source loading
    deterministic seeds/order

## Safety

Before multi-module refinement:

Run the known sensitive single-module case again.

Require:

    finite loss
    safe memory
    reproducible direction of improvement

Do not require exact floating-point identity, but results should be consistent.

Record:

    local loss
    train end-to-end target
    validation end-to-end target
    peak memory
    wall time

## Optimizer exploration

Use a SMALL bounded sweep only if necessary.

Example:

    1e-6
    3e-7
    3e-6

Do not run a large hyperparameter search.

Stage 6 showed a much larger LR could diverge.

Choose one stable default.

Commit reusable implementation.

Suggested:

    feat: productionize bounded Qwen3.8 DWQ harness

---

# Phase 3 — DWQ Scaling Curve

Target: 120-180 minutes.

Primary student:

    canonical g128

Use the Stage 5/6 end-to-end KL sensitivity ranking as the initial module
priority.

Do NOT create a new static quant.

Test refinement scales such as:

    top 4 modules
    top 8 modules
    top 16 modules
    top 32 modules

These are suggested gates, not mandatory exact counts.

The preferred procedure is incremental:

    g128
    -> refine top 4
    -> validate
    -> optionally continue to top 8
    -> validate
    -> optionally continue to top 16
    -> validate
    -> optionally continue to top 32

Do not create full duplicate checkpoint trees if a safe resume/copy-on-write
workflow exists.

Preserve enough checkpoints to compare best states.

## Per-stage metrics

On DWQ train:

    teacher loss before/after

On DWQ validation:

    teacher loss before/after

Also:

    top-1 agreement
    selected generation sanity
    model payload
    memory
    wall time

## Gate

Continue module count only if:

    validation improves or remains plausibly flat
    no practical task regression appears
    training remains stable

Stop if:

    validation worsens twice
    train improves while validation clearly worsens
    generation becomes degraded
    runtime budget becomes excessive

The best validation checkpoint becomes:

    DWQ-CANDIDATE

Do not use final held-out to choose top4/top8/top16/top32.

---

# Phase 4 — Calibration Robustness / Domain Ablation

Target: 60-90 minutes.

Question:

> Is diversity-aware calibration actually more robust than optimizing against a
> narrow domain?

Do not train many full models.

Use the DWQ harness on a small fixed module subset, e.g. top 4-8 modules.

Compare:

    diverse calibration
    coding-heavy calibration
    reasoning-heavy calibration

Evaluate on the DWQ validation set split by domain.

Report:

    in-domain gain
    cross-domain gain
    regressions

Primary decision:

    keep diverse calibration
    OR identify a clearly superior weighting strategy

Do not optimize against final held-out.

If the diverse corpus clearly dominates, stop the ablation early.

Commit.

Suggested:

    experiment: measure Qwen3.8 DWQ calibration robustness

---

# Phase 5 — DWQ Final Quality Evaluation

Target: 60-90 minutes.

Only run on:

    g128 baseline
    best DWQ-CANDIDATE

Optionally include g64 as teacher/reference metrics, not as a production
candidate.

Run:

    corrected heldout_v2
    full Quality Suite V2
    teacher-forced NLL/KL subset

Record:

    objective totals
    category totals
    coding results
    reasoning
    structured output
    context retrieval
    instruction following
    prose/semi-objective
    early EOS
    malformed outputs
    finish reason

Teacher-forced:

    NLL
    KL
    top-1 agreement

Manually inspect every disagreement between g128 and DWQ candidate.

## Decision

Promote DWQ to DEFAULT only if it offers a meaningful quality improvement
without runtime regression.

Possible outcomes:

### DWQ DEFAULT

    measurable held-out improvement
    same model-size class
    runtime essentially unchanged

### DWQ QUALITY MODE

    modest but real quality improvement
    some operational cost

### REJECTED / RESEARCH ONLY

    calibration improves but held-out does not
    or practical regressions appear

Do not force a DWQ win.

Commit.

Suggested:

    benchmark: evaluate bounded-DWQ Qwen3.8 quality

---

# Phase 6 — DWQ Runtime Validation

Target: 45-60 minutes.

Only if a DWQ candidate survives Phase 5.

Compare:

    g128
    DWQ candidate

At:

    ~8K
    ~16K

Use exact-suffix persistent chat.

Record:

    payload
    prompt t/s
    decode
    warm TTFT
    peak
    active+cache
    RSS
    correctness

DWQ should not materially change payload/residency if implemented as intended.

If it unexpectedly changes runtime behavior:

    investigate configuration/serialization differences.

One 20K smoke is allowed only if DWQ becomes DEFAULT and 16K remains healthy.

---

# Phase 7 — Reverse-Engineer Qwen3.8 MTP Architecture

Target: 60-120 minutes.

This is the second P0 research branch.

Stage 6 confirmed:

    15 mtp.* source tensors
    ~0.791 GiB BF16 total

Now determine exactly how they are used.

## Inspect

- source config;
- tensor names;
- tensor shapes;
- original/reference implementation;
- related Qwen3.x MTP model code;
- current mlx-lm code;
- current upstream branches/issues/PRs if useful.

Record exact source/revisions inspected.

Do not rely on memory or previous chat summaries.

## Produce architecture note

Create:

    research/stage7/mtp/ARCHITECTURE.md

It should map:

    each MTP tensor
    -> module
    -> input
    -> output
    -> base-model hidden state dependency
    -> position/token dependency

Derive:

    proposal generation procedure
    number of future tokens predicted
    normalization
    embedding sharing
    logits head behavior
    cache/state requirements

If architecture is ambiguous:

    explicitly document ambiguity.

## Gate

Only implement MTP if the computation graph and acceptance semantics are
sufficiently supported.

If not:

    stop implementation branch
    preserve architecture findings
    spend remaining time on DWQ/agent quality.

Commit architecture research.

---

# Phase 8 — Minimal Exact MTP Prototype

Target: 90-150 minutes.

Only if Phase 7 passes.

Do NOT begin with optimized production code.

Build the smallest correct prototype.

Preferred base:

    g128

MTP tensors may initially be loaded separately.

## Correctness oracle

Use short prompts.

Compare:

    baseline greedy token sequence
    MTP speculative token sequence

If the algorithm is supposed to be exact:

    require token-for-token equality.

Record:

    draft tokens proposed
    accepted
    rejected
    fallback tokens
    acceptance rate

Test multiple short prompts:

    coding
    reasoning
    prose
    structured response

Do not benchmark speed until correctness passes.

## Cache correctness

Qwen3.8 is hybrid recurrent + attention.

MTP rejection/rollback must preserve:

    full-attention KV cache
    recurrent ArraysCache/GatedDelta state

Do not assume ordinary transformer KV rollback is enough.

If rollback semantics are unsafe or unclear:

    use state checkpoint/copy for the prototype
    even if slow

Correctness first.

## Gate

Proceed to optimization only if:

    baseline output equivalence passes
    cache/state survives rejected proposals
    repeated generation remains coherent

Commit.

Suggested:

    experiment: implement exact Qwen3.8 MTP prototype

---

# Phase 9 — Quantized MTP Weights

Target: 45-75 minutes.

Only if exact MTP prototype works.

Raw BF16 MTP payload:

    ~0.791 GiB

This is likely too expensive for 16K general use.

Create at most:

    one 4-bit MTP candidate

Optionally:

    one 3-bit MTP candidate

only if 4-bit is clearly promising.

Do not create a quant zoo.

Use source-backed conversion.

Evaluate MTP-head proposal quality:

    acceptance rate
    exact final output
    added payload
    added residency

The final base-model output must remain correct under exact speculative
acceptance.

Select the smallest MTP precision with acceptable acceptance.

Do not overwrite the base model.

---

# Phase 10 — MTP Performance Study

Target: 60-90 minutes.

Compare:

    baseline
    best MTP mode

At:

    short
    ~4K
    ~8K
    ~16K

Use fresh processes for headline comparisons.

Record:

    decode tok/s
    speedup ratio
    acceptance rate
    proposed tokens/step
    accepted tokens/step
    peak
    active+cache
    RSS
    TTFT
    output identity

If possible separate:

    proposal cost
    verification cost
    rollback/rejection cost

## Classification

### PRACTICAL DEFAULT

    substantial sustained speedup at 8K/16K
    exact correctness
    acceptable memory

### OPTIONAL SHORT-CONTEXT

    good 4K/8K
    but long-context residency cliff

### EXPERIMENTAL

    small gain
    complexity/high rejection
    incomplete integration

### REJECTED

    no useful speedup
    unsafe memory
    correctness problems

Do not promote based on one short benchmark.

---

# Phase 11 — Real Coding-Agent Benchmark

Target: 90-120 minutes.

Use the serialized agent backend.

Final candidate set:

    g128 baseline
    DWQ candidate if it survived
    MTP mode if it survived

Do not benchmark every combination unnecessarily.

## Build deterministic local fixtures

Create several small isolated repositories/tasks.

Suggested tasks:

### Task A — bug diagnosis

Repository contains:

    bug
    tests
    misleading nearby code

Agent must:

    inspect files
    identify root cause
    propose correct fix

### Task B — failed patch recovery

Agent receives:

    initial task
    tool output
    test failure

Must:

    revise hypothesis
    produce corrected patch/plan

### Task C — multi-file change

Requires:

    inspect multiple files
    maintain an invariant
    reference earlier tool output

### Task D — structured/tool interaction

Requires:

    valid tool-role sequence
    parse tool result
    continue without replay

Use deterministic scripts/tools.

Do not let the benchmark alter the actual research repo.

## Metrics

Per task:

    success/fail
    partial success
    tool calls
    invalid tool calls
    turns
    total logical context
    processed suffix tokens
    median warm TTFT
    max warm TTFT
    decode
    final correctness
    recovery from wrong hypothesis

If native model-generated structured tool calls are not reliable:

    distinguish:
        transport/protocol correctness
        model tool-choice quality

Do not conflate them.

## Quality comparison

If DWQ candidate exists:

    compare task outcomes against g128.

Do not tune DWQ after seeing final agent benchmark.

This is final held-out practical evidence.

Commit fixtures and results.

Suggested:

    benchmark: evaluate Qwen3.8 on serialized coding-agent tasks

---

# Phase 12 — Production Integration

Target: 45-60 minutes.

Based on final Stage 7 decisions, clean up:

    research/runtime/qwen38_chat.py
    research/runtime/qwen38_agent_server.py

Integrate only validated modes.

Possible options:

    --model
    --kv-bits 8
    --mtp
    --mtp-model/path
    --greedy
    sampling controls

Do NOT expose an MTP flag as production-ready if it is experimental.

Document:

    DEFAULT model
    DEFAULT context
    16K mode
    20K caveat
    DWQ role
    MTP role
    tool-use limitations

Regression tests must cover any new MTP/DWQ/runtime bug.

Do not add concurrency.

Commit.

Suggested:

    feat: integrate Stage 7 Qwen3.8 runtime modes

---

# Phase 13 — Final Report

Create:

    research/stage7/FINAL_REPORT.md

Update:

    research/stage7/STATUS.md
    research/stage7/RESULTS.md

Before finalizing:

    inspect results.jsonl
    inspect raw DWQ curves
    inspect raw MTP correctness logs
    inspect agent benchmark outputs
    git status
    git log
    final model paths

Distinguish:

    MEASURED FACT
    INFERENCE
    HYPOTHESIS

---

# Time Allocation Guidance

Approximate target:

    Phase 0    0.5h
    Phase 1    0.75h
    Phase 2    1.0h
    Phase 3    2.0-3.0h
    Phase 4    1.0h
    Phase 5-6  1.5h
    Phase 7    1.0-2.0h
    Phase 8    1.5-2.5h
    Phase 9-10 1.0-2.0h
    Phase 11   1.5h
    Phase 12-13 1.0h

This is a guide, not a mandate.

Use gates.

If MTP architecture is blocked, spend that time on deeper DWQ validation and
agent evaluation.

If DWQ clearly fails validation early, stop refining it and prioritize MTP.

---

# Explicitly Out of Scope

Do NOT spend Stage 7 on:

    new generic static quant frontier
    g64/g128 group-size sweep
    mixed_2_6
    Q1/Q2/Q3 resurrection
    uniform 2-bit
    generic 4-bit
    MXFP4/NVFP4 sweep
    AWQ/GPTQ comparison campaign
    DSpark
    24K/32K context chasing
    multi-user server
    batching/concurrency
    generic prompt-prefix cache trimming
    wired-memory system tuning
    training a new MTP head

---

# Machine-Readable Results

Append to:

    research/stage7/results.jsonl

Recommended fields:

    timestamp
    experiment_id
    phase
    git_commit

    model_name
    model_path
    model_payload_bytes

    calibration_train_hash
    calibration_validation_hash
    final_eval_hash

    dwq_enabled
    dwq_checkpoint
    dwq_module_count
    dwq_modules
    dwq_lr
    dwq_steps
    dwq_train_loss_before
    dwq_train_loss_after
    dwq_validation_loss_before
    dwq_validation_loss_after

    mtp_enabled
    mtp_precision
    mtp_payload_bytes
    mtp_draft_width
    mtp_proposed_tokens
    mtp_accepted_tokens
    mtp_acceptance_rate
    mtp_exact_match

    suite_name
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
    peak_memory_bytes
    rss_bytes
    swap_used_bytes

    agent_task_id
    agent_task_success
    agent_partial_score
    agent_turns
    agent_tool_calls
    agent_invalid_tool_calls

    classification
    exception
    safety_stop_reason

Use null for unavailable fields.

Never invent data.

---

# Suggested Commits

Prefer logical commits:

    benchmark: define diversity-aware Qwen3.8 DWQ calibration
    feat: productionize bounded Qwen3.8 DWQ harness
    experiment: scale bounded DWQ across sensitive modules
    experiment: measure DWQ calibration robustness
    benchmark: evaluate bounded-DWQ Qwen3.8 quality
    benchmark: validate DWQ Qwen3.8 runtime
    research: map Qwen3.8 MTP architecture
    experiment: implement exact Qwen3.8 MTP prototype
    experiment: quantize Qwen3.8 MTP weights
    benchmark: measure Qwen3.8 MTP decode
    benchmark: evaluate Qwen3.8 coding-agent tasks
    feat: integrate Stage 7 Qwen3.8 runtime modes
    docs: summarize Qwen3.8 Stage 7

Do not commit model weights.

---

# FINAL_REPORT.md Must Answer

1. What Stage 7 calibration train/validation sets were created?

2. Are they materially more diverse than Stage 5 calibration?

3. What teacher/student baseline gap exists on train vs validation?

4. Was the Stage 6 bounded DWQ result reproduced?

5. What learning rate/steps were stable?

6. How many modules were successfully refined?

7. How did train loss change?

8. How did validation loss change?

9. Where did DWQ returns saturate?

10. Did diversity-aware calibration outperform narrow-domain calibration?

11. What is the best DWQ checkpoint?

12. Did DWQ improve corrected heldout_v2?

13. Did DWQ improve Quality Suite V2?

14. Did DWQ improve NLL/KL/top-1?

15. Did DWQ improve real coding-agent task success?

16. Did DWQ change model payload or runtime materially?

17. What exact architecture do the 15 mtp.* tensors implement?

18. Was a correct Qwen3.8 MTP decode prototype implemented?

19. Was baseline greedy output preserved exactly?

20. How was hybrid recurrent state handled during rejection/rollback?

21. What MTP precision was tested?

22. What was MTP payload overhead?

23. What was MTP acceptance at short/4K/8K/16K?

24. What was MTP decode speedup at 4K/8K/16K?

25. Did MTP cause a long-context memory cliff?

26. Is MTP DEFAULT, optional, experimental, or rejected?

27. How many real coding-agent tasks were tested?

28. What was task success by model/mode?

29. Were tool-role sequences valid?

30. Are native structured tool calls reliable, or only serialized transport?

31. What is the final DEFAULT checkpoint?

32. Is there a separate QUALITY MODE?

33. Is there a separate ACCELERATED MODE?

34. What context is recommended daily?

35. What exact CLI command should be used TODAY?

36. What exact server command should be used TODAY?

37. What are the highest-value remaining Stage 8 ideas?

Include:

### DWQ Table

    checkpoint
    modules refined
    train loss
    validation loss
    held-out quality
    NLL/KL
    payload
    verdict

### MTP Table

    mode
    MTP precision
    extra GiB
    acceptance
    4K decode
    8K decode
    16K decode
    exactness
    verdict

### Agent Table

    model/mode
    tasks passed
    tool errors
    median warm TTFT
    decode
    verdict

### Final Runtime Table

    mode
    model
    context
    decode
    warm TTFT
    peak
    role

---

# Completion Condition

Stage 7 is complete when:

- a diverse DWQ train/validation protocol exists;
- bounded DWQ has been scaled beyond the Stage 6 one-module proof or has been
  safely rejected;
- final DWQ held-out quality is known;
- Qwen3.8 MTP architecture has been documented;
- an MTP correctness prototype has been attempted if architecture support is
  sufficient;
- MTP performance has been measured if correctness passes;
- real serialized coding-agent tasks have been run;
- final runtime/model roles are selected;
- FINAL_REPORT.md exists;
- STATUS.md is current;
- useful work is committed.

Do not start Stage 8 automatically.

At completion return a concise summary with:

    default model
    DWQ verdict
    DWQ quality gain/loss
    MTP verdict
    MTP 4K/8K/16K speedup
    MTP memory overhead
    coding-agent benchmark verdict
    recommended context
    recommended CLI command
    recommended server command
    FINAL_REPORT path
    commit SHAs

Then STOP.
