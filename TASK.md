# Stage 6 — Exploit the Stage 5 Frontier and Finish the Practical Runtime

## Mission

Run an approximately 15-hour autonomous Stage 6.

Stage 6 is NOT a new broad exploration stage.

Stage 5 already paid the expensive cost of creating a KL-guided frontier but
stopped before proper evaluation because the machine entered an untrustworthy
high-swap state.

Stage 6 must extract maximum value from those existing artifacts.

Critical path:

    clean baseline
    -> preserve/verify Q1/Q2/Q3
    -> repair quality scoring truncation
    -> evaluate Q1 deeply
    -> gate Q2/Q3
    -> validate KL-ranking reproducibility
    -> bounded DWQ on winner
    -> final Pareto decision
    -> 20K envelope
    -> gated Qwen3.8 MTP
    -> real agent-server integration
    -> final report

Do not create new unrelated quants.

---

# Phase 0 — Clean-Boot and Artifact Recovery

Target: 30 minutes.

Read:

    AGENTS.md
    research/stage5/FINAL_REPORT.md
    research/stage5/STATUS.md
    research/stage5/RESULTS.md
    research/stage5/results.jsonl

Start:

    git status
    git log --oneline -30

Inspect running processes.

Record:

    memory_pressure
    vm_stat
    swap usage
    free disk

If the machine is still in severe post-Stage-5 swap pressure, do not run
headline benchmarks.

Create:

    research/stage6/PLAN.md
    research/stage6/STATUS.md
    research/stage6/RESULTS.md
    research/stage6/results.jsonl

## Recover Q1/Q2/Q3

Search expected locations:

    /private/tmp/qwen38-stage5-Q1-kl-g128
    /private/tmp/qwen38-stage5-Q2-kl-g128
    /private/tmp/qwen38-stage5-Q3-kl-g128

and persistent model directories.

Recover exact model paths from Stage 5 artifacts.

For each candidate verify:

    config.json
    tokenizer files
    weight shards
    payload size
    readable/loadable metadata

If candidates still exist only under /private/tmp and disk permits, move them
to persistent storage before benchmarking.

Preferred:

    /Users/artem/models/stage5-frontier/

Record final paths.

If Q2/Q3 are missing but Q1 exists:

    do NOT immediately rebuild them.

Evaluate Q1 first.

If all are missing:

    rebuild only Q1 from saved Stage 5 manifest/ranking as the first recovery.

Do not repeat the full sensitivity sweep just to rebuild candidates.

Commit scaffolding/status if useful.

---

# Phase 1 — Repair Held-Out Scoring and Generation Budgets

Target: 45-60 minutes.

Before comparing Q1/Q2/Q3, inspect Stage 5 held-out failures.

Identify cases where:

    answer is semantically correct
    but generation stopped at max_tokens
    before a required sentinel/final line

Examples may include arithmetic/version tasks.

Do not change semantic expected answers unless objectively wrong.

Preferred fix:

    increase generation budget
    OR
    make scorer accept the correct semantic answer without requiring an
    unnecessary final sentinel

Apply the same correction to ALL models.

Keep a machine-readable record:

    old case
    old rule
    reason invalid
    new rule/budget
    new hash

Freeze:

    heldout_v2

or equivalent.

Do not use Q1 output to tailor the answer key.

## Regression rerun

Re-run only the affected held-out cases first on:

    g64
    g128
    mixed34

Verify that truncation artifacts are actually removed.

Then freeze the corrected suite.

Commit.

Suggested:

    benchmark: fix Stage 5 held-out truncation artifacts

---

# Phase 2 — Q1 Full Validation

Target: 90-120 minutes.

This is the highest-priority Stage 6 experiment.

Model:

    Q1 (~10.327 GiB)

Compare against:

    g128

Use fresh processes for cold/performance comparisons.

## Quality

Run:

    full Quality Suite V2
    corrected heldout_v2
    teacher-forced NLL/KL subset

Record:

    objective totals
    per-category totals
    early EOS
    repetition
    structured validity
    NLL
    KL
    top-1 agreement
    finish reasons

Manually inspect every Q1-vs-g128 disagreement.

Do not use subjective impressions to override objective correct answers.

## Runtime

Run Q1 at:

    short smoke
    ~8K
    ~12K
    ~16K

At 16K do at least two fresh repeats if results are surprising.

Record:

    prompt t/s
    cold TTFT
    warm exact-suffix TTFT
    decode t/s
    active
    cache
    active+cache
    peak
    RSS
    memory pressure

Use a 5-turn persistent validation at:

    8K
    16K

if cold performance is healthy.

## Q1 decision gate

### Q1 wins

Advance Q1 as the primary Stage 6 candidate if it offers:

    measurable quality recovery
    with no meaningful 16K runtime cliff

or another real Pareto improvement.

### Q1 ties g128

If quality/runtime are effectively equal:

    g128 remains default because it is smaller/simpler.

Do not promote Q1 merely because it exists.

### Q1 loses

If Q1 is lower quality or significantly slower:

    reject Q1.

Then decide whether Q2/Q3 plausibly address the specific failure.

Commit.

Suggested:

    benchmark: fully evaluate Stage 5 Q1 quant

---

# Phase 3 — Gate Q2 and Q3

Target: 60-120 minutes depending on Q1 result.

Do NOT automatically run full suites on both.

## If Q1 improves quality but not enough

Test Q2 next.

Mini gate:

    selected quality disagreements
    NLL/KL subset
    8K decode
    16K decode

If Q2 improves quality further without a cliff:

    run full corrected heldout + V2.

Only then consider Q3.

## If Q1 quality equals g128 but Q1 is slower

Q2/Q3 are unlikely to be good defaults because they are larger.

Only test Q2 if:

    Q1 shows a clear monotonic quality signal that may justify a separate
    QUALITY MODE.

Otherwise skip Q2/Q3.

## If Q1 already hits a 16K residency cliff

Use Phase 1/Stage 5 cliff evidence.

Q2/Q3 are larger and likely worse for long-context decode.

Do not run expensive 16K repetitions unless testing a clear hypothesis.

They may still be tested briefly as short-context quality modes.

## If Q1 strongly wins

Q2 may be tested as a quality-extension candidate.

Q3 only if Q2 demonstrates additional useful recovery.

At the end choose:

    g128
    Q1
    Q2
    Q3

roles based on actual Pareto value.

Commit meaningful evaluation.

---

# Phase 4 — KL Ranking Reproducibility

Target: 60-90 minutes.

Stage 5's earlier correlation check changed calibration data.

Stage 6 must perform a clean reproducibility test.

Do not repeat all 498 modules unless justified.

Select:

    top ~25 Stage 5 modules
    ~10 middle-ranked modules
    ~10 low-ranked modules

Use the SAME calibration examples as the original Stage 5 ranking.

Repeat the sensitivity measurement under the same algorithm.

Measure:

    Spearman
    top-k overlap
    score magnitude stability
    sign stability

Then run a second comparison using an alternative calibration subset.

Report separately:

### Numerical repeatability

    same data
    same algorithm

### Calibration robustness

    different data

Do not conflate them.

Question:

> Is the Stage 5 ranking stable enough to be used for future fixed-budget
> allocations?

If same-data reproducibility is poor:

    diagnose numerical/reduction sensitivity.

If same-data is strong but cross-data weak:

    calibration diversity is the main issue.

Commit.

Suggested:

    experiment: validate KL quant ranking reproducibility

---

# Phase 5 — Finalize the Weight Pareto Frontier

Target: 45-60 minutes.

Using all Stage 6 evidence, create a concise table:

    g64
    g128
    mixed34
    Q1
    Q2 if tested
    Q3 if tested

Include:

    payload GiB
    effective BPW
    V2 quality
    heldout_v2 quality
    NLL/KL
    8K decode
    12K decode
    16K decode
    16K peak
    role

Select:

    DEFAULT
    optional QUALITY MODE
    optional LONG-CONTEXT/PERFORMANCE MODE
    REFERENCE/REJECTED

It is acceptable for DEFAULT to remain g128.

Do not force a new winner.

This phase defines the student for DWQ.

---

# Phase 6 — Bounded DWQ on One Selected Student

Target: 150-210 minutes.

Only begin after the weight Pareto decision.

Choose ONE primary student:

    DEFAULT candidate

Optionally one tiny comparison student only if implementation work is shared.

## Inspect upstream

Read current DWQ implementation and exact source revision.

Understand:

    what parameters are trained
    teacher target
    optimizer
    loss
    calibration pipeline
    checkpoint format

## Design bounded version

The target is a memory-safe streamed/layer-local method.

Possible architecture:

1. fixed calibration samples;
2. capture layer input activations in bounded chunks;
3. load teacher representation for one layer/module;
4. run student quantized layer;
5. optimize allowed scale/bias/non-quantized parameters;
6. validate on held-out calibration subset;
7. save updated layer;
8. free teacher/student temporary arrays;
9. continue.

Do not claim this is exact upstream DWQ if materially different.

## Dry run

Use:

    1 layer
    then a small group of layers

Measure:

    memory peak
    loss before/after
    wall time
    optimizer stability
    output/logit impact

## Expansion gate

Expand to more layers only if:

    loss consistently improves
    memory is safe
    runtime fits remaining stage window

Prioritize layers/modules identified as highly valuable by Stage 5 KL ranking.

Do not necessarily refine all 64 layers.

A targeted bounded-DWQ pass over the most sensitive layers is allowed.

## Artifact limit

At most:

    one main DWQ/refined 27B model

plus one debug partial artifact.

Do not create many refined checkpoints.

## Evaluation

If a usable refined model is produced:

    NLL/KL subset
    corrected heldout
    selected V2 cases
    8K decode
    16K decode if size/residency unchanged and safe

DWQ must produce a measurable practical improvement to become a mode.

Commit.

Suggested:

    experiment: apply bounded DWQ to Qwen3.8 winner

---

# Phase 7 — Final Weight/DWQ Tournament

Target: 60-90 minutes.

Compare at most:

    g128
    best KL-guided candidate
    DWQ-refined winner

mixed34/g64 only where useful as references.

Run:

    corrected heldout_v2
    relevant full Quality Suite V2
    teacher-forced NLL/KL

Produce final quality verdict.

This is the definitive weight-model selection for Stage 6.

---

# Phase 8 — 20K Practical Context Envelope

Target: 60-75 minutes.

Run only after final model selection.

Test DEFAULT at:

    ~20K initial context
    3 short exact-suffix persistent follow-ups

Start with normal KV.

Record:

    cold prefill
    prompt t/s
    warm TTFT
    decode
    peak
    active+cache
    RSS
    memory pressure
    correctness

If normal KV is marginal but system-safe:

    test KV8 once.

## 24K gate

One 24K smoke is allowed only if:

    20K is clearly stable
    no severe swap pressure
    useful decode remains
    projected memory is safe

Never 32K.

Report:

    recommended daily context
    practical long-context mode
    maximum tested safe context

Do not equate "survived once" with "recommended".

Commit.

---

# Phase 9 — Qwen3.8 MTP Investigation

Target: 90-150 minutes.

Only run after final weight model selection and 20K context test if time permits.

Stage 5 established that source checkpoint contains:

    15 mtp.* tensors
    ~0.791 GiB BF16 raw

Now determine whether they can be used.

## Static reverse engineering

Inspect:

    original config
    tensor names/shapes
    upstream/reference Qwen3.8 implementation
    current Qwen3.x MTP implementations in mlx-lm
    generation/speculative APIs

Document:

    exact MTP architecture
    how hidden states feed the MTP head
    how acceptance should work
    cache implications
    normalization
    token embedding/head sharing

Do not guess.

## Memory reduction

Estimate:

    BF16 MTP payload
    8-bit
    4-bit
    3-bit

where supported.

If a quantized MTP head can be safely generated from source, prefer it over raw
BF16 if correctness remains acceptable.

Do not overwrite the main model.

## Minimal runtime

Implement the smallest isolated MTP generation path needed for testing.

Do not redesign the server.

Correctness first.

## Smoke

Compare:

    normal greedy
    MTP-assisted

Measure:

    token correctness
    acceptance rate
    decode tok/s
    added active/peak memory

If the algorithm should preserve greedy output:

    require token identity.

## Context

If short smoke succeeds:

    4K
    8K

Then only if memory is comfortable:

    16K

Important:

MTP that accelerates 4K but triggers the same kind of residency cliff at 16K is
not the general default.

## Decision

Classify:

    PRACTICAL
    SHORT-CONTEXT ONLY
    NOT READY
    UNSUPPORTED

Commit.

Suggested:

    experiment: implement Qwen3.8 MTP decode prototype

---

# Phase 10 — Real Serialized Agent-Backend Validation

Target: 60-90 minutes.

Use:

    research/runtime/qwen38_agent_server.py

or current equivalent.

Run with the final DEFAULT model.

Keep one inference at a time.

## Real integration smoke

Create a scripted client that behaves like a coding agent.

Use:

    realistic system/tool prompt
    repository-like supplied context
    tool calls/results
    user/assistant/tool sequencing
    10-20 exchanges
    context growing toward ~8K-12K

The exact tool-call format must match what the server/model actually supports.

Do not invent a fake structured protocol and call it production-ready.

Check:

    session persistence
    exact-suffix token reuse
    no full reprefill
    tool-result retention
    role sequencing
    reset/session change
    warm TTFT
    decode
    no stale replay
    graceful error recovery

Measure:

    first request TTFT
    median warm TTFT
    max warm TTFT
    decode
    total logical context
    suffix tokens

If native structured tool calls are not reliable:

    document that limitation
    validate generic chat/tool-role transport separately.

## Optional client compatibility

If easy, test one OpenAI-compatible client library.

Do not spend hours adapting third-party frameworks.

Commit.

Suggested:

    test: validate Qwen3.8 serialized agent backend

---

# Phase 11 — Production Runtime Cleanup

Target: 30-60 minutes.

Based on all findings:

Update:

    qwen38_chat.py
    qwen38_agent_server.py

as needed.

Document:

    default model path
    recommended context
    optional KV8
    optional MTP mode if validated
    quality mode if one exists
    reset behavior
    known limitations

Add regression tests for any new Stage 6 bug discovered.

Do not add concurrency.

Commit.

Suggested:

    feat: finalize Qwen3.8 local runtime configuration

---

# Phase 12 — Final Report

Create:

    research/stage6/FINAL_REPORT.md

Update:

    research/stage6/STATUS.md
    research/stage6/RESULTS.md

Inspect:

    raw artifacts
    results.jsonl
    git status
    git log
    model paths

The final report must distinguish:

    MEASURED FACT
    INFERENCE
    HYPOTHESIS

---

# Explicitly Out of Scope

Do not spend Stage 6 on:

    uniform 2-bit
    mixed_2_6
    random new quant recipes
    full new fixed-budget frontier unless Q1/Q2/Q3 are lost
    generic uniform 4-bit
    DSpark
    32K
    multi-user batching
    concurrent inference
    system wired-memory tuning
    training a new MTP head

Do not do AWQ/GPTQ unless DWQ fails early AND substantial useful time remains.

---

# Machine-Readable Results

Append meaningful runs to:

    research/stage6/results.jsonl

Suggested fields:

    timestamp
    phase
    experiment_id
    git_commit

    model_name
    model_path
    payload_bytes
    effective_bpw
    quant_recipe

    suite_name
    suite_hash
    case_id
    category
    objective_pass
    finish_reason
    early_eos
    repetition
    structured_valid

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

    mtp_enabled
    mtp_payload_bytes
    mtp_acceptance_rate

    dwq_enabled
    dwq_layers_refined
    dwq_loss_before
    dwq_loss_after

    classification
    exception
    safety_stop_reason

Use null if unavailable.

Never invent measurements.

---

# Final Report Must Answer

1. Were Q1/Q2/Q3 successfully recovered/preserved?

2. What truncation/scoring defects were corrected in heldout_v2?

3. What is the corrected held-out suite hash?

4. How does Q1 compare with g128 on:
       V2 quality
       heldout_v2
       NLL/KL
       8K
       12K
       16K

5. Was Q2 tested?
   Why or why not?

6. Was Q3 tested?
   Why or why not?

7. What is the final fixed-weight Pareto frontier?

8. Is KL ranking numerically reproducible on identical calibration data?

9. How robust is the ranking across calibration subsets?

10. Was bounded DWQ implemented?

11. What memory did DWQ require?

12. Did DWQ reduce teacher/loss divergence?

13. Did DWQ improve actual held-out quality?

14. What is the final DEFAULT model?

15. Is there a separate QUALITY MODE?

16. Is there a separate PERFORMANCE/LONG-CONTEXT MODE?

17. What are final 8K/12K/16K numbers?

18. Is 20K practical?

19. Was 24K tested?

20. What context should the user actually use daily?

21. Was Qwen3.8 MTP implemented?

22. What MTP tensor precision/payload was used?

23. What was MTP acceptance rate?

24. What was MTP decode speedup at 4K/8K/16K?

25. Did MTP introduce a memory/residency cliff?

26. Did the real serialized agent smoke pass?

27. What warm TTFT/decode did the agent backend show?

28. Are structured tool calls reliable?

29. What exact CLI command should the user run TODAY?

30. What exact server command should the user run TODAY?

31. What are the three best remaining Stage 7 ideas?

Include tables:

### Final Weight Pareto

    model
    size GiB
    BPW
    V2
    heldout_v2
    NLL/KL
    8K decode
    16K decode
    role

### Context Envelope

    context
    KV mode
    warm TTFT
    decode
    peak
    verdict

### Acceleration

    baseline
    MTP
    payload overhead
    acceptance
    decode
    correctness
    verdict

---

# Completion Condition

Stage 6 is complete when:

- Q1 is fully evaluated;
- Q2/Q3 are evaluated only as justified;
- held-out truncation artifacts are corrected;
- KL ranking repeatability is measured correctly;
- bounded DWQ produces useful evidence or is safely rejected;
- final weight-model roles are chosen;
- 20K is evaluated if safe;
- MTP is investigated if time permits;
- real agent backend smoke is attempted;
- FINAL_REPORT.md exists;
- STATUS.md is current;
- useful changes are committed.

Do not start Stage 7 automatically.

At the end return a concise summary with:

    default model
    Q1/Q2/Q3 verdicts
    quality result
    8K/12K/16K/20K result
    DWQ verdict
    MTP verdict
    agent backend verdict
    recommended CLI command
    recommended server command
    FINAL_REPORT path
    commit SHAs

Then STOP.
