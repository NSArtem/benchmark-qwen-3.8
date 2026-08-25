# Stage 5 — 15-Hour Qwen3.8 Optimization Campaign

## Mission

Run an approximately 15-hour autonomous research campaign.

Stage 5 is allowed to be broad, but not uncontrolled.

The critical path is:

    recover Stage 4
    -> revalidate mixed34 16K cliff
    -> build completely held-out quality set
    -> loss/logit/gradient-guided sensitivity
    -> fixed-budget quant Pareto frontier
    -> mini tournament
    -> bounded DWQ feasibility/refinement
    -> final quality/Pareto tournament
    -> activation quantization
    -> 20K context envelope
    -> gated MTP
    -> serialized real-agent backend
    -> final report

Use gates.

Do not turn this into random quant exploration.

---

# Phase 0 — Recover State

Target: 20-30 minutes.

Start:

    git status
    git log --oneline -30

Read:

    AGENTS.md
    research/stage4/FINAL_REPORT.md
    research/stage4/STATUS.md
    research/stage4/RESULTS.md
    research/stage4/results.jsonl

Inspect Stage 4 raw artifacts for:

    Quality Suite V2
    g64 results
    g128 results
    mixed34 results
    sensitivity-guided rejected candidate
    8K/16K runtime validation
    qwen38-chat
    early-EOS study

Recover exact model paths.

Verify:

    g64
    g128
    mixed34
    high-precision source
    bounded converter
    qwen38-chat

Create:

    research/stage5/PLAN.md
    research/stage5/STATUS.md
    research/stage5/RESULTS.md
    research/stage5/results.jsonl

Record the entering Pareto table.

Do not rerun completed Stage 4 work unless a later phase requires a controlled
comparison.

---

# Phase 1 — Reproduce and Localize the mixed34 16K Decode Cliff

Target: 60-90 minutes.

This is P0.

Question:

> Does mixed34 really lose nearly half its decode speed near 16K, and if so,
> where does the cliff begin?

## Part A — Interleaved fresh-process control

Use identical:

    prompt family
    exact prompt token count
    generation length
    greedy settings
    prefill step
    cache mode
    no KV8
    no activation quant
    canonical environment

Run at ~16K:

    g128
    mixed34
    g128
    mixed34
    g128
    mixed34

Fresh Python process every run.

Do not run simultaneously.

Record:

    prompt tokens
    prefill tok/s
    TTFT
    decode tok/s
    generated tokens
    finish reason
    active
    cache
    active+cache
    peak
    RSS
    memory_pressure summary
    vm_stat summary

Use enough generated tokens to stabilize decode measurement, e.g. 128-256.

## Part B — Context sweep

One fresh run per model per useful point:

    ~8K
    ~10K
    ~12K
    ~14K
    ~16K

If the transition is sharp, optionally add one point between the last fast and
first slow condition.

Do not over-benchmark.

## Analysis

Plot/table:

    context vs decode
    context vs active+cache
    context vs peak
    context vs RSS

for both models.

Test hypotheses:

1. mixed34 result was noise;
2. decode cliff follows context;
3. decode cliff follows total model+cache residency;
4. cliff correlates with system memory pressure;
5. another runtime artifact appears at long context.

Do not assert causality from correlation alone.

## Gate

If mixed34 is actually ~10 tok/s at 16K under controlled repeats:

    clear the Stage 4 performance concern.

If mixed34 reproducibly falls to ~5-6 tok/s:

    mark a real long-context practical penalty.

This result must directly influence quant target size in later phases.

Commit.

Suggested commit:

    benchmark: localize Qwen3.8 16K decode cliff

---

# Phase 2 — Completely Held-Out Quality Suite

Target: 60-90 minutes setup + run.

Create a NEW frozen held-out set.

Target:

    ~20-30 cases

None of these cases may be used in:

    Stage 4 calibration
    Stage 4 sensitivity
    Stage 4 mini tournament
    Stage 4 Quality Suite V2
    Stage 5 sensitivity/calibration
    Stage 5 candidate selection

This is final-selection evidence.

## Categories

Use a balanced set such as:

    coding / debugging          6-8
    reasoning / arithmetic      4-5
    structured output           3-4
    instruction following       3-4
    context retrieval           3-4
    prose continuity            2-3

All objective expected answers must be manually verified before any model run.

Avoid ambiguous lease/time semantics.

Use enough output budget:

    256-512 for coding/reasoning
    shorter only when answer is intentionally short

Record finish reason.

## Initial comparison

Run:

    g64
    g128
    mixed34

Do not yet run future Stage 5 quants.

Report:

    objective total
    category totals
    malformed output
    early EOS
    repetition
    prose/rubric result

Also select a teacher-forced subset and record:

    NLL
    top-1 agreement
    optional KL/logit metrics

Freeze files and hashes.

Commit.

Suggested commit:

    benchmark: add held-out Qwen3.8 quality suite

---

# Phase 3 — End-to-End Quant Sensitivity

Target: 90-150 minutes.

This phase replaces Stage 4's weak RMS proxy.

Goal:

> rank possible precision promotions by estimated END-TO-END language-model
> quality recovery per added byte.

Inspect current upstream mlx-lm implementation of:

    dynamic_quant
    learned quantization helpers
    quant predicates
    quantized modules

Record exact upstream commit inspected.

Do not upgrade canonical runtime.

## Preferred signal

Use one or more of:

    KL divergence to teacher logits
    NLL improvement
    gradient-informed low-vs-high quant sensitivity
    logit recovery

on a calibration split.

Do NOT use held-out Stage 5 evaluation.

## Teacher

Prefer the highest-quality practical teacher that can be used safely.

Possible:

    high-precision source
    g64
    another high-precision/8-bit source

but memory must remain bounded.

Do not load two complete 27B models concurrently.

## Bounded algorithm

If upstream algorithm assumes whole-model duplication/deepcopy and is unsafe:

    adapt it.

Use existing bounded converter/layer infrastructure.

Possible design:

1. capture/replay bounded calibration activations;
2. inspect one quantizable module/layer at a time;
3. compare low and high precision effect on end-to-end or downstream logits
   through a memory-safe approximation;
4. release source tensors before next module.

Document approximation honestly.

## Sensitivity artifact

Save:

    research/stage5/quant/loss_sensitivities.json

Fields should include:

    module path
    layer index
    module family
    parameter count
    3-bit storage estimate
    4-bit storage estimate
    optional 5/6-bit estimate
    bytes added for promotion
    sensitivity score
    quality-recovery-per-byte score
    calibration sample count
    calibration hash

## Determinism

Run a small subset twice.

Measure ranking stability.

If scores are numerically noisy:

    save one frozen deterministic ranking/order
    use explicit module lists
    do not repeatedly optimize thresholds

## Validation

Check whether the ranking predicts actual calibration NLL/KL recovery for a
small number of controlled module-promotion experiments.

If it does not predict actual recovery, do not build a large frontier from it.

Gate into Phase 4 only with useful predictive evidence.

Commit.

Suggested commit:

    experiment: map loss-guided Qwen3.8 quant sensitivity

---

# Phase 4 — Fixed-Budget Quant Pareto Frontier

Target: 90-120 minutes.

Use high-precision source and bounded converter.

Create at most FOUR serious new Stage 5 checkpoints.

Base:

    affine 3-bit group128

## Candidate budgets

Aim roughly for:

    Q1 = g128 + ~150 MiB
    Q2 = g128 + ~300 MiB
    Q3 = g128 + ~450 MiB

Use module granularity; exact payload may differ.

Promotion priority:

    descending frozen quality-recovery-per-byte ranking

Primary promotion:

    3 -> 4 bit

## Matched-size precision-allocation experiment

At ONE useful budget, optionally compare:

    broad 3/4 allocation

vs:

    sparse 3/5 or 3/6 allocation

only if:

- format is supported safely;
- extreme sensitive outliers exist;
- payload can be approximately matched;
- this does not exceed the four-candidate limit.

## Important memory-cliff constraint

Use Phase 1 results.

If Phase 1 shows a strong decode cliff above a certain payload/residency region,
do not knowingly build most candidates beyond that region.

The point is a practical Pareto frontier, not quality at unusable memory.

## Conversion recording

For every candidate:

    exact module list
    ranking hash
    payload bytes
    effective BPW
    conversion time
    converter peak
    RSS
    shards
    config metadata

## Immediate smoke gate

Fresh process:

    load
    short factual prompt
    768-token prompt
    4K coding prompt
    128-256 output

Reject for:

    repetition/collapse
    semantic garbage
    config/load bug
    unacceptable decode regression
    unsafe memory

Do not full-benchmark rejected models.

Commit recipes/harnesses/results, not weights.

Suggested commit:

    experiment: build fixed-budget Qwen3.8 quant frontier

---

# Phase 5 — Fast Quant Tournament

Target: 60-90 minutes.

Compare:

    g128
    mixed34
    Q1/Q2/Q3
    optional matched-size sparse candidate

g64 may be included only as reference for selected teacher-forced metrics.

Use:

    Stage 4 Quality Suite V2 mini subset
    Stage 5 non-held-out mini subset
    teacher-forced calibration/eval subset

Do NOT use the Stage 5 final held-out set for candidate selection.

Metrics:

    objective quality
    NLL delta
    top-1/logit agreement
    model size
    effective BPW
    8K decode
    projected/quick 16K behavior
    repetition
    early EOS

Select:

    at most TWO new finalists

A valid result is:

    no new fixed-budget candidate beats g128/mixed34.

Do not tune indefinitely.

Commit.

Suggested commit:

    benchmark: select Stage 5 quant finalists

---

# Phase 6 — Bounded DWQ Investigation and Optional Refinement

Target: 120-180 minutes.

This phase is important but gated.

First inspect current upstream:

    DWQ implementation
    teacher/student residency
    optimizer state
    calibration flow
    supported quantized parameters
    memory reduction options

Record exact source commit.

## Feasibility calculation

Before a real 27B run estimate:

    teacher bytes
    student bytes
    optimizer/tunable bytes
    activation bytes
    calibration cache
    expected peak

If naive method is unsafe:

    do not run it.

## Preferred bounded approach

Explore a layer-wise/streamed design:

    load/capture calibration input for layer
    teacher layer forward
    student quantized layer forward
    optimize relevant quant parameters/scales/biases
    validate layer/calibration loss
    save updated layer
    release teacher/student tensors
    continue

Exact implementation may differ based on actual upstream DWQ semantics.

Do not claim equivalence to upstream DWQ if algorithm differs materially.

## Dry run

Before the full model:

    1-2 layers
    small calibration subset
    short sequence length

Measure:

    loss trend
    memory
    wall time
    numerical stability

## Full refinement gate

Only perform a broader refinement if:

- dry run lowers a meaningful end-to-end/local teacher loss;
- memory is clearly safe;
- projected total runtime fits remaining stage time.

Preferred students:

    g128
    OR
    best fixed-budget finalist

Do not refine every model.

At most TWO DWQ/refined serious artifacts.

## Evaluation gate

Run a compact quality/NLL subset.

If refinement has no measurable gain:

    stop DWQ branch.

If gain is promising:

    advance at most ONE DWQ model to final tournament.

Commit.

Suggested commits:

    research: prototype bounded Qwen3.8 DWQ
    experiment: refine Qwen3.8 quant with bounded DWQ

---

# Phase 7 — Final Quality and Pareto Tournament

Target: 90-120 minutes.

Finalists should be no more than:

    g128
    mixed34
    best fixed-budget quant
    best DWQ/refined quant

Prefer 3 total if weak candidates have already been eliminated.

Run:

    full Stage 4 Quality Suite V2
    full Stage 5 held-out suite

The Stage 5 held-out set must NOT have influenced candidate allocation or
selection prior to this phase.

## Report objective evidence separately

For each model:

    objective score
    category scores
    structured validity
    arithmetic/reasoning
    context retrieval
    instruction following
    coding rubric/objective checks
    early EOS count
    repetition/collapse

Teacher-forced:

    mean NLL
    median per-case NLL
    NLL delta vs g128/g64
    top-1 agreement
    KL/logit metrics where available

Behavioral:

    output length distribution
    EOS frequency
    malformed outputs
    severe contradictions

Optional blind subjective evaluation:

    coding/prose pairwise

with anonymized model labels.

Supplementary only.

## Pareto table

For all finalists include:

    payload GiB
    effective BPW
    quality
    8K decode
    16K decode
    8K peak
    16K peak

Choose roles:

    DEFAULT
    QUALITY MODE if justified
    PERFORMANCE/LONG-CONTEXT MODE if justified
    REJECTED/REFERENCE

Do not preserve roles just for symmetry.

If g128 is best overall, say so.

Commit.

Suggested commit:

    benchmark: finalize Qwen3.8 Stage 5 Pareto frontier

---

# Phase 8 — Activation Quantization

Target: 45-60 minutes.

Only run on:

    DEFAULT winner

Optionally one secondary mode if there is a strong reason.

Compare:

    normal activations
    activation quantization

At:

    ~8K
    ~16K

Primary focus:

    16K

Record:

    peak
    active+cache
    RSS
    decode
    TTFT
    prefill
    correctness smoke
    logit/output divergence if relevant

Question:

> Does activation quantization materially reduce long-context residency or
> remove a performance cliff?

Reject it if:

    memory benefit is trivial
    decode cost is large
    quality visibly degrades
    runtime becomes unstable

Do not promote it merely because it works.

Commit if meaningful.

---

# Phase 9 — Practical 20K Envelope

Target: 45-75 minutes.

Run only if the selected winner is stable at 16K.

Test:

    ~20K normal cache

If memory/headroom is marginal, also test:

    ~20K KV8

If activation quantization was validated as useful, one QA variant is allowed.

Use:

    cold prefill
    3 short persistent follow-up turns

Measure:

    cold TTFT
    warm TTFT
    decode
    peak
    active+cache
    RSS
    system pressure
    correctness

## 24K gate

ONE ~24K smoke is allowed only if:

- 20K is clearly stable;
- system remains responsive;
- peak/headroom is comfortable;
- decode remains useful;
- projected 24K is safe;
- no prior Metal instability occurred.

Do NOT test 32K.

Classify:

    recommended practical context
    largest tested safe context
    largest tested but marginal context

These may differ.

Commit.

---

# Phase 10 — Gated MTP Investigation

Target: 60-120 minutes.

Only run if primary quant/Pareto work is complete.

Goal:

> Determine whether Qwen3.8's original checkpoint has a usable MTP path that
> can materially increase decode throughput without unacceptable memory cost.

## Static inspection

Inspect:

    original source config
    source tensor names
    prior text-only conversion exclusions
    current mlx-lm MTP support
    relevant Qwen3.x model implementations

Determine:

    whether Qwen3.8 includes MTP weights
    their approximate payload
    compatibility with current architecture/runtime

Do not assume.

## Conversion

If compatible MTP tensors exist and the implementation path is clear:

    adapt bounded converter to preserve/include them

Do not rebuild all quant candidates.

Use only:

    DEFAULT weight quant

or one close equivalent.

## Smoke correctness

Require:

    baseline greedy
    MTP-assisted output

to satisfy the correctness guarantee of the actual algorithm.

If the implementation claims token-identical greedy output, verify it.

Measure:

    acceptance rate
    decode tok/s
    additional memory
    TTFT
    8K
    optionally 16K if safe

## Gate

Continue only if:

    speedup is substantial
    correctness is clean
    memory overhead is practical

If incompatible/absent, document and stop.

Do not train a new MTP head.

Commit.

Suggested commit:

    experiment: evaluate Qwen3.8 MTP decode

---

# Phase 11 — Serialized Agent Backend

Target: 60 minutes.

qwen38-chat already exists.

Create or complete a production-oriented serialized backend suitable for real
coding-agent use.

Preferred:

    OpenAI-compatible HTTP API

with exactly one active inference at a time.

Required:

    persistent session/conversation
    canonical chat template
    exact-suffix cache
    tool/system/user/assistant messages
    session reset
    deterministic mode
    normal sampling mode
    metrics
    graceful errors

Do not implement concurrency/batching.

## Real agent smoke

Run a local scripted coding-agent style session with:

    substantial system/tool prompt
    repository-like context
    several tool-call/result turns
    8K-class logical history
    at least ~10 exchanges

Check:

    no full reprefill
    suffix token accounting
    valid message sequencing
    tool-result persistence
    reset behavior
    warm TTFT
    decode
    no stale replay

If tool-call parsing is not natively standardized for this model/runtime, keep
the API generic and document the limitation instead of inventing a brittle
format.

Commit.

Suggested commit:

    feat: add serialized Qwen3.8 agent backend

---

# Phase 12 — Final Synthesis

Update:

    research/stage5/STATUS.md
    research/stage5/RESULTS.md

Create:

    research/stage5/FINAL_REPORT.md

Before writing conclusions:

    inspect raw artifacts
    inspect results.jsonl
    verify model paths
    verify git status/log
    verify no uncommitted important harness changes

The report must distinguish:

    MEASURED FACT
    INFERENCE
    HYPOTHESIS

---

# Explicitly Out of Scope

Do NOT spend Stage 5 on:

    uniform 2-bit
    mixed_2_6
    random MXFP4/NVFP4 sweep
    generic full 4-bit rebuilds
    multi-user batching
    concurrent server work
    DSpark
    32K context
    system wired-memory tuning
    training a new MTP head

AWQ/GPTQ may be inspected only if DWQ is blocked and there is real remaining
time. They are not a primary branch.

---

# Machine-Readable Results

Append meaningful experiments to:

    research/stage5/results.jsonl

Suggested fields:

    timestamp
    experiment_id
    phase
    git_commit

    model_name
    model_path
    quant_recipe
    quant_group_size
    model_payload_bytes
    effective_bpw
    promoted_modules
    ranking_hash
    calibration_hash
    heldout_hash

    context_tokens
    suffix_tokens
    generated_tokens
    finish_reason
    output_hash

    objective_pass
    category
    early_eos
    repetition
    structured_valid

    nll
    nll_delta
    kl
    top1_agreement

    ttft_seconds
    prompt_tps
    decode_tps

    active_memory_bytes
    cache_memory_bytes
    active_plus_cache_bytes
    peak_memory_bytes
    rss_bytes

    memory_pressure_summary
    exception
    safety_stop_reason
    classification

Use null for unavailable values.

Never invent data.

---

# Commit Suggestions

Prefer logical commits such as:

    benchmark: localize Qwen3.8 16K decode cliff
    benchmark: add held-out Qwen3.8 quality suite
    experiment: map loss-guided Qwen3.8 quant sensitivity
    experiment: build fixed-budget Qwen3.8 quant frontier
    benchmark: select Stage 5 quant finalists
    research: prototype bounded Qwen3.8 DWQ
    experiment: refine Qwen3.8 quant with bounded DWQ
    benchmark: finalize Qwen3.8 Stage 5 Pareto frontier
    benchmark: evaluate activation quantization
    benchmark: probe Qwen3.8 20K context
    experiment: evaluate Qwen3.8 MTP decode
    feat: add serialized Qwen3.8 agent backend
    docs: summarize Qwen3.8 Stage 5 findings

Do not commit weights.

---

# FINAL_REPORT.md Must Answer

1. Was the mixed34 16K decode cliff reproduced?

2. At what context/residency does the cliff begin, if real?

3. Does mixed34 beat g128 on the completely held-out quality suite?

4. What is the Stage 5 held-out suite composition and hash?

5. What end-to-end sensitivity method was implemented?

6. How stable/deterministic was the sensitivity ranking?

7. Which modules/families had the highest quality-recovery-per-byte?

8. Which fixed-budget candidates were built?

9. What were their exact sizes/effective BPW?

10. Did broad 3/4 or sparse higher-bit promotion win at matched size?

11. Which candidates survived the mini tournament?

12. Was bounded DWQ feasible?

13. Did DWQ measurably improve any finalist?

14. What are the final Quality Suite V2 results?

15. What are the final completely held-out quality results?

16. What are the final NLL/KL/top-1 results?

17. What is the final Pareto frontier?

18. Which model is DEFAULT?

19. Is there a separate QUALITY MODE?

20. Is there a separate PERFORMANCE/LONG-CONTEXT MODE?

21. What are the final 8K and 16K decode/TTFT/peak numbers?

22. Did activation quantization help?

23. Is 20K practical?

24. Was 24K tested, and if so was it practical or merely survivable?

25. Was MTP available for Qwen3.8?

26. If tested, what speedup/acceptance/memory cost did MTP show?

27. Is there now a serialized OpenAI-compatible/local-agent backend?

28. Did a real multi-turn agent-style smoke pass?

29. What exact command/config should the user run TODAY?

30. What are the three highest-value remaining Stage 6 ideas?

Include final tables:

### Quality Pareto

    model
    payload GiB
    effective BPW
    V2 objective
    held-out objective
    NLL
    quality role

### Runtime Pareto

    model
    8K decode
    8K warm TTFT
    8K peak
    16K decode
    16K warm TTFT
    16K peak
    20K status

### Acceleration

    mode
    decode
    memory
    quality/correctness
    verdict

---

# Completion Condition

Stage 5 is complete when:

- mixed34 cliff is resolved;
- held-out suite is frozen and run;
- loss/logit-guided sensitivity is investigated;
- fixed-budget frontier is tested;
- DWQ has either produced useful evidence or been safely rejected;
- final Pareto roles are chosen;
- winner is tested at useful contexts;
- optional acceleration branches are gated correctly;
- serialized agent backend is tested if time permits;
- FINAL_REPORT.md and STATUS.md are complete;
- useful work is committed.

Do not start Stage 6 automatically.

At the end return a concise summary with:

    default model
    alternate modes
    model sizes
    held-out quality verdict
    8K/16K/20K runtime verdict
    mixed34 cliff verdict
    DWQ verdict
    activation quant verdict
    MTP verdict
    agent backend command
    FINAL_REPORT path
    commit SHAs

Then STOP.
