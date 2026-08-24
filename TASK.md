# Stage 4 — Quality-Guided Quantization and Runtime Productionization

## Mission

Run an approximately 10-hour autonomous research stage.

Stage 3 solved the main persistent-cache problem.

Stage 4 must now answer:

> What is the best practical Qwen3.8-27B quant/runtime configuration for this
> 18 GB M3 Pro, balancing quality, model size, context capacity, and speed?

The critical path is:

    recover state
    -> Quality Suite V2
    -> early-EOS targeted control
    -> quantization sensitivity map
    -> at most 2-3 smart quant candidates
    -> mini tournament
    -> full finalist quality comparison
    -> persistent runtime validation
    -> productionize qwen38-chat
    -> optional learned-quant feasibility

This is a gated program.

Do NOT run every possible branch.

---

# Phase 0 — Recover Stage 3 State

Target: short.

Start with:

    git status
    git log --oneline -25

Inspect:

    research/stage3/FINAL_REPORT.md
    research/stage3/STATUS.md
    research/stage3/RESULTS.md
    research/stage3/results.jsonl

Inspect raw artifacts for:

    cache oracle
    8K/12K/16K scaling
    endurance turn 17
    Stage 3 quality suite
    KV8 results

Confirm:

    g64 path
    g128 path
    high-precision source availability
    bounded converter availability

Do not rerun Stage 3 cache validation.

Create:

    research/stage4/PLAN.md
    research/stage4/STATUS.md
    research/stage4/RESULTS.md
    research/stage4/results.jsonl

Record current facts explicitly.

### Required Phase 0 summary

At minimum:

    g64 size
    g128 size
    g128 persistent 8K/12K/16K performance
    Stage 3 endurance status
    Stage 3 quality-suite limitations
    existing rejected quant candidates
    existing converter paths
    high-precision source path/revision

Commit only if new project scaffolding is meaningful.

---

# Phase 1 — Quality Suite V2

Target: 90-120 minutes.

This is P0.

The Stage 3 quality suite is not sufficient for final quant decisions.

Build a corrected, compact, diverse evaluation suite.

Target:

    ~30-40 total cases

Do not optimize for quantity.

Optimize for:

    correctness
    diversity
    objective scoring
    enough output budget

## Split

Create:

    calibration/
    evaluation/

Do NOT calibrate dynamic/mixed quant selection on the exact same full evaluation
set.

A reasonable split might be:

    ~10-12 calibration items
    ~24-30 evaluation items

The exact count may differ.

## Suggested categories

### Coding / repository reasoning: ~8-10

Examples:

- race-condition diagnosis;
- invariant selection;
- patch plan;
- SQL/transaction correctness;
- API idempotency;
- code-review defect identification;
- minimal regression test design.

Output budget:

    usually 256-512 tokens

Objective scoring should focus on:

- whether the root cause is identified;
- whether required invariant/check is present;
- whether proposed fix addresses the actual bug;
- whether an explicitly requested test condition is included.

Do not require a magic keyword if equivalent language is acceptable.

### Instruction following: ~4-5

Examples:

- strict requested format;
- requested exclusions;
- exact number of items;
- required final phrase where appropriate;
- preserve supplied terminology.

Score mechanically where possible.

### Structured output: ~4-5

Examples:

- JSON object with exact keys;
- JSON list under a schema;
- table-like machine-readable record.

Check:

    parse success
    required keys
    types
    semantic constraints

### General reasoning/arithmetic: ~4-5

Use questions with unambiguous expected answers.

Manually verify every expected answer before the benchmark.

Avoid vague temporal/lease semantics unless fully specified.

### Context retrieval/consistency: ~4-5

Supply several-thousand-token local contexts.

Ask about:

- an early constraint;
- two distant facts;
- a comparison;
- a deliberately similar distractor.

Objective fact matching is preferred.

### TTRPG/prose continuity: ~3-5

Use supplied fictional state.

Check:

- names/facts preserved;
- no contradiction;
- requested tone/constraint observed;
- no invented critical facts.

Subjective judgment may be used only as a supplement.

## Generation settings

Default:

    greedy / deterministic
    enable_thinking=False if appropriate for the model template

Use output budgets large enough to complete.

Record:

    finish reason
    generated tokens
    EOS vs max-token stop
    wall time
    decode rate

## Teacher-forced subset

Select a representative subset.

Compute for g64 and g128:

    mean NLL
    per-case NLL
    token-level top-1 agreement
    optionally KL/logit similarity

Do not aggregate away category-level differences.

## Stage 1 validation gate

Before using this suite to judge new quants:

1. run g64;
2. inspect all objective failures;
3. correct wrong/ambiguous expected answers;
4. ensure obvious coherent answers are not failing due to harness mistakes;
5. run g128;
6. manually inspect all cases where g64/g128 disagree.

Only then freeze:

    evaluation_v2.jsonl
    calibration_v2.jsonl

Store hashes.

Commit.

Suggested commit:

    benchmark: build Qwen3.8 quality suite v2

---

# Phase 2 — Turn-17 Early-EOS Targeted Study

Target: 30-45 minutes.

Do not turn this into another endurance campaign.

Recover the exact Stage 3 context and turn-17 user prompt.

Compare, where practical:

    g128 persistent
    g128 fresh full-transcript
    g64 persistent
    g64 fresh full-transcript

Use the same logical transcript.

For each path capture:

    next-token top-k logits/probabilities
    EOS rank/probability
    first 16 generated tokens
    finish reason
    generated length
    cache accounting
    output

Repeat only as useful.

Greedy should be deterministic; repetitions are not required just to prove that.

## Decision

### If early EOS occurs in fresh and persistent similarly

Classify:

    model/quant generation-quality event

Close runtime-cache concern.

### If only g128 shows it

Record as g128 quality evidence.

### If only persistent shows it

This contradicts the current Stage 3 runtime conclusion.

Preserve evidence and perform one narrow follow-up.

Do NOT automatically reopen the full cache investigation unless state/token
accounting evidence points there.

Commit if useful.

Suggested commit:

    experiment: diagnose Qwen3.8 endurance early EOS

---

# Phase 3 — Quantization Sensitivity Study

Target: 60-120 minutes.

This is the main new research branch.

Goal:

> identify which quantizable modules are most sensitive, so additional precision
> can be spent selectively.

First inspect the current pinned and upstream mlx-lm implementations of:

    dynamic_quant
    quantize_model
    quant predicates
    mixed recipes

Also inspect the existing bounded converter.

Do not blindly run upstream Dynamic Quantization if it would require loading an
unsafe full model/teacher into 18 GB.

## Preferred outcome

Produce a reusable sensitivity artifact, e.g.:

    research/stage4/quant/sensitivities.json

containing:

    module path
    sensitivity score
    module type/category
    layer index
    tensor dimensions
    parameter count

## Calibration data

Use ONLY the frozen Stage 4 calibration split or a documented subset.

Do not use the full evaluation split.

## Safety gate

Before a full sensitivity pass:

1. run a small representative subset/modules;
2. measure memory;
3. verify the algorithm is bounded enough for this machine.

If unsafe:

    STOP that implementation path.

Then implement/adapt a bounded sensitivity analysis using existing
layer-oriented infrastructure.

A slower bounded method is preferable to an unsafe full-model method.

## Analysis

Aggregate sensitivity by:

    early / middle / late layers
    attention vs GatedDelta/recurrent layers
    q_proj
    k_proj
    v_proj
    o_proj
    gate_proj
    up_proj
    down_proj
    lm_head
    any architecture-specific projections

Create:

    sensitivity ranking
    cumulative parameter curve
    cumulative estimated BPW/size curve

Answer:

> If we can afford +100 MiB, +200 MiB, +400 MiB over g128, where should those
> extra bits go first?

## Cross-check

Where feasible, compare sensitivity ranking with actual g64-vs-g128 logit/NLL
degradation on calibration items.

Sensitivity that has no relationship to observed error should not be trusted
blindly.

Commit.

Suggested commit:

    experiment: map Qwen3.8 quantization sensitivity

---

# Phase 4 — Create Smart Quant Candidates

Target: 45-75 minutes.

Create at most THREE serious candidates.

Do not create a quant zoo.

Use the high-precision source and bounded converter.

All candidates should default to group_size=128 unless the evidence strongly
supports another choice.

## Candidate A — Built-in/simple mixed 3/4 control

Purpose:

    establish a standard mixed-bit reference.

Use semantics equivalent to mlx-lm mixed_3_4 where practical:

    low precision = 3-bit
    high precision = 4-bit

Adapt the existing bounded converter rather than using an unsafe whole-model
conversion.

Record exactly which modules receive 4-bit.

Call it something unambiguous, e.g.:

    qwen38-mixed34-control-g128

## Candidate B — Sensitivity-guided 3/4

This is the primary candidate.

Base:

    affine 3-bit group128

Promote the most sensitive modules to:

    affine 4-bit group128

Choose a target budget informed by the sensitivity curve.

Preferred target region:

    ~10.3-10.6 GiB

Do not force that size if the sensitivity structure suggests a better nearby
tradeoff.

Record:

    promoted modules
    promoted parameter count
    final payload
    effective BPW

## Candidate C — Optional

Only create if Candidate B leaves a clear unresolved quality gap AND the
sensitivity data shows a small extreme-outlier set.

Possible form:

    3-bit group128 base
    6-bit group128 only for the most sensitive modules

or an equivalent targeted 3/6 scheme.

Do NOT create Candidate C just because time exists.

## Immediate smoke gate for each candidate

Before broader benchmark:

    load
    33-token prompt
    ~768 prompt
    one 4K coding prompt
    128-256 output tokens

Reject immediately for:

    incoherence
    deterministic repetition loop
    loading/config mismatch
    pathological memory
    obviously unusable decode

Record conversion stats and smoke result.

Commit conversion/harness changes separately from giant model outputs.

Do not commit weights.

Suggested commit:

    experiment: build quality-guided Qwen3.8 quants

---

# Phase 5 — Mini Quant Tournament

Target: 60-90 minutes.

Compare:

    g64
    g128
    Candidate A
    Candidate B
    Candidate C if it exists

Do NOT run full evaluation suite on every weak candidate.

Use a mini subset:

    ~10-12 evaluation_v2 cases

chosen to include:

    coding
    arithmetic/reasoning
    structured output
    context retrieval
    instruction following

Also run teacher-forced metrics on a small representative subset.

For each model record:

    model payload GiB
    effective BPW
    4K peak
    8K peak if useful
    decode t/s
    objective mini-suite score
    per-category result
    NLL delta vs g64
    top-1 agreement vs g64
    repetition/collapse
    finish reason anomalies

## Tournament gate

Advance at most ONE new candidate to the full final comparison.

Select based on:

1. practical quality;
2. size;
3. decode;
4. memory.

Do not select by average score alone.

If no new candidate clearly improves the Pareto frontier, advance none.

Possible valid outcome:

    no new quant beats g64/g128 tradeoff

That is a useful result.

Commit.

Suggested commit:

    benchmark: run Qwen3.8 quant mini tournament

---

# Phase 6 — Full Finalist Quality Comparison

Target: 90-120 minutes.

Compare:

    g64
    g128
    best new candidate if one advanced

Run full frozen evaluation_v2.

Do not change expected answers after seeing one model's output unless the test
is objectively broken.

If a broken test is discovered:

- document the correction;
- rerun that case for ALL models.

## Metrics

Report separately:

### Objective

    total objective pass rate
    per-category pass rate
    JSON/schema validity
    arithmetic/retrieval correctness
    instruction compliance

### Teacher-forced

    mean NLL
    median NLL delta vs g64
    per-category NLL
    top-1 agreement

### Behavioral

    early EOS
    repetition
    malformed structure
    severe hallucination/contradiction
    first divergence token where useful

### Subjective supplementary

For selected prose/coding answers create blind A/B/C labels.

Judge under a fixed rubric.

Do not allow the judge to know model identity.

Treat this as supplementary, not ground truth.

## Final quant classification

Choose one of:

### DEFAULT

Best overall practical model.

### QUALITY MODE

A larger/slower model with meaningfully better quality.

### PERFORMANCE/LONG-CONTEXT MODE

A smaller/faster model with an accepted quality tradeoff.

### REJECTED

No useful practical role.

It is acceptable to end with:

    g64 = quality mode
    g128 = performance mode

It is also acceptable for a new sensitivity-guided model to replace one or
both.

Commit.

Suggested commit:

    benchmark: finalize Qwen3.8 quant quality tradeoffs

---

# Phase 7 — Persistent Runtime Validation of Winner(s)

Target: 45-60 minutes.

Do NOT repeat all Stage 3 experiments.

For the model selected as DEFAULT:

Run:

    ~8K persistent, 5 turns
    ~16K persistent, 5 turns

Use exact-suffix runtime.

Measure:

    cold TTFT
    warm TTFT
    decode
    peak
    active+cache
    output correctness
    replay flag

If Stage 6 establishes a separate QUALITY MODE and PERFORMANCE MODE, one shorter
validation for the secondary mode is allowed.

## Regression requirements

Must pass:

    no stale replay
    correct transcript token accounting
    short follow-ups remain near-second-scale
    no model-config/quant loading bug
    no deterministic collapse

If the new quant has materially different memory from g128/g64, record updated
practical context guidance.

Commit.

Suggested commit:

    benchmark: validate Stage 4 Qwen3.8 runtime winner

---

# Phase 8 — Productionize qwen38-chat

Target: 60-90 minutes.

The runtime is now allowed to become a normal tool.

Create or clean up a user-facing runtime, e.g.:

    qwen38-chat

or:

    research/runtime/qwen38_chat.py

Prefer a location consistent with the repository structure.

## Required features

### Model

    --model PATH

Provide a documented default only if it is stable and sensible.

### Chat

Canonical tokenizer chat template.

Maintain complete logical conversation.

Use validated exact-suffix persistent cache.

### Commands

At minimum:

    /reset
    /quit

Optional useful commands:

    /stats
    /context

### Generation

Support:

    deterministic/greedy mode

and:

    practical sampling mode

Do not silently enable thinking if the intended default is non-thinking chat.

### Memory/cache

Optional:

    --kv-bits 8

if supported by the existing runtime.

Print compact stats such as:

    logical context tokens
    processed suffix tokens
    TTFT
    decode tok/s
    peak memory

### Safety

Never call mx.set_wired_limit().

Do not change system wired settings.

Handle:

    EOS
    Ctrl-C
    reset
    basic generation exceptions

without corrupting the next turn.

### Regression tests

Add automated tests where practical for:

1. no new BOS on later turns;
2. exact suffix calculation;
3. cached logical prefix accounting;
4. stale-replay regression with a lightweight/mock/token-level harness;
5. reset clears logical transcript/cache.

Tests should not require loading the full 27B model unless explicitly marked as
integration tests.

## Optional single-user server

Only if the CLI/runtime is complete and stable:

A single-request-at-a-time OpenAI-compatible endpoint may be added.

Do NOT implement:

    concurrent batching
    shared prefix cache across clients
    rotating multi-user cache
    cache trimming across unrelated prompts

Keep it single-user/single-conversation or serialized.

Commit.

Suggested commit:

    feat: productionize Qwen3.8 persistent chat runtime

---

# Phase 9 — Optional Learned-Quant Feasibility

Target: remaining time only.

Do this only if the primary Stage 4 goals are complete.

Investigate whether a Stage 5 experiment with:

    Dynamic Quantization
    DWQ
    AWQ
    GPTQ

is practical on this machine.

Do not automatically run all methods.

## Dynamic Quant

If Stage 3 already adapted equivalent sensitivity logic successfully, compare
it against upstream's current Dynamic Quant algorithm.

Determine:

    whether our sensitivity map is equivalent in spirit
    whether upstream target-BPW selection offers value
    whether bounded conversion can reproduce it

## DWQ

Perform a feasibility analysis.

Questions:

- can the teacher be an 8-bit or otherwise memory-safe model?
- can training/distillation be done layer-wise/bounded-memory?
- what is the minimum sample/sequence setup worth testing?
- what memory does a tiny dry-run use?
- can g128 or a dynamic 3/4 candidate be refined without full-model teacher
  residency?

A tiny bounded dry-run is allowed if safe.

Do not launch hours of unsafe training simply because Stage 4 still has time.

## AWQ/GPTQ

Static feasibility plus at most one tiny calibration smoke if easy.

The deliverable is a ranked Stage 5 recommendation, not a complete study.

---

# Explicitly Out of Scope

Do not spend Stage 4 on:

    uniform 2-bit
    mixed_2_6
    generic uniform 4-bit
    MXFP4/NVFP4 zoo
    24K/32K context chasing
    DSpark
    MTP
    external draft models
    multi-user batching
    system wired-memory tuning

Optional n-gram self-speculative decode is lower priority than quant quality and
runtime productionization.

Only touch it if every primary deliverable is complete and substantial time
remains.

---

# Machine-Readable Results

Append meaningful runs to:

    research/stage4/results.jsonl

Suggested fields:

    timestamp
    experiment_id
    phase
    git_commit

    model_name
    model_path
    quant_recipe
    quant_bits_base
    quant_group_size
    promoted_modules_count
    model_payload_bytes
    effective_bpw

    calibration_hash
    evaluation_hash
    case_id
    category

    prompt_tokens
    generated_tokens
    finish_reason
    output_hash

    objective_pass
    structured_valid
    replay_detected
    early_eos

    nll
    nll_delta_vs_g64
    top1_agreement

    ttft_seconds
    decode_tps
    prompt_tps

    active_memory_bytes
    cache_memory_bytes
    active_plus_cache_bytes
    peak_memory_bytes
    rss_bytes

    classification
    exception
    safety_stop_reason

Use null when unavailable.

Never invent values.

---

# Research Memory

After every major phase update:

    research/stage4/STATUS.md

Keep:

    current best model
    current best quant tradeoff
    frozen suite hashes
    candidate model paths
    model sizes
    rejected candidates
    current runtime status
    next action

Keep human-readable synthesis in:

    research/stage4/RESULTS.md

---

# Git

Prefer logical commits:

    benchmark: build Qwen3.8 quality suite v2
    experiment: diagnose Qwen3.8 endurance early EOS
    experiment: map Qwen3.8 quantization sensitivity
    experiment: build quality-guided Qwen3.8 quants
    benchmark: run Qwen3.8 quant mini tournament
    benchmark: finalize Qwen3.8 quant quality tradeoffs
    benchmark: validate Stage 4 Qwen3.8 runtime winner
    feat: productionize Qwen3.8 persistent chat runtime
    research: assess learned quantization feasibility

Do not commit model weights.

---

# Final Report

Create:

    research/stage4/FINAL_REPORT.md

It must clearly distinguish:

    MEASURED FACT
    INFERENCE
    FUTURE HYPOTHESIS

Answer:

1. What defects in Stage 3 quality evaluation were corrected?

2. What is the frozen Quality Suite V2 composition and hash?

3. What did the turn-17 early-EOS experiment show?

4. Is g128 materially worse than g64 on corrected objective quality?

5. What are g64 vs g128 teacher-forced differences?

6. What did sensitivity analysis reveal?
   Which module families/layers are most sensitive?

7. What quant candidates were built?

8. What are their:
       payload sizes
       effective BPW
       decode speeds
       memory footprints

9. Which candidates were rejected at smoke/mini-tournament stage, and why?

10. What are the final full-suite results for:
        g64
        g128
        best new candidate if any

11. What model should be:
        DEFAULT
        QUALITY MODE
        PERFORMANCE/LONG-CONTEXT MODE

12. Did the final winner pass 8K and 16K persistent validation?

13. What are the recommended 8K and 16K runtime commands?

14. What is the current role of KV8?

15. Is qwen38-chat productionized?
    What exact command should the user run?

16. Is a single-user OpenAI-compatible endpoint available?

17. Which learned quantization methods are worth a Stage 5?

18. What are the three highest-value next experiments?

Include a final Pareto table:

    model
    size GiB
    effective BPW
    quality verdict
    8K peak
    16K peak
    decode
    warm TTFT
    recommended role

Do not claim untested configurations.

---

# Completion Condition

Stage 4 is complete when:

- Quality Suite V2 is frozen and validated;
- g64 vs g128 quality is materially clarified;
- sensitivity-guided quantization has been investigated;
- no more than three new serious quant candidates were built;
- the best candidate(s) have a final quality classification;
- the selected default model has been validated in persistent runtime;
- qwen38-chat is usable/documented;
- FINAL_REPORT.md and STATUS.md are complete;
- useful code/results are committed.

Do not start Stage 5 automatically.

At the end return a concise summary containing:

    default model
    quality mode
    performance mode
    model sizes
    quality result
    8K/16K runtime numbers
    early-EOS verdict
    qwen38-chat command
    learned-quant recommendation
    FINAL_REPORT.md path
    commit SHAs

Then STOP.
