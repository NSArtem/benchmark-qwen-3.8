# Evening Task — Maximize Practical Qwen3.8-27B MLX Context on 18 GB M3 Pro

## Mission

Continue the Qwen3.8 investigation autonomously for an extended unattended run.

Primary question:

> How far can we improve practical, correctness-safe Qwen3.8-27B context on
> this 18 GB M3 Pro using MLX 0.32.1?

The immediate baseline is the existing affine 3-bit / group-size-64 model.

The long-term target is an interactive coding-agent runtime with substantially
more usable context than the current near-memory-limit ~16K result.

This task is deliberately multi-phase.

DO NOT blindly execute every phase.

Use the decision gates below.

Re-read AGENTS.md before starting.

---

# Phase 0 — Recover State and Build Research Ledger

Start with:

    git status
    git log --oneline -15

Inspect existing Qwen3.8 experiment directories and reports.

Specifically look for recent results covering:

- MLX 0.32.1 OOM localization;
- 512/513, 640/641, 768/769 alignment experiment;
- context scaling;
- the Lantern coding prompt;
- any currently-running benchmark started before this task.

Do not rerun an experiment merely because this TASK mentions it.

If an existing artifact already answers a question with sufficient raw evidence,
reuse it and cite the artifact in the research report.

Create/update:

    research/PLAN.md
    research/STATUS.md
    research/RESULTS.md
    research/results.jsonl

PLAN.md should translate this TASK into concrete commands and experiment IDs.

STATUS.md must contain:

    Current phase
    Completed experiments
    Key measurements
    Current best configuration
    Failed/rejected configurations
    Current hypotheses
    Next action
    Safety notes

Inspect processes before launching a heavy test.

Do not overlap large Qwen MLX workloads.

Commit the research scaffolding if appropriate.

---

# Phase 1 — Consolidate Current Baseline

The baseline configuration is:

    Qwen3.8-27B
    current 3-bit affine group-64 checkpoint
    MLX 0.32.1
    mlx-lm pinned revision
    default wired state / NO forced wired limit

Before running anything expensive, consolidate all existing baseline evidence
into a single table.

Desired context points:

    ~1K
    ~2K
    ~4K
    ~8K
    ~12K
    ~16K

For each point, collect where available:

    actual prompt tokens
    classification
    TTFT
    prompt tok/s
    decode tok/s
    generated-token count
    active memory
    cache memory
    active + cache
    MLX peak
    OS-level memory
    exception
    system-instability observation

IMPORTANT:

A manual real-world run already demonstrated:

    16767 tokens
    63.348 prompt tok/s
    4.118 generation tok/s
    256 generated tokens
    17.200 GB MLX peak
    coherent output
    no Metal OOM

Do NOT repeat this ~16.8K run solely to duplicate evidence.

If a context-scale experiment already ran while this task was being prepared,
inspect its artifacts first.

Only fill missing baseline points when useful.

For new cold tests:

    direct generate_step
    default wired state
    greedy
    prefill_step_size = 2048
    fresh process
    max_tokens = 128

Do not use the stock forced wired-limit wrapper.

### Phase 1 decision

At the end report:

    largest coherent tested context
    largest memory-safe tested context
    empirical memory-growth curve
    TTFT curve
    decode curve

Then continue to Phase 2.

---

# Phase 2 — Build a Canonical No-Wired Runtime Helper

We currently use ad-hoc monkey patches to disable mlx-lm's wired-limit context.

Create a SMALL reusable local runtime helper so later experiments cannot
accidentally return to the problematic stock behavior.

Goals:

1. preserve as much stock mlx-lm behavior as possible;
2. override ONLY the wired-limit policy;
3. make the override obvious in logs;
4. keep the pinned environment unchanged.

Useful entry points may include:

    generate_no_wired
    chat_no_wired

or equivalent small Python wrappers.

Do not monkey-patch MLX itself.

Do not call mx.set_wired_limit().

For generate, support:

    model
    prompt / stdin
    max tokens
    temperature
    normal chat template
    optional KV-cache arguments if the pinned API supports them

For chat, preserve persistent prompt_cache behavior.

Add a startup line similar to:

    [qwen38-runtime] mlx-lm wired_limit wrapper disabled; MLX wired state untouched

Validate with a SHORT prompt first.

Then validate with a moderate realistic prompt, not ~16K.

Record the exact command users should run.

Do not overengineer a package.

Commit the helper separately if clean.

---

# Phase 3 — Measure Warm / Interactive Performance

This phase matters greatly.

The user observed that interactive MLX chat felt much faster than cold
large-context generation.

Test whether persistent prompt-cache use provides a practically good
interactive experience.

Do NOT start with 16K.

Use a realistic coding-agent conversation.

Suggested initial context:

    ~4K

If safe and informative, also:

    ~8K

Only use ~12K if memory measurements justify it.

Do not use ~16K for this phase unless there is an exceptional reason.

Test a scripted multi-turn conversation:

    Turn 1:
        substantial repository context / coding task

    Turn 2:
        short follow-up referring to previous answer/context

    Turn 3:
        another realistic short follow-up

    Turn 4:
        another realistic follow-up if still useful

The prompt cache must persist between turns.

Use the no-wired runtime.

Measure separately:

    initial cold TTFT
    initial prefill tok/s
    initial decode tok/s

    turn-2 incremental TTFT
    turn-2 newly processed tokens if measurable
    turn-2 decode tok/s

    turn-3 incremental TTFT
    turn-3 decode tok/s

    memory growth after each turn

Save outputs.

Determine whether the previously perceived ~high interactive generation speed
is reproducible.

Do not assume it is.

### Phase 3 decision

Classify interactive behavior as one of:

    EXCELLENT
    USEFUL
    MARGINAL
    UNUSABLE

based on measured latency, decode speed, correctness and memory.

Record this before moving on.

---

# Phase 4 — Exact Quantization Size Study

Do NOT convert a model yet.

First determine precisely what memory we can plausibly save.

Inspect the current checkpoint's architecture, tensor shapes and quantization
configuration.

Build a small analysis tool if needed.

Calculate as accurately as practical the expected stored/model parameter size
for:

A. current baseline

    affine
    3-bit
    group size 64

B. candidate 1

    affine
    3-bit
    group size 128

C. uniform lower precision control

    affine
    2-bit
    group size 128

D. built-in mixed candidate

    mixed_2_6
    group size 64

E. built-in mixed candidate

    mixed_2_6
    group size 128

F. optionally one custom mixed candidate if clearly justified

Account for:

- packed quantized weights;
- scales;
- affine biases/zero-point equivalent metadata;
- non-quantized tensors;
- tensors whose dimensions prevent a proposed group size;
- lm_head / embedding behavior;
- model-specific quantization predicate behavior.

Use actual model tensor shapes rather than simply multiplying 27B by a nominal
bits-per-weight number.

Report:

    effective bits per weight
    expected bytes
    expected GiB
    savings vs current
    percentage saved

Also estimate first-order context headroom implied by the Phase-1 empirical
memory-growth curve.

Do not claim quality based on size.

### Candidate ranking

Rank candidates by:

    expected memory saving
    likely quality risk
    conversion complexity
    runtime support risk

Candidate 1 should normally remain 3-bit g128 unless measurements reveal a
reason not to use it.

Continue to conversion feasibility only after this analysis.

---

# Phase 5 — Investigate Conversion Feasibility BEFORE Downloading

We may need original/high-precision weights to make a valid new quant.

Identify the exact source/base model of the current checkpoint using:

- local config;
- local README/model metadata if available;
- model card metadata if network access is available.

Do NOT guess.

Determine whether suitable source weights are already local.

Record their location if present.

If not local, determine expected download size.

Check:

    df -h
    available local disk
    existing HF cache size

Conservative project rule:

Do not begin a ~50+ GB source-model download unless there is at least roughly
80 GiB of genuinely free disk available after accounting for the intended
output and temporary files.

This is a project safety margin, not an MLX guarantee.

Do not delete unrelated caches to manufacture free space.

## Inspect conversion paths

Inspect the pinned mlx-lm conversion implementation.

Also inspect current upstream mlx-lm conversion/chunking work if network access
is available, but DO NOT upgrade the canonical inference environment.

Determine whether:

1. stock pinned convert is realistically safe on 18 GB;
2. an upstream streaming/chunked conversion implementation can be adapted;
3. a small local streaming converter is necessary.

Do not simply launch stock full conversion to discover its memory behavior by
OOM.

Prefer bounded-memory conversion.

Important:

A valid g128 candidate should be produced from appropriate high-precision
source weights.

Do not re-quantize the existing 3-bit checkpoint and silently present that as
equivalent.

### Phase 5 decision

Choose one:

    CONVERSION_SAFE
    STREAMING_CONVERSION_REQUIRED
    BLOCKED_BY_SOURCE
    BLOCKED_BY_DISK
    BLOCKED_BY_RUNTIME

Record evidence.

If blocked, skip to Phase 8 and do useful KV-cache work instead.

---

# Phase 6 — Produce 3-bit Affine g128 Candidate

Only enter this phase if Phase 5 establishes a safe path.

Primary candidate:

    affine
    bits = 3
    group_size = 128

Do not change unrelated quantization parameters.

Prefer original/high-precision source weights.

Use a bounded-memory conversion method if standard conversion has unacceptable
peak-memory risk.

Monitor:

    active
    cache
    peak
    OS memory pressure
    elapsed time

Do not use system wired-memory changes.

If conversion OOMs once:

    record it
    do not repeatedly retry the same implementation

Fix the memory strategy first.

After conversion verify:

    config
    quantization metadata
    tensor count
    output shard integrity
    actual disk size
    effective bpw if computable

Do not upload the model anywhere.

Do not commit model weights.

---

# Phase 7 — Validate g128 Correctness Before Long Context

Do NOT immediately run 16K.

Use fresh processes.

Test at minimum:

    short control
    ~768 tokens
    ~4K realistic coding prompt

Generate:

    256 tokens

Use:

    MLX 0.32.1
    no-wired runtime
    greedy

Classify each result.

Compare with g64 baseline.

If g128 is BROKEN or clearly much worse:

    repeat one representative test once
    if failure reproduces, reject candidate
    do not proceed to long-context scaling

If coherent:

    measure model memory after load
    compare with exact predicted saving
    continue.

---

# Phase 8 — Benchmark g128 Context Envelope

Only if Phase 7 passes.

Do not blindly run every size.

Start at:

    ~8K

Then:

    ~12K
    ~16K

Use the same realistic prompt family/baseline methodology where possible.

Measure:

    TTFT
    prompt tok/s
    decode tok/s
    active + cache
    peak
    OS memory
    correctness

If 16K is coherent and materially safer than the existing g64 baseline,
consider larger context.

Suggested adaptive ladder:

    20K
    24K

Do NOT automatically run 32K.

A single 32K probe is allowed only if:

- 24K is coherent;
- macOS remains stable;
- measured memory still shows meaningful safety headroom;
- there has been no Metal allocation failure;
- the experiment is still providing useful information.

Do not chase a number at the cost of system stability.

Stop at the first clearly unsafe point.

Record:

    maximum coherent context
    recommended practical context

These are different concepts.

---

# Phase 9 — KV Cache Quantization Study

KV quantization is an independent lever from weight quantization.

Inspect the exact pinned mlx-lm API and Qwen3.8 cache implementation before
testing.

Do not guess flags or cache semantics.

Determine which layers have context-growing KV state and which use recurrent
state.

Measure actual cache objects/shapes if useful.

Test KV8 first.

Preferred initial configuration, if supported by pinned API:

    kv_bits = 8
    kv_group_size = 64

Determine the appropriate quantized_kv_start from the actual API instead of
blindly choosing one.

First test with the ORIGINAL g64 model so the effect can be isolated.

Use:

    ~4K or ~8K

Compare:

    correctness
    TTFT
    decode speed
    active + cache
    peak

If KV8 is coherent and produces meaningful savings:

    test ~12K
    then ~16K if safe

If g128 exists and passed:

    test the combination:
        g128 weights + KV8

Do not perform a huge matrix.

The purpose is to answer:

> Does KV8 buy enough memory at long context to be worth using?

If yes, use only the most promising configuration for larger-context probing.

---

# Phase 10 — Optional Second Weight Quant Candidate

Only enter if ALL are true:

- g128 work is complete or clearly blocked;
- conversion infrastructure is safe;
- source weights are available;
- expected savings from a second candidate are materially larger;
- enough useful work can still be completed without compromising safety.

Maximum ONE additional weight quant candidate tonight.

Preferred candidate should be selected from Phase 4 evidence, probably:

    mixed_2_6

or a carefully justified custom mixed precision policy.

Do not create several quantized 27B copies merely because the command exists.

Do not run DWQ tonight unless the entire primary investigation is complete and
there is an exceptional evidence-based reason.

Treat DWQ as a likely follow-up.

For a second candidate:

    short correctness
    768
    4K

before any long-context benchmark.

Reject quickly if quality is poor.

---

# Phase 11 — Practical Runtime Comparison

At the end compare only configurations actually measured.

Likely rows:

    g64 baseline
    g64 + KV8
    g128
    g128 + KV8
    optional mixed candidate

For each report:

    model size
    context tested
    recommended context
    cold TTFT
    cold prefill tok/s
    decode tok/s
    warm-turn TTFT
    warm decode tok/s
    memory
    correctness
    system stability

Do not fill missing cells with guesses.

A configuration should only be recommended if it is coherent and operationally
stable.

---

# Phase 12 — Produce User-Facing Runtime Commands

Create exact reproducible commands/scripts for the winning configurations.

At minimum:

1. one-shot generation;
2. interactive chat;
3. optional long-context/KV8 variant if it wins.

The no-wired behavior must be explicit.

Do not require modifying installed mlx-lm files.

Do not require changing sysctl.

Prefer small wrappers committed in the repo.

Include a command to print:

    MLX version
    model path
    quant config
    wired-policy override status

so accidental environment drift is visible.

---

# Result Files

Maintain throughout:

    research/results.jsonl

Each experiment record should contain where applicable:

    timestamp
    experiment_id
    git_commit
    python_version
    mlx_version
    mlx_lm_revision

    model_path
    model_quantization

    prompt_path
    prompt_sha256
    prompt_tokens

    generated_tokens
    generated_token_hash
    classification

    prefill_step_size

    ttft_seconds
    prompt_tps
    decode_tps
    generation_seconds
    total_seconds

    active_memory
    cache_memory
    active_plus_cache
    peak_memory

    os_memory_metrics

    kv_bits
    kv_group_size

    exception
    safety_stop_reason

Never invent unavailable measurements.

---

# STATUS Discipline

After EVERY meaningful experiment update:

    research/STATUS.md

Before every new phase re-read:

    AGENTS.md
    TASK.md
    research/STATUS.md

This is mandatory for the unattended run.

If context becomes large or the investigation changes direction, STATUS.md is
the canonical compressed state.

---

# Commit Discipline

Prefer commits after logically complete milestones, for example:

    benchmark: consolidate Qwen3.8 MLX baseline
    tooling: add no-wired MLX runtime wrappers
    benchmark: measure Qwen3.8 warm cache behavior
    analysis: model Qwen3.8 quantization sizes
    tooling: add bounded-memory MLX conversion
    benchmark: evaluate Qwen3.8 3bit g128
    benchmark: evaluate Qwen3.8 KV8 cache
    docs: summarize Qwen3.8 18GB runtime findings

Do not commit huge model files.

Do not rewrite unrelated history.

---

# Blocked-Branch Policy

This is an unattended task.

Do not stop simply because one phase is blocked.

Examples:

If source weights cannot be downloaded:
    -> do KV8 analysis and interactive-cache benchmarks.

If conversion cannot fit:
    -> investigate/implement bounded-memory conversion.

If inference worker is currently busy:
    -> do static quant-size and source analysis.

If g128 quality fails:
    -> preserve the failure and investigate KV8 on g64.

If KV8 quality fails:
    -> preserve the failure and continue weight-quant work.

Do not circumvent a safety rule to make progress.

---

# Hard Stop Rules

Never:

- change iogpu.wired_limit_mb;
- increase system wired memory;
- test MLX 0.32.0;
- delete unrelated caches;
- run two huge MLX processes concurrently;
- repeatedly reproduce an OOM;
- repeatedly approach known dangerous memory pressure with no new information;
- upload model weights;
- silently upgrade the canonical environment.

If a risky branch fails, preserve evidence and move on.

---

# Final Report

Create:

    research/FINAL_REPORT.md

The report must answer:

1. What baseline results were established?

2. What exactly caused the known MLX 0.32.1 OOM?

3. How well does the canonical no-wired runtime work?

4. What is the measured cold context envelope for g64?

5. What is the measured warm/interactive performance?

6. How much memory does 3-bit g128 theoretically and actually save?

7. Was g128 successfully converted?

8. Is g128 coherent?

9. What context does g128 reach?

10. What does KV8 save?

11. Does KV8 affect correctness or speed?

12. What is the best tested configuration?

13. What is:
       maximum coherent context?
       recommended practical context?

14. What exact command should the user run for normal interactive use?

15. What unresolved bugs remain?

16. Which hypotheses were rejected?

17. What should the NEXT experiment be?

Include a compact comparison table.

Also include a section:

    "Things deliberately not done"

so that skipped unsafe/low-value work is explicit.

The final conclusion must distinguish:

    measured fact
    inference
    future hypothesis

Do not claim a configuration was tested when it was only estimated.

---

# Completion

At the end:

    git status
    git log --oneline -15

Verify research artifacts exist.

Commit any remaining appropriate small artifacts.

Then return a concise final message containing:

- best configuration;
- recommended context;
- warm interactive performance;
- memory result;
- quantization result;
- any safety limits encountered;
- FINAL_REPORT.md location;
- commit SHAs.

Do not begin another open-ended investigation after FINAL_REPORT.md is written.