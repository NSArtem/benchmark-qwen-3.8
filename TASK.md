# Stage 3 — Persistent Qwen3.8 Runtime Validation

## Mission

Run a focused 6-8 hour research stage that determines whether the current
Qwen3.8-27B MLX work can be promoted from "promising experiment" to
"practical interactive runtime".

Primary goals:

1. close persistent-cache correctness;
2. scale correct exact-suffix chat from 4K to 8K/12K/16K;
3. test long-session endurance;
4. measure g64 vs g128 quality across a diverse suite;
5. evaluate KV8 only as a long-context safety lever;
6. if justified, probe 20K practical persistent use;
7. only if the primary work finishes early, investigate one decode-speed
   optimization with no extra model-memory requirement.

This is a gated research program.

Do NOT blindly execute every phase.

Use the decision gates.

---

# Phase 0 — Recover State

Target: 15-30 minutes.

Start with:

    git status
    git log --oneline -20

Inspect:

    research/
    research/night2/
    qwen38-persistent-suffix-4k/
    previous g128 conversion artifacts
    previous g64/g128 benchmark outputs
    previous KV8 results

Do not rerun completed experiments before inspecting them.

Confirm both model paths:

    g64 reference
    g128 candidate

Confirm g128 exists at:

    /Users/artem/models/Qwen3.8-27B-3bit-g128-MLX

Confirm canonical environment:

    envs/mlx-0.32.1

Inspect whether another heavy MLX process is active.

Create/update:

    research/stage3/PLAN.md
    research/stage3/STATUS.md
    research/stage3/RESULTS.md
    research/stage3/results.jsonl

In STATUS.md record the exact facts recovered from prior artifacts.

Do not paraphrase old results from memory when raw files exist.

### Phase 0 deliverable

A compact baseline table containing at least:

    g64 model size
    g128 model size
    known 4K/8K/12K/16K-ish memory
    known exact-suffix 4K TTFT
    known fresh 4K oracle TTFT
    known KV8 16K-ish tradeoff
    current unresolved questions

Then proceed.

---

# Phase 1 — Same-Boundary Cache Correctness Oracle

Target: 60-90 minutes.

This is the highest-priority phase.

The current unresolved observation is:

- exact-suffix persistent chat at ~4K is coherent and fast;
- fresh full-transcript chat is coherent but much slower;
- their generated outputs diverge after ~42 tokens;
- their cache-state signatures also diverge.

We need to determine whether this is merely segmentation sensitivity or a true
persistent-state error.

## Required paths

Implement a diagnostic that compares THREE paths over the SAME intended
canonical conversation.

### Path A — Fresh one-shot

Start from empty cache.

Build the complete canonical turn-2 transcript as one token stream.

Process it as a normal cold full-transcript generation.

This is the semantic correctness oracle.

### Path B — Fresh segmented

Start from a NEW empty cache.

Feed the token stream with the SAME logical/model-call boundaries that occurred
in the persistent conversation.

Conceptually:

    initial canonical user prompt tokens
    -> assistant generation tokens one by one
    -> exact canonical turn-2 suffix
    -> turn-2 generation tokens

Important:

Path B must reproduce Path C's boundaries as exactly as the harness/API allows.

Do not use the state produced by Path C.

This path tests segmentation/batching sensitivity.

### Path C — True persistent

Run the real conversation:

    initial prompt
    -> assistant generation
    -> persistent cache remains alive
    -> build complete canonical turn-2 transcript
    -> calculate exact missing suffix from the logical cached prefix
    -> feed suffix only
    -> generate turn 2

Use the already-working exact-suffix protocol.

## Token-accounting rule

Do not assume which generated token is or is not already represented by cache.

Instrument the actual path.

Record:

    canonical logical tokens
    cache offsets where meaningful
    generated token sequence
    exact prefix length represented by cache
    exact suffix token IDs

If a final predicted token exists but has not yet been evaluated into cache,
account for it explicitly.

The equation must be true at the logical protocol level:

    cached_prefix_tokens + supplied_suffix
        == canonical intended prefix before next assistant generation

## Compare at boundaries

At useful boundaries compare Path B vs Path C.

At minimum:

    after initial prompt prefill
    after turn-1 generation
    after turn-2 suffix prefill
    after first turn-2 generated token
    after final turn-2 generated token

For each layer record compact state signatures:

    layer index
    cache class
    offset if available
    state shape(s)
    deterministic hash/signature

Do not serialize huge arrays.

Where practical, also compare logits at the next-token boundary:

    max absolute logit diff
    mean absolute logit diff
    top-1 token
    top-k overlap

## Decision gate

### CASE 1

If:

    Path B ~= Path C

at state/logit/output level, while A may differ,

then conclude:

    persistent exact-suffix state is consistent with identical segmented
    execution;

    A-vs-B/C divergence is segmentation/numerical sensitivity, not evidence of
    persistent-cache corruption.

Mark persistent cache VALIDATED FOR SCALING.

Do not keep digging for bitwise one-shot equivalence.

Proceed to Phase 3.

### CASE 2

If:

    Path B != Path C

materially under identical token stream and identical segmentation boundaries,

then there is evidence of a persistent-state-specific problem.

Proceed to Phase 2.

### CASE 3

If the harness cannot actually make B and C segmentation-equivalent,

document the exact limitation.

Do not falsely claim validation.

Perform the best narrower control available, then proceed to Phase 2 only if
there is actual suspicious evidence.

## Artifact directory

Create something like:

    research/stage3/cache-boundary-oracle/

Save:

    harness/
    tokens/
    state_signatures/
    outputs/
    results.json
    results.md

Commit this phase.

Suggested commit:

    experiment: validate segmented Qwen3.8 cache state

---

# Phase 2 — Root-Cause Persistent Divergence

Target: up to 90 minutes.

ONLY run this phase if Phase 1 finds material B-vs-C divergence.

Do not enter it merely because A differs from B/C.

Goal:

    identify the FIRST layer / state component / model-call boundary where
    persistent state diverges from fresh segmented state.

## Localization

Find the earliest boundary where state differs.

Then identify:

    layer index
    layer type
    cache type

For ArraysCache/recurrent layers distinguish state slots where possible:

    convolution state
    recurrent/delta state
    metadata/offset-like state

For attention layers inspect:

    KV shapes
    offsets
    token count represented

Use small deterministic probes.

Do not dump full tensors.

## Controlled hypotheses

Test only evidence-driven possibilities such as:

- one generated token is logically counted but not evaluated into cache;
- end-of-turn token handling is wrong;
- tokenizer transcript contains a boundary token missing from persistent path;
- ArraysCache state is mutated differently across generator teardown/reentry;
- mx.clear_cache / evaluation timing changes lazy state materialization;
- persistent path reuses a cache object differently from a fresh segmented path.

Do NOT jump immediately to upstream PR patches.

## Upstream inspection

If the divergence points at ArraysCache, inspect current upstream source and
current status of relevant issues/PRs.

Useful references:

    issue #980
    PR #1253 / #1254
    issue #1641 / PR #1642

Do not assume a closed PR is the right fix.

## Patch rule

If a minimal local fix is strongly supported by the localized evidence:

- patch only the experiment/helper copy if practical;
- keep canonical installed environment unchanged;
- rerun Phase 1 oracle.

If the fix requires invasive upstream surgery:

- document a minimal reproduction;
- do not spend the entire stage implementing speculative redesign;
- preserve evidence;
- mark persistent cache NOT YET VALIDATED;
- skip unsafe scaling phases and continue with Phase 5 quality work.

### Phase 2 success gate

Only promote persistent cache to validated if a post-fix Phase 1 oracle passes.

Commit any useful reproduction/harness/fix separately.

---

# Phase 3 — Persistent Exact-Suffix Scaling

Target: 60-90 minutes.

ONLY run if Phase 1/2 validates the persistent exact-suffix protocol.

Primary model:

    g128

Do not use KV8 yet.

Use one realistic coding-agent prompt family.

Test:

    ~4K   existing result may be reused
    ~8K
    ~12K
    ~16K

Do not require exact powers of two if canonical chat-template token counts
naturally differ.

For each context use a real multi-turn conversation, not a single follow-up.

Suggested shape:

    initial substantial coding/repository prompt

    assistant turn 1:
        128-256 tokens

    user turn 2:
        short question asking to prioritize one invariant

    assistant turn 2:
        64-128 tokens

    user turn 3:
        ask to modify/correct one part of the previous answer

    assistant turn 3:
        64-128 tokens

    user turn 4:
        ask a question that references information from turn 1

    assistant turn 4:
        64-128 tokens

    user turn 5:
        ask for one concise implementation recommendation

    assistant turn 5:
        64-128 tokens

The follow-ups must make stale replay or lost conversation state obvious.

## Metrics per turn

Record:

    logical transcript tokens
    exact incremental suffix tokens
    TTFT
    decode tok/s
    generated tokens
    active memory
    cache memory
    active + cache
    peak memory
    output classification
    replay detected?
    transcript consistency?

Measure initial cold prefill separately from warm turns.

## Practical expectations

Do not hardcode pass/fail solely to these values, but the desired behavior is
approximately:

    4K follow-up  ~1 s
    8K follow-up  ~1 s
    12K follow-up near-second-scale
    16K follow-up ideally <= a few seconds

If follow-up TTFT grows approximately with TOTAL conversation length, investigate
why; that suggests hidden reprefill.

## Correctness oracle

For at least ONE follow-up at 8K and ONE at the largest successful context,
also run a fresh full-transcript semantic oracle.

The persistent and fresh outputs need not be token-identical.

They must:

- answer the same current question;
- use the same conversation facts;
- avoid stale replay/corruption.

## Stop conditions

Stop upward scaling if:

- Metal OOM;
- severe system instability;
- persistent turn is BROKEN twice at the same context;
- clear hidden full reprefill makes persistence ineffective;
- memory enters a clearly unsafe zone with no useful headroom.

Do NOT automatically go beyond ~16K in this phase.

## Deliverable

A table:

    context
    cold TTFT
    warm TTFT median
    warm TTFT max
    decode t/s
    memory
    correctness
    verdict

Classify each context:

    PRACTICAL
    USABLE
    MARGINAL
    UNSAFE
    BROKEN

Commit.

Suggested commit:

    benchmark: scale exact-suffix Qwen3.8 chat

---

# Phase 4 — Long-Session Endurance

Target: 45-75 minutes.

ONLY run if persistent exact-suffix chat is validated and 8K works.

Use g128, normal cache.

Start with approximately:

    8K initial context

Run:

    20-30 sequential user/assistant turns

Use realistic but bounded turns:

    user input ~50-150 tokens
    assistant output ~64-128 tokens

The conversation must exercise references to older context.

Examples:

- ask to remember a constraint from the initial prompt;
- revise an earlier decision;
- ask about an earlier file/component;
- ask to compare current recommendation to turn 2;
- ask for a concise final patch summary.

Do not make every turn independent.

## Measure

Per turn:

    total logical context
    suffix tokens
    TTFT
    decode t/s
    generated tokens
    active/cache/peak
    cache offsets/signatures
    replay/corruption flag

Also periodically record:

    process RSS
    vm_stat / memory pressure if useful

## Look for trends

Analyze:

    TTFT vs turn number
    TTFT vs suffix size
    memory vs logical context
    decode speed vs turn number
    state-signature anomalies
    output-quality degradation
    accidental full reprefill

Do not expect exact constant memory because attention KV grows with context.

We are looking for UNEXPECTED growth/drift.

## Upstream leak awareness

Current upstream has reported ArraysCache metadata/resource issues primarily in
BatchGenerator/server-style paths.

Our direct generate_step path may not share that mechanism.

Do not import the conclusion from upstream.

Measure our actual process.

## Gate

PASS if:

- no stale replay;
- no catastrophic corruption;
- TTFT remains governed primarily by suffix size, not full history;
- memory growth is explainable by context/cache;
- no severe unexplained leak;
- decode remains practically stable.

If endurance fails, preserve exact first failing turn and stop.

Commit.

Suggested commit:

    benchmark: test Qwen3.8 persistent-chat endurance

---

# Phase 5 — g64 vs g128 Quality Suite

Target: 90-120 minutes.

This phase is independent of persistent-cache success.

If persistent-cache work becomes blocked, proceed here.

Goal:

> Determine whether g128's ~0.78 GiB saving and speed advantage causes a
> meaningful practical quality loss vs g64.

Do not decide from one perplexity number.

## Build a diverse deterministic suite

Target approximately 30-50 cases.

Keep it small enough to complete on this machine.

Suggested categories:

### Coding / repository reasoning: ~10

- bug diagnosis
- race-condition analysis
- patch planning
- API invariant reasoning
- code review
- identify likely root cause from supplied snippets

### Instruction following: ~5

- strict formatting
- concise constraints
- ordered requirements
- exclusion constraints

### Structured output: ~5

- valid JSON
- simple schema adherence
- exact requested keys
- short table/record generation

### General reasoning: ~5

- deterministic logic/reasoning tasks
- avoid trivia dependent on web freshness

### Long-context retrieval / consistency: ~5

- retrieve details from a few-thousand-token supplied context
- compare two distant facts
- obey an early constraint near the end

### TTRPG / prose continuity: ~5

- maintain names/facts
- continue a scene under explicit constraints
- summarize state without inventing facts

Favor tasks with objective or semi-objective evaluation.

Do not build a huge benchmark framework.

## Generation comparison

For each case:

    same prompt
    fresh process where practical
    greedy decoding
    same max tokens
    g64
    g128

Record:

    outputs
    token IDs
    first divergence position
    repetition
    parse/schema validity
    objective pass/fail where available

## Teacher-forced metrics

On a representative subset compute where practical:

    mean NLL
    token-level top-1 agreement
    KL or logit similarity if inexpensive

Do not over-weight perplexity.

Report per-category distributions, not only one global mean.

## Blind pairwise assessment

For subjective outputs, optionally create anonymized A/B pairs.

The Codex agent may produce a blind rubric-based comparison as a SUPPLEMENTARY
signal.

Do not treat its own preference judgment as ground truth.

Prefer objective checks whenever possible.

## Decision categories

At the end classify g128 as:

### DEFAULT

No meaningful systematic quality loss found.

### PERFORMANCE/LONG-CONTEXT MODE

Some measurable quality loss, but often acceptable in exchange for memory/speed.

### EXPERIMENTAL ONLY

Large or systematic practical degradation.

The report must clearly explain which evidence drove the classification.

Commit.

Suggested commit:

    benchmark: compare Qwen3.8 g64 and g128 quality

---

# Phase 6 — KV8 as a Long-Context Lever

Target: 45-60 minutes.

ONLY run after persistent chat is validated.

Do not test KV8 at 4K simply to produce more rows.

Primary question:

> At 16K-class persistent context, does KV8 provide enough extra headroom to
> justify its decode penalty?

Use g128.

Compare:

    normal KV
    KV8

At:

    ~16K persistent conversation

Use the same conversation shape where possible.

Measure:

    cold peak
    warm-turn peak
    follow-up TTFT
    decode t/s
    correctness
    system responsiveness

Existing prior evidence suggests around:

    ~0.465 GiB saving
    ~10-11% decode penalty

Confirm or update this under REAL persistent chat.

## Decision

KV8 should become a selectable long-context mode only if the memory saving is
operationally meaningful.

Examples:

- makes 16K materially safer;
- prevents swap/system pressure;
- enables a useful larger context;
- reduces peak enough to create comfortable OS headroom.

If it merely saves memory that we do not need, reject it as default.

Commit if useful.

---

# Phase 7 — Optional 20K Practical Probe

Target: 30-45 minutes.

ONLY run if ALL are true:

- exact-suffix persistent cache is validated;
- 16K persistent is coherent;
- system remained responsive;
- memory measurements predict meaningful safety headroom;
- no unexplained endurance leak;
- this experiment still provides useful information.

First try:

    g128
    normal cache
    ~20K initial context
    3 short follow-up turns

If normal cache is too close to the safety limit but KV8 was validated and
clearly useful, ONE KV8 20K attempt is allowed.

Measure:

    cold TTFT
    warm TTFT
    decode
    active+cache
    peak
    correctness
    system stability

Do NOT automatically test:

    24K
    32K

If 20K is surprisingly comfortable, record that 24K is a future experiment.

Do not pursue it in this stage unless the entire primary program finished early
and the measured headroom is clearly large.

Default action: STOP at 20K.

---

# Phase 8 — OPTIONAL Decode Acceleration Study

Target: remaining time only.

Run ONLY if:

- Phase 1 correctness is closed;
- Phase 3 scaling is useful;
- Phase 4 endurance passes;
- Phase 5 quality classification is complete;
- no urgent unresolved failure remains.

Do NOT use an external draft model because memory is constrained.

Investigate ONE low-memory decode optimization.

Preferred candidate:

    n-gram self-speculative decoding / self-spec decode

Current upstream issue to inspect:

    mlx-lm #1497

The proposal is specifically interesting for Qwen hybrid GDN/attention models
because it uses CPU n-gram drafting and recurrent-state checkpoint/rollback
rather than loading another model.

Do not assume the upstream patch is correct or current.

First do static feasibility analysis against our pinned revision and Qwen3.8.

Only if implementation looks small and safe:

- apply/adapt it locally, not by upgrading canonical dependencies;
- test short correctness;
- require greedy token equivalence where the algorithm promises it;
- measure decode improvement at 4K/8K;
- measure memory overhead.

Stop if it threatens primary runtime correctness.

This phase is a bonus, not a required deliverable.

---

# Explicitly Out of Scope

Do NOT spend this stage on:

- new weight quant formats;
- mixed_2_6;
- uniform 2-bit;
- DWQ;
- MXFP4;
- NVFP4;
- 32K chasing;
- mlx_lm.server productionization;
- concurrency/batching;
- external draft models;
- DSpark;
- MTP experiments;
- system wired-memory tuning.

Do not broaden the project because an interesting tangent appears.

Record tangents under:

    Future work

and continue.

---

# Results Format

Append one JSON object per meaningful run to:

    research/stage3/results.jsonl

Suggested fields:

    timestamp
    experiment_id
    phase
    git_commit

    python_version
    mlx_version
    mlx_lm_revision

    model_name
    model_path
    quant_bits
    quant_group_size

    execution_mode
    persistent_cache
    kv_bits

    prompt_path
    prompt_sha256
    total_logical_tokens
    processed_suffix_tokens

    generated_tokens
    generated_token_hash

    classification
    replay_detected
    structured_output_valid

    ttft_seconds
    prompt_tps
    decode_tps
    total_seconds

    active_memory_bytes
    cache_memory_bytes
    active_plus_cache_bytes
    peak_memory_bytes
    rss_bytes

    cache_state_signature
    exception
    safety_stop_reason

Use null for unavailable values.

Never invent measurements.

---

# Research Notes

Maintain:

    research/stage3/STATUS.md

after every major experiment or decision gate.

Maintain:

    research/stage3/RESULTS.md

as a human-readable running synthesis.

Before each new phase re-read:

    AGENTS.md
    TASK.md
    research/stage3/STATUS.md

If the agent context becomes long, STATUS.md is the canonical compressed
working state.

---

# Git

Prefer logical commits such as:

    experiment: validate segmented Qwen3.8 cache state
    fix: preserve exact Qwen3.8 persistent chat state
    benchmark: scale exact-suffix Qwen3.8 chat
    benchmark: test Qwen3.8 persistent-chat endurance
    benchmark: compare Qwen3.8 g64 and g128 quality
    benchmark: evaluate KV8 at long context
    benchmark: probe Qwen3.8 20K persistent context
    experiment: test ngram self-spec decode

Do not commit model weights or unrelated changes.

---

# Final Report

Create:

    research/stage3/FINAL_REPORT.md

The final report must distinguish:

    MEASURED FACT
    INFERENCE
    FUTURE HYPOTHESIS

It must answer:

1. Is exact-suffix persistent cache validated?

2. Did same-boundary fresh-segmented execution match the true persistent path?

3. If not, where did divergence first occur?

4. What persistent contexts were actually tested?

5. What is the largest correctness-safe persistent context?

6. What is the recommended practical context?

7. What are follow-up TTFTs at 4K/8K/12K/16K?

8. Does follow-up TTFT depend mainly on suffix size or full history?

9. Did the 20-30 turn endurance test pass?

10. Was any unexplained memory/state drift found?

11. How does g128 compare with g64 on the diverse quality suite?

12. Should g128 be:
        DEFAULT
        PERFORMANCE/LONG-CONTEXT MODE
        EXPERIMENTAL ONLY

13. Is KV8 useful as a real long-context mode?

14. Was 20K tested?
    If yes, is it practical?

15. Was optional decode acceleration tested?
    If yes, what was the measured gain and correctness result?

16. What exact runtime configuration should the user run TODAY?

17. What are the three highest-value remaining follow-ups?

Include compact tables.

Do not claim a configuration was tested if it was only estimated.

---

# Completion Condition

The stage is complete when:

- the highest-value correctness question has been resolved as far as practical;
- scaling/endurance have been tested if correctness permits;
- g64 vs g128 quality has been materially clarified;
- FINAL_REPORT.md exists;
- STATUS.md reflects the final state;
- useful harness changes/results are committed.

Do not start another open-ended research program after writing FINAL_REPORT.md.

At the end return a concise summary with:

    cache verdict
    best practical context
    warm TTFT
    decode speed
    g128 quality verdict
    KV8 verdict
    20K verdict
    optional acceleration result
    FINAL_REPORT.md path
    commit SHAs
