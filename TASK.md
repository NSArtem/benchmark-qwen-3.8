# Stage 8 — Runtime Reliability and Sequential-Equivalent MTP Verification

## Mission

Run an autonomous Stage 8 for approximately 10-15 hours.

Stage 8 has exactly three primary engineering missions:

A. fix the length-truncated completion continuation bug that produced HTTP 409;

B. profile the exact Stage 7 MTP runtime quantitatively, including time and
   memory overhead by component;

C. build, if technically possible, a sequential-equivalent efficient verifier
   for Qwen3.8's hybrid attention + GatedDelta target model.

After those tasks, validate the production server with a genuinely long
8K-16K coding-agent session.

Do not reopen model quantization or DWQ research.

Critical path:

    recover Stage 7
    -> reproduce HTTP 409 at token level
    -> fix session/truncation semantics
    -> regression tests
    -> profile exact MTP
    -> localize batched-vs-sequential first divergence
    -> design transactional verifier
    -> implement sequential-equivalent verifier
    -> exactness oracle
    -> 4K/8K/16K performance
    -> long coding-agent endurance
    -> production cleanup
    -> final report

Use decision gates.

---

# Phase 0 — Recover Stage 7 and Establish Baseline

Target: 20-30 minutes.

Start:

    git status
    git log --oneline -30

Read completely:

    AGENTS.md
    research/stage7/FINAL_REPORT.md
    research/stage7/STATUS.md
    research/stage7/RESULTS.md
    research/stage7/results.jsonl

Inspect raw Stage 7 code/artifacts for:

    qwen38_agent_server.py
    qwen38_chat.py
    MTP architecture implementation
    exact MTP verifier
    4-bit MTP artifact/config
    MTP performance harness
    batched-verifier failure
    agent benchmark
    truncation/HTTP409 reproduction

Recover exact paths/revisions.

Verify canonical model:

    /Users/artem/models/Qwen3.8-27B-3bit-g128-MLX

Record clean system baseline:

    memory_pressure
    vm_stat
    swap
    free disk
    running heavy processes

Create:

    research/stage8/PLAN.md
    research/stage8/STATUS.md
    research/stage8/RESULTS.md
    research/stage8/results.jsonl

Do not rerun Stage 7 quality/DWQ experiments.

---

# Phase 1 — Reproduce the HTTP 409 Bug at Token Level

Target: 45-75 minutes.

This is the first P0 task.

Reproduce the exact Stage 7 failure:

    serialized session
    assistant generation reaches max_tokens
    finish_reason = length
    next request continues the conversation
    HTTP 409 / exact-prefix conflict

Use a small enough prompt/context to iterate quickly.

Do not begin with an 8K model run if a short reproduction works.

## Capture exact state

At the end of the truncated assistant turn save:

    generated token IDs
    decoded response text
    re-encoded token IDs from that text
    logical transcript token IDs
    cache offsets
    cache represented-prefix length
    assistant finish reason
    server session message representation
    canonical full-transcript tokens expected on next request

On the next request, save:

    client message list
    server reconstructed canonical tokens
    expected cached prefix
    actual longest common prefix
    first mismatching token index
    token IDs around mismatch
    decoded context around mismatch

Also record cache signatures/offsets if relevant.

## Test the hypotheses

Explicitly determine whether the first mismatch is caused by:

    decode -> encode non-identity
    chat-template assistant-close boundary
    cache lookahead/off-by-one
    server message history
    client echo/reconstruction
    another cause

Do not patch until the mismatch is understood.

## Deliverable

Create:

    research/stage8/truncation/ROOT_CAUSE.md
    research/stage8/truncation/repro.json

State one concrete root cause supported by raw token evidence.

If multiple independent causes exist, separate them.

Commit the reproduction/root-cause harness if useful.

Suggested:

    test: reproduce truncated-completion session mismatch

---

# Phase 2 — Implement Correct Truncated-Completion Continuation

Target: 60-90 minutes.

Fix the server/session representation based on Phase 1 evidence.

## Required semantics

A previous assistant message ending with:

    finish_reason = length

is a valid conversation turn.

A later request extending that conversation must not fail solely because EOS was
not generated.

## Preferred architecture

Maintain authoritative server-side session state that can include:

    exact generated token IDs
    decoded assistant text
    finish reason
    exact cached logical prefix length
    canonical turn boundary tokens

If the current bug is re-tokenization, use exact stored generated IDs for cache
identity rather than reconstructing them from response text.

Do not force this design if Phase 1 disproves that hypothesis.

## Cache reuse policy

### Exact fast path

If the new request extends the authoritative session exactly:

    reuse cache
    append only canonical missing suffix

### Safe fresh fallback

If the supplied valid message history does not match the reusable cached token
prefix:

    invalidate session cache
    fresh-prefill the supplied canonical transcript
    continue

Do not return HTTP 409 merely because cache reuse is unavailable.

Only return conflict/error for a genuinely invalid/ambiguous session contract.

## OpenAI compatibility

Return correct:

    finish_reason = length

for max-token completion.

Preserve returned response text exactly as API output.

Internal exact token state may contain more information than the API response.

## Unit/regression tests

Create lightweight tests for:

    EOS continuation
    max_tokens continuation
    multiple consecutive truncated completions
    reset after truncated completion
    client retransmission
    deliberate history edit
    fresh-fallback behavior

Budgets should cover a representative boundary set, such as:

    1, 2, 3, 31, 32, 63, 64, 127, 128, 255, 256

Use token-level/mock tests where possible.

Include:

    whitespace
    code
    indentation
    Unicode
    punctuation

Do not require 27B for the full matrix.

## 27B integration

After lightweight tests pass:

Run a real g128 session with:

    at least 3 intentional length truncations
    a follow-up after each

Require:

    no 409
    coherent continuation
    exact cache accounting or explicit fresh fallback
    no stale replay

Record warm TTFT for:

    exact-reuse continuation
    fresh-fallback continuation

Commit.

Suggested:

    fix: continue sessions after length-truncated completions

---

# Phase 3 — Harden Session Recovery Semantics

Target: 30-45 minutes.

Test server behavior when the client legitimately cannot use the cached prefix.

Examples:

    client edits an earlier message
    client omits/changes assistant text
    session ID reused with a different history
    tool payload differs
    conversation resumes from a stored external transcript

Correct production behavior should be explicit.

Preferred:

    valid transcript -> fresh-prefill and continue
    invalid request -> clear structured error

Do not silently use stale cache.

Add counters/metrics:

    cache_exact_reuse
    cache_fresh_fallback
    cache_conflict_error
    truncation_continuations

Expose them through:

    /metrics

or the current metrics surface.

Commit with Phase 2 if tightly coupled.

---

# Phase 4 — Instrument the Stage 7 Exact MTP Runtime

Target: 60-90 minutes.

This is the second P0 task.

Recover the known exact 4-bit MTP implementation.

Do not optimize it yet.

Add a diagnostic profiling mode.

## Time components

Measure per speculative cycle:

    draft_proposal
    target_state_snapshot
    target_verification
    acceptance_compare
    rollback_restore
    target_replay
    draft_state_sync
    explicit mx.eval/synchronization
    mx.clear_cache if material
    other bookkeeping
    total cycle

Count:

    draft calls
    target calls
    target tokens processed
    draft tokens proposed
    accepted tokens
    rejected tokens
    replayed target tokens

## Memory components

At useful points record:

    active
    cache
    active+cache
    peak
    RSS

Checkpoints:

    baseline model loaded
    MTP weights loaded
    before snapshot
    after snapshot
    after verification
    after rollback/commit
    end of cycle

Estimate actual bytes copied for:

    16 attention KV cache components
    48 recurrent state components

where possible.

## Diagnostic vs benchmark modes

Diagnostic profiling may synchronize.

Headline benchmark mode must not inherit artificial synchronization.

Make this separation explicit in code.

## Contexts

Detailed profile:

    short / ~1K
    ~4K
    ~8K

Do not profile 16K in high-overhead diagnostic mode unless memory is clearly
safe.

## Deliverable

Create:

    research/stage8/mtp/profile_summary.json
    research/stage8/mtp/PROFILE.md

The report must quantify:

    percentage of wall time by component
    percentage/bytes of extra memory by component
    target-call overhead per emitted token
    theoretical upper bound if snapshot/replay were free

Commit.

Suggested:

    benchmark: profile exact Qwen3.8 MTP overhead

---

# Phase 5 — Build the Sequential-vs-Chunk Target Oracle

Target: 60-90 minutes.

Before designing another verifier, identify the first actual divergence.

Use exact predetermined token sequences.

Compare:

### S — normal sequential target

    token 0
    token 1
    token 2
    ...

one target decode step at a time.

### C — current chunk/batched verifier

    same exact tokens
    processed as a short multi-token verification chunk

Do not involve draft-model uncertainty in this oracle.

## Test widths

Use small widths:

    2
    3
    4

or the actual Stage 7 candidate widths.

## Per-layer instrumentation

At each layer compare where practical:

    layer input/output signatures
    cache class
    cache offsets
    recurrent state signatures
    KV state signatures

Find:

    first layer whose output differs
    first cache state that differs
    first target-logit position that differs

Separate:

    48 recurrent/GatedDelta layers
    16 attention layers

## Numeric metrics

For differing tensors/logits record:

    exact equality
    max abs diff
    mean abs diff
    top-1 identity
    state hash/signature

Do not dump giant tensors.

## Segmentation controls

Useful controls:

    chunk width 1
    chunk width 2
    same tokens but layer-specific sequential recurrent mode if easy

Question:

> Is recurrent chunk execution the first source of divergence, attention batch
> execution, or something else?

## Deliverable

Create:

    research/stage8/mtp/VERIFIER_DIVERGENCE.md
    research/stage8/mtp/divergence.json

Do not implement the optimized verifier until this evidence exists.

Commit.

Suggested:

    experiment: localize Qwen3.8 verifier segmentation divergence

---

# Phase 6 — Design Transactional Hybrid Cache Verification

Target: 45-60 minutes.

Based on Phase 4 and Phase 5 evidence, write a concrete design before coding.

Create:

    research/stage8/mtp/TRANSACTIONAL_VERIFIER_DESIGN.md

The design should answer:

## Recurrent state

How to obtain sequential-equivalent recurrent state for each candidate token?

Preferred concept if supported:

    state_before
    state_after_token_1
    state_after_token_2
    ...

Then acceptance of k tokens commits:

    state_after_token_k

without replaying the prefix.

## Attention KV

Avoid copying the full historical KV.

Preferred:

    append speculative KV
    then trim rejected suffix

or:

    short delta/shadow KV

using only cache operations proven safe for that cache type.

## Full-model sequence execution

If recurrent layers can process the candidate sequence sequentially internally,
determine whether the model can still process a short sequence in one
high-level forward while remaining sequential-equivalent.

Document operation ordering.

## Rejection

Define exact behavior when:

    first draft token rejected
    middle token rejected
    all tokens accepted

## Draft state

Define how the MTP/draft cache is synchronized without unnecessary full replay.

## Memory

Estimate additional state as a function of:

    draft width
    recurrent checkpoint size
    attention delta size
    MTP cache

Compare to Stage 7's ~1.47 GiB extra 8K peak.

## Speed model

Estimate:

    target forwards per emitted token
    recurrent scan cost
    attention verification cost
    draft cost

A design that still requires one full target forward per emitted token should
not be expected to provide a major speedup.

Gate implementation based on this analysis.

---

# Phase 7 — Implement Sequential-Equivalent Multi-Token Verifier

Target: 120-210 minutes.

This is the core Stage 8 engineering phase.

Implement the smallest version of the Phase 6 design.

Keep it local to the Qwen3.8 runtime/harness initially.

Do not modify the canonical installed mlx-lm package in-place if a local adapter
or model helper is sufficient.

## Likely implementation direction

If Phase 5 identifies recurrent chunk processing as the divergence:

Implement a verifier-specific sequence mode that forces the recurrent
GatedDelta computation to advance token-by-token inside each recurrent layer,
while allowing the short candidate sequence to continue through the model as a
sequence.

The exact implementation must follow actual model code.

Do not assume this direction if Phase 5 shows a different first divergence.

## Transactional state

Avoid deep-copying the full target historical cache.

For recurrent layers:

    store only the minimal state/checkpoints necessary to select the accepted
    boundary.

For attention:

    use safe append/trim or delta cache.

## First correctness target

Use a fixed 2-token verification width.

Do not optimize width yet.

Require:

    same target top-1 tokens
    accepted-prefix cache equivalence
    correct rollback after first-token rejection
    correct rollback after partial acceptance
    correct all-accepted commit

Compare against token-by-token sequential oracle.

## Strong exactness tests

Use:

    coding
    reasoning
    prose
    structured output
    high-rejection examples
    high-acceptance examples

Run repeated speculative cycles, not only one cycle.

Test at least one generation long enough to contain many accept/reject events.

If state drifts after several cycles:

    STOP performance work and localize.

## Commit

Suggested:

    experiment: implement sequential-equivalent Qwen3.8 verifier

---

# Phase 8 — Verifier Exactness and State Oracle

Target: 45-75 minutes.

Before headline performance:

Run a dedicated oracle suite.

Compare baseline greedy vs optimized MTP:

    token IDs
    full output hash
    acceptance trace
    cache offsets
    recurrent state signatures
    attention KV signatures

At several checkpoints.

Use:

    short contexts
    ~1K
    ~4K

One ~8K exactness run if memory is comfortable.

Required outcome for exact MTP:

    baseline greedy output token-identical

and committed cache state must remain equivalent to sequential target behavior.

If logits have tiny differences but tokens/states remain exact, document.

If committed state differs:

    do not call the verifier exact.

---

# Phase 9 — Optimize Only Proven Bottlenecks

Target: 45-90 minutes.

Only after exactness passes.

Use Phase 4 profile to optimize the largest remaining components.

Potential optimizations may include:

    avoiding redundant mx.eval
    avoiding full cache copies
    preallocating recurrent checkpoints
    reducing Python loops
    compiling small recurrent verification helpers
    improving draft-state synchronization
    using a short cache delta representation

Do not apply speculative micro-optimizations without measurement.

After each meaningful change:

    rerun exactness
    rerun profile

Do not trade exactness for speed silently.

---

# Phase 10 — MTP Headline Performance

Target: 60-90 minutes.

Compare fresh-process:

    baseline g128
    optimized 4-bit MTP

Use the best exact verifier.

Contexts:

    ~4K
    ~8K
    ~16K

At least two repeats for surprising results.

Record:

    decode t/s
    speedup
    warm TTFT if relevant
    acceptance
    proposed tokens
    accepted tokens
    target calls
    draft calls
    peak
    active+cache
    RSS
    swap/system pressure
    output exactness

## Draft width

Only after the width-2/default path is exact and benchmarked:

Test at most a few widths, e.g.:

    1
    2
    3

if the MTP architecture/prototype supports them.

Select by real throughput, not acceptance alone.

## Production gate

Classify:

### DEFAULT ACCELERATION

    exact
    useful sustained gain at 8K and 16K
    safe memory

### OPTIONAL SHORT-CONTEXT

    exact
    useful gain at 4K/8K
    but 16K memory/performance penalty

### EXPERIMENTAL

    exact but marginal/no gain
    or implementation complexity remains high

### REJECTED

    cannot preserve correctness
    or clearly slower/unsafe

Do not keep optimizing after a well-supported negative result.

Commit.

Suggested:

    benchmark: measure optimized Qwen3.8 MTP verifier

---

# Phase 11 — Long 8K-16K Coding-Agent Session

Target: 90-120 minutes.

This is the production reliability gate.

Use:

    qwen38_agent_server.py

with:

    final g128 runtime
    optimized MTP only if it passed Phase 10

Build one or more deterministic local repository fixtures.

The principal session must grow to:

    >= 8K logical tokens

Preferred:

    8K-12K

One ~16K session or late-session phase is allowed if safe.

## Required interaction pattern

The session should include:

    system instructions
    repository context
    multiple file inspections
    tool requests/results
    an initial wrong hypothesis
    a failing test/result
    model correction
    multi-file reasoning
    a long tool result
    references to early constraints
    at least one 128-256 token forced truncation
    continuation after truncation
    additional turns after that continuation
    final task summary/solution

Target:

    ~20-30 exchanges

depending on tool payload sizes.

## Strict tool semantics

If the model requests the wrong tool/file:

    do not inject the expected result anyway.

Tool validity must affect task success.

Do not make the harness permissive merely to complete the run.

## Truncation reliability

Intentionally trigger at least:

    3 length-truncated assistant turns

across the long session.

Require:

    0 HTTP 409 from cache-prefix optimization
    correct continuation
    metrics show exact reuse or explicit fresh fallback
    no stale replay

## Metrics

Record per turn:

    role
    logical context
    suffix tokens
    exact cache reuse vs fresh fallback
    finish reason
    TTFT
    decode
    generated tokens
    tool validity
    HTTP status
    replay flag

Summary:

    task success
    valid/invalid tools
    truncation count
    truncation recovery success
    fallback count
    median warm TTFT
    p95/max warm TTFT
    decode median
    final context

## Baseline/MTP comparison

If MTP became a viable mode, compare at least one representative long agent task
with baseline and MTP.

Do not duplicate the entire 30-turn run unless useful.

Commit.

Suggested:

    benchmark: run long Qwen3.8 coding-agent session

---

# Phase 12 — Production Runtime Cleanup

Target: 30-60 minutes.

Integrate only validated behavior.

Update:

    research/runtime/qwen38_agent_server.py
    research/runtime/qwen38_chat.py

as appropriate.

Required production semantics:

    length truncation continuation works
    cache mismatch can fresh-prefill safely
    metrics expose reuse/fallback
    session reset remains correct

If MTP passed production gate:

    expose it behind an explicit option

Example:

    --mtp
    --mtp-path ...

If MTP is experimental:

    do not make it default.

Document:

    default model
    recommended 12K-16K context
    MTP role
    fallback behavior
    tool-use limitations

Add regression tests for every Stage 8 production bug fixed.

Do not add concurrency/batching.

Commit.

Suggested:

    feat: harden Qwen3.8 production agent runtime

---

# Phase 13 — Final Synthesis

Create:

    research/stage8/FINAL_REPORT.md

Update:

    research/stage8/STATUS.md
    research/stage8/RESULTS.md

Before writing conclusions inspect:

    truncation root-cause token dumps
    server regression tests
    MTP profile raw results
    verifier divergence artifacts
    transactional verifier exactness results
    MTP performance logs
    long agent transcripts
    results.jsonl
    git status
    git log

Clearly distinguish:

    MEASURED FACT
    INFERENCE
    HYPOTHESIS

---

# Suggested Time Allocation

Approximate:

    Phase 0        0.5h
    Phase 1-3      2.0-2.5h
    Phase 4        1.0-1.5h
    Phase 5-6      1.5-2.0h
    Phase 7-9      3.0-5.0h
    Phase 10       1.0h
    Phase 11       1.5-2.0h
    Phase 12-13    1.0h

Use gates.

If the verifier branch proves impossible within the window, finish the server
reliability and long-agent work thoroughly rather than forcing an unsafe
half-solution.

---

# Explicitly Out of Scope

Do NOT spend Stage 8 on:

    new static quants
    DWQ
    AWQ/GPTQ
    Q1/Q2/Q3
    generic draft-model search
    training an MTP head
    DSpark
    20K/24K/32K context records
    multi-user serving
    batching
    generic prefix-cache research
    system wired-memory changes

---

# Machine-Readable Results

Append meaningful runs to:

    research/stage8/results.jsonl

Suggested fields:

    timestamp
    phase
    experiment_id
    git_commit

    model_name
    model_path

    session_id
    finish_reason
    truncation_count
    truncation_recovered
    cache_reuse_mode
    cache_lcp_tokens
    cache_mismatch_index
    fresh_fallback

    mtp_enabled
    mtp_precision
    mtp_payload_bytes
    mtp_width
    mtp_proposed_tokens
    mtp_accepted_tokens
    mtp_acceptance_rate
    mtp_exact_match

    mtp_draft_seconds
    mtp_snapshot_seconds
    mtp_verify_seconds
    mtp_compare_seconds
    mtp_rollback_seconds
    mtp_replay_seconds
    mtp_sync_seconds
    mtp_other_seconds

    target_calls
    target_tokens_processed
    draft_calls
    replayed_target_tokens

    verifier_mode
    verifier_first_divergent_layer
    verifier_first_divergent_cache_type
    verifier_max_logit_diff
    verifier_state_exact

    context_tokens
    suffix_tokens
    generated_tokens
    ttft_seconds
    decode_tps

    active_memory_bytes
    cache_memory_bytes
    peak_memory_bytes
    rss_bytes
    swap_used_bytes

    agent_task_id
    agent_task_success
    agent_turn
    agent_tool_valid
    agent_http_status
    replay_detected

    classification
    exception
    safety_stop_reason

Use null for unavailable values.

Never invent measurements.

---

# Suggested Commits

Prefer logical commits:

    test: reproduce truncated-completion session mismatch
    fix: continue sessions after length-truncated completions
    test: harden Qwen3.8 session cache fallback
    benchmark: profile exact Qwen3.8 MTP overhead
    experiment: localize Qwen3.8 verifier segmentation divergence
    docs: design transactional Qwen3.8 verifier
    experiment: implement sequential-equivalent Qwen3.8 verifier
    benchmark: validate Qwen3.8 verifier exactness
    perf: reduce Qwen3.8 MTP verification overhead
    benchmark: measure optimized Qwen3.8 MTP verifier
    benchmark: run long Qwen3.8 coding-agent session
    feat: harden Qwen3.8 production agent runtime
    docs: summarize Qwen3.8 Stage 8

Do not commit model weights.

---

# FINAL_REPORT.md Must Answer

1. What exactly caused HTTP 409 after a length-truncated completion?

2. At what token index did the old session representation diverge?

3. Was decode->encode token non-identity involved?

4. Were assistant chat-boundary tokens involved?

5. Was there any cache off-by-one/lookahead bug?

6. What authoritative session representation is used now?

7. Do EOS and length-truncated messages both continue correctly?

8. How many truncation regression cases passed?

9. What happens when a valid request cannot reuse the cache?

10. Is fresh-prefill fallback correct and observable in metrics?

11. What percentage of old exact MTP runtime was spent in:
        draft
        snapshot
        verification
        rollback/replay
        synchronization
        other

12. What caused the extra ~1.47 GiB 8K MTP peak?

13. How much was full-history KV copying responsible?

14. What was the first layer where chunk verification diverged from sequential?

15. Was the first divergence recurrent or attention?

16. What exact state differed first?

17. What sequential-equivalent verifier design was implemented?

18. How are recurrent states checkpointed/committed?

19. How are attention KV speculative deltas committed/trimmed?

20. Does the optimized verifier avoid full historical cache duplication?

21. Does optimized MTP remain token-identical to baseline greedy?

22. Does committed cache state match sequential continuation?

23. What is MTP acceptance after optimization?

24. What is MTP decode at 4K?

25. What is MTP decode at 8K?

26. What is MTP decode at 16K?

27. What are the baseline comparisons/speedups?

28. What is optimized MTP peak overhead at 8K and 16K?

29. What is the final MTP classification?

30. How long was the final coding-agent session?

31. What maximum logical context did it reach?

32. How many tool requests were valid/invalid?

33. How many forced truncations occurred?

34. How many truncations continued successfully?

35. Were there any HTTP 409s?

36. How many exact cache reuses vs fresh fallbacks occurred?

37. What was median/p95/max warm TTFT?

38. What was long-session decode throughput?

39. Did the final coding-agent task succeed?

40. What exact server command should the user run TODAY?

41. What exact CLI command should the user run TODAY?

42. What are the three highest-value remaining Stage 9 ideas, if any?

Include:

### Truncation Table

    termination type
    test count
    exact reuse
    fresh fallback
    failures
    verdict

### MTP Profile Table

    component
    seconds/token or cycle
    percent wall
    memory contribution
    verdict

### Verifier Table

    mode
    exactness
    target calls/token
    memory overhead
    4K decode
    8K decode
    16K decode
    verdict

### Agent Table

    final context
    turns
    valid tools
    invalid tools
    truncations
    recovered
    409s
    median TTFT
    decode
    task success

---

# Completion Condition

Stage 8 is complete when:

- the truncated-completion HTTP 409 has a proven root cause;
- production continuation is fixed or a clearly documented blocker remains;
- lightweight and real-27B truncation regressions have run;
- exact MTP overhead has been quantitatively profiled;
- batched-vs-sequential first divergence has been localized;
- a sequential-equivalent verifier has been implemented and tested, OR a
  precise blocker has been demonstrated;
- optimized MTP has been benchmarked if exactness passes;
- a real 8K+ long coding-agent session has been run;
- production runtime behavior is documented;
- FINAL_REPORT.md exists;
- STATUS.md is current;
- useful work is committed.

Do not start Stage 9 automatically.

At completion return a concise summary with:

    truncation root cause
    truncation fix verdict
    exact reuse/fallback behavior
    MTP bottleneck breakdown
    verifier verdict
    MTP 4K/8K/16K speedup
    MTP memory overhead
    long-agent context/turns/success
    recommended server command
    recommended CLI command
    FINAL_REPORT path
    commit SHAs

Then STOP.
