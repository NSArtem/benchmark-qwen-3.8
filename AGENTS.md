# Qwen3.8 MLX Investigation — Stage 8 Project Instructions

This repository is now primarily a runtime-engineering project.

Stages 3-7 have converged on the static model:

    Qwen3.8-27B
    affine 3-bit
    group_size = 128

Path:

    /Users/artem/models/Qwen3.8-27B-3bit-g128-MLX

Payload:

    ~10.180 GiB

This is the production default entering Stage 8.

Stage 8 must solve three concrete engineering problems:

1. fix continuation after a length-truncated assistant completion, including the
   Stage 7 HTTP 409 failure;
2. profile the exact MTP implementation deeply enough to explain its time and
   memory overhead quantitatively;
3. implement, if technically possible, a sequential-equivalent efficient
   verifier for Qwen3.8's hybrid attention + GatedDelta architecture.

The final validation is a genuinely long serialized coding-agent workload at
8K-16K context.

Do not reopen broad weight-quantization or DWQ research.

---

## Hardware

Primary machine:

- MacBook Pro
- Apple M3 Pro
- 18 GB unified memory

The project is constrained by unified-memory residency.

Priority order:

1. correctness;
2. system stability;
3. production agent reliability;
4. sustained 8K-16K decode throughput;
5. warm-turn TTFT;
6. memory overhead;
7. cold prefill.

A short-context speedup that destroys 16K performance is not a general win.

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

Do not silently upgrade the canonical environment.

Current upstream source may be inspected for ideas/reference.

If a newer upstream implementation must be executed, use an isolated checkout
or environment and do not mutate the canonical runtime.

---

## System Safety

NEVER modify:

    sudo sysctl iogpu.wired_limit_mb

Never call:

    mx.set_wired_limit()

in the canonical runtime.

Never run two heavy Qwen/MLX workloads concurrently.

Hard stop conditions:

- Metal OOM;
- GPU timeout;
- severe WindowServer/UI instability;
- major system unresponsiveness;
- repeated failed allocations;
- clearly untrustworthy near-exhausted swap state.

Record memory/swap baseline before headline performance work.

Do not deliberately reproduce unsafe memory pressure repeatedly.

---

## Current Baseline Runtime

The validated default model is g128.

Representative historical results:

    ~8K:
        decode ~11.1 tok/s
        warm TTFT ~1.1 s
        peak ~13.98 GiB

    ~12K:
        decode ~10.8 tok/s
        warm TTFT ~1.1 s
        peak ~14.61 GiB

    ~16K:
        decode ~10.4-10.6 tok/s
        warm TTFT ~1.2 s
        peak ~15.24 GiB

    ~20K clean-host:
        decode ~10.37 tok/s
        technically working
        not the everyday recommendation

Recommended practical context:

    12K-16K

Do not spend Stage 8 re-benchmarking baseline unless needed for an A/B control.

---

## Persistent Cache Is Solved Infrastructure

Exact-suffix persistent chat has been validated.

The previous same-boundary oracle demonstrated:

    fresh segmented == true persistent

for:

    48 recurrent ArraysCache layers
    16 full-attention KVCache layers
    cache state
    first-token logits
    generated tokens

Do not reopen the old stale-replay investigation.

Stage 8 may modify session/token bookkeeping for truncation recovery, but the
model-cache continuation protocol itself remains exact-suffix.

---

## Static Quantization / DWQ Status

Static quant search is closed for Stage 8.

Previous variants did not beat g128 overall:

- g64;
- mixed34;
- RMS-guided mixed quant;
- KL-guided Q1;
- Q2/Q3 were not justified after Q1;
- bounded DWQ optimized its calibration objective but did not improve final
  held-out quality.

Do NOT spend Stage 8 on:

- new static weight quants;
- Q1/Q2/Q3;
- DWQ;
- AWQ;
- GPTQ;
- uniform 2-bit;
- mixed_2_6;
- generic 4-bit;
- MXFP4/NVFP4 sweeps.

The model is fixed unless MTP needs a separate draft-head artifact.

---

## Agent Backend

Expected path:

    research/runtime/qwen38_agent_server.py

or the current equivalent.

The backend is serialized / one active inference at a time.

It already supports or previously supported:

    /v1/chat/completions
    /v1/models
    /health
    /metrics
    session persistence
    session reset
    system/user/assistant/tool roles
    exact-suffix persistent cache

Stage 7 real smoke:

    4/4 small agent tasks passed
    5/5 expected tool requests valid
    0 invalid tool calls
    0 protocol errors

However the contexts were small, roughly sub-1K.

Stage 8 must validate long 8K-16K sessions.

---

## Known Production Bug: Length-Truncated Completion -> HTTP 409

Stage 7 observed a reproducible server edge case:

    assistant generation reaches max_tokens
    finish_reason = length / truncated completion
    client sends next request continuing the conversation
    server attempts exact-prefix reuse
    request fails with HTTP 409

This must be treated as a P0 production bug.

Do not assume the cause.

Likely hypotheses include, but are not limited to:

1. generated token IDs are decoded to text and later re-tokenized differently;
2. assistant closing chat-template tokens are inserted differently on the next
   request;
3. internal cache represents a different prefix than the reconstructed message
   transcript;
4. a generator lookahead/cache-offset accounting edge case occurs at a hard
   max-token boundary;
5. the client retransmits text that is semantically identical but tokenically
   different;
6. server session bookkeeping assumes EOS-terminated assistant messages.

Stage 8 must localize the exact first token/state mismatch.

---

## Truncated Completion Semantics

A length-truncated assistant response is still a valid assistant message.

The next user turn must NOT fail merely because the previous message ended due
to max_tokens instead of EOS.

Production behavior should be:

### Fast path

If the request extends the server's exact known session history:

    reuse exact stored token state
    feed only the canonical missing suffix

### Safe fallback

If cache reuse cannot be proven but the supplied conversation is valid:

    invalidate/rebuild the prompt cache
    fresh-prefill the full canonical transcript
    continue correctly

Do not return HTTP 409 simply because a cache optimization cannot be reused.

Reserve conflict/error behavior only for genuinely ambiguous or invalid session
semantics.

Correctness is more important than cache hit rate.

---

## Authoritative Session Representation

Do not assume decoded assistant text is sufficient to reproduce generated token
identity.

Stage 8 should strongly consider preserving, per assistant turn:

    exact generated token IDs
    decoded response text
    finish reason
    logical token range represented by cache
    canonical special-token boundary information

The server may use decoded text for OpenAI-compatible API responses while using
exact token IDs internally for cache continuity.

Before implementing this, prove whether re-tokenization is actually the Stage 7
bug.

Do not design around an unverified hypothesis.

---

## Required Truncation Regression Coverage

At minimum test completions ending by:

    EOS
    max_tokens / length
    explicit stop sequence if supported
    user cancellation / interruption if runtime supports it

Length tests should cover several budgets, e.g.:

    1
    2
    3
    31
    32
    63
    64
    127
    128
    255
    256

You do not need to load 27B for every unit test.

Build lightweight token-level/session tests.

Include content likely to stress decode/re-encode boundaries:

    normal prose
    leading/trailing whitespace
    code fences
    indentation
    punctuation
    Unicode
    partial-looking code/text

A subsequent turn must either:

    exact-reuse successfully

or:

    safely fresh-prefill

but must remain semantically correct.

---

## MTP Stage 7 Status

Stage 7 successfully reconstructed and implemented a correctness-first
Qwen3.8 MTP prototype.

Source MTP:

    15 mtp.* tensors
    BF16 payload ~0.791 GiB

Recovered architecture approximately:

    shared token embedding + target hidden state
        -> two RMSNorm paths
        -> concat
        -> projection to hidden size
        -> one full-attention decoder block
        -> RMSNorm
        -> shared LM head

Exact architecture details must be recovered from Stage 7 artifacts rather than
this summary.

---

## MTP Exactness Result

Stage 7 demonstrated exact greedy speculative behavior on the correctness suite
for the correctness-first implementation.

It handled both:

    16 full-attention KV caches
    48 recurrent ArraysCache / GatedDelta states

BF16 MTP was exact on tested prompt classes.

A 4-bit MTP candidate was also exact on the tested suite.

Approximate 4-bit MTP payload:

    ~0.210 GiB

Approximate short-suite acceptance:

    ~68% overall

with significant prompt-category variation.

This is a valuable result.

Do NOT rebuild the MTP architecture from scratch.

Recover and reuse the Stage 7 implementation/artifacts.

---

## MTP Stage 7 Performance

The correctness-first implementation was much slower than baseline.

Representative Stage 7 measurements:

    context    baseline      4-bit MTP
    ~1K        ~11.90 t/s    ~6.29 t/s
    ~4K        ~11.65 t/s    ~4.07 t/s
    ~8K        ~11.15 t/s    ~2.65 t/s
    ~16K       ~10.63 t/s    safety/memory stop

The MTP predictor itself is therefore NOT yet a production acceleration.

The current implementation strategy is the likely bottleneck.

---

## MTP Memory Result

At ~8K, Stage 7 observed roughly:

    baseline peak ~13.98 GiB
    MTP peak      ~15.46 GiB

Approximate increase:

    ~1.47 GiB

But 4-bit MTP weights themselves are only about:

    ~0.210 GiB

Therefore most additional peak memory is likely runtime state/intermediates,
such as:

    target cache snapshot/copy
    verification cache
    recurrent checkpoints
    draft cache
    replay state
    lazy MLX intermediates

Stage 8 must measure this directly rather than assume.

---

## Current MTP Verifier

Stage 7's exact implementation prioritizes correctness.

It performs substantial snapshot / sequential verification / rollback or replay
work.

This preserved exact target semantics but removed the normal efficiency
advantage of speculative decoding.

Stage 8 must first profile it component-by-component.

Do not optimize before establishing the measured time/memory budget.

---

## Current Upstream Speculative Decoder Reference

At an upstream mlx-lm revision inspected before Stage 8
(`f4f3b57971c39b0bc0bcc2e0cab27e77ea9d53f5`), generic
`speculative_generate_step()`:

- verifies multiple draft tokens in a target model call;
- rewinds using `trim_prompt_cache()`;
- explicitly requires the target prompt cache to be trimmable;
- raises when cache types are non-trimmable.

Qwen3.8 has recurrent ArraysCache layers which are not generically trimmable.

Therefore generic transformer speculative decoding cannot simply be dropped
into this hybrid runtime.

Inspect current upstream again during Stage 8 if useful, but do not blindly
replace the validated local implementation.

---

## Failed Batched Verification Result

Stage 7 attempted an obvious batched target verifier.

It was not exact for all prompt classes and produced rollback/state failure in
at least one reasoning case.

Do NOT re-enable it unchanged.

The key Stage 8 research question is:

> Why does batched/chunked target verification differ from token-by-token target
> execution, and can we construct a multi-token verifier whose layer/cache state
> is sequential-equivalent?

---

## Sequential-Equivalent Means

The required correctness oracle is stronger than "text looks similar."

For a proposed verification chunk, compare against token-by-token target
execution on the same exact token stream.

At useful boundaries compare:

    target logits / top-1
    48 recurrent cache states
    16 attention KV cache states / offsets
    accepted-prefix state
    post-rejection state

The ideal result is exact equality where feasible.

At minimum, the committed target state after accepted tokens must be equivalent
to the normal baseline continuation strongly enough that long greedy output
remains token-identical.

Do not call an approximate verifier "exact."

---

## Stage 8 Verifier Strategy

Stage 8 should investigate the first divergence before implementing a solution.

Potential strategies include:

### A. Per-layer sequential recurrence inside a multi-token target call

Process the proposed sequence as a short sequence, but force recurrent
GatedDelta/ArraysCache updates to occur token-by-token in the exact order used
by normal decode.

The rest of the layer/model may retain sequence-level execution where that is
proven equivalent.

This may reduce Python/model-call overhead while preserving recurrence
semantics.

### B. Transactional recurrent state

During verification, retain:

    state before candidate chunk
    recurrent state after each proposed token

Then after acceptance of k draft tokens:

    commit recurrent state at accepted boundary k

without replaying the entire target prefix.

For full-attention KV caches, use append + trim/offset rollback where the cache
type safely supports it.

Do not deep-copy the full historical attention KV cache merely to support a
2-4-token speculative transaction.

### C. Shadow/delta cache

Share immutable historical cache and allocate only short speculative deltas.

Commit accepted deltas or discard rejected suffixes.

This is preferable to duplicating the whole long-context cache if supported.

### D. Hybrid approach

It may be necessary to combine:

    sequential recurrent processing
    + attention delta cache
    + per-token recurrent checkpoints

The implementation should follow measured first-divergence evidence.

---

## Important Limitation

A verifier that still performs one full target-model forward per accepted draft
token is unlikely to outperform baseline once draft overhead is included.

Eliminating cache copies alone may reduce memory, but it does not create the
main speculative-decoding compute win.

Therefore Stage 8 must distinguish:

    memory optimization

from:

    actual target-compute reduction.

Before claiming expected speedup, count:

    target model calls
    target tokens processed
    draft model calls
    accepted tokens per verification cycle

Build a simple throughput model and compare it to measured data.

---

## MTP Profiling Requirements

Instrument the exact Stage 7 MTP path.

Measure at minimum:

    draft proposal time
    target cache snapshot/checkpoint time
    target verification time
    token comparison/acceptance time
    rollback/restore time
    replay time
    draft-state synchronization time
    mx.eval / synchronization time
    cache clearing time if material
    total iteration time

Also count:

    proposed tokens
    accepted tokens
    rejected tokens
    target calls
    draft calls
    replayed target calls
    bytes/state copied where measurable

Memory checkpoints should distinguish:

    before MTP weights
    after MTP weights
    before snapshot
    after snapshot
    after verification
    after rollback/commit
    end of iteration

Use:

    active memory
    cache memory
    peak memory
    RSS

where practical.

---

## Profiling Mode vs Benchmark Mode

Detailed component profiling may require explicit synchronization and therefore
distort performance.

Keep two modes:

### Diagnostic profile mode

May force synchronization to attribute component time accurately.

Use for bottleneck decomposition.

### Headline benchmark mode

Do not add artificial synchronizations beyond those required by the real
algorithm.

Use for final throughput comparison.

Never mix profile-mode timing with production benchmark timing without labeling
it.

---

## Draft Width

Recover the actual Stage 7 MTP draft width/semantics.

Do not assume a fixed value.

After correctness of an optimized verifier is established, test a very small
width set if useful, e.g.:

    1
    2
    3

or the architecture-appropriate equivalents.

Do not run a large tuning sweep.

The optimum may depend on acceptance and verification cost.

---

## MTP Production Gate

MTP may become a production mode only if:

1. exactness passes;
2. no cache-state corruption;
3. memory is safe;
4. sustained throughput improves materially at useful context;
5. the long-context result does not show a residency cliff.

Preferred evaluation:

    4K
    8K
    16K

A useful general MTP mode should improve 8K and ideally 16K.

If it only improves 4K:

    classify SHORT-CONTEXT OPTIONAL.

If exact optimized MTP still cannot beat baseline:

    preserve the bottleneck proof and close the branch.

Do not keep optimizing indefinitely.

---

## Long Agent Benchmark Requirements

Stage 8 must go beyond Stage 7's sub-1K fixtures.

Use local deterministic repository fixtures.

At least one serious run should reach:

    >= 8K logical context

Preferred:

    8K-12K

If system/runtime is healthy, one run may approach:

    ~16K

The session should involve many real dependencies across turns:

- inspect files;
- request tool output;
- diagnose a bug;
- propose a patch;
- receive a failing test;
- revise;
- inspect another file;
- maintain earlier constraints;
- handle at least one long tool result;
- handle at least one deliberately length-truncated assistant completion;
- continue after truncation;
- finish with a correct summary/solution.

Do not inject a tool result after an invalid/unexpected tool request and still
count the task as success.

Tool validity must be part of task success.

---

## Agent Benchmark Success

For each task/session record:

    final success
    tool request validity
    invalid tool calls
    turns
    logical context
    suffix tokens
    cache hit/reuse vs fresh fallback
    length truncations
    truncation recoveries
    HTTP errors
    median warm TTFT
    max warm TTFT
    decode throughput
    final task correctness

Separate:

    server transport/protocol reliability

from:

    model autonomous tool-choice quality.

Do not call both the same thing.

---

## Results and Project Memory

Maintain:

    research/stage8/PLAN.md
    research/stage8/STATUS.md
    research/stage8/RESULTS.md
    research/stage8/results.jsonl
    research/stage8/FINAL_REPORT.md

Before every major phase re-read:

    AGENTS.md
    TASK.md
    research/stage8/STATUS.md

STATUS.md must include:

    current phase
    memory/swap baseline
    truncation-bug status
    exact mismatch localization
    server fallback status
    current MTP profiler findings
    current first-divergence layer
    verifier implementation status
    MTP exactness status
    MTP 4K/8K/16K performance
    long-agent benchmark status
    next action

---

## Git Discipline

Always begin with:

    git status
    git log --oneline -30

Preserve unrelated changes.

Never:

    git reset --hard
    git clean -fd
    discard unrelated work

Do not commit:

    model weights
    HF cache
    virtualenvs
    giant traces
    huge cache snapshots

Commit:

    server fixes
    regression tests
    MTP profiler
    verifier code
    compact state-signature artifacts
    benchmark harnesses
    agent fixtures
    JSON/JSONL results
    reports

Use focused logical commits.

---

## Autonomy

Expected Stage 8 window:

    approximately 10-15 hours

Do not ask routine clarification questions.

If one branch is blocked:

    document the exact reason
    preserve evidence
    continue with an independent high-value branch

Examples:

- if optimized verifier is blocked, still finish truncation reliability and
  long-agent validation;
- if server truncation bug is fixed early, do not create new research topics;
  spend the time on MTP and agent endurance.

Do not start Stage 9 automatically.

---

## Explicitly Out of Scope

Do NOT spend Stage 8 on:

- static weight quantization;
- DWQ;
- AWQ/GPTQ;
- Q1/Q2/Q3;
- new draft-model training;
- new MTP-head training;
- DSpark;
- 20K/24K/32K context chasing;
- multi-user batching;
- concurrent serving;
- generic prefix-cache trimming;
- wired-memory system tuning.

---

## Final Standard

Stage 8 should answer:

1. What exactly caused the length-truncated completion HTTP 409?

2. Can a truncated assistant completion now continue reliably?

3. Does cache reuse remain fast after truncation?

4. When cache reuse is impossible, does the server safely fresh-prefill instead
   of failing?

5. What fraction of current MTP time is spent in:
       draft
       snapshot
       target verification
       rollback/replay
       synchronization
       other

6. What causes the ~1.47 GiB extra 8K MTP peak?

7. What is the first layer/cache component where batched verification diverges
   from sequential target execution?

8. Can a sequential-equivalent multi-token verifier be implemented?

9. How does it commit accepted recurrent state without full-prefix replay?

10. How does it rollback rejected recurrent state?

11. Does it avoid duplicating the full attention KV history?

12. Is optimized MTP exact?

13. What is optimized MTP throughput at 4K/8K/16K?

14. What is optimized MTP memory overhead at 8K/16K?

15. Is MTP production-useful or should the branch be closed?

16. Does a real 8K-16K coding-agent session survive length truncation,
    long tool results, and many turns?

17. What exact CLI/server configuration should the user run TODAY?
