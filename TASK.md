# Night 2 Task — Fix Qwen3.8 Persistent Chat Correctness and Validate g128 Interactive Runtime

## Mission

Continue the Qwen3.8 MLX investigation autonomously overnight.

The previous run successfully produced a source-backed affine 3-bit/group-128
checkpoint with ~0.783 GiB lower model payload and coherent cold generation
through ~16.7K.

However, post-run inspection found a serious correctness problem in the prior
8K persistent-cache chat benchmark:

> turn 2 generated exactly the first 64 generated tokens of turn 1,
> token-for-token, despite a different follow-up query.

Therefore the highest-value objective is now:

> determine why persistent multi-turn cache can replay stale output, build a
> correctness-safe chat protocol, and then validate the g128 model as a real
> interactive runtime.

Do not spend the night generating additional weight quants.

Do not treat low TTFT as success unless follow-up semantics are correct.

---

# Phase 0 — Recover state, preserve g128, and create Night-2 ledger

Start with:

    git status
    git log --oneline -15

Read completely:

    AGENTS.md
    TASK.md
    research/FINAL_REPORT.md
    research/STATUS.md
    research/runtime/qwen38_runtime.py
    research/harness/run_warm_conversation.py

Inspect raw prior warm artifacts:

    research/warm-cache/
    research/warm-cache-8k/

Confirm programmatically that the old 8K turn-2 token sequence equals the first
64 turn-1 tokens. Record the exact hashes and overlap.

Do not alter the raw historical files.

Create:

    research/night2/PLAN.md
    research/night2/STATUS.md
    research/night2/RESULTS.md
    research/night2/results.jsonl

The first STATUS entry must explicitly record that the old P3-WARM8K
`EXCELLENT` correctness conclusion is superseded/invalid pending revalidation.

## Preserve the g128 checkpoint

Expected previous temporary path:

    /private/tmp/qwen38-g128-mlx

Choose a persistent untracked location, preferably:

    ~/models/Qwen3.8-27B-3bit-g128-MLX

Rules:

1. If the persistent path already exists, verify it rather than overwriting.
2. If only the temporary path exists, move or copy it to the persistent path
   and verify config/index/shard integrity.
3. If neither exists but the high-precision source remains local, regenerate
   g128 with the existing bounded converter into the persistent path.
4. If the source is also unavailable, record the blocker and continue cache
   diagnosis with g64.
5. Never commit model weights.

Verification should include:

    quantization mode/bits/group_size
    shard count
    tensor/index key agreement
    payload size approximately 10.180 GiB

Record the final g128 path in:

    research/night2/STATUS.md

Do not delete the high-precision source or unrelated caches.

---

# Phase 1 — Correct the research record

Do not rewrite history silently.

Create:

    research/night2/ERRATA.md

Document:

- old report claim: 8K persistent cache = EXCELLENT;
- raw evidence: turn 2 repeats first 64 tokens of turn 1 exactly;
- corrected interpretation: performance numbers are real, correctness result is
  invalid;
- 4K did not show exact replay but still requires the new oracle;
- g128 warm behavior was never established.

If useful, add a short clearly labeled erratum note to the top or bottom of the
existing `research/FINAL_REPORT.md`, pointing to `research/night2/ERRATA.md`.
Do not delete or alter old raw measurements.

Commit this correction separately if clean.

Suggested commit:

    docs: correct Qwen3.8 warm-cache conclusion

---

# Phase 2 — Build a protocol-aware cache correctness harness

The old heuristic classifier is insufficient.

Build a new harness under:

    research/night2/harness/

It must be able to run one initial large user prompt and at least two strongly
different follow-ups.

For each turn save:

    query text/hash
    query token IDs
    prompt-fragment token IDs
    generated token IDs
    generated token SHA256
    raw text
    first-64-token overlap with every previous answer
    longest common prefix with previous answers
    n-gram replay metrics
    timing
    memory
    cache diagnostics

Use deterministic greedy generation.

The harness must implement THREE execution modes.

## Mode A — current persistent incremental protocol

Reproduce the current `NoWiredChat` behavior:

    one persistent prompt_cache
    each turn passes the freshly chat-templated user fragment

This is the suspect path.

## Mode B — exact-token concatenation fresh oracle

For turn N, reconstruct in a completely fresh process the exact token sequence
that Mode A logically claims is represented by its cache:

    prior prompt fragment tokens
    + prior generated token IDs
    + next prompt fragment tokens
    ...

Then generate from scratch.

Do not rely on decoded/re-encoded assistant text for this oracle when exact
previous generated token IDs are available.

Purpose:

- if A != B, persistent cache state is not equivalent to replaying its supposed
  token stream;
- suspect cache/runtime implementation.

## Mode C — full-transcript fresh chat oracle

Maintain logical messages:

    user
    assistant
    user
    assistant
    ...

Serialize the full conversation with the model's actual chat template and run
fresh generation.

Purpose:

- if B is wrong/replays but C responds correctly, the incremental turn
  serialization/protocol is wrong even if cache mechanics are internally
  consistent.

Save all three outputs side-by-side.

## Strong follow-up prompts

Follow-ups should be intentionally distinguishable.

Example structure:

Turn 1:
    diagnose the Lantern incident

Turn 2:
    answer ONLY which single database invariant should be enforced first

Turn 3:
    answer ONLY with one focused regression test and its expected failure

Avoid vague prompts where repeating a diagnosis could plausibly appear relevant.

## Acceptance signals

A follow-up is BROKEN if any of the following occurs:

- inappropriate exact replay of a prior answer;
- >=90% prefix-token overlap with prior answer without semantic justification;
- ignores the new task and repeats stale content;
- token soup / pathological loop.

Do not classify grammatical replay as COHERENT.

---

# Phase 3 — Reproduce and locate the cache failure boundary

First use g64 so the historical result is directly comparable.

Run sequentially, fresh process per conversation session:

    ~4K initial prompt
    ~6K initial prompt
    ~7K initial prompt
    ~8K initial prompt

You may reuse/derive the existing Lantern prompt family.

Do not run all modes for every point if early results already reveal the
failure class. Start with Mode A, then run B/C at the first clearly failing
length and one known-passing length.

Initial output:

    96 or 128 tokens

Follow-ups:

    48 or 64 tokens

Enough output must be generated to detect replay.

## Boundary questions

Determine:

1. Is replay reproducible at 8K?
2. Does it begin sharply between 4K and 8K?
3. Does it depend on exact prompt length?
4. Does it depend on number of generated tokens in turn 1?
5. Does it depend on prefill_step_size?

Only if useful, at the first failing length test:

    prefill_step_size = 2048
    prefill_step_size = 1024

Do not sweep many chunk sizes.

If 8K replay does NOT reproduce, repeat once and inspect whether the previous
harness itself constructed turn state incorrectly.

## Decision gate

Classify the problem as one of:

A. CACHE_STATE_BUG
    persistent Mode A differs materially from exact-token fresh Mode B

B. INCREMENTAL_PROTOCOL_BUG
    A approximately matches B, but both are wrong vs full-transcript Mode C

C. BOTH

D. HARNESS_BUG
    corrected reproduction shows prior test construction was invalid

E. UNRESOLVED

Do not proceed as if fixed until one classification has evidence.

---

# Phase 4 — Inspect hybrid cache internals

The model uses a hybrid cache with recurrent and attention layers.

Instrument cache state around the failing and passing cases.

For every layer record compact metadata, not giant tensor dumps:

    layer index
    cache class
    offset / logical position when available
    state tensor shapes
    state dtypes
    state nbytes
    trimmable status
    small deterministic state signatures/checksums when safe

Expected rough split from prior work:

    48 ArraysCache
    16 KVCache

Capture at least:

    after cache creation
    after initial prompt prefill
    after initial generation
    immediately before follow-up prefill
    after follow-up prompt processing
    after first follow-up generated token

Do not copy entire multi-GiB states to CPU just to hash them.
Use small slices/scalars/statistics where appropriate.

Check whether any cache layer:

- resets unexpectedly;
- stops advancing;
- wraps/clamps at a suspicious boundary;
- has offsets inconsistent with other layers;
- changes state when it should not;
- fails to change when it should;
- receives duplicated special/BOS/control tokens in the incremental protocol.

Also inspect the exact token boundary between:

    turn-1 prompt
    generated answer
    turn-2 fragment

Print token IDs and decoded special tokens around the join.

This is important: a cache may be mechanically correct while the concatenated
turn token stream is semantically invalid.

---

# Phase 5 — Upstream source/issue investigation

Use primary sources only.

Compare pinned mlx-lm revision against current upstream for relevant files,
including at minimum:

    mlx_lm/generate.py
    mlx_lm/chat.py
    mlx_lm/models/cache.py (or equivalent)
    Qwen3.5/Qwen3.8 model/cache implementation

Search official MLX/mlx-lm issues and PRs for:

    Qwen3.5 hybrid cache
    Qwen3.8 cache
    ArraysCache
    GatedDelta / recurrent cache
    long conversation
    prompt_cache
    chat cache
    cache offset
    repeated output / stale output

Record exact issue/PR/commit references in:

    research/night2/upstream.md

Do not upgrade the canonical environment.

If current upstream contains a plausible relevant fix:

1. identify the minimal code delta;
2. explain why it matches the observed failure;
3. test it in an isolated local patch/helper or disposable environment;
4. do not mutate canonical env files silently.

If no upstream fix exists, say so and continue locally.

---

# Phase 6 — Implement a correctness-safe chat path

Do not merely hide the replay with sampling.

The implementation depends on Phase-3 diagnosis.

Possible solution classes include:

1. corrected incremental token framing;
2. correct boundary/control-token processing between assistant and next user;
3. cache-state fix adapted from upstream;
4. transcript-aware cache reuse;
5. full-transcript re-prefill fallback when safe reuse cannot be guaranteed.

Prefer the smallest correctness-preserving change.

Do not optimize before correctness.

## Required modes

The user-facing runtime should eventually expose at least:

    chat-safe

and, only if proven correct:

    chat-fast

`chat-safe` may sacrifice warm TTFT if necessary, but it must implement true
logical conversation semantics.

`chat-fast` may use persistent reuse only after passing the oracle tests.

If the existing `chat` command is fixed, keep backward compatibility where
reasonable.

## Acceptance tests

At minimum the fixed/recommended chat path must pass:

    g64 ~4K, 3 follow-ups
    g64 ~8K, 3 follow-ups

with:

- no stale replay;
- responses relevant to each current follow-up;
- two independent fresh session repeats at the formerly failing ~8K case;
- protocol-aware comparison against Mode B and Mode C;
- saved token IDs and hashes.

If persistent cache reuse cannot be made correct, do NOT call it solved.
Ship/document the safe fallback and preserve the unresolved fast-cache bug.

Commit a correct runtime fix separately.

Suggested commit:

    fix: make Qwen3.8 multi-turn cache semantics safe

---

# Phase 7 — Validate g128 as an interactive model

Only enter after a correctness-safe chat path exists.

Use the persistent g128 model path established in Phase 0.

Do NOT regenerate the quant unless needed.

Run adaptive warm-conversation validation:

    ~4K
    ~8K
    ~12K

If all are correct and stable, test:

    ~16K

Use the same strong follow-up set and correctness oracles.

Measure per turn:

    actual prompt/new-token counts
    TTFT
    prompt tok/s where meaningful
    decode tok/s
    active memory
    cache memory
    active+cache
    peak GiB
    OS memory pressure/swap snapshot

Compare against g64 at matching useful points.

## Recommendation gate

Determine separately:

    largest correctness-safe warm context
    largest memory-safe warm context
    recommended practical warm context

Do not automatically choose the largest.

A 16K initial cold TTFT of several minutes may still be acceptable for a coding
agent if subsequent turns are fast and correct, but state this tradeoff rather
than hiding it.

---

# Phase 8 — Verify g128 quality delta vs g64

The current evidence proves g128 is coherent; it does not quantify subtle
quality loss from doubling group size.

Build a small objective comparison that does NOT require loading g64 and g128
simultaneously.

Preferred method:

- use a held-out text/coding corpus already available in the repo or create a
  small deterministic non-secret corpus;
- calculate teacher-forced average negative log-likelihood / perplexity for
  g64 and g128 in separate fresh processes;
- use bounded chunks/cache so logits do not create unsafe memory spikes;
- compare the same exact token sequence.

Also optionally record next-token top-1 agreement / target-token logprob on a
small set of sampled positions.

Do not use an LLM-as-judge as the only quality metric.

If implementing perplexity safely becomes disproportionately complex, use a
small deterministic task suite with objectively checkable answers and explain
the limitation.

Decision:

    g128 quality delta negligible / small / material / unresolved

This phase should not endanger the machine.

---

# Phase 9 — Recheck g128 performance advantage

Prior cold results suggest g128 may decode faster than g64 in addition to using
less memory, especially at 8K/12K, but those were not repeated for performance
statistics.

If time permits after cache correctness:

Run controlled fresh-process repeats at:

    4K
    8K

for:

    g64
    g128

Two measured runs per model/length are enough.

Keep:

    same prompt
    greedy
    same output length
    prefill_step_size 2048
    no KV8
    no MTP/speculation

Report medians for:

    TTFT
    prompt tok/s
    decode tok/s
    peak memory

Do not overfit tiny differences.

---

# Phase 10 — Long-context headroom: KV8 and prefill chunking

Only enter after:

- cache correctness work is complete or clearly blocked;
- g128 is validated at useful contexts;
- no safety incident occurred.

The goal is NOT to chase 32K.
The goal is to learn whether 16K+ can be made safer.

## Step 10A — g128 + KV8 at ~16K

Use:

    kv_bits = 8
    kv_group_size = 64

Verify the pinned API and quantized_kv_start behavior from source first.
Do not blindly assume start=0 is optimal; use the simplest justified setting.

Compare against existing normal-g128 ~16.7K result.

Measure:

    correctness
    peak saving
    active+cache saving
    decode penalty
    TTFT

At 8K KV8 saved ~0.231 GiB and slowed decode. At ~16K the memory saving should
be larger if the expected KV growth dominates; measure rather than extrapolate.

## Step 10B — prefill_step_size control

At a safe long context, compare at most:

    2048
    1024

only if reducing prefill chunk could materially lower transient peak.

Do not sweep lower values unless a clear allocation problem requires it.

## Step 10C — optional ~20K probe

A ~20K g128 probe is allowed only if ALL are true:

- 16K/g128 is coherent;
- no UI/GPU/system instability was observed/reported;
- measured binary-GiB peak remains comfortably below the historical danger
  envelope after unit conversion;
- `memory_pressure`/OS telemetry is not severe;
- the best available long-context configuration (normal or KV8) has a
  conservative projected peak.

Prefer the safer of normal g128 and g128+KV8.

Generate only 96–128 tokens.

Stop after ~20K unless the 24K gate below is extraordinarily safe.

## Step 10D — optional ~24K gate

24K is NOT an automatic task.

Run a single ~24K probe only if ALL are true:

- ~20K completed coherently;
- measured ~20K peak is <= about 15.5 GiB;
- system memory-pressure telemetry remains healthy enough;
- no safety stop occurred;
- a conservative extrapolation keeps 24K below about 15.8 GiB;
- KV8 or another already-proven non-weight lever provides enough headroom.

If these conditions are not met, explicitly SKIP 24K.

Do not test 32K tonight.

Do not turn safety thresholds into targets.

---

# Phase 11 — Do not reopen rejected quant branches

Do NOT run:

    mixed_2_6
    uniform 2-bit
    DWQ
    MXFP4
    NVFP4
    new custom mixed weight recipes

unless every primary cache/runtime task is complete, there is substantial time
remaining, and a new result creates a specific evidence-based reason.

Default action: skip all new weight quants tonight.

The source-backed g128 converter is already a major success; exploit it rather
than generating a quant zoo.

---

# Phase 12 — User-facing runtime commands

Once the safe chat path is known, update/create exact commands for:

1. one-shot g128 generation;
2. correctness-safe interactive g128 chat;
3. optional 16K+ safer variant if KV8/prefill tuning genuinely wins.

Commands must:

- use the canonical uv environment;
- make no-wired behavior explicit;
- print model path and quantization;
- print MLX version / pinned mlx-lm revision;
- never require sysctl changes;
- never modify installed mlx-lm files in place.

If fast persistent cache remains unresolved, the documented default must be the
safe path, not the fast broken one.

---

# Result schema

Append each meaningful experiment to:

    research/night2/results.jsonl

Fields where applicable:

    timestamp
    experiment_id
    git_commit
    python_version
    mlx_version
    mlx_lm_revision

    model_path
    quantization

    protocol_mode
    prompt_path
    prompt_sha256
    prompt_tokens
    incremental_prompt_tokens

    generated_tokens
    generated_token_hash
    previous_turn_prefix_overlap
    replay_detected
    classification

    exact_token_oracle_hash
    full_transcript_oracle_hash
    oracle_relation

    ttft_seconds
    prompt_tps
    decode_tps
    total_seconds

    active_gib
    cache_gib
    active_plus_cache_gib
    peak_gib
    os_memory_metrics

    cache_layer_summary
    exception
    safety_stop_reason

Use null when unavailable.
Never invent measurements.

---

# STATUS discipline

After every meaningful experiment or decision gate update:

    research/night2/STATUS.md

Before each new major phase re-read:

    AGENTS.md
    TASK.md
    research/night2/STATUS.md

STATUS must always contain:

    Current phase
    Last completed experiment
    Current diagnosis of the cache bug
    Current trusted chat mode
    Persistent g128 path
    Best validated warm context
    Best cold context
    Safety state
    Next action

This is the canonical external memory for the unattended run.

---

# Git discipline

Commit logical milestones separately.

Suggested commits:

    docs: correct Qwen3.8 warm-cache conclusion
    test: add protocol-aware Qwen3.8 cache oracle
    debug: localize Qwen3.8 persistent cache replay
    fix: make Qwen3.8 multi-turn cache semantics safe
    benchmark: validate Qwen3.8 g128 warm context
    benchmark: compare Qwen3.8 g64 and g128 quality
    benchmark: test Qwen3.8 long-context headroom
    docs: summarize Qwen3.8 night2 findings

Do not commit model weights or HF source shards.

---

# Hard stop rules

Never:

- change iogpu.wired_limit_mb;
- call mx.set_wired_limit for canonical inference;
- run MLX 0.32.0;
- run multiple huge Qwen MLX workloads concurrently;
- repeatedly reproduce an OOM/dangerous point;
- delete unrelated model caches;
- upload model weights;
- silently upgrade canonical MLX/mlx-lm;
- test 32K tonight;
- call a replaying chat path correct because its TTFT is good.

If a safety gate fails, preserve evidence and move to another safe phase.

---

# Final report

Create:

    research/night2/FINAL_REPORT.md

It must answer explicitly:

1. Was the old 8K replay reproduced?
2. What exact token/cache/protocol condition caused it?
3. Was the bug in cache state, turn serialization, harness behavior, or more
   than one layer?
4. Did upstream already contain a relevant fix?
5. What code/protocol change fixed it, if any?
6. What is the trusted user-facing chat mode now?
7. Does 4K multi-turn pass the new oracle?
8. Does 8K multi-turn pass the new oracle?
9. Does g128 pass warm chat at 4K/8K/12K/16K?
10. What are warm follow-up TTFT and decode rates for g128?
11. Is g128 quality measurably worse than g64?
12. Is g128 actually faster than g64 in controlled repeats?
13. What does KV8 save at ~16K, and what decode penalty does it impose?
14. Was 20K tested? If not, why was it correctly skipped?
15. Was 24K tested? If not, which safety gate prevented it?
16. What is the maximum correctness-safe tested context?
17. What is the recommended practical interactive context?
18. What exact command should the user run tomorrow?
19. What remains unresolved?
20. What single next experiment has highest expected value?

Include sections:

    Measured facts
    Corrections to prior conclusions
    Cache diagnosis
    g128 interactive results
    Quality/performance comparison
    Long-context safety
    Recommended runtime
    Things deliberately not done
    Unresolved questions

Clearly distinguish:

    measured fact
    inference
    hypothesis

At the end run:

    git status
    git log --oneline -20

Commit appropriate small artifacts.

Then return a concise final message with:

- cache bug diagnosis;
- whether it was fixed;
- best trusted model/configuration;
- recommended warm context;
- warm TTFT/decode;
- long-context result;
- persistent g128 path;
- FINAL_REPORT location;
- relevant commit SHAs.

Do not start a new open-ended branch after FINAL_REPORT.md is complete.
