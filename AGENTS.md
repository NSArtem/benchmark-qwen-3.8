# Qwen3.8 MLX Investigation — Stage 4 Project Instructions

This repository is an empirical engineering project for running Qwen3.8-27B
well on an 18 GB Apple Silicon Mac.

Stage 3 established that the exact-suffix persistent chat protocol is correct
and practical. Stage 4 therefore shifts focus from "can persistent chat work?"
to:

1. quality validation;
2. quantization quality/memory tradeoffs;
3. productionizing the validated runtime.

The goal is not to maximize novelty. The goal is to identify the best practical
Qwen3.8 configuration for daily local use.

---

## Hardware

Primary machine:

- MacBook Pro
- Apple M3 Pro
- 18 GB unified memory
- macOS 15.7.8

Priority order:

1. correctness;
2. system stability;
3. practical output quality;
4. persistent chat correctness;
5. useful context length;
6. follow-up latency;
7. decode throughput;
8. model size.

A smaller/faster quant is not a win if it materially damages real task quality.

---

## Canonical Environment

Use ONLY:

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

Do not silently upgrade inference dependencies.

A separate isolated comparison environment may be created only if TASK.md
explicitly calls for one. The canonical environment must remain unchanged.

---

## Canonical Models

### g64 reference / quality baseline

Path:

    /Users/artem/.cache/huggingface/hub/models--lukaskremla--Qwen3.8-27B-3bit-MLX-TextOnly/snapshots/c98bba5926f51fec1c8d8737e577221673f524d7

Quantization:

    affine 3-bit
    group_size = 64

Measured model payload:

    ~10.963 GiB

Treat g64 as the current quality/reference baseline.

### g128 performance/memory candidate

Path:

    /Users/artem/models/Qwen3.8-27B-3bit-g128-MLX

Quantization:

    affine 3-bit
    group_size = 128

Measured model payload:

    ~10.18 GiB

Approximate advantage vs g64:

    ~0.783 GiB smaller

Observed practical decode advantage in prior controlled runs:

    roughly 7-8% in several 4K-12K comparisons

g128 is the current practical performance candidate, but its quality relative
to g64 is NOT yet finally classified.

---

## High-Precision Source

A high-precision Qwen3.8-27B source snapshot was previously downloaded into the
Hugging Face cache.

Inspect existing conversion artifacts and local cache before downloading
anything.

A bounded-memory layer-oriented converter already exists in the repository:

    research/harness/convert_qwen38_textonly_g128.py

or a closely related path from previous stages.

Reuse/extend the existing bounded-memory infrastructure.

Do not write a completely separate model conversion system without first
inspecting what already exists.

Do not re-download ~52 GB of source weights if already present.

---

## Validated Persistent Chat Result

Stage 3 validated the exact-suffix protocol.

The critical control compared:

    A. fresh one-shot
    B. fresh segmented with the same model-call boundaries
    C. true persistent cache

For B vs C:

- token streams matched;
- all 64 cache layers matched;
- 48/48 ArraysCache matched;
- 16/16 KVCache matched;
- first-token logprobs matched exactly;
- generated turn outputs matched token-for-token.

Therefore:

> exact-suffix persistent state is validated against identical segmented
> execution.

A-vs-B/C differences are segmentation/numerical sensitivity, not evidence of
persistent cache corruption.

Do NOT re-open this issue without new contradictory evidence.

---

## Validated Persistent Performance

Representative Stage 3 g128 results:

    ~4K warm TTFT   ~1.0 s
    ~8K warm TTFT   ~1.1 s
    ~12K warm TTFT  ~1.1 s
    ~16.7K warm TTFT ~1.2 s

Representative decode:

    ~8K   ~11.1 tok/s
    ~12K  ~10.8 tok/s
    ~16K  ~10.4 tok/s

Representative peak memory:

    ~8K   ~13.98 GiB
    ~12K  ~14.61 GiB
    ~16K  ~15.24 GiB

The persistent protocol therefore already satisfies the main interactive
runtime objective.

Do not spend Stage 4 re-proving it from scratch.

---

## Stage 3 Endurance Result

An 8K persistent endurance conversation completed 16 sequential follow-up turns
without replay/cache/accounting failure.

Observed:

    warm TTFT ~1.55-1.62 s late in the session
    decode ~10.6-11.2 tok/s
    peak memory stable around the expected range
    no stale replay
    no cache-accounting error

At turn 17 the model emitted a very short incomplete response and EOS.

This is currently classified as:

    unresolved EARLY-EOS / generation-quality event

NOT:

    proven persistent-cache corruption

Stage 4 should perform one targeted reproduction/control study and then close
the question unless evidence points back to runtime state.

---

## Stage 3 g64 vs g128 Quality Result Is NOT Final

The Stage 3 quality suite was useful but flawed.

Known problems:

1. many coding tasks used a 64-token cap, so correct answers were often cut
   before required keywords/checks could appear;

2. at least one reasoning task had an arguably wrong/ambiguous expected answer;

3. some aggregated pass-rate differences therefore reflect harness design
   rather than model quality.

Existing evidence is mixed:

- both models produce coherent outputs on many coding prompts;
- one teacher-forced comparison showed a large NLL gap against g128;
- another code-heavy subset showed only a small gap;
- at least one clear arithmetic/reasoning error was observed from g128.

Therefore Stage 4 must build QUALITY SUITE V2 before making final quant decisions.

Do not claim:

    g128 == g64

or:

    g128 is materially worse than g64

until Stage 4's corrected suite is complete.

---

## Quantization Work Already Performed

Previous stages explored:

### 3-bit affine group 64

Working.
Current quality/reference baseline.

### 3-bit affine group 128

Working.
Saves ~0.783 GiB.
Main performance candidate.

### mixed_2_6

Rejected.

A source-backed mixed 2/6 model was created and loaded, but produced
deterministic pathological repetition / semantic collapse.

Do not retest unless a new mechanism directly explains why that exact result
was invalid.

### uniform 2-bit

Previously showed unacceptable quality.

Do not revisit in Stage 4.

### 4-bit class

Too large to be the default target on this 18 GB machine at useful long
contexts.

Do not spend Stage 4 rebuilding generic uniform 4-bit models.

### KV8

Validated as an independent long-context lever:

around 16K:

    ~0.465 GiB memory saving
    ~9-11% decode penalty
    warm TTFT nearly unchanged
    correctness passed

Treat KV8 as an optional context/headroom mode, not a weight-quant substitute.

---

## Relevant Upstream Quantization Options

Current mlx-lm supports:

- standard affine quantization;
- built-in mixed recipes:
    mixed_2_6
    mixed_3_4
    mixed_3_6
    mixed_4_6
- Dynamic Quantization;
- Distilled Weight Quantization (DWQ);
- Activation-aware Weight Quantization (AWQ);
- GPTQ.

Current upstream documentation describes:

Dynamic Quantization
- measures sensitivity of quantizable layers;
- uses higher precision for more sensitive layers;
- saves per-layer sensitivity results;
- is the fastest learned-quant option.

DWQ
- distills quantized weights/parameters against a higher-precision teacher;
- generally slower;
- often better quality;
- particularly relevant in the 2-4 bit range.

AWQ/GPTQ
- calibration-based;
- can require substantial runtime;
- not mandatory Stage 4 branches.

Use current upstream source as reference, but keep canonical runtime pinned.

Do not blindly upgrade mlx-lm to access a helper if the equivalent logic can be
adapted locally.

---

## Stage 4 Quantization Philosophy

Do not create a quant zoo.

The primary quantization question is:

> Can we recover g64-like practical quality while retaining most of g128's
> memory/speed advantage?

Preferred strategy:

    3-bit g128 base
    +
    selectively allocate more bits to the most sensitive tensors/modules

rather than:

    globally increase all weights.

Target region for a promising new candidate:

    roughly 10.3-10.6 GiB

This is a TARGET REGION, not a hard requirement.

A candidate is valuable if it spends only a fraction of the ~0.78 GiB saved by
g128 but materially recovers quality.

---

## Quant Candidate Limit

Stage 4 may create at most THREE serious new weight-quant candidates.

Preferred order:

1. one built-in/simple mixed 3/4 control;
2. one sensitivity-guided 3/4 candidate;
3. only if justified, one higher-precision-outlier candidate such as
   sensitivity-guided 3/6.

Do not create every recipe available.

A candidate that fails a mini quality gate should be rejected without full
benchmarking.

---

## Bounded-Memory Conversion Rule

Any new 27B conversion must use bounded-memory/layer-wise processing.

Do not load the complete high-precision model into unified memory.

Record:

    conversion wall time
    process RSS
    MLX peak
    active+cache when relevant
    output tensor payload
    shard count

Conversion safety matters.

If an upstream quant helper assumes loading the entire teacher/source model and
is unsafe on 18 GB, do not run it directly.

Instead:

- inspect the algorithm;
- determine whether sensitivity can be computed in a bounded/layer-wise way;
- adapt only the required logic.

---

## Quality Suite Rules

The quality suite must distinguish:

    OBJECTIVE CHECKS
    SEMI-OBJECTIVE RUBRIC
    SUBJECTIVE PAIRWISE JUDGMENT
    TEACHER-FORCED METRICS

Do not collapse all signals into one naive score.

Objective checks should dominate where possible.

Examples:

- arithmetic answer;
- JSON parse/schema validity;
- explicit instruction compliance;
- retrieval of supplied facts;
- code invariant identification;
- presence/absence of required structural elements.

Subjective model judging is supplementary only.

---

## Output-Length Rule

Do not repeat Stage 3's 64-token truncation mistake.

For coding/reasoning tasks use enough room to finish:

    usually 256-512 generated tokens

Use shorter caps only where the answer is intentionally short and the expected
answer fits comfortably.

A model must not be marked wrong merely because the harness truncated a
correctly progressing answer before its required conclusion.

Track finish reason / EOS / length cap.

---

## Learned-Quant Calibration Rule

Calibration data must be local and reproducible.

Do not depend on live web content.

Prefer:

- the Stage 4 quality-suite prompts;
- selected coding-agent prompt segments;
- representative prose/reasoning contexts;
- a compact fixed calibration corpus checked into the repo when licensing
  permits.

Record hashes/seeds.

Do not optimize on the exact full evaluation set and then pretend it is an
independent evaluation.

Separate:

    calibration subset
    evaluation subset

---

## Memory Measurement

For important experiments capture where practical:

    mx.get_active_memory()
    mx.get_cache_memory()
    mx.get_peak_memory()

Compute:

    active + cache

Also record RSS when useful.

Do not compare decimal GB and GiB as if identical.

Do not use one memory number from an old phase if a new candidate materially
changes weights.

---

## System Safety

NEVER modify:

    sudo sysctl iogpu.wired_limit_mb

or any equivalent system GPU/wired-memory setting.

Never explicitly increase MLX wired limit.

Canonical runtime leaves wired state untouched/default.

Hard stop conditions:

- Metal OOM;
- GPU timeout;
- WindowServer/UI instability;
- major machine unresponsiveness;
- repeated allocation failure;
- runaway unexplained memory growth.

Do not repeatedly reproduce an unsafe condition.

Never run two heavy Qwen inference/conversion jobs concurrently.

---

## Runtime Productionization Rule

Stage 4 may productionize the validated direct runtime.

The production runtime should preserve:

- exact canonical chat transcript;
- exact-suffix persistent cache;
- no forced wired-limit behavior;
- optional KV8;
- reset support;
- deterministic mode;
- practical sampling mode;
- metrics;
- safe failure handling.

Do not productionize:

- multi-user batching;
- concurrent request scheduling;
- server-wide prefix reuse;
- speculative cache trimming.

Those are separate upstream-hard problems.

A simple single-user CLI or single-request-at-a-time OpenAI-compatible server is
acceptable.

---

## Persistent Stage 4 Research Memory

Maintain:

    research/stage4/PLAN.md
    research/stage4/STATUS.md
    research/stage4/RESULTS.md
    research/stage4/results.jsonl
    research/stage4/FINAL_REPORT.md

Before every major phase re-read:

    AGENTS.md
    TASK.md
    research/stage4/STATUS.md

STATUS.md must contain:

    current phase
    completed experiments
    validated facts
    candidate models and sizes
    rejected candidates
    current best configuration
    active risks
    next action

---

## Git Discipline

Always begin with:

    git status
    git log --oneline -20

Preserve unrelated changes.

Never:

    git reset --hard
    git clean -fd
    discard unrelated modifications

Do not commit model weights or huge source checkpoints.

Commit:

- harnesses;
- quant recipes/configuration;
- sensitivity JSON if reasonably sized;
- result summaries;
- small representative outputs;
- runtime code;
- tests;
- documentation.

Prefer one logical commit per completed phase.

---

## Final Standard

Stage 4 should finish with a measured answer to:

1. Is g128 actually materially worse than g64 on a corrected quality suite?

2. If yes, can a targeted 3/4 or 3/6 mixed quant recover most of that quality
   while staying materially smaller/faster than g64?

3. What model should the user run by default TODAY?

4. Is the validated persistent runtime ready to be used as a normal local chat /
   coding-agent backend?

5. Which quantization ideas deserve a Stage 5, if any?

Do not continue open-ended exploration after these questions are answered.
