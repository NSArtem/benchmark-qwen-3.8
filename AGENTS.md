# Qwen3.8 MLX Investigation — Project Instructions

This repository is an empirical investigation of running Qwen3.8-27B correctly,
quickly, and interactively on an 18 GB Apple Silicon Mac.

The project is now past the exploratory "does the model run?" stage.

The primary engineering goal is:

> Build and validate a practical Qwen3.8-27B MLX runtime for long-lived,
> multi-turn coding-agent and conversational workloads on an 18 GB M3 Pro.

Treat saved artifacts and raw measurements as the source of truth.

Do not rely on conversational memory when an experiment can be inspected directly.

---

## Hardware

Primary machine:

- MacBook Pro
- Apple M3 Pro
- 18 GB unified memory
- macOS 15.7.8

The practical objective is NOT maximum context at any cost.

Prioritize, in order:

1. correctness;
2. system stability;
3. persistent/incremental chat correctness;
4. useful context length;
5. follow-up TTFT;
6. decode throughput;
7. cold-prefill throughput.

A configuration that technically reaches a larger context but makes normal
interactive use unpleasant is not the winner.

---

## Canonical MLX Environment

Use ONLY:

    envs/mlx-0.32.1

Canonical command prefix:

    uv run --project envs/mlx-0.32.1 --no-sync ...

Expected:

    Python 3.14.7
    mlx == 0.32.1

Pinned mlx-lm Git revision:

    254d153fdeb6f150edd4fc5a54f9828638481fa8

Do NOT:

    uv sync
    uv lock
    uv update
    uvx

Do not silently upgrade mlx, mlx-lm, transformers, tokenizers, or other
inference dependencies.

Upstream code may be inspected, copied into a local experimental helper, or
tested in an isolated environment when TASK.md explicitly asks for it.

The canonical environment itself must remain pinned unless TASK.md explicitly
creates a separate comparison environment.

---

## Canonical Models

### Quality/reference baseline: g64

Model:

    lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly

Exact local snapshot:

    /Users/artem/.cache/huggingface/hub/models--lukaskremla--Qwen3.8-27B-3bit-MLX-TextOnly/snapshots/c98bba5926f51fec1c8d8737e577221673f524d7

Quantization:

    affine
    bits = 3
    group_size = 64

Measured model payload:

    ~10.963 GiB

Use this as the quality/reference baseline.

### Main performance candidate: g128

Persistent path:

    /Users/artem/models/Qwen3.8-27B-3bit-g128-MLX

Quantization:

    affine
    bits = 3
    group_size = 128

Measured model payload:

    ~10.18 GiB

This model was converted from the original high-precision Qwen3.8-27B weights,
not by re-quantizing the g64 checkpoint.

Observed saving vs g64:

    ~0.783 GiB

This is currently the main practical candidate, but its quality relative to
g64 is NOT yet considered fully validated.

Do not move/delete either checkpoint.

---

## Original High-Precision Source

A high-precision Qwen3.8-27B source snapshot was previously downloaded into the
Hugging Face cache.

Inspect the local cache and previous conversion artifacts before assuming an
exact path.

Do not re-download it unless it is genuinely absent.

A bounded-memory conversion tool already exists in this repository from the
previous research phase.

Do not write another converter unless TASK.md explicitly requires it.

---

## Critical Wired-Memory Safety Rule

NEVER modify:

    sudo sysctl iogpu.wired_limit_mb

or any equivalent system-wide wired/GPU-memory setting.

A previous experiment changing the system iGPU wired limit froze/rebooted the
machine.

Do not recommend it.
Do not test it.
Do not automate it.

Also do not explicitly raise MLX's wired limit.

The canonical runtime leaves MLX wired state untouched/default.

---

## Known mlx-lm Wired-Limit Problem

The pinned mlx-lm generation wrapper uses a `wired_limit()` context that calls
approximately:

    mx.set_wired_limit(
        mx.device_info()["max_recommended_working_set_size"]
    )

On this machine that recommended working set is around 12 GiB.

This was experimentally shown to trigger Metal OOM for Qwen3.8 even around
~768 prompt tokens.

Controlled result:

    direct generate_step
    default wired state
    ~768 prompt tokens
        -> PASS

    same basic prefill
    wired limit forced to recommended working set
        -> Metal OOM

Therefore:

- do not use the stock forced-wired behavior for canonical Qwen3.8 experiments;
- use the existing no-wired helper / direct generate_step path;
- never call mx.set_wired_limit() in the canonical runtime;
- never alter the system wired limit.

---

## Known-Good Core Inference Path

Current known-good core path:

    MLX 0.32.1
    Qwen3.8
    direct generate_step
    default wired state
    greedy/argmax decoding
    no MTP
    no external speculative model
    no activation quantization

A prior controlled alignment test ran 18 independent fresh-process cases across
512/513, 640/641, and 768/769 final prompt tokens.

Result:

    18 / 18 COHERENT
    0 OOM
    deterministic token sequences within each condition

Therefore the old small-length alignment hypothesis is rejected.

---

## Long-Context Cold Baseline

A real coding-agent prompt around 16.7K tokens has been demonstrated coherent.

For the original g64 checkpoint, a manual no-wired run showed approximately:

    prompt = 16,767 tokens
    prompt throughput = 63.35 tok/s
    decode = 4.12 tok/s in that specific run
    MLX reported peak = 17.20 GB decimal
    output = coherent

Later controlled g128 runs were materially better.

Representative g128 cold measurements:

    4K peak      ~13.36 GiB
    8K peak      ~13.98 GiB
    12K peak     ~14.61 GiB
    ~16.7K peak  ~15.24 GiB
    ~16.7K decode ~10.6 tok/s

Treat units carefully:

    GB decimal != GiB

Do not compare them as if they were identical.

Historical UI flicker occurred in an older MLX 0.32.0 run near an extremely
high memory state. Do not reproduce old 0.32.0 stress tests.

---

## Persistent Chat: Current State

The old warm-chat harness was WRONG.

It effectively appended each follow-up as a fresh standalone chat-template
fragment, including a new BOS/user framing, on top of an already-cached
conversation.

At ~8K this produced a stale replay where turn 2 repeated the beginning of
turn 1 token-for-token.

A fresh process given the same malformed token stream reproduced the same stale
replay.

Therefore that original replay was explained by an invalid incremental token
protocol, not by persistent cache alone.

A later exact-suffix experiment fixed the protocol:

- build the full canonical transcript with the tokenizer;
- determine the exact logical token prefix already represented by cache;
- append ONLY the canonical missing suffix;
- do not inject a new BOS;
- do not template the new user message independently.

At ~4K the exact-suffix persistent path worked:

    initial prompt ~4K
    cached logical prefix after turn 1 = 4192 tokens in that experiment
    turn-2 exact suffix = 38 tokens
    persistent turn-2 TTFT ~0.98 s
    persistent decode ~11.6 tok/s
    fresh full-transcript turn-2 TTFT ~54.7 s
    stale replay = NO
    both outputs coherent

Persistent and fresh outputs shared the first ~42/64 generated tokens and then
diverged.

Cache-state summaries also differed after the full turn.

This divergence is NOT yet classified as a correctness bug.

The next required question is whether it is merely a normal segmentation /
floating-point sensitivity of GatedDeltaNet recurrent state or a true
persistent-cache state error.

---

## Persistent Cache Correctness Standard

Do not require token-identical output between:

    one-shot full prefill

and:

    an identical token stream split across multiple model calls

unless an experiment proves the runtime should be bitwise-equivalent.

For hybrid recurrent models, small numerical differences may eventually change
greedy output.

Correctness requires:

- same intended conversation token stream;
- correct message boundaries;
- response to the CURRENT user turn;
- no stale replay;
- no catastrophic corruption;
- no permanent loss of conversation state;
- stable long-session behavior.

However, when comparing two paths with IDENTICAL token stream and IDENTICAL
segmentation boundaries, state/output differences are much more suspicious and
must be investigated.

---

## Cache Types

Qwen3.8 uses a hybrid architecture.

The model has:

- full-attention layers with context-growing KV cache;
- recurrent / GatedDeltaNet layers using ArraysCache-like recurrent state.

Do not treat every layer as a standard KV cache.

When debugging cache behavior, inspect layer types separately.

Useful compact per-layer observations include:

    layer index
    cache class
    offset where available
    state shapes
    small deterministic hashes/signatures

Do not dump giant tensors into logs.

---

## Upstream Hybrid-Cache Context

Current upstream mlx-lm has known/open work around hybrid-model prompt cache
reuse and ArraysCache behavior.

Relevant historical/current references include:

    mlx-lm issue #980
        prefix cache reuse broken for hybrid architectures

    mlx-lm PR #1253 / #1254
        attempted ArraysCache trim/reuse support
        closed without merge

    mlx-lm issue #1641 / PR #1642
        ArraysCache metadata lazy-graph / Metal-buffer-object leak
        primarily relevant to BatchGenerator/server decode paths

    mlx-lm issue #1497
        proposed n-gram self-speculative decoding with recurrent-state rollback

Treat these as upstream evidence and ideas, not as automatically correct fixes
for our direct-generate exact-suffix runtime.

Before relying on upstream status, inspect current GitHub state.

Do not blindly cherry-pick a closed PR into the canonical environment.

---

## Memory Measurement

Do NOT use mx.get_peak_memory() alone.

For important experiments capture when practical:

    mx.get_active_memory()
    mx.get_cache_memory()
    mx.get_peak_memory()

Calculate:

    active + cache

When useful also capture:

    RSS / process footprint
    vm_stat
    memory_pressure

Do not require sudo.

Do not let monitoring itself materially distort benchmark performance.

Memory units must be explicit:

    bytes
    GB decimal
    GiB

---

## System Stability

Swap usage alone is NOT failure.
Pageouts alone are NOT failure.

Hard stop conditions:

- Metal OOM;
- GPU timeout;
- repeated severe system memory pressure;
- substantial machine unresponsiveness;
- WindowServer/UI instability if observable;
- repeated failed allocation at the same configuration;
- runaway memory growth without new information.

Do not repeatedly hammer a configuration that already failed.

---

## Heavy Workload Concurrency

Never run two large Qwen/MLX inference workloads concurrently.

Before starting a heavy phase inspect relevant running processes.

Static analysis, report writing, source inspection, and small CPU-only analysis
may proceed while another heavy job is active.

---

## Fresh-Process Rule

Cold benchmarks and quality comparisons should use fresh Python processes per
configuration unless a phase explicitly measures persistence.

Persistent chat/endurance experiments intentionally reuse one process/cache.

Do not mix cold and persistent measurements in the same result without labeling
them clearly.

---

## Correctness Classification

For generated text use:

COHERENT
- meaningful response to the actual task/current turn;
- normal syntax and progression;
- no pathological repetition.

DEGRADED
- partly meaningful but visibly corrupted/repetitive/structurally damaged.

BROKEN
- token soup;
- repeated punctuation;
- repeated one-token/substring loops;
- stale response replay;
- failure to respond to the current turn;
- structurally unusable output.

For structured-output tests also record schema/parse validity.

Technical imperfection is not catastrophic inference corruption.

---

## g128 Quality Status

g128 is currently promising but NOT yet fully quality-cleared.

Existing evidence is mixed:

- several generated coding outputs are coherent;
- one teacher-forced evaluation showed a large NLL/PPL degradation vs g64;
- another code-heavy comparison showed only a very small difference.

Therefore do not claim "g128 is quality-equivalent to g64" until a broader suite
is completed.

Likewise do not reject g128 based on one synthetic perplexity result.

Use diverse tasks and multiple metrics.

---

## KV8 Status

KV8 is an independent context-memory lever.

At around 16.7K on g128, one prior measurement was approximately:

    memory saving ~0.465 GiB
    decode penalty ~10-11%

Therefore KV8 is NOT interesting as a default for small contexts.

It may be worthwhile for 16K+ if the extra headroom materially improves system
stability or enables a larger practical context.

Do not use KV8 in quality/reference measurements unless a phase explicitly
tests it.

---

## Rejected / Low-Priority Weight Quantization Work

A mixed_2_6 g128-style candidate was created previously and produced
deterministic pathological repetition.

Treat that candidate as rejected.

Do not spend this stage on:

- mixed_2_6;
- uniform 2-bit;
- DWQ;
- MXFP4;
- NVFP4;
- another weight-quant zoo.

The current project bottleneck is runtime correctness/usability, not lack of
weight formats.

---

## Experimental Discipline

Every experiment must have:

1. a question;
2. controlled variables;
3. a predeclared decision gate;
4. raw artifacts;
5. an explicit comparison baseline;
6. a conclusion that distinguishes:
       measured fact
       inference
       hypothesis

Do not silently tune until something works.

Negative results are valuable and must be recorded.

---

## Persistent Research Memory

Maintain:

    research/stage3/PLAN.md
    research/stage3/STATUS.md
    research/stage3/RESULTS.md
    research/stage3/results.jsonl
    research/stage3/FINAL_REPORT.md

Before each major phase re-read:

    AGENTS.md
    TASK.md
    research/stage3/STATUS.md

STATUS.md should always contain:

    Current phase
    Completed experiments
    Key measurements
    Current best configuration
    Rejected configurations/hypotheses
    Current risks
    Next action

Do not rely on the conversational transcript as the sole project memory.

---

## Git Discipline

Always start with:

    git status

Preserve unrelated changes.

Never:

    git reset --hard
    git clean -fd
    discard unrelated modifications

Prefer focused commits after meaningful phases.

Do not commit:

- model weights;
- Hugging Face cache;
- virtual environments;
- huge raw temporary files;
- unrelated changes.

Commit:

- harnesses;
- small output samples;
- JSON/JSONL measurements;
- Markdown analysis;
- reproduction commands.

---

## Unattended Operation

This stage may run for 6-8 hours.

Do not stop for ordinary clarification questions.

If one branch is blocked, record why and continue to an independent useful phase
when safe.

Do not improvise around a safety rule.

Do not define success as "every planned phase ran".

Success means the highest-value uncertainties were reduced with reliable
evidence.

---

## Final Standard

The ideal end state of this stage is a measured answer to:

> Can Qwen3.8-27B on this 18 GB Mac sustain a correctness-safe 8K-16K
> persistent conversation where short follow-up turns have near-second-scale
> TTFT and ~10+ tok/s decode?

And:

> Is g128 good enough to be the practical default, or should g64 remain the
> quality mode?

Everything else is secondary.
