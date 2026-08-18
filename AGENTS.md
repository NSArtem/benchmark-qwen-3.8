# Qwen3.8 MLX Investigation — Project Instructions

This repository is an empirical investigation of running Qwen3.8-27B efficiently
and correctly on an 18 GB Apple Silicon Mac.

Treat measurements and saved artifacts as the source of truth.

Do not rely on memory when an earlier experiment can be inspected directly.

## Hardware

Primary test machine:

- MacBook Pro
- Apple M3 Pro
- 18 GB unified memory
- macOS 15.7.8

The practical goal is not merely "can the model technically run".

The goal is:

> Find a correctness-safe, memory-safe, practically useful Qwen3.8-27B MLX
> configuration for interactive coding-agent and long-conversation workloads
> on this 18 GB Mac.

Prioritize:

1. coherent output;
2. system stability;
3. useful context length;
4. TTFT / warm-turn latency;
5. decode throughput.

Synthetic maximum context is secondary.

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

Do not:

    uv sync
    uv lock
    uv update
    uvx

Do not silently upgrade mlx, mlx-lm, transformers, or other core runtime
dependencies.

Do not modify:

    envs/mlx-0.32.1/pyproject.toml
    envs/mlx-0.32.1/uv.lock
    root pyproject.toml
    root uv.lock

unless a task explicitly requires it.

## Canonical Existing Model

Current baseline checkpoint:

    lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly

Exact local snapshot:

    /Users/artem/.cache/huggingface/hub/models--lukaskremla--Qwen3.8-27B-3bit-MLX-TextOnly/snapshots/c98bba5926f51fec1c8d8737e577221673f524d7

Known quantization:

    mode = affine
    bits = 3
    group_size = 64

This is weight-only MLX affine group quantization.

Do not replace this checkpoint when establishing baseline results.

## Critical Safety Rule: Wired Memory

NEVER modify:

    sudo sysctl iogpu.wired_limit_mb

or any other system-wide wired-memory setting.

A previous experiment changing the system iogpu wired limit caused the Mac to
freeze/reboot.

Do not recommend it.
Do not test it.
Do not automate it.

Also do not explicitly increase MLX's wired limit.

## Known mlx-lm Wired-Limit Problem

The pinned mlx-lm implementation wraps normal stream generation in a helper
which calls approximately:

    mx.set_wired_limit(
        mx.device_info()["max_recommended_working_set_size"]
    )

On this machine the recommended working set is approximately 12 GiB.

This has been experimentally shown to cause a Metal OOM for this Qwen3.8 model
even around a ~768-token prompt.

Controlled result:

    direct generate_step
    default wired state
    ~768 prompt tokens
        -> PASS

    same basic prefill
    wired limit forced to recommended ~12 GiB
        -> Metal OOM

The OOM occurred during prompt-cache-state evaluation.

Therefore:

- do not use the stock wired-limit behavior for Qwen3.8 experiments;
- leave MLX wired state untouched/default;
- never call mx.set_wired_limit() in the canonical runtime path.

## Canonical Correctness Runtime

The current known-good baseline is:

    MLX 0.32.1
    direct generate_step
    default wired state
    greedy decoding
    no speculative decoding
    no MTP
    no APC
    no activation quantization

The controlled alignment experiment tested:

    final prompt    main prefill
    512             511
    513             512
    640             639
    641             640
    768             767
    769             768

Each was run three times in fresh processes.

Result:

    18 / 18 COHERENT
    0 DEGRADED
    0 BROKEN
    0 OOM

All three repeats at each length produced identical complete 128-token output
token sequences.

Simple 512/640/768 prefill alignment therefore did NOT explain the historical
corruption.

## Important Real-World 16K Result

A real coding-agent prompt was run manually using MLX 0.32.1 with the normal
mlx_lm.generate implementation except that the mlx_lm wired_limit context
manager was monkey-patched to a no-op.

Prompt file:

    qwen38-mtp-smoke/coding-prompt.txt

Observed result:

    Prompt: 16767 tokens
    Prompt throughput: 63.348 tok/s
    Generation: 256 tokens
    Decode throughput: 4.118 tok/s
    Peak MLX memory: 17.200 GB
    Output: coherent
    OOM: no

The generated answer correctly understood the Lantern repository/task.

This is strong evidence that:

- the current checkpoint can work at ~16.8K;
- MLX 0.32.1 can work at ~16.8K;
- the 3-bit g64 quant itself is not intrinsically broken at long context;
- the forced mlx-lm wired limit is a distinct problem.

However, 17.2 GB reported MLX peak on an 18 GB machine is NOT comfortable
headroom.

Do not repeatedly reproduce a ~17+ GB state without a good reason.

## Historical 0.32.0 Result

MLX 0.32.0 is retired from active investigation.

Do not run it.

It previously completed a ~16.7K run but produced corrupted output and caused
visible macOS screen/UI flicker near ~17.1 GB reported MLX peak.

Do not reproduce this.

Use historical artifacts only.

## GGUF Control

A llama.cpp GGUF control exists and demonstrated that the model architecture
itself can operate at long context.

Known useful control:

    unsloth/Qwen3.8-27B-GGUF:UD-IQ3_XXS

Representative results:

    16K:
        prefill ~52.7 tok/s
        decode ~4.39 tok/s

    24K:
        prefill ~42.5 tok/s
        decode ~3.91 tok/s

    32K:
        prefill ~28.3 tok/s
        decode ~3.36 tok/s

Do not rerun GGUF unless explicitly required by the task.

## Memory Measurement

Do NOT use mx.get_peak_memory() alone.

For every important run capture when practical:

    mx.get_active_memory()
    mx.get_cache_memory()
    mx.get_peak_memory()

Also calculate:

    active + cache

The MLX allocator may retain freed buffers in its cache, so allocator pressure
can be materially higher than active/peak numbers alone suggest.

Capture OS-level memory information when practical without sudo:

    ps
    vm_stat
    footprint
    memory_pressure

Do not make monitoring itself materially affect benchmark performance.

## System Stability

Swap usage alone is NOT failure.
Pageouts alone are NOT failure.

Hard stop conditions include:

- Metal OOM;
- GPU timeout;
- repeated severe OS memory-pressure indications;
- substantial system unresponsiveness;
- WindowServer/UI instability if observable;
- repeated failed allocation at the same configuration.

Because unattended runs cannot reliably observe UI flicker, be conservative
once measured memory approaches previously dangerous territory.

Never repeatedly hammer a configuration that has already failed.

## Heavy Workload Concurrency

Do not run two large MLX/Qwen inference or conversion workloads concurrently.

At startup inspect relevant processes.

If another benchmark from this repository is actively using the model, do not
start a competing heavy Metal workload.

Static analysis, source inspection, report generation, and size calculations
may proceed while waiting for heavy inference to become available.

## Fresh-Process Rule

For cold correctness/context benchmarks, use a fresh Python process per
configuration.

Do not load the model once and then use that process to compare unrelated cold
configurations.

Persistent caches are allowed only in experiments explicitly measuring
interactive/warm-cache behavior.

## Output Correctness

Do not judge correctness from only the first token.

For important smoke tests generate enough output to detect delayed corruption.

Prefer:

    128 tokens for context-ladder probes
    256 tokens for final candidate validation

Classification:

COHERENT:
- meaningful response;
- normal linguistic/syntactic progression;
- no pathological repetition.

DEGRADED:
- meaningful content mixed with obvious corruption/repetition.

BROKEN:
- token soup;
- repeated punctuation;
- repeated one-token/word loops;
- meaningless fragments;
- pathological substring loops.

Technical imperfection is not catastrophic inference corruption.

Save raw output and token IDs.

## Experimental Discipline

For every experiment:

1. state the question;
2. change as few variables as possible;
3. define the decision gate before running;
4. save raw measurements;
5. compare against the appropriate baseline;
6. record negative results;
7. do not silently tune parameters until something works.

Never reinterpret a failed experiment as success merely because a later
configuration worked.

## External Research

Prefer primary sources for MLX / mlx-lm behavior:

- MLX source;
- mlx-lm source;
- official repository issues and pull requests;
- model config/model card.

When using current upstream behavior, record the commit or source revision.

Do not upgrade the canonical runtime merely because current upstream differs.

Upstream code may be inspected or adapted into a local experimental helper.

## Quantization Rules

Do not casually re-quantize the existing 3-bit model into another quant and
treat it as equivalent to conversion from the original higher-precision
weights.

For a serious quality comparison, new weight quantizations should be produced
from the appropriate original/high-precision source whenever practical.

If an experiment must quantize from already-quantized weights, label it
explicitly as a compounded-quantization experiment and do not use it for final
quality conclusions.

## Conversion Safety

Do not blindly launch a full ~27B conversion just to see whether it OOMs.

Before conversion:

- identify source weights precisely;
- determine whether they are already local;
- check free disk space;
- estimate final output size;
- inspect conversion memory behavior;
- choose stock vs streaming/chunked conversion deliberately.

If a large source-model download is necessary, first verify that there is
comfortable free disk space for source weights, output weights, temporary
files, and existing model caches.

Do not delete unrelated model caches automatically.

## Git Discipline

Always begin with:

    git status

Preserve unrelated user changes.

Never:

    git reset --hard
    git clean -fd
    discard unrelated modifications

Commit logical completed phases separately when appropriate.

Use focused commit messages.

Do not commit:

- Hugging Face model weights;
- large generated model files;
- cache directories;
- virtual environments;
- giant raw temporary files;
- unrelated changes.

Experiment scripts, small logs, JSON results, Markdown reports, and small text
outputs may be committed when useful.

## Persistent Research Memory

Long-running work must maintain external state in the repository.

Use:

    research/PLAN.md
    research/STATUS.md
    research/RESULTS.md
    research/results.jsonl
    research/FINAL_REPORT.md

Before each major phase, re-read:

    AGENTS.md
    TASK.md
    research/STATUS.md

Update STATUS.md after every meaningful experiment or decision gate.

Append machine-readable measurements to results.jsonl.

Do not rely on the conversational context as the sole memory of the experiment.

## Unattended Operation

The user may be away for hours.

Do not stop merely to ask a nonessential clarification.

Resolve ordinary ambiguities conservatively and document the choice.

If one branch becomes blocked:

- record why;
- continue with an independent useful phase where safe.

Do not improvise around safety constraints.

Do not perform destructive operations merely to keep making progress.

## Final Standard

A successful evening is NOT defined by executing every planned phase.

A successful evening means:

- reliable measurements;
- preserved system stability;
- useful negative results;
- a clearly improved understanding of the memory/context envelope;
- and, if supported by evidence, a better practical runtime configuration.

Stop unsafe or unproductive branches early.