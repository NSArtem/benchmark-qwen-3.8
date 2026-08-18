# Qwen3.8 MLX Investigation — Project Instructions

This repository is an empirical investigation of running Qwen3.8-27B correctly
and efficiently on an 18 GB Apple Silicon Mac.

This file supersedes the previous AGENTS.md for the next unattended run. Keep
all old experiment artifacts; do not erase history.

## Primary goal

Find a correctness-safe, memory-safe, practically useful Qwen3.8-27B MLX
runtime for interactive coding-agent and long-conversation workloads on this
machine.

Optimize in this order:

1. correctness;
2. macOS / Metal stability;
3. useful interactive context;
4. warm-turn latency;
5. decode throughput;
6. cold TTFT;
7. maximum context.

A technically runnable context that risks UI instability is not a win.

## Hardware and canonical environment

Machine:

- MacBook Pro, Apple M3 Pro
- 18 GB unified memory
- macOS 15.7.8

Canonical MLX environment:

    envs/mlx-0.32.1

Always use:

    uv run --project envs/mlx-0.32.1 --no-sync ...

Expected:

    Python 3.14.7
    mlx == 0.32.1

Pinned mlx-lm revision:

    254d153fdeb6f150edd4fc5a54f9828638481fa8

Do not silently upgrade or modify the canonical environment.

Do not use:

    uvx
    uv sync
    uv lock
    uv update

for canonical experiments.

## Baseline g64 model

Original baseline checkpoint:

    lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly

Exact local snapshot used by prior work:

    /Users/artem/.cache/huggingface/hub/models--lukaskremla--Qwen3.8-27B-3bit-MLX-TextOnly/snapshots/c98bba5926f51fec1c8d8737e577221673f524d7

Quantization:

    affine
    bits = 3
    group_size = 64

Exact tensor payload measured previously:

    10.963 GiB

## Source-backed g128 candidate

A source-backed affine 3-bit/group-128 checkpoint was successfully produced
from the original high-precision Qwen3.8 source.

Previous temporary path:

    /private/tmp/qwen38-g128-mlx

Expected properties:

    affine
    bits = 3
    group_size = 128
    1,847 tensors
    19 shards
    tensor payload = 10.180 GiB

Measured saving vs g64:

    ~0.783 GiB
    ~7.14%

The high-precision source is expected at:

    /Users/artem/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0

The bounded converter is:

    research/harness/convert_qwen38_textonly_g128.py

Prior bounded conversion completed in roughly 25 seconds and kept converter
peak far below full-model memory. Prefer this converter over stock full-model
mlx_lm.convert for 27B conversion on this machine.

Do not re-quantize g64 into g128 and call it equivalent. Serious weight
candidates must come from the source weights.

## Proven g128 cold behavior

Source-backed g128 produced coherent fresh-process outputs at:

    short
    768
    4K
    8K
    12K
    16.7K

Representative cold measurements:

    4K     peak ~13.360 GiB, decode ~11.61 tok/s
    8K     peak ~13.983 GiB, decode ~11.20 tok/s
    12K    peak ~14.609 GiB, decode ~10.83 tok/s
    16.7K  peak ~15.236 GiB, decode ~10.64 tok/s

The 16.7K probe is evidence of correctness and headroom, not permission to
blindly scale much farther while unattended.

## Critical correction: previous 8K warm-cache conclusion is NOT valid

The prior final report classified both 4K and 8K persistent-cache chat as
EXCELLENT.

That conclusion must be corrected.

Inspection of the raw 8K token artifacts found:

    turn 2 generated token sequence == first 64 tokens of turn 1

exactly, token-for-token, despite a semantically different follow-up query.

Therefore:

- P3-WARM8K is NOT correctness-safe evidence;
- its low TTFT and ~10.65 tok/s decode numbers are performance observations
  only;
- do not use P3-WARM8K to recommend interactive 8K until cache semantics are
  understood and fixed;
- preserve the old artifacts as evidence; do not rewrite them away.

The old 4K warm run did not show the same exact replay and appears semantically
reasonable, but it must also be revalidated under the new correctness oracle.

The highest-priority problem for this run is persistent-cache correctness.

## No-wired runtime and wired-memory safety

The known-good inference path is:

    MLX 0.32.1
    direct generate_step
    MLX wired state untouched/default

Pinned mlx-lm's normal stream wrapper calls a wired-limit helper using the
recommended working-set size (~12 GiB on this machine). Controlled tests showed
that forcing this wired limit can cause Metal OOM around a ~768-token prefill,
while the equivalent direct/default-wired path passes.

Never call:

    mx.set_wired_limit(...)

in canonical experiments.

NEVER modify:

    sudo sysctl iogpu.wired_limit_mb

or any system-wide wired-memory setting.

A previous system wired-memory change caused a freeze/reboot.

The reusable local no-wired helper currently lives at:

    research/runtime/qwen38_runtime.py

It may contain a chat-cache protocol bug. Do not treat its chat mode as trusted
until this run validates it.

## Mixed quantization is closed

The corrected source-backed mixed_2_6/group-128 candidate had approximately:

    8.989 GiB tensor payload

but two fresh short runs produced the same pathological repeated-sentence loop
with identical token hashes.

Therefore mixed_2_6 g128 is REJECTED.

Do not:

- rerun it at 768/4K/long context;
- tune it tonight;
- try to rescue it with sampling;
- use its smaller memory as evidence of practical quality.

Uniform 2-bit and DWQ are also outside the primary scope of this run unless all
cache/runtime goals are complete and TASK.md explicitly reaches an optional
follow-up gate.

## KV8 status

Pinned Qwen3.8 has approximately:

    48 recurrent ArraysCache layers
    16 attention KVCache layers

Previous KV8/group64/start0 tests were coherent.

At 8K, g128 + KV8 measured approximately:

    -0.231 GiB peak vs normal g128
    ~7% slower decode

This is a poor tradeoff at 8K, but the saving should grow with context because
only the attention KV portion scales with context.

Therefore:

- KV8 is rejected for normal 4K/8K use;
- KV8 is NOT permanently rejected for 16K+ safety/headroom;
- long-context KV8 may be tested only behind TASK.md gates.

## Units matter

Prior logs mix decimal GB and binary GiB.

Never compare them without conversion.

Use binary GiB internally for new measurements:

    bytes / 2**30

When quoting an old decimal `GB` measurement, label it as decimal and, when it
matters for a safety comparison, convert it to GiB.

Do not casually say that 15.2 GiB is two full GiB below an old 17.2 GB result;
17.2 decimal GB is only about 16.0 GiB.

## System-stability policy

Swap alone is not failure.
Pageouts alone are not failure.

Hard stop conditions include:

- Metal OOM;
- GPU timeout;
- severe/repeated memory-pressure warnings;
- substantial machine unresponsiveness if observable;
- WindowServer/UI instability if observable;
- repeated allocation failure at the same configuration.

Because this is an unattended run, do not depend on a human noticing screen
flicker. Use conservative measured limits.

Do not repeatedly probe near the historical danger envelope merely to find an
absolute maximum.

## Heavy workload serialization

Never run two large Qwen MLX inference/conversion workloads concurrently.

Static source analysis, report generation, small CPU-only scripts, and file
inspection may be done while a heavy workload is unavailable.

If process inspection is denied by the sandbox, do not infer that no external
heavy process exists. Still serialize all heavy commands you launch yourself.

## Fresh-process and persistent-process rules

Cold benchmark:

    fresh Python process per configuration

Warm/chat benchmark:

    persistent process is allowed only when persistent cache behavior is the
    variable being tested

Do not contaminate cold comparisons with allocator/cache history.

## Correctness standard

Never classify only from a superficial text heuristic.

For cache experiments, correctness must include protocol-aware checks.

At minimum save:

- complete generated token IDs;
- raw decoded output;
- prompt/token construction metadata;
- generated token SHA256;
- overlap with previous turn output;
- semantic relation to the current user query;
- comparison against a fresh-process reference.

A follow-up that simply repeats the previous answer is BROKEN even if the text
is grammatical.

Classifications:

COHERENT:
- responds to the current user turn;
- normal linguistic progression;
- no pathological repetition/replay.

DEGRADED:
- partly relevant but visibly corrupted/replayed/confused.

BROKEN:
- token soup;
- pathological loops;
- exact or near-exact inappropriate replay of an earlier answer;
- response ignores the new turn because stale cache state dominates.

## Cache correctness oracles

For multi-turn debugging distinguish at least two references:

1. EXACT-TOKEN CONCATENATION ORACLE

   Reconstruct in a fresh process the exact token sequence that should have
   existed before the new generation:

       initial prompt tokens
       + generated assistant token IDs
       + next incremental prompt-fragment tokens

   This tests whether persistent cache state is equivalent to replaying the
   same token stream from scratch.

2. FULL-TRANSCRIPT CHAT ORACLE

   Re-serialize the logical conversation using the tokenizer chat template:

       user
       assistant
       user
       ...

   then generate in a fresh process.

   This tests whether the incremental turn protocol itself represents the
   intended conversation.

If persistent cache differs from exact-token concatenation, suspect cache-state
implementation/runtime behavior.

If persistent cache matches exact-token concatenation but both are wrong vs the
full-transcript oracle, suspect the turn serialization/protocol.

Do not conflate these failure classes.

## External source research

For current MLX/mlx-lm behavior use primary sources only:

- official MLX source/issues/PRs;
- official mlx-lm source/issues/PRs;
- official model/config metadata.

Do not upgrade the canonical runtime just because main contains a change.

A current-main fix may be copied/adapted into an isolated local experiment or a
separate disposable environment after source review.

Record exact commits.

## Git discipline

Begin with:

    git status
    git log --oneline -15

Preserve unrelated changes.

Never use:

    git reset --hard
    git clean -fd

Do not commit:

- HF source weights;
- converted 27B model weights;
- virtual environments;
- huge raw temporary files;
- unrelated files.

Commit small harnesses, logs/results, JSON, and reports when useful.

## Night-2 research state

Create a new namespace for this run:

    research/night2/

Use:

    research/night2/PLAN.md
    research/night2/STATUS.md
    research/night2/RESULTS.md
    research/night2/results.jsonl
    research/night2/FINAL_REPORT.md

Do not destroy the previous `research/` results.

Before every major phase re-read:

    AGENTS.md
    TASK.md
    research/night2/STATUS.md

Update STATUS after every meaningful experiment/decision gate.

## Unattended behavior

The user will be away for hours.

Do not stop for ordinary clarification.
Make conservative documented choices.

If one branch is blocked:

- record the blocker;
- move to another independent safe branch.

Do not violate safety constraints to keep working.

A good unattended run may end with a bug localized rather than fixed. Evidence
is more valuable than a pile of speculative changes.
