# Qwen3.8 MLX Investigation — Stage 5 Project Instructions

This repository is an empirical engineering project for making Qwen3.8-27B
a genuinely useful local model on an 18 GB Apple Silicon Mac.

Stage 3 established correctness and usability of exact-suffix persistent chat.

Stage 4 established that:
- g128 is much better than the earlier flawed quality harness suggested;
- mixed 3/4 is an interesting candidate but has a suspicious 16K decode cliff;
- Stage 4 RMS/module-output sensitivity was not a strong enough predictor of
  end-to-end model quality;
- qwen38-chat exists as a usable persistent CLI runtime.

Stage 5 is a larger optimization campaign.

The objective is NOT to create many quant formats.

The objective is to identify the best Pareto-optimal model/runtime combination
for daily use and to determine whether learned/loss-guided quantization and
decode acceleration can materially improve it.

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
4. persistent-chat correctness;
5. useful context capacity;
6. warm-turn TTFT;
7. decode throughput;
8. model size.

A quant that wins one benchmark but crosses a memory-residency cliff at useful
context is not a default winner.

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

Do not silently upgrade the canonical environment.

Current upstream source may be inspected and algorithms may be adapted locally.

If a later upstream feature requires a newer checkout for comparison, create an
isolated comparison environment/path. Do not mutate the canonical runtime.

---

## Canonical Model Roles Entering Stage 5

### g64 — reference

Path:

    /Users/artem/.cache/huggingface/hub/models--lukaskremla--Qwen3.8-27B-3bit-MLX-TextOnly/snapshots/c98bba5926f51fec1c8d8737e577221673f524d7

Quant:

    affine 3-bit
    group_size = 64

Payload:

    ~10.963 GiB

Current role:

    reference / historical quality baseline

Stage 4 did NOT find convincing evidence that g64 is practically better than
g128 on the corrected suite.

Do not assume g64 deserves to remain a production mode.

### g128 — current practical default candidate

Path:

    /Users/artem/models/Qwen3.8-27B-3bit-g128-MLX

Quant:

    affine 3-bit
    group_size = 128

Payload:

    ~10.180 GiB

Stage 4 Quality Suite V2:

    objective: 23/24
    prose: 4/4
    teacher-forced NLL approximately 1.369
    top-1 agreement vs reference approximately 69.46%

Representative runtime:

    ~8K:
        warm TTFT ~1.1 s
        decode ~11.1 tok/s
        peak ~13.98 GiB

    ~12K:
        warm TTFT ~1.1 s
        decode ~10.8 tok/s
        peak ~14.61 GiB

    ~16K:
        warm TTFT ~1.2 s
        decode ~10.4 tok/s
        peak ~15.24 GiB

Current role entering Stage 5:

    practical default candidate

### mixed34 control — quality candidate

Expected path:

    /Users/artem/models/Qwen3.8-27B-mixed-3-4-control-g128

or the exact Stage 4 path recorded in artifacts.

Payload:

    ~10.665 GiB

Effective BPW:

    ~3.406

Stage 4 Quality Suite V2:

    objective: 24/24
    prose: 4/4
    teacher-forced NLL approximately 1.386
    top-1 agreement approximately g64-level

But Stage 4 showed suspicious long-context decode:

    ~8K:
        ~10.7 tok/s

    ~16K:
        ~5.7 tok/s

with peak around:

    ~15.72 GiB

This performance cliff MUST be revalidated before mixed34 can be considered a
general default.

---

## High-Precision Source

A high-precision Qwen3.8-27B source snapshot is already cached locally.

Inspect Stage 1-4 artifacts and Hugging Face cache to recover its exact path and
revision.

Do not re-download the full source if present.

Previous measured source size was around 52 GiB on disk.

A bounded layer-oriented converter already exists.

Reuse and extend that infrastructure.

---

## Validated Persistent Runtime

Exact-suffix persistent chat is considered validated.

Do not re-open that investigation unless new evidence contradicts it.

The Stage 3 same-boundary oracle showed:

    fresh segmented == true persistent

across:

    48 ArraysCache layers
    16 KVCache layers
    first-token logits
    generated tokens

The runtime objective is therefore already achieved:

    long initial context
    + tiny exact suffix
    -> near-second-scale follow-up TTFT

Stage 5 uses this runtime as infrastructure, not as an open research question.

---

## Early-EOS Status

The Stage 3 endurance turn-17 short EOS was investigated in Stage 4.

It reproduced under fresh and segmented/persistent paths and was not associated
with cache-accounting corruption.

Treat it as:

    model/generation behavior

not:

    persistent-cache failure

Do not reopen unless a new quant or runtime systematically increases early-EOS
frequency.

---

## Quality Suite V2 Status

Stage 4 corrected the flawed 64-token evaluation approach.

The Stage 4 suite is useful and frozen for comparison, but it is no longer
sufficient as the only quality evidence because:

- mixed34 selection saw related evaluation/calibration information;
- the observed 24/24 vs 23/24 difference is only one objective case;
- Stage 5 will perform further quant optimization.

Therefore Stage 5 MUST add a completely held-out set that is not used for:

    calibration
    sensitivity
    candidate allocation
    mini-tournament selection

Do not modify held-out answers after observing model identities unless the test
is objectively invalid. If a test is corrected, rerun it for all models and
document the change.

---

## Previous Quantization Results

### Keep

- g64 affine 3-bit group64
- g128 affine 3-bit group128
- mixed34 control
- Stage 4 sensitivity artifacts
- bounded source converter
- Quality Suite V2

### Reject / do not reopen

- uniform 2-bit
- mixed_2_6
- generic quant zoo
- repeated uniform 4-bit
- blind low-bit experiments

mixed_2_6 previously produced deterministic semantic collapse/repetition.

Do not re-run it.

---

## Stage 4 Sensitivity Result

Stage 4 created a module-output/RMS-style sensitivity map.

It found high sensitivity in several late attention/recurrent/MLP projections,
but the resulting sensitivity-guided candidate did not outperform the simpler
mixed34 control.

Therefore:

> local module reconstruction distortion is not a sufficient optimization
> target.

Stage 5 must prioritize:

    end-to-end loss
    KL/logit divergence
    gradient-informed sensitivity
    quality recovery per added byte

Do not reuse the Stage 4 ranking as if it were ground truth.

---

## Upstream Learned Quantization Context

Current mlx-lm exposes learned quantization families including:

    Dynamic Quantization
    DWQ
    AWQ
    GPTQ

Dynamic Quantization is especially relevant because it uses calibration data to
identify sensitive modules and allocate higher precision.

Recent upstream work/issues also indicate that:

- target-BPW/threshold allocation can have edge cases near the low end;
- sensitivity/allocation may be affected by numerical/reduction noise.

Therefore Stage 5 should prefer:

    explicit fixed budget
    deterministic saved ranking
    explicit module list

over:

    repeatedly searching a threshold until a target BPW happens to appear.

Inspect current upstream implementation before adapting it.

Do not trust remembered details if source can be checked.

---

## Stage 5 Quantization Philosophy

The main target is:

> maximize quality recovery per extra byte above g128.

Preferred base:

    affine 3-bit group128

Promote selected modules to:

    4-bit
    or, for a very small set of outliers, 5/6-bit if supported and justified

The key metric is conceptually:

    quality recovery
    ----------------
       bytes added

Use a real end-to-end proxy such as:

    held-out/calibration KL improvement
    NLL improvement
    logit recovery

not only module-output RMS.

---

## Fixed-Budget Frontier

Do not create uncontrolled candidates.

Preferred Stage 5 budget ladder relative to g128:

    Q0 = g128 baseline

    Q1 ~ +150 MiB

    Q2 ~ +300 MiB

    Q3 ~ +450 MiB

Exact sizes may differ depending on module granularity.

At most FOUR new serious Stage 5 weight candidates may be created in the main
frontier.

This limit excludes tiny debug/synthetic test artifacts but includes real 27B
checkpoints.

At a matched size, one experiment may compare:

    broad 3->4 promotion

vs

    sparse 3->5/6 promotion

if supported safely.

Do not generate dozens of threshold variants.

---

## Bounded-Memory Conversion

All 27B source-backed conversion must remain bounded/layer-oriented.

Never load the complete high-precision source into unified memory.

For each conversion record:

    source revision
    recipe/ranking hash
    promoted modules
    payload bytes
    effective BPW
    wall time
    peak MLX memory
    RSS
    shard count

Do not commit weights.

Do not duplicate source weights unnecessarily.

---

## DWQ Safety

DWQ is allowed in Stage 5, but only if implemented or adapted safely.

A naive simultaneous teacher + student 27B setup is not acceptable on 18 GB.

Before real DWQ:

1. inspect current implementation;
2. calculate expected resident memory;
3. determine whether teacher/student can be streamed/layered;
4. perform a small dry run;
5. measure memory.

Preferred concept:

    bounded / layer-wise distillation

Possible teachers:

    high-precision source
    memory-safe 8-bit teacher
    g64 reference

but only if teacher/student residency is controlled.

The goal is not to reproduce upstream CLI behavior at any cost.

A useful outcome may be:

    proof that bounded DWQ is feasible

or:

    proof that it is not worth implementing on this hardware.

Do not burn the whole research window fighting an unsafe training path.

---

## Activation Quantization

Activation quantization is allowed as a runtime experiment only after a weight
winner exists.

The relevant question is:

> can activation quantization reduce the long-context residency cliff without
> unacceptable speed/quality cost?

It is not automatically beneficial.

Test it only on the selected winner(s), primarily around 16K.

---

## KV8

KV8 remains a separate context-memory lever.

Prior result around 16K on g128:

    saves ~0.465 GiB
    costs ~9-11% decode
    warm TTFT nearly unchanged

Do not use KV8 in ordinary quality measurements.

Use it for 20K/long-context headroom tests if justified.

---

## MTP

MTP is allowed as a gated Stage 5 branch.

It is potentially high value because decode is now the major remaining runtime
bottleneck.

Before using MTP:

- inspect the original Qwen3.8 source/config;
- verify whether compatible MTP weights exist;
- identify whether text-only conversion previously excluded them;
- inspect current mlx-lm support/state;
- quantify added model memory;
- preserve canonical runtime separately.

Do not assume Qwen3.5/Qwen3.6 MTP implementation is directly compatible with
Qwen3.8.

Do not fabricate or train an MTP head.

If source weights/support are absent or incompatible, record and stop.

---

## Context Expansion

Stage 5 may test:

    20K

and, only behind strong gates:

    one 24K smoke

Do not test 32K.

The goal is practical context, not a record.

A larger context is only useful if:

    system remains stable
    warm TTFT remains good
    decode remains useful
    memory has reasonable OS headroom

---

## Performance Cliff Investigation

The mixed34 16K cliff is a P0 Stage 5 question.

We need to determine whether decode degradation correlates with:

    model payload
    active+cache
    peak residency
    context length
    system memory pressure
    Metal allocation behavior

Do not assume the Stage 4 result was either noise or a universal memory cliff.

Re-measure it.

Interleave models/runs to reduce thermal/background bias.

Use fresh processes.

---

## System Safety

NEVER modify:

    sudo sysctl iogpu.wired_limit_mb

Never explicitly raise MLX wired limits.

Never call mx.set_wired_limit() in the canonical runtime.

Hard stop:

- Metal OOM;
- GPU timeout;
- severe UI/WindowServer instability;
- major unresponsiveness;
- repeated allocation failure;
- dangerous unexplained memory growth.

Do not repeatedly reproduce unsafe failures.

Never run two large Qwen jobs concurrently.

---

## Measurement

Important inference runs should record when practical:

    prompt tokens
    suffix tokens
    generated tokens
    finish reason
    TTFT
    prompt tok/s
    decode tok/s

Memory:

    mx.get_active_memory()
    mx.get_cache_memory()
    active + cache
    mx.get_peak_memory()
    process RSS

For cliff studies also record:

    memory_pressure output
    vm_stat summary

where practical.

Use explicit units:

    bytes
    decimal GB
    GiB

---

## Runtime / Server Scope

Stage 4 produced qwen38-chat.

Stage 5 may add a productionized serialized API/backend suitable for a real
coding-agent harness.

Allowed:

    single user
    one active inference at a time
    persistent session
    OpenAI-compatible chat endpoint
    tool messages
    reset/session boundaries
    metrics

Out of scope:

    concurrent batching
    multi-user shared prefix cache
    arbitrary cache trimming
    server-wide cache reuse across unrelated clients

Keep the runtime simple and correctness-first.

---

## Research Memory

Maintain:

    research/stage5/PLAN.md
    research/stage5/STATUS.md
    research/stage5/RESULTS.md
    research/stage5/results.jsonl
    research/stage5/FINAL_REPORT.md

Before every major phase re-read:

    AGENTS.md
    TASK.md
    research/stage5/STATUS.md

STATUS.md must include:

    current phase
    completed experiments
    current Pareto frontier
    current default candidate
    candidate model paths/sizes
    held-out suite hash
    rejected branches
    safety observations
    next action

The user may be away for the entire run.

Do not rely on chat history as project memory.

---

## Git Discipline

Start with:

    git status
    git log --oneline -30

Preserve unrelated changes.

Never:

    git reset --hard
    git clean -fd
    discard unrelated modifications

Do not commit:

    model weights
    source model cache
    virtual environments
    huge transient logs

Commit:

    harnesses
    candidate recipes/module lists
    sensitivity rankings
    held-out suite
    small output samples
    JSON/JSONL results
    runtime/server code
    tests
    reports

Use focused commits.

---

## Autonomy

This stage is expected to run for roughly 15 hours.

Do not ask routine clarification questions.

Use decision gates.

If a branch is blocked or unsafe:

    record it
    preserve evidence
    move to an independent useful phase

Do not spend hours forcing one failed idea.

Do not run every optional phase simply because it is listed.

---

## Final Standard

Stage 5 should answer:

1. Is mixed34's 16K performance cliff real and reproducible?

2. Does mixed34 really outperform g128 on a genuinely held-out quality set?

3. Can loss/logit-guided fixed-budget allocation beat both g128 and mixed34?

4. Can bounded DWQ improve the best low-bit candidate enough to matter?

5. What is the final Pareto frontier?

6. What model is the production default?

7. What is the practical context recommendation:
       12K
       16K
       20K
       other measured value

8. Can activation quantization help long-context residency?

9. Is MTP available and worthwhile for Qwen3.8?

10. Is there a usable serialized local agent backend?

Do not start Stage 6 automatically.
