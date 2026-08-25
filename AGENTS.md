# Qwen3.8 MLX Investigation — Stage 6 Project Instructions

This repository is an empirical engineering project for making Qwen3.8-27B
a practical local model on an 18 GB Apple Silicon Mac.

Stage 6 starts from an unusual position:

- the persistent runtime is already validated;
- g128 is already a strong practical default;
- Stage 5 produced a much better end-to-end KL-guided quantization ranking;
- three fixed-budget candidate checkpoints (Q1/Q2/Q3) were physically created;
- Stage 5 then stopped because macOS swap pressure became too high to continue
  trustworthy heavy benchmarking.

Stage 6 must therefore NOT begin by inventing new candidates.

Its first job is to exploit and evaluate the expensive Stage 5 artifacts under
a clean post-reboot memory state.

The main engineering questions are:

1. Does Q1 actually beat g128 on the practical Pareto frontier?
2. If not, does Q2 or Q3?
3. Is the Stage 5 KL-guided ranking genuinely predictive and reproducible?
4. Can bounded DWQ materially improve the winning low-bit model?
5. Can MTP materially improve decode without destroying the memory advantage?
6. Is the serialized agent backend ready for real use?
7. What is the final recommended context/runtime configuration?

Do not treat Stage 6 as another open-ended quantization search.

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
3. practical quality;
4. sustained decode throughput;
5. useful context;
6. warm TTFT;
7. model size.

Stage 5 demonstrated that a model can look good at 8K and still hit a serious
decode cliff at ~16K.

Therefore long-context decode is a first-class selection criterion.

---

## Canonical Environment

Use ONLY:

    envs/mlx-0.32.1

Canonical prefix:

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

Do not silently upgrade canonical MLX/mlx-lm.

Current upstream may be inspected for algorithm/reference purposes.

If a newer implementation must be experimentally compared, isolate it outside
the canonical environment.

---

## System Safety

NEVER modify:

    sudo sysctl iogpu.wired_limit_mb

Never call:

    mx.set_wired_limit()

in the canonical runtime.

Do not change any equivalent GPU/wired-memory system setting.

Hard stop conditions:

- Metal OOM;
- GPU timeout;
- WindowServer/UI instability;
- major machine unresponsiveness;
- repeated allocation failure;
- clearly exhausted swap/residency state;
- uncontrolled memory growth.

Never run two large Qwen/MLX jobs concurrently.

---

## Clean-Boot Requirement

Stage 5 ended under severe swap pressure.

Observed near the stop:

    encrypted swap used ~12.2 GiB
    swap free ~1.1 GiB

Stage 6 benchmark conclusions are only trustworthy if the machine starts from a
clean memory state.

At Stage 6 start:

1. inspect memory_pressure / vm_stat / swap usage;
2. confirm no stale heavy MLX/Qwen processes;
3. if the user has rebooted, record the clean baseline;
4. if the machine is still in a badly polluted swap state, do not perform
   headline benchmark comparisons until the state is clean.

Do not try to "fix swap" with unsafe sysctls.

---

## Existing Canonical Models

### g128 — current production default

Path:

    /Users/artem/models/Qwen3.8-27B-3bit-g128-MLX

Payload:

    ~10.180 GiB

Quant:

    affine 3-bit
    group_size 128

Validated approximate runtime:

    8K:
        warm TTFT ~1.1 s
        decode ~11.1 tok/s
        peak ~13.98 GiB

    12K:
        warm TTFT ~1.1 s
        decode ~10.8 tok/s
        peak ~14.61 GiB

    16K:
        warm TTFT ~1.2 s
        decode ~10.4-10.6 tok/s
        peak ~15.24 GiB

Quality evidence entering Stage 6:

    approximately equal to g64 on corrected Stage 4/5 evidence

Current role:

    DEFAULT unless Stage 6 proves a better candidate.

### g64 — reference

Path:

    /Users/artem/.cache/huggingface/hub/models--lukaskremla--Qwen3.8-27B-3bit-MLX-TextOnly/snapshots/c98bba5926f51fec1c8d8737e577221673f524d7

Payload:

    ~10.963 GiB

Role:

    reference only unless new evidence unexpectedly shows a practical advantage.

### mixed34 — long-context cautionary control

Exact Stage 4/5 path must be recovered from artifacts.

Payload:

    ~10.665 GiB

Stage 5 revalidated approximately:

    16.7K decode ~4.9 tok/s

while:

    g128 ~10.6 tok/s

with similar prefill throughput.

Therefore mixed34 is NOT the current default.

Keep it as a control for:

    quality
    memory-cliff behavior
    bit-allocation comparisons

Do not spend Stage 6 trying to rescue mixed34 unless a new mechanism directly
explains the cliff.

---

## Stage 5 Q1/Q2/Q3 Candidates

Stage 5 physically created three KL-guided candidates.

Expected temporary paths were:

    /private/tmp/qwen38-stage5-Q1-kl-g128
    /private/tmp/qwen38-stage5-Q2-kl-g128
    /private/tmp/qwen38-stage5-Q3-kl-g128

The user may have moved them to persistent storage before Stage 6.

Recover exact paths from:

    research/stage5/STATUS.md
    research/stage5/RESULTS.md
    research/stage5/results.jsonl
    research/stage5/FINAL_REPORT.md
    filesystem inspection

Approximate properties:

### Q1

    +150 MiB vs g128
    payload ~10.327 GiB
    effective BPW ~3.298
    ~112 promoted 4-bit modules

Stage 5 short smoke:

    coherent
    decode ~11.7 tok/s in that specific short test
    peak ~10.64 GiB

This was NOT a complete performance/quality validation.

### Q2

    +300 MiB
    payload ~10.473 GiB
    effective BPW ~3.345
    ~137 promoted modules

Created but not properly evaluated.

### Q3

    +450 MiB
    payload ~10.619 GiB
    effective BPW ~3.392
    ~161 promoted modules

Created but not properly evaluated.

These candidates are the highest-priority Stage 6 artifacts.

Do not recreate them if intact.

---

## Candidate Preservation Rule

Before any heavy benchmark:

- verify Q1/Q2/Q3 directories exist;
- verify config/tokenizer/weights are complete;
- calculate payload size;
- optionally hash manifests/recipes;
- if candidates still live under /private/tmp and persistent disk space is
  available, move/copy them to a stable user model directory before proceeding.

Preferred stable form:

    /Users/artem/models/stage5-frontier/...

Do not delete or overwrite them until Stage 6 final conclusions are written.

Do not commit weights to git.

---

## Stage 5 KL-Guided Sensitivity Result

Stage 5 performed an expensive end-to-end sensitivity sweep:

    498 / 498 quantizable modules
    12 calibration cases
    ~477 target tokens
    ~4579 seconds wall
    peak ~14.48 GiB

The method evaluated the end-to-end logit/KL effect of temporarily promoting a
module from 3-bit to 4-bit relative to the g64/reference teacher.

This was substantially different from Stage 4 RMS sensitivity.

Observed Stage 4-vs-Stage 5 ranking relation:

    top-25 overlap ~1 / 25
    Spearman around -0.12

This strongly suggests Stage 4 local-RMS sensitivity was optimizing the wrong
proxy.

Stage 5 aggregate evidence suggested linear-attention modules were especially
important.

Approximate mean KL recovery by family:

    linear_attn > self_attn > MLP

with several high-ranked:

    linear_attn.in_proj_a
    linear_attn.in_proj_b

especially in selected mid/late layers.

Treat exact rankings as artifact-derived facts; inspect the saved JSON rather
than relying on this summary.

---

## Sensitivity Reproducibility Caveat

Stage 5 reported a low/modest ranking correlation from a second smaller run, but
that run used a DIFFERENT calibration subset.

Therefore it did NOT prove numerical nondeterminism.

It mostly measured calibration-set dependence.

Stage 6 should separately test:

1. same calibration set, same modules, repeated measurement;
2. different calibration subset robustness.

Do not conflate these.

---

## Stage 5 Held-Out Quality Caveat

The Stage 5 held-out suite was useful but still contained some max-token
truncation artifacts.

Examples found in raw inspection:

- a model computed the correct arithmetic result but failed because generation
  stopped before an obligatory `Final: ...` sentinel;
- multiple models reached the correct version/state but were truncated before
  printing the exact final marker.

Therefore raw formal scores such as:

    17/21
    18/21

must NOT be treated as unquestionable quality differences.

Stage 6 must repair the affected scoring/generation budgets BEFORE using this
suite to decide Q1/Q2/Q3.

Do not rebuild the entire held-out set unnecessarily.

Correct the known truncation defects symmetrically for ALL models.

---

## Persistent Runtime

Exact-suffix persistent chat is validated.

Do not re-investigate cache correctness.

Use existing:

    research/runtime/qwen38_chat.py

or the current renamed/equivalent runtime.

The runtime must:

- preserve canonical chat template;
- use exact suffix;
- never force wired-limit settings;
- keep one conversation/cache;
- support reset.

---

## Agent Backend

Stage 5 created:

    research/runtime/qwen38_agent_server.py

or an equivalent final path.

It includes a serialized OpenAI-compatible API and protocol/runtime tests.

Expected functionality:

    /v1/chat/completions
    /v1/models
    /health
    /metrics
    session reset
    system/user/assistant/tool roles
    single active inference

Stage 5 had protocol/unit tests but did NOT complete a full real 27B multi-turn
agent integration smoke because of the swap safety gate.

Stage 6 should validate it for real.

Do not add concurrency/batching.

---

## Stage 5 DWQ Status

Naive whole-model DWQ is not memory-safe.

Approximate impossible/simple residency examples:

    BF16 teacher + g128 student ~61.9 GiB weights alone

    g64 teacher + g128 student ~21.1 GiB weights alone

before activations/optimizer state.

However a synthetic bounded/local DWQ test showed actual improvement:

    MSE ~0.8813 -> ~0.7180
    ~18.5% reduction

Therefore bounded/layer-streamed DWQ remains plausible.

Stage 6 may invest substantially in it, but only after Q1/Q2/Q3 evaluation.

---

## MTP Status

Stage 5 statically found real MTP tensors in the original Qwen3.8 source:

    15 mtp.* tensors
    BF16
    ~849,398,784 bytes
    ~0.791 GiB

The current text-only conversion/runtime excludes or does not use them.

Pinned runtime has no proven native Qwen3.8 MTP decode path.

This means:

    MTP is physically present in the source
    but implementation compatibility is still unresolved.

Stage 6 may investigate/implement it after the main quant frontier is settled.

Do not train a new MTP head.

---

## Quantization Philosophy for Stage 6

Do NOT create another broad fixed-budget frontier unless Q1/Q2/Q3 evidence
clearly demands it.

Stage 6 selection order is:

    Q1 first

If Q1 is clearly better than g128:

    validate thoroughly
    then test Q2 only if it can plausibly improve quality without crossing the
    long-context cliff

If Q1 is clearly worse:

    test Q2 only if the failure mode plausibly improves with more promoted
    modules

If Q1 is equal quality but slower/larger:

    g128 remains winner; Q2/Q3 require a strong reason.

Do not automatically benchmark all three merely because they exist.

---

## Decision Standard

A candidate must improve the Pareto frontier, not one metric.

A meaningful win could be:

### Quality win

Same practical runtime as g128 but measurably better held-out quality.

### Runtime win

Same quality but faster or safer at useful context.

### Context win

Same quality and speed class but more long-context headroom.

A candidate that is only:

    +150 MiB
    same quality
    same speed

is not a meaningful default improvement.

---

## Performance Evaluation

For candidate selection, emphasize:

    4K smoke
    8K practical
    12K useful long context
    16K decisive long context

At 16K use fresh-process repeats where needed.

Record:

    prompt throughput
    TTFT
    decode tok/s
    active
    cache
    active+cache
    peak
    RSS
    memory_pressure

A repeatable decode cliff is disqualifying for default long-context use unless
quality gains are large enough to justify a separate role.

---

## Quality Evaluation

Use:

1. corrected Stage 4 Quality Suite V2;
2. corrected Stage 5 held-out suite;
3. teacher-forced NLL/KL subset;
4. selected coding/retrieval cases;
5. objective checks;
6. supplementary blind subjective comparison where useful.

Avoid:

    magic-keyword scoring
    insufficient generation caps
    evaluation leakage

Always record finish reason.

A max-token-truncated answer must not be counted as a clean semantic failure
without inspection.

---

## DWQ Philosophy

If a final Q1/Q2/g128 winner emerges, Stage 6 may attempt bounded refinement.

Primary objective:

> improve teacher/logit/loss agreement without increasing weight payload.

Preferred targets:

    g128
    OR
    best KL-guided candidate

Do not refine multiple weak candidates.

At most TWO serious refined 27B artifacts.

Do not spend the whole stage implementing a theoretically perfect DWQ system.

A bounded partial-layer or scale/bias refinement with measurable held-out
benefit is a valid research result.

---

## MTP Philosophy

MTP is potentially one of the highest-impact remaining performance ideas.

But it is gated because +0.791 GiB BF16 raw MTP weights are non-trivial on an
18 GB machine.

Required questions:

1. what exactly are the 15 tensors?
2. how are they used in Qwen3.8 reference inference?
3. can they be quantized?
4. what is the minimal runtime implementation?
5. what acceptance rate is achieved?
6. does the added residency destroy 16K performance?

Do not call MTP a win based only on short-context speed.

---

## Context Expansion

After winner selection, Stage 6 may test:

    20K

and, only if clearly safe:

    one 24K smoke

Never 32K.

Use:

    normal KV
    optional KV8

Activation quantization may be revisited only if Stage 5 did not already settle
it and there is a concrete reason.

---

## Research Memory

Maintain:

    research/stage6/PLAN.md
    research/stage6/STATUS.md
    research/stage6/RESULTS.md
    research/stage6/results.jsonl
    research/stage6/FINAL_REPORT.md

Before every major phase re-read:

    AGENTS.md
    TASK.md
    research/stage6/STATUS.md

STATUS.md must include:

    current phase
    current memory/swap baseline
    Q1/Q2/Q3 preservation status
    current candidate frontier
    quality-suite corrections
    current default
    DWQ status
    MTP status
    agent-backend status
    next action

---

## Git Discipline

Start with:

    git status
    git log --oneline -30

Preserve unrelated changes.

Never:

    git reset --hard
    git clean -fd

Do not commit:

    model weights
    model cache
    venvs
    giant transient traces

Commit:

    harness changes
    corrected evaluation data
    manifests
    rankings
    recipes
    results
    runtime/backend code
    tests
    reports

---

## Autonomy

Expected runtime:

    ~15 hours

Do not ask routine clarification questions.

If a branch fails:

    record why
    preserve evidence
    move on if another independent high-value branch exists

Do not burn hours reproducing the same failure.

Do not automatically create Stage 7.

---

## Final Standard

Stage 6 should answer:

1. Is Q1 a better practical model than g128?

2. If not, does Q2 or Q3 improve the Pareto frontier?

3. How reproducible is the KL-guided ranking on the SAME calibration data?

4. How robust is it across calibration subsets?

5. Does bounded DWQ improve the selected model?

6. Does MTP work for Qwen3.8 and is it practical at 8K/16K?

7. What is the recommended practical context?

8. Does 20K work?

9. Is the serialized agent backend validated with a real multi-turn 27B run?

10. What exact model/runtime command should the user use TODAY?
