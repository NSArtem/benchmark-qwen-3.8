# Qwen3.8-27B Stage 7 Final Report

Date: 2026-08-30
Repository: `/Users/artem/projects/benchmark-qwen-3.8`
Hardware: Apple M3 Pro, 18 GB unified memory

## Executive decision

### MEASURED FACT

The affine 3-bit, group-128 checkpoint remains the best production model:

`/Users/artem/models/Qwen3.8-27B-3bit-g128-MLX`

Its MLX model payload is 10,930,715,648 bytes (10.180022 GiB). It is the only
Stage 7 model selected for the daily CLI and serialized agent server.

Bounded diversity-aware DWQ was successfully reproduced and scaled to four
modules without materially changing the static payload class, but the
candidate lost one case on each final practical suite and had worse
teacher-forced NLL. It is rejected for production.

The source-backed Qwen3.8 MTP architecture was derived and an exact greedy
prototype was implemented. A source-backed affine 4-bit MTP candidate passed
short/4K/8K exactness checks, but its sequential verifier was slower than
direct g128 decoding and its 16K run hit the safety stop. It is research-only
and rejected for production.

The final g128 serialized coding-agent benchmark passed all four deterministic
tasks with five valid text-protocol tool requests and no protocol errors.
Provider-native structured `tool_calls` were not validated.

### INFERENCE

The current practical Pareto point is unchanged from Stage 6: g128 with a
12K–16K daily context envelope and ordinary exact-suffix persistent chat.
There is no validated Stage 7 quality mode or accelerated mode.

### HYPOTHESIS

A fused Qwen3.8 MTP verifier with native hybrid-state rollback could change the
speed result, but that implementation was not established here. Further
generic static quantization search is not justified by the Stage 7 evidence.

## Environment, safety, and baseline

### MEASURED FACT

- Canonical execution environment: `envs/mlx-0.32.1`, MLX 0.32.1, pinned
  mlx-lm revision `254d153fdeb6f150edd4fc5a54f9828638481fa8`.
- Stage 6 artifacts, model paths, runtime paths, and corrected evaluation
  manifests were recovered before new experiments.
- The host preload baseline reported 19,327,352,832 bytes total, 89% free,
  and 63.25 MiB swap used. No GPU wired-memory setting was modified and no
  `mx.set_wired_limit()` call was used.
- The g128 reference runtime remains:

| Context | Decode | Warm TTFT | Peak MLX memory | Role |
|---:|---:|---:|---:|---|
| ~8K | 11.137 tok/s | ~1.1 s | 13.983 GiB | reference |
| ~12K | 10.803 tok/s | ~1.1 s | 14.609 GiB | practical |
| ~16K | 10.4–10.6 tok/s | ~1.2 s | 15.236 GiB | practical upper daily |
| ~20K | 10.367 tok/s | 0.72 s follow-up in one run | 15.859 GiB | clean-host upper envelope |

The 20K result is not the daily recommendation.

## DWQ investigation

### Protocol and controls

#### MEASURED FACT

The frozen DWQ train corpus contains 16 cases across eight domains (two each):
coding, patch reasoning, structured output, retrieval, general reasoning,
instruction following, tool/agent transcripts, and prose continuity. The
validation corpus contains eight disjoint cases (one per domain). Final
held-out cases were not used for ranking, learning-rate selection, module-count
selection, or DWQ tuning.

- Train SHA-256:
  `3fedf6d50c6f202f189f0d585fae418e3475eb4458e5f352af509e4f9a58db98`
- Validation SHA-256:
  `1193e48e77bee8ada73d27ab5a025eb7100c8072c03051bec8fe0fb6feaa91dd`
- Corrected final heldout SHA-256:
  `1749a8b47e99bb3e0e04347d75e24d7022ed3ab8d5ffa5bc56926b9cf02cbcbd`
- Quality Suite V2 SHA-256:
  `f24e3983efbd898084852c3fde7b0f824604bc6a63349de5f32efeaa36ca3de0`

The g64 teacher and g128 student were evaluated in separate memory-safe
processes. The original g128 aggregate baseline was train KL 0.156751577,
teacher-argmax NLL 0.285186946, top-1 0.917525773; validation KL
0.142996969, teacher-argmax NLL 0.277587394, top-1 0.918316832.

#### MEASURED FACT: bounded refinement curves

All completed refinements were finite at 40 steps with learning rate 1e-6.
The raw curves are in `research/stage7/artifacts/dwq/`.

| Checkpoint | Modules | Representative local loss | Train KL | Validation KL | Patch bytes |
|---|---:|---:|---:|---:|---:|
| layer-60 reproduction | 1 | 0.265708178 → 0.231171161 | 0.156751577 → 0.155614787 | 0.142996969 → 0.142216881 | 8,098 |
| cumulative top-4 | 4 | all four curves decreased | 0.156751577 → 0.150977940 | 0.142996969 → 0.136866009 | 31,881 |
| cumulative top-8 | 8 | all four curves decreased | 0.150977940 → 0.156082136 | 0.136866009 → 0.142290316 | 63,591 |

The four refined module paths were:

1. `language_model.model.layers.60.linear_attn.in_proj_a`
2. `language_model.model.layers.49.linear_attn.in_proj_b`
3. `language_model.model.layers.18.linear_attn.in_proj_b`
4. `language_model.model.layers.21.linear_attn.in_proj_b`

The top-8 step was stopped after its validation regression. No top-16 or
top-32 run was launched. The one-module reproduction reduced local loss by
12.998%; the top-4 cumulative patch improved internal teacher KL but reduced
validation top-1 from 0.918316832 to 0.913366337.

#### MEASURED FACT: diversity robustness

Using the same four modules and the same eight-case validation set:

| Training selection | Validation KL | Validation teacher NLL | Validation top-1 |
|---|---:|---:|---:|
| diverse, 16 cases | 0.136866009 | 0.275872929 | 0.913366337 |
| coding-heavy, 8 cases | 0.140906769 | 0.277519246 | 0.918316832 |
| reasoning-heavy, 8 cases | 0.142019810 | 0.279135490 | 0.918316832 |

The diverse selection was preferred for internal selection because it had the
best validation distributional loss without using final held-out data.

### Final DWQ quality gate

#### MEASURED FACT

The fresh final comparisons were:

| Model | Final heldout objective | Final heldout semi | Quality V2 objective | Quality V2 semi | Teacher-forced NLL | Top-1 |
|---|---:|---:|---:|---:|---:|---:|
| g128 | 20/21 | 3/3 | 23/24 | 4/4 | 1.369011976 | 0.694610778 |
| g128 + top-4 DWQ | 19/21 | 3/3 | 22/24 | 4/4 | 1.372005988 | 0.694610778 |

Both runs had zero harness exceptions. Top-4 lost `h5-coding-06` on the final
heldout suite and `coding-07` on Quality V2. The shared reasoning repetition
flag was mechanically triggered after both models gave the correct answer and
was manually classified as a false positive. The teacher-forced comparison
used 167 target tokens.

The top-4 patch adds only small overlay metadata; the base model payload stays
in the same 10.180022 GiB class. The complete research artifact is
`research/stage7/artifacts/dwq/top4.safetensors`; it is not a production model
directory.

#### INFERENCE

DWQ achieved the intended internal calibration improvement but no practical
quality improvement. Relative to g128, the selected top-4 candidate regressed
final objective coverage by 1/21 case (4.76 percentage points) on heldout and
1/24 case (4.17 percentage points) on Quality V2, while teacher-forced NLL
worsened by 0.002994012 nats/token. The correct verdict is DWQ REJECTED;
g128 remains DEFAULT. No DWQ runtime comparison was warranted after this gate.

#### HYPOTHESIS

The top-4 calibration objective may be correcting teacher-distribution error
in ways that perturb generation-critical recurrent dynamics. The evidence is
consistent with this explanation but does not prove causality.

## MTP investigation

### Architecture derivation

#### MEASURED FACT

The original source checkpoint is
`/Users/artem/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
Its nested text configuration specifies hidden size 5120, 64 target layers,
full attention every fourth layer, 24 query heads, 4 KV heads, head dimension
256, linear-attention key/value dimensions 128/128, `mtp_num_hidden_layers=1`,
`mtp_use_dedicated_embeddings=false`, and block size 3.

All 15 `mtp.*` tensors are in source shard 18 and total 849,398,784 BF16
bytes (0.791 GiB). They map to:

- `mtp.fc.weight`;
- two pre-FC RMSNorm vectors for token embedding and target hidden state;
- one full-attention decoder block with input/post-attention norms,
  q/k/v/o projections, q/k norms, and three MLP projections; and
- one final MTP RMSNorm vector.

There is no dedicated MTP embedding or MTP LM head. The derived graph is:

```text
shared token embedding + target hidden
        ↓ separate centered RMSNorms
        ↓ concatenate (2H) → mtp.fc (2H → H)
        ↓ one full-attention decoder block
        ↓ final MTP RMSNorm → shared target LM head
```

The derivation is recorded in `research/stage7/mtp/ARCHITECTURE.md`. It was
cross-checked against the local executable `mlx_vlm==0.6.13` Qwen3.5 MTP
reference and the pinned MLX Qwen3.5 hybrid cache implementation. The pinned
text-only runtime itself strips `mtp.*` tensors.

#### INFERENCE

The graph and cache semantics were sufficiently supported to justify a guarded
prototype, but not to assume that a batched verifier was correct. A proposal
round can contain at most two draft tokens, and rejection must restore both
full-attention KV state and linear/GatedDelta `ArraysCache` state.

### Exactness and MTP precision

#### MEASURED FACT

The prototype is `research/stage7/harness/run_mtp_correctness.py`. It uses
explicit state snapshots and replay and compares greedy output token-for-token
with the direct g128 target.

- BF16 MTP: 849,398,784 bytes live; exact greedy identity and all observed
  rollback checks passed on coding, reasoning, prose, and structured prompts.
  Aggregate short-gate acceptance was 66/102 (64.7%).
- 4-bit MTP: source-backed affine linear quantization, group size 128,
  225,659,904 quantized parameter bytes and 225,663,093 bytes on disk. Active
  MTP overhead after loading was 225,661,184 bytes (approximately 0.210 GiB).
  Exact greedy identity and rollback checks passed on the same four prompts;
  aggregate short-gate acceptance was 68/100 (68.0%).
- The batched verification shortcut failed exactness on coding and reasoning
  and failed rollback on reasoning. It is not used by the headline results.

The raw exactness artifacts are under
`research/stage7/artifacts/mtp/correctness-bf16-v3/`,
`research/stage7/artifacts/mtp/correctness-4bit-g128/`, and
`research/stage7/artifacts/mtp/correctness-4bit-g128-batch/`.

### MTP performance and residency gate

#### MEASURED FACT

Fresh canonical-runtime processes used the same target, prompts, 64-token
budgets, and baseline token files. The tested mode was the sequential
target-verification plus snapshot/replay prototype.

| Context | Baseline decode | 4-bit MTP decode | MTP/base throughput | Accepted / proposed | Exact |
|---:|---:|---:|---:|---:|---|
| short (1,023) | 11.898 tok/s | 6.289 tok/s | 0.529x (-47.1%) | 39/52 (75.00%) | yes |
| 4K (4,096) | 11.647 tok/s | 4.070 tok/s | 0.349x (-65.1%) | 29/68 (42.65%) | yes |
| 8K (8,191) | 11.147 tok/s | 2.651 tok/s | 0.238x (-76.2%) | 34/60 (56.67%) | yes |
| 16K (16,727) | 10.628 tok/s | safety stop | N/A | N/A | not reached |

Peak memory was 12.077 vs 12.287 GiB at short, 13.360 vs 13.621 GiB at
4K, and 13.983 vs 15.456 GiB at 8K for baseline vs MTP. The 16K MTP process
was terminated when the host reported 28% free memory, 909 free pages, 544,794
compressor-occupied pages, and rapidly rising swapouts. The host recovered to
89% free after termination. No 16K acceptance, speed, or exactness result is
claimed.

Raw logs are in
`research/stage7/artifacts/performance/mtp-4bit-g128-v1/`; the safety record
is `16k-mtp-safety-stop.json`.

#### INFERENCE

The exact 4-bit sequential MTP prototype is slower than direct g128 decoding
at every completed context and is unsafe to carry into the 16K daily mode.
MTP is therefore REJECTED as DEFAULT, OPTIONAL SHORT-CONTEXT, or ACCELERATED
MODE. Its precision verdict is “4-bit exact research candidate,” not “safe
production acceleration.”

#### HYPOTHESIS

The widening slowdown is likely dominated by unfused proposal/verification and
state-copy/replay cost, not merely by MTP weight precision. A future fused
implementation could improve the ratio, but the failed batched exactness probe
means that such an optimization cannot be assumed correct.

## Serialized coding-agent benchmark

### MEASURED FACT

The final benchmark is
`research/stage7/harness/run_agent_benchmark.py`, with isolated deterministic
fixtures in `research/stage7/agent/fixtures/`. It used the g128 model through
`research/runtime/qwen38_agent_server.py`, one serialized session at a time,
greedy decoding, a 192-token generation cap, and an explicit instruction to
keep responses under 120 tokens.

| Task | Description | Assistant completions | Valid tool calls | Final logical context | Result |
|---|---|---:|---:|---:|---|
| A | mutable-default diagnosis | 3 | 1 | 551 | pass |
| B | idempotent retry recovery | 3 | 1 | 660 | pass |
| C | multi-file terminal-event ordering | 5 | 2 | 983 | pass |
| D | structured formatter result | 3 | 1 | 356 | pass |

Aggregate results: 4/4 tasks passed, five valid text-protocol tool calls, zero
invalid tool calls, zero HTTP/protocol request errors, four successful reset
checks, and server-reported `serialized=true`. Median warm TTFT was 1.367139 s
(maximum 1.892843 s); median decode was 11.879811 tok/s. Task D returned valid
JSON with the required `action`, `path`, and `reason` fields.

Tasks B and C ended their final assistant response at the 192-token length cap,
but their rubric checks passed. A separate max-256 stress artifact recorded an
HTTP 409 exact-prefix failure after a truncated response and is excluded from
the pass score. The final raw transcript and per-fixture hashes are in
`research/stage7/artifacts/agent/g128-coding-agent-final.json`.

#### INFERENCE

The g128 serialized backend is useful for small local coding-agent workflows:
it requested files, diagnosed bugs, recovered after an intentionally wrong
hypothesis, handled a multi-file exchange, and returned a structured result
without transport errors. The result validates serialized text-protocol
tool-use and session/reset behavior, not autonomous provider-native tool
selection.

#### HYPOTHESIS

A wrapper that reserves response budget and handles continuation before the
hard cap could reduce the observed length-related 409 risk. That wrapper was
not implemented or benchmarked in Stage 7.

## Production recommendation today

### MEASURED FACT

- DEFAULT model: `/Users/artem/models/Qwen3.8-27B-3bit-g128-MLX`
- QUALITY MODE: none; the four-module DWQ artifact is rejected.
- ACCELERATED MODE: none; MTP is research-only.
- Recommended daily context: 12K–16K. Treat 20K as a clean-host upper
  envelope, not a default.
- Keep one serialized session, exact canonical transcript/suffix handling,
  direct generation, and default wired-memory state.

### CLI command

```bash
source research/stage4/canonical_env.sh
uv run --project envs/mlx-0.32.1 --no-sync python research/runtime/qwen38_chat.py \
  --model /Users/artem/models/Qwen3.8-27B-3bit-g128-MLX \
  --mode greedy --max-tokens 512 --max-kv-size 16384
```

### Serialized server command

```bash
source research/stage4/canonical_env.sh
uv run --project envs/mlx-0.32.1 --no-sync python research/runtime/qwen38_agent_server.py \
  --model /Users/artem/models/Qwen3.8-27B-3bit-g128-MLX \
  --host 127.0.0.1 --port 8000 \
  --mode greedy --max-tokens 512 --max-kv-size 16384
```

### INFERENCE

These commands expose only the validated g128 runtime. KV-cache quantization
flags remain available for separate controlled experiments, but were not
needed to establish the Stage 7 default and are not silently enabled.

## Highest-value future ideas (not started)

### HYPOTHESIS

1. Implement a fused Qwen3.8 MTP proposal/verification path with explicit
   hybrid KV and recurrent rollback, then re-run exact token identity before
   any performance measurement.
2. Profile and eliminate snapshot/replay allocations in that fused path before
   testing 16K again; stop immediately on the same host-pressure signals.
3. Improve the serialized agent wrapper's length budgeting/continuation
   behavior and validate native structured tool calls separately from text
   protocol transport.

No Stage 8 work was started automatically.

## Evidence and commit index

Important raw evidence:

- DWQ curves and checkpoints: `research/stage7/artifacts/dwq/`
- Final heldout and Quality V2 raw outputs:
  `research/stage7/artifacts/final/`
- MTP architecture: `research/stage7/mtp/ARCHITECTURE.md`
- MTP exactness and quantization: `research/stage7/artifacts/mtp/`
- MTP performance and safety stop:
  `research/stage7/artifacts/performance/mtp-4bit-g128-v1/`
- Agent transcripts: `research/stage7/artifacts/agent/`
- Machine-readable ledger: `research/stage7/results.jsonl`

Stage 7 logical commits:

`beb408b`, `db31936`, `1214093`, `e08ace5`, `3dbf39e`, `f6da2df`,
`797f3e3`, `c09185c`, `9c37a1d`, `391c98f`, `82ab556`, `735ea30`, and the
final report commit containing this file.
