# Qwen3.8-27B MLX Investigation — Stage 6 Final Report

Date: 2026-08-26

This report distinguishes measured facts from inferences. Model weights and
large transient runtime traces remain outside git.

## Executive decision

**MEASURED FACT:** The fixed-weight winner is g128:

`/Users/artem/models/Qwen3.8-27B-3bit-g128-MLX`

**INFERENCE:** Keep g128 as the production default. Use normal KV at 12K–16K
for routine work. A clean-host 20K prompt completed coherently, but its 15.859
GiB MLX peak and 23% free-memory completion snapshot make it an opportunistic
upper-envelope mode rather than a daily setting. 24K was not run because the
20K result did not leave enough projected headroom. 32K remains out of scope.

Q1 did not beat g128, and the explicit Q1 gate correctly skipped Q2/Q3. A
bounded DWQ pass improved local objectives but did not produce a complete
refined model. MTP is not ready in the pinned runtime. The serialized backend
passed a real ten-exchange 27B smoke.

## Environment and safety

The initial clean baseline showed 90% free memory, zero compressor pages, zero
swap I/O, and no stale heavy MLX/Qwen process. Q1/Q2/Q3 were found intact on
stable storage and were never rebuilt, deleted, or overwritten. All heavy work
used the pinned canonical environment and did not call `mx.set_wired_limit()` or
modify `iogpu.wired_limit_mb`. No two heavy Qwen/MLX processes were run
concurrently.

Candidate preservation:

| candidate | payload | effective BPW | verdict |
|---|---:|---:|---|
| g128 | 10.180022 GiB | 3.2513 | DEFAULT |
| g64 | 10.962722 GiB | 3.5012 | reference |
| mixed34 | 10.664947 GiB | 3.4061 | long-context cautionary control |
| Q1 | 10.326507 GiB | 3.2980 | rejected |
| Q2 | 10.472991 GiB | 3.3448 | preserved, skipped |
| Q3 | 10.619475 GiB | 3.3916 | preserved, skipped |

Q1/Q2/Q3 each have 20/20 weight shards plus config, index, tokenizer, and
chat-template files under `/Users/artem/models/stage5-frontier/`.

## Corrected quality evidence

The frozen `heldout_v2` suite is
`research/stage6/heldout/heldout_v2.jsonl`, SHA-256
`1749a8b47e99bb3e0e04347d75e24d7022ed3ab8d5ffa5bc56926b9cf02cbcbd`.
Nine known truncation defects were corrected symmetrically by raising affected
budgets to 1024 tokens. Prompts, semantic answers, and scorer rules were not
changed.

| model | Quality Suite V2 objective/semi | heldout_v2 objective/semi | teacher NLL / top-1 |
|---|---:|---:|---:|
| g128 | 23/24 + 4/4 | 20/21 + 3/3 | 1.369012 / 0.694611 |
| Q1 | 22/24 + 4/4 | 17/21 + 2/3 | 1.415419 / 0.682635 |
| g64 | 23/24 + 4/4 | 19/21 + 2/3 | 1.403443 / 0.700599 |
| mixed34 | 23/24 + 4/4 | 18/21 + 2/3 | 1.385479 / 0.700599 |

The formal heldout numbers retain repetition flags and finish reasons. Raw
inspection found Q1 coding-04 and reasoning-02 reached required semantic
content but were flagged by repetition/format conditions; that does not turn
Q1 into a win. Q1 remains behind g128 on the corrected formal score and on
teacher-forced NLL/top-1.

## Q1 runtime and frontier decision

| model/context | decode tok/s | TTFT | peak |
|---|---:|---:|---:|
| g128 8K reference | 11.137 | 113.20 s | 13.983 GiB |
| Q1 8K | 11.013 | 106.48 s | 14.129 GiB |
| g128 12K reference | 10.803 | — | 14.609 GiB |
| Q1 12K | 10.443 | 164.26 s | 14.756 GiB |
| g128 16K reference | 10.583 | 242.12 s | 15.236 GiB |
| Q1 16K | 10.260 | 232.92 s | 15.382 GiB |

Q1 was coherent at all three cold contexts and passed exact-suffix five-turn
persistent smokes at approximately 8K and 16K. It was nevertheless larger,
slower on decode, heavier at 16K, and lower on quality. **Q1 verdict: no
Pareto win; reject as default and optional quality mode.** Q2 and Q3 were not
loaded because no plausible Q1 failure mode justified their larger payloads.

## KL-ranking reproducibility

**Same data:** Repeating 45 Stage 5 rank-band modules on the exact same 12
calibration cases produced Spearman 1.000, Pearson 1.000, top-10 10/10,
top-25 25/25, 100% sign agreement, and zero mean/max absolute score delta.

**Different subset:** The same modules on a separate two-case subset produced
Spearman 0.410, Pearson 0.317, top-10 5/10, top-25 20/25, and 73.3% sign
agreement.

**VERDICT:** The ranking is numerically reproducible but calibration-set
sensitive. The earlier Stage 5 low correlation was not a nondeterminism proof;
it used a different subset.

## Bounded DWQ

**MEASURED FACT:** Whole-model teacher/student DWQ is impossible within the
18 GB envelope. A streamed one-module pass on
`language_model.model.layers.60.linear_attn.in_proj_a` kept one 983,040-byte
source tensor resident. At learning rate `1e-6`, 40 scale/bias steps reduced
local MSE from 0.253556 to 0.218637 (13.77%) and reduced one end-to-end
calibration KL from 0.091406 to 0.090705. Peak MLX memory was 10.681 GiB and
no wired-limit call occurred.

**VERDICT:** Bounded/layer-streamed DWQ is feasible as a research method, but
the optimizer is step-size sensitive and no complete refined 27B checkpoint
was materialized. No heldout or production quality improvement is claimed.

## Context envelope

The final g128 20K normal-KV probe used exactly 20,480 prompt tokens and
generated 128 tokens coherently:

- prompt throughput: 72.688 tok/s;
- TTFT: 281.756 s;
- decode: 10.367 tok/s;
- active+cache after completion: 11.622 GiB;
- MLX peak: 15.859 GiB;
- completion snapshot: 23% free memory, 488.62 MiB swap used / 1,559.38 MiB free.

The same prompt then passed three exact-suffix follow-ups, reaching logical
context 20,686 with warm TTFT 0.724–0.726 s, warm decode 10.419–10.433
tok/s, no replay flags, and a reset back to zero.

**VERDICT:** 8K, 12K, and 16K are the practical tested envelope; 20K works
once on a clean host but is not recommended for routine use. Recommend 16K
normal KV, or 12K when system headroom is more important. No KV8 run was
needed for the decision. 24K was gated off; 32K was never attempted.

## MTP investigation

Header-only inspection of the original source shard found 15 top-level BF16
`mtp.*` tensors totaling 849,398,784 bytes (0.791 GiB). They are one
`mtp.layers.0` attention/MLP block with `mtp.fc`, Q/K/V/O projections, MLP
projections, and normalization weights. Exact names, shapes, and byte counts
are in [mtp_static_inspection.json](mtp_static_inspection.json).

The pinned MLX-LM `qwen3_5.py` strips `mtp.*` before loading the validated text
model and has no native Qwen3.8 MTP head/reference decode path. Its generic
speculative interface is not an MTP implementation. Ideal raw payload math is
approximately 0.396 GiB at 8-bit, 0.198 GiB at 4-bit, and 0.148 GiB at 3-bit
before scales and metadata; no quantized MTP artifact was created.

**VERDICT:** `NOT READY / UNSUPPORTED` for today's affine g128 runtime.
Acceptance rate, token identity, decode speed, and MTP residency are N/A. No
head was fabricated and no new MTP head was trained.

## Serialized agent backend

The real g128 backend smoke completed ten sequential 27B exchanges. It
included system/user/assistant/tool roles, two tool-call/result injections,
`/health`, `/v1/models`, `/metrics`, exact suffix/cache-offset checks, and a
session reset. All ten completion requests succeeded, replay flags were false,
and reset returned zero messages and zero logical context. Ordinary warm TTFT
was approximately 0.583–0.593 s; longer tool-result suffixes reached about
1.38 s; short-response decode was approximately 11.9–12.0 tok/s.

**VERDICT:** Ready for one serialized local session. It is not a concurrent or
batching server. Tool-call payloads are normalized as text; this smoke does
not claim provider-native structured tool-call generation.

## Recommended commands today

Interactive exact-suffix chat, normal KV:

```sh
source research/stage4/canonical_env.sh
uv run --project envs/mlx-0.32.1 --no-sync python research/runtime/qwen38_chat.py \
  --model /Users/artem/models/Qwen3.8-27B-3bit-g128-MLX \
  --max-tokens 512
```

Serialized OpenAI-compatible backend:

```sh
source research/stage4/canonical_env.sh
uv run --project envs/mlx-0.32.1 --no-sync python research/runtime/qwen38_agent_server.py \
  --model /Users/artem/models/Qwen3.8-27B-3bit-g128-MLX \
  --host 127.0.0.1 --port 8000 --mode greedy --max-tokens 512
```

Keep one inference active at a time. Optional `--kv-bits 8` is a reserve
headroom tradeoff, not part of the default recommendation.

## Key artifacts and commits

- Status: [STATUS.md](STATUS.md)
- Ledger: [results.jsonl](results.jsonl)
- Corrected suite: [heldout_v2.jsonl](heldout/heldout_v2.jsonl)
- Context results: [g128-20k-normal/result.json](artifacts/context/g128-20k-normal/result.json) and [g128-20k-persistent/result.json](artifacts/context/g128-20k-persistent/result.json)
- Agent result: [g128-serialized-smoke.json](artifacts/agent/g128-serialized-smoke.json)
- DWQ result: [g128-layer60-in_proj_a-stable.json](artifacts/dwq/g128-layer60-in_proj_a-stable.json)

The report was prepared from parent commit `1439ec7`; the final Stage 6 commit
SHA is supplied in the handoff after the final audit.
