# Qwen3.8-27B MLX Stage 3 Final Report

Date: 2026-08-24  
Machine: Apple M3 Pro, 18 GB unified memory, macOS 15.7.8  
Canonical runtime: Python 3.14.7, MLX 0.32.1, mlx-lm 0.31.3, pinned revision
`254d153fdeb6f150edd4fc5a54f9828638481fa8`

## Executive result

MEASURED FACT: source-backed affine 3-bit/group-128 g128 sustained correct
exact-suffix persistent chat through a 16,727-token initial prompt and
17,219-token normal-KV logical offset. Five-turn runs at 8K, 12K, and
16K-class contexts were coherent, non-replaying, and exactly suffix-accounted.

INFERENCE: the recommended practical configuration today is g128, normal KV,
direct `generate_step`, default MLX wired state, and exact canonical suffixes,
with an approximately 8K initial context. It gives near-second follow-ups and
about 11 tok/s decode with measured headroom. 12K is usable; 16K-class is a
bounded long-context mode, not an endurance guarantee.

MEASURED FACT: g128 is not quality-equivalent to g64 on the completed suite.
The classification is **PERFORMANCE/LONG-CONTEXT MODE**. KV8 is a validated
**SELECTABLE LONG-CONTEXT MODE** at 16K-class context, not the default. The
20K and optional acceleration branches were not run.

## 1. Persistent-cache correctness

MEASURED FACT: the old 8K warm protocol was invalid and replayed turn 1. It
is excluded from correctness evidence. The corrected protocol renders the
full canonical transcript, measures the logical cache prefix, and supplies
only the missing canonical suffix.

At 4K, the same-boundary oracle compared:

| path | purpose | result |
|---|---|---|
| A | fresh one-shot full transcript | coherent, no replay; diverged after 42 generated tokens from segmented paths |
| B | fresh segmented execution | coherent; same token stream and call boundaries as C |
| C | true persistent exact-suffix execution | coherent; same token stream and call boundaries as B |

MEASURED FACT: B and C matched at every required boundary. All 48 recurrent
`ArraysCache` and 16 attention `KVCache` layers had equal compact signatures
and offsets; first-token logprob max/mean absolute differences were 0.0 and
top-8 overlap was 8/8. The logical equation `4,192 + 38 = 4,230` held.

INFERENCE: exact-suffix persistence is **VALIDATED FOR SCALING**. A-vs-B/C
divergence is segmentation/floating-point sensitivity of the hybrid recurrent
model, not evidence of a persistent-cache-specific bug. The corrected 8K
fresh semantic oracle passed and was token-equal to its persistent turn for
29/29 generated tokens; the corrected 16K-class semantic oracle passed too.

Artifacts: `research/stage3/cache-boundary-oracle/` and
`research/stage3/persistent-scaling-final/`. The pre-fix 8K oracle that
serialized 16,536 tokens is preserved but explicitly excluded.

## 2. Context and latency ladder

MEASURED FACT: all values below are from the corrected g128 normal-KV
persistent run. Warm values are follow-up turns; cold TTFT is the initial
full prefill.

| initial prompt | final offset | cold TTFT | warm TTFT median / max | warm decode | peak | verdict |
|---:|---:|---:|---:|---:|---:|---|
| 8,191 tokens | 8,676 | 110.729 s | 1.071 / 1.098 s | 11.105 tok/s | 13.983 GiB | PRACTICAL |
| 12,288 tokens | 12,760 | 193.462 s | 1.117 / 1.121 s | 10.803 tok/s | 14.609 GiB | USABLE |
| 16,727 tokens | 17,219 | 246.012 s | 1.205 / 1.223 s | 10.409 tok/s | 15.236 GiB | MARGINAL / bounded |

MEASURED FACT: the 4K same-boundary exact-suffix follow-up measured 1.027 s
TTFT and 11.455 tok/s decode. The preserved 4K fresh full-transcript oracle
measured about 54.745 s TTFT. These are different execution modes.

INFERENCE: follow-up TTFT is governed primarily by the short incremental
suffix, not by total history. Across 8K–16K-class runs, suffixes were 33–39
tokens while warm TTFT stayed approximately 0.7–1.2 s. Initial cold prefill
still scales with full prompt length.

## 3. Endurance

MEASURED FACT: the g128 normal-KV endurance run completed 17 sequential turns
from an 8K start. Turns 1–16 were coherent, non-replaying, and exactly
suffix-accounted. Warm TTFT was 1.549–1.624 s, decode was 10.58–11.17 tok/s,
and logical context reached 10,439 tokens. Peak stayed at 13.983 GiB, RSS was
approximately 6.68 GB, and periodic memory-pressure checks showed no severe
pressure or unexplained cache/state drift.

The first failing turn generated four tokens: `The smallest worker` plus EOS.
It was not replay or protocol corruption, but was incomplete and therefore
`BROKEN`. The 20–30-turn endurance gate is **STOPPED_OR_UNRESOLVED**, not
passed. No continuation was run.

## 4. g64 versus g128 quality

MEASURED FACT: separate fresh-process greedy runs completed 38 cases per model.

| category | cases | g64 pass | g128 pass | delta |
|---|---:|---:|---:|---:|
| coding | 10 | 2 | 0 | -20.0 pp |
| instruction | 6 | 6 | 6 | 0.0 pp |
| structured | 6 | 6 | 6 | 0.0 pp |
| reasoning | 6 | 4 | 3 | -16.7 pp |
| retrieval | 5 | 3 | 3 | 0.0 pp |
| prose | 5 | 4 | 4 | 0.0 pp |
| **total** | **38** | **25 / 65.8%** | **22 / 57.9%** | **-7.9 pp** |

The suite had no inference exceptions. Fresh-case median decode was about
11.223 tok/s for g64 and 12.017 tok/s for g128; measured peak was 14.456 and
13.673 GiB respectively. Prior 4K teacher-forced evidence measured mean NLL
4.1387 nats for g64 versus 5.8335 for g128 over 4,095 targets (+1.6947 nats).

INFERENCE: g128 is **PERFORMANCE/LONG-CONTEXT MODE**, not `DEFAULT`. The
objective disadvantage is concentrated in coding and reasoning, while
instruction following, structured output, retrieval, and prose were tied.
Several coding answers were relevant but cut off by the 64-token cap, so the
raw suite does not prove a literal 20-point unconstrained coding deficit. It
does establish a real quality risk for coding-agent use. g64 remains the
quality/reference choice when that risk outweighs g128's 0.783-GiB saving and
speed advantage.

Artifacts: `research/stage3/quality-final/` and
`research/stage3/results.jsonl`.

## 5. KV8 at 16K-class context

MEASURED FACT: normal KV and KV8 were run in separate fresh g128 processes on
the same prompt family and five exact-suffix turns.

| mode | cold TTFT | warm TTFT median | warm decode mean | active+cache peak | peak | correctness |
|---|---:|---:|---:|---:|---:|---|
| normal KV | 224.005 s | 1.169 s | 10.513 tok/s | 11.418 GiB | 15.236 GiB | PASS |
| KV8, group 64, start 0 | 223.990 s | 1.169 s | 9.522 tok/s | 11.390 GiB | 14.771 GiB | PASS |

KV8 saved 0.465 GiB of peak and imposed a 9.4% warm-decode penalty. Its
attention caches were `QuantizedKVCache` in all 16 attention layers; the 48
recurrent layers remained `ArraysCache`. Every turn was coherent,
non-replaying, offset-consistent, and suffix-accounted.

INFERENCE: KV8 is useful as a selectable 16K safety lever when the additional
0.465 GiB headroom is operationally important. It is not a default for 4K/8K
because normal KV already passed the 16K-class safety envelope. One KV8
turn-5 recommendation differed materially from normal KV while remaining
current-turn relevant; treat this as a quantization-quality sensitivity
caveat, not a cache-state failure.

## 6. 20K and decode acceleration gates

MEASURED FACT: 20K was **not tested**. The repository's natural prompt family
ends at 16,727 tokens. Normal KV already measured 15.236 GiB at that point,
and the endurance quality gate stopped at turn 17.

INFERENCE: synthetic padding to 20K was not justified. It would probe close to
the unattended safety envelope without a natural workload and without closing
the more important quality/endurance uncertainty. No 24K or 32K work was
performed.

MEASURED FACT: optional decode acceleration was **not tested**. TASK.md makes
it conditional on a passed endurance gate and no urgent primary issue.

## 7. Exact configuration to run today

For the normal interactive coding-agent workload:

- Model: `/Users/artem/models/Qwen3.8-27B-3bit-g128-MLX`.
- Environment: `envs/mlx-0.32.1`, MLX 0.32.1, pinned mlx-lm revision above.
- Runtime: direct `generate_step`, greedy decoding, exact canonical
  incremental suffixes, one persistent cache.
- KV: normal KV (`kv_bits=None`) for approximately 8K initial context.
- Wired memory: leave MLX at its default; do not call
  `mx.set_wired_limit()` and do not change any system wired setting.
- Safety expectation: approximately 13.983 GiB peak at 8K, approximately
  1.07 s median warm TTFT, and approximately 11.105 tok/s warm decode in the
  measured five-turn run.

For a bounded 16K-class conversation, use normal KV first. If measured
headroom is insufficient, select KV8 with `kv_bits=8`, `kv_group_size=64`,
and `quantized_kv_start=0`, accepting approximately 9.4% lower decode. Use
g64 instead when quality-critical coding behavior is more important than
memory/speed.

## 8. Highest-value follow-ups

FUTURE HYPOTHESIS / follow-up 1: repeat the coding/reasoning comparison with
longer output caps and a small blind rubric or teacher-forced top-1 agreement
subset. This will separate truncation effects from intrinsic g128 quality loss.

FUTURE HYPOTHESIS / follow-up 2: investigate the turn-17 early-EOS endurance
failure with a bounded reproduction and raw token/logprob analysis before
attempting another long unattended session. Do not treat the current 16K
five-turn pass as a 20–30-turn guarantee.

FUTURE HYPOTHESIS / follow-up 3: after an appropriate natural 20K prompt is
available and the safety policy is revisited, test one conservative KV8 or
normal-KV 20K run. Alternatively, isolate a pinned upstream cache/runtime
change in a disposable environment; do not upgrade the canonical runtime.

## 9. Phase commits

| phase | commit |
|---|---|
| initialization | `051ac88` |
| Phase 1 same-boundary oracle | `acc8fad` |
| Phase 3 persistent scaling | `7ff69be` |
| Phase 4 endurance | `b4c26d5` |
| Phase 5 g64/g128 quality | `573cf69` |
| Phase 6 KV8 long-context comparison | `ce30989` |

The final report itself is committed separately after the final status and
repository audit.
