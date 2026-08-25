# Qwen3.8 Stage 5 Final Report

Date: 2026-08-25
Hardware: Apple M3 Pro, 18 GB unified memory, macOS 15.7.8
Environment: `envs/mlx-0.32.1`, MLX 0.32.1, pinned mlx-lm revision
`254d153fdeb6f150edd4fc5a54f9828638481fa8`

## Executive result

The production default is **g128 affine 3-bit/group-128**:
`/Users/artem/models/Qwen3.8-27B-3bit-g128-MLX`.

It is the smallest complete candidate with the best practical runtime, no
reproduced mixed34 long-context cliff, and no held-out quality disadvantage
once the semi-objective cases are included. Mixed34 remains a quality
alternate/control for short-to-medium contexts, not a general default. The
loss-guided frontier was successfully built but not quality-tested far enough
to promote a finalist because the first Q1 smoke drove the host into unsafe
swap pressure.

The most important unresolved result is therefore explicitly negative:
Stage 5 did **not** establish that a new fixed-budget candidate beats g128 or
mixed34. It established a bounded end-to-end ranking and three reproducible
weight checkpoints, then stopped before unsafe model loads.

## 1. Mixed34 16K cliff

### MEASURED FACT

The cliff is real and reproducible in fresh processes.

| model | actual prompt | cold TTFT | prefill tok/s | decode tok/s | peak | end active+cache | result |
|---|---:|---:|---:|---:|---:|---:|---|
| g128, fresh Stage 5 | 16,727 | 242.118 s | 69.088 | 10.583 | 15.236 GiB | 11.403 GiB | coherent |
| mixed34, fresh Stage 5 retry | 16,727 | 240.786 s | 69.470 | 4.890 | 15.721 GiB | 11.873 GiB | coherent |

Stage 4 independently measured mixed34 at 5.719 tok/s over five persistent
16K-class turns, versus g128's same-harness normal-KV baseline of about 10.409
tok/s. At 8,191 tokens, Stage 4 measured mixed34 at 10.743 tok/s and g128 at
about 11.105 tok/s. The transition is therefore bounded between 8,191 and
16,727 tokens; the exact onset was not localized because additional 27B
prefills drove encrypted swap to 12–14 GiB and crossed the safety gate.

### INFERENCE

The penalty correlates with long-context residency: mixed34 is 0.485 GiB
larger than g128 and its measured peak is 0.485 GiB higher at 16,727 tokens.
That is sufficient to make it a practical long-context penalty.

### HYPOTHESIS

The measurements are consistent with a Metal allocation/cache behavior cliff,
but do not prove that payload size alone causes it. Context length, active/cache
state, OS pressure, and allocator behavior were not independently isolated.

## 2. Held-out quality suite

The suite is frozen at
`research/stage5/heldout/heldout_v1.jsonl` with SHA-256
`6fe7bd7da6732e9057a75764292b18b29c3262a96eb39391b700f038f641851b`.
It contains 24 cases: coding/debugging 6, instruction following 4, structured
output 4, reasoning/arithmetic 4, context retrieval 3, and prose continuity 3.
Six teacher-forced anchors are supplementary only. Direct prompt and case-ID
overlap checks against Stage 4 calibration/evaluation passed. The suite was
never used for calibration, sensitivity, allocation, or candidate selection.

| model | coding | instruction | structured | reasoning | retrieval | prose | objective | semi-objective | combined |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| g64 | 5/6 | 4/4 | 4/4 | 2/4 | 3/3 | 2/3 | 18/21 | 2/3 | 20/24 |
| g128 | 5/6 | 4/4 | 4/4 | 1/4 | 3/3 | 3/3 | 17/21 | 3/3 | 20/24 |
| mixed34 | 5/6 | 4/4 | 4/4 | 2/4 | 3/3 | 2/3 | 18/21 | 2/3 | 20/24 |

All three runs had zero exceptions, early-EOS events, and repetition flags.

### MEASURED FACT

Mixed34 is one objective case above g128, but ties g128 on the complete
objective-plus-semi suite. It does not beat g128 on a robust aggregate quality
measure, and it carries the reproduced 16K penalty.

## 3. End-to-end sensitivity method

The pinned Dynamic Quant and DWQ implementations were inspected but not run as
written: they duplicate whole-model state, and the pinned DWQ flow calls
`mx.set_wired_limit()`.

The replacement method is bounded and calibration-only:

1. Capture g64 teacher logits for the 12-case Stage 4 calibration split in a
   separate process; teacher tensors are stored as float16 and are not kept as
   a resident teacher during the sweep.
2. Keep g128 resident.
3. Load one high-precision source chunk at a time, quantize one candidate
   module to affine 4-bit/group-128, and evaluate its effect on final logits.
4. Score teacher-KL recovery per added byte, with teacher-argmax NLL, top-1,
   and logit MSE as secondary signals.
5. Release the source chunk and proceed deterministically through all 498
   modules.

The 12 calibration cases contained 477 target tokens. Baseline g128 versus
g64-teacher KL was 0.115868; maximum MLX peak was 14.480 GiB and maximum
active+cache was 12.302 GiB. The frozen full ranking list hash is
`e9d561a11807810dc44a9a6e38080623f2b41207d03643515b0661c8a188dd6b`.

The highest recovery-per-byte slots were:

1. layer 60 `linear_attn.in_proj_a` — `1.2313e-7`;
2. layer 49 `linear_attn.in_proj_b` — `1.2210e-7`;
3. layer 18 `linear_attn.in_proj_b` — `1.2174e-7`;
4. layer 21 `linear_attn.in_proj_b` — `1.0835e-7`;
5. layer 17 `linear_attn.in_proj_b` — `1.0491e-7`.

Mean KL recovery by family was 0.000572 for `linear_attn`, 0.000203 for
`self_attn`, and 0.000091 for `mlp`. The new top-25 overlapped the old Stage 4
RMS ranking in only 1/25 modules; all-module Spearman correlation was -0.121.

### Stability

The ranking was deterministic within the full run but content-sensitive. A
two-case all-layer repeat completed safely with Spearman correlation 0.257,
top-10 overlap 3/10, and top-25 overlap 8/25. The full 12-case order was
frozen for allocation rather than threshold-searched repeatedly. This is a
material limitation: the method is a useful allocation signal, not yet a
validated predictor of held-out model quality.

## 4. Fixed-budget frontier and tournament

Exactly three new serious checkpoints were converted from the frozen ranking:

| candidate | promotions | added payload | payload | effective BPW | 3-bit / 4-bit modules |
|---|---:|---:|---:|---:|---:|
| Q1 | 112 | +150 MiB | 11,088,002,048 B / 10.326507 GiB | 3.2980 | 386 / 112 |
| Q2 | 137 | +300 MiB | 11,245,288,448 B / 10.472991 GiB | 3.3448 | 361 / 137 |
| Q3 | 161 | +450 MiB | 11,402,574,848 B / 10.619475 GiB | 3.3916 | 337 / 161 |

All conversions were bounded and stayed around 3.33 GiB converter peak. No
weights were committed. No sparse 3-to-5/6 matched-size branch was attempted:
MLX support and the safety gate did not justify opening it.

Q1 passed one isolated 37-token factual smoke at 11.728 tok/s and 10.642 GiB
peak with coherent output. The required roughly-1K smoke was not launched
after that process left 12.205 GiB encrypted swap used and 1.107 GiB free.
Q2/Q3 were not loaded. Consequently:

- no Q1/Q2/Q3 mini tournament was completed;
- no new finalist was selected;
- no claim that loss-guided allocation beats g128 or mixed34 is justified;
- the held-out suite was not used to reject the candidates.

## 5. DWQ

### MEASURED FACT

The pinned whole-model DWQ path is unsafe on this device. The minimum source
plus g128 student payload is 61.927 GiB before activations, optimizer state,
targets, or cache. Even g64 plus g128 is 21.143 GiB before those overheads.

A synthetic layer-sized bounded dry run optimized affine 3-bit scales/biases
against fixed teacher outputs: local MSE fell from 0.881265 to 0.718006 in 40
steps (18.5% reduction), with 13,600 bytes maximum MLX allocation. This
validates the toy optimization mechanics only.

### VERDICT

No real 27B DWQ refinement was attempted, and no finalist was improved. A
useful real branch would require a genuinely streamed layer-wise teacher/
student design with new measured residency; it was safely rejected for this
Stage 5 window.

## 6. Final quality Pareto

The NLL values below are the Stage 4 six-anchor teacher-forced subset, not the
held-out suite. The held-out suite supplied objective/semi-objective results;
it was not used to tune candidates.

| model | payload GiB | effective BPW | V2 objective | held-out objective | held-out combined | NLL | quality role |
|---|---:|---:|---:|---:|---:|---:|---|
| g64 | 10.962722 | 3.5012 | 23/24 | 18/21 | 20/24 | 1.4034 | reference |
| g128 | 10.180022 | 3.2513 | 23/24 | 17/21 | 20/24 | 1.3690 | DEFAULT; performance/long-context |
| mixed34 | 10.664947 | 3.4061 | 24/24 | 18/21 | 20/24 | 1.3855 | quality alternate/control, short-to-medium context |
| Q1 | 10.326507 | 3.2980 | not run | not run | not run | not measured | unvalidated research checkpoint |
| Q2 | 10.472991 | 3.3448 | not run | not run | not run | not measured | unvalidated research checkpoint |
| Q3 | 10.619475 | 3.3916 | not run | not run | not run | not measured | unvalidated research checkpoint |

There is no separately justified production QUALITY MODE: mixed34 is the
closest alternate, but its held-out aggregate ties g128 and its NLL is worse.
There is no separate smaller PERFORMANCE MODE: g128 is both the default and
the measured long-context mode.

## 7. Runtime Pareto and context recommendation

Warm values are the Stage 4 persistent-runtime medians; Stage 5 fresh-process
controls are included below where available. A fresh cold prefill at a 16K
prompt necessarily has a much larger TTFT than a warm exact-suffix follow-up.

| model | 8K decode | 8K warm TTFT | 8K peak | 16K decode | 16K warm TTFT | 16K peak | 20K status |
|---|---:|---:|---:|---:|---:|---:|---|
| g128 | 11.105 tok/s | ~1.071 s | 13.983 GiB | 10.409 tok/s | ~1.205 s | 15.236 GiB | not run |
| mixed34 | 10.743 tok/s | 0.989 s | 14.468 GiB | 5.719 tok/s | 1.174 s | 15.721 GiB | not run |

Fresh Stage 5 controls matched the story: g128 was 11.137 tok/s at 8,191
tokens and 10.583 tok/s at 16,727 tokens; mixed34 was 4.890 tok/s at 16,727
tokens. The fresh long-prompt TTFTs were 113.198 s for g128 at 8,191,
242.118 s for g128 at 16,727, and 240.786 s for mixed34 at 16,727.

The conservative practical recommendation is **12K normal KV**. g128 has a
validated 16K persistent path and remains useful there after the host is
healthy, but 16K leaves little system headroom during cold fresh runs. **20K
was not established as practical**: neither normal KV nor KV8 was launched in
Stage 5 after swap pressure became unsafe. **24K was not tested**, and 32K was
explicitly excluded.

## 8. Acceleration and gated branches

| mode | decode/memory evidence | quality/correctness | verdict |
|---|---|---|---|
| g128 normal KV | 11.105 tok/s at 8K; 10.409 tok/s at 16K; 13.983/15.236 GiB peak | persistent exact-suffix validated | default |
| g128 KV8 | prior Stage 4: saves ~0.465 GiB at 16K, costs ~9–11% decode, warm TTFT nearly unchanged | not used in ordinary quality runs | reserve for a future safe 20K probe |
| activation quantization | pinned path supports only NVFP4/MXFP8; affine g128 is rejected before inference | no runtime test | unsupported for current winner |
| DWQ | toy MSE improvement only; 27B dual residency is unsafe | no real finalist refinement | reject for Stage 5 |
| MTP | 15 source tensors, 0.791 GiB BF16, but stripped by pinned runtime | no compatible Qwen3.8 MTP path | stop; do not fabricate a head |

## 9. Serialized agent backend

`research/runtime/qwen38_agent_server.py` implements a single-user,
single-inference OpenAI-compatible service with:

- `POST /v1/chat/completions`;
- `GET /health`, `GET /v1/models`, and `GET /metrics`;
- `POST /v1/sessions/{id}/reset`;
- persistent exact-suffix session state;
- system/user/assistant/tool text messages;
- greedy or sampled generation;
- prompt/suffix/TTFT/decode/cache metrics;
- reset-on-cache-error behavior.

Provider-specific tool-call objects are not parsed; tool-call payloads are
accepted and normalized as text, which keeps the API generic.

Protocol self-tests and the existing runtime token tests pass 9/9 without a
Metal model load. A real 10-exchange coding-agent smoke was not run because
the host had only 861 MiB swap free at final audit. Thus the backend is
protocol/code validated, not model-backed smoke validated.

Run after the host has recovered and no other MLX job is active:

```sh
source research/stage4/canonical_env.sh
uv run --project envs/mlx-0.32.1 --no-sync python \
  research/runtime/qwen38_agent_server.py \
  --model /Users/artem/models/Qwen3.8-27B-3bit-g128-MLX \
  --host 127.0.0.1 --port 8000 --mode greedy
```

## 10. Required answers and remaining work

1. Mixed34's 16K cliff: **yes, reproduced**.
2. Onset: **bounded between 8,191 and 16,727 tokens**, with the slow point at
   15.721 GiB peak; exact onset was safety-limited.
3. Mixed34 versus g128 held out: **not a robust win; 20/24 combined tie**.
4. Held-out composition/hash: **24 cases, hash above**.
5. Sensitivity: **bounded final-logit teacher-KL recovery per added byte**.
6. Stability: **full run deterministic; repeat Spearman 0.257**.
7. Highest families: **linear_attn, then self_attn, then mlp**; top slots listed
   above.
8. Fixed candidates: **Q1/Q2/Q3 at +150/+300/+450 MiB**.
9. Sizes/BPW: **table above**.
10. Broad 3/4 versus sparse 3/5/6: **only broad 3/4 was built**.
11. Tournament survivors: **no new survivor; Q1/Q2/Q3 remained unvalidated**.
12. Bounded DWQ: **toy layer feasible; whole-model 27B path unsafe**.
13. DWQ finalist gain: **none measured**.
14. Final V2: **g64/g128 23/24, mixed34 24/24**.
15. Final held out: **g64/mixed34 20/24 combined, g128 20/24**.
16. Final NLL/KL/top-1: **V2 NLL 1.4034/1.3690/1.3855 for g64/g128/mixed34;
   sensitivity baseline KL 0.115868; V2 top-1 70.06%/69.46%/70.06%**.
17. Pareto: **g128 default/performance, mixed34 alternate, g64 reference**.
18. DEFAULT: **g128**.
19. Separate QUALITY MODE: **no independently justified one; mixed34 is an
   optional quality alternate**.
20. Separate PERFORMANCE/LONG-CONTEXT MODE: **g128 itself**.
21. 8K/16K runtime: **table above**.
22. Activation quantization: **not supported for affine winner; no benefit
   measured**.
23. 20K: **not tested; not established**.
24. 24K: **not tested**.
25. MTP available: **source weights yes; pinned runtime path no**.
26. MTP speedup/acceptance/memory: **not measured**.
27. Serialized backend: **yes, protocol/code validated**.
28. Real agent smoke: **not run because of swap safety**.
29. Today's command: **backend command above, only after host recovery**.
30. Highest-value Stage 6 ideas:
   - validate Q1 first, then at most one frontier finalist on the frozen
     calibration/V2/held-out tournament after a clean OS recovery;
   - investigate the mixed34/g128 Metal residency cliff with allocator and
     cache instrumentation rather than more blind quant formats;
   - implement a genuinely streamed DWQ or activation-friendly quant path and
     measure a safe 20K/KV8 envelope.

## Evidence, safety, and commits

All claims above are tied to raw artifacts under `research/stage5/artifacts/`,
the JSONL ledger `research/stage5/results.jsonl`, the frozen quant artifacts,
and `research/stage5/mtp/static_inspection.json`. The held-out, Stage 4
evaluation/calibration, and cliff prompt hashes were rechecked before this
report. No model weights, source cache, virtual environment, or wired-memory
setting was modified or committed. The final handoff includes the focused
Stage 5 commits shown by the final `git log` and the pre-existing Stage 5
commits `1711a41` and `808f4b2`.

Stage 5 stops here. Stage 6 is not started automatically.
