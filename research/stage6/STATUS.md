# Stage 6 Status

Last updated: 2026-08-26 03:38 +04 (2026-08-25T23:38:00Z)

## Current phase

Phases 0–10 are complete; the final report, raw-artifact audit, and commit are
the remaining Phase 11/12 handoff work. Q1 passed the cold runtime probes
without a 16K cliff and passed the exact-suffix five-turn smokes at
approximately 8K and 16K, but it did not improve the fixed-weight Pareto
frontier. Q2/Q3 were not loaded because Q1 supplied no credible quality, speed,
memory, or context win to justify their larger payloads. The same-data KL
repeat was numerically exact; the alternate calibration subset showed
meaningful rank movement. A bounded one-module DWQ pass on g128 produced
feasibility evidence but no production checkpoint. The selected default is
g128.

## Memory/swap baseline

The initial `memory_pressure` snapshot reported 90% system-wide free memory,
221,405 free pages, 404,428 active pages, 398,562 inactive pages, 108,954
wired pages, zero compressor pages, and zero swap-ins/swap-outs. `vm_stat`
reported the same clean shape. The machine therefore appears recovered from
Stage 5's polluted state. A read-only escalated query confirmed
`vm.swapusage: total = 0.00M, used = 0.00M, free = 0.00M (encrypted)`. The
baseline disk report showed 129,883,828 KiB available on the data volume (~124
GiB).

The first process-list attempts (`pgrep`, `ps`) were sandbox-restricted. The
read-only escalated process table contained no stale MLX/Qwen/Python/uv
workload; the only matching strings were the diagnostic `find` command and the
inspection command itself. The serialized heldout-v2 regressions and Q1
quality/runtime runs temporarily raised swap, but each process exited cleanly.
The latest post-regression check reported 87% free memory and 255.69 MiB swap
used / 768.31 MiB free from a 1 GiB encrypted swap device. The final g128 20K
probe recorded 23% free memory and `vm.swapusage` of 488.62 MiB used / 1,559.38
MiB free at completion; after the process exited, the host returned to 90%
free memory with no heavy process active. The final 20K persistent smoke also
exited cleanly; the latest read-only check reports 91% free memory with no
heavy process active.

## Q1/Q2/Q3 preservation

All three candidates are present on stable persistent storage:

| candidate | path | payload bytes | effective BPW | files verified |
|---|---|---:|---:|---|
| Q1 | `/Users/artem/models/stage5-frontier/Qwen3.8-27B-Q1-kl-g128` | 11,088,002,048 | 3.2980 | 20/20 shards, config, index, tokenizer, chat template |
| Q2 | `/Users/artem/models/stage5-frontier/Qwen3.8-27B-Q2-kl-g128` | 11,245,288,448 | 3.3448 | 20/20 shards, config, index, tokenizer, chat template |
| Q3 | `/Users/artem/models/stage5-frontier/Qwen3.8-27B-Q3-kl-g128` | 11,402,574,848 | 3.3916 | 20/20 shards, config, index, tokenizer, chat template |

The sizes match `research/stage5/quant/frontier_manifests_v1.json`. No copy or
rebuild was needed. The user-owned untracked `archive_3.8_stage5.zip` is
untouched.

## Current candidate frontier

g128 is the fixed-weight and final default. Q1 is not a Pareto improvement;
Q2/Q3 remain preserved but unvalidated and were correctly skipped by the Q1
gate. Stage 5/Stage 6 evidence retains g64 as the reference and mixed34 as a
quality/control model with a reproduced 16K decode cliff.

## Quality-suite corrections

Frozen as `research/stage6/heldout/heldout_v2.jsonl` with SHA-256
`1749a8b47e99bb3e0e04347d75e24d7022ed3ab8d5ffa5bc56926b9cf02cbcbd`.
The nine targeted coding/reasoning cases have 1024-token generation budgets;
prompts, answer regexes, semantic answers, and scorer are unchanged. The
correction record is `research/stage6/heldout/heldout_v2_manifest.json`.

Final-hash full merges are: Q1 17/21 objective + 2/3 semi-objective, g64
19/21 + 2/3, g128 20/21 + 3/3, and mixed34 18/21 + 2/3. There were no
exceptions or early-EOS events; repetition flags were Q1 2, g64 2, g128 1,
and mixed34 3. Raw inspection shows Q1 coding-04 and reasoning-02 are not
clean semantic failures (required content / Final answer is present), while
coding-06, instruction-02, and prose-01 remain substantive misses.

Q1's corrected Quality Suite V2 result is 22/24 objective + 4/4
semi-objective, versus g128's 23/24 + 4/4. This is not a quality win.

Q1 teacher-forced evaluation on the six-case representative target subset
completed with no exceptions: weighted NLL 1.415419 nats over 167 target
tokens and weighted top-1 agreement 0.682635. The corresponding g128 artifact
is 1.369012 nats and 0.694611, so Q1 is currently slightly worse on both
teacher-forced measures. Q1 artifact:
`research/stage6/artifacts/teacher-forced/Q1/evaluation.json`.

Q1 fresh cold probes were coherent at all requested contexts: 8K 11.013
tok/s decode / 106.48 s TTFT / 14.13 GiB peak; 12K 10.443 tok/s / 164.26 s /
14.76 GiB; 16K 10.260 tok/s / 232.92 s / 15.38 GiB. The g128 8K reference is
11.137 tok/s / 113.20 s / 13.98 GiB, and the Stage 5 g128 16K reference is
10.583 tok/s / 242.12 s / 15.24 GiB. Q1 is therefore slightly slower on
decode and heavier at 16K despite a somewhat faster prefill in these fresh
processes; there is no Q1 long-context win.

Q1 exact-suffix persistent chat passed five turns at both tested envelopes.
The 8K smoke reached logical context 7,825 with warm TTFT 0.633–0.641 s and
decode 11.239–11.275 tok/s; the 16K smoke reached 15,731 with warm TTFT
0.695–0.701 s and decode 10.511–10.580 tok/s. Both had no replay flags, exact
cache-offset checks, and reset returned logical context to zero. Peak MLX
memory was 14.84 GiB at 8K and 16.18 GiB at 16K.

### Q1 gate decision

Q1 is rejected as the default and as an optional quality mode: it is larger,
slightly worse on teacher-forced NLL/top-1, behind g128 on both corrected
formal suites, and not faster or safer at 16K. Q2 and Q3 are explicitly
skipped under the Stage 6 gate; their larger payloads do not address an
observed Q1 failure mode that could plausibly improve the practical frontier.
The fixed-weight winner entering later phases is g128.

## DWQ status

The pinned whole-model DWQ path remains rejected because full teacher/student
residency is unsafe. Stage 6 first reproduced the safe synthetic gate, then
ran a genuinely streamed one-module pass on selected g128 module
`language_model.model.layers.60.linear_attn.in_proj_a`. At 1e-3 the local
optimizer diverged to infinity; that failure is preserved in
`research/stage6/artifacts/dwq/g128-layer60-in_proj_a.json`. At 1e-6, 40
scale/bias steps reduced local source-output MSE 0.253556 -> 0.218637
(13.77%) and improved one calibration case's end-to-end KL 0.091406 ->
0.090705. Peak MLX memory was 10.681 GiB with one 983,040-byte source tensor;
no full teacher was resident and no wired-limit call occurred. The partial
scales/biases artifact is
`research/stage6/artifacts/dwq/g128-layer60-in_proj_a-stable-partial.safetensors`.
No full refined model was materialized, so no heldout quality improvement is
claimed; g128 remains the default.

## KL-ranking reproducibility

The controlled Stage 6 repeat measured 45 modules (Stage 5 ranks 1–25,
245–254, and 489–498) against the exact 12-case Stage 5 calibration set and
hash. It completed in 465.289 s with 14.480 GiB peak and 12.302 GiB
active+cache. Compared with the corresponding Stage 5 scores, Spearman was
1.000, top-10 overlap 10/10, top-25 overlap 25/25, sign agreement 100%, and
mean/max absolute score delta were both zero.

The same 45 modules on the separate `cal-instruction-01` plus
`cal-reasoning-01` subset completed in 101.988 s with 13.932 GiB peak.
Compared with the full 12-case ranking, Spearman was 0.410, top-10 overlap
5/10, top-25 overlap 20/25, and sign agreement 73.3%. This is calibration-set
robustness evidence, not numerical nondeterminism. Artifacts:
`research/stage6/artifacts/kl-repro/{same-data-repeat,alternate-2case,analysis}.json`.

## Context envelope

The final g128 normal-KV 20K cold probe was coherent at exactly 20,480 prompt
tokens: prompt throughput 72.688 tok/s, TTFT 281.756 s, decode 10.367 tok/s,
active+cache 11.622 GiB after completion, and 15.859 GiB MLX peak. At the
completion snapshot the host had 23% free memory and 488.62 MiB swap used;
memory recovered to 90% free after exit. This establishes 20K as a one-shot
upper-envelope mode, not the recommended daily context. A follow-up smoke also
passed three exact-suffix turns, reaching logical context 20,686 with warm TTFT
0.724–0.726 s, warm decode 10.419–10.433 tok/s, exact cache offsets, no replay
flags, and reset to zero. The recommended normal-KV context is 16K; 12K is the
safer sustained-work context. A 24K smoke was not run because the 20K peak
leaves insufficient projected headroom. 32K remains explicitly out of scope.

Artifacts: `research/stage6/artifacts/context/g128-20k-normal/result.json` and
`research/stage6/artifacts/context/g128-20k-persistent/result.json`.

## MTP status

Static Stage 6 header inspection confirms 15 top-level `mtp.*` BF16 tensors in
the source, totaling 849,398,784 bytes (~0.791 GiB). They comprise one
`mtp.layers.0` attention/MLP block plus `mtp.fc`, normalization, and projection
weights; exact names/shapes are frozen in
`research/stage6/mtp_static_inspection.json`. The pinned `qwen3_5.py` strips
`mtp.*`, exposes no native Qwen3.8 MTP head/reference decode path, and its
generic speculative path is not an MTP implementation. No conversion, head
fabrication, or MTP decode smoke was attempted. Verdict: NOT READY / UNSUPPORTED
for the current affine g128 runtime. Ideal raw payload estimates are ~0.396
GiB at 8-bit, ~0.198 GiB at 4-bit, and ~0.148 GiB at 3-bit before scales and
metadata; these are estimates, not measured artifacts.

## Agent-backend status

The serialized backend passed the Stage 6 real 27B smoke using g128: ten
sequential exchanges, system/user/assistant/tool roles, two tool-call/result
transcript injections, exact suffix/cache-offset checks, `/health`,
`/v1/models`, `/metrics`, and session reset. All ten requests returned 200, all
replay flags were false, and reset returned message count and logical context
to zero. The transport accepts tool payloads as normalized text; native
provider-specific tool-call generation is not claimed. Artifact:
`research/stage6/artifacts/agent/g128-serialized-smoke.json`.

## Final handoff

Fixed-weight winner: g128. Q1 is rejected; Q2/Q3 are preserved and skipped.
Bounded DWQ is a research feasibility result only. MTP is not ready in the
pinned runtime. The serialized server is validated for serialized text/tool
role transport. Remaining work is to write `FINAL_REPORT.md`, append the final
ledger records, audit paths/results/git state, and commit code/reports without
including weights or transient giant traces.
