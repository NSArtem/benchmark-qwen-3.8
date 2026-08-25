# Stage 5 Status

Last updated: 2026-08-25 — static acceleration gates recorded; backend and final synthesis remain.

## Current phase

Phase 0, the safety-limited Phase 1 cliff study, the Phase 2 baseline
comparison, Phase 3's bounded g64-teacher/g128-student KL/logit ranking, the
Phase 4 Q1/Q2/Q3 conversions, the Phase 5/7 tournament safety gates, the Phase
6 bounded DWQ feasibility gate, and the static Phase 8/9/10 acceleration gates
are complete. Candidate smoke validation is safety-limited after Q1's first
short run drove encrypted swap to 12.2 GiB. The exact cliff onset below 16K is
intentionally unmeasured because repeated 27B prefills accumulated dangerous
swap pressure.

## Completed experiments

- Read `AGENTS.md` and `TASK.md` completely before taking task actions.
- Recovered Stage 4 final report, status, results, and machine-readable ledger.
- Verified the clean starting worktree and recent Stage 4 commit history.
- Verified the exact g64, g128, mixed34, high-precision source, bounded
  converter, and qwen38-chat paths are present.
- Initialized the Stage 5 plan, status, results, and JSONL ledger.
- Frozen a same-source cliff prompt family at 8,191/10,240/12,288/14,336/
  16,727 chat tokens; source SHA-256 is
  `67dcccb037993b679c5593073bbc1875ad61541827a6fc3bb6720410b8096903`.
- `P1A-16K-G128-R1` completed coherently at 10.583 tok/s with 15.236 GiB
  MLX peak. The interleaved mixed34 process loaded to 10.666 GiB but was
  interrupted before first token when system swap reached ~14.4 GiB; this is a
  safety stop, not a model OOM verdict.
- The gated isolated retry `P1A-16K-MIXED34-RETRY` completed coherently at
  4.890 tok/s, 15.721 GiB MLX peak, 11.873 GiB active+cache, and 69.470
  prefill tok/s. Its prompt and output token hashes match the Stage 4 mixed34
  16K artifact, so the ~5 tok/s result is reproducible across fresh processes.
- The fresh 8K g128 point `P1B-8192-G128` completed coherently at 11.137
  tok/s and 13.983 GiB peak. The following mixed34 8K process was stopped
  during load/setup at 32% free memory and ~13.7 GiB swap.
- Frozen Stage 5 held-out suite v1: 24 cases, categories coding 6,
  instruction 4, structured 4, reasoning 4, retrieval 3, prose 3; six
  teacher-forced anchors. Suite SHA-256:
  `6fe7bd7da6732e9057a75764292b18b29c3262a96eb39391b700f038f641851b`.
  A direct prompt and ID overlap check against Stage 4 calibration/evaluation
  passed. This suite is final-selection evidence only.
- Fresh final-selection baselines completed for all three entering models. g64
  scored 18/21 objective and 2/3 semi-objective; g128 scored 17/21 and 3/3;
  mixed34 scored 18/21 and 2/3. Every run had zero exceptions, early-EOS
  events, and repetition flags. Raw per-case outputs and diagnostics are under
  `research/stage5/artifacts/heldout/{g64,g128,mixed34}`.

## Current Pareto frontier

| role | model/path | payload | quality evidence | runtime evidence |
|---|---|---:|---|---|
| reference | g64: `/Users/artem/.cache/huggingface/hub/models--lukaskremla--Qwen3.8-27B-3bit-MLX-TextOnly/snapshots/c98bba5926f51fec1c8d8737e577221673f524d7` | 11,771,132,928 B / 10.962722 GiB | V2 23/24; held-out 18/21 objective, 2/3 semi | Stage 4 long-context not rerun |
| performance/default candidate | g128: `/Users/artem/models/Qwen3.8-27B-3bit-g128-MLX` | 10,930,715,648 B / 10.180022 GiB | V2 23/24; held-out 17/21 objective, 3/3 semi, NLL 1.3690 | fresh 8K/16K 11.137/10.583 tok/s; no cliff observed |
| quality alternate | mixed34: `/Users/artem/models/Qwen3.8-27B-mixed-3-4-control-g128` | 11,451,399,168 B / 10.664947 GiB | V2 24/24; held-out 18/21 objective, 2/3 semi, NLL 1.3855 | fresh 16K 4.890 tok/s; cliff reproduced |

## Candidate model paths/sizes

Q1/Q2/Q3 are temporary Stage 5 research checkpoints outside the repository;
weights are not committed. None has advanced to a Stage 5 finalist because the
required multi-prompt smoke gate was stopped for system swap safety. Do not
count Stage 4 Candidate B as a Stage 5 candidate; it was rejected at the Stage
4 mini gate.

Temporary candidate paths and sizes:

- Q1: `/private/tmp/qwen38-stage5-Q1-kl-g128`, 11,088,002,048 B,
  10.326507 GiB, 3.2980 BPW.
- Q2: `/private/tmp/qwen38-stage5-Q2-kl-g128`, 11,245,288,448 B,
  10.472991 GiB, 3.3448 BPW.
- Q3: `/private/tmp/qwen38-stage5-Q3-kl-g128`, 11,402,574,848 B,
  10.619475 GiB, 3.3916 BPW.

## Held-out suite

Frozen v1 at `research/stage5/heldout/heldout_v1.jsonl`; SHA-256
`6fe7bd7da6732e9057a75764292b18b29c3262a96eb39391b700f038f641851b`.
It is excluded from all calibration, sensitivity, allocation, and mini-
tournament inputs.

Baseline raw summaries:

- g64: `research/stage5/artifacts/heldout/g64/logs/summary.json`, 18/21
  objective, 2/3 semi-objective, 528.175 s.
- g128: `research/stage5/artifacts/heldout/g128/logs/summary.json`, 17/21
  objective, 3/3 semi-objective, 485.303 s.
- mixed34: `research/stage5/artifacts/heldout/mixed34/logs/summary.json`,
  18/21 objective, 2/3 semi-objective, 505.363 s.

## Rejected branches

Uniform 2-bit, `mixed_2_6`, generic quant-zoo work, repeated uniform 4-bit,
DSpark, and 32K remain closed. Stage 4 sensitivity Candidate B remains
rejected. Whole-model upstream DWQ is rejected by the memory/safety gate; its
bounded synthetic dry run passed numerically. Activation quantization was not
run because the pinned path rejects affine g128, while MTP weights are present
but stripped by the pinned runtime and lack a native Qwen3.8 path.

## Safety observations

- Starting git worktree was clean on `main`.
- `P1A-16K-G128-R1` ended with 22% free-memory report and ~11.2 GiB swap;
  the interrupted mixed34 process ended with 31% free-memory report and
  ~14.4 GiB swap. Current free-memory report recovered to 88% after stopping.
- The six-run 16K interleave was stopped after the first g128 and an interrupted
  mixed34 attempt; do not repeat it blindly. The one isolated mixed34 retry
  completed safely enough to establish the cliff. Phase 1 is now closed.
- Fresh runs repeatedly drove the system from roughly 89% free to 22–32% free
  and grew swap to 12–14 GiB even though MLX peaks stayed below the 16.2 GiB
  unattended stop. No Metal OOM occurred; two processes were intentionally
  interrupted before prefill completion.
- No wired-memory setting has been changed; no `mx.set_wired_limit()` call is
  permitted in the canonical runtime.
- The three held-out runs each completed in a fresh serialized process and the
  system recovered to 89% free memory afterward.
- The full sensitivity process stayed below the 15 GiB active+cache guard and
  completed at 14.480 GiB peak. The first Q1 smoke began at 87% free memory but
  ended at 26% free with 12.205 GiB encrypted swap used; the next heavy Q1
  process was stopped before launch. Do not work around this gate by launching
  Q2/Q3 or by running concurrent jobs.
- Pinned learned-quant source inspection is recorded in
  `research/stage5/quant/upstream_inspection.json`. Dynamic Quant and DWQ are
  unsafe as whole-model flows on this machine because they duplicate 27B
  residency; the Stage 5 path is serialized, calibration-only, and removes any
  wired-limit behavior. The held-out suite is not an input.
- Phase 3 teacher capture used g64 logits for all 12 Stage 4 calibration cases;
  the local float16 tensor payload is retained but ignored by git. The bounded
  one-module-at-a-time sweep evaluated all 498 quantization modules in 4,579.2
  s, with 14.480 GiB peak and 12.302 GiB maximum active+cache. Its primary
  signal is final-logit KL recovery per added byte.
- The frozen full-split ranking hash is
  `e9d561a11807810dc44a9a6e38080623f2b41207d03643515b0661c8a188dd6b`.
  A two-case all-layer repeat had Spearman 0.257 with the full ranking, with
  top-10 overlap 3/10 and top-25 overlap 8/25; the full split is retained as
  the deterministic allocation order, while this instability is a validation
  warning.
- Phase 4 converted exactly three temporary frontier checkpoints: Q1 +150 MiB
  (112 promotions, 11,088,002,048 B), Q2 +300 MiB (137, 11,245,288,448 B),
  and Q3 +450 MiB (161, 11,402,574,848 B). Q1's isolated short factual smoke
  was coherent at 11.728 tok/s with 10.642 GiB peak. A second Q1 smoke was
  refused after the first process left 12.205 GiB encrypted swap used and only
  1.107 GiB swap free; Q2/Q3 have not been advanced to finalists.
- The pinned upstream DWQ flow was not launched because it retains a full
  teacher/student path and calls `mx.set_wired_limit()`. The synthetic bounded
  scale/bias dry run reduced local MSE 18.5% in 40 steps with 13,600 bytes MLX
  peak. Static source-plus-g128 minimum residency is 61.927 GiB, so a real
  whole-model DWQ run is rejected; only a future genuinely streamed layer-wise
  implementation could reopen it.
- The Q1 multi-prompt smoke gate and final candidate tournament were not
  launched after the swap safety stop; the held-out suite was not used for that
  decision. Q1/Q2/Q3 remain unvalidated temporary checkpoints.
- Pinned activation quantization supports only `nvfp4`/`mxfp8`, not the affine
  3-bit/group-128 winner; no format-changing activation branch was opened.
- Static MTP inspection found 15 BF16 `mtp.*` source tensors totaling
  849,398,784 B / 0.791 GiB, but the pinned Qwen3.5 sanitizer strips them and
  exposes no native Qwen3.8 MTP path. No conversion or speculative smoke was
  attempted.
- No 20K or 24K run was launched after the swap safety gate; 32K remains
  explicitly closed.
- The serialized backend `research/runtime/qwen38_agent_server.py` and its
  protocol tests pass 9/9 in the canonical environment without loading MLX.
  It exposes serialized OpenAI-compatible chat, reset, health, models, and
  metrics routes; a real model-backed agent smoke remains deferred by the swap
  gate.

## Next action

Perform the final raw-artifact audit and report. If a safe single-process
window returns before finalization, validate Q1 first; otherwise keep
g128/mixed34 as the only advanced quality/runtime roles and report Q1/Q2/Q3 as
unvalidated research checkpoints.
