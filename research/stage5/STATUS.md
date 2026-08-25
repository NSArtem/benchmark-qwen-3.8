# Stage 5 Status

Last updated: 2026-08-25 — Phase 1 complete under the safety gate; Phase 2 is next.

## Current phase

Phase 0 and the safety-limited Phase 1 cliff study are complete. No Stage 5
candidate weights have been created. The exact onset below 16K is intentionally
unmeasured because repeated 27B prefills accumulated dangerous swap pressure.

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

## Current Pareto frontier

| role | model/path | payload | quality evidence | runtime evidence |
|---|---|---:|---|---|
| reference | g64: `/Users/artem/.cache/huggingface/hub/models--lukaskremla--Qwen3.8-27B-3bit-MLX-TextOnly/snapshots/c98bba5926f51fec1c8d8737e577221673f524d7` | 11,771,132,928 B / 10.962722 GiB | V2 23/24 objective, 4/4 semi | Stage 4 long-context not rerun |
| performance candidate | g128: `/Users/artem/models/Qwen3.8-27B-3bit-g128-MLX` | 10,930,715,648 B / 10.180022 GiB | V2 23/24 objective, NLL 1.3690 | prior 8K/12K/16K-class decode 11.105/10.803/10.409 tok/s |
| default candidate | mixed34: `/Users/artem/models/Qwen3.8-27B-mixed-3-4-control-g128` | 11,451,399,168 B / 10.664947 GiB | V2 24/24 objective, 4/4 semi, NLL 1.3855 | prior 8K/16K-class decode 10.743/5.719 tok/s |

## Candidate model paths/sizes

No new Stage 5 model exists yet. Do not count Stage 4 Candidate B as a Stage
5 candidate; it was rejected at the Stage 4 mini gate.

## Held-out suite

Not created or hashed yet. It must be frozen before any Stage 5 candidate
selection and must remain absent from calibration/sensitivity inputs.

## Rejected branches

Uniform 2-bit, `mixed_2_6`, generic quant-zoo work, repeated uniform 4-bit,
DSpark, and 32K remain closed. Stage 4 sensitivity Candidate B remains
rejected. No DWQ, activation-quant, or MTP run has occurred in Stage 5.

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

## Next action

Create and freeze the completely held-out Stage 5 quality suite, using no
Stage 5 calibration or candidate-selection data from it.
