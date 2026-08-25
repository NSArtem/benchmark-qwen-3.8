# Qwen3.8 Stage 6 Plan

Stage 6 exploits the fixed Stage 5 KL-guided frontier. It does not open a new
quantization search.

## Completion state

Phases 0–10 are complete. Q1 was evaluated and rejected; Q2/Q3 were preserved
and skipped by the explicit gate. KL reproducibility, bounded DWQ feasibility,
the 20K context gate, static MTP investigation, and the real serialized 27B
agent smoke are recorded. Phase 11/12 is the final audit, report, and commit.

## Safety and environment

- Canonical environment: `envs/mlx-0.32.1` with `uv run --project envs/mlx-0.32.1 --no-sync ...`.
- Never call `mx.set_wired_limit()` or modify `iogpu.wired_limit_mb`.
- Run at most one heavy Qwen/MLX process at a time.
- Stop on Metal OOM, GPU timeout, UI instability, exhausted swap, or uncontrolled memory growth.
- Re-read `AGENTS.md`, `TASK.md`, and `research/stage6/STATUS.md` before each major phase.

## Phase gates

1. **Phase 0 — clean baseline and recovery.** Record memory/swap/process/disk state; preserve Q1/Q2/Q3; verify metadata and manifests.
2. **Phase 1 — heldout_v2.** Inspect the Stage 5 raw outputs and scorer. Correct only truncation/budget defects symmetrically, record the old/new rule and hash, and freeze the suite before candidate selection.
3. **Phase 2 — Q1 first.** Run corrected V2/held-out/NLL-KL evidence, then 8K/12K/16K performance and persistent checks against g128. Inspect all disagreements and gate Q2/Q3 from actual evidence.
4. **Phase 3 — justified Q2/Q3.** Run only the mini gates justified by Q1. Do not spend a full tournament on candidates that cannot improve the Pareto frontier.
5. **Phase 4 — KL reproducibility.** Repeat selected modules on the identical calibration data, then separately compare an alternative subset. Report numerical repeatability separately from calibration robustness.
6. **Phase 5 — fixed-weight decision.** Choose default, optional quality mode, long-context/performance mode, reference, and rejected candidates.
7. **Phase 6/7 — bounded DWQ.** After the fixed-weight decision, perform a safe one-layer/small-group dry run and expand only if streamed memory and loss gates pass. Refine at most one main student.
8. **Phase 8 — context envelope.** Test 20K normal KV, KV8 only if needed and safe, and at most one gated 24K smoke. Never 32K.
9. **Phase 9 — MTP.** Reverse-engineer the existing Qwen3.8 tensors and reference path; do not train a head. Prototype only if correctness and memory gates are satisfied.
10. **Phase 10 — agent backend.** Run a real serialized 27B coding-agent-style multi-turn smoke with the final default; document structured tool-call limits.
11. **Phase 11/12 — production handoff.** Update runtime/docs/tests, audit raw artifacts and paths, write `FINAL_REPORT.md`, update `STATUS.md`/`RESULTS.md`, and commit useful code/evidence only (never weights).

## Current artifact paths

- g128: `/Users/artem/models/Qwen3.8-27B-3bit-g128-MLX`
- g64 reference: `/Users/artem/.cache/huggingface/hub/models--lukaskrem...--Qwen3.8-27B-3bit-MLX-TextOnly/snapshots/c98bba5926f51fec1c8d8737e577221673f524d7`
- mixed34 control: `/Users/artem/models/Qwen3.8-27B-mixed-3-4-control-g128`
- Q1: `/Users/artem/models/stage5-frontier/Qwen3.8-27B-Q1-kl-g128`
- Q2: `/Users/artem/models/stage5-frontier/Qwen3.8-27B-Q2-kl-g128`
- Q3: `/Users/artem/models/stage5-frontier/Qwen3.8-27B-Q3-kl-g128`

The Q1/Q2/Q3 weights are preserved outside git and must not be overwritten or
deleted before the final report.
