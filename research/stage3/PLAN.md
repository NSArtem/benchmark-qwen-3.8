# Stage 3 plan

This namespace records the current unattended run. It follows `TASK.md` and
does not rewrite prior `research/` or `research/night2/` evidence.

## Gates and order

1. Phase 0: recover raw state and establish the baseline.
2. Phase 1: compare fresh one-shot, fresh segmented, and true persistent
   exact-suffix execution at identical logical boundaries.
3. Phase 2: only if the Phase 1 B-vs-C gate shows material divergence,
   localize the first differing cache boundary/layer and rerun the oracle.
4. Phase 3: only after persistent exact-suffix validation, scale g128 through
   approximately 4K, 8K, 12K, and 16K with bounded multi-turn prompts.
5. Phase 4: only if validated 8K works, run a bounded 20–30 turn endurance
   session and stop on the first repeated failure or safety condition.
6. Phase 5: independently compare g64 and source-backed g128 on a diverse,
   deterministic quality suite.
7. Phase 6: only after cache validation, measure KV8 at approximately 16K.
8. Phase 7: only if the measured headroom and stability gates justify it, make
   one bounded approximately-20K probe; do not chase 24K/32K.
9. Phase 8: only if all primary questions are closed, perform one low-memory
   decode-acceleration feasibility/measurement study.

## Phase 1 outcome

The 4K g128 B-vs-C gate passed. Phase 2 root-cause localization is therefore
skipped for this run; the A-vs-C divergence is treated as one-shot versus
segmented numerical sensitivity, not persistent-cache corruption. Phase 3 is
authorized with the exact-suffix protocol and normal KV.

## Runtime invariants

- Canonical command prefix remains `uv run --project envs/mlx-0.32.1
  --no-sync ...`.
- The archive-backed invocation exposes the already-pinned MLX 0.32.1 and
  mlx-lm revision read-only because the project venv contains no installed
  packages. It does not run `uv sync`, `uv lock`, `uv update`, or `uvx`.
- The direct `generate_step` no-wired helper is used. No call to
  `mx.set_wired_limit()` or system wired-memory setting is permitted.
- Large MLX/Qwen workloads are serialized. Fresh-process comparisons are kept
  separate from persistent-process measurements.
- Model weights and environments remain outside commits.
