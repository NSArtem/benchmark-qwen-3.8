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

## Phase 3 outcome

The corrected exact-suffix ladder passed at 8K, 12K, and 16K-class context
with five coherent, non-replaying turns at each context. Persistent cache
offsets reached 8,676, 12,760, and 17,219 respectively, with peak memory
13.983, 14.609, and 15.236 GiB. Warm TTFT stayed below 1.224 s and decode
averaged about 11.105, 10.803, and 10.409 tok/s. Corrected fresh semantic
oracles passed at 8K turn 2 and 16K turn 5. Phase 4 endurance is authorized;
KV8 remains gated to a later 16K safety comparison.

## Phase 4 outcome

The 25-turn endurance attempt stopped at the first BROKEN output, turn 17,
after 17 sequential assistant turns and logical offset 10,443. Turns 1–16
were coherent, non-replaying, and exactly suffix-accounted. Turn 17 generated
only four tokens (`The smallest worker` plus EOS), so the 20–30-turn endurance
gate did not pass. Memory, state signatures, and protocol accounting remained
normal through the failure; this is recorded as a quality/early-termination
failure rather than cache corruption. Do not continue this branch unattended.
