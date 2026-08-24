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

## Phase 5 outcome

The independent fresh-process suite completed 38 deterministic cases for each
model across coding, instruction following, structured output, reasoning,
retrieval, and prose continuity. Objective pass rates were 25/38 (65.8%) for
g64 and 22/38 (57.9%) for g128. Instruction, structured, retrieval, and prose
categories were tied; g128 trailed in coding (0/10 vs 2/10) and reasoning
(3/6 vs 4/6). The coding evaluator used a 64-token cap, so several raw g128
answers were relevant but truncated; this makes the suite a conservative
objective signal rather than a claim of a 20-point intrinsic coding defect.
Prior 4K teacher-forced NLL also favored g64 by 1.6947 nats per target, as
supplementary evidence. The decision is `PERFORMANCE/LONG-CONTEXT MODE` for
g128: useful memory saving/speed, with measurable quality risk. Phase 6 is
authorized as the final primary comparison.

## Phase 6 outcome

The real 16K-class persistent conversation was run in two separate fresh child
processes with g128: normal KV and KV8/group-64/start-0. Both completed five
coherent, non-replaying turns with exact suffix accounting and no exceptions.
Normal KV peaked at 15.236 GiB; KV8 peaked at 14.771 GiB, saving 0.465 GiB.
Warm median TTFT was effectively identical at 1.169 s, while mean warm decode
fell from 10.513 to 9.522 tok/s (9.4% penalty). The 16 attention layers were
`QuantizedKVCache` under KV8 and the 48 recurrent layers remained `ArraysCache`.
The result is `SELECTABLE_LONG_CONTEXT_MODE`: useful as a 16K safety lever,
not a normal default. The quality of the KV8 turn-5 answer was coherent and
current-turn relevant, though its implementation recommendation diverged
from normal KV; this is recorded as quantization sensitivity, not cache
corruption.

## Phase 7/8 gate outcome

No 20K probe was run. The repository has no natural 20K prompt family, normal
16K is already 15.236 GiB, and a synthetic extension would add an unsafe,
low-value probe after the endurance quality stop. The stage therefore stops at
the measured 16K-class envelope. Optional decode acceleration is skipped
because the Phase 4 endurance gate did not pass and no urgent primary issue
should be displaced by a bonus experiment.
