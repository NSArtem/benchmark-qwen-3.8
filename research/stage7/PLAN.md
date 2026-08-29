# Qwen3.8 Stage 7 Plan

This plan follows `TASK.md` and records the gated execution order. Stage 7 is
not a new static quantization search.

## Baseline and safety

- Primary student: `/Users/artem/models/Qwen3.8-27B-3bit-g128-MLX`
- Primary environment: `envs/mlx-0.32.1`, MLX 0.32.1, mlx-lm revision
  `254d153fdeb6f150edd4fc5a54f9828638481fa8`
- Source/reference: `/Users/artem/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- No wired-memory setting changes, no concurrent heavy model jobs, and no
  overwrite of the canonical checkpoint.
- Re-read `AGENTS.md`, `TASK.md`, and `research/stage7/STATUS.md` before each
  major phase.

## Gates

### Phase 0 — recover Stage 6 and establish baseline

Inspect Stage 6 reports, raw artifacts, model paths, runtime paths, git state,
and a clean host memory snapshot. Do not repeat Stage 6 headline benchmarks.

### Phase 1 — freeze diverse calibration/evaluation protocol

Create disjoint deterministic local DWQ train and validation corpora covering
eight usage domains. Freeze their manifests and SHA-256 hashes. Reuse the
corrected Stage 6 `heldout_v2`/Quality Suite V2 only for final evaluation.
Measure teacher/student baseline KL/NLL/top-1 on train and validation.

Gate: no final-heldout case is used for optimization, ranking, or tuning.

### Phase 2 — productionize and reproduce bounded DWQ

Recover the Stage 6 streamed one-module primitive. Refactor it into a small
reusable harness with explicit module lists, conservative LR/steps,
checkpoint/resume, disjoint train/validation data, deterministic ordering,
per-module metrics, and safe source loading. First reproduce layer 60
`linear_attn.in_proj_a`; reject non-finite optimization and retain rollback.

Gate: finite loss, safe memory, and consistent improvement direction.

### Phase 3 — incremental DWQ scaling

Use the Stage 5/6 sensitivity ranking only as an initial priority. Refine
top-4, then top-8, top-16, and top-32 only when validation and practical
sanity checks pass. Preserve each best validation state without duplicating
large weight trees. Stop after two validation regressions, clear train/valid
divergence, generation degradation, instability, or excessive cost.

### Phase 4 — calibration robustness ablation

On a small fixed top-4/top-8 subset, compare diverse, coding-heavy, and
reasoning-heavy calibration against the frozen diverse validation domains.
Stop early if diverse calibration clearly dominates. Do not inspect final
heldout outputs during this choice.

### Phases 5–6 — final DWQ quality and runtime gates

Evaluate only g128 and the best internally selected DWQ candidate on corrected
heldout_v2, Quality Suite V2, teacher-forced NLL/KL/top-1, and selected output
disagreements. Promote only a meaningful same-payload quality win without
runtime regression. If it survives, compare exact-suffix runtime at about 8K
and 16K.

### Phase 7 — derive the exact MTP architecture

Inspect source config/index/tensors, pinned runtime source, official/reference
code available locally or via current authoritative source, and related Qwen3.x
implementations. Write `mtp/ARCHITECTURE.md` mapping every tensor and cache /
state dependency. Do not write generation code from names alone.

Gate: computation graph and acceptance/rollback semantics must be supported;
otherwise classify MTP unsupported and continue to agent validation.

### Phases 8–10 — exact MTP prototype, quantization, performance

If Phase 7 passes, build a minimal state-copying prototype, test greedy token
identity and rejection/rollback (including recurrent and attention state), then
test at short, 4K, 8K, and 16K. Try at most one 4-bit MTP artifact, and one
3-bit artifact only if 4-bit is clearly useful. Measure speed only after exact
correctness passes and classify the resulting mode conservatively.

### Phase 11 — real serialized coding-agent benchmark

Use isolated deterministic toy repositories and the serialized backend for
bug diagnosis, failed-patch recovery, multi-file invariants, and structured
tool transport. Separate protocol correctness from native model tool-choice
quality. Do not tune DWQ against these final practical cases.

### Phases 12–13 — integrate validated modes and report

Expose only validated flags/modes, keep one serialized session and exact
suffix semantics, update all reports and JSONL evidence, inspect raw curves,
outputs, MTP logs, agent transcripts, paths, git status/log, and commit useful
code/fixtures/reports without weights or giant transient tensors. Do not start
Stage 8.

## Decision record

Every experiment must state its question, baseline, controlled variables,
decision gate, raw artifact paths, measured result, and conclusion. Reports
label `MEASURED FACT`, `INFERENCE`, and `HYPOTHESIS` explicitly.
