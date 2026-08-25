# Qwen3.8 Stage 5 Research Plan

Stage 5 is an evidence-gated optimization campaign for an 18 GB Apple
Silicon machine. The objective is a practical Pareto frontier, not a quant
format collection.

## Guardrails

- Canonical commands use `uv run --project envs/mlx-0.32.1 --no-sync ...`.
- No canonical-environment upgrade, wired-memory setting, concurrent heavy
  Qwen job, source download, or committed model weight.
- The Stage 5 held-out suite is frozen before any Stage 5 candidate selection
  and is not used for calibration, sensitivity, or mini-tournament decisions.
- At most four serious new Stage 5 weight candidates and at most two new
  finalists; no rejected Stage 3 branches or 32K tests.
- Any Metal OOM, GPU timeout, repeated allocation failure, severe system
  instability, or unexplained memory growth stops that branch.

## Gates and execution order

1. **Phase 0 — recover state.** Verify Stage 4 artifacts, model paths,
   bounded converter, source snapshot, and runtime; initialize the Stage 5
   ledger.
2. **Phase 1 — cliff localization.** Run fresh, interleaved g128/mixed34
   controls at 16K and a compact 8K–16K sweep. Record memory and system
   pressure. This result constrains every later candidate budget.
3. **Phase 2 — held-out suite.** Author, manually verify, hash, and freeze a
   balanced 20–30-case suite. Run only g64/g128/mixed34 initially. Keep this
   split isolated from all allocation and selection work.
4. **Phase 3 — end-to-end sensitivity.** Inspect pinned upstream code and
   implement a bounded local-data loss/logit/gradient proxy. Save scores,
   ranking hash, determinism check, and calibration-only validation.
5. **Phases 4–5 — fixed-budget frontier.** Build Q1/Q2/Q3 only when the
   ranking and cliff gate justify them, using explicit frozen module lists.
   Smoke each fresh process, then select at most two new finalists using only
   non-held-out evidence.
6. **Phase 6 — DWQ gate.** Calculate residency, adapt only if safe, perform a
   1–2-layer dry run, and stop if there is no measurable local/end-to-end gain
   or if memory is not clearly safe.
7. **Phases 7–9 — final/runtime.** Run full V2 plus the untouched held-out
   suite, assign Pareto roles, test activation quant only on the winner, and
   probe 20K with normal KV and KV8 only if needed. One 24K smoke is allowed
   only after a comfortable 20K gate.
8. **Phases 10–11 — optional acceleration/backend.** Inspect Qwen3.8 MTP
   tensors/support before any conversion. If absent or incompatible, record
   and stop. If time remains, add a serialized local OpenAI-compatible backend
   around the validated runtime and run a real agent-style smoke.
9. **Phase 12 — synthesis.** Audit raw artifacts, paths, hashes, results, and
   git state; write `FINAL_REPORT.md`, update `STATUS.md`/`RESULTS.md`, commit
   useful harnesses and evidence, and stop.

## Entering evidence

| role | model | payload | effective BPW | entering evidence |
|---|---|---:|---:|---|
| reference | g64 group64 | 11,771,132,928 B / 10.962722 GiB | 3.5012 | V2 23/24; conservative reference |
| performance | g128 group128 | 10,930,715,648 B / 10.180022 GiB | 3.2513 | V2 23/24; ~10.409 tok/s at 16K-class |
| quality candidate | mixed34 g128 | 11,451,399,168 B / 10.664947 GiB | 3.4061 | V2 24/24; ~5.719 tok/s at 16K-class, not yet independently localized |

## Required evidence ledger

Every meaningful experiment appends one JSON object to `results.jsonl`. Raw
outputs and machine summaries live below `artifacts/`; recipes and rankings
live below `quant/`; the held-out inputs and hashes live below `heldout/`.

