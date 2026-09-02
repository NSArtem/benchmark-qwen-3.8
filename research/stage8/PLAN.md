# Stage 8 plan

Stage 8 is the runtime-engineering continuation of Stage 7. The fixed
production model is the affine 3-bit/group-128 checkpoint; no new static
quantization or DWQ work is in scope.

## Gates

1. Phase 0: recover Stage 7, record a clean host baseline, and establish the
   implementation/artifact inventory.
2. Phases 1–3: reproduce the length-truncated HTTP 409 at token level, prove
   the first mismatch, fix continuation semantics, add exact reuse and safe
   fresh-prefill fallback, and run lightweight plus real-model regressions.
3. Phase 4: profile the existing exact 4-bit MTP prototype in diagnostic mode
   without changing its algorithm; quantify time, calls, copied state, and
   memory checkpoints.
4. Phases 5–6: compare sequential and chunk verification on predetermined
   tokens, localize the first divergence, and write a transactional verifier
   design based on evidence.
5. Phases 7–9: implement the smallest sequential-equivalent verifier, prove
   token and committed-cache exactness, then optimize only measured costs.
6. Phase 10: fresh-process 4K/8K/16K MTP headline benchmarks with safety stops.
7. Phase 11: run a strict serialized coding-agent session to at least 8K
   logical context with three forced truncations and real tool validity checks.
8. Phases 12–13: integrate only validated behavior, update the reports, inspect
   raw evidence and git state, and stop without starting Stage 9.

## Safety and measurement rules

- Run at most one heavy Qwen/MLX process at a time.
- Use `uv run --project envs/mlx-0.32.1 --no-sync ...` for canonical runs.
- Do not alter `iogpu.wired_limit_mb` or call `mx.set_wired_limit()`.
- Diagnostic timing may synchronize; headline timing must not inherit those
  artificial synchronizations.
- Treat cache state as exact only after token, offset, recurrent-state, and KV
  evidence supports it. Valid transcripts may fall back to fresh prefill.
- Append measured runs to `results.jsonl`; unavailable values remain null.

## Current Phase 0 inventory

- Worktree was clean at entry; Stage 7 HEAD was `7ba8ba4` (`stage 8`).
- Canonical g128 model directory exists at
  `/Users/artem/models/Qwen3.8-27B-3bit-g128-MLX`.
- The tracked Stage 7 MTP harness is
  `research/stage7/harness/run_mtp_correctness.py`; its exact path is retained
  as the correctness oracle.
- The ignored Stage 7 raw MTP, agent, and performance directories are not
  present in this checkout. The Stage 7 reports and tracked harnesses are
  available; missing raw artifacts must be recorded and regenerated where safe.
- The Stage 7 source-checkpoint symlinks exist but their blob targets are
  currently missing, so MTP source recovery requires an explicit local/cache
  recovery check before any model load.

## Phase 1–3 completion record

- `truncation/ROOT_CAUSE.md` proves the real newline/template-boundary mismatch
  and the independent partial-byte decode/re-encode mismatch.
- `truncation/repro.json` contains the 13-case tokenizer matrix; the repaired
  canonical stream is exact-prefix for every case.
- `truncation/real-targeted.json` contains the real g128 projected-409 case and
  successful exact reuse after the patch.
- `truncation/real-recovery-final.json` contains three forced truncations,
  three exact follow-ups, and one valid-history fresh fallback.
- The server now preserves generated IDs/finish reason and exposes cumulative
  exact-reuse, fallback, conflict, and truncation-continuation counters.

## Phase 4–10 completion record

- `mtp/profile_summary.json` and `mtp/PROFILE.md` quantify the unchanged
  exact Stage 7 path at short/4K/8K. At 8K, target prefill is 85.68% of
  diagnostic wall time; the full cache snapshot is 707,592,192 bytes and the
  clean same-run peak delta is 1,581,699,296 bytes.
- `mtp/divergence.json` localizes the first sequential-vs-batched mismatch to
  layer 0's `ArraysCache` at width 2. The layerwise follow-up also fails at
  full-attention layer 3, so it is not a valid optimized verifier.
- `mtp/transactional_verifier.py` passes token and independent full-cache state
  oracles at short, 4K, and 8K while sharing historical KV state and copying
  only recurrent state. It does not reduce target compute.
- Fresh-process headline runs are token-exact at 4K and 8K but slower than
  baseline (5.511 vs 11.652 tok/s; 3.563 vs 11.115 tok/s). The guarded 16K
  transactional run was stopped at the memory safety boundary before output.
- MTP is closed as a production acceleration branch for this stage; baseline
  g128 remains the default. The remaining gate is long-agent reliability.

## Deliverables

- `truncation/ROOT_CAUSE.md`, `truncation/repro.json`, and regression evidence.
- `mtp/profile_summary.json`, `mtp/PROFILE.md`, `mtp/divergence.json`,
  `mtp/VERIFIER_DIVERGENCE.md`, and
  `mtp/TRANSACTIONAL_VERIFIER_DESIGN.md`.
- exactness/performance artifacts, long-agent transcript/results, `RESULTS.md`,
  `STATUS.md`, and `FINAL_REPORT.md`.
