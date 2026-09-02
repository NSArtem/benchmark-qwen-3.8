# Stage 8 results

This ledger distinguishes measured facts, inferences, and hypotheses. Values
from Stage 7 are explicitly labeled historical controls; Stage 8 measurements
are appended as the phases run.

## Phase 0 — entry recovery

### MEASURED FACT

- `AGENTS.md`, `TASK.md`, and all four required Stage 7 reports/ledgers were
  read completely before Stage 8 actions.
- The worktree was clean at entry. `git log --oneline -30` showed Stage 8
  entry commit `7ba8ba4` above the Stage 7 commits.
- The canonical model directory exists. The tracked Stage 7 runtime and MTP
  harnesses are present.
- Stage 7 raw MTP, performance, and agent artifact directories are ignored and
  absent in this checkout. The Stage 7 reports remain the available historical
  evidence.
- Current host `memory_pressure` reports about 77% free memory. `vm_stat` is
  available; `vm.swapusage` and `ps` are restricted by the current sandbox.

### INFERENCE

The Stage 8 server and token-level tests can be developed without the missing
ignored artifacts. Real MTP reruns may require recovering the missing source
checkpoint blobs and the quantized MTP artifact, or rebuilding the candidate
only if the source tensors can be recovered without violating scope.

### HYPOTHESIS

The Stage 7 server likely treated an exact-prefix/cache optimization miss as a
session-state conflict. This is now supported by the token and real-model
evidence below.

## Phases 1–3 — truncation continuation and recovery

### MEASURED FACT

- The tokenizer-only oracle covered 13 cases: EOS, max-token budgets 1, 2, 3,
  31, 32, 63, 64, 127, 128, 255, and 256, plus a partial-byte Unicode case.
  The old logic projected 3 HTTP 409 cases; the repaired stream was an exact
  prefix in all cases. Two cases had decode/re-encode non-identity.
- The real g128 newline-heavy probe ran budgets 31, 128, and 256. All reached
  `finish_reason=length`; the old logic projected one 409 at budget 256. Its
  generated stream ended in ID 198 (`\\n`), while the raw next template put ID
  248046 (`<|im_end|>`) at absolute index 314. The patched continuation reused
  the exact cache and returned HTTP 200.
- The real repeated-session probe produced three forced length truncations,
  three exact-cache follow-ups, and a valid edited-history fresh fallback. It
  recorded zero HTTP errors, three exact reuses, one fresh fallback, and three
  truncation continuations. See
  `truncation/real-recovery-final.json`.
- The partial-byte case generated `[9008, 239, 102, 373]`, decoded as `👩�`,
  and re-encoded as `[9008, 239, 102, 5513]`. The first mismatch was generated
  index 3. The explicit EOS case remained an exact prefix despite decoded
  special-token omission.
- Lightweight token/session tests pass 14/14.

### INFERENCE

The old HTTP 409 was an optimization miss misclassified as a conflict. A
length-truncated assistant turn is valid; exact stored IDs are required for
identity, while valid but unreusable transcripts must be fresh-prefilled.

### HYPOTHESIS

Other client/template combinations may expose additional non-lossless text
boundaries. The fallback policy handles them without trusting stale cache.
