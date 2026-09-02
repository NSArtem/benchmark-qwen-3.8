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

## Phase 4 — exact MTP diagnostic profile

### MEASURED FACT

- The recovered Stage 7 source-backed sidecar contains 15 tensors. Its BF16
  source payload is 849,398,784 bytes; the reused 4-bit candidate is
  225,659,904 parameter bytes with block size 3. The target model remained the
  fixed g128 artifact.
- The exact implementation proposes at most two draft tokens per round and
  uses diagnostic synchronization at attribution boundaries. At 8K, the
  126.873 s profile broke down as follows:

| component | seconds | wall share |
| --- | ---: | ---: |
| target prefill | 108.702 | 85.68% |
| target verification | 12.343 | 9.73% |
| draft proposal | 3.043 | 2.40% |
| rollback/replay | 1.880 | 1.48% |
| full-cache snapshot | 0.856 | 0.67% |
| draft sync + logits + compare + other | 0.050 | 0.04% |

- The 8K run made 30 draft calls, 114 target calls, processed 8,300 target
  tokens, proposed 60, accepted 34, rejected 26, and replayed 19 target
  tokens. Token comparison was below 0.001% of wall time.
- Baseline and MTP peaks in the clean 8K profile were 15,013,952,566 and
  16,595,651,862 bytes, a 1,581,699,296-byte delta. MTP weights contributed
  225,661,186 bytes. A full snapshot was 707,592,192 bytes: 153,944,064
  recurrent-state bytes plus 553,648,128 attention-KV bytes.

### INFERENCE

The weights plus one full snapshot explain 933,253,378 bytes (59.0%) of the
same-run peak delta. Full-history attention copying alone accounts for the
measured 553,648,128-byte snapshot component, while repeated recurrent copies
account for 4,618,321,920 bytes of cumulative 8K copy traffic rather than a
simultaneous 4.62 GB allocation.

Removing snapshot and rollback/replay from the 8K diagnostic total would leave
approximately 124.137 s. That is an arithmetic upper bound for this exact
algorithm, not a benchmark: target calls/tokens and draft work remain.

### HYPOTHESIS

The unassigned 648,445,918-byte peak residual is mostly lazy MLX graph or
target/draft intermediate residency. The current aggregate counters cannot
identify every Metal allocation by tensor owner.

## Phases 5–10 — divergence, transactional verifier, and performance gate

### MEASURED FACT

- The predetermined stream oracle is exact at width 1. At width 2, the first
  divergence is layer 0 `linear_attention`, position 0, in `ArraysCache`:
  hidden max-absolute difference 0.0048828125; recurrent state differences
  0.25 and 0.0911827087. Width 3 reproduces the same first divergence. The
  short-run top-1 remained equal, but logits and state were not exact.
- The implemented transactional verifier copies only recurrent state, shares
  historical KV objects, appends/trim-rolls speculative KV, checkpoints each
  recurrent boundary, and commits the checkpoint at the accepted boundary.
  Independent token and full-cache state oracles pass at short, 4K, and 8K.
- The layerwise alternative (sequential recurrent layer calls plus batched
  attention) fails the short gate at full-attention layer 3 `KVCache`, with key
  and value differences 0.28125 and 0.1044921875. It is rejected.
- Fresh-process headline results were:

| prompt | baseline | transactional | ratio | acceptance | MTP peak delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 11.652 tok/s | 5.511 tok/s | 0.473x (-52.7%) | 29/68 = 42.65% | +280,195,304 B |
| 8,191 | 11.115 tok/s | 3.563 tok/s | 0.321x (-67.9%) | 34/60 = 56.67% | +1,581,699,390 B |
| 16,727 | 10.727 tok/s | safety stop | n/a | n/a | n/a |

- The 4K and 8K transactional token streams and committed states are exact.
  The 16K baseline completed, but the MTP attempt was stopped before
  generation at 4,100 free `vm_stat` pages (~64 MiB); no 16K MTP result is
  claimed.

### INFERENCE

The verifier is sequentially equivalent as an exact state transaction, but it
does not provide the compute reduction required for speculative acceleration:
it still performs one full target forward per candidate token and adds draft
and transaction work. It is therefore a validated memory/state experiment, not
a production speed mode.

### HYPOTHESIS

A future fused recurrent kernel could remove the first divergence, but that
would require a new numerical implementation and a new exactness proof. It is
not justified as a Stage 8 production change.

## Phase 11 — long serialized coding-agent gate

### MEASURED FACT

- The principal real-model artifact is
  `agent/long-session-final2.json`. It used the serialized
  `qwen38_agent_server.py` with baseline g128 and reached 10,589 logical
  tokens over 29 assistant turns. It included multiple fixture inspections,
  wrong-hypothesis correction, failing-test revisions, a 30.9 KB deterministic
  tool payload, and references to the initial constraints.
- Three forced `finish_reason=length` checkpoints all recovered with HTTP 200
  and exact cache reuse. There were zero HTTP errors, zero HTTP 409s, zero
  invalid calls, zero missing expected calls, 28 exact-reuse requests, and no
  fresh fallbacks. The runner's summary counts seven explicitly injected tool
  results; the raw transcript contains 12 recognized valid `read_file`
  request matches and zero invalid matches.
- Median warm TTFT was 1.062 s, p95 1.966 s, maximum 92.613 s (the long
  payload prefill); median decode was 11.811 tok/s. Final task success was
  true. The server's cumulative counter reports eight truncation continuations
  because five other assistant turns in the same session also ended at their
  requested length; the deliberate checkpoint count is three.
- Earlier strict negative controls remain in `agent/long-session-r2.json` and
  `r3.json`; they demonstrate that unknown paths and malformed prefixes were
  not given fabricated results.

### INFERENCE

The final session demonstrates production transport and cache reliability past
8K under a real serialized workload. The 92.613 s maximum is cold processing
of the large tool-result suffix; warm turns remained near one second.

### HYPOTHESIS

The harness deliberately maps the model's predictable ancillary test paths to
existing deterministic fixture contents. A provider-native structured-tool
adapter would need its own contract and was not inferred from this text-only
backend.

## Phases 12–13 — closeout

### MEASURED FACT

- The final lightweight suite ran 14 tests and passed; the server protocol
  self-test also passed.
- The production README documents exact-ID continuation, fresh fallback,
  metrics, and the decision to keep MTP off by default. No model weights,
  virtualenvs, or MTP sidecar files were added to git.

### INFERENCE

Stage 8 meets its completion condition. The recommended production runtime is
the fixed g128 baseline at a practical 12K–16K context; MTP remains an
experimental/rejected acceleration branch.
