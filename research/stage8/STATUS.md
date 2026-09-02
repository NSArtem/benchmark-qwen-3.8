# Stage 8 status

Last updated: 2026-09-02, after Phases 4–10 MTP work.

## Current phase

Phase 11 — long serialized coding-agent reliability benchmark. The truncation
fix is production-integrated; MTP remains experimental/rejected as an
acceleration mode after the exact transactional result stayed slower.

## Memory/swap baseline and safety

The Stage 8 entry baseline reported 19,327,352,832 bytes total, approximately
77% free memory, 159,912 wired-down pages, and 93,486 compressor-occupied
pages. `vm_stat` was available; `sysctl vm.swapusage` and `ps` were
permission-restricted by the sandbox. After each completed 27B run the host
returned to roughly 89–90% free memory.

One guarded 16K transactional MTP attempt was stopped by the agent when
`vm_stat` reached 4,100 free pages (~64 MiB), ~4.46 GiB occupied compressor
pages, and ~3.67 GiB wired pages. The process was interrupted during prefill;
there is no 16K MTP correctness or throughput result. The host recovered to
90% free. No wired-memory setting was changed and no `mx.set_wired_limit()`
call was made.

## Truncation bug

- Status: reproduced, localized, fixed, and real-model regression-tested.
- Root cause: the old server used decoded assistant text as cache identity.
  Qwen's assistant template trims content, so a real g128 max-256 response
  ending in token 198 (`\\n`) reconstructed with `<|im_end|>` 248046 at
  absolute index 314. A separate partial-byte oracle showed that decoded
  assistant text can also be non-lossy only in appearance: generated
  `[9008,239,102,373]` decoded as `👩�` and re-encoded as
  `[9008,239,102,5513]`.
- Evidence: `truncation/ROOT_CAUSE.md`, `truncation/real-targeted.json`,
  `truncation/repro.json`, and `truncation/real-recovery-final.json`.
- Fix: preserve generated IDs and finish reason, splice authoritative IDs into
  the canonical assistant span, reuse exact known suffixes, and fresh-prefill
  valid but unproven histories instead of returning 409.

## Server fallback

Implemented and covered by 14/14 lightweight tests plus real g128 tests. The
real serialized recovery artifact records three forced length truncations,
three exact-cache follow-ups, one deliberate valid-history edit using fresh
fallback, zero HTTP errors, three exact reuses, one fallback, and zero
conflict errors. Remaining 409s are reserved for genuinely invalid or
ambiguous session semantics.

## MTP artifact and profiler findings

The Stage 7 source-backed BF16 sidecar was recovered from the published
source-compatible MTP artifact and quantized locally only for the 4-bit draft
artifact; no target model quantization was performed. The candidate has 15
MTP tensors, a 849,398,784-byte BF16 source payload, a 225,659,904-byte
quantized parameter payload, and block size 3. Weights remain outside git in
`/private/tmp`.

The diagnostic profiler is in `research/stage8/mtp/profile_exact.py` and is
opt-in. At 8K it measured 126.873 s total diagnostic wall time: target
prefill 85.68%, target verification 9.73%, draft proposal 2.40%,
rollback/replay 1.48%, snapshot 0.67%, and all remaining components below
0.1% each. Counts were 30 draft calls, 114 target calls, 8,300 target tokens,
60 proposed, 34 accepted, 26 rejected, and 19 replayed target tokens.
See `mtp/profile_summary.json` and `mtp/PROFILE.md`.

## First divergence

The predetermined-token oracle is `mtp/divergence.json`. Width 1 is exact.
At width 2, the first mismatch is layer 0, `linear_attention`,
`ArraysCache`, position 0: hidden max absolute difference 0.0048828125;
the recurrent state differences are 0.25 and 0.09118270874023438 for the
two state slots. Width 3 reproduces the same first divergence. Logit top-1
was still equal in this short oracle, but logits and cache state were not.

## Verifier implementation

`mtp/transactional_verifier.py` implements the evidence-driven transactional
design: copy only the fixed 48-layer recurrent state, share historical
attention KV arrays, append speculative KV, trim rejected suffixes, and
select the recurrent checkpoint at the accepted boundary. The sequential
mode passes independent token and full-cache state oracles at short, 4K, and
8K.

An opt-in layerwise variant forces recurrent layers to single-token updates
and batches full-attention layers. It fails the short exactness gate at full
attention layer 3's `KVCache` (key/value max differences 0.28125/0.1044921875)
and emits different tokens. It is not used for benchmarks or production.

## MTP exactness and performance

Fresh-process headline results, 64 output-token budget:

| prompt | baseline decode | transactional exact decode | speed ratio | MTP peak delta |
| --- | ---: | ---: | ---: | ---: |
| 4,096 | 11.652 tok/s | 5.511 tok/s | 0.473x | +280,195,304 B |
| 8,191 | 11.115 tok/s | 3.563 tok/s | 0.321x | +1,581,699,390 B |
| 16,727 | 10.727 tok/s | safety-stopped | n/a | n/a |

The 4K and 8K transactional outputs are token-identical to fresh baseline
references. The 16K baseline completed at 10.727 tok/s and 16,359,177,270 B
peak; the MTP attempt was stopped before generation when system free pages
became unsafe. The sequential transaction is therefore an exact state/memory
experiment, not a production acceleration.

## Long-agent benchmark

Not started. The next action is to run the strict serialized backend fixture
with three forced length truncations, long tool payloads, invalid-tool
accounting, and at least 8K logical context. Use baseline g128 only; the MTP
branch did not pass the Phase 10 production gate.

## Next action

Read AGENTS/TASK/STATUS again, inspect the Stage 7 agent harness and current
server protocol, implement the deterministic long-session harness, then run
and record the production reliability gate before final synthesis.
