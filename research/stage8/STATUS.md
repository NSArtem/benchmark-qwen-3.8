# Stage 8 status

Last updated: 2026-09-02, after Phases 1–3.

## Current phase

Phase 4 — exact MTP recovery and diagnostic profiler preparation.

## Memory/swap baseline

The Stage 8 entry baseline reported 19,327,352,832 bytes total, approximately
77% system-wide free memory, 159,912 wired-down pages, and 93,486
compressor-occupied pages. `vm_stat` was available; `sysctl vm.swapusage` and
`ps` were permission-restricted in the default sandbox. After the real g128
recovery probes the host returned to 89% free memory, 113,009 wired-down pages,
and 10,021 pages occupied by the compressor. No unsafe-memory condition was
observed. The Stage 7 clean-host reference was 89% free and 63.25 MiB swap
used; it remains historical.

No wired-memory setting was changed and no `mx.set_wired_limit()` call was
made.

## Truncation bug

- Status: reproduced and fixed.
- Root cause: the old server used decoded assistant text as cache identity.
  Qwen's assistant template trims content, so a real g128 max-256 response
  ending in token `198` (`\\n`) reconstructed with `<|im_end|>` `248046` at
  absolute token 314. A separate partial-byte oracle reproduced a
  decode/re-encode mismatch inside the generated stream.
- Evidence: `truncation/repro.json`, `truncation/real-targeted.json`, and
  `truncation/ROOT_CAUSE.md`.
- Fix: authoritative generated IDs are spliced into the canonical assistant
  span at each recorded prompt boundary. The API still returns decoded text.

## Server fallback

Implemented. Exact extensions use the repaired token stream and append only the
canonical suffix. A valid history edit or an unproven cache state clears the
hybrid cache and fresh-prefills instead of returning 409. Session metrics expose
`cache_exact_reuse`, `cache_fresh_fallback`, `cache_conflict_error`, and
`truncation_continuations`.

Lightweight coverage is 14/14. The real recovery artifact
`truncation/real-recovery-final.json` records three forced truncations, three
successful exact follow-ups, zero HTTP errors, and a successful fresh fallback
after a deliberate history edit.

## MTP profiler findings

Not started. The Stage 7 exact algorithm is sequential target verification with
per-round deep cache copy plus replay. Its tracked implementation and call
graph are known, but Stage 7 raw profile logs and the ignored 4-bit artifact are
absent locally. MTP source/artifact recovery is the next action.

## First divergence

Not measured. The Stage 7 report says the obvious batched verifier failed, but
does not localize a layer/cache component. A predetermined-token oracle is
required before verifier implementation.

## Verifier implementation

No Stage 8 verifier exists. The Stage 7 sequential snapshot/replay oracle must
remain unchanged until a divergence artifact and transactional design exist.

## MTP exactness/performance

Historical Stage 7 reference only: 4-bit MTP was exact on short/4K/8K, with
decode 6.289/4.070/2.651 tok/s versus baseline 11.898/11.647/11.147; 16K
was safety-stopped. Stage 8 values are pending and must not be inferred from
these numbers.

## Long-agent benchmark

Not started. Stage 7 reached only 983 logical tokens in its largest task and
did not validate recovery after truncation.

## Next action

Recover the Stage 7 MTP source/4-bit artifact if available without mutating the
canonical environment; otherwise preserve the blocker and run the target-only
sequential-vs-chunk divergence oracle.
