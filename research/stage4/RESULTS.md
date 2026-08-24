# Stage 4 Results

## Phase 0 — recovery

### Measured facts

Stage 3 established a correct exact-suffix persistent protocol for the hybrid
Qwen3.8 cache. The g128 baseline is 10.1800222 GiB versus 10.9627218 GiB for
g64. At 8K/12K/16K-class contexts, g128 measured 13.983/14.609/15.236 GiB
peak and approximately 11.105/10.803/10.409 tok/s warm decode.

The Stage 3 endurance artifact shows turn 17 generated exactly four tokens,
`The smallest worker`, followed by EOS. The logical cache equations still held
and no replay was detected, so this is an unresolved generation-quality event,
not evidence of cache corruption.

### Inference

Quality Suite V2 must separate objective checks, rubric judgments, pairwise
judgment, and teacher-forced metrics. The old 64-token coding cap cannot be used
to make the final g64/g128 quality decision.

### Future hypothesis

A corrected suite may narrow the apparent g128 coding deficit, but the prior
teacher-forced NLL gap and observed arithmetic error make a quality difference
plausible. This will be tested, not assumed.

The local high-precision source is present at snapshot
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (55,562,855,904 bytes across 18
shards). The existing bounded converter self-test passed in the pinned MLX
0.32.1 environment: 498 quantization keys, text-only model schema, and zero
wired-limit calls. The archive-backed runtime helper only exposes existing
cached packages and the matching MLX 0.32.1 Metal library; it does not sync or
upgrade the canonical environment.
