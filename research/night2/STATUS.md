# Night-2 status

Last updated: 2026-08-19T02:25:40+04:00, Phase 0 ledger created.

## Current phase

Phase 0 — recover state, preserve the source-backed g128 checkpoint, and
establish the correction ledger.

## Last completed experiment

`P0-HISTORICAL-REPLAY`: the preserved old 8K artifact was checked directly.
Turn 1 has 128 tokens; turn 2 has 64 tokens; turn 2 equals turn 1's first 64
tokens exactly. Turn-2 hash and first-64 turn-1 hash are both
`cb7ec89a619202c1a75d78f8b67140100a66b9ec3bfcd36af3ea3be66b0a32ce`.
Longest common prefix and prefix overlap are 64 and 1.0. The old
`P3-WARM8K` `EXCELLENT` correctness conclusion is explicitly superseded and
invalid pending revalidation; its timing/memory observations remain historical
performance data only.

`P0-G128-PRESERVE`: the temporary source-backed g128 checkpoint was moved to
`/Users/artem/models/Qwen3.8-27B-3bit-g128-MLX`. Config is affine 3-bit,
group-size 128; 19 shards and 1,847 tensor keys are intact; header payload is
10,930,715,648 bytes (10.180022 GiB), exactly equal to the index metadata.

## Current diagnosis of the cache bug

The stale replay is reproduced from raw artifacts, but its mechanism is not
localized yet. The existing `NoWiredChat` passes a freshly templated one-user
fragment into one persistent hybrid `prompt_cache`; this is the suspect path.
The next evidence must compare it with exact-token concatenation and a fresh
full-transcript chat reference.

## Current trusted chat mode

None for multi-turn correctness at this point. The old persistent-cache mode
is untrusted. One-shot direct/no-wired generation remains trusted for the
previously validated cold probes.

## Persistent g128 path

`/Users/artem/models/Qwen3.8-27B-3bit-g128-MLX` (untracked model weights; do not
commit). It is independent of `/private/tmp` and has passed header/index
integrity verification.

## Best validated warm context

None under the Night-2 oracle. Historical g64 4K/8K warm timings are retained
as performance observations, but 8K correctness is invalid and 4K must be
revalidated.

## Best cold context

Source-backed g128: 16,727 actual prompt tokens, coherent, 15.236 GiB peak in
the prior saved fresh-process probe. This is historical cold evidence, not a
permission to probe larger contexts.

## Safety state

No Night-2 inference workload is running. No wired-memory setting has been
changed. `ps` is unavailable in this sandbox, so process absence cannot be
independently established; all heavy launches will remain serialized. The
g128 checkpoint is preserved outside temporary storage.

## Next action

Create the erratum and implement the protocol-aware harness under
`research/night2/harness/`, then run a single fresh g64 Mode-A reproduction at
the first safe boundary before spending more memory exposure.

