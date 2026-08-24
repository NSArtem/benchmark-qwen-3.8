# Stage 3 results ledger

This is the human-readable synthesis for the current run. Raw prior evidence
is preserved under `research/` and `research/night2/`; new machine-readable
records are appended to `results.jsonl`.

## Phase 0 — recovered baseline

Measured facts and source paths are recorded in `STATUS.md`. The current
candidate is source-backed affine 3-bit/group-128 g128. The highest-value
unresolved question is whether exact-suffix persistent cache state is
consistent with a fresh execution using the same token stream and call
boundaries.

Decision gate: do not scale persistent exact-suffix chat until the Phase 1 B-vs-C
comparison passes, or until a documented limitation prevents a stronger claim.

## Phase 1 — same-boundary cache correctness oracle

### Measured fact

The source-backed g128 model was run with direct pinned `generate_step`, greedy
decoding, normal KV, and default MLX wired state. The preserved 4K canonical
conversation had 4,096 initial prompt tokens, 96 turn-1 tokens, a measured
cached prefix of 4,192 tokens, a 38-token canonical exact suffix, and a 4,230
token full turn-2 transcript. The logical accounting equation held in both
segmented paths: `4,192 + 38 = 4,230`.

Path B (fresh segmented) and Path C (true persistent) matched at every saved
boundary: after turn 1, before the suffix, after the suffix plus the first
scheduled generation step, after the first turn-2 yield, and after turn 2.
All 64 layers had equal compact state signatures and offsets, with 48
`ArraysCache` and 16 `KVCache` layers. The B/C first-token logprob vectors had
0.0 maximum and mean absolute difference at both generation stages and 8/8
top-k overlap. Turn-1 and turn-2 token sequences were identical, both outputs
were coherent, and no stale replay was detected. The gate is `PASS` and exact
suffix persistent cache is validated for scaling under the tested boundaries.

### Inference

The old persistent-versus-fresh full-transcript divergence is not evidence of a
persistent-state bug in this 4K protocol. Path A was coherent and non-replaying
but shared only 42 initial generated tokens with B/C, which is consistent with
one-shot versus segmented numerical sensitivity in this hybrid recurrent model.
Phase 2 localization is not justified by the B-vs-C evidence.

### Artifact locations

The full result is `research/stage3/cache-boundary-oracle/results.json`;
complete generated IDs and decoded outputs are under `tokens/` and `outputs/`,
and compact per-layer signatures are under `state_signatures/`. The first
attempt's bfloat16 host-conversion failure is preserved in the same namespace
and in `research/stage3/results.jsonl` as a harness error, not inference data.
