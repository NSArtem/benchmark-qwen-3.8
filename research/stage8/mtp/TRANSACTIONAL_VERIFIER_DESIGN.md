# Transactional sequential-equivalent verifier design

## Goal and gate

The verifier must produce the same greedy tokens and the same committed target
state as ordinary one-token target execution.  A useful speed claim additionally
requires fewer target tokens/calls; a memory-only improvement is reported
separately.

The Phase 5 oracle proves that a batched `S>1` call is not exact beginning at
layer 0's `ArraysCache`, so the first implementation keeps recurrent updates
sequential.  It removes only the unnecessary full-prefix snapshot and replay.

## State ownership

At the start of a round the committed target cache contains the prompt and all
previously committed output tokens.  The round input is `[bonus] + draft`
tokens, exactly as in the Stage 7 oracle.

### Recurrent `ArraysCache`

- Allocate a transaction-local `ArraysCache` containing the current fixed-size
  convolution state and GatedDelta recurrent state.
- Process each round token with the normal one-token layer call.
- After each processed token, evaluate the cache transition and retain a
  checkpoint for that token boundary.  The checkpoint may retain immutable MLX
  array references when the implementation proves that later cache updates
  replace rather than mutate those arrays; otherwise copy only these fixed
  recurrent arrays.
- On rejection, select the checkpoint after `[bonus] + accepted_draft_prefix`.
  That becomes the committed recurrent state.  Discard the rejected suffix
  checkpoint without replaying the full prompt.
- On full acceptance, commit the final transaction-local recurrent state.

The measured 8K recurrent state footprint is 153,944,064 bytes.  It is
independent of historical context, unlike attention KV storage.

### Full-attention `KVCache`

- Keep the historical KV arrays shared; do not deep-copy their full context.
- Record each KV offset before the transaction.
- Append speculative keys/values into the existing capacity or a short growth
  delta.  On rejection, trim each KV cache back to the offset after the
  accepted boundary.  On full acceptance, retain the new offset.
- If an append operation reallocates, the reallocated storage remains safe to
  keep after accepting; on rejection the offset trim discards only the
  speculative suffix.

This design relies on the pinned `KVCache.trim()` contract and on the fact that
one serialized request is active.  It does not generalize to concurrent
readers or arbitrary cache implementations.

## Round sequence

1. Propose the small Stage 7 draft block.
2. Save committed KV offsets and create recurrent-only transaction state.
3. Run target tokens one at a time, storing hidden outputs and recurrent
   checkpoints after every token.
4. Compare target predictions with draft IDs left to right.
5. Select the accepted recurrent checkpoint and trim rejected KV suffixes.
6. Synchronize the draft state using the accepted target hidden and next bonus,
   following the existing Stage 7 semantics.
7. Commit the selected target cache and continue.

The fallback token is a prediction, not an input already in the target cache;
the committed target offset therefore advances by `1 + accepted_draft_count`
target inputs, matching the Stage 7 oracle.

## Exactness checks

For every proposed implementation compare, on the same prompt and proposed
stream:

- emitted token IDs and finish reason;
- target top-1/logits at each verification position;
- all 48 recurrent cache arrays and 16 KV offsets/accepted arrays;
- accepted-prefix state after a full-accept round;
- post-rejection state after accepted-prefix rollback;
- long continuation tokens after several mixed rounds.

The existing Stage 7 snapshot/replay implementation remains the correctness
oracle.  A transaction is not called exact until these state checks pass.

## Cost model

This first design still performs one target model call per candidate token,
just like baseline decode, and it adds draft proposals.  It can reduce:

- copied attention-history bytes;
- rejected-prefix target replay;
- cache-transaction memory.

It cannot by itself reduce target compute.  If the measured per-token target
work plus draft overhead remains above baseline, the branch will be classified
as a memory optimization or closed rather than tuned indefinitely.

## Risks and mitigations

- **In-place KV mutation:** test offsets, array identity, and rejected suffix
  contents on the pinned `KVCache`; fall back to a short delta if unsafe.
- **MLX lazy recurrent arrays:** force evaluation at checkpoints and compare
  state signatures against the sequential oracle.
- **Draft state mismatch:** keep Stage 7 draft synchronization unchanged until
  target exactness is established.
- **No compute win:** report target calls/tokens and retain baseline as the
  production default.
