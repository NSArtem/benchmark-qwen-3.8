# Sequential versus chunked Qwen3.8 target oracle

## Experiment

`divergence_oracle.py` loaded the production g128 target and used the first 64
tokens of the local short context as a common committed prefix.  It then ran
the predetermined stream `[123, 456, 789]` from identical copied caches in:

- sequential mode: one target token call at a time, with cache evaluation
  before the next token;
- chunk mode: one target call over the complete candidate prefix.

The comparison mirrors the pinned `Qwen3_5TextModel.__call__` layer loop and
records every decoder-layer output, all recurrent/KV state arrays, offsets,
target logits, and top-1 IDs.  The raw compact artifact is
[divergence-short.json](divergence-short.json).

## MEASURED FACT

| candidate length | first hidden mismatch | first cache mismatch | max logit difference | top-1 equal |
|---:|---|---|---:|---:|
| 1 | none | none | 0 | yes |
| 2 | layer 0, `linear_attention`, position 0, `0.0048828125` | layer 0 `ArraysCache`: state 0 `0.25`, state 1 `0.0911827087` | `0.296875` | yes |
| 3 | layer 0, `linear_attention`, position 0, `0.0048828125` | layer 0 `ArraysCache`: state 0 `0.25`, state 1 `0.1224336624` | `0.328125` | yes |

At length 2 the layer-0 output also differs at position 1 by `0.00390625`.
The first divergent cache is therefore the recurrent hybrid layer, not one of
the 16 full-attention KV layers.  State index 0 is the GatedDelta convolution
history and state index 1 is its recurrent state in the pinned implementation.

## INFERENCE

The direct multi-token `gated_delta_update` path is not sequential-equivalent
to repeated one-token execution for this target.  Its fused sequence update
changes both the per-position output and the committed recurrent state.  The
equal top-1 IDs are accidental masking, not evidence of exactness.

The result also rules out the following as the first cause for this stream:

- full-attention KV cache behavior;
- a cache offset/lookahead error;
- a final LM-head-only numerical difference;
- token comparison or acceptance policy.

## HYPOTHESIS

The difference is caused by the numerical/algorithmic sequence kernel used by
`gated_delta_update` when `S > 1`, possibly including its recurrent scan and
the order in which the convolution state is materialized.  The oracle does not
claim the internal kernel instruction as a proven root cause; it proves the
first externally observable divergence at the layer/cache boundary.

## Consequence for verifier work

Batching the entire hybrid target call cannot be enabled unchanged.  Any exact
candidate must run the recurrent layer transitions in token order and retain
the state at each candidate boundary.  Full-attention KV history can still be
transactional if its append/trim behavior is verified separately.
