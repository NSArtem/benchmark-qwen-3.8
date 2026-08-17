# Pinned mlx-lm source inspection

Source inspected from mlx-lm commit
`254d153fdeb6f150edd4fc5a54f9828638481fa8`. No source was patched.

## Strongest suspect: hybrid recurrent state plus the Metal gated-delta kernel

Qwen3.8 is represented by the pinned implementation as a Qwen3.5 hybrid model:

- `mlx_lm/models/qwen3_5.py:44` sets `full_attention_interval=4`.
- `qwen3_5.py:210-217` makes layers 1/2/3 linear-attention layers and every
  fourth layer a full-attention layer.
- `qwen3_5.py:345-346` creates `ArraysCache(size=2)` for every linear layer and
  `KVCache` for every full-attention layer.
- `qwen3_5.py:281-297` constructs one full-attention mask and one SSM mask for
  each model call, then sends the selected mask and cache to every layer.

The linear path is stateful in two distinct ways:

- `qwen3_5.py:149-167` carries a 3-token convolution state across calls,
  concatenates it with the current input chunk, and stores the latest state.
- `qwen3_5.py:179-199` carries the gated-delta recurrent state in
  `ArraysCache[1]` and advances it by the current call length.
- `gated_delta.py:262-283` initializes recurrent state when empty and selects
  the custom Metal kernel whenever the model is in evaluation mode on GPU.
- `gated_delta.py:214-259` contains the alternative sequential ops reference
  implementation, but the normal evaluated Metal path does not use it.

This is consistent with the measurements: changing `prefill_step_size` changes
the `T` shape and the number/remainder of calls into the stateful path, and can
change a fixed prompt from coherent to repeated tokens or back. It does not
prove which state update is wrong.

## Prefill control flow

`mlx_lm/generate.py:298-465` is the other important path:

- `generate_step` defaults to `prefill_step_size=2048` at line 307.
- It uses `mx.argmax` when no sampler is supplied at line 377, so the primary
  run is greedy rather than temperature/sampling driven.
- It leaves one prompt token for `_step`, processing the preceding tokens in
  chunks at lines 421-444.
- After every chunk it evaluates all cache state at line 433 and calls
  `mx.clear_cache()` at line 442.
- The first generated token is produced by the final `_step` and then the
  autoregressive loop begins at lines 446-461.

Thus a 512-token prompt with step 512 is not the same execution as step 256:
the former processes 511 tokens then leaves one; the latter processes 256,
255, then leaves one. The exact chunk remainder reaches the hybrid model and
its cache state.

## Cache behavior relevant to the result

- `cache.py:325-355` grows `KVCache` in 256-token allocation units but advances
  its logical offset by the actual number of tokens.
- `cache.py:594-695` implements `ArraysCache` as two arbitrary arrays. In the
  normal non-batched path, `lengths` and `left_padding` are unset, so
  `ArraysCache.make_mask` returns `None`; `advance` does not maintain a normal
  absolute position counter.
- `base.py:58-60` makes the SSM mask by delegating to the cache. Therefore the
  recurrent state is carried directly, while full-attention position/masking
  is carried through `KVCache.offset`.

The 256-token KV allocation granularity is a possible interaction point, but
the observed results are not a simple 256-token threshold: the same 512-token
prompt was coherent at step 256 and broken at step 512, while the same 640
prompt was coherent at both 256 and 512. The stronger current hypothesis is
the stateful hybrid kernel/chunk interaction, not allocation size alone.

## Evidence against two tempting explanations

1. **Not an OOM cliff.** Primary peaks rise gradually from 13.09 GB at 512 to
   15.19 GB at 4096/4608. The primary cases complete without memory errors, and
   there is no abrupt memory discontinuity at 512/640/768.
2. **Not unstable recurrent parameter loading.** Two fresh `mlx_lm.load`
   processes produced identical first-layer `A_log` and `dt_bias` heads. The
   source constructor initializes `A_log` with `mx.random.uniform` at
   `qwen3_5.py:124-125`, but the checkpoint load supplies stable values; this
   was checked rather than assumed.

## Current conclusion from inspection

The source provides a concrete, testable runtime hypothesis: the Qwen hybrid
`GatedDeltaNet`/`ArraysCache` state update, especially the evaluated Metal
kernel path, is sensitive to prefill call shape and state boundaries. The
first-token failures show that corruption can already exist after prefill, but
some runs begin with a plausible token and collapse immediately during decode.
No source-level bug is claimed until a future controlled comparison against the
ops path or a fresh conversion isolates it.
