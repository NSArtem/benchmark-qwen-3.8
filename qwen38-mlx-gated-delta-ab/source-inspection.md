# Pinned source inspection

Inspected from mlx-lm Git commit
`254d153fdeb6f150edd4fc5a54f9828638481fa8`. No upstream source file was
modified. The temporary uv environment path used for this run was:

```text
/private/tmp/qwen38-mlx-threshold-uv-cache/archive-v0/BA2xb8MR4jSwbL5E/lib/python3.14/site-packages/mlx_lm/
```

## A/B selection is real

`mlx_lm/models/qwen3_5.py:17` imports `gated_delta_update` into the
`qwen3_5` module namespace. `GatedDeltaNet.__call__` invokes it at lines
184–195 and passes `use_kernel=not self.training` at line 194. The loaded
layer reports `training=False`.

`mlx_lm/models/gated_delta.py:262–283` computes `beta` and `g`, initializes
the float32 state, then selects `gated_delta_ops` when `use_kernel=False` and
`gated_delta_kernel` otherwise on a Metal GPU. The runner patched
`qwen3_5.gated_delta_update`, not merely `gated_delta.gated_delta_update`,
and recorded `ops_patch_active=True` plus a nonzero wrapper call count in
every OPS run. Metal runs recorded the original symbol identity.

There is no hidden alternate GatedDelta implementation in this path.

## Kernel versus ops implementation

The custom Metal source is constructed at `gated_delta.py:13–123`. It maps
the value head to the corresponding key head at line 34, initializes the
per-thread state at lines 48–56, and advances q/k/v/y/g/beta pointers at
lines 87–93. It writes the final recurrent state at lines 95–98. The launch
configuration is `grid=(32, Dv, B * Hv)` with `threadgroup=(32, 4, 1)` at
lines 197–210.

The ops implementation is a sequential loop at lines 214–259. It repeats
q/k for `Hv // Hk` at lines 242–244, calls the compiled single-step
reference at lines 246–256, and stacks the per-token output at line 258.
The single-step reference performs decay, state-to-memory projection, delta
update, and q projection at lines 150–168.

The code inspection found no obvious pointer increment or head-repeat error.
The measured operator comparison is the more important evidence: the two
implementations agree closely for the actual Qwen dimensions, including
`Hv/Hk=3`.

## Higher-level state and chunk paths

The Qwen hybrid layer carries two different state paths:

- `qwen3_5.py:149–168` prepends the three-token convolution state to each
  input call and stores the newest three tokens in `ArraysCache[0]`.
- `qwen3_5.py:179–199` passes `ArraysCache[1]` into GatedDelta, stores the
  returned recurrent state, and calls `cache.advance(S)`.
- `qwen3_5.py:44`, `:213`, and `:281–297` implement the periodic full-
  attention/linear-attention hybrid and share an SSM mask across linear
  layers.
- `qwen3_5.py:345–346` creates `ArraysCache(size=2)` for linear layers and
  `KVCache` for full-attention layers.

`mlx_lm/generate.py:307` defaults `prefill_step_size` to 2048. The primary
prefill loop at lines 421–444 processes chunks, evaluates every cache state
at line 433, and clears the MLX cache at line 442 before the final `_step`.
Autoregressive decode then uses `_step` again at lines 444–460.

`ArraysCache` at `models/cache.py:594–699` stores arbitrary arrays. Its
`advance` method at lines 685–689 adjusts optional `lengths` and
`left_padding`, but does not maintain a normal absolute recurrent offset.
By contrast, `KVCache` has an explicit 256-token allocation step and an
absolute `offset` at `models/cache.py:325–355`.

These paths are consistent with a possible higher-level hybrid
cache/prefill/integration defect, but the tests here do not prove one. The
standalone kernel and one real-weight layer are both numerically well behaved.

## Interpretation

The strongest current suspect is therefore not the arithmetic of
`gated_delta_kernel` itself. It is the combination of the Qwen hybrid
layer's convolution/recurrent `ArraysCache`, periodic full-attention cache,
and generation-time chunk/state evaluation—or the MLX checkpoint/conversion
feeding that integration. The full-model forced-ops path failing too means
that replacing the kernel alone is insufficient to isolate the final failure.
