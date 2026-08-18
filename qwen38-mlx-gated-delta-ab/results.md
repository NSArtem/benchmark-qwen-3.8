# Qwen3.8 MLX GatedDeltaNet Metal-vs-ops A/B result

Date: 2026-08-17/18

Model: `lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly`  
mlx-lm: Git commit `254d153fdeb6f150edd4fc5a54f9828638481fa8`  
Runtime: MLX 0.32.0, Apple M3 Pro, 18 GB unified memory

## Decision

This experiment does **not** reproduce an incorrect or nondeterministic
standalone `gated_delta_kernel`.

The custom Metal kernel agrees closely with `gated_delta_ops` for the actual
Qwen dimensions and is deterministic for identical inputs. Processing one
sequence as a whole or as 256/128-token chunks is exactly equivalent in the
recorded comparisons. A one-layer real-weight test also agrees closely.

At the full-model level, however, forcing `use_kernel=False` does not restore
correct generation: successful OPS runs are broken or incomplete, and later
OPS repeats hit Metal insufficient-memory errors before the first token. The
result is therefore **Case B**:

> The standalone Metal kernel is not the demonstrated cause. Replacing it
> with the reference ops path does not eliminate the full-model failure.
> The next high-value experiment is a fresh self-conversion of the official
> Qwen3.8 checkpoint, followed by the same short threshold test.

This is not evidence that the Metal kernel is perfect in every possible shape
or model, only that this diagnostic did not isolate a kernel-level failure.

## 1–7. Synthetic operator tests

The actual checkpoint dimensions were used:

```text
B=1, Hk=16, Hv=48, Dk=128, Dv=128
dtype: q/k/v bfloat16; recurrent state, g, beta float32
seed: 20260817
```

The same tensors and initial state were passed to both implementations.

| T | Metal output max abs error vs ops | Metal state max abs error vs ops | Output cosine | State cosine | NaN/Inf |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 1.12e-8 | 1.0000000000000002 | 0.9999999999999968 | 0/0 |
| 2 | 0 | 1.49e-8 | 1.0000000000000002 | 0.9999999999999950 | 0/0 |
| 16 | 1.22e-4 | 2.98e-8 | 0.9999999998415 | 0.9999999999999899 | 0/0 |
| 128 | 2.44e-4 | 2.98e-8 | 0.9999999998753 | 0.9999999999999896 | 0/0 |
| 256 | 4.88e-4 | 2.98e-8 | 0.9999999996936 | 0.9999999999999899 | 0/0 |
| 512 | 4.88e-4 | 2.24e-8 | 0.9999999997624 | 0.9999999999999895 | 0/0 |
| 640 | 2.44e-4 | 2.24e-8 | 0.9999999998157 | 0.9999999999999896 | 0/0 |
| 768 | 4.88e-4 | 2.98e-8 | 0.9999999997595 | 0.9999999999999891 | 0/0 |

These are small bfloat16/float32 numerical differences, not an
order-of-magnitude divergence or a sequence-length cliff.

For T=128, 512, and 768, ten repeated Metal calls with identical inputs and
state matched repeat 0 exactly. The ten repeated ops controls also matched
repeat 0 exactly. No NaNs or Infs occurred.

Whole-sequence versus chunked execution also matched exactly in the recorded
float32 comparisons:

```text
totals: 512, 640, 768
chunks: 256 and 128
implementations: Metal and ops
output max error vs whole: 0 for every case
state max error vs whole: 0 for every case
```

Therefore:

1. `gated_delta_kernel` numerically agrees with `gated_delta_ops` for this
   workload.
2. Disagreement does not grow meaningfully with sequence length.
3. The Metal kernel is deterministic for identical inputs.
4. The ops implementation is also deterministic.
5. Whole-block and chunked processing are equivalent here.
6. Chunk equivalence holds for ops.
7. Chunk equivalence holds for Metal.

## Real-weight layer control

One actual layer (`DecoderLayer` 0, including its real projections,
convolution, normalization, GatedDelta call, and cache state) was run with
the same embedded prompt input in Metal and forced-ops modes.

| T | Output max abs error | State max abs error | Output cosine | State cosine | NaN/Inf |
|---:|---:|---:|---:|---:|---:|
| 128 | 1.22e-4 | 5.72e-6 | 0.9999999999941 | 0.9999999999999885 | 0/0 |
| 512 | 9.77e-4 | 8.58e-6 | 0.9999999999505 | 0.9999999999999738 | 0/0 |
| 768 | 9.77e-4 | 1.34e-5 | 0.9999999999898 | 0.9999999999999662 | 0/0 |

This is additional evidence against a large arithmetic error in the custom
kernel on actual checkpoint weights.

## 8–12. Full-model A/B

Settings were identical except for the runtime switch:

```text
greedy argmax
enable_thinking=false
max_output_tokens=128
default prefill_step_size=2048
no KV quantization, activation quantization, MTP, APC, or prompt caching
fresh process for every run
```

The required matrix was Metal x5 and OPS x3 at each of 512, 640, and 768
tokens. `sanity-*` runs are separate preliminary runs and are not included in
the repeat counts below.

| Implementation | Prompt | Repeat outcomes | First-token IDs for successful runs | Memory / failure |
|---|---:|---|---|---|
| Metal | 512 | 5/5 BROKEN | 2, 271, 2, 2, 2 | 13.07–13.09 GB |
| Metal | 640 | 1 DEGRADED, 4 BROKEN | 332, 271, 12, 4558, 12 | 13.22–13.23 GB |
| Metal | 768 | 5/5 BROKEN | 10525, 1343, 271, 10, 205213 | 13.36–13.38 GB |
| OPS | 512 | 1 BROKEN, 1 DEGRADED, 1 OOM | 69267, 248068 | successful runs 14.68 GB |
| OPS | 640 | 2 BROKEN, 1 OOM | 332, 262 | successful runs 15.23 GB |
| OPS | 768 | 1 BROKEN, 2 OOM | 2083 | successful run 15.72 GB |

The OPS 512 DEGRADED run begins with useful coding-agent prose but reaches
the 128-token cap before answering the task; it is not counted as a coherent
completion. The other successful OPS runs are immediate EOS, unrelated
repetition, or token/number soup. The OOM runs fail during
`mx.eval([c.state for c in prompt_cache])` before a first token.

Thus:

8. At 512, Metal is broken in all five repeats. OPS does not restore a
   completion: one run is broken, one incomplete, and one OOM.
9. At 640, Metal is mostly broken with one incomplete normal-looking start;
   OPS is broken in both successful runs and OOM in the third.
10. At 768, Metal is broken in all five; OPS is broken in its one successful
    run and OOM in the other two.
11. Metal first-token IDs are not stable across fresh full-model runs.
12. Successful OPS first-token IDs are also not a stable correctness signal,
    and the OPS path has a higher memory footprint and OOMs.
13. No: forcing `use_kernel=False` does not eliminate token soup or produce a
    coherent full-model completion.

The first-token variation is real at the full-model level, but the isolated
kernel test did not reproduce it. It therefore points to a higher-level
model/cache/integration or checkpoint/conversion problem rather than proving
that the Metal GatedDelta kernel is the source.

## Memory observation

Metal memory rises gradually from about 13.07 GB at 512 to 13.38 GB at 768;
there is no correctness cliff paired with an abrupt memory jump. The OPS
reference path is materially more expensive in the full model: successful
peaks were 14.68, 15.23, and 15.72 GB, and subsequent fresh OPS runs failed
with Metal insufficient-memory errors. This is descriptive telemetry, not a
memory-tuning result, and it does not explain the already-broken successful
OPS outputs.

## 14. Strongest suspect after source inspection

The strongest current suspect is the higher-level Qwen3.5/Qwen3.8 hybrid
integration:

- `qwen3_5.py:149–168` carries a three-token convolution state across model
  calls.
- `qwen3_5.py:179–199` carries GatedDelta state through `ArraysCache[1]` and
  advances the cache by each call length.
- `qwen3_5.py:281–297` shares the SSM mask across periodic linear layers and
  full-attention layers.
- `qwen3_5.py:345–346` combines `ArraysCache` and `KVCache` in the hybrid
  model.
- `generate.py:307` defaults to 2048-token prefill chunks; lines 421–444
  evaluate cache state and clear MLX's allocation cache between chunks.
- `ArraysCache.advance()` at `models/cache.py:685–689` does not maintain a
  normal absolute recurrent offset, while `KVCache` has an explicit offset.

This is consistent with the original context-length failure and with
full-model fresh-process variability. It is not a proven source-level bug.

Evidence against the custom Metal kernel specifically:

- direct operator error remains at bfloat16-scale rounding;
- Metal is exactly deterministic for ten identical-input repeats;
- Metal and ops are exactly chunk-equivalent in the standalone recurrence;
- one real-weight layer agrees closely at 128/512/768;
- forced-ops full-model generation is also broken and can OOM.

## 15. Upstream issue readiness

There is **not** enough evidence for a minimal upstream issue claiming that
`gated_delta_kernel` itself is incorrect or racy: the requested synthetic
reproduction failed to reproduce that claim.

There **is** enough evidence for a broader upstream mlx-lm issue about
Qwen3.5/Qwen3.8 hybrid full-model correctness under this checkpoint/runtime:
the same prompts, pinned revision, and fresh processes produce malformed
outputs; replacing the kernel does not rescue them; and the isolated kernel
and real-weight layer controls are clean. A stronger issue should first add a
fresh self-converted checkpoint or a smaller non-OOM reference run.

## 16. Next experiment

Self-conversion is now justified and is the highest-value next step. Use the
official Qwen3.8 text checkpoint and the pinned converter, for example:

```text
UV_CACHE_DIR=/private/tmp/qwen38-mlx-threshold-uv-cache \
  uvx --from 'git+https://github.com/ml-explore/mlx-lm@254d153fdeb6f150edd4fc5a54f9828638481fa8' \
  mlx_lm.convert \
  --hf-path /path/to/official/Qwen3.8-27B \
  --mlx-path /private/tmp/qwen38-self-converted-q3 \
  --quantize --q-bits 3 --q-group-size 64 --q-mode affine \
  --dtype bfloat16
```

Then run only the existing 512/640/768 prompts with the default Metal path,
and, if memory permits, one 512 OPS control. A clean self-converted result
would make the current `lukaskremla` conversion the leading suspect; the same
cliff would strengthen the higher-level mlx-lm hybrid implementation theory.

No model was downloaded or converted in this diagnostic.
