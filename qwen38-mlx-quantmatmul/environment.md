# Qwen3.8 MLX quantized-matmul diagnostic environment

Run date: 2026-08-18 (Asia/Yerevan)

## Host

- Model: MacBook Pro Mac15,6 (MRX33KS/A)
- Chip: Apple M3 Pro, 11 cores
- Physical unified memory: 19,327,352,832 bytes; Apple reports 18 GB
- macOS: 15.7.8, build 24G824
- Darwin: 24.6.0 arm64
- Free disk at capture: 244 GiB
- `iogpu.wired_limit_mb`: `0` (read only; unchanged)
- No sudo, model conversion, or system configuration change was used.

## Pinned runtime

All MLX commands used uv with:

```text
git+https://github.com/ml-explore/mlx-lm@254d153fdeb6f150edd4fc5a54f9828638481fa8
```

- Python: 3.14.7
- MLX package: 0.32.1
- mlx-lm package metadata: 0.31.3
- mlx-lm source revision: `254d153fdeb6f150edd4fc5a54f9828638481fa8`
- uv: 0.12.5
- Default device: GPU
- Metal: available
- Metal device: Apple M3 Pro (`applegpu_g15s`)
- MLX recommended working-set size: 12,884,918,272 bytes
- MLX reported memory size: 19,327,528,832 bytes

## Model

- Repository: `lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly`
- Snapshot: `c98bba5926f51fec1c8d8737e577221673f524d7`
- Local cache path: `/Users/artem/.cache/huggingface/hub/models--lukaskremla--Qwen3.8-27B-3bit-MLX-TextOnly/snapshots/c98bba5926f51fec1c8d8737e577221673f524d7`
- Snapshot size through symlinks: approximately 11 GiB
- Loaded class: `mlx_lm.models.qwen3_5.Model`
- Text architecture: Qwen3.5 hybrid, 64 decoder layers
- Hidden size: 5120
- Full-attention interval: 4
- GatedDelta heads: 16 key heads / 48 value heads
- GatedDelta head dimensions: 128 / 128
- MLP intermediate size: 17,408
- Vocabulary: 248,320
- QuantizedLinear modules: 497
- Quantization: affine, 3 bits, group size 64
- Packed weight dtype: `uint32`
- Scale and affine-bias dtype: `bfloat16`

The complete selected-module manifest and SHA256 fingerprints are in
[`module-manifest.json`](module-manifest.json).

## Source path checked

The pinned source was inspected without modification. `mlx.nn.QuantizedLinear.__call__`
passes the stored packed weight, scales, affine biases, `transpose=True`,
`group_size`, `bits`, and `mode` to `mx.quantized_matmul`, then applies an
ordinary linear bias if present. The Qwen3.5 projections are loaded as this
class at the paths recorded in the manifest.
