# Qwen3.8 MLX threshold environment

Captured 2026-08-17 on the shared Apple Silicon machine. The experiment used
only the pinned mlx-lm commit below; no GGUF or additional MLX tuning was run.

## Machine

```text
Model:          MacBook Pro (Mac15,6, MRX33KS/A)
Chip:           Apple M3 Pro
CPU:            11 cores (5 performance, 6 efficiency)
Unified memory: 18 GB (19,327,352,832 bytes)
macOS:          15.7.8 (24G824)
Kernel:         Darwin 24.6.0 arm64
Metal device:   Apple M3 Pro
Metal memory:   19,327,352,832 bytes
Metal recommended working set: 12,884,918,272 bytes
Metal max buffer: 9,663,676,416 bytes
GPU architecture: applegpu_g15s
Resource limit: 499000
Free disk at capture: 242 GiB
iogpu.wired_limit_mb: 0
```

The wired limit was not changed. No sudo, swap changes, mlock, CPU-only
forcing, KV quantization, activation quantization, MTP, APC, or prompt-cache
reuse was used.

## Runtime

```text
uv:          0.12.5
Python:      3.14.7
mlx:         0.32.0
mlx-lm pkg:  0.31.3
mlx-lm git:  254d153fdeb6f150edd4fc5a54f9828638481fa8
transformers: 5.15.0
tokenizers:   0.22.2
```

The package metadata still reports mlx-lm `0.31.3`; reproducibility is defined
by the fetched Git commit, not that version string.

## Primary checkpoint

```text
Repository: lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly
Local snapshot: c98bba5926f51fec1c8d8737e577221673f524d7
Quantization: affine, group size 64, 3 bits
Architecture: qwen3_5 / Qwen3.5 hybrid text model
Layers: 64
Full-attention interval: 4
Linear convolution kernel: 4
```

The model was loaded from the existing Hugging Face cache. No weights are
stored in this repository.

## Alternative checkpoint check

The locally cached `rapid-mlx/Qwen3.8-27B-mixed-3.5bpw-MLX` checkpoint was the
one allowed alternate-quant control. It is a materially different mixed 2/3/4
bit recipe, but its short sanity reached 13.67 GB MLX peak and failed with a
Metal insufficient-memory error. Its below/above-boundary runs were therefore
not launched. The second cached 2-bit checkpoint was not used because the
one-alternate diagnostic scope had already been consumed.
