# Qwen3.8 MLX GatedDeltaNet Metal-vs-ops A/B environment

Captured 2026-08-17 on the shared Apple Silicon machine; the interrupted
full-model matrix was resumed on 2026-08-18 after a host reboot. The
diagnostic uses only mlx-lm Git commit
`254d153fdeb6f150edd4fc5a54f9828638481fa8` through uv.

## Machine

```text
Model:          MacBook Pro (Mac15,6, MRX33KS/A)
Chip:           Apple M3 Pro
CPU:            11 cores (5 performance, 6 efficiency)
Unified memory: 18 GB (19,327,352,832 bytes)
macOS:          15.7.8 (24G824)
Kernel:         Darwin 24.6.0 arm64
Metal device:   Apple M3 Pro; 14 GPU cores; Metal supported
Free disk:      243 GiB at capture
iogpu.wired_limit_mb: 0 (read only; not changed)
```

`sysctl` value reads were restricted by the execution sandbox for some
hardware keys; the machine/Metal values above are also recorded in the
preceding threshold experiment's environment capture. No sudo, sysctl,
swap, mlock, CPU-only forcing, or memory tuning was used.

## Runtime

```text
uv:          0.12.5
Python:      3.14.7
mlx:         0.32.0
mlx-lm pkg:  0.31.3
mlx-lm Git:  254d153fdeb6f150edd4fc5a54f9828638481fa8
```

Pinned source files were imported from the uv environment and were not
modified. The verification reported:

```text
qwen3_5.gated_delta_update is gated_delta.gated_delta_update: True
gated_delta_update(..., use_kernel=True) is the normal API path
gated_delta_update(..., use_kernel=False) calls gated_delta_ops
```

`qwen3_5.py` calls `gated_delta_update(..., use_kernel=not self.training)`.
The full-model OPS runner patches the symbol in `qwen3_5`'s own module
namespace and verifies that the wrapper is called. Metal mode leaves the
symbol untouched.

## Checkpoint and dimensions

```text
Repository: lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly
Snapshot:   c98bba5926f51fec1c8d8737e577221673f524d7
Quant:      affine, group size 64, 3 bits
Architecture: qwen3_5 hybrid text model
Layers:     64
Hk:         16 (linear_num_key_heads)
Hv:         48 (linear_num_value_heads)
Dk:         128 (linear_key_head_dim)
Dv:         128 (linear_value_head_dim)
```

The existing threshold prompt files under `qwen38-mlx-threshold/prompts/`
were reused byte-for-byte. No model weights are stored in this repository.
