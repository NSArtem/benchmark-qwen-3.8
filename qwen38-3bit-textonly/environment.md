# Qwen3.8 27B 3-bit TextOnly benchmark environment

Captured 2026-08-17, Asia/Yerevan, on the shared Mac before the long-context sweep.

## Machine

- Model: MacBook Pro `Mac15,6`
- Chip: Apple M3 Pro, 11 cores (5 performance, 6 efficiency)
- Unified memory: 18 GB (`19,327,352,832` bytes reported by MLX)
- macOS: 15.7.8, build `24G824`
- Free disk at the initial snapshot: 271 GiB; after the sanity run: 267 GiB
- MLX device: `Device(gpu, 0)`
- MLX public device info: Apple M3 Pro; max recommended working set 12,884,918,272 bytes; max buffer length 9,663,676,416 bytes

## Runtime

- `uv`: 0.12.5 (Homebrew, aarch64)
- Python: 3.14.7
- MLX: 0.32.0
- `mlx-lm`: 0.31.3
- `huggingface-hub`: 1.27.0
- `transformers`: 5.15.0
- `tokenizers`: 0.22.2
- `mlx-vlm`: 0.6.13 is installed but was not used for this text-only model

All Python commands were run through `uv run --no-sync`. Because the sandbox cannot write the
default uv cache, commands used the temporary cache directory
`/private/tmp/qwen38-uv-cache`; this does not alter the project environment.

## Memory policy checks

```text
iogpu.wired_limit_mb: 0
```

The wired limit was not changed. No `sudo` command was used, and swap configuration was not
modified.

Startup idle snapshot, before benchmark generation:

```text
vm.swapusage: total = 2048.00M  used = 876.06M  free = 1171.94M (encrypted)
vm_stat page size: 16384 bytes
Pages free: 548666
Pages active: 212637
Pages inactive: 193250
Pages speculative: 20677
Pages wired down: 131695
Pages stored in compressor: 222598
Pages occupied by compressor: 31946
Pageins: 8771267
Pageouts: 9167
Swapins: 534464
Swapouts: 611672
```

Post-sanity idle snapshot (the OS had reclaimed/reallocated memory between snapshots):

```text
vm.swapusage: total = 2048.00M  used = 967.00M  free = 1081.00M (encrypted)
Pageins: 9837626
Pageouts: 10341
Swapins: 850462
Swapouts: 964392
```

Swap values are observations, not rejection criteria. Per-run samples are in the corresponding
`*-memory.jsonl` files.
