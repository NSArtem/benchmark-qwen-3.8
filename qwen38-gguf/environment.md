# Qwen3.8-27B GGUF / llama.cpp environment

Capture date: 2026-08-17, Asia/Yerevan. This branch used the installed
llama.cpp build on the shared Apple Silicon Mac. No second llama.cpp was
compiled, Ollama was not used, and no sudo/mlock/system memory setting was
used.

## Machine

- Model: MacBook Pro `Mac15,6` (`MRX33KS/A`)
- Chip: Apple M3 Pro, 11 cores (5 performance, 6 efficiency)
- Unified memory: 18 GB; `hw.memsize` = `19,327,352,832` bytes
- macOS: 15.7.8, build `24G824`
- Kernel: Darwin 24.6.0, arm64
- `sysctl iogpu.wired_limit_mb`: `0` (read-only check; unchanged)
- Free disk at initial branch capture: 268 GiB

The relevant read-only checks were:

```text
Hardware:

    Model Name: MacBook Pro
    Model Identifier: Mac15,6
    Model Number: MRX33KS/A
    Chip: Apple M3 Pro
    Total Number of Cores: 11 (5 performance and 6 efficiency)
    Memory: 18 GB
```

```text
Filesystem      Size  Used  Avail Capacity  Mounted on
/dev/disk3s1    460Gi 164Gi 268Gi 38%        /System/Volumes/Data
```

## llama.cpp

- `llama-cli`: `/opt/homebrew/bin/llama-cli`
- `llama-bench`: `/opt/homebrew/bin/llama-bench`
- `llama-tokenize`: `/opt/homebrew/bin/llama-tokenize`
- Version: `0.1.0-dev (build 10450, commit ece963f41)`
- Build: AppleClang 17.0.0.17000604, Darwin arm64
- Backends exposed by `llama-bench --help`: BLAS, MTL, CPU (`apple_m2_m3`)

The installed CLI exposes the required controls: native `-hf` loading,
`--no-mmproj`, `--reasoning off`, `-ctk/-ctv`, `-ub`, `-fa`,
`--fit`, and `--load-mode`. The installed parser accepts the short
`-ctk/-ctv` spellings used in the commands.

## Metal and hybrid-model startup

Successful Qwen3.8 startup logs report:

- Metal device: `MTL0 Apple M3 Pro`, 12,288 MiB working set, 12,287 MiB
  free at device selection.
- GGUF architecture: `qwen35`
- `qwen35.block_count = 65`
- `qwen35.full_attention_interval = 4`
- `n_layer_all = 65`
- Model type: 27B; model parameters: 27.32B
- The 16K Q8/ubatch-512 run offloaded 56/66 layers to GPU, confirming the
  hybrid path rather than treating the model as an all-full-attention cache.
- Its model buffers were CPU 1,948.17 MiB and MTL0 9,208.05 MiB.
- Its KV cache was 561 MiB total: CPU 70.12 MiB + MTL0 490.88 MiB,
  covering 16 full-attention KV layers.
- Its recurrent state was 149.63 MiB total: CPU 24.94 MiB + MTL0 124.69 MiB.
- Its compute buffers were MTL0 148.78 MiB and CPU 148.78 MiB.
- Final 16K MTL breakdown: 2,314 MiB free, 9,208 MiB model, 615 MiB
  context, 148 MiB compute.
- The final 32K run retained 2,219 MiB free in the MTL breakdown, with
  8,897 MiB model, 1,005 MiB context, and 164 MiB compute.

The installed build's default/mmap-backed path retained an oversized Metal
mapping for the larger Q3_K_M candidate and failed even a tiny generation.
A single targeted `--load-mode none` check was therefore used for the
practical candidate; this was not a broad mmap/no-mmap sweep.

## Model cache

At the initial capture:

```text
number of models in cache: 1
   1. unsloth/Qwen3.5-9B-GGUF:Q4_K_M
```

At completion, llama.cpp reported:

```text
number of models in cache: 3
   1. unsloth/Qwen3.5-9B-GGUF:Q4_K_M
   2. unsloth/Qwen3.8-27B-GGUF:IQ3_XXS
   3. unsloth/Qwen3.8-27B-GGUF:Q3_K_M
```

These cached model files remain outside the repository.

## Safety and repository hygiene

- No model weights, HF cache, llama.cpp cache, temporary binaries, or
  virtual environments are in Git.
- No `mlock), CPU-only forcing, swap configuration changes, or attempts to
  pin nearly all unified memory were used.
- Swap is descriptive telemetry, not an automatic rejection criterion.
- GGUF model files are ignored if accidentally placed under the experiment
  directory.
