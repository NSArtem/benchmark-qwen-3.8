# Qwen3.8-27B GGUF / llama.cpp environment

Initial capture: 2026-08-17, Asia/Yerevan. This branch uses the installed
llama.cpp build on the shared Apple Silicon Mac; it does not compile a second
llama.cpp and does not use Ollama.

## Machine

- Model: MacBook Pro `Mac15,6` (`MRX33KS/A`)
- Chip: Apple M3 Pro, 11 cores (5 performance, 6 efficiency)
- Unified memory: 18 GB; `hw.memsize` = `19,327,352,832` bytes
- macOS: 15.7.8, build `24G824`
- Kernel: Darwin 24.6.0, arm64
- `sysctl iogpu.wired_limit_mb`: `0` (read-only check; unchanged)
- Free disk at initial GGUF-branch snapshot: 268 GiB

Hardware capture:

```text
Hardware:

    Hardware Overview:

      Model Name: MacBook Pro
      Model Identifier: Mac15,6
      Model Number: MRX33KS/A
      Chip: Apple M3 Pro
      Total Number of Cores: 11 (5 performance and 6 efficiency)
      Memory: 18 GB
```

Disk capture:

```text
Filesystem      Size  Used  Avail Capacity  Mounted on
/dev/disk3s1    460Gi 164Gi 268Gi 38%        /System/Volumes/Data
```

The sandbox initially denied `sysctl hw.memsize` and
`sysctl iogpu.wired_limit_mb`; the exact read-only calls were repeated with
approval and returned the values above. No `sudo` was used. The wired limit
was not changed.

## llama.cpp

- `llama-cli`: `/opt/homebrew/bin/llama-cli`
- `llama-bench`: `/opt/homebrew/bin/llama-bench`
- `llama-tokenize`: `/opt/homebrew/bin/llama-tokenize`
- Version: `0.1.0-dev (build 10450, commit ece963f41)`
- Build: AppleClang 17.0.0.17000604, Darwin arm64
- Backends observed from `llama-bench --help`: BLAS, MTL, CPU (`apple_m2_m3`)

The installed CLI exposes the required controls: native `--hf-repo`,
`--no-mmproj`, `--reasoning off`, `--cache-type-k`, `--cache-type-v`,
`--ubatch-size`, and `--flash-attn`. Startup Metal/offload and hybrid-model
details will be recorded from the first successful model load below.

`llama-cli --list-devices` returned BLAS and a blank-named device entry in
this build; the backend loader still reported the Metal backend, so actual
device/offload evidence is taken from model startup logs rather than this
summary listing.

## Initial cache

```text
number of models in cache: 1
   1. unsloth/Qwen3.5-9B-GGUF:Q4_K_M
```

No Qwen3.8 GGUF was cached at the start of this branch. Native HF model
loading will use llama.cpp's normal cache outside this repository.

## Safety constraints

- No model weights, HF cache, llama.cpp cache, temporary binaries, or virtual
  environments belong in Git.
- No `mlock`, CPU-only forcing, swap configuration changes, or attempts to
  pin nearly all unified memory will be used.
- Swap is descriptive telemetry, not an automatic rejection criterion.

