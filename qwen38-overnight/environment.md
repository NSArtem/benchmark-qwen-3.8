# Qwen3.8 overnight environment

Safety gate: `iogpu.wired_limit_mb` must remain zero for all runs.

## Machine

- Platform: `macOS-15.7.8-arm64-arm-64bit-Mach-O`
- Architecture: `arm64`
- Physical memory: `19327352832` bytes
- Hardware report:

```text
Hardware:

    Hardware Overview:

      Model Name: MacBook Pro
      Model Identifier: Mac15,6
      Model Number: MRX33KS/A
      Chip: Apple M3 Pro
      Total Number of Cores: 11 (5 performance and 6 efficiency)
      Memory: 18 GB
      System Firmware Version: mBoot-18000.161.9
      OS Loader Version: 11881.140.96.701.1
```

- macOS:

```text
ProductName:		macOS
ProductVersion:		15.7.8
BuildVersion:		24G824
```

- Free disk at start: `Filesystem      Size    Used   Avail Capacity iused ifree %iused  Mounted on
/dev/disk3s1   460Gi   128Gi   305Gi    30%    530k  3.2G    0%   /System/Volumes/Data`
- `iogpu.wired_limit_mb`: `0`

## Tooling

- uv: `uv 0.12.5 (Homebrew 2026-08-14 aarch64-apple-darwin)`
- Python: `3.14.7`
- mlx: `0.32.0`
- mlx-lm: `0.31.3`
- mlx-vlm: `0.6.13`
- mlx-dspark: `0.12.0`
- transformers: `5.15.0`
- huggingface-hub: `1.27.0`
- httpx: `0.28.1`

## Prompt provenance

- Original: `/Users/artem/projects/benchmark-qwen-3.8/coding_prompt.md`
- Copied prompt: `/Users/artem/projects/benchmark-qwen-3.8/qwen38-overnight/coding-prompt.txt`
- SHA256: `67dcccb037993b679c5593073bbc1875ad61541827a6fc3bb6720410b8096903`
- Size: 61240 bytes / 61236 UTF-8 characters
- Actual model-formatted token counts are recorded in each model run JSON.

## Baseline idle memory

The same `memory_pressure -Q`, `vm_stat`, and `sysctl vm.swapusage` metrics used by the watchdog were sampled three times before model work.

```json
[
  {
    "compressor_pages": 14629,
    "free_percent": 89.0,
    "page_size": 16384,
    "pageins": 454746,
    "pageouts": 1163,
    "swap": {
      "free_mb": 0.0,
      "total_mb": 0.0,
      "used_mb": 0.0
    },
    "swapins": 0,
    "swapouts": 0,
    "timestamp": 1786918904.700932
  },
  {
    "compressor_pages": 14629,
    "free_percent": 89.0,
    "page_size": 16384,
    "pageins": 454749,
    "pageouts": 1163,
    "swap": {
      "free_mb": 0.0,
      "total_mb": 0.0,
      "used_mb": 0.0
    },
    "swapins": 0,
    "swapouts": 0,
    "timestamp": 1786918905.717101
  },
  {
    "compressor_pages": 14629,
    "free_percent": 89.0,
    "page_size": 16384,
    "pageins": 454752,
    "pageouts": 1163,
    "swap": {
      "free_mb": 0.0,
      "total_mb": 0.0,
      "used_mb": 0.0
    },
    "swapins": 0,
    "swapouts": 0,
    "timestamp": 1786918906.748554
  }
]
```
