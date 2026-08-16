# Qwen3.8-27B native MTP feasibility test

Run date: 2026-08-17

## Environment

- Hardware: MacBook Pro Mac15,6, Apple M3 Pro (11 cores), 18 GB unified memory
- macOS: 15.7.8 (24G824)
- uv: 0.12.5 (Homebrew, aarch64)
- Python: 3.14.7
- MLX: 0.32.0
- mlx-vlm: 0.6.13
- mlx-lm: 0.31.3
- mlx-dspark: 0.12.0
- transformers: 5.15.0
- huggingface-hub: 1.27.0

The installed `mlx-vlm.generate --help` exposes `--draft-model`, native
`--draft-kind mtp`, and `--draft-block-size`. No dependency change was needed.
The native-MTP path was not reached because the target-only feasibility gate
failed first.

## Models and workload

- Target: `mlx-community/Qwen3.8-27B-4bit`
- Native MTP drafter: `mlx-community/Qwen3.8-27B-MTP-4bit`
- Prompt: [`coding-prompt.txt`](coding-prompt.txt), copied byte-for-byte from
  the prior Qwen3.5 coding workload
- Prompt SHA256: `67dcccb037993b679c5593073bbc1875ad61541827a6fc3bb6720410b8096903`
- Actual Qwen3.8-formatted prompt count: 16,767 tokens, as recorded for this
  exact prompt/model in the earlier feasibility artifact
- Requested generation: 128 tokens, batch size 1, temperature 0, thinking disabled
- Prefill step size: 2,048
- Published metadata context: target approximately 16.1 GB; MTP approximately
  239 MB / 66.4M parameters; native MTP block size 3
- Runtime-observed MTP block size: not available because MTP was not loaded

## Feasibility gate result

The target-only process loaded the target and began the required short 4-token
sanity generation. It then failed before processing the long prompt with:

`[METAL] Command buffer execution failed: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory).`

This is a target-only failure at the `target_sanity_generation` stage, not an
MTP compatibility failure and not a long-prompt throughput result. The process
was not retried with a smaller context or altered quantization. Per the stop
condition, the MTP drafter was not loaded or tested.

| Mode           | Input tokens | Prefill tok/s | TTFT | Decode tok/s | Wall time | Peak MLX | Paging | Verdict |
| -------------- | -----------: | ------------: | ---: | -----------: | --------: | -------: | ------ | ------- |
| Qwen3.8 target |       16,767* |             — |    — |            — |         — |        — | Genuine pressure/OOM during 4-token sanity | Not viable |
| + native MTP   |             — |             — |    — |            — |         — |        — | Not attempted | Stopped by gate |

\* The long prompt was not processed in this run; 16,767 is the previously
recorded formatted count for this exact prompt and Qwen3.8 target.

## Memory evidence

The lightweight sampler preserved observations in [`memory.log`](memory.log):

- Start: 87% free memory; swap used 399.44 MB.
- First sample after 5.2 s: 15% free; swap used 1,170.31 MB; page-ins about
  166,909/s and swap-outs about 10,378/s.
- Second sample after another 16.5 s: 9% free; swap used 1,896.19 MB; page-ins
  about 5,893/s and swap-outs about 27,079/s.
- The target process then received the Metal out-of-memory error.

This is genuine memory failure, not merely harmless macOS swap allocation:
free memory collapsed, compressed-memory pages rose sharply, swap grew while
the command buffer failed, and the process could not complete even the short
sanity generation. No MTP memory overhead or throughput can be measured.

## Direct answers

1. No. Qwen3.8-27B-4bit did not remain usable on this 18 GB Mac; it failed
   during the short sanity generation after loading.
2. No target-only decode speed was obtained.
3. The evidence shows genuine memory pressure and a Metal OOM, not ordinary
   inactive-page swap use.
4. Native MTP was not loaded because the target-only gate failed.
5. MTP additional memory is unavailable; no drafter measurement was attempted.
6. MTP acceptance behavior is unavailable.
7. MTP decode speed is unavailable.
8. No speedup can be calculated.
9. No APC/server experiment is justified by this result.
10. The precise failure stage was target-only short sanity generation: target
    loading completed, then the first tiny generation hit Metal OOM before the
    16K prompt prefill.

## Verdict and limitations

The target itself is not practically viable at the requested memory envelope
under this current `mlx-vlm` generation path. A tiny native MTP drafter cannot
rescue a target that already reaches severe pressure and Metal OOM before the
real context is processed. The experiment stops here as required; no MTP,
APC, context reduction, quantization change, or optimization was attempted.

No generated text or token IDs exist because the sanity generation failed.

## Artifacts and commits

- Target log: [`baseline.log`](baseline.log)
- Memory observations: [`memory.log`](memory.log)
- Partial harness result: [`baseline.json`](baseline.json)
- Harness: [`benchmark.py`](benchmark.py)
- Commands: [`commands.txt`](commands.txt)
- Harness/config commit: `266fca3` (`benchmark: add Qwen3.8 native MTP feasibility test`)
- Measurement/report commit: `0ed5bec` (`benchmark: record Qwen3.8 native MTP results`)

No `mtp.log` exists because the native-MTP stage was intentionally not
attempted. Model weights, Hugging Face caches, `.venv`, and Python caches were
not staged. Pre-existing unrelated root archives were preserved.
