# mlx-dspark smoke-test results

Date: 2026-08-16

Status: complete. Qwen3-8B paired run measured; Qwen3.8-27B baseline feasibility attempt aborted for genuine paging/thrashing. Qwen3.8-27B DSpark was not attempted afterward.

## Environment

- Apple Silicon: Apple M3 Pro, arm64
- Unified memory: 18 GB
- macOS: 15.7.8 (24G824)
- `uv`: 0.12.5
- Python through `uv`: 3.14.7
- MLX: 0.32.0
- mlx-dspark: 0.12.0
- mlx-lm: 0.31.3
- `mlx-dspark doctor`: passed; Metal device detected

## Prompt and configuration

- Exact prompt: `coding-prompt.txt`
- Exact prompt text is shared by every run, but tokenization differs: 15,409 tokens for Qwen3-8B and 16,767 for Qwen3.8-27B
- Detected Qwen3 context limit: 40,960 tokens; the hybrid Qwen3.8 wrapper did not expose a limit; requested context target: 32,768+
- Models used for the paired test: `mlx-community/Qwen3-8B-8bit`; DSpark drafter `deepseek-ai/dspark_qwen3_8b_block7`
- Qwen3-8B KV cache: 8-bit for both baseline and DSpark
- Qwen3.8-27B KV8 was rejected by the library because its hybrid recurrent-state cache is not a KV cache; the supported feasibility retry used the normal cache (`--kv-bits 0`) for baseline
- Decoding: greedy/deterministic, one 8-token same-prompt warm-up, one 256-token measured generation
- DSpark configuration: automatic cap calibration selected cap 7; default lookup drafts enabled

## Final comparison

| Model | Input | KV | Baseline tok/s | DSpark tok/s | Speedup | Mean accept | MLX peak | Paging | Verdict |
| ----- | ----: | -- | -------------: | -----------: | ------: | ----------: | -------: | ------ | ------- |
| Qwen3-8B-8bit | 15,409 | 8 | 8.9 | 8.6 | 0.97x | 2.876 | 13.6 / 15.2 GB | No sustained high-rate page-outs; swap activity observed | DSpark not useful in this run |
| Qwen3.8-27B-4bit | 16,767 | normal | — | — | — | — | 17.8 GB | Genuine sustained paging/stall during warm-up | Baseline aborted; DSpark not attempted |

## Qwen3-8B measurements

### Baseline

- Prefill: 72.376 s, 212.9 tok/s
- Time to first token: 81.466 s
- Decode: 28.835 s, 8.9 tok/s
- Total generation wall time: 101.211 s
- MLX peak: 13,620.6 MB
- Memory monitor: free-memory percentage 22% minimum; 1,281 page-outs and 188,156 swap-outs over the process window; swap peaked at 1,661.2 MB and ended at 675.8 MB above the 0.0 MB starting point
- The process completed without an abort, and decode remained stable. The baseline memory window starts at process launch and includes load, warm-up, and measurement.

### DSpark

- Automatic cap: 7; lookup rounds: 4
- Prefill: 73.891 s, 208.5 tok/s
- Time to first token: 84.997 s
- Decode: 29.764 s, 8.6 tok/s
- Total generation wall time: 103.655 s
- Mean accepted draft length: 2.876; target forwards: 90; rounds: 89
- MLX peak: 15,172.7 MB
- Measurement-window memory monitor: free-memory percentage 20% minimum; 305 page-outs and 170,960 swap-outs; swap was 784.25 MB before measurement, peaked at 2,559.8 MB, and ended at 855.5 MB
- No sustained-paging abort was triggered. The machine retained reasonable free-memory headroom and page-outs stayed low, although swap-in/out activity was nontrivial.

### Preliminary conclusion before the 27B test

DSpark speedup is `8.6 / 8.9 = 0.97x`, so this run shows no useful acceleration rather than a modest or strong gain. Long-context prefill is intrinsically slow here (about 72–74 seconds for 15.4K tokens), while the measured decode rates were close and stable. Swap grew and macOS performed swap I/O, but the observations did not show severe memory pressure or sustained high-rate page-outs that collapsed generation; swap growth alone was not used as an abort criterion.

## Qwen3.8-27B feasibility test

### KV8 compatibility attempt

The first attempt used `--kv-bits 8` as requested but failed before generation with the library's explicit hybrid-cache error: KV8 is unsupported because only 16 of 64 layers have KV caches and the recurrent-state caches are not KV caches. The raw log is `qwen38-27b-baseline-kv8-unsupported.log`.

### Supported normal-cache baseline

- The model loaded with `--kv-bits 0`; the same prompt encoded to 16,767 tokens.
- MLX peak during load/calibration: 17,847.5 MB.
- The 8-token warm-up did not complete after several minutes and produced no usable prefill, TTFT, or decode measurement.
- During the load/warm-up window, free-memory percentage reached 25%, page-outs increased by 6,762, swap-outs by 475,052, and swap peaked at 7,810.5 MB from a 731.9 MB starting point.
- Independent samples while the warm-up was stalled showed swap-ins and swap-outs continuing at roughly 1.5 million pages per minute. The process was manually terminated after this sustained paging plus the prolonged warm-up stall; the final observation is in `qwen38-27b-baseline-abort-observation.txt`.
- After termination, the machine recovered to 89% free memory and 388.3 MB swap, confirming the pressure was associated with the 27B attempt. No DSpark run was started after baseline failed feasibility.

## Direct answers

1. Qwen3-8B-8bit achieved 8.9 baseline decode tok/s and 8.6 DSpark tok/s at 15,409 tokens with KV8.
2. DSpark was 0.97x baseline, slightly slower; no useful acceleration was measured.
3. Qwen3-8B caused substantial swap/compression activity, but free memory stayed at 20–22% minimum, high-rate page-outs were not sustained, and decode remained stable. This is not evidence of a memory-pressure collapse; it is best described as normal macOS paging with a caveat that the swap I/O was nontrivial.
4. No sustained high-rate page-out behavior was observed during either completed 8B generation.
5. No: on this Mac and this long-context coding prompt, baseline Qwen3-8B is more practical than adding DSpark.
6. Qwen3.8-27B-4bit did not fit as a practically usable long-context run without genuine thrashing.
7. No 27B decode performance was obtained; the warm-up stalled and was aborted.
8. No further parameter sweep or TTRPG benchmark is justified by this smoke test. A future test should first change the hardware/memory conditions or use a model/configuration known to fit, then repeat only the paired comparison.
