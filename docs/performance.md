# Performance

All figures are historical measurements: g128 model, M3 Pro with 18 GB, MLX
0.32.1, greedy decoding, normal KV cache. They were recorded between
2026-08-24 and 2026-09-02. Treat them as approximate. Peak is MLX's reported
peak memory.

## Model

| Item | Value |
| --- | --- |
| Weight payload | 10,930,715,648 bytes (10.180 GiB) |
| Quantization | affine, 3 bits, group size 128, 3.25 bits per weight effective |
| Layers | 64 total: 48 GatedDelta (recurrent), 16 full attention |

## Persistent chat by context

Warm turns are follow-ups that use exact-suffix cache reuse. Cold TTFT is the
first full prefill of the initial prompt.

| Initial prompt | Cold TTFT | Warm TTFT | Warm decode | Peak |
| ---: | ---: | ---: | ---: | ---: |
| ~8K (8,191) | ~111 s | ~1.1 s | ~11.1 tok/s | ~13.98 GiB |
| ~12K (12,288) | ~193 s | ~1.1 s | ~10.8 tok/s | ~14.61 GiB |
| ~16K (16,727) | ~242–246 s | ~1.2 s | ~10.4–10.6 tok/s | ~15.24 GiB |
| ~20K (20,480), clean host | ~282 s | ~0.7 s | ~10.4 tok/s | ~15.86 GiB |

Cold prefill runs at about 70–80 tok/s. The 20K row is an upper bound on what
works, not a recommendation.

## Long coding-agent session

Reference run of `benchmarks/long_agent.py`, completed 2026-09-02:

| Metric | Result |
| --- | ---: |
| Final logical context | 10,589 tokens |
| Assistant turns | 29 |
| Valid tool requests / invalid | 12 / 0 |
| Forced length truncations / recovered | 3 / 3 |
| Exact cache reuse / fresh fallback | 28 / 0 |
| HTTP errors (409s) | 0 (0) |
| Median / p95 warm TTFT | 1.06 s / 1.97 s |
| Max TTFT (long tool-result prefill) | 92.6 s |
| Median decode | 11.8 tok/s |
| Total wall time | ~257 s |

The benchmark uses short output caps of 8–160 tokens per turn. Its decode
figure is not directly comparable to the persistent-chat table above.
