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

## Pi coding agent (Stage 9)

Pi 1.0.0 owned the agent loop and its built-in tools (`read`, `bash`, `edit`,
`write`) against `qwen38-server` through a plain `models.json` provider,
streaming, greedy decoding, 16K window with the example compaction settings.
Seven fixture tasks with hidden evaluators (`benchmarks/pi_agent.py`), run on
2026-10-03 after the reconciliation fix:

| Task | Result | Pi turns | Tool calls | Max context | Wall |
| --- | --- | ---: | ---: | ---: | ---: |
| single-file bug | pass | 6 | 6 | 2,862 | 66 s |
| multi-file bug | pass | 7 | 10 | 4,316 | 123 s |
| two stacked bugs | fail (model loop) | 58 | 56 | 12,306 | 773 s |
| search/read across files | pass | 7 | 12 | 3,363 | 93 s |
| long command output | pass | 4 | 4 | 9,776 | 168 s |
| early constraint | pass | 9 | 9 | 4,772 | 165 s |
| three-prompt session (`pi -c`) | pass | 13 | 11 | 7,188 | 320 s |

| Metric | Result |
| --- | ---: |
| Tasks passed | 6 / 7 |
| Model requests | 105 (all streamed, all HTTP 200) |
| Tool calls / invalid (schema or unknown tool) | 108 / 0 |
| Tool-call parse failures | 2 (both calls cut by Pi's output budget) |
| Exact cache reuse / fresh prefill | 96 / 9 (7 task starts, 1 compaction) |
| Warm TTFT median / p95 / max | 1.4 s / 7.4 s / 54 s (3.5K-token log result) |
| Cold first turn (~1.7K-token Pi prompt) | ~22 s |
| Fresh prefill after compaction (11.7K tokens) | 156 s |
| Decode median (range) | 11.6 tok/s (11.1–11.9) |
| MLX peak | 14.3 GiB |
| Total wall time | 28.5 min |

Every warm turn prefilled only its new tokens (median 67, p95 577). The one
failure was a greedy repetition loop: the model reran one debugging command
with identical results until Pi's compaction and output budget ran out. An
earlier run (before the reconciliation fix, scored with the same evaluators)
also passed 6/7, failing the same task with a fix that passed the tests but
broke day arithmetic.
