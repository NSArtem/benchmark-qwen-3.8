# 16K tuning notes

The f16 run is the baseline that fits with the targeted `--load-mode none`
loader. Q8 K/V is the preferred cache for this 18 GB machine.

| Run | KV | Batch | ubatch | FA | Prefill tok/s | Decode tok/s | KV MiB | Peak RSS MiB | Swap before -> after | Verdict |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| baseline-16k-f16-load-none | f16/f16 | 256 | 128 | auto | 32.49 | 4.05 | 1056 | 12558.6 | 537.56 -> 873.25 | coherent; more memory/swap |
| correctness-16k-q8-load-none | q8_0/q8_0 | 256 | 128 | auto | 46.92 | 4.63 | 561 | 12211.8 | 545.56 -> 537.56 | coherent; safe fallback |
| tuning-16k-q8-ub512 | q8_0/q8_0 | 512 | 512 | auto | 52.69 | 4.39 | 561 | 12345.2 | 873.25 -> 865.25 | coherent; selected |

Q8 saves 495 MiB of KV at 16K, a 46.9% reduction from f16. The larger
ubatch improves prefill by about 12% over the ubatch-128 run while leaving
about 2.31 GiB free in the final Metal breakdown. Decode is effectively
unchanged, so ubatch 512 is the practical prefill/memory sweet spot.

The only Flash Attention comparison used the selected Q8 K/V and ubatch 512.
FA off did not create a context: llama.cpp reported that quantized V cache
requires Flash Attention. Therefore `-fa auto` is retained; no performance
claim is made for a Q8 FA-off path.
