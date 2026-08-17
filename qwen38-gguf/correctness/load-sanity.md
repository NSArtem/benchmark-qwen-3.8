# Load sanity record

## Primary Q3_K_M

Selected model: `unsloth/Qwen3.8-27B-GGUF:Q3_K_M`  
File size: 13,818,690,528 bytes (12.86 GiB as reported by llama.cpp)  
Build: llama.cpp `b10450-ece963f41`  
Device: `MTL0 Apple M3 Pro`, 12,288 MiB recommended working set

The first normal/default load used the native HF cache path with `--no-mmproj`
and default repacking. It downloaded the model successfully but remained at
about 98–100% CPU and ~10.1 GB RSS without producing startup output for more
than 17 minutes. It was stopped as an initialization failure; it was not
counted as a model-quality result.

The completed cache was then retried with `--no-repack`. The model metadata
loaded correctly, but decode failed with repeated Metal
`kIOGPUCommandBufferCallbackErrorOutOfMemory` errors. The same error remained
with a 128-token physical batch, `-ngl 32`, `-ngl 16`, and explicit
`--load-mode mmap`.

The verbose startup log identifies the relevant allocation:

```text
MTL0: Apple M3 Pro (12288 MiB, 12287 MiB free)
general.architecture = qwen35
qwen35.block_count = 65
qwen35.context_length = 262144
qwen35.full_attention_interval = 4
model type = 27B
model params = 27.32 B
file size = 12.86 GiB (4.04 BPW)
recurrent-state size = 149.62 MiB
```

At `-ngl 16`, llama.cpp reported 16/66 layers offloaded, but the Metal
allocation still reached `9216.00 MiB + 4725.02 MiB = 13941.02 MiB`, above
the 12,288 MiB Metal working-set limit. The run exited with `Error: Compute
error` and generated no assistant tokens. This is a memory/load-path failure,
not malformed output, so the correctness gate has not been declared passed.

Because the conventional Q3_K_M cannot reach a tiny decode on this installed
Metal path, the next step is exactly one pre-confirmed smaller Q3-class
diagnostic: Unsloth `UD-IQ3_XXS`. No quant sweep is planned.

