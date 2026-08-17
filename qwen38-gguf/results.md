# Qwen3.8-27B GGUF / llama.cpp results

Status: setup and model selection complete; model load sanity and correctness
gate are pending.

The MLX branch is complete and is not being rerun. Its control result was
coherent at short context but malformed at the realistic ~4K, ~8K, and ~16K
variants despite reducing the ~16K peak to about 13.42 GB with KV8 and small
prefill chunks. This GGUF branch is therefore both a practical configuration
experiment and a correctness control.

## Direct answers

1. Exact selected GGUF: pending load sanity; selected
   `unsloth/Qwen3.8-27B-GGUF:Q3_K_M`, file `Qwen3.8-27B-Q3_K_M.gguf`.
2. Selection rationale: conventional imatrix-tagged Q3_K_M with useful
   memory headroom and separate optional mmproj.
3–20. Measurements pending. Values will be recorded as `null` rather than
   inferred until the corresponding run is actually performed.

## Run table

| Config | KV | ubatch | FA | Prefill tok/s | Decode tok/s | KV MB | Coherent | Swap | Verdict |
| --- | --- | ---: | --- | ---: | ---: | ---: | :---: | ---: | --- |
| — | — | — | — | — | — | — | — | — | pending |

