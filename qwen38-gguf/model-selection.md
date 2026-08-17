# Qwen3.8-27B GGUF selection

Discovery date: 2026-08-17. Hugging Face metadata was queried through
`uv` and llama.cpp's native Hugging Face loader was used for model loading.
No unconfirmed quant name was assumed.

## Initial primary candidate

**Repository:** `unsloth/Qwen3.8-27B-GGUF`
**Exact file/quant:** `Qwen3.8-27B-Q3_K_M.gguf` / `Q3_K_M`
**File size:** 13,818,690,528 bytes (13.82 GB decimal; llama.cpp reports
12.86 GiB)
**Model metadata:** Qwen3.8-27B, `qwen35`, 65 blocks, 27.32B parameters
**Separate mmproj:** yes (`mmproj-BF16.gguf` and `mmproj-F16.gguf`); all
runs used `--no-mmproj`.

This was the sensible first control because it is a conventional,
imatrix-tagged Q3_K_M quant with separate projector files and broad
llama.cpp support. It was not viable with this installed build: default
repacking stalled for more than 17 minutes, while `--no-repack` and
explicit partial-offload attempts still reported a roughly 13.95 GiB Metal
allocation and failed even a tiny deterministic generation.

## Final practical candidate

**Repository:** `unsloth/Qwen3.8-27B-GGUF`
**Exact file/quant:** `Qwen3.8-27B-UD-IQ3_XXS.gguf` /
`UD-IQ3_XXS`
**File size:** 11,913,559,104 bytes (11.91 GB decimal; llama.cpp reports
11.09 GiB and 3.49 BPW)
**Model metadata:** Qwen3.8-27B, `qwen35`, 65 blocks, 27.32B parameters
**Separate mmproj:** yes; it was explicitly disabled.

This is the one planned alternate Q3-class diagnostic, not a quant sweep. It
was selected because it is materially smaller while remaining a confirmed
27B Q3-class text GGUF. Its tiny sanity generation produced `OK`, and the
targeted `--load-mode none` path then produced coherent 4K, 8K, and 16K
answers. The winning long-context configuration uses this exact file through
llama.cpp's normal HF cache.

The cache display shortens the entry to `IQ3_XXS`; the repository metadata
and llama.cpp model path identify the actual downloaded file as
`Qwen3.8-27B-UD-IQ3_XXS.gguf`.

## Enumerated alternatives

These were confirmed by metadata discovery but not benchmarked in this
branch:

| Repository | Exact file | Quant | Bytes | Separate mmproj |
| --- | --- | --- | ---: | :---: |
| `unsloth/Qwen3.8-27B-GGUF` | `Qwen3.8-27B-Q3_K_S.gguf` | Q3_K_S | 12,574,489,568 | yes |
| `unsloth/Qwen3.8-27B-GGUF` | `Qwen3.8-27B-UD-Q3_K_XL.gguf` | UD-Q3_K_XL | 13,441,059,904 | yes |
| `bartowski/Qwen3.8-27B-GGUF` | `Qwen3.8-27B-IQ3_M.gguf` | IQ3_M | 13,903,517,280 | yes |
| `bartowski/Qwen3.8-27B-GGUF` | `Qwen3.8-27B-Q3_K_M.gguf` | Q3_K_M | 14,605,735,520 | yes |
| `AtomicChat/Qwen3.8-27B-GGUF` | `Qwen3.8-27B-AD-IQ3_S.gguf` | AD-IQ3_S | 13,838,267,872 | yes |
| `AtomicChat/Qwen3.8-27B-GGUF` | `Qwen3.8-27B-AD-IQ3_S-IQ3_XXS.gguf` | mixed AD-IQ3 | 12,984,128,992 | yes |
| `Chungulus/Qwen3.8-27B-GGUF` | `Qwen3.8-27B-Q3_K_M.gguf` | Q3_K_M | 13,500,736,896 | not listed |

## Sources

- [Unsloth Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
- [bartowski Qwen3.8-27B-GGUF](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF)
- [AtomicChat Qwen3.8-27B-GGUF](https://huggingface.co/AtomicChat/Qwen3.8-27B-GGUF)
- [Chungulus Qwen3.8-27B-GGUF](https://huggingface.co/Chungulus/Qwen3.8-27B-GGUF)
