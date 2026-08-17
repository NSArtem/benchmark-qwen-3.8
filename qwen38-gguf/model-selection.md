# Qwen3.8-27B GGUF selection

Discovery date: 2026-08-17. Repository and file metadata were queried through
the Hugging Face API using `uv`; no unconfirmed quant name was assumed.

## Primary candidate

**Repository:** `unsloth/Qwen3.8-27B-GGUF`  
**Exact file/quant:** `Qwen3.8-27B-Q3_K_M.gguf` / `Q3_K_M`  
**File size:** 13,818,690,528 bytes (13.82 GB decimal, 12.87 GiB)  
**Parameter count:** Qwen3.8-27B (27B class; exact GGUF metadata will be
confirmed in the llama.cpp startup log)  
**Separate mmproj:** yes (`mmproj-BF16.gguf` and `mmproj-F16.gguf` are present
in the repository); this experiment uses `--no-mmproj`.

This is the first/main candidate because it is a conventional, widely
supported Q3_K_M quant, carries the publisher's imatrix repository tag, and
leaves more room for a 16K hybrid-model state/KV/compute allocation than the
larger conventional Q3 variants. It also keeps the comparison interpretable:
the first answer is about a normal Q3-class GGUF rather than an unusual
mixed-format experiment.

## Confirmed alternatives, not benchmarked initially

Metadata enumeration found these relevant files:

| Repository | Exact file | Quant | Bytes | Separate mmproj |
| --- | --- | --- | ---: | :---: |
| `unsloth/Qwen3.8-27B-GGUF` | `Qwen3.8-27B-Q3_K_S.gguf` | Q3_K_S | 12,574,489,568 | yes |
| `unsloth/Qwen3.8-27B-GGUF` | `Qwen3.8-27B-UD-IQ3_XXS.gguf` | UD-IQ3_XXS | 11,913,559,104 | yes |
| `unsloth/Qwen3.8-27B-GGUF` | `Qwen3.8-27B-UD-Q3_K_XL.gguf` | UD-Q3_K_XL | 13,441,059,904 | yes |
| `bartowski/Qwen3.8-27B-GGUF` | `Qwen3.8-27B-IQ3_M.gguf` | IQ3_M | 13,903,517,280 | yes |
| `bartowski/Qwen3.8-27B-GGUF` | `Qwen3.8-27B-Q3_K_M.gguf` | Q3_K_M | 14,605,735,520 | yes |
| `AtomicChat/Qwen3.8-27B-GGUF` | `Qwen3.8-27B-AD-IQ3_S.gguf` | AD-IQ3_S | 13,838,267,872 | yes |
| `AtomicChat/Qwen3.8-27B-GGUF` | `Qwen3.8-27B-AD-IQ3_S-IQ3_XXS.gguf` | mixed AD-IQ3 | 12,984,128,992 | yes |
| `Chungulus/Qwen3.8-27B-GGUF` | `Qwen3.8-27B-Q3_K_M.gguf` | Q3_K_M | 13,500,736,896 | no projector listed |

The primary Q3_K_M is not a claim that it is the global best quant. It is a
defensible first control. Per the stopping rule, only one alternate quant
will be tested if the primary already produces the same malformed output at
4K/8K.

## Sources

- [Unsloth Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
- [bartowski Qwen3.8-27B-GGUF](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF)
- [AtomicChat Qwen3.8-27B-GGUF](https://huggingface.co/AtomicChat/Qwen3.8-27B-GGUF)
- [Chungulus Qwen3.8-27B-GGUF](https://huggingface.co/Chungulus/Qwen3.8-27B-GGUF)

