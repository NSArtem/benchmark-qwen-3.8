# Correctness gate

The same MLX-branch coding-agent prompts were reused. Token counts below are
from llama.cpp for the selected `UD-IQ3_XXS` GGUF; the exact prompt files
and SHA256 hashes are recorded in `results.md`.

| Target | Prompt file | Context slot | Prompt tokens | Output | Classification | Evidence |
| --- | --- | ---: | ---: | ---: | --- | --- |
| ~4K | `prompts/4k.txt` | 4608 | 4035 | 128 | coherent | [output](../outputs/correctness-4k-q8-load-none.txt), [log](../logs/correctness-4k-q8-load-none.log) |
| ~8K | `prompts/8k.txt` | 8192 | 7903 | 128 | coherent | [output](../outputs/correctness-8k-q8-load-none.txt), [log](../logs/correctness-8k-q8-load-none.log) |
| ~16K | `prompts/16k.txt` | 16896 | 16359 | 128 | coherent | [output](../outputs/correctness-16k-q8-load-none.txt), [log](../logs/correctness-16k-q8-load-none.log) |

Settings for these gate runs: Q8 K/V, Flash Attention auto, batch 256,
ubatch 128, `--fit on --fit-target 2048 --load-mode none`,
`--reasoning off`, temperature 0, deterministic seed 42, no mmproj, no
speculative decoding, no MTP, and no context truncation.

The `-o` files contain the prompt/session transcript. The generated answer
begins after the final assistant marker; all three answers are normal
coding-agent diagnoses with no digit loops, punctuation loops, unrelated
token loops, or repeated single-word failure.

The conventional `Q3_K_M` candidate failed load/memory sanity before a
correctness gate. The single planned alternate quant diagnostic initially
also hit the mmap-backed Metal allocation limit at 4K, but the targeted
`--load-mode none` loader path produced the successful gate runs above.
