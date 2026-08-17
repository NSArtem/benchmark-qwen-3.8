# Qwen3.8 27B 3-bit TextOnly — bounded native MLX rescue

Status: rescue stages are in progress. The required prompt has been prepared
with the actual tokenizer and Qwen chat template; see
[`prompt-manifest.json`](prompt-manifest.json).

## Prompt

| Input | Value |
| --- | --- |
| Raw input tokens | 16,354 |
| Final chat-formatted tokens | 16,365 |
| Target | 16,384 |
| SHA256 | `1c9259d016122d12af579ca701ffdf0e102884c425b3340efeade6ae042e3f68` |

The raw prompt is realistic Lantern coding-agent repository material. The
compact variant removes only a duplicate Task heading and redundant summary
paragraphs; it is not padded with meaningless repetition or silently
truncated during generation.

## Stage results

| Config | Prompt tokens | Coherent | Prefill tok/s | Decode tok/s | Peak MLX | Swap delta | OOM | Verdict |
| --- | ---: | :---: | ---: | ---: | ---: | ---: | :---: | --- |
| pending | 16,365 | — | — | — | — | — | — | — |

Per-case JSON, logs, memory telemetry, and output text are written under
[`logs/`](logs/) and [`outputs/`](outputs/) immediately after each process.

## Decision

Not yet classified. The bounded sequence will stop at the first coherent,
stable ~16K configuration, or recommend GGUF/llama.cpp after the explicitly
allowed MLX attempts are exhausted.

