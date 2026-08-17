# Qwen3.8 27B 3-bit TextOnly — bounded native MLX rescue

Status: complete. Native MLX does **not** have a practical stable ~16K
configuration for this checkpoint on the 18 GB M3 Pro with the installed
`mlx-lm` stack. Do not continue the MLX tuning branch.

## Prompt

The actual Qwen tokenizer and chat template were used with
`add_generation_prompt=true` and `enable_thinking=false`:

| Input | Value |
| --- | --- |
| Raw input tokens | 16,354 |
| Final chat-formatted tokens | 16,365 |
| Target | 16,384 |
| SHA256 | `1c9259d016122d12af579ca701ffdf0e102884c425b3340efeade6ae042e3f68` |

The prompt is realistic Lantern coding-agent repository material. The compact
variant removes only a duplicate Task heading and redundant summary paragraphs;
it is not padded with meaningless repetition or silently truncated during
generation. See [`prompt-manifest.json`](prompt-manifest.json) and
[`prompt-16k.txt`](prompt-16k.txt).

## Stage results

`Peak MLX` is decimal GB from `mx.get_peak_memory()`. Swap is descriptive;
page-outs and swap growth were not used as stop conditions. Every case used
128 requested greedy output tokens and the normal public
`mlx_lm.load`/`mlx_lm.stream_generate` path.

| Config | Prompt tokens | Coherent | Prefill tok/s | Decode tok/s | Peak MLX | Swap delta | OOM | Verdict |
| --- | ---: | :---: | ---: | ---: | ---: | ---: | :---: | --- |
| Plain baseline, default prefill/cache | 16,365 | no | 176.32 | 8.76 | 17.16 GB | -91.6 MB final; 2.01 GB peak | no | malformed/repetitive |
| `cache_limit=0`, default prefill | 16,365 | no | 117.60 | 8.98 | 17.16 GB | +101.2 MB final; 691 MB peak | no | cache freed, peak unchanged; malformed |
| `prefill_step_size=1024` | 16,365 | no | 85.58 | 6.93 | 15.44 GB | -24.1 MB final; 683 MB peak | yes | cleanup OOM; malformed |
| `prefill_step_size=512` | 16,365 | no | 87.94 | 7.03 | 14.73 GB | -56.0 MB final; 659 MB peak | yes | cleanup OOM; malformed |
| `prefill_step_size=256` | 16,365 | no | 76.92 | 9.87 | 13.89 GB | -16.0 MB final; 595 MB peak | no | pure repeated `!` |
| `cache_limit=0 + prefill_step_size=512` | 16,365 | no | — | — | 14.67 GB | -16.0 MB final; 579 MB peak | yes | prefill OOM; no output |
| Activation quantization short sanity | short | no | — | — | 0.00001 GB | 0 MB | no | incompatible model mode |
| Normal-KV ~8K control | 7,909 | no | 118.32 | 10.66 | 15.52 GB | 0 MB | no | ordinary KV already repeated `0` |
| Normal-KV ~4K fallback control | 4,035 | no | 123.68 | 11.05 | 15.09 GB | 0 MB | no | ordinary KV already repeated `dis` |
| Normal-KV short control | 33 | yes | 52.34 | 11.40 | 12.11 GB | 0 MB | no | coherent control |
| KV8 short control, `quantized_kv_start=0` | 33 | yes | 50.73 | 11.35 | 12.11 GB | -8.0 MB | no | coherent control |
| **KV8 + `prefill_step_size=256` final 16K** | **16,365** | **no** | **73.76** | **8.79** | **13.42 GB** | **+96.9 MB final; 652 MB peak** | **no** | **repeated `!`; not a success** |

Per-case JSON, logs, memory telemetry, and output text were persisted under
[`logs/`](logs/) and [`outputs/`](outputs/) immediately after each process.
The final output is [`outputs/kv8-prefill256-16k.txt`](outputs/kv8-prefill256-16k.txt).

## Direct answers

1. Plain exact ~16K did not work. It fully processed 16,365 tokens and
   completed, but generated malformed/repetitive text.
2. Smaller prefill chunks reduced peak memory the most. `256` reduced the
   standalone peak from 17.16 GB to 13.89 GB; the final KV8+256 attempt was
   13.42 GB.
3. Yes for memory, no for usability: 1024 and 512 still OOMed, while 256
   completed but produced only repeated punctuation.
4. The standalone 256 chunk reduced prefill from 176.32 to 76.92 tok/s,
   a 56.4% penalty. The final KV8+256 run measured 73.76 tok/s.
5. `mx.set_cache_limit(0)` freed the reusable cache (`0` bytes after the run),
   but did not reduce the 17.16 GB peak and slowed prefill to 117.60 tok/s.
6. Activation quantization did not help: the checkpoint fails at model load
   with `Mode ({m.mode}) does not support activation quantization`.
7. It therefore never reached a short generated answer or a 16K run; no
   coherent-output preservation claim is possible.
8. KV8 is not intrinsically broken here: it remained coherent on the same
   known-good short prompt as normal KV. The earlier 16K KV8 result was
   confounded by the already-broken ordinary baseline. At 16K, however, the
   controlled final KV8 attempt still produced repeated punctuation.
9. There is no successful working 16K configuration. The best resource-
   completing attempt was KV8 + `prefill_step_size=256`, but it is not usable.
10. No 16K command should be recommended. The exact final attempted runner
    was:

    ```bash
    UV_CACHE_DIR=/private/tmp/qwen38-uv-cache uv run --no-sync python \
      qwen38-3bit-mlx-rescue/harness/run_case.py \
      --name kv8-prefill256-16k \
      --prompt-file qwen38-3bit-mlx-rescue/prompt-16k.txt \
      --prefill-step-size 256 --kv-bits 8 --quantized-kv-start 0
    ```

11. That final attempt measured 16,365 prompt tokens, 73.76 prefill tok/s,
    8.79 decode tok/s, and 13.42 GB peak MLX memory. It completed without
    OOM but failed the coherence criterion.
12. MLX 16K is not practical for this model/machine/stack combination.
13. Next step: **GGUF / llama.cpp Q3-class experiment**. Do not download or
    benchmark GGUF as part of this rescue.

## Why the GGUF branch is justified

The bounded native MLX plan exhausted exact 16K baseline, free-cache limit,
prefill chunking at 1024/512/256, the one cache+prefill combination,
activation quantization compatibility, and a controlled KV8 re-evaluation.
GGUF/llama.cpp is the appropriate next experiment because it offers explicit
KV-cache quantization, controllable prefill microbatch size, more direct memory
controls, and the same 16K target. This is a handoff, not a GGUF benchmark.

