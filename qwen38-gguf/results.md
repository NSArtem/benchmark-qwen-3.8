# Qwen3.8-27B GGUF / llama.cpp results

Date: 2026-08-17. Machine: Apple M3 Pro, 18 GB unified memory. The MLX
branch was not rerun.

## Executive result

A confirmed Q3-class GGUF can produce coherent coding-agent output at 4K,
8K, 16K, 24K, and 32K on this machine. The practical recommendation is
16K with Q8 K/V, batch 512, ubatch 512, Flash Attention auto, and the
non-mmap `--load-mode none` path.

The exact selected model is:

```text
Repository: unsloth/Qwen3.8-27B-GGUF
File:       Qwen3.8-27B-UD-IQ3_XXS.gguf
Quant:      UD-IQ3_XXS
Size:       11,913,559,104 bytes (11.91 GB decimal; 11.09 GiB in llama.cpp)
```

The first/main conventional candidate,
`Qwen3.8-27B-Q3_K_M.gguf` (13,818,690,528 bytes), could not pass tiny
load sanity with this installed build: default repacking stalled, and the
no-repack/partial-offload path retained about a 13.95 GiB Metal allocation
and failed with Metal out-of-memory. Per the stop rule, exactly one smaller
Q3-class diagnostic was tested. The selected UD-IQ3_XXS candidate passed the
tiny `OK` test and then passed the long-context gate using
`--load-mode none`.

## Direct answers

1. **Which exact GGUF Q3-class model was selected?**
   `unsloth/Qwen3.8-27B-GGUF:UD-IQ3_XXS`,
   file `Qwen3.8-27B-UD-IQ3_XXS.gguf`. llama.cpp reports Qwen3.8-27B,
   `qwen35`, 65 blocks, and 27.32B parameters. A separate mmproj exists
   in the repository but was disabled.

2. **Why was it selected?**
   It is a confirmed 27B text GGUF in the Q3 quality/size class and is small
   enough for a partial Metal/CPU split. The conventional Q3_K_M was a
   useful control but did not load on this installed Metal path.

3. **Does it produce coherent output at 4K?**
   Yes. The selected Q8 gate run consumed 4,035 prompt tokens and generated
   128 coherent tokens. Prefill: 76.12 tok/s. Decode: 5.75 tok/s.

4. **At 8K?**
   Yes. It consumed 7,903 prompt tokens and generated 128 coherent tokens.
   Prefill: 62.04 tok/s. Decode: 5.39 tok/s.

5. **At 16K?**
   Yes. It consumed 16,359 prompt tokens in a 16,896-token slot, generated
   128 coherent tokens, and reported no truncation or Metal error. The
   conservative ubatch-128 gate run measured 46.92 prompt tok/s and
   4.63 decode tok/s.

6. **Does this differ from the MLX result?**
   Yes. MLX was coherent at short context but malformed at approximately 4K,
   8K, and 16K even after its 16K peak was reduced to about 13.42 GB.
   GGUF/llama.cpp is coherent at all three gate sizes. This is strong
   empirical evidence that the practical issue was specific to the tested
   MLX conversion/runtime path, rather than Qwen3.8 being inherently unable
   to handle these contexts. It is not absolute proof of the underlying
   software bug.

7. **What is baseline 16K prefill speed?**
   The f16-KV baseline that fit with the targeted loader was 32.49 tok/s.
   The preferred Q8/ubatch-512 configuration is 52.69 tok/s.

8. **What is baseline 16K decode speed?**
   The f16-KV baseline was 4.05 tok/s. The preferred Q8/ubatch-512 result
   was 4.39 tok/s.

9. **Does q8_0 KV work correctly?**
   Yes. Q8 K/V was coherent at 4K, 8K, 16K, 24K, and 32K.

10. **How much memory does q8_0 KV save?**
    At 16K, f16 K/V was 1,056 MiB and q8 K/V was 561 MiB: a saving of
    495 MiB, or 46.9%. The selected Q8 run also had lower peak RSS than the
    f16 comparison (12,345 MiB versus 12,559 MiB), though swap is
    descriptive and the runs were sequential.

11. **What ubatch is the practical sweet spot?**
    512. At 16K it improved prefill from 46.92 to 52.69 tok/s, kept decode
    near the same (4.63 to 4.39 tok/s), remained coherent, and left about
    2.31 GiB free in the final Metal breakdown. Ubatch 128 is the safer
    fallback if the machine is under more memory pressure.

12. **Does Flash Attention help?**
    A Q8 V-cache cannot run with FA off in this build: llama.cpp exits with
    `quantized V cache requires flash_attn to be enabled`. Therefore the
    winning command uses `-fa auto`; no unsupported OFF performance number
    is claimed.

13. **What is the best exact 16K command?**

    ```sh
    llama-cli \
      -hf unsloth/Qwen3.8-27B-GGUF:UD-IQ3_XXS \
      --no-mmproj --no-repack --load-mode none \
      --fit on --fit-target 2048 \
      -c 16896 -b 512 -ub 512 -fa auto \
      -ctk q8_0 -ctv q8_0 \
      --reasoning off --reasoning-format none \
      --conversation --multiline-input
    ```

    `--load-mode none` is important for this installed build. The command
    does not use mlock, CPU-only forcing, speculative decoding, MTP, or a
    projector.

14. **Does 24K work?**
    Yes. The selected settings consumed 24,552 prompt tokens in a 25,088
    slot, generated 128 coherent tokens, and measured 42.52 prompt tok/s and
    3.91 decode tok/s. Q8 KV was 833 MiB.

15. **Does 32K work?**
    Yes. It consumed 31,569 prompt tokens in a 32,768 slot, generated 128
    coherent tokens, and measured 28.33 prompt tok/s and 3.36 decode tok/s.
    Q8 KV was 1,088 MiB.

16. **What is the largest practical context?**
    The largest tested coherent context is 32K. The recommended practical
    context is 16K; 24K is usable for occasional long sessions, while 32K
    has a roughly 19.2-minute fresh prefill in this setup.

17. **What is the decode speed there?**
    At the largest tested 32K context: 3.36 tok/s. At the recommended 16K
    sweet spot: 4.39 tok/s.

18. **Is GGUF/llama.cpp now preferable to MLX for this Mac?**
    Yes for Qwen3.8-27B long-context work. It is the only tested branch here
    that passed coherent 4K/8K/16K output, and it also reached 24K/32K. It
    is not fast, but it is functionally usable.

19. **Is the setup practical for coding agents and long-running TTRPG sessions?**
    For coding agents: yes at 16K, with the caveat that a fresh 16K prompt
    takes about 5.7 minutes to prefill; shorter contexts are much more
    responsive. Context reuse in a live interactive session was not measured.
    For TTRPG sessions: yes if turn latency of roughly 4.4 tok/s is
    acceptable; 16K is the sensible default, 24K is an occasional extension,
    and 32K is a slow archival/maximum-context mode.

20. **What is the single highest-value next experiment?**
    Measure a real multi-turn interactive session with the winning command:
    context reuse, subsequent-turn latency, and whether the retained hybrid
    state behaves coherently after several turns. A broad quant or llama.cpp
    feature sweep is lower value now.

## Correctness and prompt provenance

The prompt files reuse the MLX branch's realistic Lantern coding-agent
material. The task tail is identical across all sizes.

| Prompt | SHA256 | llama.cpp tokens |
| --- | --- | ---: |
| [4k.txt](prompts/4k.txt) | `12ba3518c53566fea9073b822819c2ec0ac7540f5208f34f72e5aa82246372e6` | 4035 |
| [8k.txt](prompts/8k.txt) | `bc917834cd4303c66182a9936759e9222e1051ace9b3f3bf4fe67a07145729de` | 7903 |
| [16k.txt](prompts/16k.txt) | `1c9259d016122d12af579ca701ffdf0e102884c425b3340efeade6ae042e3f68` | 16359 |
| [24k.txt](prompts/24k.txt) | `5636b80d934e7b8fd7b798864e5f33b2884cfcdf945e0e8b06c258207fbefe1a` | 24552 |
| [32k.txt](prompts/32k.txt) | `cacb38daa2a93ac4a5688bf92bca592d419d88dc9b6fd07afbc04d0fc8697b75` | 31569 |

Common task-tail SHA256: `794952f16a704e84c6a224fda4ed0a10cc1d31967caff4daf6f6cedf749f47ea`.

## 16K comparison table

| Config | KV | ubatch | FA | Prefill tok/s | Decode tok/s | KV MB | Coherent | Swap | Verdict |
| --- | --- | ---: | --- | ---: | ---: | ---: | :---: | --- | --- |
| f16 baseline | f16/f16 | 128 | auto | 32.49 | 4.05 | 1056 | yes | 537.56 -> 873.25 | works, but more memory/swap |
| Q8 gate | q8_0/q8_0 | 128 | auto | 46.92 | 4.63 | 561 | yes | 545.56 -> 537.56 | safe fallback |
| Q8 selected | q8_0/q8_0 | 512 | auto | 52.69 | 4.39 | 561 | yes | 873.25 -> 865.25 | practical sweet spot |
| Q8 FA off | q8_0/q8_0 | 512 | off | — | — | — | no | 865.25 -> 865.25 | unsupported: quantized V requires FA |

Memory breakdown for the selected 16K row was MTL model 9,208 MiB,
context 615 MiB, compute 148 MiB, with 2,314 MiB free. The model is a
hybrid Qwen3.8 architecture: 65 blocks, full-attention interval 4, and a
16-layer full-attention KV cache.

## Context extension table

| Context | Prompt tokens | Slot | KV MB | Prefill tok/s | Decode tok/s | Total seconds | Coherent |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 4K | 4035 | 4608 | 153 | 76.12 | 5.75 | 75.09 | yes |
| 8K | 7903 | 8192 | 272 | 62.04 | 5.39 | 150.94 | yes |
| 16K | 16359 | 16896 | 561 | 52.69* | 4.39* | 339.42* | yes |
| 24K | 24552 | 25088 | 833 | 42.52 | 3.91 | 609.84 | yes |
| 32K | 31569 | 32768 | 1088 | 28.33 | 3.36 | 1152.26 | yes |

*The 16K starred row is the selected ubatch-512 tuning run. The correctness
gate at ubatch 128 was 46.92/4.63 tok/s.

## Artifacts

- [environment.md](environment.md)
- [model-selection.md](model-selection.md)
- [commands.txt](commands.txt)
- [results.json](results.json)
- [correctness summary](correctness/summary.md)
- [tuning summary](tuning/summary.md)
- [harness](harness/run_case.py)

The complete per-run logs, metadata JSON files, and exact transcripts are in
[logs](logs/), [outputs](outputs/), and [prompts](prompts/).

## Sources

- [Unsloth Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
- [bartowski Qwen3.8-27B-GGUF](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF)
