# Qwen3.8-27B overnight exploration

Generated: 2026-08-17 03:12:28 +04

This report contains only measurements actually attempted in this run. Failed gates are retained as evidence; skipped stages are not presented as measurements.

## Safety and scope

- Machine: Apple M3 Pro with 18 GB unified memory.
- `iogpu.wired_limit_mb`: `0`; it was never modified.
- Qwen3.8 27B 4-bit was not retried; no sudo, wired-memory policy change, swap hack, or unrelated-process termination was used.
- Each model run used a separate child process, a one-second watchdog, and a bounded wall-clock timeout.

### Watchdog policy

The watchdog samples `memory_pressure -Q`, `vm_stat`, and `sysctl vm.swapusage` about once per second. Swap growth alone is not failure. It aborts only for: free memory ≤8% for 3 samples; free memory ≤12% with meaningful page-outs/swap-outs for 3 samples; critical swap-out pressure for 2 samples; a child Metal OOM; or progress stalled for 25 seconds during worsening pressure. Meaningful pressure is ≥25 page-outs/s or ≥500 swap-outs/s. The process group contains only the benchmark child and descendants.

## Workload and runtime

- Prompt source: `/Users/artem/projects/benchmark-qwen-3.8/coding_prompt.md`
- Prompt SHA256: `67dcccb037993b679c5593073bbc1875ad61541827a6fc3bb6720410b8096903`
- Prompt size: `61240` bytes / `61236` characters.
- Practical runs use the same prompt, batch size 1, temperature 0, thinking disabled, prefill step 2048, and 128 generated tokens.
- The actual chat-formatted token count is model-specific and is recorded in each practical run.

## Candidate table

| Target | Mode | 16K stable | Prefill tok/s | Decode tok/s | MTP speedup | Peak MLX | Min free | Paging | APC hit | Warm TTFT | Quality | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |
| Qwen3.8-27B uniform 3-bit text-only | plain | no | 138.8 | 10.10 | — | 17.28 GB | 18.0% | meaningful_sustained_paging | — | — | 3/5 | not practically viable: sustained paging |
| Qwen3.8-27B 2-bit text-only | plain | yes | 74.9 | 13.14 | — | 13.92 GB | 20.0% | no_meaningful_sustained_paging | — | — | 0/5 | stable run but quality smoke 0/5 |

`Peak MLX` is MLX’s reported allocation peak, not total system RSS. `Min free` and paging are system-level watchdog observations; headroom claims use both rather than treating swap allocation alone as a failure.

## Metadata and attempted stages

- `lukaskremla/Qwen3.8-27B-2bit-MLX-TextOnly`: status `ok`, revision `33b90b60fd7ba16b668854e049bd65e22d6afddf`, reported weight download `8.41 GB`, text-only inference `{'basis': 'repository_name', 'text_only': True}`.
- `lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly`: status `ok`, revision `c98bba5926f51fec1c8d8737e577221673f524d7`, reported weight download `11.77 GB`, text-only inference `{'basis': 'repository_name', 'text_only': True}`.
- `rapid-mlx/Qwen3.8-27B-mixed-3.5bpw-MLX`: status `ok`, revision `d8b68e21eab505332d4a456dacec6d4125c96d73`, reported weight download `13.91 GB`, text-only inference `{'basis': 'vision_or_processor_metadata_present', 'processor_files': ['preprocessor_config.json', 'processor_config.json', 'video_preprocessor_config.json'], 'text_only': False, 'vision_keys': ['image_token_id', 'video_token_id', 'vision_config']}`.
- `q38-2bit/target-practical`: `complete`; watchdog `False` ``.
- `q38-2bit/target-tiny`: `failed_or_aborted`; watchdog `False` ``.
- `q38-2bit/target-tiny-text-only-fixed`: `complete`; watchdog `False` ``.
- `q38-3bit/target-practical`: `complete`; watchdog `False` ``.
- `q38-3bit/target-tiny`: `failed_or_aborted`; watchdog `False` ``.
- `q38-3bit/target-tiny-text-only`: `failed_or_aborted`; watchdog `False` ``.
- `q38-3bit/target-tiny-text-only-detokenizer`: `complete`; watchdog `False` ``.
- `q38-3bit/target-tiny-text-only-embedding`: `failed_or_aborted`; watchdog `False` ``.
- `q38-3bit/target-tiny-text-only-stopping`: `failed_or_aborted`; watchdog `False` ``.
- `q38-mixed-3.5/target-tiny`: `watchdog_aborted`; watchdog `True` `critical_swapout_rate_with_pressure`.

## Quality smoke

- `q38-2bit`: 0/5; status `complete`.
  - A-coding: `fail` — case [] returned None, expected []
  - B-structured-json: `fail` — invalid JSON: Expecting value: line 1 column 1 (char 0)
  - C-reasoning: `fail` — got 'Here, 323 is the product of two primes, which is a nice number to have. Compute the product of two primes, namely $a$ = $17$ and $b$ = $19$, then output: $323$ = $17 \\times 19$. This problem asks you to compute $323$ exactly, which is a nice number to have. Return the exact value of $323$ in the first place, then use it to compute $17 \\times 19$ to get $323$? Then, a nice number to have. Return exactly `ANSWER:`\n\n323\n\n**Core Thought process to answer**\n17 × 19 = ?\n**Core evidence**\n- The product of two primes, namely $a$ = $17$ and $b$ = $19$, is a nice number to have. Compute the product of two primes, namely $a$ = $17$ and $b$ = $19$, then output: $323$ = $17 \\times 19$. This problem asks you to compute $323$ exactly, which is a nice'
  - D-long-context-retrieval: `fail` — got 'Based on the analysis of the provided text, the pattern for each record follows a consistent cycle of `tenant-X` (where X is the record number) leading to `lantern-X-blue` or `lantern-X-amber`.\n\nThe pattern is clear:\n- Records with `record` leading to `blue` (records where the record number is divisible by the cycle of the tenant).\n- Records with `record` leading to `blue` (records where the record number is divisible by the cycle of the Tenant).\n\nTherefore, the record `057` corresponds to the `lantern-057-blue` checksum, which matches the requirement exactly.\n\nANSWER: lantern-057-blue'
  - E-instruction-following: `fail` — got '```\ninline\n```\n</think>'
- `q38-3bit`: 3/5; status `complete`.
  - A-coding: `fail` — case [[5, 7], [1, 2], [2, 4]] returned [[1, 4], [5, 7]], expected [[1, 7]]
  - B-structured-json: `pass` — exact schema/value match
  - C-reasoning: `pass` — exact objective answer
  - D-long-context-retrieval: `fail` — got '-1\n\n\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**\n**'
  - E-instruction-following: `pass` — exact constrained format

## Direct answers

1. **No Qwen3.8-27B configuration met all three requirements (quality, stable 16K operation, and practical unattended behavior).** Uniform 3-bit had the best smoke result (3/5) but failed the 16K paging gate; 2-bit passed the memory gate but scored 0/5.
2. Uniform 3-bit measured **10.10 tok/s** at `16727` tokens; the stable 2-bit control measured **13.14 tok/s**.
3. Uniform 3-bit: MLX peak **17.28 GB**, minimum free **18.0%**. 2-bit: MLX peak **13.92 GB**, minimum free **20.0%**.
4. Uniform 3-bit showed **meaningful_sustained_paging** with page-out delta `449` and no swap-out delta; 2-bit showed **no_meaningful_sustained_paging** with no swap-outs.
5. Native MTP was **not attempted** for a practical candidate: 3-bit failed the paging gate, 2-bit failed the quality gate, and mixed 3.5 bpw was watchdog-aborted during tiny loading.
6. MTP acceptance and speedup: **not measured**; no unsupported estimate is made.
7. APC: **not run** because no candidate qualified for a persistent coding-agent server test.
8. APC cold/warm TTFT: **not measured**. For context, 3-bit cold TTFT was approximately 120.66 s at 16,727 tokens, and 2-bit cold TTFT was approximately 223.47 s; neither is an APC result.
9. Compared with the existing Qwen3.5-9B-4bit reference (24.16 tok/s plain, 29.85 tok/s native MTP, APC cold/warm about 64.83/1.49 s), Qwen3.8 3-bit was slower at 10.10 tok/s and paged; 2-bit was 13.14 tok/s and stable but failed every smoke task.
10. Is the extra size worth it? **Not for this unattended 18 GB / 16K use case based on these measurements.** 3-bit showed some quality signal (3/5) but insufficient memory headroom; 2-bit supplied no useful quality signal.
11. Mixed 3.5 bpw provided **no demonstrated quality advantage**: its tiny target run triggered conservative watchdog termination under sustained swap-out pressure, so no practical or quality run was authorized.
12. 2-bit is **not usable for the tested coding-agent smoke** (0/5), despite being the only Qwen3.8 target that completed the 16K memory gate cleanly.
13. Recommended configuration: keep using **Qwen3.5-9B-4bit + native MTP + APC** for both the coding agent and TTRPG. Do not run Qwen3.8-27B 3-bit unattended at 16K; treat 2-bit as a stable but quality-failed experiment only.
14. Single most useful next experiment: run the same five deterministic tasks against the existing Qwen3.5 stack, then compare task-level quality and latency directly with the Qwen3.8 3-bit 3/5 result—without any memory-policy changes.

## Prior reference (not rerun)

The existing Qwen3.5-9B-4bit reference remains outside this overnight run: approximately 24–25 tok/s plain, 30 tok/s native MTP, ~98.6% APC prefix reuse, and cold/warm TTFT approximately 65 s / 1.5 s at the practical coding context.

## Artifacts

Raw logs, stage JSON, watchdog telemetry, metadata, and quality outputs live under `qwen38-overnight/`.

Archive: `qwen38-overnight-report.tar.gz` (214700 bytes).
Git verification: `verified: intended qwen38-overnight commits are scoped; no model weights or MLX/HF/APC caches are tracked; unrelated pre-existing root files and the report archive remain untracked`.
