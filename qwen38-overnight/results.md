# Qwen3.8-27B overnight exploration

Generated: 2026-08-17 03:00:50 +04

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
| Qwen3.8-27B uniform 3-bit text-only | plain | no | 138.8 | 10.10 | — | 17.28 GB | 18.0% | meaningful_sustained_paging | — | — | pending | completed |

`Peak MLX` is MLX’s reported allocation peak, not total system RSS. `Min free` and paging are system-level watchdog observations; headroom claims use both rather than treating swap allocation alone as a failure.

## Metadata and attempted stages

- `lukaskremla/Qwen3.8-27B-2bit-MLX-TextOnly`: status `ok`, revision `33b90b60fd7ba16b668854e049bd65e22d6afddf`, reported weight download `8.41 GB`, text-only inference `{'basis': 'repository_name', 'text_only': True}`.
- `lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly`: status `ok`, revision `c98bba5926f51fec1c8d8737e577221673f524d7`, reported weight download `11.77 GB`, text-only inference `{'basis': 'repository_name', 'text_only': True}`.
- `rapid-mlx/Qwen3.8-27B-mixed-3.5bpw-MLX`: status `ok`, revision `d8b68e21eab505332d4a456dacec6d4125c96d73`, reported weight download `13.91 GB`, text-only inference `{'basis': 'vision_or_processor_metadata_present', 'processor_files': ['preprocessor_config.json', 'processor_config.json', 'video_preprocessor_config.json'], 'text_only': False, 'vision_keys': ['image_token_id', 'video_token_id', 'vision_config']}`.
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

- Not run yet; performance gates must identify viable candidates first.

## Direct answers

1–14. No practical winner exists yet; the report will be regenerated after the staged gates and quality smoke complete.

## Prior reference (not rerun)

The existing Qwen3.5-9B-4bit reference remains outside this overnight run: approximately 24–25 tok/s plain, 30 tok/s native MTP, ~98.6% APC prefix reuse, and cold/warm TTFT approximately 65 s / 1.5 s at the practical coding context.

## Artifacts

Raw logs, stage JSON, watchdog telemetry, metadata, and quality outputs live under `qwen38-overnight/`. The final archive path and Git verification are added at completion.
