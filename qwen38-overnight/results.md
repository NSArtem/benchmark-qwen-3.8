# Qwen3.8-27B overnight exploration

Generated: 2026-08-17 02:21:53 +04

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
| — | — | pending | — | — | — | — | — | — | — | — | pending | No practical run has completed yet |

`Peak MLX` is MLX’s reported allocation peak, not total system RSS. `Min free` and paging are system-level watchdog observations; headroom claims use both rather than treating swap allocation alone as a failure.

## Metadata and attempted stages

- Metadata collection pending.

## Quality smoke

- Not run yet; performance gates must identify viable candidates first.

## Direct answers

1–14. No practical winner exists yet; the report will be regenerated after the staged gates and quality smoke complete.

## Prior reference (not rerun)

The existing Qwen3.5-9B-4bit reference remains outside this overnight run: approximately 24–25 tok/s plain, 30 tok/s native MTP, ~98.6% APC prefix reuse, and cold/warm TTFT approximately 65 s / 1.5 s at the practical coding context.

## Artifacts

Raw logs, stage JSON, watchdog telemetry, metadata, and quality outputs live under `qwen38-overnight/`. The final archive path and Git verification are added at completion.
