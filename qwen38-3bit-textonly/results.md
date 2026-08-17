# Qwen3.8 27B 3-bit TextOnly on Apple M3 Pro / 18 GB

## Executive summary

`lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly` loads and produces coherent short answers through
the supported native `mlx-lm` runtime. The 16,727-token realistic Lantern coding-agent prompt
does not produce a usable long-context result on this 18 GB M3 Pro: the public API emitted
blank punctuation and multilingual/gibberish output, then raised a native Metal insufficient
memory error during normal stream cleanup. A direct `mlx_lm.generate` CLI cross-check reproduced
the malformed output at the same context, so this is not a benchmark-harness-only artifact.

The 16K run was allowed to progress through all prefill chunks and 256 generated tokens. Swap
and page-outs were recorded, but they were not used as an abort condition and there was no clear
continuous paging storm with loss of progress. The failure is the malformed generation plus
Metal memory failure, not merely the existence of swap.

Per the requested stopping rule, ~24K and ~32K were not run after the 16K gate failed. MTP and
prefix-cache benchmarks were also not run; their results are deliberately `null`, not inferred.

## Stage 1 sanity gate

The public `mlx_lm.load` + `mlx_lm.stream_generate` path answered the required OSR-dungeon prompt
coherently for 256 tokens. It reached the length cap rather than EOS and measured about 11.0
decode tok/s. The standard interactive `mlx_lm.chat` CLI was also run with the same short prompt
and produced coherent OSR guidance. The saved API-path output is
[`harness/short-output.txt`](harness/short-output.txt).

## Baseline context scaling

`prefill tok/s` below is measured from native `mlx-lm` prompt-progress callbacks through the
last prompt chunk, excluding the first-token logit step. TTFT includes prompt processing and the
first generated token. The API run is the primary saved measurement; the direct CLI cross-check
reported 174.906 prompt tok/s and 9.191 decode tok/s at the same prompt.

| Context | Prompt tokens | Coherent | Prefill tok/s | TTFT | Decode tok/s | Peak memory | Swap delta | Verdict |
| ------- | ------------: | -------: | ------------: | ---: | -----------: | ----------: | ---------: | ------- |
| ~16K | 16,727 | no | 178.4 | 96.38 s | 6.23 | 17.20 GB | +662.5 MB peak; -543.2 MB final | malformed output; native Metal OOM during cleanup |
| ~24K | — | — | — | — | — | — | — | not run after 16K gate |
| ~32K | — | — | — | — | — | — | — | not run after 16K gate |

The 16K run loaded the model in 4.40 s, emitted 256 tokens, and had a 141.96 s process wall
time. Its swap snapshots were 1,486.56 MB before, 2,149.06 MB peak, and 943.38 MB after. The
saved result is [`baseline/16k.json`](baseline/16k.json); the malformed text is preserved in
[`baseline/16k-output.txt`](baseline/16k-output.txt), with telemetry in
[`baseline/16k-memory.jsonl`](baseline/16k-memory.jsonl).

The direct CLI cross-check exited normally but emitted repeated `因`/gibberish sequences and
reported 17.2 GB peak memory. This confirms that the public supported CLI and the measurement
API agree on the long-context quality failure.

## Memory and paging interpretation

- `iogpu.wired_limit_mb` was `0` at startup and remained unchanged.
- No `sudo` was used and swap configuration was not modified.
- Page-outs and swap-ins occurred during the 16K run, but samples were intermittent rather than
  a continuous adjacent heavy-paging storm.
- There was no observed prolonged loss of prefill progress: all 16,727 prompt tokens advanced
  through the normal `mlx-lm` chunks.
- Therefore `real_thrashing` is recorded as `false`; paging was observed, with no clear paging-
  driven collapse. The native Metal OOM and broken output remain genuine failure evidence.

## MTP

| Context | Baseline tok/s | MTP tok/s | Speedup | Acceptance | Extra memory | Verdict |
| ------- | -------------: | --------: | ------: | ---------: | -----------: | ------- |
| — | — | — | — | — | — | not measured after target-only baseline gate |

Installed `mlx-lm` 0.31.3 exposes generic `--draft-model` and `--num-draft-tokens` speculative
decoding. The same installed package has a `qwen3_5.py` target implementation but no
`qwen3_5_mtp.py` loader module; the cached companion metadata is
`lukaskremla/Qwen3.8-27B-MTP-3bit-MLX`. No MTP generation was attempted after the target-only
failure, and no acceptance or speedup is claimed.

## Prefix caching

No two-turn cache test was run. Static inspection found the supported current mechanisms:

- `mlx_lm.cache_prompt` / `mlx_lm.generate --prompt-cache-file` for a saved prompt KV cache;
- `mlx_lm.server --prompt-cache-size` and `--prompt-cache-bytes` for persistent server-side
  prefix reuse, with response metadata exposing cached-token counts.

Because no viable long-context target-only configuration was established, there is no cold/warm
TTFT measurement or cache-hit evidence to report.

## Recommended configuration

For this machine today, use the target-only normal chat path that is already known to work:

```bash
uv run mlx_lm.chat \
  --model lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly \
  -m 2048
```

Do not claim 16K as a practical coding-agent or TTRPG-GM context: it was processed, but the
answer was not coherent and the API run hit Metal insufficient memory. The measured short sanity
case is evidence that the model itself and native text-only runtime work; it does not establish
a long-context sweet spot. The practical long-context sweet spot therefore remains
**undetermined below 16K**, with the user’s previously verified 2,048-token chat configuration
the only concrete recommended setting from this experiment.

This is not yet a credible long-running local coding-agent or long-context TTRPG-GM model on an
18 GB M3 Pro. It is a credible short-context local model, subject to the prompt-size limit just
described.

## Direct answers

1. Yes for short native `mlx-lm` use; no for a reliable 16K realistic context.
2. Short generation is coherent; 16K native generation is not coherent.
3. Measured 16K prefill was 178.4 tok/s (CLI cross-check 174.9); 24K/32K were not run.
4. Measured 16K decode was 6.23 tok/s in the instrumented API run (CLI cross-check 9.19); 24K/32K were not run.
5. 16,727 prompt tokens were processed and 256 tokens emitted, but not as a successful/coherent completion; no usable maximum long context was established.
6. No long-context sweet spot was established; use the known 2,048-token configuration pending a lower-bound test.
7. 16K: 1,486.56 MB before, 2,149.06 MB peak, 943.38 MB after. 24K/32K: not measured.
8. Paging did not show a clear throughput-collapse pattern by itself; malformed output and native Metal OOM did.
9. No genuine sustained thrashing was identified.
10. Native MTP was not run; this installed `mlx-lm` does not expose a clean `qwen3_5_mtp` model loader.
11. No MTP speedup measured.
12. No MTP memory cost measured.
13. Prefix caching was not run; no cache hit is claimed.
14. No warm-TTFT improvement measured.
15. Run target-only native `mlx_lm.chat` with `-m 2048`; do not enable MTP/cache based on this failed baseline.
16. Not yet for long-running coding-agent use; short-context use is credible.
17. Not yet for a long-running context-heavy TTRPG-GM session; short sessions are credible.
18. Highest-value next experiment: one supported target-only **8K** run with the same realistic prompt/settings to locate the first coherent boundary below 16K, before revisiting MTP or caching.

## Artifacts

- Machine-readable summary: [`results.json`](results.json)
- Environment: [`environment.md`](environment.md)
- Commands: [`commands.txt`](commands.txt)
- Prompt/token manifest: [`prompt-manifest.json`](prompt-manifest.json)
- Harness source: [`harness/benchmark.py`](harness/benchmark.py)
