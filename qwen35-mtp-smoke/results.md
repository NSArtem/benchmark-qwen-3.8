# Qwen3.5-9B native MTP smoke-test results

Date: 2026-08-16

This is the deliberately small A/B experiment requested: one Qwen3.5-9B-4bit
baseline and one native Qwen3.5 MTP run through the same mlx-vlm generation path
on the 18 GB Apple Silicon Mac. DFlash, DSpark, Qwen3, Qwen3.8, alternate
quantizations, sweeps, and block-size tuning were not tested.

## Environment

- Machine: MacBook Pro, model identifier `Mac15,6`
- Apple Silicon: Apple M3 Pro, 11 cores
- Unified memory: 18 GB (`hw.memsize=19327352832` bytes)
- Architecture: `arm64`
- macOS: 15.7.8, build `24G824`
- `uv`: 0.12.5 (Homebrew, aarch64)
- Python through uv: 3.14.7
- MLX: 0.32.0
- mlx-vlm: 0.6.13
- mlx-lm: 0.31.3
- transformers: 5.15.0
- huggingface-hub: 1.27.0

The repository already contains `mlx-dspark` 0.12.0 as an older project
dependency; it was not used by this experiment. `mlx-vlm>=0.6.13` was added as
an explicit uv dependency.

## Models and workload

- Target: `mlx-community/Qwen3.5-9B-4bit`
- Native MTP drafter: `mlx-community/Qwen3.5-9B-MTP-4bit`
- The drafter was passed explicitly as native `draft_kind="mtp"`; it was never
  loaded or benchmarked as a standalone language model.
- Prompt: `coding-prompt.txt`, a coherent fictional Lantern repository/debugging
  task reused from the existing repository prompt.
- Raw prompt: 61,236 UTF-8 characters.
- Chat-formatted prompt: 61,304 characters.
- Actual input tokens after Qwen3.5 chat formatting/tokenization: **16,727**.
  The earlier repository note of 15,409 tokens used a different Qwen3
  tokenizer/template. The resulting Qwen3.5 count is slightly above the nominal
  16K target but remains the intended practical long-context workload; it was
  not reduced to a short artificial context.
- Both runs used the same formatted-prompt SHA-256:
  `f42f2dc8ff460f33c456cc28a8150ed2e54553d6c3e155d1594a7e4cfa5d9e9b`.

## Decoding configuration

- Batch size: 1
- Maximum generated tokens: 256
- Temperature: 0.0, greedy decoding
- Thinking: disabled in the chat template and generation call
- Prefill step size: 2,048 (identical in both modes)
- Warm-up: one same-prompt 8-token generation per process, excluded from the
  reported measurement
- Measurement: one run per mode after warm-up; no repeated-trial averaging
- Generation path: the harness consumes mlx-vlm `stream_generate`, the same
  underlying path used by the mlx-vlm CLI. It records mlx-vlm's own prompt and
  generation timing fields.
- MTP block-size override: none. The downloaded checkpoint's actual
  `config.json` reports `block_size: 3`; with the installed mlx-vlm semantics
  this produced two drafted tokens per round. This is the checkpoint/runtime
  default, not a tuning choice. It differs from the expected “native block size
  2” wording in the test request, so the discrepancy is recorded here rather
  than silently changing it.

## Results

| Mode       | Input tokens | Prefill tok/s | Decode tok/s | Wall time | Peak MLX memory | MTP acceptance | Verdict |
| ---------- | -----------: | ------------: | -----------: | --------: | --------------: | -------------: | ------- |
| Baseline   |       16,727 |        261.37 |        24.16 |  74.64 s |       9.915 GB |              — | Valid mlx-vlm baseline; close to the independent ~25 tok/s sanity reference |
| Native MTP |       16,727 |        261.40 |        29.85 |  72.60 s |      10.052 GB | 54.5%; 2.09 tokens/round | Modest but potentially useful decode improvement |

### Timing details

The mlx-vlm prompt rate implies approximately 63.997 s of prompt processing for
the baseline and 63.991 s for MTP. Observed time to the first generated token was
64.024 s and 64.012 s respectively. The measured post-first-token decode tails
were 10.617 s baseline and 8.590 s MTP. The primary decode rates above are
mlx-vlm's final `generation_tps` values, 24.1551 and 29.8489 tok/s.

The baseline is consistent with the prior independent `mlx_lm.benchmark` sanity
reference of roughly 25 tok/s at this context. The small difference is expected
from the different generation path and exact 16,727-token prompt.

### Native MTP statistics

The measured MTP run exposed 122 speculative rounds:

- Drafted tokens: 244 (2.00 per round)
- Accepted drafted tokens: 133
- Draft acceptance rate: 133 / 244 = **54.5%**
- Mean accepted drafted tokens per round: 1.09
- Mean accepted tokens per round including the target verifier bonus: **2.09**
- Target verification rounds: 122 speculative rounds were recorded; a separate
  target-forward-pass counter was not exposed by mlx-vlm.
- mlx-vlm summary: `2.09 accepted tokens/round (1.09 accepted drafts/round,
  54.5% of drafted, avg draft 2.00) over 122 rounds`.

The per-round `accept_lens` and `draft_lens` arrays are retained in `mtp.json`.
The lifetime counters in that file include the warm-up and therefore are larger
than the measured-only round totals.

### Memory observations

MLX peak memory increased from 9.915 GB to 10.052 GB, approximately +0.137 GB
(+137 MB, about 1.4%). This was not a material increase for this machine.

The measurements were separate fresh processes, so the free-memory percentages
are only directional: baseline was 42% before and 40% after; MTP was 39% before
and 39% after. Swap was 147.06 MB before and after both measured runs, with zero
swap-outs during each run. Page-outs increased by 99 for baseline and 140 for
MTP, without accompanying swap growth or signs of sustained paging. The MTP run
was therefore not memory-pressure limited.

### Speedup

`29.8489007253 / 24.1550814739 = 1.235719×`, or **+23.57%** decode throughput
(an absolute gain of **5.694 tok/s**). By the requested classification this is
`1.15–1.4×`: modest but potentially useful. Total wall time improved only about
2.8% because the approximately 64-second prompt prefill dominates a 256-token
completion.

### Correctness sanity check

Both runs produced 256 generated token IDs. The first 174 zero-based token
positions were identical. They first diverged at zero-based index **174** (the
175th generated token): baseline token `466` decodes as `" or"`, while MTP token
`314` decodes as `" of"`. There were 82 differing positions in the 256-token
outputs. The outputs then remain coherent coding-agent answers; there is no
first-token divergence, truncation, or obvious corruption.

This is sufficient for a smoke-test performance result, but it is not exact
greedy output identity. The later, local divergence is plausibly a floating-point
or tie-breaking difference in the native speculative verification path rather
than an obvious catastrophic correctness failure. Exact token identity should
be checked again before treating MTP as a drop-in deterministic replacement.

## Direct answers

1. **Baseline at ~16K:** 24.16 decode tok/s through mlx-vlm at 16,727 input tokens.
2. **Native-MTP speed:** 29.85 decode tok/s.
3. **Measured multiplier:** 1.2357×, or +23.57%.
4. **Acceptance behavior:** 54.5% of drafted tokens accepted; 1.09 accepted
   drafts and 2.09 total accepted tokens per speculative round.
5. **Memory pressure:** no material increase in practical pressure; MLX peak was
   +137 MB and neither run added swap-outs or swap usage.
6. **Correctness:** no obvious corruption and divergence begins only at token
   175, but outputs are not byte/token-identical, so this passes a smoke-test
   sanity check rather than a strict equivalence check.
7. **Worth using:** conditionally yes for coding-agent workloads if a roughly
   24% decode gain is worth accepting non-identical greedy text; it is useful but
   not transformative, and total long-prompt latency remains prefill-dominated.
8. **Later DFlash comparison:** yes, this is interesting enough to justify a
   separate DFlash experiment, especially because native MTP added little memory
   pressure. DFlash was not run here.

## Limitations and saved artifacts

- Only one measured run per mode was performed, so run-to-run variance is not
  estimated.
- The nominal prompt target was exceeded slightly by the Qwen3.5 tokenizer
  (16,727 tokens). Both A/B runs used exactly the same prompt and token count.
- `prompt_tps` is mlx-vlm's reported field; its prompt timing includes the work
  up to the first generated token according to this release's implementation.
- The MTP checkpoint metadata/runtime block-size discrepancy is documented above;
  no override or tuning was used.

Important files in this directory:

- `results.json`: machine-readable summary
- `baseline.json`, `mtp.json`: per-run metrics and native acceptance arrays
- `baseline.log`, `mtp.log`: complete stdout/stderr from each run
- `baseline-output.txt`, `mtp-output.txt`: exact generated texts
- `baseline-tokens.json`, `mtp-tokens.json`: exact generated token IDs
- `coding-prompt.txt`: exact benchmark prompt
- `benchmark.py`: reproducible uv/mlx-vlm harness
- `commands.txt`: important reproduction commands

Git commits:

- Preparation: `66b00c4` (`benchmark: add Qwen3.5 native MTP smoke test`)
- Results commit: `718ecbb` (`benchmark: record Qwen3.5 native MTP results`)
