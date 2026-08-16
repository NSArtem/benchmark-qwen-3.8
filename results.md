# mlx-dspark smoke-test results

Date: 2026-08-16

## Environment

- Apple Silicon: Apple M3 Pro, arm64
- Unified memory: 18 GB
- macOS: 15.7.8
- `uv`: 0.12.5
- Python through `uv`: 3.14.7
- MLX: 0.32.0
- mlx-dspark: 0.12.0
- `mlx-dspark doctor`: passed; Metal device detected

## Method

- Workload: fictional coding-agent repository/debugging prompt
- Prompt size: 15,409 tokens with Qwen3 chat formatting
- Requested context target: approximately 32K; the generate API exposed no explicit context-window cap, and Qwen3 reported a 40,960-token limit
- Output: deterministic greedy decoding, 256 tokens
- One same-prompt 8-token warm-up before each measured attempt
- No parameter sweeps or repeated trials
- Automatic DSpark drafter and cap selection used
- Optional TTRPG workload skipped because the preceding runs were not trouble-free

## Measurements

| Model | Workload | Input tokens | Baseline decode tok/s | DSpark tok/s | Speedup | Mean accept | Peak memory | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Qwen3-8B-8bit | Coding | 15,409 | 9.0 | — | — | — | MLX peak 11.3 GB; no measured swap growth | Baseline completed; DSpark stopped by swap safety limit |
| Qwen3.8-27B-4bit | Coding | 15,409 | — | — | — | — | System swap reached 6.16 GB total during load/calibration | Stopped immediately; no generation run |

### Qwen3-8B baseline

- Prefill: 59.7 seconds, 258 tok/s
- Time to first token: 65.8 seconds
- Decode: 28.3 seconds for 256 tokens, 9.0 tok/s
- Total generation wall time: 88.0 seconds
- Peak MLX memory: 11.3 GB

DSpark automatically selected cap 7 and hybrid lookup drafts. It repeatedly hit sustained swap growth before the 16K prompt generation could be measured. No valid DSpark decode speed or speedup was recorded.

### Qwen3.8-27B feasibility

The target and automatic drafter began loading/calibration, but system swap grew from roughly 1.8 GB to 6.16 GB total on the 18 GB Mac. The process was terminated immediately. No baseline or DSpark generation was attempted.

## Verdict

1. mlx-dspark works at the installation, Metal, doctor, and calibration levels.
2. Qwen3-8B-8bit baseline achieved 9.0 decode tok/s at this context.
3. DSpark speedup was not measurable because of memory pressure.
4. Qwen3.8-27B-4bit did not remain usable with its automatic drafter on this Mac.
5. Qwen3.8 decode speed was not measured.
6. This smoke test does not establish a DSpark performance benefit; a later controlled run may be worthwhile after resolving memory pressure.
7. For local use today, choose Qwen3-8B-8bit baseline for both a coding agent and a TTRPG gamemaster. The 27B configuration and DSpark path were not usable under this test condition.
