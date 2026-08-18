# Investigation Status

Last updated: 2026-08-18, Phase 1 after P1-G64-12288.

## Current phase

Phase 0 complete; Phase 1 cold g64 consolidation is complete. Phase 2 no-wired
runtime helper is next. No heavy workload is running. Process inspection was
attempted, but this execution sandbox denied
both `ps` and `pgrep` (`operation not permitted` / `sysmond service not found`);
heavy launches will therefore remain strictly serialized and short enough to
observe through their command sessions.

## Completed experiments

- `P0-ALIGNMENT-REUSE`: reused `qwen38-mlx0321-alignment/`. MLX 0.32.1,
  mlx-lm revision `254d153fdeb6f150edd4fc5a54f9828638481fa8`, g64 affine,
  default wired state, direct `generate_step`, 18/18 coherent at final prompt
  lengths 512, 513, 640, 641, 768, and 769. No OOM; all three repeats at
  each length had identical 128-token sequences.
- `P0-OOM-REUSE`: reused `qwen38-mlx0321-oom/`. The 768-token direct default
  path passed; forcing the recommended 12.000 GiB wired limit OOMed during
  `mx.eval([c.state for c in prompt_cache])` before the first token.
- `P0-16K-POLICY`: existing `AGENTS.md` evidence records a manual 16,767-token
  real coding prompt on 0.32.1 with the wired-limit wrapper made a no-op:
  coherent, 256 generated tokens, 63.348 prompt tok/s, 4.118 decode tok/s,
  17.200 GB MLX peak, no OOM. Raw run files are not in the checkout.
- `P1-PROMPTS`: prepared `research/context-ladder/prompt-manifest.json` and
  five raw/tokenized prompt pairs using the same compact Lantern source and
  chat template. Actual final counts are 1,023, 2,048, 4,096, 8,191, and
  12,288; 1,023 and 8,191 are the closest natural token boundaries to the
  requested 1,024 and 8,192 targets.
- `P1-G64-1024`: fresh-process direct `generate_step`, actual 1,023 prompt
  tokens, 128 greedy output. COHERENT; TTFT 13.352 s; prompt 76.615 tok/s;
  decode 11.112 tok/s; peak 12.860 GiB; after-first-token active+cache
  11.389 GiB. End snapshot reported 28% system free memory and 41.69 MB swap;
  no exception or safety stop.
- `P1-G64-2048`: fresh-process direct `generate_step`, 2,048 prompt tokens,
  128 greedy output. COHERENT; TTFT 25.671 s; prompt 79.780 tok/s; decode
  11.002 tok/s; peak 13.824 GiB; after-first-token active+cache 11.399 GiB.
  End snapshot reported 26% system free memory and 41.69 MB swap; no exception
  or safety stop.
- `P1-G64-4096`: fresh-process direct `generate_step`, 4,096 prompt tokens,
  128 greedy output. COHERENT; TTFT 51.383 s; prompt 79.715 tok/s; decode
  10.858 tok/s; peak 14.142 GiB; after-first-token active+cache 11.449 GiB.
  End snapshot reported 25% system free memory and 41.69 MB swap; no exception
  or safety stop.
- `P1-G64-8192`: fresh-process direct `generate_step`, actual 8,191 prompt
  tokens, 128 greedy output. COHERENT; TTFT 104.330 s; prompt 78.511 tok/s;
  decode 10.415 tok/s; peak 14.766 GiB; after-first-token active+cache
  11.634 GiB. End snapshot reported 24% system free memory and 41.69 MB swap;
  no exception or safety stop.
- `P1-G64-12288`: fresh-process direct `generate_step`, 12,288 prompt tokens,
  128 greedy output. COHERENT; TTFT 160.908 s; prompt 76.366 tok/s; decode
  7.734 tok/s; peak 15.392 GiB; after-first-token active+cache 12.047 GiB.
  End snapshot reported 24% system free memory and 235.75 MB swap; no
  exception or safety stop. Raw result and output are saved under
  `research/context-ladder/`.

## Key measurements

- Device: Apple M3 Pro, 18 GiB MLX-reported unified memory.
- Model active memory after load in the 0.32.1 alignment runs: about 10.963
  GiB; model file/materialized bytes: 11,771,132,928 bytes.
- Cold g64 peak memory rose from 12.860 GiB at 1,023 prompt tokens to 13.824,
  14.142, 14.766, and 15.392 GiB at 2,048, 4,096, 8,191, and 12,288 tokens.
  After-first-token active+cache rose from 11.389 to 12.047 GiB over the same
  range. This is a measured curve, not a claim of a safe extrapolated limit.
- Cold TTFT was 13.352, 25.671, 51.383, 104.330, and 160.908 s at those
  same prompt sizes. Prompt throughput stayed 76.366–79.780 tok/s; decode
  throughput fell from 11.112 to 7.734 tok/s.
- At 769 final prompt tokens / 768 main prefill: peak 12.459 GiB; after-first
  token active about 11.199 GiB and active+cache about 11.363 GiB in one
  repeat. The adjacent-length peak differences were immaterial.
- The known forced-wired OOM snapshot reached about 11.958 GiB active+cache in
  the direct test; the default path passed at the same prompt.

## Current best configuration

Best measured practical candidate remains g64 affine, MLX 0.32.1, direct or
no-wired execution, greedy, no speculation/MTP/APC/activation quantization.
The largest fresh-process cold probe is 12,288 coherent at 15.392 GiB peak;
the separate 16,767-token policy-summary run was coherent at 17.200 GB but is
not a comfortable recommendation.

## Failed/rejected configurations

- Forced pinned `mlx_lm` recommended wired limit: rejected; reproducible Metal
  OOM at 768-token prefill.
- MLX 0.32.0 long-context artifacts: historical/rejected by project policy;
  no new 0.32.0 runs.
- Historical 0.32.0 3-bit rescue, chunking, and KV8 results: not canonical
  evidence for this task and not used to claim 0.32.1 behavior.

## Current hypotheses

- The 0.32.1 stock OOM is a wired-residency/accounting interaction: the
  forced recommended limit changes the cache-state evaluation allocation path,
  while default wired state can pass.
- Persistent prompt caching may reduce incremental prefill latency, but its
  usefulness and memory growth are unmeasured here.
- Larger-group quantization may save model memory, but quality/runtime support
  and a valid high-precision source path are unresolved.

## Next action

Re-read the governing files, then enter Phase 2: build and short-prompt
validate a small reusable no-wired runtime helper. Preserve the pinned
environment and leave MLX wired state untouched. Do not run another large g64
context probe solely to fill the gap between 12K and the existing 16.8K
policy-summary run.

## Safety notes

- No system wired-memory setting was modified.
- No model weights, source downloads, or conversions have been started.
- Do not use the older 0.32.0 environment.
- `ps`/`pgrep` process checks are unavailable in the sandbox; do not infer that
  an external process is absent beyond the failed checks.
