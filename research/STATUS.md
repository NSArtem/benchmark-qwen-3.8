# Investigation Status

Last updated: 2026-08-18, Phase 0 recovery.

## Current phase

Phase 0 complete; Phase 1 cold g64 consolidation is next. No heavy workload is
running. Process inspection was attempted, but this execution sandbox denied
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

## Key measurements

- Device: Apple M3 Pro, 18 GiB MLX-reported unified memory.
- Model active memory after load in the 0.32.1 alignment runs: about 10.963
  GiB; model file/materialized bytes: 11,771,132,928 bytes.
- At 769 final prompt tokens / 768 main prefill: peak 12.459 GiB; after-first
  token active about 11.199 GiB and active+cache about 11.363 GiB in one
  repeat. The adjacent-length peak differences were immaterial.
- The known forced-wired OOM snapshot reached about 11.958 GiB active+cache in
  the direct test; the default path passed at the same prompt.

## Current best configuration

Best measured practical candidate remains g64 affine, MLX 0.32.1, direct or
no-wired execution, greedy, no speculation/MTP/APC/activation quantization.
The 16.8K result has uncomfortable 17.200 GB peak; a recommended practical
context has not yet been established by this ledger.

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

Re-read `AGENTS.md`, `TASK.md`, and this file, then build/run the Phase 1
0.32.1 direct-generate cold ladder only for missing useful points. Avoid
repeating the known 16.8K run unless a later decision genuinely requires it.

## Safety notes

- No system wired-memory setting was modified.
- No model weights, source downloads, or conversions have been started.
- Do not use the older 0.32.0 environment.
- `ps`/`pgrep` process checks are unavailable in the sandbox; do not infer that
  an external process is absent beyond the failed checks.
