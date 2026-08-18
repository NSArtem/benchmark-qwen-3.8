# Investigation Status

Last updated: 2026-08-18, Phase 4 after P4-QUANT-SIZES.

## Current phase

Phase 0 through Phase 4 are complete. Phase 5 conversion-feasibility review is
next. No heavy workload is running. Process inspection was
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
- `P2-HELPER-INFO`: `research/runtime/qwen38_runtime.py info` reported Python
  3.14.7, MLX 0.32.1, pinned mlx-lm revision, the exact local model path,
  affine 3-bit g64 metadata, and `MLX wired state untouched`.
- `P2-HELPER-SHORT`: the no-wired helper generated a coherent 37-token answer
  from a 25-token chat-templated prompt. Prompt 23.859 tok/s; decode 11.497
  tok/s; peak 12.075 GB decimal. Raw response is saved in
  `research/runtime/validation-short.txt`.
- `P2-HELPER-4K`: the corrected no-wired helper generated a coherent 64-token
  diagnosis from the 4,096-token Lantern prompt. Prompt 79.688 tok/s; decode
  11.027 tok/s; peak 15.185 GB decimal. Raw response is saved in
  `research/runtime/validation-moderate.txt`.
- `P3-WARM4K`: one persistent-cache process ran a 4,096-token initial coding
  turn plus three short follow-ups (128, 64, 64, and 64 output tokens).
  All four turns were COHERENT. Initial TTFT 51.484 s at 79.710 prompt tok/s
  and 10.943 decode tok/s. Follow-up TTFTs were 0.726, 1.103, and 0.728 s;
  follow-up decode was 11.024, 11.017, and 11.005 tok/s. After-turn
  active+cache was 11.463, 11.433, 11.447, and 11.447 GiB; cumulative peak
  was 14.142 GiB. OS snapshots showed 26% free memory and 203.75 MB swap at
  the end. Responses and token IDs are in `research/warm-cache/`.
- `P3-WARM8K`: the same persistent-cache script ran an 8,191-token initial
  coding turn plus the same three short follow-ups. All four turns were
  COHERENT. Initial TTFT 104.222 s at 78.667 prompt tok/s and 10.547 decode
  tok/s. Follow-up TTFTs were 0.816, 1.202, and 0.828 s; follow-up decode was
  10.664, 10.651, and 10.645 tok/s. After-turn active+cache was 11.714,
  11.684, 11.697, and 11.697 GiB; cumulative peak was 14.766 GiB. OS
  snapshots showed 25% free memory and 203.75 MB swap at the end. Responses
  and token IDs are in `research/warm-cache-8k/`.
- `P4-QUANT-SIZES`: header-only analysis of all three current safetensor
  shards. It found 1,847 tensors, 498 quantized modules, 5,291,008 bytes of
  non-quantized storage, and exact current storage of 11,771,132,928 bytes.
  The affine g64 packing formula reproduced the current byte count exactly.
  All 498 module input widths are divisible by 128. Estimates: affine 3-bit
  g128 = 10.180 GiB (0.783 GiB / 7.14% saved; 3.251 effective bpw),
  mixed_2_6 g64 = 9.772 GiB (1.191 GiB / 10.87% saved; 3.121 bpw),
  mixed_2_6 g128 = 8.989 GiB (1.974 GiB / 18.00% saved; 2.871 bpw), and
  affine 2-bit g128 = 7.049 GiB (3.913 GiB / 35.70% saved; 2.251 bpw).
  The current index's 26,895,993,856 parameter count is reconciled after
  excluding 4,608 saved `A_log`/`dt_bias` buffer elements. No weight arrays
  were loaded.

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
- The reusable helper reproduced the moderate 4K baseline behavior and emits
  an explicit no-wired startup line. Its CLI's initial raw-file mode was
  briefly wrong because it bypassed the chat template; that tooling bug
  produced a one-token EOS response, was corrected, and the corrected 4K
  validation passed. This is not evidence of model corruption.
- Persistent prompt caching is confirmed useful at 4K: the initial cold turn
  retains the expected ~51 s prefill, while short incremental turns complete
  in about 0.7–1.1 s to first token and decode at about 11 tok/s without
  material memory growth. On this evidence the 4K interactive classification
  is EXCELLENT.
- The 8K confirmation has the same classification: short follow-ups remain
  below 1.21 s TTFT and about 10.65 tok/s decode with no material cache growth.
  The measured practical warm context is therefore 8K; 12K remains a cold
  one-shot measurement only.
- The pinned conversion predicate supports a useful ranking: affine 3-bit g128
  has the lowest expected quality risk but modest savings; mixed_2_6 g128 is
  the larger built-in tradeoff; affine 2-bit g128 saves the most but has the
  highest expected quality risk. These rankings are hypotheses until source
  weights and correctness tests exist.
- At 769 final prompt tokens / 768 main prefill: peak 12.459 GiB; after-first
  token active about 11.199 GiB and active+cache about 11.363 GiB in one
  repeat. The adjacent-length peak differences were immaterial.
- The known forced-wired OOM snapshot reached about 11.958 GiB active+cache in
  the direct test; the default path passed at the same prompt.

## Current best configuration

Best measured practical candidate remains g64 affine, MLX 0.32.1, direct or
no-wired execution, greedy, no speculation/MTP/APC/activation quantization.
The reusable no-wired helper now provides the canonical user-facing path. The
4K and 8K persistent-cache profiles are both classified EXCELLENT. For a
conservative interactive recommendation, use 8K prompt context; 12K is a
coherent cold probe but has not been validated as a warm conversation.
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

Re-read `AGENTS.md`, `TASK.md`, and this file, then enter Phase 5. Identify the
exact source/base model from local metadata, inspect disk/cache headroom, and
review the pinned conversion path for bounded-memory feasibility. Do not
download source weights or start conversion until the source and disk gate is
explicitly recorded.

## Safety notes

- No system wired-memory setting was modified.
- No model weights, source downloads, or conversions have been started.
- Do not use the older 0.32.0 environment.
- `ps`/`pgrep` process checks are unavailable in the sandbox; do not infer that
  an external process is absent beyond the failed checks.
