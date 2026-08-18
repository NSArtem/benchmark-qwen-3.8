# Investigation Status

Last updated: 2026-08-18, Phase 12 final report drafted.

## Current phase

Phase 0 through Phase 10 are complete: all three Phase 7 g128 correctness gates
passed, the Phase 8 8K/12K/~16K probes passed, and the Phase 9 original-g64
KV8 plus g128+KV8 probes are complete. The bounded text-only affine 3-bit
g128 conversion succeeded and passed header/index verification. KV8 is
rejected as a practical lever. Phase 10's mixed_2_6 predicate self-test and
source-backed conversion passed; the first load attempt found and preserved a
converter metadata-path bug before generation, and the corrected rebuild now
passes config and header/index integrity. The corrected candidate loaded, but
its first 256-token short output was manually classified BROKEN for
pathological repetition; the one fresh repeat reproduced the identical token
hash and loop. The candidate is rejected and no 768/4K or long-context gates
will run. No MLX inference workload is running. Process inspection was
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
- `P5-CONVERT-FEAS`: exact baseline model-card provenance is
  `Qwen/Qwen3.8-27B`; no suitable high-precision source is local. Read-only
  source metadata reports 18 safetensor shards totaling 55,563,006,776 bytes.
  Disk passes with 246 GiB free and a 106G HF cache. The pinned whole-model
  converter is rejected as unbounded for an 18 GB conversion. An official
  upstream chunked-conversion commit (`a0803664761e2c9c52ecbcb2bb012aff3e8eef06`)
  was inspected; it requires a local adaptation for the source’s
  `model.language_model.layers` paths and for stripping 333 vision plus MTP
  tensors. Decision: `STREAMING_CONVERSION_REQUIRED`. Gate details are in
  `research/conversion/phase5-gate.md`.
- `P6-CONVERTER-SELFTEST`: the local bounded converter found exactly 498
  quantizable text modules, planned the non-layer chunk plus 64 transformer
  layer chunks, verified source-to-MLX key mapping, preserved the pinned
  global norm shift on a full-attention chunk (`1.0` to `2.0`), and made zero
  wired-policy calls. No source arrays were loaded. Tool commit: `c12bace`.
- `P6-SOURCE-DOWNLOAD`: downloaded the exact `Qwen/Qwen3.8-27B` revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` into the HF cache. Verification
  found all 18 shards, 1,199 indexed tensors, 333 vision tensors, 15 MTP
  tensors, 851 retained text tensors, and 55,563,006,776 bytes on disk. The
  cache is 52G and 194 GiB remains free. No output model or inference process
  was started.
- `P6-G128-CONVERT`: one fresh-process bounded conversion processed 65 chunks
  (non-layer plus 64 transformer layers) in 24.920 s. It produced affine
  3-bit g128 output with 1,847 tensors across 19 shards. Tensor payload is
  exactly 10,930,715,648 bytes (10.180022 GiB), with 10,930,954,231 bytes on
  disk including safetensor headers. The index parameter count matches the
  baseline. Header-only verification found no missing/corrupt shard map.
  Maximum converter MLX peak was 3.330 GiB; maximum active+cache was 0.508
  GiB; maximum process RSS was 3.436 GiB. The completion memory-pressure
  snapshot reported 92% system-wide free memory. Output remains untracked at
  `/private/tmp/qwen38-g128-mlx`; no model weights are committed.
- `P7-G128-SHORT`: the converted source-backed affine 3-bit g128 model loaded
  at 10.180602 GiB active MLX memory and produced 256 greedy tokens from the
  33-token short control. Output is COHERENT; TTFT 1.066 s; prompt 30.972
  tok/s; decode 11.947 tok/s; peak 10.505 GiB; after-completion active+cache
  10.418 GiB. No exception or safety stop. Raw output, token IDs, and
  telemetry are under `research/g128-correctness/`.
- `P7-G128-768`: the same candidate produced a coherent 256-token diagnosis
  from the exact 768-token alignment prompt. TTFT 10.122 s; prompt 75.876
  tok/s; decode 11.882 tok/s; peak 11.678 GiB; after-completion active+cache
  10.444 GiB. The model materialized at 10.180602 GiB; no exception or safety
  stop. Raw output, token IDs, and telemetry are saved beside the short case.
- `P7-G128-4096`: the candidate produced a coherent 256-token diagnosis from
  the 4,096-token realistic Lantern coding prompt. TTFT 51.272 s; prompt
  79.888 tok/s; decode 11.614 tok/s; peak 13.360 GiB; after-completion
  active+cache 10.685 GiB. The model materialized at 10.180602 GiB; no
  exception or safety stop. Raw output, token IDs, and telemetry are saved
  beside the other g128 gates.
- `P7-HARNESS-FIXES`: the first g128 launch stopped before useful generation
  because the cold harness imported the runtime from the wrong script path;
  the next attempt exposed a tuple-vs-`GenerationResponse` collector mismatch.
  Both were corrected without changing model/runtime behavior. Neither is a
  model-quality or memory result.
- `P8-G128-8192`: a fresh-process 8,191-token realistic coding prompt and 128
  greedy output tokens were COHERENT. TTFT 103.864 s; prompt 78.863 tok/s;
  decode 11.204 tok/s; peak 13.983 GiB; after-completion active+cache
  11.392 GiB. System-wide free memory was 30% after the run; swap used was
  139.75 MB; no exception or safety stop.
- `P8-G128-12288`: a fresh-process 12,288-token realistic coding prompt and
  128 greedy output tokens were COHERENT. TTFT 158.238 s; prompt 77.655
  tok/s; decode 10.830 tok/s; peak 14.609 GiB; after-completion active+cache
  11.227 GiB. System-wide free memory was 28% after the run; swap used was
  139.75 MB; no exception or safety stop.
- `P8-G128-16767`: the existing real Lantern coding prompt produced a
  coherent 128-token response at an actual 16,727 prompt tokens (the runner's
  16,767 target label is retained for provenance). TTFT 219.611 s; prompt
  76.167 tok/s; decode 10.641 tok/s; peak 15.236 GiB; after-completion
  active+cache 11.397 GiB. System-wide free memory was 27% after the run and
  swap used was 326.19 MB. No exception or safety stop.
- `P9-G64-KV8-4096`: original g64 with pinned KV8/group64/start0 produced a
  coherent 128-token response at 4,096 prompt tokens. TTFT 51.579 s; prompt
  79.413 tok/s; decode 10.575 tok/s; peak 14.029 GiB; after-completion
  active+cache 11.316 GiB. Relative to `P1-G64-4096`, peak was only 0.113 GiB
  lower and decode was about 2.6% slower. System-wide free memory was 28%
  after the run; swap used was 302.19 MB; no exception or safety stop.
- `P9-G64-KV8-8192`: original g64 with the same pinned KV8 settings produced
  coherent 128-token output at 8,191 prompt tokens. TTFT 104.621 s; prompt
  78.292 tok/s; decode 9.862 tok/s; peak 14.535 GiB; after-completion
  active+cache 11.413 GiB. Relative to `P1-G64-8192`, peak was 0.231 GiB
  lower but decode was about 5.3% slower; TTFT was effectively unchanged.
  System-wide free memory was 27% after the run; swap used was 294.19 MB; no
  exception or safety stop.
- `P9-G128-KV8-8192`: the source-backed g128 plus pinned KV8/group64/start0
  combination produced coherent 128-token output at 8,191 prompt tokens.
  TTFT 104.235 s; prompt 78.582 tok/s; decode 10.417 tok/s; peak 13.752 GiB;
  after-completion active+cache 10.909 GiB. Relative to `P8-G128-8192`, peak
  was 0.231 GiB lower but decode was about 7.0% slower; TTFT was effectively
  unchanged. System-wide free memory was 31% after the run; swap used was
  294.19 MB; no exception or safety stop.
- `P10-MIXED-SELFTEST`: the bounded converter's pinned-recipe adaptation split
  exactly 498 text quantization modules into 457 two-bit and 41 six-bit
  modules at group128, matching the Phase 4 mixed_2_6 estimate. The static
  chunk plan, Qwen sanitization, and zero wired-policy calls also passed. No
  source arrays were loaded and no model was converted at this gate.
- `P10-MIXED-CONVERT`: one bounded source-backed conversion from the exact
  `Qwen/Qwen3.8-27B` revision completed in 23.527 s without OOM. It produced
  1,847 tensors in 17 shards with 9,651,780,608 bytes of tensor payload
  (8.988921 GiB), 457 two-bit and 41 six-bit modules, and matching index/header
  key sets. Converter peak was 3.626 GiB and max RSS was 3.398 GiB; the final
  OS snapshot reported 92% free memory. The candidate remains untracked at
  `/private/tmp/qwen38-mixed26-g128-mlx`.
- `P10-MIXED-LOAD-FAIL`: the first mixed artifact was rejected before any
  generation because its per-module config keys retained `.weight`, while the
  pinned mlx-lm loader's class predicate expects module paths. With the base
  2-bit setting, `mlp.down_proj` was initialized for the wrong packed shape and
  load failed with `Expected shape (5120, 1088) but received shape (5120, 3264)`.
  No model-quality or memory conclusion was drawn; the failure is preserved in
  `research/mixed26-correctness/result-short.json`.
- `P10-MIXED-CONVERT-FIXED`: after the metadata-path fix, one fresh bounded
  rebuild completed in 25.656 s with the same 9,651,780,608-byte payload,
  17 shards, 1,847 tensors, 457 two-bit and 41 six-bit modules, and 3.626 GiB
  converter peak. The corrected config has zero `.weight` suffix entries;
  all 1,847 index keys match safetensor headers. The candidate remains
  untracked at `/private/tmp/qwen38-mixed26-g128-mlx-fixed`.
- `P10-MIXED-SHORT-FIXED`: the corrected candidate loaded at 8.9893 GiB
  active memory and completed 256 tokens without OOM. The harness reported
  `COHERENT`, but manual inspection found the answer repeated the same
  sentence/pathological substring throughout, so the authoritative project
  classification is `BROKEN`. TTFT was 1.026 s, prompt throughput 32.173
  tok/s, decode 12.895 tok/s, peak 9.330 GiB, and system-wide free memory was
  40% after the run. The raw output and token IDs are preserved under
  `research/mixed26-correctness/`.
- `P10-MIXED-SHORT-REPEAT`: the required fresh-process repeat reproduced the
  same `05a91786aa064bdcb8bb3589938cebeff61f487270c9b2acd9c84097b124f623`
  256-token hash and the same pathological sentence loop. It is therefore
  `BROKEN`; no 768/4K or long-context mixed tests were justified.

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
- A bounded, layer-oriented converter should keep peak memory tied to one
  Qwen3.8 layer plus output buffers, but this is an implementation hypothesis
  until a source-backed conversion run measures it.
- The self-test confirms the planned layer boundaries and sanitization logic;
  source-backed output integrity and peak memory remain unmeasured.
- The exact high-precision source is now available locally, so the g128 branch
  is no longer source-blocked; its next gate is bounded conversion memory and
  output integrity.
- The source-backed streaming path is empirically memory-safe for conversion
  on this machine. Affine g128 inference is also coherent through the tested
  16,727-token cold probe; warm-cache behavior remains unmeasured.
- The source-backed g128 candidate is coherent on the short control, coding
  prompts, and longer context. Its measured materialized model size is
  10.1806 GiB, consistent with the exact 10.1800 GiB tensor-payload estimate.
- The source-backed g128 candidate is also coherent at 768 tokens, with a
  measured 11.678 GiB cold peak. This is below the g64 12.459 GiB peak from
  the adjacent 769-token alignment evidence.
- The source-backed g128 candidate is coherent through the required 4K coding
  gate. Its 4K peak is 0.782 GiB below the measured g64 4K peak, matching the
  expected model-storage saving within allocator/runtime differences.
- The g128 8K peak is 0.783 GiB below the measured g64 8,191-token peak
  (14.766 GiB), matching the storage delta. The first long-context probe did
  not show correctness degradation or an allocation failure.
- The g128 12K peak is 0.783 GiB below the measured g64 12,288-token peak
  (15.392 GiB), and the 16,727-token adaptive probe also remained coherent
  without allocation failure. Larger probes were intentionally not started.
- The g128 16,727-token peak was 15.236 GiB, materially below the existing
  g64 real-prompt result of 17.200 GB decimal. It remained coherent, but the
  long cold TTFT (219.6 s) and increased swap make this a maximum cold probe,
  not yet a warm interactive recommendation.
- The corrected source-backed mixed_2_6 g128 artifact is structurally valid
  and matches the 8.988921 GiB header estimate, but two fresh short runs
  produced an identical pathological loop. It is rejected; its skipped 768/4K
  and long-context gates are not evidence of quality.
- The first KV8 isolation result is coherent and has a small 4K peak saving,
  but not yet a meaningful practical improvement. Because only the 16
  full-attention layers have KVCache objects, an 8K probe is needed before
  rejecting the lever for longer contexts.
- At 8K, KV8 still saves only 0.231 GiB while reducing decode by about 5.3%.
  This is a negative practical tradeoff for g64, but the requested g128+KV8
  combination was tested once at 8K and showed the same 0.231 GiB saving with
  about a 7.0% decode penalty. KV8 is rejected for larger probes and normal
  interactive use; no 12K/16K KV8 runs are justified.

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
- Original g64 + KV8 and g128 + KV8: coherent, but rejected because the
  measured 8K peak saving was only 0.231 GiB while decode slowed 5.3% and
  7.0%, respectively.
- Corrected mixed_2_6 g128: rejected after identical pathological repetition
  in two fresh short runs, despite successful load and stable memory.

## Current hypotheses

- The 0.32.1 stock OOM is a wired-residency/accounting interaction: the
  forced recommended limit changes the cache-state evaluation allocation path,
  while default wired state can pass.
- Persistent prompt caching is measured EXCELLENT at 4K and 8K, with
  sub-1.21-second follow-up TTFT, roughly 10.65–11.02 tok/s decode, and no
  material cache growth.
- Larger-group quantization may save model memory, but quality/runtime support
  and a valid high-precision source path are unresolved. The mixed_2_6
  predicate is statically verified and the corrected source-backed artifact is
  structurally valid, but this source-backed mixed policy is not usable on the
  short control because it deterministically loops.

## Next action

Phase 8's planned ladder is complete. Do not automatically start 20K/24K:
the 16,727-token result already establishes a materially safer cold envelope,
while larger synthetic probes would add memory exposure without improving the
interactive recommendation. Phase 10 is closed as a negative result. Phase
11/12 comparison and exact no-wired command documentation are recorded in
`research/FINAL_REPORT.md`. Do not rerun the rejected mixed candidate or its
skipped context gates.

## Safety notes

- No system wired-memory setting was modified. The Phase 5 gate, converter
  self-test, and source verification were completed before conversion.
- Source weights are cached but must not be committed. The output directory
  must remain untracked and must not be committed.
- Conversion succeeded without Metal OOM or severe memory pressure; inference
  validation must still stop at the first broken/degraded result.
- Do not run the pinned stock converter. Do not commit source or converted
  model weights.
- Do not use the older 0.32.0 environment.
- `ps`/`pgrep` process checks are unavailable in the sandbox; do not infer that
  an external process is absent beyond the failed checks.
