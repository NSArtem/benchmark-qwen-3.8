# Qwen3.8-27B MLX Research Results

This report is a running ledger. Values marked `reuse` come from saved raw
artifacts; values marked `policy-summary` come from the current project
instructions and have no separate raw run directory in this checkout.

## Phase 0 evidence inventory

| Experiment | Runtime | Prompt/context | Result | Performance / memory | Provenance |
| --- | --- | ---: | --- | --- | --- |
| `P0-ALIGNMENT-REUSE` | MLX 0.32.1, direct `generate_step`, default wired, g64 affine | 512/513/640/641/768/769 final tokens; 128 output | 18/18 COHERENT; 0 OOM; deterministic within length | Peaks: 12.188, 12.187, 12.325, 12.327, 12.461, 12.459 GiB respectively; after-first-token active+cache roughly 11.21–11.40 GiB | `qwen38-mlx0321-alignment/results.json`, commit `4418c08` |
| `P0-OOM-REUSE` | MLX 0.32.1, direct `generate_step`, g64 affine | 768 final tokens | Default wired PASS; forced recommended 12.000 GiB wired limit OOM | Default after-first-token peak 12.461 GiB; forced OOM direct snapshot active+cache 11.958 GiB | `qwen38-mlx0321-oom/results.json`, commit `664941d` |
| `P0-16K-POLICY` | MLX 0.32.1, normal generate with wired wrapper made no-op, g64 affine | 16,767 actual prompt tokens; 256 output | COHERENT; no OOM; understands Lantern task | 63.348 prompt tok/s; 4.118 decode tok/s; 17.200 GB MLX peak | `AGENTS.md` project-policy summary; raw run not present |

## Baseline context table (to be filled by Phase 1)

The 512–769 rows are correctness probes with a compact Lantern prompt family,
not interchangeable with the larger coding prompt. The 16,767 row is retained
without duplication because the task explicitly supplies it as existing
evidence. Missing cells are intentionally blank, not estimates.

Phase 1 prompt manifest: `research/context-ladder/prompt-manifest.json`.
The exact requested 1,024 and 8,192 counts were not reachable at a character
boundary without artificial padding; the saved natural prefixes are 1,023 and
8,191 tokens.

| Target | Actual prompt tokens | Classification | TTFT s | Prompt tok/s | Decode tok/s | Generated tokens | Active GiB | Cache GiB | Active+cache GiB | MLX peak GiB | OS memory | Exception / stability |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ~1K | 1,023 | COHERENT | 13.352 | 76.615 | 11.112 | 128 | 11.201 | 0.189 | 11.389 | 12.860 | 28% free; 41.69 MB swap after | no exception; stable |
| ~2K | 2,048 | COHERENT | 25.671 | 79.780 | 11.002 | 128 | 11.304 | 0.095 | 11.399 | 13.824 | 26% free; 41.69 MB swap after | no exception; stable |
| ~4K | 4,096 | COHERENT | 51.383 | 79.715 | 10.858 | 128 | 11.449 | 0.000 | 11.449 | 14.142 | 25% free; 41.69 MB swap after | no exception; stable |
| ~8K | 8,191 | COHERENT | 104.330 | 78.511 | 10.415 | 128 | 11.634 | 0.000 | 11.634 | 14.766 | 24% free; 41.69 MB swap after | no exception; stable |
| ~12K | 12,288 | COHERENT | 160.908 | 76.366 | 7.734 | 128 | 11.997 | 0.050 | 12.047 | 15.392 | 24% free; 235.75 MB swap after | no exception; stable |
| ~16K | 16,767 | COHERENT (policy-summary) |  — | 63.348 | 4.118 | 256 | — | — | — | 17.200 | no OOM reported; raw OS telemetry unavailable | Existing manual run; do not duplicate solely for evidence |

## Reuse conclusions

- The simple prefill alignment hypothesis is rejected by the 18/18 coherent
  result in the saved 0.32.1 experiment.
- The forced stock wired-limit policy is a distinct failure mechanism from the
  default-wired direct path; it must not be used by subsequent experiments.
- The fresh cold g64 ladder is coherent through 12,288 prompt tokens. Peak
  memory reached 15.392 GiB there, while the separate 16,767-token policy run
  reached 17.200 GB; this ledger therefore treats 12,288 as the largest
  directly measured memory-safe point and does not recommend the 16.8K point
  for routine use.
- Older `qwen38-3bit-mlx-rescue/` and `qwen38-3bit-textonly/` reports used MLX
  0.32.0 and are not valid current-baseline rows.

## Phase 2 no-wired runtime

`research/runtime/qwen38_runtime.py` is a small local wrapper around the
pinned `generate_step`/`speculative_generate_step` APIs. It deliberately does
not enter mlx-lm's stock wired-limit context and prints:

`[qwen38-runtime] mlx-lm wired_limit wrapper disabled; MLX wired state untouched`

The metadata command confirmed MLX 0.32.1, Python 3.14.7, revision
`254d153fdeb6f150edd4fc5a54f9828638481fa8`, and affine 3-bit g64 metadata. A
short 25-token prompt produced a coherent 37-token answer at 11.497 tok/s
decode. The corrected moderate 4,096-token Lantern prompt produced a coherent
64-token answer at 79.688 prompt tok/s and 11.027 decode tok/s, with 15.185 GB
decimal MLX peak. The first moderate attempt bypassed the chat template and
returned one EOS token; this helper-only interface bug was fixed before the
recorded passing validation.

## Phase 3 warm / interactive behavior

The persistent `NoWiredChat` session was tested in one process with the 4,096
token Lantern coding prompt followed by three short follow-ups. The initial
turn reproduced the cold baseline (51.484 s TTFT, 79.710 prompt tok/s, 10.943
decode tok/s, 14.142 GiB cumulative peak). All four responses were coherent.
The incremental turns processed only 33, 35, and 30 new prompt tokens and had
TTFTs of 0.726, 1.103, and 0.728 s, with decode rates of 11.024, 11.017, and
11.005 tok/s. Active+cache after the turns was 11.463, 11.433, 11.447, and
11.447 GiB. The measured 4K interactive profile is classified **EXCELLENT**;
the 8K confirmation below was run before choosing the practical context.

The same four-turn script at 8,191 initial prompt tokens was also **EXCELLENT**:
initial TTFT 104.222 s, 78.667 prompt tok/s, 10.547 decode tok/s, and 14.766
GiB cumulative peak. Follow-up TTFTs were 0.816, 1.202, and 0.828 s, with
10.664, 10.651, and 10.645 tok/s decode. Active+cache after turns was 11.714,
11.684, 11.697, and 11.697 GiB, with 25% system free memory. This supports an
8K practical warm-context recommendation; 12K remains measured only as a cold
one-shot point.

## Phase 4 exact quantization size study

The tool `research/harness/analyze_quant_sizes.py` parsed only safetensor
headers, config, and the pinned mlx-lm conversion predicate. It found 1,847
tensors and 498 quantized modules in the three-shard checkpoint. The current
affine g64 formula reproduced the exact 11,771,132,928-byte stored model. Both
the embedding and lm_head are among the quantized modules; all observed input
widths are divisible by 128. The index's parameter metadata excludes 4,608
saved `A_log`/`dt_bias` buffer elements, which reconciles the count.

| Candidate | Effective bpw | Stored GiB | Saved GiB | Saved vs current | First-order extra tokens to 16 GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| affine 3-bit g64 (current) | 3.501 | 10.963 | 0.000 | 0.00% | 0 |
| affine 3-bit g128 | 3.251 | 10.180 | 0.783 | 7.14% | 4,019 |
| mixed_2_6 g64 | 3.121 | 9.772 | 1.191 | 10.87% | 6,116 |
| mixed_2_6 g128 | 2.871 | 8.989 | 1.974 | 18.00% | 10,134 |
| affine 2-bit g128 | 2.251 | 7.049 | 3.913 | 35.70% | 20,093 |

The cold-curve least-squares fit is peak GiB ≈ 13.120 + 0.0001948 × prompt
tokens. The extra-token and 16 GiB columns are first-order planning math only;
they do not model quality, conversion overhead, allocator behavior, or a safe
OS operating point.

Candidate ranking: affine 3-bit g128 has the lowest expected quality risk and
same runtime family but only modest savings; mixed_2_6 g128 is the strongest
built-in memory/quality tradeoff to investigate if valid source weights are
available; affine 2-bit g128 saves the most but should not be treated as a
recommendation without quality testing.

## Phase 5 — conversion-feasibility review

The exact baseline model card identifies `Qwen/Qwen3.8-27B` as the source and
states that the existing text-only checkpoint was converted with mlx-lm 0.31.2.
The source is not local. Read-only Hugging Face metadata at revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` reports 18 safetensor shards totaling
55,563,006,776 bytes (about 51.74 GiB). The source index contains 333 vision
tensors and MTP tensors in addition to the text tower; its text layer keys are
`model.language_model.layers.N.*`.

The disk gate passes: `df -h` reported 246 GiB free and the existing HF cache is
106G. Source plus the estimated 10.180 GiB g128 output remains within the
project’s roughly 80 GiB free-space rule without deleting caches.

The pinned mlx-lm converter was inspected but rejected for this machine without
running it. Although it begins with lazy loading, it quantizes a complete model
graph and `save_model` flattens all model parameters before output sharding. The
current upstream source at commit
`13425df69fdf066ed84d8f1bbdc8182a7576740d` has the same standard structure.
Official upstream issue #876 and chunked-conversion commit
`a0803664761e2c9c52ecbcb2bb012aff3e8eef06` document a layer-oriented streaming
approach. That code is not in the pinned runtime and needs Qwen3.8 text-only
adaptation, so the Phase 5 decision is:

> **STREAMING_CONVERSION_REQUIRED**

No source weights were downloaded and no conversion was started at this gate.

## Phase 6 — bounded converter self-test

`research/harness/convert_qwen38_textonly_g128.py` was compiled and run with
the pinned MLX 0.32.1 environment before any source download. The self-test
found exactly 498 quantizable modules in the text-only baseline, planned a
non-layer chunk followed by 64 transformer-layer chunks, and verified the
`model.language_model.*` to `language_model.model.*` mapping plus `lm_head`
mapping. A full-attention norm-only chunk preserved the pinned Qwen3.5
sanitization rule by shifting a test value from 1.0 to 2.0. The tool contains
no wired-limit setter call. This proves only the static planning/sanitization
path; source-backed integrity, conversion time, and peak memory remain open.

The exact high-precision source was then downloaded at revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. Verification found 18 safetensor
shards, 1,199 indexed tensors, 333 vision tensors, 15 MTP tensors, and 851
retained text tensors. The source shard bytes sum to 55,563,006,776 (51.747
GiB); the source index metadata reports 55,562,855,904 bytes of tensor payload.
The source cache occupies 52G and 194 GiB remains free. No conversion output
has been created yet.

The bounded conversion then succeeded in one fresh process in 24.920 seconds.
It processed the non-layer chunk plus all 64 transformer-layer chunks and
created affine 3-bit g128 output with 1,847 tensors in 19 shards. The exact
tensor payload is 10,930,715,648 bytes (10.180022 GiB), matching the header
estimate; physical shard files total 10,930,954,231 bytes after safetensor
headers. The output index has the baseline 26,895,993,856 trainable-parameter
count and the header verifier reports `parameter_count_match: true`.

Conversion memory was substantially below the inference memory envelope:
maximum MLX peak 3.330 GiB, maximum active+cache 0.508 GiB, maximum process RSS
3.436 GiB, and 92% system-wide free memory at completion. This is a measured
conversion result, not an inference or quality result. The untracked output is
at `/private/tmp/qwen38-g128-mlx`; the per-chunk ledger is
`research/conversion/g128-progress.jsonl` and header analysis is under
`research/quant-size-g128/`.
