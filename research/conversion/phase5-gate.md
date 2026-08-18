# Phase 5 conversion-feasibility gate

Date: 2026-08-18

Decision: **STREAMING_CONVERSION_REQUIRED**

## Source provenance

The exact baseline model card was fetched read-only from:

<https://huggingface.co/lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly/raw/main/README.md>

It identifies the source/base model as `Qwen/Qwen3.8-27B` and says the
checkpoint was converted with mlx-lm 0.31.2. The local baseline config is the
Qwen3.5-family text-only architecture (`Qwen3_5ForConditionalGeneration`) with
64 layers, 5,120 hidden size, 24 attention heads, 4 KV heads, and 248,320
vocabulary entries. The exact local source location is absent; only quantized
Qwen3.8 derivatives are cached.

Read-only Hugging Face metadata for `Qwen/Qwen3.8-27B` at revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` reports 18 safetensor shards totaling
55,563,006,776 bytes. Its index contains 1,199 tensors: 851 text/MTP-related
tensors and 333 vision tensors. The source text keys use
`model.language_model.layers.N.*`; the current text-only MLX checkpoint has
already removed the vision tower and MTP tensors.

## Disk gate

- Free disk at gate: 246 GiB (`df -h` on the data volume).
- Existing Hugging Face cache: 106G.
- Planned source download: 55,563,006,776 bytes (about 51.74 GiB).
- Estimated g128 output: 10.180 GiB from `P4-QUANT-SIZES`.
- The project’s roughly 80 GiB free-disk rule remains satisfied after source
  plus output; no unrelated cache will be deleted.

## Conversion-path review

Pinned mlx-lm revision:
`254d153fdeb6f150edd4fc5a54f9828638481fa8`

Pinned `mlx_lm.convert` calls `load(..., lazy=True)`, constructs the complete
model graph, applies `nn.quantize`, then `save_model` collects all flattened
parameters before splitting them into output shards. The lazy load avoids
immediately materializing the source, but the full-model quantization/save
graph is not a bounded-memory conversion strategy for this 27B source on an
18 GB machine. Stock conversion is therefore rejected without running it.

Current upstream `mlx-lm` at commit
`13425df69fdf066ed84d8f1bbdc8182a7576740d` still has the same standard
whole-model conversion structure. Official upstream issue #876 documents the
memory bottleneck and references chunked-conversion commit
`a0803664761e2c9c52ecbcb2bb012aff3e8eef06`, which processes one layer at a
time. That implementation is not in the pinned environment and its generic
layer matcher rejects vision configs, so this investigation will adapt the
bounded approach locally for the text-only Qwen3.8 key layout.

## Gate consequences

Proceed to Phase 6 only with a fresh-process, layer-oriented converter that:

1. filters vision and MTP source keys;
2. applies the pinned Qwen3.5 text-model sanitization per chunk;
3. quantizes one layer at a time with affine 3-bit group size 128;
4. flushes small output shards and clears MLX cache between chunks; and
5. stops on the first Metal OOM or severe memory-pressure signal.

No source weights have been downloaded and no conversion has started at the
time this gate was recorded.
