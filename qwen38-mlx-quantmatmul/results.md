# Qwen3.8-27B MLX 3-bit quantized-matmul investigation

## Outcome

The MLX 3-bit affine `QuantizedLinear` path did not show incorrect or
nondeterministic behavior in this post-reboot run. Stored packed weights,
scales, and affine biases were identical across five fresh loads. The first
GatedDelta projection was deterministic within one process and across ten
fresh processes at T=1/128/512/768. Its output agreed with an explicit
dequantize-then-dense reference without a sequence-length-sensitive error.

The reduced suite across GatedDelta, full-attention, MLP, and `lm_head`
representatives was also stable. Whole-prompt and exact chunked full-model
activation fingerprints matched across fresh processes. The normal pinned
`generate_step` first token was stable at both the existing 512- and
768-token prompts.

Therefore this run does **not** identify an earliest divergent decoder layer
or suboperation. The previously reported fresh-process full-model divergence
was not reproduced after the hard reboot, so assigning it to
`mx.quantized_matmul` would be unsupported.

## 1. Model and quantization

The exact checkpoint is `lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly` at
snapshot `c98bba5926f51fec1c8d8737e577221673f524d7`. It loads as
`mlx_lm.models.qwen3_5.Model`, with 64 hybrid decoder layers and 497
`QuantizedLinear` modules. The stored quantization metadata is:

```text
mode=affine
bits=3
group_size=64
weight=uint32 packed
scales=bfloat16
biases=bfloat16
```

Representative paths and exact parameter hashes are in
[`module-manifest.json`](module-manifest.json). The selected paths include:

- `language_model.model.layers.0.linear_attn.in_proj_qkv/z/a/b`
- `language_model.model.layers.0.linear_attn.out_proj`
- `language_model.model.layers.3.self_attn.q_proj/k_proj/v_proj`
- layer-0 MLP gate/up/down projections
- `language_model.lm_head`

## 2. Quantized parameter fingerprints

All selected packed weights, scales, and affine biases matched across five
fresh processes. The input/configuration was unchanged, and all selected
hashes were present. This argues against fresh-load parameter mutation or
lazy materialization producing different stored model parameters.

## 3. Real `QuantizedLinear` determinism

For layer-0 `in_proj_qkv`, the same evaluated input was run 20 times in one
process at T=1, 16, 128, 512, and 768. Every output hash matched repeat 0;
the maximum numerical difference was zero at every length, with zero NaNs and
zero Infs.

The same layer was then run in ten fresh processes at T=1, 128, 512, and 768.
For each length there was exactly one input hash, one stored weight/scales/
biases hash set, and one output hash across all ten processes. There were no
NaNs or Infs.

## 4. Quantized path versus exact dequantized dense reference

The reference used the actual stored arrays and the pinned MLX API:

```text
W = mx.dequantize(weight, scales, biases, group_size=64, bits=3, mode="affine", dtype=float32)
reference = mx.matmul(x.astype(float32), W.T)
```

The module has no ordinary linear bias. Results for `in_proj_qkv` were:

| T | max abs error | mean abs error | RMS error | relative L2 | cosine |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.07443 | 0.009393 | 0.01205 | 0.010303 | 0.99994695 |
| 16 | 0.03019 | 0.001235 | 0.001861 | 0.001659 | 0.99999862 |
| 128 | 0.05970 | 0.001244 | 0.001878 | 0.001662 | 0.99999862 |
| 512 | 0.03054 | 0.001240 | 0.001869 | 0.001660 | 0.99999862 |
| 768 | 0.03174 | 0.001243 | 0.001877 | 0.001660 | 0.99999862 |

The larger relative error at T=1 is not accompanied by pathological values or
instability. From T=16 through T=768 the error is essentially flat. This is
consistent with different numeric accumulation paths (the dense reference is
float32) rather than a sequence-length-dependent quantized kernel failure.

The dense reference itself was repeated ten times in one process and once in
five fresh processes at T=512/768. Each length had one output hash and no
NaNs/Infs.

## 5. Representative matrix shapes

Five fresh processes tested T=1 and T=512 for:

```text
GatedDelta: in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, out_proj
Attention:  q_proj, k_proj, v_proj
MLP:        gate_proj, up_proj, down_proj
```

Every representative had exactly one quantized output hash per shape and
length, with zero NaNs/Infs. At T=512, the worst dense-reference relative L2
error among these representatives was approximately 0.004438 (GatedDelta
`in_proj_a`), with minimum cosine approximately 0.99999015. At T=1, the
largest relative L2 error was approximately 0.011445 (`v_proj`), with minimum
cosine approximately 0.99993454.

The quantized `lm_head` was tested at T=1 in five fresh processes and had one
output hash, one input hash, and no NaNs/Infs. A T=512 explicit dense reference
for the 248,320-by-5,120 head was intentionally skipped because materializing
that reference in float32 would consume roughly 5 GB and add avoidable memory
risk on an 18 GB machine.

## 6. Full-model activation localization

The first whole-prompt control used the existing 512-token prompt and the
normal Metal path. Three fresh processes were fingerprinted at embeddings,
decoder layers 0, 7, 15, 23, 31, 39, 47, 55, 63, and final normalization.
All 33 comparable records agreed with process 0.

Because the normal generator splits a 512-token prompt into 511-token prefill
plus a final one-token decode with its default `prefill_step_size=2048`, the
same schedule was tested explicitly. Three fresh processes produced 66
comparable prefill/decode records; all agreed. The first token was ID 2 (`#`)
in every process, with identical top-10 logits:

```text
24.125:2, 21.25:550, 20.25:1919, 19.5:332,
19.125:71093, 19.125:1206, 19.0:27775, 18.5:21018,
18.0:13962, 17.875:760
```

The normal pinned `generate_step` control then produced identical first-token
IDs and top-10 log-probabilities in five fresh processes at 512 tokens and in
ten stored fresh-process records at 768 tokens. Both prompts produced token
ID 2 (`#`) in every recorded process.

There is consequently no measured interval such as “layer 15 matches, layer
23 diverges,” and no first divergent suboperation to bisect. The correct
localization result is **not reproduced in the current post-reboot run**.

## 7. Answers to the requested questions

1. Loaded packed weights are identical across fresh processes: **yes**.
2. Scales are identical: **yes**.
3. Quantization biases are identical: **yes**.
4. A real Qwen `QuantizedLinear` is deterministic within one process: **yes**.
5. It is deterministic across fresh processes: **yes**, for the tested layer and representatives.
6. `mx.quantized_matmul` agrees reasonably with explicit dequantize + dense matmul: **yes**.
7. Disagreement grows with sequence length: **no**; it is flat from T=16 through T=768.
8. The 3-bit path differs from a 4-bit control: **not tested**, because the 3-bit path was clean and the stop rule was reached.
9. Failures specific to a tested matrix shape/layer type: **none observed**.
10. Direct `mx.quantized_matmul` reproduction: **not needed**; no real `QuantizedLinear` problem was observed to reduce.
11. Smallest direct reproduction: **none**.
12. First divergent full-model decoder layer: **none observed in this run**.
13. First divergent suboperation: **none observed**.
14. Current evidence implicates: **unresolved historical full-model behavior**, not the stored parameters or tested 3-bit quantized matmuls.
15. Self-conversion: **still justified as the next model-level control**, but it was not performed here. It should compare the same 3-bit affine/group-64 recipe against the existing conversion at the 512/768 prompts.
16. Upstream MLX `mx.quantized_matmul` issue: **not ready**. There is no minimal primitive failure from this run. The prior full-model issue remains worth reporting only after its fresh-process divergence is reproduced under a captured environment.

## Decision and next experiment

This is neither a quantized-matmul confirmation nor a localized full-model
failure. The evidence rules out, for the tested shapes and lengths:

- differing stored checkpoint parameters;
- same-process quantized projection nondeterminism;
- fresh-process quantized projection nondeterminism;
- a large or length-sensitive error relative to exact dequantization;
- divergence at the coarse full-model activation checkpoints;
- divergence in the normal first-token path at 512/768 after the reboot.

The highest-value next experiment is a controlled rerun that first captures a
fresh failing `stream_generate` completion and its exact system/runtime state,
then compares a self-converted Qwen3.8 checkpoint using the same affine
3-bit/group-64 format. If the failure returns only with the published
checkpoint, conversion metadata/weights become the leading suspect; if it
returns with both, the next investigation should target the higher-level
Qwen3.5 hybrid cache/forward integration rather than `mx.quantized_matmul`.
