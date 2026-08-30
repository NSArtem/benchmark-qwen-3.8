# Qwen3.8 MTP architecture derivation

Status: Phase 7 architecture gate passed on 2026-08-30.

This document records the derivation used before attempting a decode
prototype. It is based on the original Qwen3.8 source checkpoint, the pinned
MLX text model, and the local MLX-VLM Qwen3.5 MTP reference implementation.
The source checkpoint is not modified.

## Source and reference evidence

Measured source paths:

- Original source checkpoint:
  `/Users/artem/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- Source index: `model.safetensors.index.json`, with all 15 MTP tensors in
  `model-00018-of-00018.safetensors`.
- Pinned MLX-LM implementation:
  `/Users/artem/.cache/uv/archive-v0/GD4ii1Q6IhKy2-zq/mlx_lm/models/qwen3_5.py`
  from the canonical `mlx-lm==0.31.3` environment.
- Reference draft implementation:
  `/Users/artem/.cache/uv/archive-v0/CIPJVfCDFLVPFe_o/mlx_vlm/speculative/drafters/qwen3_5_mtp/qwen3_5_mtp.py`
  and its `split.py`, from local `mlx_vlm` package metadata version 0.6.13.

The original config identifies `Qwen3_5ForConditionalGeneration`,
`model_type=qwen3_5`, `mtp_num_hidden_layers=1`, and
`mtp_use_dedicated_embeddings=false`. The current official Transformers
Qwen3.5 model source still lists `mtp.*` under unexpected keys to ignore and
does not expose an MTP forward path:
<https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py>.
The Qwen3.8 discussion independently records the two MTP config fields:
<https://github.com/QwenLM/Qwen3.8/discussions/111>.

The local MLX-VLM implementation is the first trustworthy executable
reference found for these exact tensor names. It uses the same one-layer
full-attention Qwen3.5 decoder block and the same source-to-MLX norm shift as
the pinned MLX-LM sanitizer. This is sufficient support for a guarded
prototype, but not evidence that MTP is production-ready.

## Tensor inventory

The 15 source tensors are all BF16 and total 849,398,784 bytes (0.791 GiB).
The source names are shown verbatim; the MLX draft key is obtained by removing
the leading `mtp.`.

| source tensor | shape | MLX draft role |
| --- | ---: | --- |
| `mtp.fc.weight` | `[5120, 10240]` | concatenate two normalized hidden vectors, then project to 5120 |
| `mtp.pre_fc_norm_embedding.weight` | `[5120]` | RMS norm of the current token embedding |
| `mtp.pre_fc_norm_hidden.weight` | `[5120]` | RMS norm of the target model hidden |
| `mtp.layers.0.input_layernorm.weight` | `[5120]` | MTP decoder pre-attention RMS norm |
| `mtp.layers.0.self_attn.q_proj.weight` | `[12288, 5120]` | gated Q projection, 24 heads × 2 × 256 |
| `mtp.layers.0.self_attn.k_proj.weight` | `[1024, 5120]` | 4 KV heads × 256 |
| `mtp.layers.0.self_attn.v_proj.weight` | `[1024, 5120]` | 4 KV heads × 256 |
| `mtp.layers.0.self_attn.o_proj.weight` | `[5120, 6144]` | attention output projection |
| `mtp.layers.0.self_attn.q_norm.weight` | `[256]` | query RMS norm |
| `mtp.layers.0.self_attn.k_norm.weight` | `[256]` | key RMS norm |
| `mtp.layers.0.post_attention_layernorm.weight` | `[5120]` | MLP pre-norm |
| `mtp.layers.0.mlp.gate_proj.weight` | `[17408, 5120]` | SwiGLU gate projection |
| `mtp.layers.0.mlp.up_proj.weight` | `[17408, 5120]` | SwiGLU up projection |
| `mtp.layers.0.mlp.down_proj.weight` | `[5120, 17408]` | SwiGLU down projection |
| `mtp.norm.weight` | `[5120]` | final MTP RMS norm |

There are no dedicated MTP embedding or LM-head tensors. The reference binds
the MTP draft to the target model's `embed_tokens` and `lm_head`, as required
by `mtp_use_dedicated_embeddings=false` and the source tensor inventory.

## Computation graph

Let `e_t` be the target embedding of the token currently being proposed and
let `h_t` be the target model's final normalized hidden at the preceding
position. The MTP input is:

```text
u_t = fc(concat(RMSNorm_embedding(e_t), RMSNorm_hidden(h_t)))
h_t' = MTPDecoderLayer(u_t, causal/full-attention KV cache)
z_t = RMSNorm(h_t')
logits_t = target_lm_head(z_t)
```

The single MTP decoder layer is forced to full attention (`full_attention_interval=1`)
even though the 64-layer target alternates three linear/GatedDelta layers with
one full-attention layer. Its block is the ordinary Qwen3.5 gated attention
plus residual MLP:

```text
r = gated_qwen_attention(input_layernorm(x), cache)
h = x + r
out = h + SwiGLU(post_attention_layernorm(h))
```

The draft is autoregressive. After the first MTP input, each following draft
step embeds the previous draft token and pairs it with the previous MTP hidden.
For `mtp_num_hidden_layers=1`, the reference block size is
`mtp_num_hidden_layers + 2 = 3`, which means at most two newly proposed draft
tokens per verification round. The target verifies the bonus token followed by
those draft tokens in one forward pass.

## Position and acceptance semantics

The committed target cache contains the prompt and all previously accepted
tokens. A round starts with `bonus`, the next token after that committed
prefix. The draft proposes `d[0], d[1]` (when the output budget permits). The
target evaluates `[bonus, d[0], d[1]]`; the hidden at each position predicts the
next target token. Greedy acceptance walks the target argmax against the draft
tokens from left to right:

1. accept `d[0]` if target argmax at `bonus` equals `d[0]`;
2. accept `d[1]` only if the previous comparison succeeded and the next target
   argmax equals `d[1]`;
3. on the first mismatch, emit the target argmax as the fallback token;
4. if all draft tokens match, emit the final target argmax after `d[1]`.

The target cache therefore advances by `accepted_draft_tokens + 1` committed
tokens per round. The draft cache is advanced speculatively and must retain
only the accepted portion plus the fallback token's state for the next round.

This is exact greedy speculative decoding when the target logits, tokenizer,
and cache transitions are the same as the baseline. The prototype must compare
the complete emitted token sequence with a fresh baseline greedy run before any
speed claim.

## Hybrid rollback requirement

The target has two different state families:

- full-attention layers use trimmable KV caches;
- the other 48 layers use `ArraysCache(size=2)`, containing convolution state
  and the recurrent GatedDelta state.

Rejecting a draft suffix must roll back both families. Trimming only the
full-attention KV caches is incorrect: the linear/GatedDelta recurrent state
would still include rejected tokens. The reference implementation captures
GatedDelta intermediate state during verification and restores/replays the
accepted prefix; a minimal prototype may use a slower snapshot-and-replay
implementation, but it must test rejection explicitly. The final MTP path
must never silently discard recurrent state.

The draft layer has its own one-layer KV cache. Target shared K/V state is not
fed into that cache in the local reference; only the target position/valid
length is used to position the draft cache. This is intentional for this
Qwen3.8 tensor layout and is recorded here so a future implementation does not
invent cross-model KV sharing.

## Weight conversion and precision gate

The original Qwen3.5 RMSNorm stores the centered parameter used as
`normalized_x * (1 + weight)`. MLX `nn.RMSNorm` stores the multiplier directly.
For raw source tensors, all MTP RMSNorm vectors therefore require `+1.0` during
conversion, matching the pinned MLX-LM sanitizer and the local MTP splitter.
This applies to the two pre-FC norms, the MTP layer input/post norms, q/k norms,
and final MTP norm. Projection tensors remain BF16 for the first correctness
prototype.

The raw BF16 payload is measured at 0.791 GiB. A 4-bit MTP artifact is only
eligible after the BF16 prototype passes exactness and shows useful acceptance;
it must be measured rather than inferred from the raw byte count. A BF16 MTP
path is not eligible for the daily 16K mode unless its measured residency stays
within the Stage 7 memory gate.

## Phase 7 decision

**MEASURED FACT:** the source config, all 15 tensor shapes, the pinned MLX
layer implementation, and a local executable MTP reference agree on a
one-layer full-attention draft with shared target embeddings/LM head.

**INFERENCE:** implementing a guarded canonical-runtime prototype is technically
justified; no architecture guess is required for the graph above.

**HYPOTHESIS:** exact greedy acceptance may produce enough accepted tokens to
offset the extra one-layer proposal cost. This is untested and cannot promote
MTP until Phase 8 correctness and Phase 10 short/4K/8K/16K measurements pass.
