"""Layer-at-a-time Qwen3.5-family forward helpers for Stage 10."""

from __future__ import annotations

import contextlib
import gc
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

from .source import load_sanitized_layer


def text_args(config: dict[str, Any]):
    from mlx_lm.models.qwen3_5 import ModelArgs, TextModelArgs

    model_args = ModelArgs.from_dict(config)
    return TextModelArgs.from_dict(model_args.text_config)


def fresh_decoder_layer(args, layer_idx: int, sanitized_weights: dict[str, Any] | None = None):
    from mlx_lm.models.qwen3_5 import DecoderLayer

    layer = DecoderLayer(args=args, layer_idx=layer_idx)
    if sanitized_weights is not None:
        prefix = f"language_model.model.layers.{layer_idx}."
        weights = []
        for key, value in sanitized_weights.items():
            if not key.startswith(prefix):
                raise ValueError(f"sanitized layer {layer_idx} contains unexpected key {key}")
            weights.append((key[len(prefix) :], value))
        layer.load_weights(weights, strict=True)
    layer.eval()
    return layer


class _LinearInputTap:
    """MLX module that forwards a linear call and exposes its input array."""

    def __init__(self, inner, path: str, callback: Callable[[str, Any], None]):
        self.inner = inner
        self.path = path
        self.callback = callback

    def __call__(self, inputs):
        self.callback(self.path, inputs)
        return self.inner(inputs)


@contextlib.contextmanager
def capture_linear_inputs(
    layer, callback: Callable[[str, Any], None], *, prefix: str = ""
) -> Iterator[None]:
    """Temporarily wrap every ``nn.Linear`` in ``layer`` and call ``callback``.

    The callback receives ``(module_path, input_array)`` immediately before
    each linear operation. It can accumulate statistics without retaining the
    whole layer's activation history.
    """
    import mlx.nn as nn

    replaced = []

    def visit(module, path: str) -> None:
        for name, child in list(module.items()):
            child_path = f"{path}.{name}" if path else name
            if isinstance(child, nn.Linear):
                module[name] = _LinearInputTap(
                    child,
                    f"{prefix}.{child_path}".strip("."),
                    callback,
                )
                replaced.append((module, name, child))
            elif isinstance(child, nn.Module):
                visit(child, child_path)

    visit(layer, "")
    try:
        yield
    finally:
        for parent, name, original in reversed(replaced):
            parent[name] = original


def decoder_masks(hidden, args):
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask

    return create_attention_mask(hidden, None), create_ssm_mask(hidden, None)


def run_decoder_layer(
    layer,
    hidden,
    args,
    layer_idx: int,
    capture_hook: Callable[[int, str, Any], None] | None = None,
):
    fa_mask, ssm_mask = decoder_masks(hidden, args)
    mask = ssm_mask if layer.is_linear else fa_mask
    if capture_hook is None:
        return layer(hidden, mask=mask, cache=None)
    with capture_linear_inputs(
        layer,
        lambda name, value: capture_hook(layer_idx, name, value),
        prefix=f"layers.{layer_idx}",
    ):
        return layer(hidden, mask=mask, cache=None)


def run_loaded_layers(
    text_model,
    hidden,
    *,
    args=None,
    capture_hook: Callable[[int, str, Any], None] | None = None,
    eval_each_layer: bool = True,
):
    import mlx.core as mx

    args = args or text_model.args
    for layer_idx, layer in enumerate(text_model.layers):
        next_hidden = run_decoder_layer(layer, hidden, args, layer_idx, capture_hook)
        if eval_each_layer:
            mx.eval(next_hidden)
        del hidden
        hidden = next_hidden
    return hidden


def run_source_layers(
    hidden,
    *,
    config: dict[str, Any],
    source_plan: dict[str, Any],
    sanitizer_model,
    capture_hook: Callable[[int, str, Any], None] | None = None,
    sequence_batch_size: int = 1,
):
    """Run all sequences through each freshly loaded BF16 source layer."""
    import mlx.core as mx

    args = text_args(config)
    if hidden.shape[0] < 1 or sequence_batch_size < 1:
        raise ValueError("hidden and sequence_batch_size must be non-empty")
    for layer_idx in range(args.num_hidden_layers):
        sanitized = load_sanitized_layer(
            sanitizer_model,
            source_plan["source"],
            source_plan["weight_map"],
            layer_idx,
        )
        layer = fresh_decoder_layer(args, layer_idx, sanitized)
        del sanitized
        outputs = []
        for start in range(0, hidden.shape[0], sequence_batch_size):
            batch = hidden[start : start + sequence_batch_size]
            output = run_decoder_layer(layer, batch, args, layer_idx, capture_hook)
            mx.eval(output)
            outputs.append(output)
            del batch, output
        next_hidden = mx.concatenate(outputs, axis=0)
        mx.eval(next_hidden)
        del outputs, layer, hidden
        hidden = next_hidden
        gc.collect()
        mx.clear_cache()
    return hidden, args


def scores_from_logits(logits, token_ids: np.ndarray, *, start: int = 0, top_k: int = 1):
    """Score target tokens and return full-vocabulary argmax/top-k per position."""
    import mlx.core as mx

    stop = start + logits.shape[1]
    targets = mx.array(token_ids[:, start + 1 : stop + 1], dtype=mx.int32)
    logits32 = logits.astype(mx.float32)
    log_probs = logits32 - mx.logsumexp(logits32, axis=-1, keepdims=True)
    picked = mx.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
    top1 = mx.argmax(log_probs, axis=-1).astype(mx.int32)
    result = {"token_logp": picked, "top1": top1}
    if top_k > 1:
        vocab = log_probs.shape[-1]
        top_ids = mx.argpartition(log_probs, vocab - top_k, axis=-1)[..., -top_k:]
        top_logp = mx.take_along_axis(log_probs, top_ids, axis=-1)
        result["topk_ids"] = top_ids.astype(mx.int32)
        result["topk_logp"] = top_logp
    return result


def _host_scores(result: dict[str, Any]) -> dict[str, np.ndarray]:
    import mlx.core as mx

    mx.eval(*result.values())
    return {key: np.asarray(value) for key, value in result.items()}


def score_logits(logits, token_ids: np.ndarray, *, chunk_size: int = 256, top_k: int = 1):
    pieces: dict[str, list[np.ndarray]] = {}
    positions = token_ids.shape[1] - 1
    for start in range(0, positions, chunk_size):
        stop = min(start + chunk_size, positions)
        values = _host_scores(
            scores_from_logits(logits[:, start:stop, :], token_ids, start=start, top_k=top_k)
        )
        for key, array in values.items():
            pieces.setdefault(key, []).append(array)
    return {key: np.concatenate(value, axis=1) for key, value in pieces.items()}


def score_hidden(
    hidden,
    lm_head,
    token_ids: np.ndarray,
    *,
    chunk_size: int = 256,
    top_k: int = 1,
):
    pieces: dict[str, list[np.ndarray]] = {}
    positions = token_ids.shape[1] - 1
    for start in range(0, positions, chunk_size):
        stop = min(start + chunk_size, positions)
        logits = lm_head(hidden[:, start:stop, :])
        values = _host_scores(scores_from_logits(logits, token_ids, start=start, top_k=top_k))
        for key, array in values.items():
            pieces.setdefault(key, []).append(array)
        del logits
    return {key: np.concatenate(value, axis=1) for key, value in pieces.items()}


def forward_loaded_model_hidden(model, token_ids):
    import mlx.core as mx

    text = model.language_model
    hidden = text.model.embed_tokens(token_ids)
    hidden = run_loaded_layers(text.model, hidden, args=text.args)
    hidden = text.model.norm(hidden)
    mx.eval(hidden)
    return hidden


def forward_source_hidden(
    token_ids,
    *,
    source_plan: dict[str, Any],
    sanitizer_model,
    sequence_batch_size: int = 1,
    capture_hook: Callable[[int, str, Any], None] | None = None,
):
    import mlx.core as mx

    from .source import load_sanitized_tensors, output_key

    source = source_plan["source"]
    weight_map = source_plan["weight_map"]
    raw_key = "model.language_model.embed_tokens.weight"
    embed = load_sanitized_tensors(sanitizer_model, source, weight_map, [raw_key])
    embed_weight = embed[output_key(raw_key)]
    hidden = mx.take(embed_weight, token_ids, axis=0)
    mx.eval(hidden)
    del embed, embed_weight
    mx.clear_cache()
    hidden, args = run_source_layers(
        hidden,
        config=source_plan["config"],
        source_plan=source_plan,
        sanitizer_model=sanitizer_model,
        capture_hook=capture_hook,
        sequence_batch_size=sequence_batch_size,
    )
    norm_key = "model.language_model.norm.weight"
    norm = load_sanitized_tensors(sanitizer_model, source, weight_map, [norm_key])
    norm_weight = norm[output_key(norm_key)]
    # Match the source model's final RMSNorm without instantiating another 27B tree.
    from mlx.nn import RMSNorm

    final_norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
    final_norm.load_weights([("weight", norm_weight)], strict=True)
    hidden = final_norm(hidden)
    mx.eval(hidden)
    del norm, norm_weight, final_norm
    mx.clear_cache()
    return hidden


def source_lm_head(source_plan: dict[str, Any], sanitizer_model):
    from mlx.nn import Linear

    from .source import load_sanitized_tensors, output_key

    key = "lm_head.weight"
    sanitized = load_sanitized_tensors(
        sanitizer_model, source_plan["source"], source_plan["weight_map"], [key]
    )
    weight = sanitized[output_key(key)]
    head = Linear(weight.shape[-1], weight.shape[-2], bias=False)
    head.load_weights([("weight", weight)], strict=True)
    head.eval()
    return head


def source_embedding(source_plan: dict[str, Any], sanitizer_model):
    """Load the BF16 embedding table into an MLX Embedding module."""
    from mlx.nn import Embedding

    from .source import load_sanitized_tensors, output_key

    key = "model.language_model.embed_tokens.weight"
    sanitized = load_sanitized_tensors(
        sanitizer_model, source_plan["source"], source_plan["weight_map"], [key]
    )
    weight = sanitized[output_key(key)]
    embedding = Embedding(weight.shape[0], weight.shape[1])
    embedding.load_weights([("weight", weight)], strict=True)
    embedding.eval()
    return embedding


def output_key_for_layer(layer_idx: int, suffix: str) -> str:
    return f"language_model.model.layers.{layer_idx}.{suffix}"

