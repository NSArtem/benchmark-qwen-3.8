"""Partial layer-by-layer AWQ scale search, then 3-bit RTN (TASK.md section 5)."""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

import numpy as np

from .gptq import (
    cast_layer_weights,
    calibration_embeddings,
    input_group,
    install_quantized,
    linear_children,
    run_layer_batches,
    split_batches,
)

SEARCH_SEQUENCES = 16
ALPHAS = tuple(round(0.1 * i, 1) for i in range(11))
TOKEN_CHUNK = 4096
# Guards the s / sqrt(max(s) * min(s)) normalization against all-zero channels.
MIN_MEAN_ABS = 1e-8

# Searched groups: input straight from an RMSNorm, plus down_proj.
NORM_GROUPS = {
    "linear_attn.in": "input_layernorm",
    "self_attn.in": "input_layernorm",
    "mlp.in": "post_attention_layernorm",
}
DOWN_GROUP = "mlp.down_proj"


class InputCapture:
    """Keeps every captured input once per shared-input group."""

    def __init__(self):
        self.X: dict[str, list] = {}
        self._done: set[str] = set()

    def __call__(self, layer_idx: int, name: str, value) -> None:
        group = input_group(name.split(".", 2)[2])
        if group in self._done:
            return
        self._done.add(group)
        self.X.setdefault(group, []).append(value.reshape(-1, value.shape[-1]))

    def flush(self) -> None:
        self._done = set()

    def inputs(self, group: str):
        import mlx.core as mx

        return mx.concatenate(self.X[group], axis=0)


def awq_scales(X, alpha: float):
    import mlx.core as mx

    mean_abs = mx.maximum(mx.mean(mx.abs(X.astype(mx.float32)), axis=0), MIN_MEAN_ABS)
    s = mean_abs**alpha
    return s / mx.sqrt(mx.max(s) * mx.min(s))


def group_mse(X, weights: list, s, bits: int = 3, group_size: int = 128) -> float:
    """Output MSE of the group with weights ``W·diag(s)`` (RTN) and inputs ``X / s``."""
    import mlx.core as mx

    from .qformat import rtn

    deq = []
    for W in weights:
        q, sc, b = rtn((W.astype(mx.float32) * s).astype(mx.bfloat16), bits, group_size)
        deq.append(mx.dequantize(q, sc, b, group_size=group_size, bits=bits).astype(mx.float32))
    Wq = mx.concatenate(deq, axis=0)
    Wf = mx.concatenate([W.astype(mx.float32) for W in weights], axis=0)
    total, count = 0.0, 0
    for start in range(0, X.shape[0], TOKEN_CHUNK):
        x = X[start : start + TOKEN_CHUNK].astype(mx.float32)
        diff = x @ Wf.T - (x / s) @ Wq.T
        value = mx.sum(diff * diff)
        mx.eval(value)
        total += float(value.item())
        count += diff.size
        del x, diff
    del Wq, Wf, deq
    return total / count


def search_alpha(X, weights: list) -> tuple[float, list[float], Any]:
    mses = [group_mse(X, weights, awq_scales(X, alpha)) for alpha in ALPHAS]
    best = int(np.argmin(mses))
    return ALPHAS[best], mses, awq_scales(X, ALPHAS[best])


def fold_scales(sanitized: dict[str, Any], layer_idx: int, group: str, members: list[str], s) -> None:
    """Divide the producer by ``s`` and multiply each consumer's columns by ``s`` (in place)."""
    import mlx.core as mx

    prefix = f"language_model.model.layers.{layer_idx}."

    def scaled(key, factor, axis):
        value = sanitized[key]
        f = factor if axis == -1 else factor[:, None]
        sanitized[key] = (value.astype(mx.float32) * f).astype(value.dtype)

    if group in NORM_GROUPS:
        scaled(prefix + NORM_GROUPS[group] + ".weight", 1.0 / s, -1)
    elif group == DOWN_GROUP:
        scaled(prefix + "mlp.up_proj.weight", 1.0 / s, 0)
    else:
        raise ValueError(group)
    for local in members:
        scaled(prefix + local + ".weight", s, -1)
    mx.eval(*sanitized.values())


def build_awq(exp_id: str, recipe_name: str, output: Path) -> dict:
    import mlx.core as mx

    from .build import RECIPES, _store_tensor, module_records
    from .candidate import PRODUCTION, CandidateWriter, candidate_config, quantize_module
    from .data import load_split
    from .layerwise import fresh_decoder_layer, text_args
    from .source import SOURCE, build_model, load_json, load_sanitized_layer, load_sanitized_tensors, read_source

    recipe = RECIPES[recipe_name]
    plan = read_source(SOURCE)
    model = build_model(plan["config"])
    cast_predicate = getattr(model, "cast_predicate", lambda _key: True)
    records = module_records(model, recipe)
    by_path = {record["path"]: record for record in records}
    args = text_args(plan["config"])
    log_path = Path(__file__).resolve().parents[3] / "benchmarks/results/stage10" / exp_id / "build_progress.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    writer = CandidateWriter(output)
    for raw in ("model.language_model.embed_tokens.weight", "lm_head.weight", "model.language_model.norm.weight"):
        loaded = load_sanitized_tensors(model, plan["source"], plan["weight_map"], [raw])
        for key in sorted(loaded):
            _store_tensor(writer, key, loaded.pop(key), by_path, recipe.seed, cast_predicate)
        del loaded
        mx.clear_cache()

    tokens = load_split("calibration")[:SEARCH_SEQUENCES]
    hidden = split_batches(calibration_embeddings(plan, model, tokens, recipe.rotate, recipe.seed))
    chosen: list[dict] = []

    for layer_idx in range(args.num_hidden_layers):
        t0 = time.monotonic()
        sanitized = cast_layer_weights(
            load_sanitized_layer(model, plan["source"], plan["weight_map"], layer_idx), cast_predicate
        )
        layer = fresh_decoder_layer(args, layer_idx, sanitized)
        capture = InputCapture()
        run_layer_batches(layer, hidden, args, layer_idx, hook=capture, keep=False)
        linears = linear_children(layer)
        members: dict[str, list[str]] = {}
        for local in sorted(linears):
            members.setdefault(input_group(local), []).append(local)
        searched = [g for g in members if g in NORM_GROUPS or g == DOWN_GROUP]
        results = {}
        for group in searched:
            X = capture.inputs(group)
            alpha, mses, s = search_alpha(X, [linears[local].weight for local in members[group]])
            mx.eval(s)
            results[group] = s
            chosen.append({"layer": layer_idx, "group": group, "alpha": alpha, "mse_by_alpha": mses})
            del X
            mx.clear_cache()
        del capture, layer, linears
        gc.collect()
        mx.clear_cache()
        for group, s in results.items():
            fold_scales(sanitized, layer_idx, group, members[group], s)

        prefix = f"language_model.model.layers.{layer_idx}."
        quantized: dict[str, dict] = {}
        for key in sorted(sanitized):
            path = key[: -len(".weight")] if key.endswith(".weight") else None
            if path in by_path:
                arrays = quantize_module(sanitized[key], by_path[path], recipe.seed)
                quantized[path[len(prefix) :]] = arrays
                for name, value in arrays.items():
                    writer.add(f"{path}.{name}", value)
            else:
                writer.add(key, sanitized[key])
        layer = fresh_decoder_layer(args, layer_idx, sanitized)
        install_quantized(layer, layer_idx, quantized, by_path)
        next_hidden = run_layer_batches(layer, hidden, args, layer_idx)
        del layer, quantized, hidden, sanitized
        hidden = next_hidden
        gc.collect()
        mx.clear_cache()
        alphas = {c["group"]: c["alpha"] for c in chosen if c["layer"] == layer_idx}
        log(f"layer {layer_idx} done in {time.monotonic() - t0:.1f}s; alphas {alphas}; peak {mx.get_peak_memory() / 2**30:.2f} GiB")

    config = candidate_config(load_json(PRODUCTION / "config.json"), exp_id, recipe_name, recipe.seed, records)
    writer.finish(config)
    return {
        "search_sequences": SEARCH_SEQUENCES,
        "alphas_tried": list(ALPHAS),
        "min_mean_abs": MIN_MEAN_ABS,
        "chosen": chosen,
    }
