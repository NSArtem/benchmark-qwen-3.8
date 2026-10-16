"""Layer-by-layer GPTQ with 2/3/4-bit packing (TASK.md section 5).

The per-column algorithm is copied from mlx-lm ``mlx_lm/quant/gptq.py``
(0.32.1, which contains the error-propagation fix of commit 9974d3f):
damping ``0.01 * mean(diag(H))``, ``Hinv = chol_upper(inv(H))``, per-group
scales from ``mx.quantize`` on the working weights, in-group updates per
column and one trailing update per group. Two changes: codes are packed with
``qformat.pack`` so 3 bits work, and the in-group updates act on the
``[out, group]`` block instead of re-assigning the whole matrix per column
(``Hinv`` is upper triangular, so the arithmetic is the same).
"""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

import numpy as np

from .qformat import hadamard_rotate, make_signs, pack

CALIBRATION_SEQUENCES = 96
HESSIAN_CHUNK = 8
MSE_SEQUENCES = 8
DAMP = 0.01
MAX_DAMP_RETRIES = 3
MSE_TOKEN_CHUNK = 2048

# Linears that read the same input share one Hessian.
SHARED_INPUT = {
    "linear_attn.in_proj_qkv": "linear_attn.in",
    "linear_attn.in_proj_z": "linear_attn.in",
    "linear_attn.in_proj_b": "linear_attn.in",
    "linear_attn.in_proj_a": "linear_attn.in",
    "self_attn.q_proj": "self_attn.in",
    "self_attn.k_proj": "self_attn.in",
    "self_attn.v_proj": "self_attn.in",
    "mlp.gate_proj": "mlp.in",
    "mlp.up_proj": "mlp.in",
}


def input_group(local_name: str) -> str:
    return SHARED_INPUT.get(local_name, local_name)


def rotate_hessian(H, block: int, signs):
    """``R^T H R`` for the rotation ``x -> hadamard_rotate(x)`` (= ``x R``)."""
    HR = hadamard_rotate(H, block, signs)
    return hadamard_rotate(HR.T, block, signs).T


def inverse_hessian_factor(H, damp_scale: float = DAMP):
    """Upper Cholesky factor of ``inv(H + damp*I)`` on the CPU, or None if not finite."""
    import mlx.core as mx

    try:
        with mx.stream(mx.cpu):
            damp = damp_scale * mx.mean(mx.diag(H))
            Hd = H + damp * mx.eye(H.shape[0], dtype=H.dtype)
            L = mx.linalg.cholesky(Hd)
            Hinv = mx.linalg.cholesky_inv(L)
            U = mx.linalg.cholesky(Hinv, upper=True)
            U = mx.triu(U)
            ok = mx.all(mx.isfinite(U))
            mx.eval(U, ok)
    except RuntimeError:
        return None
    return U if bool(ok.item()) else None


def gptq_quantize_weight(W, Hinv, bits: int, group_size: int = 128):
    """GPTQ on float32 ``W`` [out, in]; returns (codes uint8 np, scales f32, biases f32)."""
    import mlx.core as mx

    n_bins = 2**bits - 1
    n = W.shape[-1]
    rest = W.astype(mx.float32)
    codes, all_scales, all_biases = [], [], []
    for i in range(0, n, group_size):
        j = i + group_size
        block = rest[:, :group_size]
        Hblk = Hinv[i:j, i:j]
        _, scales, biases = mx.quantize(block, bits=bits, group_size=group_size)
        errs, qs = [], []
        for k in range(group_size):
            w = block[:, k : k + 1]
            q = mx.clip(mx.round((w - biases) / scales), 0.0, n_bins)
            e = (w - (scales * q + biases)) / Hblk[k, k]
            block = block - e @ Hblk[k : k + 1, :]
            errs.append(e)
            qs.append(q)
        err = mx.concatenate(errs, axis=1)
        q_group = mx.concatenate(qs, axis=1).astype(mx.uint8)
        rest = rest[:, group_size:]
        if j < n:
            rest = rest - err @ Hinv[i:j, j:]
        mx.eval(rest, q_group, scales, biases)
        codes.append(np.asarray(q_group))
        all_scales.append(scales)
        all_biases.append(biases)
    return (
        np.concatenate(codes, axis=1),
        mx.concatenate(all_scales, axis=-1),
        mx.concatenate(all_biases, axis=-1),
    )


def gptq_module(W, H, bits: int, group_size: int = 128, log=print, name: str = ""):
    """Full GPTQ for one module with damping retries; RTN fallback. Returns arrays + info."""
    import mlx.core as mx

    from .qformat import rtn

    damp_scale = DAMP
    Hinv = None
    retries = 0
    for attempt in range(MAX_DAMP_RETRIES + 1):
        Hinv = inverse_hessian_factor(H, damp_scale)
        if Hinv is not None:
            break
        if attempt < MAX_DAMP_RETRIES:
            retries += 1
            damp_scale *= 10
            log(f"cholesky retry {retries} for {name}: damping scale {damp_scale:g}")
    if Hinv is None:
        log(f"cholesky failed for {name}; RTN fallback")
        q, s, b = rtn(W.astype(mx.bfloat16), bits, group_size)
        return {"weight": q, "scales": s, "biases": b}, {"retries": retries, "fallback_rtn": True}
    codes, scales, biases = gptq_quantize_weight(W, Hinv, bits, group_size)
    del Hinv
    orig = W.dtype if W.dtype != mx.float32 else mx.bfloat16
    out = {"weight": pack(codes, bits), "scales": scales.astype(orig), "biases": biases.astype(orig)}
    return out, {"retries": retries, "fallback_rtn": False, "damp_scale": damp_scale}


def output_mse(X, W, arrays: dict, bits: int, group_size: int = 128) -> float:
    """Mean squared difference of ``X @ W.T`` and ``X @ deq(arrays).T`` (float32)."""
    import mlx.core as mx

    Wq = mx.dequantize(arrays["weight"], arrays["scales"], arrays["biases"], group_size=group_size, bits=bits)
    Wq = Wq.astype(mx.float32)
    Wf = W.astype(mx.float32)
    total = 0.0
    count = 0
    for start in range(0, X.shape[0], MSE_TOKEN_CHUNK):
        x = X[start : start + MSE_TOKEN_CHUNK].astype(mx.float32)
        diff = x @ Wf.T - x @ Wq.T
        value = mx.sum(diff * diff)
        mx.eval(value)
        total += float(value.item())
        count += diff.size
        del x, diff
    return total / count


# ---------------------------------------------------------------- layer plumbing


def calibration_embeddings(plan, sanitizer_model, tokens: np.ndarray, rotate: bool, seed: int):
    """BF16 embedding rows for ``tokens``; rotated and inverse-rotated when ``rotate``."""
    import mlx.core as mx

    from .source import load_sanitized_tensors, output_key

    raw = "model.language_model.embed_tokens.weight"
    table = load_sanitized_tensors(sanitizer_model, plan["source"], plan["weight_map"], [raw])[output_key(raw)]
    table = table.astype(mx.bfloat16)
    hidden = mx.take(table, mx.array(tokens, dtype=mx.int32), axis=0)
    mx.eval(hidden)
    del table
    if rotate:
        from .build import BLOCK

        signs = mx.array(make_signs(seed, "language_model.model.embed_tokens", hidden.shape[-1]))
        rows = []
        for i in range(hidden.shape[0]):
            r = hadamard_rotate(hadamard_rotate(hidden[i], BLOCK, signs).astype(mx.bfloat16), BLOCK, signs, inverse=True)
            mx.eval(r)
            rows.append(r.astype(mx.bfloat16))
        hidden = mx.stack(rows)
        mx.eval(hidden)
    mx.clear_cache()
    return hidden


def cast_layer_weights(sanitized: dict[str, Any], cast_predicate) -> dict[str, Any]:
    import mlx.core as mx

    out = {}
    for key, value in sanitized.items():
        if cast_predicate(key) and mx.issubdtype(value.dtype, mx.floating):
            value = value.astype(mx.bfloat16)
        out[key] = value
    return out


def linear_children(layer) -> dict[str, Any]:
    import mlx.nn as nn

    return {path: module for path, module in layer.named_modules() if isinstance(module, nn.Linear)}


def install_quantized(layer, layer_idx: int, quantized: dict[str, dict], records: dict[str, dict]):
    """Replace each nn.Linear of ``layer`` with a populated QLinear."""
    from mlx.utils import tree_unflatten

    from .candidate import QLinear

    replacements = []
    for local, module in linear_children(layer).items():
        path = f"language_model.model.layers.{layer_idx}.{local}"
        record = records[path]
        arrays = quantized[local]
        rows, width = module.weight.shape
        q = QLinear(width, rows, record["bits"], record["group_size"], record["block"], record["rotate_dtype"])
        q.weight = arrays["weight"]
        q.scales = arrays["scales"]
        q.biases = arrays["biases"]
        if record["block"]:
            q.signs = arrays["signs"]
        q.freeze()
        replacements.append((local, q))
    layer.update_modules(tree_unflatten(replacements))
    return layer


def split_batches(hidden, batch: int = HESSIAN_CHUNK) -> list:
    return [hidden[start : start + batch] for start in range(0, hidden.shape[0], batch)]


def run_layer_batches(layer, batches: list, args, layer_idx: int, hook=None, keep: bool = True) -> list:
    """Run each batch of sequences through ``layer``; returns a new list of batches."""
    import mlx.core as mx

    from .layerwise import run_decoder_layer

    outputs = []
    for batch in batches:
        out = run_decoder_layer(layer, batch, args, layer_idx, hook)
        mx.eval(out)
        if keep:
            outputs.append(out)
        del out
        if hook is not None:
            hook.flush()
    return outputs


class HessianHook:
    """Accumulates ``H = Σ XᵀX`` per shared input and keeps the first 8 sequences' X."""

    def __init__(self, keep_sequences: int = MSE_SEQUENCES):
        self.H: dict[str, Any] = {}
        self.X: dict[str, list] = {}
        self.keep_sequences = keep_sequences
        self.seen_rows = 0
        self._batch_rows = 0

    def __call__(self, layer_idx: int, name: str, value) -> None:
        import mlx.core as mx

        local = name.split(".", 2)[2]
        group = input_group(local)
        if group in self._done_this_batch:
            return
        self._done_this_batch.add(group)
        self._batch_rows = value.shape[0]
        xf = value.reshape(-1, value.shape[-1]).astype(mx.float32)
        update = xf.T @ xf
        self.H[group] = update if group not in self.H else self.H[group] + update
        if self.seen_rows < self.keep_sequences:
            take = min(value.shape[0], self.keep_sequences - self.seen_rows)
            self.X.setdefault(group, []).append(value[:take].reshape(-1, value.shape[-1]))
        mx.eval(self.H[group])

    def start(self):
        self._done_this_batch = set()

    def flush(self) -> None:
        self.seen_rows += self._batch_rows
        self._done_this_batch = set()

    def inputs(self, group: str):
        import mlx.core as mx

        return mx.concatenate(self.X[group], axis=0)


def build_gptq(exp_id: str, recipe_name: str, output: Path, max_layers: int | None = None) -> dict:
    import mlx.core as mx

    from .build import RECIPES, _store_tensor, module_records
    from .candidate import PRODUCTION, CandidateWriter, candidate_config
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
    log_path = Path(__file__).resolve().parents[3] / "benchmarks/results/stage10" / exp_id / ("build_progress.log" if max_layers is None else "smoke_progress.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    writer = CandidateWriter(output)
    # Non-layer tensors: RTN with the recipe's bits (embed_tokens, lm_head); norm as-is.
    for raw in ("model.language_model.embed_tokens.weight", "lm_head.weight", "model.language_model.norm.weight"):
        loaded = load_sanitized_tensors(model, plan["source"], plan["weight_map"], [raw])
        for key in sorted(loaded):
            _store_tensor(writer, key, loaded.pop(key), by_path, recipe.seed, cast_predicate)
        del loaded
        mx.clear_cache()

    tokens = load_split("calibration")[:CALIBRATION_SEQUENCES]
    hidden = calibration_embeddings(plan, model, tokens, recipe.rotate, recipe.seed)
    log(f"calibration hidden {tuple(hidden.shape)}")
    hidden = split_batches(hidden)
    mse_before: list[float] = []
    mse_after: list[float] = []
    module_order: list[str] = []
    events: list[dict] = []

    for layer_idx in range(args.num_hidden_layers if max_layers is None else max_layers):
        t0 = time.monotonic()
        sanitized = cast_layer_weights(
            load_sanitized_layer(model, plan["source"], plan["weight_map"], layer_idx), cast_predicate
        )
        layer = fresh_decoder_layer(args, layer_idx, sanitized)
        hook = HessianHook()
        hook.start()
        run_layer_batches(layer, hidden, args, layer_idx, hook=hook, keep=False)
        quantized: dict[str, dict] = {}
        for local, module in sorted(linear_children(layer).items()):
            path = f"language_model.model.layers.{layer_idx}.{local}"
            record = by_path[path]
            group = input_group(local)
            W = module.weight.astype(mx.float32)
            H = hook.H[group]
            X = hook.inputs(group)
            signs = None
            if record["block"]:
                signs = mx.array(make_signs(recipe.seed, path, W.shape[-1]))
                W = hadamard_rotate(W, record["block"], signs)
                H = rotate_hessian(H, record["block"], signs)
                X = hadamard_rotate(X, record["block"], signs)
            mx.eval(W, H, X)
            Wb = W.astype(mx.bfloat16)
            from .qformat import rtn

            q, s, b = rtn(Wb, record["bits"], record["group_size"])
            rtn_arrays = {"weight": q, "scales": s, "biases": b}
            arrays, info = gptq_module(Wb, H, record["bits"], record["group_size"], log=log, name=path)
            if signs is not None:
                arrays["signs"] = signs
            if info["retries"] or info["fallback_rtn"]:
                events.append({"module": path, **info})
            mse_before.append(output_mse(X, Wb, rtn_arrays, record["bits"]))
            mse_after.append(output_mse(X, Wb, arrays, record["bits"]))
            module_order.append(path)
            quantized[local] = arrays
            del W, H, X, Wb, rtn_arrays
            mx.clear_cache()
        del hook
        gc.collect()
        mx.clear_cache()

        # Store this layer: unquantized tensors as-is, quantized modules from GPTQ.
        prefix = f"language_model.model.layers.{layer_idx}."
        for key in sorted(sanitized):
            path = key[: -len(".weight")] if key.endswith(".weight") else None
            if path in by_path:
                for name, value in quantized[path[len(prefix) :]].items():
                    writer.add(f"{path}.{name}", value)
            else:
                writer.add(key, sanitized[key])
        del sanitized

        install_quantized(layer, layer_idx, quantized, by_path)
        next_hidden = run_layer_batches(layer, hidden, args, layer_idx)
        del layer, quantized, hidden
        hidden = next_hidden
        gc.collect()
        mx.clear_cache()
        log(
            f"layer {layer_idx} done in {time.monotonic() - t0:.1f}s; "
            f"peak {mx.get_peak_memory() / 2**30:.2f} GiB"
        )

    config = candidate_config(load_json(PRODUCTION / "config.json"), exp_id, recipe_name, recipe.seed, records)
    writer.finish(config)
    return {
        "calibration_sequences": CALIBRATION_SEQUENCES,
        "damp": DAMP,
        "act_order": False,
        "mse_sequences": MSE_SEQUENCES,
        "mse_modules": module_order,
        "mse_before_gptq": mse_before,
        "mse_after_gptq": mse_after,
        "cholesky_events": events,
    }
