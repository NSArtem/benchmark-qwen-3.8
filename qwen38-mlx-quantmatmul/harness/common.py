#!/usr/bin/env python
"""Shared helpers for the Qwen3.8 MLX quantized-matmul diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np


MODEL_ID = "lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly"
MLX_LM_COMMIT = "254d153fdeb6f150edd4fc5a54f9828638481fa8"
DEFAULT_MODEL_PATH = (
    "/Users/artem/.cache/huggingface/hub/models--lukaskremla--"
    "Qwen3.8-27B-3bit-MLX-TextOnly/snapshots/"
    "c98bba5926f51fec1c8d8737e577221673f524d7"
)


def synchronize() -> None:
    """Evaluate pending work and wait for the device before reading values."""

    mx.synchronize()


def materialize(value: mx.array, dtype: Any = np.float32) -> np.ndarray:
    """Evaluate an MLX array and return an independent NumPy copy."""

    mx.eval(value)
    synchronize()
    try:
        return np.asarray(value, dtype=dtype).copy()
    except Exception:
        converted = value.astype(mx.float32)
        mx.eval(converted)
        synchronize()
        return np.asarray(converted, dtype=dtype).copy()


def exact_numpy(value: mx.array) -> np.ndarray:
    """Return a stable host copy, preserving integer dtypes where possible."""

    mx.eval(value)
    synchronize()
    try:
        return np.asarray(value).copy()
    except Exception:
        converted = value.astype(mx.float32)
        mx.eval(converted)
        synchronize()
        return np.asarray(converted, dtype=np.float32).copy()


def array_hash(value: mx.array) -> str:
    """Hash evaluated shape, dtype, and host bytes of an MLX array."""

    array = exact_numpy(value)
    descriptor = f"{array.dtype}:{array.shape}:".encode("utf-8")
    return hashlib.sha256(descriptor + array.tobytes(order="C")).hexdigest()


def input_hash(value: mx.array) -> str:
    return array_hash(value)


def stats(value: np.ndarray) -> dict[str, Any]:
    """Compact numerical statistics for an already materialized array."""

    array = np.asarray(value, dtype=np.float64)
    finite = np.isfinite(array)
    return {
        "sum": float(np.sum(array)) if array.size else 0.0,
        "mean": float(np.mean(array)) if array.size else 0.0,
        "std": float(np.std(array)) if array.size else 0.0,
        "min": float(np.min(array)) if array.size else None,
        "max": float(np.max(array)) if array.size else None,
        "l2": float(np.linalg.norm(array)) if array.size else 0.0,
        "nan_count": int(np.isnan(array).sum()),
        "inf_count": int(np.isinf(array).sum()),
        "nonfinite_count": int((~finite).sum()),
    }


def comparison(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    """Compare two arrays after converting to float64."""

    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    delta = a - b
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    denominator = b_norm
    return {
        "max_abs_error": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "mean_abs_error": float(np.mean(np.abs(delta))) if delta.size else 0.0,
        "rms_error": float(np.sqrt(np.mean(np.square(delta)))) if delta.size else 0.0,
        "relative_l2_error": float(np.linalg.norm(delta) / denominator) if denominator else None,
        "cosine_similarity": (
            float(np.sum(a * b) / (a_norm * b_norm)) if a_norm and b_norm else None
        ),
        "nan_count_left": int(np.isnan(a).sum()),
        "inf_count_left": int(np.isinf(a).sum()),
        "nan_count_right": int(np.isnan(b).sum()),
        "inf_count_right": int(np.isinf(b).sum()),
    }


def resolve_module(root: Any, path: str) -> Any:
    """Resolve a dotted MLX named_modules path through modules and lists."""

    current = root
    if not path:
        return current
    for part in path.split("."):
        if isinstance(current, (list, tuple)):
            current = current[int(part)]
        elif hasattr(current, part):
            current = getattr(current, part)
        else:
            current = current[part]
    return current


def module_array(module: Any, name: str) -> mx.array | None:
    """Get a module parameter without confusing absent ordinary bias."""

    try:
        value = module.get(name)
    except AttributeError:
        value = None
    return value


def module_metadata(module: Any) -> dict[str, Any]:
    weight = module["weight"]
    scales = module["scales"]
    biases = module.get("biases")
    linear_bias = module.get("bias")
    bits = int(module.bits)
    packed_rows, packed_cols = map(int, weight.shape)
    input_dims = packed_cols * 32 // bits
    return {
        "class": type(module).__name__,
        "input_dims": input_dims,
        "output_dims": packed_rows,
        "group_size": int(module.group_size),
        "bits": bits,
        "mode": str(module.mode),
        "weight_shape": list(map(int, weight.shape)),
        "weight_dtype": str(weight.dtype),
        "scales_shape": list(map(int, scales.shape)),
        "scales_dtype": str(scales.dtype),
        "biases_shape": list(map(int, biases.shape)) if biases is not None else None,
        "biases_dtype": str(biases.dtype) if biases is not None else None,
        "linear_bias_shape": list(map(int, linear_bias.shape)) if linear_bias is not None else None,
        "linear_bias_dtype": str(linear_bias.dtype) if linear_bias is not None else None,
        "weight_hash": array_hash(weight),
        "scales_hash": array_hash(scales),
        "biases_hash": array_hash(biases) if biases is not None else None,
        "linear_bias_hash": array_hash(linear_bias) if linear_bias is not None else None,
    }


def update_results(path: Path, result: dict[str, Any], collection: str = "tests") -> None:
    """Persist one result immediately while replacing only the same test id."""

    if path.exists():
        aggregate = json.loads(path.read_text(encoding="utf-8"))
    else:
        aggregate = {"model": MODEL_ID, "mlx_lm_commit": MLX_LM_COMMIT}
    entries = [item for item in aggregate.get(collection, []) if item.get("test_id") != result.get("test_id")]
    entries.append(result)
    aggregate[collection] = entries
    aggregate["last_updated_test"] = result.get("test_id")
    path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
