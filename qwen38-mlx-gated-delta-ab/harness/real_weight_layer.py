#!/usr/bin/env python
"""Compare one real-weight Qwen GatedDeltaNet layer in Metal and ops modes."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from mlx_lm import load
from mlx_lm.models import gated_delta as gd
from mlx_lm.models import qwen3_5 as q35
from mlx_lm.models.base import create_ssm_mask
from mlx_lm.models.cache import ArraysCache


MODEL_ID = "lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly"
MLX_LM_COMMIT = "254d153fdeb6f150edd4fc5a54f9828638481fa8"


def materialize(value: mx.array) -> np.ndarray:
    mx.eval(value)
    mx.synchronize()
    try:
        return np.asarray(value, dtype=np.float32).copy()
    except Exception:
        converted = value.astype(mx.float32)
        mx.eval(converted)
        mx.synchronize()
        return np.asarray(converted, dtype=np.float32).copy()


def metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float | int | None]:
    delta = left.astype(np.float64) - right.astype(np.float64)
    denominator = float(np.linalg.norm(right.astype(np.float64)))
    left_norm = float(np.linalg.norm(left.astype(np.float64)))
    right_norm = float(np.linalg.norm(right.astype(np.float64)))
    return {
        "max_abs_error": float(np.max(np.abs(delta))),
        "mean_abs_error": float(np.mean(np.abs(delta))),
        "rms_error": float(np.sqrt(np.mean(np.square(delta)))),
        "relative_error": float(np.linalg.norm(delta) / denominator) if denominator else None,
        "cosine": float(np.sum(left.astype(np.float64) * right.astype(np.float64)) / (left_norm * right_norm))
        if left_norm and right_norm
        else None,
        "nan_count": int(np.isnan(left).sum()),
        "inf_count": int(np.isinf(left).sum()),
    }


def chat_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    elif hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    if encoded and isinstance(encoded[0], (list, tuple)):
        encoded = encoded[0]
    return [int(value) for value in encoded]


def update_results(path: Path, result: dict[str, Any]) -> None:
    aggregate = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "model": MODEL_ID,
        "mlx_lm_commit": MLX_LM_COMMIT,
        "tests": [],
    }
    tests = [item for item in aggregate.get("tests", []) if item.get("test_id") != result["test_id"]]
    tests.append(result)
    aggregate["tests"] = tests
    aggregate["last_updated_test"] = result["test_id"]
    path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_mode(layer: Any, hidden: mx.array, mode: str) -> tuple[np.ndarray, np.ndarray, int | None]:
    original = gd.gated_delta_update
    calls: dict[str, int] = {"count": 0}
    if mode == "ops":
        def force_ops(*args: Any, **kwargs: Any) -> Any:
            calls["count"] += 1
            kwargs["use_kernel"] = False
            return original(*args, **kwargs)

        q35.gated_delta_update = force_ops
    else:
        q35.gated_delta_update = original

    cache = ArraysCache(size=2)
    mask = create_ssm_mask(hidden, cache)
    inputs = layer.input_layernorm(hidden)
    output = layer.linear_attn(inputs, mask=mask, cache=cache)
    output_np = materialize(output)
    state_np = materialize(cache[1])
    return output_np, state_np, calls["count"] if mode == "ops" else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--root", default="qwen38-mlx-gated-delta-ab")
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    mx.set_default_device(mx.gpu)
    model, tokenizer = load(args.model_path)
    layer = model.layers[0]
    print(f"model={MODEL_ID}")
    print(f"mlx_lm_commit={MLX_LM_COMMIT}")
    print(f"layer_type={type(layer).__name__}")
    print(f"layer_training={layer.training}")
    print(f"qwen3_5_symbol_is_gd_symbol={q35.gated_delta_update is gd.gated_delta_update}")

    for target, prompt_file in ((128, "qwen38-mlx-threshold/prompts/0512.txt"), (512, "qwen38-mlx-threshold/prompts/0512.txt"), (768, "qwen38-mlx-threshold/prompts/0768.txt")):
        ids = chat_ids(tokenizer, Path(prompt_file).read_text(encoding="utf-8"))[:target]
        hidden = model.model.embed_tokens(mx.array([ids]))
        mx.eval(hidden)
        mx.synchronize()
        input_hash = hashlib.sha256(materialize(hidden).tobytes()).hexdigest()
        started = time.perf_counter()
        metal_output, metal_state, _ = run_mode(layer, hidden, "metal")
        ops_output, ops_state, ops_calls = run_mode(layer, hidden, "ops")
        output_metrics = metrics(metal_output, ops_output)
        state_metrics = metrics(metal_state, ops_state)
        result = {
            "test_id": f"real-weight-layer0-{target:04d}",
            "category": "real_weight_layer",
            "model": MODEL_ID,
            "mlx_lm_commit": MLX_LM_COMMIT,
            "sequence_length": target,
            "prompt_file": prompt_file,
            "actual_input_tokens": len(ids),
            "layer_index": 0,
            "layer_type": type(layer).__name__,
            "input_hidden_hash": input_hash,
            "implementation_comparison": "metal_vs_ops",
            "output_max_abs_error_vs_ops": output_metrics["max_abs_error"],
            "output_mean_abs_error_vs_ops": output_metrics["mean_abs_error"],
            "output_rms_error_vs_ops": output_metrics["rms_error"],
            "output_relative_error_vs_ops": output_metrics["relative_error"],
            "output_cosine_vs_ops": output_metrics["cosine"],
            "state_max_abs_error_vs_ops": state_metrics["max_abs_error"],
            "state_mean_abs_error_vs_ops": state_metrics["mean_abs_error"],
            "state_rms_error_vs_ops": state_metrics["rms_error"],
            "state_relative_error_vs_ops": state_metrics["relative_error"],
            "state_cosine_vs_ops": state_metrics["cosine"],
            "nan_count": int(output_metrics["nan_count"] + state_metrics["nan_count"]),
            "inf_count": int(output_metrics["inf_count"] + state_metrics["inf_count"]),
            "ops_wrapper_call_count": ops_calls,
            "duration_seconds": time.perf_counter() - started,
            "failure_reason": None,
        }
        update_results(root / "results.json", result)
        print(json.dumps(result, sort_keys=True))
        mx.clear_cache()


if __name__ == "__main__":
    main()
