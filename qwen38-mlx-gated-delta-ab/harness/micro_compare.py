#!/usr/bin/env python
"""Compare the pinned MLX GatedDeltaNet Metal kernel with the ops reference.

This script intentionally calls the two public implementation functions
directly.  It does not patch mlx-lm or construct model caches.  All tensors
are deterministic and the result file is updated after every completed test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import numpy as np
from mlx_lm.models import gated_delta as gd


SEED = 20260817
B = 1
HK = 16
HV = 48
DK = 128
DV = 128
DEFAULT_LENGTHS = (1, 2, 16, 128, 256, 512, 640, 768)
DEFAULT_CHUNK_LENGTHS = (512, 640, 768)
DEFAULT_CHUNKS = (256, 128)


def materialize(value: mx.array) -> np.ndarray:
    """Synchronize an MLX value and return a float32 NumPy copy."""

    mx.eval(value)
    mx.synchronize()
    try:
        return np.asarray(value, dtype=np.float32).copy()
    except (BaseException,):
        # NumPy's buffer importer does not understand MLX bfloat16 on this
        # Python build.  Convert on the MLX side before crossing the boundary.
        converted = value.astype(mx.float32)
        mx.eval(converted)
        mx.synchronize()
        return np.asarray(converted, dtype=np.float32).copy()


def tensor_hash(value: mx.array) -> str:
    array = materialize(value)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    left_flat = left.astype(np.float64, copy=False).ravel()
    right_flat = right.astype(np.float64, copy=False).ravel()
    left_norm = float(np.linalg.norm(left_flat))
    right_norm = float(np.linalg.norm(right_flat))
    denominator = left_norm * right_norm
    if denominator == 0.0:
        return None
    return float(np.dot(left_flat, right_flat) / denominator)


def error_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float | int | None]:
    delta = left.astype(np.float64, copy=False) - right.astype(np.float64, copy=False)
    finite_delta = delta[np.isfinite(delta)]
    if finite_delta.size:
        max_abs = float(np.max(np.abs(finite_delta)))
        mean_abs = float(np.mean(np.abs(finite_delta)))
        rms = float(math.sqrt(np.mean(np.square(finite_delta))))
        denominator = float(np.linalg.norm(right.astype(np.float64, copy=False)))
        relative = float(np.linalg.norm(delta) / denominator) if denominator else None
    else:
        max_abs = mean_abs = rms = relative = None
    return {
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "rms_error": rms,
        "relative_error": relative,
        "cosine": cosine(left, right),
        "nan_count": int(np.isnan(left).sum()),
        "inf_count": int(np.isinf(left).sum()),
    }


def diff_metrics(
    left: np.ndarray, right: np.ndarray, prefix: str
) -> dict[str, float | int | None]:
    metrics = error_metrics(left, right)
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def append_result(path: Path, result: dict[str, Any]) -> None:
    if path.exists():
        aggregate = json.loads(path.read_text(encoding="utf-8"))
    else:
        aggregate = {
            "model": "lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly",
            "mlx_lm_commit": "254d153fdeb6f150edd4fc5a54f9828638481fa8",
            "seed": SEED,
            "dimensions": {"B": B, "Hk": HK, "Hv": HV, "Dk": DK, "Dv": DV},
            "tests": [],
        }
    tests = [item for item in aggregate.get("tests", []) if item.get("test_id") != result["test_id"]]
    tests.append(result)
    aggregate["tests"] = tests
    aggregate["last_updated_test"] = result["test_id"]
    path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_inputs(length: int, dtype: mx.Dtype) -> dict[str, mx.array | int | str]:
    # q/k/v use the model's configured bfloat16 inference dtype.  The recurrent
    # state and the scalar gates use float32, matching gated_delta_update().
    q = (0.1 * mx.random.normal((B, length, HK, DK))).astype(dtype)
    k = (0.1 * mx.random.normal((B, length, HK, DK))).astype(dtype)
    v = (0.2 * mx.random.normal((B, length, HV, DV))).astype(dtype)
    g = mx.random.uniform(low=0.80, high=0.999, shape=(B, length, HV)).astype(mx.float32)
    beta = mx.random.uniform(low=0.10, high=0.90, shape=(B, length, HV)).astype(mx.float32)
    state = (0.01 * mx.random.normal((B, HV, DV, DK))).astype(mx.float32)
    for value in (q, k, v, g, beta, state):
        mx.eval(value)
    mx.synchronize()
    hashes = {
        "q": tensor_hash(q),
        "k": tensor_hash(k),
        "v": tensor_hash(v),
        "g": tensor_hash(g),
        "beta": tensor_hash(beta),
        "state": tensor_hash(state),
    }
    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "state": state,
        "sequence_length": length,
        "dtype": str(dtype),
        "input_hashes": hashes,
    }


def run_impl(
    implementation: str,
    inputs: dict[str, mx.array | int | str],
    state: mx.array | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    q = inputs["q"]
    k = inputs["k"]
    v = inputs["v"]
    g = inputs["g"]
    beta = inputs["beta"]
    initial_state = inputs["state"] if state is None else state
    if implementation == "metal":
        output, final_state = gd.gated_delta_kernel(q, k, v, g, beta, initial_state)  # type: ignore[arg-type]
    elif implementation == "ops":
        output, final_state = gd.gated_delta_ops(q, k, v, g, beta, initial_state)  # type: ignore[arg-type]
    else:
        raise ValueError(f"unknown implementation {implementation}")
    output_np = materialize(output)
    state_np = materialize(final_state)
    return output_np, state_np


def sliced_inputs(
    inputs: dict[str, mx.array | int | str], start: int, end: int
) -> dict[str, mx.array | int | str]:
    return {
        "q": inputs["q"][:, start:end],  # type: ignore[index]
        "k": inputs["k"][:, start:end],  # type: ignore[index]
        "v": inputs["v"][:, start:end],  # type: ignore[index]
        "g": inputs["g"][:, start:end],  # type: ignore[index]
        "beta": inputs["beta"][:, start:end],  # type: ignore[index]
        "state": inputs["state"],
        "sequence_length": end - start,
        "dtype": inputs["dtype"],
        "input_hashes": inputs["input_hashes"],
    }


def run_chunked(
    implementation: str,
    inputs: dict[str, mx.array | int | str],
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    total = int(inputs["sequence_length"])
    state = inputs["state"]  # type: ignore[assignment]
    outputs: list[np.ndarray] = []
    for start in range(0, total, chunk_size):
        end = min(total, start + chunk_size)
        part = sliced_inputs(inputs, start, end)
        output, state_np = run_impl(implementation, part, state=state)  # type: ignore[arg-type]
        outputs.append(output)
        state = mx.array(state_np, dtype=mx.float32)
        mx.eval(state)
        mx.synchronize()
    output_np = np.concatenate(outputs, axis=1)
    return output_np, np.asarray(state, dtype=np.float32)


def common_fields(
    test_id: str,
    inputs: dict[str, mx.array | int | str],
    implementation: str,
    repeat_index: int | None,
) -> dict[str, Any]:
    return {
        "test_id": test_id,
        "category": None,
        "sequence_length": int(inputs["sequence_length"]),
        "chunk_size": None,
        "implementation": implementation,
        "repeat_index": repeat_index,
        "dtype": str(inputs["dtype"]),
        "seed": SEED,
        "dimensions": {"B": B, "Hk": HK, "Hv": HV, "Dk": DK, "Dv": DV},
        "input_hashes": inputs["input_hashes"],
        "output_max_abs_error_vs_ops": None,
        "output_mean_abs_error_vs_ops": None,
        "output_rms_error_vs_ops": None,
        "output_relative_error_vs_ops": None,
        "output_cosine_vs_ops": None,
        "state_max_abs_error_vs_ops": None,
        "state_mean_abs_error_vs_ops": None,
        "state_rms_error_vs_ops": None,
        "state_relative_error_vs_ops": None,
        "state_cosine_vs_ops": None,
        "nan_count": None,
        "inf_count": None,
        "output_nan_count": None,
        "output_inf_count": None,
        "state_nan_count": None,
        "state_inf_count": None,
        "deterministic_vs_repeat0": None,
        "output_max_abs_error_vs_repeat0": None,
        "state_max_abs_error_vs_repeat0": None,
        "output_max_abs_error_vs_whole": None,
        "output_mean_abs_error_vs_whole": None,
        "output_rms_error_vs_whole": None,
        "output_cosine_vs_whole": None,
        "state_max_abs_error_vs_whole": None,
        "state_mean_abs_error_vs_whole": None,
        "state_rms_error_vs_whole": None,
        "state_cosine_vs_whole": None,
        "duration_seconds": None,
        "failure_reason": None,
    }


def add_array_error(result: dict[str, Any], prefix: str, left: np.ndarray, right: np.ndarray) -> None:
    metrics = error_metrics(left, right)
    for key, value in metrics.items():
        result[f"{prefix}_{key}"] = value
    result[f"{prefix}_nan_count"] = int(np.isnan(left).sum())
    result[f"{prefix}_inf_count"] = int(np.isinf(left).sum())


def direct_tests(
    root: Path,
    results_path: Path,
    lengths: tuple[int, ...],
    dtype: mx.Dtype,
) -> dict[int, dict[str, Any]]:
    baselines: dict[int, dict[str, Any]] = {}
    for length in lengths:
        inputs = make_inputs(length, dtype)
        started = time.perf_counter()
        try:
            ops_output, ops_state = run_impl("ops", inputs)
            metal_output, metal_state = run_impl("metal", inputs)
            result = common_fields(f"direct-{str(dtype).replace(' ', '-')}-{length:04d}", inputs, "metal", 0)
            result["category"] = "direct_vs_ops"
            result["duration_seconds"] = time.perf_counter() - started
            add_array_error(result, "output_vs_ops", metal_output, ops_output)
            add_array_error(result, "state_vs_ops", metal_state, ops_state)
            result["output_max_abs_error_vs_ops"] = result.pop("output_vs_ops_max_abs_error")
            result["output_mean_abs_error_vs_ops"] = result.pop("output_vs_ops_mean_abs_error")
            result["output_rms_error_vs_ops"] = result.pop("output_vs_ops_rms_error")
            result["output_relative_error_vs_ops"] = result.pop("output_vs_ops_relative_error")
            result["output_cosine_vs_ops"] = result.pop("output_vs_ops_cosine")
            result["state_max_abs_error_vs_ops"] = result.pop("state_vs_ops_max_abs_error")
            result["state_mean_abs_error_vs_ops"] = result.pop("state_vs_ops_mean_abs_error")
            result["state_rms_error_vs_ops"] = result.pop("state_vs_ops_rms_error")
            result["state_relative_error_vs_ops"] = result.pop("state_vs_ops_relative_error")
            result["state_cosine_vs_ops"] = result.pop("state_vs_ops_cosine")
            result["nan_count"] = int(np.isnan(metal_output).sum() + np.isnan(metal_state).sum())
            result["inf_count"] = int(np.isinf(metal_output).sum() + np.isinf(metal_state).sum())
            result["output_nan_count"] = int(np.isnan(metal_output).sum())
            result["output_inf_count"] = int(np.isinf(metal_output).sum())
            result["state_nan_count"] = int(np.isnan(metal_state).sum())
            result["state_inf_count"] = int(np.isinf(metal_state).sum())
            baselines[length] = {
                "inputs": inputs,
                "ops_output": ops_output,
                "ops_state": ops_state,
                "metal_output": metal_output,
                "metal_state": metal_state,
            }
        except BaseException as error:
            result = common_fields(f"direct-{str(dtype).replace(' ', '-')}-{length:04d}", inputs, "metal", 0)
            result["category"] = "direct_vs_ops"
            result["duration_seconds"] = time.perf_counter() - started
            result["failure_reason"] = f"{type(error).__name__}: {error}"
        append_result(results_path, result)
        print(json.dumps(result, sort_keys=True))
        mx.clear_cache()
    return baselines


def deterministic_tests(root: Path, results_path: Path, baselines: dict[int, dict[str, Any]]) -> None:
    for length in (128, 512, 768):
        if length not in baselines:
            continue
        inputs = baselines[length]["inputs"]
        for implementation in ("metal", "ops"):
            repeat_outputs: list[np.ndarray] = []
            repeat_states: list[np.ndarray] = []
            for repeat in range(10):
                started = time.perf_counter()
                result = common_fields(f"determinism-{implementation}-{length:04d}-r{repeat}", inputs, implementation, repeat)
                result["category"] = "determinism"
                try:
                    output, state = run_impl(implementation, inputs)
                    reference_output = baselines[length]["ops_output"]
                    reference_state = baselines[length]["ops_state"]
                    add_array_error(result, "output_vs_ops", output, reference_output)
                    add_array_error(result, "state_vs_ops", state, reference_state)
                    result["output_max_abs_error_vs_ops"] = result.pop("output_vs_ops_max_abs_error")
                    result["output_mean_abs_error_vs_ops"] = result.pop("output_vs_ops_mean_abs_error")
                    result["output_rms_error_vs_ops"] = result.pop("output_vs_ops_rms_error")
                    result["output_relative_error_vs_ops"] = result.pop("output_vs_ops_relative_error")
                    result["output_cosine_vs_ops"] = result.pop("output_vs_ops_cosine")
                    result["state_max_abs_error_vs_ops"] = result.pop("state_vs_ops_max_abs_error")
                    result["state_mean_abs_error_vs_ops"] = result.pop("state_vs_ops_mean_abs_error")
                    result["state_rms_error_vs_ops"] = result.pop("state_vs_ops_rms_error")
                    result["state_relative_error_vs_ops"] = result.pop("state_vs_ops_relative_error")
                    result["state_cosine_vs_ops"] = result.pop("state_vs_ops_cosine")
                    result["nan_count"] = int(np.isnan(output).sum() + np.isnan(state).sum())
                    result["inf_count"] = int(np.isinf(output).sum() + np.isinf(state).sum())
                    result["output_nan_count"] = int(np.isnan(output).sum())
                    result["output_inf_count"] = int(np.isinf(output).sum())
                    result["state_nan_count"] = int(np.isnan(state).sum())
                    result["state_inf_count"] = int(np.isinf(state).sum())
                    repeat_outputs.append(output)
                    repeat_states.append(state)
                    if repeat == 0:
                        result["deterministic_vs_repeat0"] = True
                    else:
                        output_delta = error_metrics(output, repeat_outputs[0])
                        state_delta = error_metrics(state, repeat_states[0])
                        result["deterministic_vs_repeat0"] = (
                            output_delta["max_abs_error"] == 0.0
                            and state_delta["max_abs_error"] == 0.0
                        )
                        result["output_max_abs_error_vs_repeat0"] = output_delta["max_abs_error"]
                        result["state_max_abs_error_vs_repeat0"] = state_delta["max_abs_error"]
                except BaseException as error:
                    result["failure_reason"] = f"{type(error).__name__}: {error}"
                result["duration_seconds"] = time.perf_counter() - started
                append_result(results_path, result)
                print(json.dumps(result, sort_keys=True))
                mx.clear_cache()


def chunk_tests(results_path: Path, baselines: dict[int, dict[str, Any]]) -> None:
    for length in DEFAULT_CHUNK_LENGTHS:
        if length not in baselines:
            continue
        inputs = baselines[length]["inputs"]
        for implementation in ("metal", "ops"):
            whole_output = baselines[length][f"{implementation}_output"]
            whole_state = baselines[length][f"{implementation}_state"]
            for chunk_size in DEFAULT_CHUNKS:
                started = time.perf_counter()
                result = common_fields(
                    f"chunk-{implementation}-{length:04d}-s{chunk_size}",
                    inputs,
                    implementation,
                    0,
                )
                result["category"] = "chunk_equivalence"
                result["chunk_size"] = chunk_size
                try:
                    chunk_output, chunk_state = run_chunked(implementation, inputs, chunk_size)
                    add_array_error(result, "output_vs_whole", chunk_output, whole_output)
                    add_array_error(result, "state_vs_whole", chunk_state, whole_state)
                    result["output_max_abs_error_vs_whole"] = result.pop("output_vs_whole_max_abs_error")
                    result["output_mean_abs_error_vs_whole"] = result.pop("output_vs_whole_mean_abs_error")
                    result["output_rms_error_vs_whole"] = result.pop("output_vs_whole_rms_error")
                    result["output_cosine_vs_whole"] = result.pop("output_vs_whole_cosine")
                    result["state_max_abs_error_vs_whole"] = result.pop("state_vs_whole_max_abs_error")
                    result["state_mean_abs_error_vs_whole"] = result.pop("state_vs_whole_mean_abs_error")
                    result["state_rms_error_vs_whole"] = result.pop("state_vs_whole_rms_error")
                    result["state_cosine_vs_whole"] = result.pop("state_vs_whole_cosine")
                    result["nan_count"] = int(np.isnan(chunk_output).sum() + np.isnan(chunk_state).sum())
                    result["inf_count"] = int(np.isinf(chunk_output).sum() + np.isinf(chunk_state).sum())
                    result["deterministic_vs_repeat0"] = None
                except BaseException as error:
                    result["failure_reason"] = f"{type(error).__name__}: {error}"
                result["duration_seconds"] = time.perf_counter() - started
                append_result(results_path, result)
                print(json.dumps(result, sort_keys=True))
                mx.clear_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="qwen38-mlx-gated-delta-ab")
    parser.add_argument("--stage", choices=("all", "direct", "determinism", "chunks"), default="all")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    results_path = root / "results.json"
    dtype = getattr(mx, args.dtype)
    mx.set_default_device(mx.gpu)
    mx.random.seed(SEED)
    print(f"mlx_version={mx.__version__ if hasattr(mx, '__version__') else 'unknown'}")
    print(f"default_device={mx.default_device()}")
    print(f"metal_available={mx.metal.is_available()}")
    print(f"seed={SEED}")
    print(f"dtype={dtype}")
    print(f"dimensions=B{B},Hk{HK},Hv{HV},Dk{DK},Dv{DV}")

    baselines: dict[int, dict[str, Any]] = {}
    if args.stage in ("all", "direct", "determinism", "chunks"):
        # Direct tests are needed to populate the shared deterministic input
        # tensors for later stages.  Re-running them is cheap compared with
        # loading the 27B checkpoint, and keeps each invocation self-contained.
        baselines = direct_tests(root, results_path, DEFAULT_LENGTHS, dtype)
    if args.stage in ("all", "determinism"):
        deterministic_tests(root, results_path, baselines)
    if args.stage in ("all", "chunks"):
        chunk_tests(results_path, baselines)


if __name__ == "__main__":
    main()
