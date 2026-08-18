#!/usr/bin/env python
"""Determinism and dense-reference tests for real MLX QuantizedLinear layers."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from mlx_lm import load

from common import (
    DEFAULT_MODEL_PATH,
    MLX_LM_COMMIT,
    MODEL_ID,
    array_hash,
    comparison,
    materialize,
    module_metadata,
    resolve_module,
    stats,
    update_results,
)


DEFAULT_LENGTHS = (1, 16, 128, 512, 768)
FRESH_LENGTHS = (1, 128, 512, 768)


def make_input(sequence_length: int, input_dims: int, seed: int) -> mx.array:
    mx.random.seed(seed)
    value = mx.random.normal((1, sequence_length, input_dims)).astype(mx.bfloat16)
    mx.eval(value)
    mx.synchronize()
    return value


def module_forward(module: Any, value: mx.array) -> tuple[str, np.ndarray]:
    output = module(value)
    output_hash = array_hash(output)
    output_np = materialize(output)
    return output_hash, output_np


def dense_reference(module: Any, value: mx.array) -> tuple[str, np.ndarray]:
    """Reproduce QuantizedLinear.__call__ using its exact affine parameters."""

    dequantized = mx.dequantize(
        module["weight"],
        scales=module["scales"],
        biases=module.get("biases"),
        group_size=module.group_size,
        bits=module.bits,
        mode=module.mode,
        dtype=mx.float32,
    )
    dense = mx.matmul(value.astype(mx.float32), dequantized.T)
    if module.get("bias") is not None:
        dense = dense + module["bias"].astype(mx.float32)
    dense_hash = array_hash(dense)
    dense_np = materialize(dense)
    return dense_hash, dense_np


def layer_result(
    *,
    test_id: str,
    module_path: str,
    layer_label: str,
    module: Any,
    value: mx.array,
    sequence_length: int,
    process_index: int | None,
    repeat_index: int,
    output_hash: str,
    output_np: np.ndarray,
    reference_hash: str | None = None,
    reference_np: np.ndarray | None = None,
    input_seed: int,
) -> dict[str, Any]:
    metadata = module_metadata(module)
    output_stats = stats(output_np)
    result: dict[str, Any] = {
        "test_id": test_id,
        "model": MODEL_ID,
        "mlx_lm_commit": MLX_LM_COMMIT,
        "process_index": process_index,
        "repeat_index": repeat_index,
        "layer_path": module_path,
        "layer_label": layer_label,
        "layer_type": type(module).__name__,
        "bits": metadata["bits"],
        "group_size": metadata["group_size"],
        "mode": metadata["mode"],
        "input_shape": list(map(int, value.shape)),
        "input_hash": array_hash(value),
        "input_seed": input_seed,
        "weight_hash": metadata["weight_hash"],
        "scales_hash": metadata["scales_hash"],
        "biases_hash": metadata["biases_hash"],
        "quantized_output_hash": output_hash,
        "quantized_sum": output_stats["sum"],
        "quantized_mean": output_stats["mean"],
        "quantized_std": output_stats["std"],
        "quantized_min": output_stats["min"],
        "quantized_max": output_stats["max"],
        "quantized_l2": output_stats["l2"],
        "nan_count": output_stats["nan_count"],
        "inf_count": output_stats["inf_count"],
        "reference_output_hash": reference_hash,
        "reference_sum": None,
        "reference_mean": None,
        "reference_std": None,
        "reference_l2": None,
        "max_abs_error": None,
        "mean_abs_error": None,
        "rms_error": None,
        "relative_l2_error": None,
        "cosine_similarity": None,
        "reference_nan_count": None,
        "reference_inf_count": None,
        "deterministic_same_process": None,
        "deterministic_cross_process": None,
    }
    if reference_np is not None:
        reference_stats = stats(reference_np)
        result.update(
            {
                "reference_sum": reference_stats["sum"],
                "reference_mean": reference_stats["mean"],
                "reference_std": reference_stats["std"],
                "reference_l2": reference_stats["l2"],
                "reference_nan_count": reference_stats["nan_count"],
                "reference_inf_count": reference_stats["inf_count"],
                **comparison(output_np, reference_np),
            }
        )
    return result


def run_same_process(args: argparse.Namespace) -> None:
    model, _ = load(args.model_path)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    path = args.layer_path or manifest["selected"][args.layer_label]
    module = resolve_module(model, path)
    metadata = module_metadata(module)
    results_path = Path(args.results)
    baseline_by_length: dict[int, np.ndarray] = {}
    for sequence_length in DEFAULT_LENGTHS:
        seed = args.seed + sequence_length
        value = make_input(sequence_length, metadata["input_dims"], seed)
        started = time.perf_counter()
        for repeat_index in range(args.repeats):
            output_hash, output_np = module_forward(module, value)
            result = layer_result(
                test_id=f"same-process-{args.layer_label}-t{sequence_length:04d}-r{repeat_index:02d}",
                module_path=path,
                layer_label=args.layer_label,
                module=module,
                value=value,
                sequence_length=sequence_length,
                process_index=0,
                repeat_index=repeat_index,
                output_hash=output_hash,
                output_np=output_np,
                input_seed=seed,
            )
            if repeat_index == 0:
                baseline_by_length[sequence_length] = output_np
            else:
                delta = comparison(output_np, baseline_by_length[sequence_length])
                result["difference_vs_repeat0"] = delta
            result["elapsed_since_sequence_start_seconds"] = time.perf_counter() - started
            result["deterministic_same_process"] = (
                repeat_index == 0
                or result["difference_vs_repeat0"]["max_abs_error"] == 0.0
            )
            update_results(results_path, result, collection="quantized_layer_tests")
            print(json.dumps(result, sort_keys=True))
        mx.clear_cache()


def run_fresh_suite(args: argparse.Namespace) -> None:
    model, _ = load(args.model_path)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    path = args.layer_path or manifest["selected"][args.layer_label]
    module = resolve_module(model, path)
    metadata = module_metadata(module)
    results_path = Path(args.results)
    for sequence_length in FRESH_LENGTHS:
        seed = args.seed + sequence_length
        value = make_input(sequence_length, metadata["input_dims"], seed)
        output_hash, output_np = module_forward(module, value)
        result = layer_result(
            test_id=f"fresh-process-{args.layer_label}-p{args.process_index:02d}-t{sequence_length:04d}",
            module_path=path,
            layer_label=args.layer_label,
            module=module,
            value=value,
            sequence_length=sequence_length,
            process_index=args.process_index,
            repeat_index=0,
            output_hash=output_hash,
            output_np=output_np,
            input_seed=seed,
        )
        update_results(results_path, result, collection="quantized_layer_tests")
        print(json.dumps(result, sort_keys=True))
        mx.clear_cache()


def run_reference(args: argparse.Namespace) -> None:
    model, _ = load(args.model_path)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    path = args.layer_path or manifest["selected"][args.layer_label]
    module = resolve_module(model, path)
    metadata = module_metadata(module)
    results_path = Path(args.results)
    for sequence_length in DEFAULT_LENGTHS:
        seed = args.seed + sequence_length
        value = make_input(sequence_length, metadata["input_dims"], seed)
        output_hash, output_np = module_forward(module, value)
        reference_hash, reference_np = dense_reference(module, value)
        result = layer_result(
            test_id=f"dense-reference-{args.layer_label}-p{args.process_index:02d}-t{sequence_length:04d}",
            module_path=path,
            layer_label=args.layer_label,
            module=module,
            value=value,
            sequence_length=sequence_length,
            process_index=args.process_index,
            repeat_index=0,
            output_hash=output_hash,
            output_np=output_np,
            reference_hash=reference_hash,
            reference_np=reference_np,
            input_seed=seed,
        )
        update_results(results_path, result, collection="quantized_layer_tests")
        print(json.dumps(result, sort_keys=True))
        mx.clear_cache()


def run_reference_repeats(args: argparse.Namespace) -> None:
    model, _ = load(args.model_path)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    path = args.layer_path or manifest["selected"][args.layer_label]
    module = resolve_module(model, path)
    metadata = module_metadata(module)
    results_path = Path(args.results)
    for sequence_length in (512, 768):
        seed = args.seed + sequence_length
        value = make_input(sequence_length, metadata["input_dims"], seed)
        baseline: np.ndarray | None = None
        for repeat_index in range(args.repeats):
            reference_hash, reference_np = dense_reference(module, value)
            output_hash = reference_hash
            result = layer_result(
                test_id=f"dense-reference-repeat-{args.layer_label}-p{args.process_index:02d}-t{sequence_length:04d}-r{repeat_index:02d}",
                module_path=path,
                layer_label=args.layer_label,
                module=module,
                value=value,
                sequence_length=sequence_length,
                process_index=args.process_index,
                repeat_index=repeat_index,
                output_hash=output_hash,
                output_np=reference_np,
                input_seed=seed,
            )
            if baseline is None:
                baseline = reference_np
            else:
                result["difference_vs_repeat0"] = comparison(reference_np, baseline)
                result["deterministic_same_process"] = (
                    result["difference_vs_repeat0"]["max_abs_error"] == 0.0
                )
            result["reference_kind"] = "dequantize_then_dense"
            update_results(results_path, result, collection="reference_repeat_tests")
            print(json.dumps(result, sort_keys=True))
        mx.clear_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--manifest", default="qwen38-mlx-quantmatmul/module-manifest.json")
    parser.add_argument("--layer-label", default="gated_delta_in_proj_qkv")
    parser.add_argument("--layer-path")
    parser.add_argument("--mode", choices=("same-process", "fresh-suite", "reference", "reference-repeats"), required=True)
    parser.add_argument("--process-index", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=38027)
    parser.add_argument("--results", default="qwen38-mlx-quantmatmul/results.json")
    args = parser.parse_args()
    mx.set_default_device(mx.gpu)
    if args.mode == "same-process":
        run_same_process(args)
    elif args.mode == "fresh-suite":
        run_fresh_suite(args)
    elif args.mode == "reference":
        run_reference(args)
    else:
        run_reference_repeats(args)


if __name__ == "__main__":
    main()
