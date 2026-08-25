#!/usr/bin/env python3
"""Bounded one-module DWQ-style refinement on the selected g128 student.

This is deliberately a local streamed experiment, not upstream whole-model
DWQ: the g128 student remains resident, one BF16 source tensor is loaded, and
only the selected module's quantized scales/biases are optimized against local
source outputs for captured student activations.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

ROOT = Path(__file__).resolve().parents[3]
for relative in ("research/harness",):
    path = ROOT / relative
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from convert_qwen38_textonly_g128 import output_key, source_weight_map  # noqa: E402


GROUP_SIZE = 128
BITS = 3


def memory_snapshot(label: str) -> dict[str, Any]:
    try:
        mx.synchronize()
    except BaseException:
        pass
    active = int(mx.get_active_memory())
    cache = int(mx.get_cache_memory())
    peak = int(mx.get_peak_memory())
    return {
        "label": label,
        "active_bytes": active,
        "cache_bytes": cache,
        "active_plus_cache_bytes": active + cache,
        "peak_bytes": peak,
        "active_gib": active / (1024**3),
        "active_plus_cache_gib": (active + cache) / (1024**3),
        "peak_gib": peak / (1024**3),
        "rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }


def os_snapshot() -> dict[str, str]:
    result: dict[str, str] = {}
    for name, command in (("memory_pressure", ["memory_pressure", "-Q"]), ("vm_swapusage", ["sysctl", "vm.swapusage"])):
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=10)
            result[name] = (completed.stdout + completed.stderr).strip()
        except BaseException as exc:
            result[name] = f"{type(exc).__name__}: {exc}"
    return result


def read_manifest(path: Path, case_ids: list[str]) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_id = {item["case_id"]: item for item in payload["cases"]}
    missing = [case_id for case_id in case_ids if case_id not in by_id]
    if missing:
        raise ValueError(f"missing teacher cases: {missing}")
    return [by_id[case_id] for case_id in case_ids]


def mean_kl(teacher_logits: Any, candidate_logits: Any) -> float:
    teacher = teacher_logits[:-1].astype(mx.float32)
    candidate = candidate_logits[:-1].astype(mx.float32)
    teacher_logp = nn.log_softmax(teacher, axis=-1)
    candidate_logp = nn.log_softmax(candidate, axis=-1)
    value = mx.mean(mx.sum(mx.exp(teacher_logp) * (teacher_logp - candidate_logp), axis=-1))
    mx.eval(value)
    return float(value.item())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--teacher-manifest", required=True)
    parser.add_argument("--target-path", default="language_model.model.layers.60.linear_attn.in_proj_a")
    parser.add_argument("--output", required=True)
    parser.add_argument("--partial-output", required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    args = parser.parse_args()

    started = time.perf_counter()
    case_ids = ["cal-coding-01", "cal-instruction-01", "cal-reasoning-01"]
    manifest_path = Path(args.teacher_manifest).resolve()
    cases = read_manifest(manifest_path, case_ids)
    result: dict[str, Any] = {
        "experiment_id": "P6-DWQ-BOUNDED-G128-LAYER60-INPROJ-A",
        "phase": 6,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": None,
        "algorithm_note": "streamed one-module local scale/bias refinement against source outputs; not upstream whole-model DWQ",
        "model_path": str(Path(args.model).resolve()),
        "source_path": str(Path(args.source).resolve()),
        "teacher_manifest": str(manifest_path),
        "teacher_calibration_sha256": json.loads(manifest_path.read_text(encoding="utf-8"))["calibration_sha256"],
        "case_ids": case_ids,
        "target_path": args.target_path,
        "group_size": GROUP_SIZE,
        "bits": BITS,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "full_teacher_model_resident": False,
        "full_source_model_resident": False,
        "full_student_model_resident": True,
        "wired_limit_calls": 0,
        "memory": [],
        "os_memory_before": os_snapshot(),
        "exception": None,
        "safety_stop_reason": None,
    }
    model = tokenizer = None
    original_call = None
    target_module = None
    try:
        from mlx_lm.utils import load

        model, tokenizer = load(args.model)
        result["memory"].append(memory_snapshot("after_student_load"))
        modules = dict(model.named_modules())
        target_module = modules.get(args.target_path)
        if target_module is None or not isinstance(target_module, nn.QuantizedLinear):
            raise TypeError(f"target is not a QuantizedLinear: {args.target_path}")
        module_bias_value = getattr(target_module, "bias", None)
        if module_bias_value is not None:
            module_bias = module_bias_value.astype(mx.float32)
        else:
            module_bias = None
        qweight = target_module.weight
        original_scales = target_module.scales
        original_biases = target_module.biases
        if original_biases is None:
            raise ValueError("affine g128 target unexpectedly has no quantization biases")
        params = {
            "scales": original_scales.astype(mx.float32),
            "biases": original_biases.astype(mx.float32),
        }
        mx.eval(qweight, params)
        result["student_module_payload_bytes"] = int(qweight.nbytes + original_scales.nbytes + original_biases.nbytes)
        result["student_module_shape"] = list(original_scales.shape)

        source_dir = Path(args.source).resolve()
        weight_map = source_weight_map(source_dir)
        source_key = next(
            (key for key in weight_map if key.endswith(".weight") and output_key(key) == f"{args.target_path}.weight"),
            None,
        )
        if source_key is None:
            raise KeyError(f"source tensor for {args.target_path} not found")
        source_file = source_dir / weight_map[source_key]
        loaded_source = mx.load(str(source_file))
        source_weight = loaded_source[source_key].astype(mx.float32)
        mx.eval(source_weight)
        result["source_key"] = source_key
        result["source_file"] = str(source_file)
        result["source_weight_bytes"] = int(source_weight.nbytes)
        del loaded_source
        gc.collect()
        mx.clear_cache()
        result["memory"].append(memory_snapshot("after_one_source_tensor"))

        captured: list[Any] = []
        original_call = nn.QuantizedLinear.__call__

        def capture_call(module: Any, x: Any) -> Any:
            value = original_call(module, x)
            if module is target_module:
                captured.append(x)
            return value

        nn.QuantizedLinear.__call__ = capture_call
        activation_samples: list[Any] = []
        target_samples: list[Any] = []
        baseline_final_logits = None
        first_teacher_logits = None
        for case_index, case in enumerate(cases):
            captured.clear()
            inputs = mx.array([case["token_ids"]])
            logits = model(inputs)
            mx.eval(logits)
            if not captured:
                raise RuntimeError(f"target module was not called for {case['case_id']}")
            activation = captured[-1]
            mx.eval(activation)
            rows = min(128, int(activation.shape[-2]))
            sample = activation[0, -rows:, :].astype(mx.float32)
            teacher_local = sample @ source_weight.T
            mx.eval(sample, teacher_local)
            activation_samples.append(sample)
            target_samples.append(teacher_local)
            if case_index == 0:
                baseline_final_logits = logits[0].astype(mx.float16)
                teacher_payload = mx.load(case["logits_path"])
                first_teacher_logits = teacher_payload["logits"].astype(mx.float16)
                mx.eval(baseline_final_logits, first_teacher_logits)
                del teacher_payload
            del logits, inputs, activation
            mx.clear_cache()
        nn.QuantizedLinear.__call__ = original_call
        original_call = None
        result["activation_rows"] = sum(int(item.shape[0]) for item in activation_samples)
        x_all = mx.concatenate(activation_samples, axis=0)
        y_all = mx.concatenate(target_samples, axis=0)
        mx.eval(x_all, y_all)

        def loss_fn(current: dict[str, Any]) -> Any:
            student_local = mx.quantized_matmul(
                x_all,
                qweight,
                scales=current["scales"],
                biases=current["biases"],
                transpose=True,
                group_size=GROUP_SIZE,
                bits=BITS,
                mode="affine",
            )
            if module_bias is not None:
                student_local = student_local + module_bias
            return mx.mean((student_local - y_all) ** 2)

        value_and_grad = mx.value_and_grad(loss_fn)
        initial_loss = loss_fn(params)
        mx.eval(initial_loss)
        loss_trace = [float(initial_loss.item())]
        for step in range(args.steps):
            loss, grads = value_and_grad(params)
            mx.eval(loss, grads)
            params = {name: params[name] - args.learning_rate * grads[name] for name in params}
            mx.eval(params)
            if step == 0 or (step + 1) % 10 == 0 or step + 1 == args.steps:
                checkpoint = loss_fn(params)
                mx.eval(checkpoint)
                loss_trace.append(float(checkpoint.item()))
        final_loss = loss_fn(params)
        mx.eval(final_loss)
        result["local_loss_before"] = loss_trace[0]
        result["local_loss_after"] = float(final_loss.item())
        result["local_loss_trace"] = loss_trace
        result["local_relative_loss_reduction"] = (loss_trace[0] - float(final_loss.item())) / max(loss_trace[0], 1e-12)
        result["memory"].append(memory_snapshot("after_local_scale_bias_optimization"))

        optimized_scales = params["scales"].astype(original_scales.dtype)
        optimized_biases = params["biases"].astype(original_biases.dtype)
        partial_path = Path(args.partial_output).resolve()
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(
            str(partial_path),
            {
                f"{args.target_path}.scales": optimized_scales,
                f"{args.target_path}.biases": optimized_biases,
            },
            metadata={"format": "mlx", "note": "partial bounded DWQ debug artifact; not a complete model"},
        )
        result["partial_output"] = str(partial_path)
        result["partial_output_bytes"] = partial_path.stat().st_size

        # Apply only the debug update in-memory and measure its end-to-end KL
        # effect on one calibration case. The main g128 directory is never
        # overwritten.
        target_module.scales = optimized_scales
        target_module.biases = optimized_biases
        inputs = mx.array([cases[0]["token_ids"]])
        updated_logits = model(inputs)[0].astype(mx.float16)
        mx.eval(updated_logits)
        result["end_to_end_kl_before"] = mean_kl(first_teacher_logits, baseline_final_logits)
        result["end_to_end_kl_after"] = mean_kl(first_teacher_logits, updated_logits)
        result["end_to_end_kl_delta"] = result["end_to_end_kl_before"] - result["end_to_end_kl_after"]
        result["end_to_end_teacher_case"] = cases[0]["case_id"]
        del updated_logits, inputs
        result["memory"].append(memory_snapshot("after_end_to_end_probe"))
        finite_local = all(math.isfinite(float(value)) for value in loss_trace) and math.isfinite(float(result["local_loss_after"]))
        if finite_local and result["local_loss_after"] < result["local_loss_before"]:
            result["classification"] = (
                "LOCAL_LOSS_AND_END_TO_END_KL_IMPROVED"
                if result["end_to_end_kl_after"] < result["end_to_end_kl_before"]
                else "LOCAL_DWQ_IMPROVED_BUT_NO_END_TO_END_WIN"
            )
        else:
            result["classification"] = "UNSTABLE_LOCAL_DWQ_NO_VALID_REFINEMENT"
    except BaseException as exc:
        result["exception"] = f"{type(exc).__name__}: {exc}"
        result["classification"] = "BOUNDED_DWQ_FAILED_SAFE"
        result["safety_stop_reason"] = "exception during bounded one-module experiment" if "Memory" in type(exc).__name__ else None
    finally:
        if original_call is not None:
            nn.QuantizedLinear.__call__ = original_call
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        mx.clear_cache()
    result["memory_after_cleanup"] = memory_snapshot("after_cleanup")
    result["os_memory_after"] = os_snapshot()
    result["rss_max_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    result["total_elapsed_seconds"] = time.perf_counter() - started
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 1 if result["exception"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
