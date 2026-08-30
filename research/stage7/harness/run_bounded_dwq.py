#!/usr/bin/env python3
"""Reusable bounded, streamed DWQ refinement for the Qwen3.8 g128 student.

This is intentionally a bounded research harness rather than a full-model
teacher/student optimizer.  The g128 model stays resident; source weights are
loaded one module at a time; only affine quantization scales and biases are
optimized; and cumulative patch checkpoints make refinement resumable without
copying the 10 GiB base model.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import resource
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn


ROOT = Path(__file__).resolve().parents[3]
CONVERTER_DIR = ROOT / "research/harness"
if str(CONVERTER_DIR) not in sys.path:
    sys.path.insert(0, str(CONVERTER_DIR))

from convert_qwen38_textonly_g128 import output_key, source_weight_map  # noqa: E402


BITS = 3
GROUP_SIZE = 128


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def memory_snapshot(label: str) -> dict[str, Any]:
    mx.synchronize()
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


def read_teacher_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    calibration = Path(manifest["calibration_path"]).resolve()
    if "heldout" in str(calibration).lower():
        raise ValueError(f"held-out calibration is forbidden: {calibration}")
    if manifest["calibration_sha256"] != sha256_file(calibration):
        raise ValueError(f"calibration hash mismatch: {calibration}")
    cases = list(manifest["cases"])
    if not cases:
        raise ValueError(f"empty teacher manifest: {path}")
    return manifest, cases


def select_cases(
    cases: list[dict[str, Any]], requested: list[str] | None, label: str
) -> list[dict[str, Any]]:
    if requested is None:
        return cases
    by_id = {str(case["case_id"]): case for case in cases}
    missing = [case_id for case_id in requested if case_id not in by_id]
    if missing:
        raise KeyError(f"unknown {label} case IDs: {missing}")
    if not requested:
        raise ValueError(f"{label} case selection is empty")
    return [by_id[case_id] for case_id in requested]


def load_patch(path: Path | None, modules: dict[str, Any]) -> set[str]:
    if path is None:
        return set()
    payload = mx.load(str(path.resolve()))
    paths: set[str] = set()
    for key, value in payload.items():
        if not (key.endswith(".scales") or key.endswith(".biases")):
            continue
        module_path, parameter = key.rsplit(".", 1)
        if module_path not in modules:
            raise KeyError(f"patch module not present in model: {module_path}")
        module = modules[module_path]
        if not isinstance(module, nn.QuantizedLinear):
            raise TypeError(f"patch target is not QuantizedLinear: {module_path}")
        current = getattr(module, parameter)
        if current is None or list(current.shape) != list(value.shape):
            raise ValueError(f"patch shape mismatch for {key}: {value.shape}")
        setattr(module, parameter, value.astype(current.dtype))
        paths.add(module_path)
    mx.eval(*(getattr(modules[path], parameter) for path in paths for parameter in ("scales", "biases")))
    return paths


def save_patch(path: Path, patch: dict[str, dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    tensors: dict[str, Any] = {}
    for module_path, values in sorted(patch.items()):
        tensors[f"{module_path}.scales"] = values["scales"]
        tensors[f"{module_path}.biases"] = values["biases"]
    mx.eval(*tensors.values())
    mx.save_safetensors(
        str(path),
        tensors,
        metadata={
            "format": "mlx",
            "stage": "7",
            "algorithm": "bounded_streamed_affine_scale_bias_dwq",
            "note": "cumulative patch; apply with the Stage 7 harness/materializer",
        },
    )
    return path.stat().st_size


def metric_sums(candidate_logits: Any, teacher_logits: Any, token_ids: list[int]) -> dict[str, float]:
    candidate = candidate_logits.astype(mx.float32)
    teacher = teacher_logits.astype(mx.float32)
    if candidate.shape != teacher.shape:
        raise ValueError(f"candidate/teacher shape mismatch: {candidate.shape} != {teacher.shape}")
    candidate_slice = candidate[:-1]
    teacher_slice = teacher[:-1]
    candidate_logp = nn.log_softmax(candidate_slice, axis=-1)
    teacher_logp = nn.log_softmax(teacher_slice, axis=-1)
    teacher_prob = mx.exp(teacher_logp)
    teacher_ids = mx.argmax(teacher_slice, axis=-1)
    candidate_ids = mx.argmax(candidate_slice, axis=-1)
    actual_ids = mx.array(token_ids[1:])
    values = (
        mx.sum(teacher_prob * (teacher_logp - candidate_logp)),
        -mx.sum(mx.take_along_axis(candidate_logp, teacher_ids[:, None], axis=-1).squeeze(-1)),
        -mx.sum(mx.take_along_axis(candidate_logp, actual_ids[:, None], axis=-1).squeeze(-1)),
        mx.sum((candidate_ids == teacher_ids).astype(mx.float32)),
        mx.sum(mx.mean((candidate_slice - teacher_slice) ** 2, axis=-1)),
    )
    mx.eval(*values)
    return {
        "kl_sum": float(values[0].item()),
        "teacher_argmax_nll_sum": float(values[1].item()),
        "actual_token_nll_sum": float(values[2].item()),
        "top1_count": float(values[3].item()),
        "logit_mse_sum": float(values[4].item()),
        "target_count": float(candidate.shape[0] - 1),
    }


def add_sums(total: dict[str, float], item: dict[str, float]) -> None:
    for key, value in item.items():
        total[key] = total.get(key, 0.0) + value


def finalize_sums(total: dict[str, float]) -> dict[str, Any]:
    count = max(total.get("target_count", 0.0), 1.0)
    return {
        "target_count": int(total.get("target_count", 0.0)),
        "kl": total.get("kl_sum", 0.0) / count,
        "teacher_argmax_nll": total.get("teacher_argmax_nll_sum", 0.0) / count,
        "actual_token_nll": total.get("actual_token_nll_sum", 0.0) / count,
        "top1_agreement": total.get("top1_count", 0.0) / count,
        "logit_mse": total.get("logit_mse_sum", 0.0) / count,
    }


def model_metrics(model: Any, cases: list[dict[str, Any]]) -> dict[str, Any]:
    total: dict[str, float] = {}
    by_domain: dict[str, dict[str, float]] = {}
    per_case: dict[str, Any] = {}
    for case in cases:
        logits_payload = mx.load(str(Path(case["logits_path"]).resolve()))
        teacher_logits = logits_payload["logits"]
        token_ids = [int(value) for value in case["token_ids"]]
        candidate = model(mx.array([token_ids]))[0]
        mx.eval(candidate, teacher_logits)
        item = metric_sums(candidate, teacher_logits, token_ids)
        add_sums(total, item)
        domain = str(case.get("category", "unknown"))
        domain_total = by_domain.setdefault(domain, {})
        add_sums(domain_total, item)
        per_case[case["case_id"]] = finalize_sums(item)
        del candidate, teacher_logits, logits_payload
        mx.clear_cache()
        gc.collect()
    return {
        "case_count": len(cases),
        "overall": finalize_sums(total),
        "by_domain": {name: finalize_sums(value) for name, value in sorted(by_domain.items())},
        "cases": per_case,
    }


def find_source_tensor(source: Path, target_path: str) -> tuple[Path, str]:
    weight_map = source_weight_map(source)
    wanted = f"{target_path}.weight"
    source_key = next((key for key in weight_map if output_key(key) == wanted), None)
    if source_key is None:
        raise KeyError(f"source tensor for {target_path} not found")
    return source / weight_map[source_key], source_key


def capture_samples(
    model: Any,
    target_module: Any,
    cases: list[dict[str, Any]],
    source_weight: Any,
    rows_per_case: int,
) -> tuple[Any, Any, int]:
    captured: list[Any] = []
    original_call = nn.QuantizedLinear.__call__

    def capture_call(module: Any, x: Any) -> Any:
        value = original_call(module, x)
        if module is target_module:
            captured.append(x)
        return value

    activation_samples: list[Any] = []
    target_samples: list[Any] = []
    nn.QuantizedLinear.__call__ = capture_call
    try:
        for case in cases:
            captured.clear()
            inputs = mx.array([[int(value) for value in case["token_ids"]]])
            logits = model(inputs)
            mx.eval(logits)
            if not captured:
                raise RuntimeError(f"target module was not called for {case['case_id']}")
            activation = captured[-1]
            mx.eval(activation)
            rows = min(rows_per_case, int(activation.shape[-2]))
            sample = activation[0, -rows:, :].astype(mx.float32)
            target = sample @ source_weight.T
            mx.eval(sample, target)
            activation_samples.append(sample)
            target_samples.append(target)
            del logits, inputs, activation
            mx.clear_cache()
    finally:
        nn.QuantizedLinear.__call__ = original_call
    x_all = mx.concatenate(activation_samples, axis=0)
    y_all = mx.concatenate(target_samples, axis=0)
    mx.eval(x_all, y_all)
    return x_all, y_all, sum(int(item.shape[0]) for item in activation_samples)


def finite_array(value: Any) -> bool:
    check = mx.all(mx.isfinite(value))
    mx.eval(check)
    return bool(check.item())


def optimize_module(
    module: Any,
    x_all: Any,
    y_all: Any,
    steps: int,
    learning_rate: float,
    clip_fraction: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    qweight = module.weight
    original_scales = module.scales
    original_biases = module.biases
    if original_biases is None:
        raise ValueError("affine g128 target unexpectedly has no quantization biases")
    params = {
        "scales": original_scales.astype(mx.float32),
        "biases": original_biases.astype(mx.float32),
    }
    original = {name: value for name, value in params.items()}
    module_bias = getattr(module, "bias", None)
    if module_bias is not None:
        module_bias = module_bias.astype(mx.float32)
    mx.eval(qweight, *params.values())

    def loss_fn(current: dict[str, Any]) -> Any:
        value = mx.quantized_matmul(
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
            value = value + module_bias
        return mx.mean((value - y_all) ** 2)

    value_and_grad = mx.value_and_grad(loss_fn)
    initial = loss_fn(params)
    mx.eval(initial)
    initial_value = float(initial.item())
    trace = [initial_value]
    finite = math.isfinite(initial_value)
    diverged_step: int | None = None
    for step in range(steps):
        if not finite:
            diverged_step = step
            break
        loss, grads = value_and_grad(params)
        mx.eval(loss, grads)
        loss_value = float(loss.item())
        finite = math.isfinite(loss_value) and all(finite_array(value) for value in grads.values())
        if not finite:
            diverged_step = step
            break
        params = {
            name: params[name] - learning_rate * grads[name]
            for name in params
        }
        if clip_fraction > 0:
            bounded: dict[str, Any] = {}
            for name in params:
                delta = clip_fraction * mx.maximum(mx.abs(original[name]), mx.array(1e-4))
                bounded[name] = mx.clip(params[name], original[name] - delta, original[name] + delta)
            params = bounded
        mx.eval(*params.values())
        finite = all(finite_array(value) for value in params.values())
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == steps:
            checkpoint = loss_fn(params)
            mx.eval(checkpoint)
            checkpoint_value = float(checkpoint.item())
            trace.append(checkpoint_value)
            finite = finite and math.isfinite(checkpoint_value)
        if not finite:
            diverged_step = step + 1
            break
    final = loss_fn(params)
    mx.eval(final)
    final_value = float(final.item())
    finite = finite and math.isfinite(final_value)
    if not finite:
        diverged_step = diverged_step if diverged_step is not None else steps
    record = {
        "local_loss_before": initial_value,
        "local_loss_after": final_value if finite else None,
        "local_loss_trace": trace,
        "local_relative_loss_reduction": ((initial_value - final_value) / max(initial_value, 1e-12)) if finite else None,
        "finite": finite,
        "diverged_step": diverged_step,
        "steps_completed": steps if diverged_step is None else diverged_step,
        "max_scale_delta": float(mx.max(mx.abs(params["scales"] - original["scales"])).item()),
        "max_bias_delta": float(mx.max(mx.abs(params["biases"] - original["biases"])).item()),
    }
    output = {
        "scales": params["scales"].astype(original_scales.dtype),
        "biases": params["biases"].astype(original_biases.dtype),
    }
    mx.eval(*output.values())
    return output, record


def ranking_paths(path: Path, top_n: int) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "records" in payload:
        paths = [str(item["path"]) for item in payload["records"]]
    elif "ranking" in payload:
        paths = [str(item["path"]) for item in payload["ranking"]]
    else:
        raise ValueError(f"ranking has no records/ranking: {path}")
    if len(paths) < top_n:
        raise ValueError(f"ranking has only {len(paths)} entries; need {top_n}")
    return paths[:top_n]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--train-teacher-manifest", required=True)
    parser.add_argument("--validation-teacher-manifest", required=True)
    parser.add_argument("--ranking", required=True)
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--module-paths", nargs="*", default=None)
    parser.add_argument("--initial-patch", default=None)
    parser.add_argument("--train-case-ids", nargs="*", default=None)
    parser.add_argument("--validation-case-ids", nargs="*", default=None)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--clip-fraction", type=float, default=0.25)
    parser.add_argument("--activation-rows-per-case", type=int, default=128)
    args = parser.parse_args()
    if args.top_n is None and not args.module_paths:
        raise ValueError("provide --top-n or --module-paths")
    if args.top_n is not None and args.top_n <= 0:
        raise ValueError("--top-n must be positive")
    output_root = Path(args.output_root).resolve()
    output_path = output_root / f"{args.run_name}.json"
    patch_path = output_root / f"{args.run_name}.safetensors"
    if output_path.exists() or patch_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path} or {patch_path}")
    train_manifest_path = Path(args.train_teacher_manifest).resolve()
    valid_manifest_path = Path(args.validation_teacher_manifest).resolve()
    train_manifest, train_cases = read_teacher_manifest(train_manifest_path)
    valid_manifest, valid_cases = read_teacher_manifest(valid_manifest_path)
    train_cases = select_cases(train_cases, args.train_case_ids, "train")
    valid_cases = select_cases(valid_cases, args.validation_case_ids, "validation")
    if args.module_paths:
        selected_paths = list(args.module_paths)
    else:
        selected_paths = ranking_paths(Path(args.ranking).resolve(), int(args.top_n))
    result: dict[str, Any] = {
        "experiment_id": "P2-P3-BOUNDED-DWQ-G128",
        "phase": 2 if args.run_name.startswith("reproduce") else 3,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model_path": str(Path(args.model).resolve()),
        "source_path": str(Path(args.source).resolve()),
        "ranking_path": str(Path(args.ranking).resolve()),
        "train_teacher_manifest": str(train_manifest_path),
        "validation_teacher_manifest": str(valid_manifest_path),
        "train_calibration_sha256": train_manifest["calibration_sha256"],
        "validation_calibration_sha256": valid_manifest["calibration_sha256"],
        "selected_paths": selected_paths,
        "train_case_ids": [case["case_id"] for case in train_cases],
        "validation_case_ids": [case["case_id"] for case in valid_cases],
        "initial_patch": str(Path(args.initial_patch).resolve()) if args.initial_patch else None,
        "learning_rate": args.learning_rate,
        "steps": args.steps,
        "clip_fraction": args.clip_fraction,
        "activation_rows_per_case": args.activation_rows_per_case,
        "full_teacher_resident": False,
        "full_source_resident": False,
        "module_records": [],
        "stages": [],
        "memory": [],
        "exception": None,
        "safety_stop_reason": None,
    }
    model = tokenizer = None
    patch_values: dict[str, dict[str, Any]] = {}
    patched_paths: set[str] = set()
    started = time.perf_counter()
    try:
        from mlx_lm.utils import load

        model, tokenizer = load(args.model)
        result["memory"].append(memory_snapshot("after_model_load"))
        modules = dict(model.named_modules())
        if args.initial_patch:
            patched_paths = load_patch(Path(args.initial_patch), modules)
            payload = mx.load(str(Path(args.initial_patch).resolve()))
            for path in sorted(patched_paths):
                patch_values[path] = {
                    "scales": payload[f"{path}.scales"],
                    "biases": payload[f"{path}.biases"],
                }
            del payload
            result["memory"].append(memory_snapshot("after_initial_patch"))
        missing = [path for path in selected_paths if path not in modules]
        if missing:
            raise KeyError(f"selected modules missing from model: {missing}")
        result["baseline"] = {
            "train": model_metrics(model, train_cases),
            "validation": model_metrics(model, valid_cases),
        }
        result["memory"].append(memory_snapshot("after_baseline_metrics"))
        for module_path in selected_paths:
            if module_path in patched_paths:
                continue
            module = modules[module_path]
            if not isinstance(module, nn.QuantizedLinear):
                raise TypeError(f"selected target is not QuantizedLinear: {module_path}")
            source_file, source_key = find_source_tensor(Path(args.source).resolve(), module_path)
            loaded_source = mx.load(str(source_file))
            source_weight = loaded_source[source_key].astype(mx.float32)
            mx.eval(source_weight)
            result["memory"].append(memory_snapshot(f"{module_path}:after_source_load"))
            x_all = y_all = None
            try:
                x_all, y_all, activation_rows = capture_samples(
                    model, module, train_cases, source_weight, args.activation_rows_per_case
                )
                optimized, local_record = optimize_module(
                    module, x_all, y_all, args.steps, args.learning_rate, args.clip_fraction
                )
                local_record.update({
                    "path": module_path,
                    "source_key": source_key,
                    "source_file": str(source_file),
                    "source_weight_bytes": int(source_weight.nbytes),
                    "activation_rows": activation_rows,
                })
                result["module_records"].append(local_record)
                if not local_record["finite"]:
                    result["exception"] = f"non-finite optimizer at {module_path}"
                    result["safety_stop_reason"] = "DWQ optimizer became non-finite"
                    break
                patch_values[module_path] = optimized
                setattr(module, "scales", optimized["scales"])
                setattr(module, "biases", optimized["biases"])
                mx.eval(module.scales, module.biases)
                patched_paths.add(module_path)
            finally:
                del loaded_source, source_weight
                if x_all is not None:
                    del x_all
                if y_all is not None:
                    del y_all
                gc.collect()
                mx.clear_cache()
            result["memory"].append(memory_snapshot(f"{module_path}:after_optimization"))
            if result["exception"]:
                break
        if not result["exception"]:
            patch_bytes = save_patch(patch_path, patch_values)
            result["patch_path"] = str(patch_path)
            result["patch_bytes"] = patch_bytes
            result["stages"].append({
                "module_count": len(patched_paths),
                "module_paths": sorted(patched_paths),
                "train": model_metrics(model, train_cases),
                "validation": model_metrics(model, valid_cases),
                "patch_path": str(patch_path),
                "patch_bytes": patch_bytes,
            })
            result["memory"].append(memory_snapshot("after_refinement_metrics"))
        result["completed_module_count"] = len(patched_paths)
    except BaseException as exc:
        result["exception"] = f"{type(exc).__name__}: {exc}"
        result["error_memory"] = memory_snapshot("exception")
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        mx.clear_cache()
    result["memory_after_cleanup"] = memory_snapshot("after_cleanup")
    result["max_peak_gib"] = max((item.get("peak_gib", 0.0) for item in result["memory"]), default=0.0)
    result["rss_max_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    result["total_elapsed_seconds"] = time.perf_counter() - started
    output_root.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "run_name": args.run_name,
        "completed_module_count": result.get("completed_module_count"),
        "exception": result.get("exception"),
        "max_peak_gib": result.get("max_peak_gib"),
        "total_elapsed_seconds": result.get("total_elapsed_seconds"),
    }, sort_keys=True), flush=True)
    return 1 if result["exception"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
