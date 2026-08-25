#!/usr/bin/env python3
"""Bounded end-to-end KL/logit sensitivity for 3->4 promotions.

The pinned Dynamic Quant implementation keeps a full teacher and a copied
student resident and searches a threshold by repeatedly copying the model.
That is not acceptable for a 27B model on this machine.  This harness instead
captures g64 teacher logits in a separate process, loads g128 once, and
promotes exactly one source-backed module at a time.  Source tensors are read
one converter chunk at a time.  The resulting score is measured at the final
language-model logits on the non-held-out Stage 4 calibration split.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn


ROOT = Path(__file__).resolve().parents[3]
STAGE4_HARNESS = ROOT / "research/stage4/harness"
CONVERTER_DIR = ROOT / "research/harness"
for path in (STAGE4_HARNESS, CONVERTER_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from measure_quant_sensitivity import (  # noqa: E402
    calibration_prompt,
    load_chunk,
    make_q4_map,
    module_family,
    module_kind,
    q4_embedding_output,
    q4_linear_output,
    quantization_keys,
    sha256_file,
)
from convert_qwen38_textonly_g128 import (  # noqa: E402
    build_model,
    load_json,
    memory_snapshot,
    plan_chunks,
    source_headers,
    source_weight_map,
    verify_source_plan,
)


BASE_BITS = 3
PROMOTED_BITS = 4
GROUP_SIZE = 128


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def layer_index(path: str) -> int | None:
    marker = ".layers."
    if marker not in path:
        return None
    tail = path.split(marker, 1)[1]
    try:
        return int(tail.split(".", 1)[0])
    except (IndexError, ValueError):
        return None


class PromotionState:
    def __init__(self) -> None:
        self.active: dict[int, dict[str, Any]] = {}
        self.promote_path: str | None = None


def install_promotion_wrappers(state: PromotionState) -> tuple[Any, Any]:
    original_linear = nn.QuantizedLinear.__call__
    original_embedding = nn.QuantizedEmbedding.__call__

    def wrapped_linear(module: Any, x: Any) -> Any:
        entry = state.active.get(id(module))
        if entry is not None and entry["path"] == state.promote_path:
            return q4_linear_output(module, x, entry)
        return original_linear(module, x)

    def wrapped_embedding(module: Any, x: Any) -> Any:
        entry = state.active.get(id(module))
        if entry is not None and entry["path"] == state.promote_path:
            return q4_embedding_output(entry, x)
        return original_embedding(module, x)

    nn.QuantizedLinear.__call__ = wrapped_linear
    nn.QuantizedEmbedding.__call__ = wrapped_embedding
    return original_linear, original_embedding


def remove_promotion_wrappers(original_linear: Any, original_embedding: Any) -> None:
    nn.QuantizedLinear.__call__ = original_linear
    nn.QuantizedEmbedding.__call__ = original_embedding


def snapshot(label: str) -> dict[str, Any]:
    value = memory_snapshot(label, 0.0)
    value["max_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value


def load_teacher_cases(manifest_path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    calibration_path = Path(manifest["calibration_path"]).resolve()
    if "heldout" in str(calibration_path).lower():
        raise ValueError("held-out data is forbidden for sensitivity")
    if manifest["calibration_sha256"] != file_sha256(calibration_path):
        raise ValueError("teacher manifest calibration hash mismatch")
    cases: dict[str, dict[str, Any]] = {}
    for item in manifest["cases"]:
        logits_path = Path(item["logits_path"]).resolve()
        loaded = mx.load(str(logits_path))
        teacher_logits = loaded["logits"]
        mx.eval(teacher_logits)
        if list(teacher_logits.shape) != item["logits_shape"]:
            raise ValueError(f"teacher shape mismatch for {item['case_id']}")
        teacher_logits = teacher_logits.astype(mx.float16)
        teacher_slice = teacher_logits[:-1].astype(mx.float32)
        teacher_logp = nn.log_softmax(teacher_slice, axis=-1).astype(mx.float16)
        teacher_argmax = mx.argmax(teacher_slice, axis=-1)
        mx.eval(teacher_logits, teacher_logp, teacher_argmax)
        cases[item["case_id"]] = {
            "case_id": item["case_id"],
            "token_ids": [int(value) for value in item["token_ids"]],
            "teacher_logits": teacher_logits,
            "teacher_logp": teacher_logp,
            "teacher_argmax": teacher_argmax,
            "target_count": int(item["target_count"]),
        }
    return manifest, cases


def metric_sums(case: dict[str, Any], candidate_logits: Any) -> dict[str, float]:
    candidate = candidate_logits.astype(mx.float32)
    candidate_slice = candidate[:-1]
    teacher_logits = case["teacher_logits"].astype(mx.float32)[:-1]
    teacher_logp = case["teacher_logp"].astype(mx.float32)
    teacher_prob = mx.exp(teacher_logp)
    candidate_logp = nn.log_softmax(candidate_slice, axis=-1)
    kl = mx.sum(teacher_prob * (teacher_logp - candidate_logp), axis=-1)
    teacher_ids = case["teacher_argmax"]
    teacher_nll = -mx.take_along_axis(
        candidate_logp, teacher_ids[:, mx.newaxis], axis=-1
    ).squeeze(-1)
    logit_mse = mx.mean((candidate_slice - teacher_logits) ** 2, axis=-1)
    candidate_argmax = mx.argmax(candidate_slice, axis=-1)
    top1 = (candidate_argmax == teacher_ids).astype(mx.float32)
    values = (kl.sum(), teacher_nll.sum(), logit_mse.sum(), top1.sum())
    mx.eval(*values)
    return {
        "kl_sum": float(values[0].item()),
        "teacher_argmax_nll_sum": float(values[1].item()),
        "logit_mse_sum": float(values[2].item()),
        "teacher_top1_count": float(values[3].item()),
        "target_count": float(case["target_count"]),
    }


def add_sums(total: dict[str, float], item: dict[str, float]) -> None:
    for key, value in item.items():
        total[key] = total.get(key, 0.0) + value


def finalize_metric_sums(total: dict[str, float]) -> dict[str, float]:
    count = max(total["target_count"], 1.0)
    return {
        "kl": total["kl_sum"] / count,
        "teacher_argmax_nll": total["teacher_argmax_nll_sum"] / count,
        "logit_mse": total["logit_mse_sum"] / count,
        "teacher_top1_agreement": total["teacher_top1_count"] / count,
        "target_count": int(total["target_count"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--teacher-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case-ids", nargs="*", default=None)
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument(
        "--module-selection-reference",
        default=None,
        help="Stage 5 ranking JSON used to select top-25, middle-10, and low-10 modules",
    )
    parser.add_argument("--memory-budget-gib", type=float, default=15.0)
    args = parser.parse_args()

    started = time.monotonic()
    source = Path(args.source).resolve()
    baseline = Path(args.baseline).resolve()
    teacher_manifest_path = Path(args.teacher_manifest).resolve()
    output_path = Path(args.output).resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_path}")
    if "heldout" in str(teacher_manifest_path).lower():
        raise ValueError("held-out teacher manifest is forbidden")

    teacher_manifest, teacher_cases = load_teacher_cases(teacher_manifest_path)
    selected_ids = list(teacher_cases)
    if args.case_ids:
        wanted = set(args.case_ids)
        missing = wanted - set(teacher_cases)
        if missing:
            raise ValueError(f"unknown teacher case IDs: {sorted(missing)}")
        selected_ids = [case_id for case_id in selected_ids if case_id in wanted]
    if not selected_ids:
        raise ValueError("no calibration cases selected")
    teacher_cases = {case_id: teacher_cases[case_id] for case_id in selected_ids}

    result: dict[str, Any] = {
        "experiment_id": "P3-END-TO-END-KL-LOGIT-SENSITIVITY",
        "phase": 3,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": "unknown",
        "method": "serialized one-module 3->4 promotion against g64 teacher logits",
        "method_limits": [
            "The teacher is the practical g64 reference, not the 55 GiB high-precision source.",
            "Each score promotes one module while all other modules remain g128.",
            "Scores are calibration-only and are not a held-out quality result.",
            "KL is the primary recovery signal; logit MSE and teacher-top1 are supplementary."
        ],
        "source": str(source),
        "baseline": str(baseline),
        "teacher_manifest": str(teacher_manifest_path),
        "teacher_model_path": teacher_manifest["model_path"],
        "calibration_path": teacher_manifest["calibration_path"],
        "calibration_sha256": teacher_manifest["calibration_sha256"],
        "calibration_case_ids": selected_ids,
        "calibration_sample_count": len(selected_ids),
        "quantization": {
            "baseline_bits": BASE_BITS,
            "promoted_bits": PROMOTED_BITS,
            "group_size": GROUP_SIZE,
            "mode": "affine",
        },
        "bounded_memory": {
            "memory_budget_gib": args.memory_budget_gib,
            "source_chunking": "converter plan_chunks",
            "full_source_model_resident": False,
            "full_teacher_model_resident": False,
            "wired_limit_calls": 0,
        },
        "records": [],
        "memory": [],
        "exception": None,
    }

    from mlx_lm.utils import load

    model = tokenizer = None
    original_linear = original_embedding = None
    state = PromotionState()
    try:
        model, tokenizer = load(str(baseline))
        result["model"] = {
            "path": str(baseline),
            "config": load_json(baseline / "config.json"),
        }
        result["memory"].append(snapshot("after_model_load"))

        schema_model = build_model(load_json(baseline / "config.json"))
        qkeys = quantization_keys(schema_model)
        if len(qkeys) != 498:
            raise ValueError(f"expected 498 quantization keys, got {len(qkeys)}")
        module_by_path = {
            path: module
            for path, module in model.named_modules()
            if f"{path}.weight" in qkeys
        }
        if len(module_by_path) != len(qkeys):
            raise ValueError("baseline module census does not match quantization keys")

        weight_map = source_weight_map(source)
        headers = source_headers(source, weight_map)
        result["source_plan"] = verify_source_plan(
            baseline, source, weight_map, headers, schema_model
        )
        result["source_plan"]["source_index_sha256"] = file_sha256(
            source / "model.safetensors.index.json"
        )

        selected_paths: set[str] | None = None
        if args.module_selection_reference:
            reference_path = Path(args.module_selection_reference).resolve()
            reference = json.loads(reference_path.read_text(encoding="utf-8"))
            reference_order = [item["path"] for item in reference["records"]]
            rank_indices = (
                list(range(min(25, len(reference_order))))
                + list(range(244, min(254, len(reference_order))))
                + list(range(488, min(498, len(reference_order))))
            )
            selected_paths = {reference_order[index] for index in rank_indices}
            missing = selected_paths - set(module_by_path)
            if missing:
                raise ValueError(f"module selection contains unknown paths: {sorted(missing)}")
            result["module_selection"] = {
                "reference": str(reference_path),
                "rank_bands_zero_based": [[0, min(25, len(reference_order))], [244, min(254, len(reference_order))], [488, min(498, len(reference_order))]],
                "selected_count": len(selected_paths),
                "selected_paths": sorted(selected_paths),
            }

        base_outputs: dict[str, Any] = {}
        base_metrics_total: dict[str, float] = {}
        for case_id in selected_ids:
            case = teacher_cases[case_id]
            inputs = mx.array([case["token_ids"]])
            base_logits = model(inputs)[0]
            mx.eval(base_logits)
            base_outputs[case_id] = base_logits.astype(mx.float16)
            mx.eval(base_outputs[case_id])
            add_sums(base_metrics_total, metric_sums(case, base_outputs[case_id]))
            del base_logits, inputs
            mx.clear_cache()
        base_metrics = finalize_metric_sums(base_metrics_total)
        result["baseline_metrics_vs_teacher"] = base_metrics
        result["memory"].append(snapshot("after_baseline_logits"))

        original_linear, original_embedding = install_promotion_wrappers(state)
        records_by_path: dict[str, dict[str, Any]] = {}
        for chunk_index, (chunk_layer, chunk_keys) in enumerate(
            plan_chunks(weight_map), start=1
        ):
            if chunk_layer is not None and args.max_layers is not None and chunk_layer >= args.max_layers:
                break
            chunk_started = time.monotonic()
            loaded = load_chunk(source, weight_map, schema_model, chunk_keys)
            active, metadata = make_q4_map(loaded, qkeys, module_by_path)
            del loaded
            gc.collect()
            mx.clear_cache()
            result["memory"].append(snapshot(f"chunk_{chunk_index}_q4_ready"))
            q4_by_path = {
                item["path"]: active[module_id]
                for module_id, item in (
                    (module_id, entry) for module_id, entry in active.items()
                )
            }
            # make_q4_map stores the path in each entry; the comprehension
            # above deliberately avoids relying on Python module ordering.
            q4_by_path = {entry["path"]: entry for entry in active.values()}

            for path, metadata_item in sorted(metadata.items()):
                if selected_paths is not None and path not in selected_paths:
                    continue
                module = module_by_path[path]
                state.active = {id(module): q4_by_path[path]}
                state.promote_path = path
                candidate_total: dict[str, float] = {}
                for case_id in selected_ids:
                    case = teacher_cases[case_id]
                    inputs = mx.array([case["token_ids"]])
                    candidate_logits = model(inputs)[0]
                    mx.eval(candidate_logits)
                    add_sums(candidate_total, metric_sums(case, candidate_logits))
                    del candidate_logits, inputs
                    mx.clear_cache()
                candidate_metrics = finalize_metric_sums(candidate_total)
                extra_bytes = int(metadata_item["q4_payload_bytes"] - metadata_item["q3_payload_bytes"])
                kl_recovery = base_metrics["kl"] - candidate_metrics["kl"]
                nll_recovery = (
                    base_metrics["teacher_argmax_nll"]
                    - candidate_metrics["teacher_argmax_nll"]
                )
                mse_recovery = base_metrics["logit_mse"] - candidate_metrics["logit_mse"]
                record = {
                    "path": path,
                    "layer_index": layer_index(path),
                    "family": module_family(path),
                    "module": module_kind(path),
                    "parameter_count": int(metadata_item["parameter_count"]),
                    "q3_storage_bytes": int(metadata_item["q3_payload_bytes"]),
                    "q4_storage_bytes": int(metadata_item["q4_payload_bytes"]),
                    "bytes_added_for_promotion": extra_bytes,
                    "baseline_kl": base_metrics["kl"],
                    "candidate_kl": candidate_metrics["kl"],
                    "sensitivity_score": kl_recovery,
                    "quality_recovery_per_byte": kl_recovery / max(extra_bytes, 1),
                    "kl_recovery": kl_recovery,
                    "baseline_teacher_argmax_nll": base_metrics["teacher_argmax_nll"],
                    "candidate_teacher_argmax_nll": candidate_metrics["teacher_argmax_nll"],
                    "teacher_argmax_nll_recovery": nll_recovery,
                    "baseline_logit_mse": base_metrics["logit_mse"],
                    "candidate_logit_mse": candidate_metrics["logit_mse"],
                    "logit_mse_recovery": mse_recovery,
                    "baseline_teacher_top1_agreement": base_metrics["teacher_top1_agreement"],
                    "candidate_teacher_top1_agreement": candidate_metrics["teacher_top1_agreement"],
                    "teacher_top1_recovery": (
                        candidate_metrics["teacher_top1_agreement"]
                        - base_metrics["teacher_top1_agreement"]
                    ),
                    "calibration_sample_count": len(selected_ids),
                    "calibration_token_count": base_metrics["target_count"],
                    "calibration_hash": teacher_manifest["calibration_sha256"],
                }
                records_by_path[path] = record
                state.active = {}
                state.promote_path = None
                gc.collect()
                mx.clear_cache()
                if snapshot(path)["active_plus_cache_gib"] > args.memory_budget_gib:
                    raise MemoryError(f"memory budget exceeded after {path}")

            state.active = {}
            state.promote_path = None
            gc.collect()
            mx.clear_cache()
            result["memory"].append(snapshot(f"chunk_{chunk_index}_released"))
            print(
                json.dumps(
                    {
                        "chunk_index": chunk_index,
                        "layer_index": chunk_layer,
                        "module_count": len(metadata),
                        "elapsed_seconds": round(time.monotonic() - chunk_started, 3),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        result["records"] = sorted(
            records_by_path.values(),
            key=lambda item: (-item["quality_recovery_per_byte"], item["path"]),
        )
        result["processed_module_count"] = len(result["records"])
        expected_module_count = len(selected_paths) if selected_paths is not None else len(qkeys)
        result["expected_module_count"] = expected_module_count
        result["complete"] = len(result["records"]) == expected_module_count
        result["ranking"] = [
            {
                "rank": index,
                "path": item["path"],
                "quality_recovery_per_byte": item["quality_recovery_per_byte"],
                "kl_recovery": item["kl_recovery"],
                "bytes_added_for_promotion": item["bytes_added_for_promotion"],
            }
            for index, item in enumerate(result["records"], start=1)
        ]
    except BaseException as exc:
        result["exception"] = f"{type(exc).__name__}: {exc}"
        result["complete"] = False
        result["error_memory"] = snapshot("exception")
        raise
    finally:
        if original_linear is not None and original_embedding is not None:
            remove_promotion_wrappers(original_linear, original_embedding)
        state.active = {}
        state.promote_path = None
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        mx.clear_cache()

    result["total_elapsed_seconds"] = round(time.monotonic() - started, 3)
    result["max_peak_gib"] = max(
        (item.get("peak_gib", 0.0) for item in result["memory"]), default=0.0
    )
    result["max_active_plus_cache_gib"] = max(
        (item.get("active_plus_cache_gib", 0.0) for item in result["memory"]),
        default=0.0,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "complete",
                    "processed_module_count",
                    "total_elapsed_seconds",
                    "max_peak_gib",
                    "max_active_plus_cache_gib",
                )
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
