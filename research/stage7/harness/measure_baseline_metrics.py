#!/usr/bin/env python3
"""Measure g128 teacher-agreement metrics on frozen Stage 7 calibration splits.

The g64 reference logits are captured in a separate process.  This script only
keeps g128 and one case's float16 reference logits resident, so it is safe for
the 18 GB host and cannot accidentally use final held-out data.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn


ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def memory_snapshot(label: str) -> dict[str, Any]:
    mx.synchronize()
    active = int(mx.get_active_memory())
    cache = int(mx.get_cache_memory())
    return {
        "label": label,
        "active_bytes": active,
        "cache_bytes": cache,
        "active_plus_cache_bytes": active + cache,
        "peak_bytes": int(mx.get_peak_memory()),
        "active_gib": active / (1024**3),
        "active_plus_cache_gib": (active + cache) / (1024**3),
        "peak_gib": int(mx.get_peak_memory()) / (1024**3),
        "rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }


def read_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    calibration = Path(manifest["calibration_path"]).resolve()
    if "heldout" in str(calibration).lower():
        raise ValueError(f"held-out calibration is forbidden: {calibration}")
    if manifest["calibration_sha256"] != sha256_file(calibration):
        raise ValueError(f"calibration hash mismatch: {calibration}")
    return manifest, list(manifest["cases"])


def metric_sums(candidate_logits: Any, teacher_logits: Any, token_ids: list[int]) -> dict[str, float]:
    candidate = candidate_logits[0].astype(mx.float32)
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


def add(total: dict[str, float], item: dict[str, float]) -> None:
    for key, value in item.items():
        total[key] = total.get(key, 0.0) + value


def finalize(total: dict[str, float]) -> dict[str, Any]:
    count = max(total.get("target_count", 0.0), 1.0)
    return {
        "target_count": int(total.get("target_count", 0.0)),
        "kl": total.get("kl_sum", 0.0) / count,
        "teacher_argmax_nll": total.get("teacher_argmax_nll_sum", 0.0) / count,
        "actual_token_nll": total.get("actual_token_nll_sum", 0.0) / count,
        "top1_agreement": total.get("top1_count", 0.0) / count,
        "logit_mse": total.get("logit_mse_sum", 0.0) / count,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-teacher-manifest", required=True)
    parser.add_argument("--validation-teacher-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    train_manifest_path = Path(args.train_teacher_manifest).resolve()
    valid_manifest_path = Path(args.validation_teacher_manifest).resolve()
    train_manifest, train_cases = read_manifest(train_manifest_path)
    valid_manifest, valid_cases = read_manifest(valid_manifest_path)
    result: dict[str, Any] = {
        "experiment_id": "P1-G128-BASELINE-VS-G64-TEACHER",
        "phase": 1,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model_path": str(Path(args.model).resolve()),
        "train_teacher_manifest": str(train_manifest_path),
        "validation_teacher_manifest": str(valid_manifest_path),
        "train_calibration_sha256": train_manifest["calibration_sha256"],
        "validation_calibration_sha256": valid_manifest["calibration_sha256"],
        "splits": {},
        "memory": [],
        "exception": None,
    }
    model = tokenizer = None
    try:
        from mlx_lm.utils import load

        model, tokenizer = load(args.model)
        result["memory"].append(memory_snapshot("after_model_load"))
        for split_name, cases in (("train", train_cases), ("validation", valid_cases)):
            split: dict[str, Any] = {"case_count": len(cases), "cases": {}, "by_domain": {}}
            total: dict[str, float] = {}
            for case in cases:
                logits_payload = mx.load(str(Path(case["logits_path"]).resolve()))
                teacher_logits = logits_payload["logits"]
                token_ids = [int(value) for value in case["token_ids"]]
                inputs = mx.array([token_ids])
                candidate_logits = model(inputs)
                mx.eval(candidate_logits, teacher_logits)
                item = metric_sums(candidate_logits, teacher_logits, token_ids)
                add(total, item)
                category = case.get("category", "unknown")
                domain_total = split["by_domain"].setdefault(category, {})
                add(domain_total, item)
                split["cases"][case["case_id"]] = finalize(item)
                del candidate_logits, inputs, teacher_logits, logits_payload
                mx.clear_cache()
                gc.collect()
            split["overall"] = finalize(total)
            split["by_domain"] = {name: finalize(values) for name, values in split["by_domain"].items()}
            result["splits"][split_name] = split
            result["memory"].append(memory_snapshot(f"after_{split_name}"))
    except BaseException as exc:
        result["exception"] = f"{type(exc).__name__}: {exc}"
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        mx.clear_cache()
    result["memory_after_cleanup"] = memory_snapshot("after_cleanup")
    result["rss_max_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    result["total_elapsed_seconds"] = time.perf_counter() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 1 if result["exception"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
