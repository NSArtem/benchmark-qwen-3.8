#!/usr/bin/env python3
"""Compare corrected full-suite finalist artifacts for Stage 4."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summary(
    label: str,
    final_root: Path,
    raw_root: Path,
    teacher_path: Path,
    payload_bytes: int,
    effective_bpw: float,
) -> dict[str, Any]:
    final = read(final_root / "model.json")
    raw = read(raw_root / "model.json")
    teacher = read(teacher_path)
    cases = final["cases"]
    categories: dict[str, dict[str, Any]] = {}
    for category in sorted({case["category"] for case in cases}):
        subset = [case for case in cases if case["category"] == category]
        objective = [case for case in subset if case["checks"].get("objective_pass") is not None]
        semi = [case for case in subset if case["checks"].get("semi_objective_pass") is not None]
        categories[category] = {
            "cases": len(subset),
            "objective_pass": sum(bool(case["checks"].get("objective_pass")) for case in objective),
            "objective_total": len(objective),
            "semi_objective_pass": sum(bool(case["checks"].get("semi_objective_pass")) for case in semi),
            "semi_objective_total": len(semi),
        }
    return {
        "label": label,
        "model_path": final.get("model_path"),
        "payload_bytes": payload_bytes,
        "payload_gib": payload_bytes / (1 << 30),
        "effective_bpw": effective_bpw,
        "suite_sha256": final.get("suite_sha256"),
        "objective_pass": sum(bool(case["checks"].get("objective_pass")) for case in cases),
        "objective_total": sum(case["checks"].get("objective_pass") is not None for case in cases),
        "semi_objective_pass": sum(bool(case["checks"].get("semi_objective_pass")) for case in cases),
        "semi_objective_total": sum(case["checks"].get("semi_objective_pass") is not None for case in cases),
        "objective_failures": [case["case_id"] for case in cases if case["checks"].get("objective_pass") is False],
        "semi_objective_failures": [case["case_id"] for case in cases if case["checks"].get("semi_objective_pass") is False],
        "category_summary": categories,
        "finish_reasons": {
            reason: sum(case.get("finish_reason") == reason for case in cases)
            for reason in sorted({case.get("finish_reason") for case in cases})
        },
        "early_eos_count": sum(bool(case.get("early_eos")) for case in cases),
        "repetition_count": sum(bool(case.get("checks", {}).get("repetition_detected")) for case in cases),
        "generated_tokens_mean": statistics.mean(case.get("generated_tokens", 0) for case in cases),
        "decode_tps_mean": statistics.mean(case.get("decode_tps", 0.0) for case in cases),
        "decode_tps_median": statistics.median(case.get("decode_tps", 0.0) for case in cases),
        "ttft_seconds_mean": statistics.mean(case.get("ttft_seconds", 0.0) for case in cases),
        "peak_gib_max": max((case.get("memory_after_generation") or {}).get("peak_gib", 0.0) for case in cases),
        "active_plus_cache_gib_max": max(
            (case.get("memory_after_generation") or {}).get("active_plus_cache_gib", 0.0)
            for case in cases
        ),
        "raw_total_elapsed_seconds": raw.get("total_elapsed_seconds"),
        "teacher_forced": {
            "case_count": len(teacher.get("cases", [])),
            "mean_nll_nats": teacher.get("mean_nll_nats"),
            "weighted_mean_nll_nats": teacher.get("weighted_mean_nll_nats"),
            "mean_top1_accuracy": teacher.get("mean_top1_accuracy"),
            "weighted_top1_accuracy": teacher.get("weighted_top1_accuracy"),
            "total_target_tokens": teacher.get("total_target_tokens"),
            "per_case": [
                {
                    "case_id": case["case_id"],
                    "category": case["category"],
                    "average_nll_nats": case["average_nll_nats"],
                    "top1_accuracy": case["top1_accuracy"],
                }
                for case in teacher.get("cases", [])
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g64-final", type=Path, required=True)
    parser.add_argument("--g128-final", type=Path, required=True)
    parser.add_argument("--mixed-final", type=Path, required=True)
    parser.add_argument("--g64-raw", type=Path, required=True)
    parser.add_argument("--g128-raw", type=Path, required=True)
    parser.add_argument("--mixed-raw", type=Path, required=True)
    parser.add_argument("--g64-teacher", type=Path, required=True)
    parser.add_argument("--g128-teacher", type=Path, required=True)
    parser.add_argument("--mixed-teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    models = [
        summary("g64", args.g64_final, args.g64_raw, args.g64_teacher, 11771132928, 3.50123),
        summary("g128", args.g128_final, args.g128_raw, args.g128_teacher, 10930715648, 3.25125),
        summary("mixed34", args.mixed_final, args.mixed_raw, args.mixed_teacher, 11451399168, 3.40613),
    ]
    g64_teacher = models[0]["teacher_forced"]["weighted_mean_nll_nats"]
    for model in models:
        teacher = model["teacher_forced"]
        teacher["weighted_nll_delta_vs_g64"] = teacher["weighted_mean_nll_nats"] - g64_teacher
        teacher["weighted_top1_delta_vs_g64"] = (
            teacher["weighted_top1_accuracy"] - models[0]["teacher_forced"]["weighted_top1_accuracy"]
        )
    case_ids = [case["case_id"] for case in read(args.g64_final / "model.json")["cases"]]
    case_matrix = []
    for case_id in case_ids:
        row = {"case_id": case_id}
        for model, root in zip(models, (args.g64_final, args.g128_final, args.mixed_final)):
            case = next(item for item in read(root / "model.json")["cases"] if item["case_id"] == case_id)
            row[model["label"]] = {
                "objective_pass": case["checks"].get("objective_pass"),
                "semi_objective_pass": case["checks"].get("semi_objective_pass"),
                "finish_reason": case.get("finish_reason"),
                "generated_tokens": case.get("generated_tokens"),
            }
        case_matrix.append(row)
    result = {
        "experiment_id": "P6-FULL-FINALIST-COMPARISON-V2",
        "phase": 6,
        "suite_sha256": models[0]["suite_sha256"],
        "rubric_correction": {
            "case_id": "coding-03",
            "change": "accepted transaction boundary and atomic transaction as equivalent single-transaction wording",
            "corrected_suite_sha256": models[0]["suite_sha256"],
            "rerun_artifacts": "research/stage4/artifacts/quality-v2/rerun-coding03",
        },
        "models": models,
        "case_matrix": case_matrix,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
