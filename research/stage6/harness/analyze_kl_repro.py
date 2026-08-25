#!/usr/bin/env python3
"""Compare Stage 5 KL ranks with Stage 6 controlled repeats."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index], reverse=True)
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            ranks[order[position]] = rank
        cursor = end
    return ranks


def correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_rank = average_ranks(left)
    right_rank = average_ranks(right)
    left_mean = sum(left_rank) / len(left_rank)
    right_mean = sum(right_rank) / len(right_rank)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left_rank, right_rank))
    left_norm = math.sqrt(sum((a - left_mean) ** 2 for a in left_rank))
    right_norm = math.sqrt(sum((b - right_mean) ** 2 for b in right_rank))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else None


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_norm = math.sqrt(sum((a - left_mean) ** 2 for a in left))
    right_norm = math.sqrt(sum((b - right_mean) ** 2 for b in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else None


def load_records(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {item["path"]: item for item in payload["records"]}


def compare(reference: dict[str, dict[str, Any]], candidate: dict[str, dict[str, Any]], label: str) -> dict[str, Any]:
    paths = sorted(set(reference) & set(candidate))
    ref_scores = [float(reference[path]["quality_recovery_per_byte"]) for path in paths]
    cand_scores = [float(candidate[path]["quality_recovery_per_byte"]) for path in paths]
    ref_order = sorted(paths, key=lambda path: (float(reference[path]["quality_recovery_per_byte"]), path), reverse=True)
    cand_order = sorted(paths, key=lambda path: (float(candidate[path]["quality_recovery_per_byte"]), path), reverse=True)
    signs = [
        (float(reference[path]["quality_recovery_per_byte"]) > 0)
        == (float(candidate[path]["quality_recovery_per_byte"]) > 0)
        for path in paths
    ]
    abs_deltas = [abs(a - b) for a, b in zip(ref_scores, cand_scores)]
    return {
        "label": label,
        "module_count": len(paths),
        "spearman": correlation(ref_scores, cand_scores),
        "pearson_score": pearson(ref_scores, cand_scores),
        "top10_overlap": len(set(ref_order[:10]) & set(cand_order[:10])),
        "top25_overlap": len(set(ref_order[:25]) & set(cand_order[:25])),
        "sign_agreement": sum(signs) / len(signs) if signs else None,
        "mean_absolute_score_delta": sum(abs_deltas) / len(abs_deltas) if abs_deltas else None,
        "max_absolute_score_delta": max(abs_deltas) if abs_deltas else None,
        "reference_score_range": [min(ref_scores), max(ref_scores)] if ref_scores else None,
        "candidate_score_range": [min(cand_scores), max(cand_scores)] if cand_scores else None,
        "reference_top10": ref_order[:10],
        "candidate_top10": cand_order[:10],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--same-data", required=True)
    parser.add_argument("--alternate-data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    reference = load_records(Path(args.reference))
    same_data = load_records(Path(args.same_data))
    alternate = load_records(Path(args.alternate_data))
    result = {
        "reference": str(Path(args.reference)),
        "same_data_repeat": str(Path(args.same_data)),
        "alternate_subset": str(Path(args.alternate_data)),
        "same_data_calibration_hash": json.loads(Path(args.same_data).read_text(encoding="utf-8"))["calibration_sha256"],
        "alternate_calibration_hash": json.loads(Path(args.alternate_data).read_text(encoding="utf-8"))["calibration_sha256"],
        "same_data_case_ids": json.loads(Path(args.same_data).read_text(encoding="utf-8"))["calibration_case_ids"],
        "alternate_case_ids": json.loads(Path(args.alternate_data).read_text(encoding="utf-8"))["calibration_case_ids"],
        "comparisons": {
            "same_calibration_data": compare(reference, same_data, "v1 12-case order vs Stage 6 repeat on identical 12 cases"),
            "different_calibration_subset": compare(reference, alternate, "v1 12-case order vs Stage 6 2-case alternate subset"),
            "repeat_vs_alternate": compare(same_data, alternate, "Stage 6 identical-data repeat vs alternate subset"),
        },
        "interpretation": {
            "same_data_is_numerical_repeatability": True,
            "alternate_is_calibration_robustness": True,
            "heldout_data_used": False,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
