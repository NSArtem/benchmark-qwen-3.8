#!/usr/bin/env python3
"""Merge the unchanged Stage 5 heldout_v1 cases with v2 affected reruns."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
V2_HASH = "1749a8b47e99bb3e0e04347d75e24d7022ed3ab8d5ffa5bc56926b9cf02cbcbd"
CORRECTED_CASES = {
    *(f"h5-coding-{index:02d}" for index in range(1, 7)),
    *(f"h5-reasoning-{index:02d}" for index in (1, 2, 4)),
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--affected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    baseline = read_json(args.baseline)
    affected = read_json(args.affected)
    if baseline.get("suite_sha256") != "6fe7bd7da6732e9057a75764292b18b29c3262a96eb39391b700f038f641851b":
        raise AssertionError("baseline is not heldout_v1")
    if affected.get("suite_sha256") != V2_HASH:
        raise AssertionError("affected run is not heldout_v2")
    baseline_cases = {item["case_id"]: item for item in baseline["cases"]}
    affected_cases = {item["case_id"]: item for item in affected["cases"]}
    if set(affected_cases) != CORRECTED_CASES:
        raise AssertionError(f"unexpected affected cases: {sorted(affected_cases)}")
    merged_cases = [affected_cases.get(item["case_id"], item) for item in baseline["cases"]]
    merged = dict(baseline)
    merged.update(
        {
            "experiment_id": f"P1-HELDOUT-V2-FULL-{baseline['model_label'].upper()}",
            "suite_path": str(ROOT / "research/stage6/heldout/heldout_v2.jsonl"),
            "suite_sha256": V2_HASH,
            "cases": merged_cases,
            "source_v1_model": str(args.baseline),
            "source_affected_model": str(args.affected),
            "correction_policy": "Only the nine targeted coding/reasoning budget-correction cases were replaced; prompts/scorer/answers unchanged.",
        }
    )
    merged["case_count"] = len(merged_cases)
    merged["selected_case_ids"] = [item["case_id"] for item in merged_cases]
    merged["overall_classification"] = "PASS" if not [item for item in merged_cases if item.get("exception") or item.get("checks", {}).get("mechanical_pass") is False] else "OBJECTIVE_OR_HARNESS_FAILURES_PRESENT"
    summary = {
        "experiment_id": merged["experiment_id"],
        "model_label": merged["model_label"],
        "model_path": merged["model_path"],
        "suite_sha256": V2_HASH,
        "case_count": len(merged_cases),
        "objective_pass": sum(bool(item.get("checks", {}).get("objective_pass")) for item in merged_cases),
        "objective_total": sum(item.get("score_kind") == "objective" for item in merged_cases),
        "semi_objective_pass": sum(bool(item.get("checks", {}).get("semi_objective_pass")) for item in merged_cases),
        "semi_objective_total": sum(item.get("score_kind") == "semi-objective" for item in merged_cases),
        "early_eos": sum(bool(item.get("early_eos")) for item in merged_cases),
        "repetition": sum(bool(item.get("checks", {}).get("repetition_detected")) for item in merged_cases),
        "exceptions": sum(bool(item.get("exception")) for item in merged_cases),
        "overall_classification": merged["overall_classification"],
        "source_v1_model": str(args.baseline),
        "source_affected_model": str(args.affected),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
