#!/usr/bin/env python3
"""Merge a candidate's earlier v2 full run with final corrected reruns.

Q1 had no heldout_v1 full run before the Stage 6 budget audit.  This helper
therefore preserves the 15 unaffected cases from its earlier v2 run and
replaces the nine audited cases with the final-hash rerun.
"""

from __future__ import annotations

import argparse
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
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--affected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    base = read_json(args.base)
    affected = read_json(args.affected)
    if affected.get("suite_sha256") != V2_HASH:
        raise AssertionError("affected run is not the final heldout_v2")
    base_cases = {item["case_id"]: item for item in base["cases"]}
    affected_cases = {item["case_id"]: item for item in affected["cases"]}
    if len(base_cases) != 24 or set(affected_cases) != CORRECTED_CASES:
        raise AssertionError("unexpected candidate/affected case shape")
    if set(base_cases) & CORRECTED_CASES:
        # The base records may be old v2 reruns, but the final merge must use
        # the corrected run for every audited case.
        pass
    merged_cases = [affected_cases.get(item["case_id"], item) for item in base["cases"]]
    merged = dict(base)
    merged.update(
        {
            "experiment_id": f"P2-HELDOUT-V2-FULL-{base['model_label'].upper()}",
            "suite_path": str(ROOT / "research/stage6/heldout/heldout_v2.jsonl"),
            "suite_sha256": V2_HASH,
            "cases": merged_cases,
            "source_base_model": str(args.base),
            "source_affected_model": str(args.affected),
            "correction_policy": "Final corrected reruns replace all nine audited coding/reasoning cases; the other 15 cases are preserved from the earlier candidate v2 run.",
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
        "source_base_model": str(args.base),
        "source_affected_model": str(args.affected),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
