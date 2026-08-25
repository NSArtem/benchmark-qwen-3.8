#!/usr/bin/env python3
"""Rescore completed Quality Suite V2 outputs against the frozen rubric.

This is used after a rubric-only correction. It never loads a model and never
changes the preserved raw generation run; the resulting model.json records
the exact source run and the current suite hash.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from run_quality_v2 import evaluate_case, read_jsonl, sha256_text, write_json


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", choices=("calibration", "evaluation"), default="evaluation")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "logs").mkdir(exist_ok=True)
    suite_path = ROOT / ("calibration/calibration_v2.jsonl" if args.split == "calibration" else "evaluation/evaluation_v2.jsonl")
    specs = read_jsonl(suite_path)
    raw_model = json.loads((raw_root / "model.json").read_text(encoding="utf-8"))
    result: dict[str, Any] = {
        "experiment_id": f"P1-{args.split.upper()}-{args.label.upper()}-QUALITY-V2-RESCORED",
        "phase": 1,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": raw_model.get("git_commit"),
        "model_label": args.label,
        "model_path": raw_model.get("model_path"),
        "split": args.split,
        "suite_path": str(suite_path),
        "suite_sha256": sha256_text(suite_path.read_text(encoding="utf-8")),
        "case_count": len(specs),
        "source_raw_run": str(raw_root.resolve()),
        "source_raw_suite_sha256": raw_model.get("suite_sha256"),
        "model": raw_model.get("model"),
        "cases": [],
        "exception": None,
    }
    for spec in specs:
        case_id = spec["case_id"]
        raw_record = next(item for item in raw_model["cases"] if item["case_id"] == case_id)
        output_path = raw_root / "outputs" / f"{case_id}.txt"
        token_path = raw_root / "tokens" / f"{case_id}.json"
        generated_text = output_path.read_text(encoding="utf-8")
        token_ids = [int(value) for value in json.loads(token_path.read_text(encoding="utf-8"))["token_ids"]]
        record = dict(raw_record)
        record["checks"] = evaluate_case(spec, generated_text, token_ids)
        record["raw_output_path"] = str(output_path.resolve())
        record["raw_token_path"] = str(token_path.resolve())
        result["cases"].append(record)

    failures = [item for item in result["cases"] if item.get("exception") or item.get("checks", {}).get("mechanical_pass") is False]
    result["overall_classification"] = "PASS" if not failures else "OBJECTIVE_OR_HARNESS_FAILURES_PRESENT"
    result["total_elapsed_seconds"] = 0.0
    write_json(output_root / "model.json", result)
    summary = {
        "experiment_id": result["experiment_id"],
        "model_label": args.label,
        "split": args.split,
        "case_count": result["case_count"],
        "overall_classification": result["overall_classification"],
        "objective_pass": sum(bool(item.get("checks", {}).get("objective_pass")) for item in result["cases"]),
        "semi_objective_pass": sum(bool(item.get("checks", {}).get("semi_objective_pass")) for item in result["cases"]),
        "exceptions": sum(bool(item.get("exception")) for item in result["cases"]),
        "source_raw_run": str(raw_root.resolve()),
        "suite_sha256": result["suite_sha256"],
    }
    write_json(output_root / "logs" / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
