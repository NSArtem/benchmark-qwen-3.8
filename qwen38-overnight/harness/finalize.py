"""Add derived final verdicts and packaging metadata to the overnight registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--archive-path")
    parser.add_argument("--archive-size-bytes", type=int)
    parser.add_argument("--git-verification")
    args = parser.parse_args()

    registry = read_json(args.registry)
    quality = {
        item.get("candidate"): item
        for item in registry.get("quality", [])
        if isinstance(item, dict)
    }
    for run in registry.get("runs", []):
        if not isinstance(run, dict):
            continue
        candidate = run.get("candidate")
        stage = run.get("stage")
        metrics = run.setdefault("metrics", {})
        if stage == "target-practical":
            smoke = quality.get(candidate)
            if smoke:
                metrics["quality_smoke_results"] = {
                    "passed": smoke.get("passed"),
                    "total": smoke.get("total"),
                }
            if metrics.get("paging_assessment") in {
                "severe_sustained_paging",
                "meaningful_sustained_paging",
            }:
                metrics["final_verdict"] = "not_practically_viable_due_to_sustained_paging"
            elif smoke and smoke.get("passed") == 0:
                metrics["final_verdict"] = "stable_but_quality_smoke_failed_0_of_5"
            elif run.get("result", {}).get("status") == "complete":
                metrics["final_verdict"] = "completed_with_recorded_quality_smoke"
        if run.get("supervisor", {}).get("watchdog_abort"):
            metrics["final_verdict"] = "aborted_by_conservative_watchdog"

    if args.archive_path is not None:
        registry["archive"] = {
            "path": args.archive_path,
            "size_bytes": args.archive_size_bytes,
            "weights_and_caches_excluded": True,
        }
    if args.git_verification is not None:
        registry["git_verification"] = args.git_verification
    write_json(args.registry, registry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
