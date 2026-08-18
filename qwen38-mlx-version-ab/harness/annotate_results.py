#!/usr/bin/env python3
"""Apply human-reviewed classifications to completed A/B run records."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("qwen38-mlx-version-ab")
CLASSIFICATIONS = {
    "mlx0320-short": ("COHERENT", "normal prose/code answer"),
    "mlx0321-short": ("COHERENT", "normal prose/code answer"),
    "mlx0320-0768-r1": ("BROKEN", "repeated irl/newline token pattern"),
    "mlx0320-0768-r2": ("BROKEN", "repeated assoluto/assoluzione fragments"),
    "mlx0320-0768-r3": ("BROKEN", "repeated Chinese token 中 with spaces"),
    "mlx0321-0768-r1": ("BROKEN", "no completion; Metal Insufficient Memory before first token"),
    "mlx0321-0768-r2": ("BROKEN", "no completion; Metal Insufficient Memory before first token"),
    "mlx0321-0768-r3": ("BROKEN", "no completion; Metal Insufficient Memory before first token"),
}


def main() -> None:
    aggregate_path = ROOT / "results.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    by_id = {item["test_id"]: item for item in aggregate["tests"]}
    for test_id, (classification, note) in CLASSIFICATIONS.items():
        item = by_id[test_id]
        item["classification"] = classification
        item["classification_note"] = note
        per_run = Path(item["log_json_file"])
        per_run_payload = json.loads(per_run.read_text(encoding="utf-8"))
        per_run_payload["classification"] = classification
        per_run_payload["classification_note"] = note
        per_run.write_text(json.dumps(per_run_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    aggregate["tests"] = [by_id[item["test_id"]] for item in aggregate["tests"]]
    aggregate["classification_reviewed"] = True
    aggregate_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
