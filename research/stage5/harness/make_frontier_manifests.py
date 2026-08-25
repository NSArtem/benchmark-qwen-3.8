#!/usr/bin/env python3
"""Freeze explicit Q1/Q2/Q3 promotion manifests from one ranking."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensitivity", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    sensitivity_path = Path(args.sensitivity).resolve()
    baseline_path = Path(args.baseline).resolve()
    output_path = Path(args.output).resolve()
    data = json.loads(sensitivity_path.read_text(encoding="utf-8"))
    if not data.get("complete") or len(data.get("records", [])) != 498:
        raise ValueError("sensitivity artifact is incomplete")
    if "heldout" in data["calibration_path"].lower():
        raise ValueError("held-out data cannot allocate candidates")
    records = data["records"]
    ranking_hash = canonical_hash(
        [
            {
                "path": item["path"],
                "quality_recovery_per_byte": item["quality_recovery_per_byte"],
                "bytes_added_for_promotion": item["bytes_added_for_promotion"],
            }
            for item in records
        ]
    )
    baseline_config = json.loads((baseline_path / "config.json").read_text())
    baseline_index = json.loads((baseline_path / "model.safetensors.index.json").read_text())
    baseline_disk_bytes = sum(
        int(path.stat().st_size)
        for path in baseline_path.glob("*.safetensors")
    )
    baseline_bytes = int(baseline_index["metadata"]["total_size"])
    parameter_count = int(baseline_index["metadata"]["total_parameters"])
    candidates = []
    for name, budget_mib in (("Q1", 150), ("Q2", 300), ("Q3", 450)):
        budget_bytes = budget_mib * 1024 * 1024
        selected = []
        selected_bytes = 0
        for item in records:
            extra = int(item["bytes_added_for_promotion"])
            if selected_bytes + extra <= budget_bytes:
                selected.append(item)
                selected_bytes += extra
        output_bytes = baseline_bytes + selected_bytes
        candidates.append(
            {
                "name": name,
                "budget_mib": budget_mib,
                "selection": "greedy frozen KL recovery per added byte",
                "selected_paths": [item["path"] for item in selected],
                "selected_module_count": len(selected),
                "selected_extra_bytes": selected_bytes,
                "selected_extra_mib": selected_bytes / (1024 * 1024),
                "predicted_output_bytes": output_bytes,
                "predicted_output_gib": output_bytes / (1 << 30),
                "predicted_effective_bpw": output_bytes * 8 / parameter_count,
                "family_counts": {
                    family: sum(item["family"] == family for item in selected)
                    for family in sorted({item["family"] for item in selected})
                },
            }
        )
    result = {
        "experiment_id": "P4-FIXED-BUDGET-MANIFESTS",
        "phase": 4,
        "sensitivity_artifact": str(sensitivity_path),
        "sensitivity_sha256": hashlib.sha256(sensitivity_path.read_bytes()).hexdigest(),
        "calibration_path": data["calibration_path"],
        "calibration_sha256": data["calibration_sha256"],
        "ranking_hash": ranking_hash,
        "heldout_excluded": True,
        "baseline_path": str(baseline_path),
        "baseline_bytes_from_shards": baseline_bytes,
        "baseline_disk_bytes": baseline_disk_bytes,
        "baseline_bytes_from_config": baseline_config.get("quantization"),
        "parameter_count": parameter_count,
        "candidates": candidates,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ranking_hash": ranking_hash, "candidates": [(x["name"], x["selected_module_count"], x["selected_extra_mib"]) for x in candidates]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
