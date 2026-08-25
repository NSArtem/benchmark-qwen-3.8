#!/usr/bin/env python3
"""Summarize the bounded sensitivity map and write promotion manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def mixed_3_4(path: str, num_layers: int = 64) -> bool:
    """Match pinned mlx-lm mixed_quant_predicate_builder('mixed_3_4')."""

    if "lm_head" in path:
        return True
    match = LAYER_RE.search(path)
    index = int(match.group(1)) if match else 0
    use_more_bits = (
        index < num_layers // 8
        or index >= 7 * num_layers // 8
        or (index - num_layers // 8) % 3 == 2
    )
    return (
        ("v_proj" in path or "v_a_proj" in path or "v_b_proj" in path)
        and use_more_bits
    ) or ("down_proj" in path and use_more_bits)


def module_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["family"], []).append(record)
    result = []
    for family, items in sorted(grouped.items()):
        result.append(
            {
                "family": family,
                "count": len(items),
                "mean_relative_rms_output_delta": sum(
                    item["relative_rms_output_delta"] for item in items
                )
                / len(items),
                "mean_rms_output_delta": sum(item["rms_output_delta"] for item in items)
                / len(items),
                "total_extra_payload_bytes": sum(item["extra_payload_bytes"] for item in items),
                "top_by_delta_energy_per_parameter": max(
                    items, key=lambda item: item["delta_energy_per_parameter"]
                )["path"],
                "top_by_relative_rms": max(
                    items, key=lambda item: item["relative_rms_output_delta"]
                )["path"],
            }
        )
    return result


def candidate_record(
    name: str,
    recipe: str,
    selected: list[dict[str, Any]],
    base_bytes: int,
    base_gib: float,
    selection: dict[str, Any],
) -> dict[str, Any]:
    paths = sorted(item["path"] for item in selected)
    extra = sum(item["extra_payload_bytes"] for item in selected)
    return {
        "name": name,
        "recipe": recipe,
        "bits_low": 3,
        "bits_high": 4,
        "group_size": 128,
        "mode": "affine",
        "selected_module_count": len(paths),
        "selected_paths": paths,
        "selection": selection,
        "predicted_extra_payload_bytes": extra,
        "predicted_extra_payload_gib": extra / (1 << 30),
        "predicted_output_bytes": base_bytes + extra,
        "predicted_output_gib": (base_bytes + extra) / (1 << 30),
        "base_g128_bytes": base_bytes,
        "base_g128_gib": base_gib,
        "manifest_sha256": sha256_json(paths),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--sensitivity-budget-mib",
        type=float,
        default=384.0,
        help="maximum predicted extra payload for the sensitivity candidate",
    )
    args = parser.parse_args()

    map_path = Path(args.map).resolve()
    baseline_path = Path(args.baseline).resolve()
    output_path = Path(args.output).resolve()
    mapped = read_json(map_path)
    records = mapped["records"]
    if not mapped.get("complete") or len(records) != 498:
        raise ValueError("sensitivity map must be complete with 498 records")

    baseline_index = read_json(baseline_path / "model.safetensors.index.json")
    base_bytes = int(baseline_index["metadata"]["total_size"])
    base_gib = base_bytes / (1 << 30)

    by_path = {record["path"]: record for record in records}
    builtin_paths = sorted(path for path in by_path if mixed_3_4(path))
    builtin_selected = [by_path[path] for path in builtin_paths]

    budget_bytes = int(args.sensitivity_budget_mib * (1 << 20))
    ranked = sorted(
        records,
        key=lambda item: (
            -item["delta_energy_per_parameter"],
            -item["relative_rms_output_delta"],
            item["path"],
        ),
    )
    selected: list[dict[str, Any]] = []
    selected_bytes = 0
    skipped_large: list[str] = []
    for record in ranked:
        extra = int(record["extra_payload_bytes"])
        if selected_bytes + extra <= budget_bytes:
            selected.append(record)
            selected_bytes += extra
        else:
            skipped_large.append(record["path"])

    builtin = candidate_record(
        "mixed-3-4-control-g128",
        "mixed_3_4_pinned_predicate",
        builtin_selected,
        base_bytes,
        base_gib,
        {
            "kind": "pinned_upstream_predicate",
            "source": "mlx_lm.convert.mixed_quant_predicate_builder",
            "num_layers": 64,
        },
    )
    sensitivity = candidate_record(
        "sensitivity-3-4-gain-per-parameter",
        "sensitivity_3_4",
        selected,
        base_bytes,
        base_gib,
        {
            "kind": "greedy_budgeted_rank",
            "rank_key": "delta_energy_per_parameter",
            "proxy": mapped["method"],
            "budget_bytes": budget_bytes,
            "budget_mib": args.sensitivity_budget_mib,
            "skipped_large_count": len(skipped_large),
        },
    )

    summary = {
        "experiment_id": "P3-SENSITIVITY-SUMMARY",
        "phase": 3,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "map_path": str(map_path),
        "map_sha256": hashlib.sha256(map_path.read_bytes()).hexdigest(),
        "calibration_sha256": mapped["calibration_sha256"],
        "record_count": len(records),
        "module_summary": module_summary(records),
        "top_by_delta_energy_per_parameter": [
            {
                "path": item["path"],
                "family": item["family"],
                "layer_index": item["layer_index"],
                "delta_energy_per_parameter": item["delta_energy_per_parameter"],
                "relative_rms_output_delta": item["relative_rms_output_delta"],
                "extra_payload_bytes": item["extra_payload_bytes"],
            }
            for item in ranked[:50]
        ],
        "top_by_relative_rms_output_delta": [
            {
                "path": item["path"],
                "family": item["family"],
                "layer_index": item["layer_index"],
                "delta_energy_per_parameter": item["delta_energy_per_parameter"],
                "relative_rms_output_delta": item["relative_rms_output_delta"],
                "extra_payload_bytes": item["extra_payload_bytes"],
            }
            for item in sorted(records, key=lambda item: -item["relative_rms_output_delta"])[:50]
        ],
        "candidates": [builtin, sensitivity],
        "candidate_census": {
            "builtin_high_bit_module_count": len(builtin_paths),
            "sensitivity_high_bit_module_count": len(selected),
            "sensitivity_budget_bytes": budget_bytes,
            "sensitivity_selected_bytes": selected_bytes,
        },
        "future_stage5": {
            "dynamic_quant": "not run: pinned implementation deep-copies full 27B model and is unsafe on 18 GB",
            "dwq": "not run: learned teacher/distillation path requires a separate bounded feasibility study",
            "awq": "not run: calibration search and full-model/block residency need a dedicated safety audit",
            "gptq": "not run: Hessian/calibration path needs a dedicated bounded feasibility study",
        },
    }
    write_json(output_path, summary)
    print(json.dumps({
        "builtin_modules": len(builtin_paths),
        "builtin_gib": builtin["predicted_output_gib"],
        "sensitivity_modules": len(selected),
        "sensitivity_gib": sensitivity["predicted_output_gib"],
        "sensitivity_extra_mib": sensitivity["predicted_extra_payload_bytes"] / (1 << 20),
        "output": str(output_path),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
