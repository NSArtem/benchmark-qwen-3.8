#!/usr/bin/env python
"""Add compact comparable activation records to results.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import update_results


def flatten_file(path: Path, phase_name: str | None = None) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    if phase_name is None:
        phases = [(None, payload.get("checkpoints", {}))]
    else:
        phases = [
            ("prefill", payload.get("prefill_checkpoints", {})),
            ("decode", payload.get("decode_checkpoints", {})),
        ]
    for phase, checkpoints in phases:
        for checkpoint, values in checkpoints.items():
            record = {
                "test_id": f"forward-fingerprint-{path.stem}-{phase or 'whole'}-{checkpoint}",
                "process_index": payload["process_index"],
                "prompt_hash": payload.get("prompt_sha256"),
                "token_ids_hash": payload.get("token_ids_hash"),
                "checkpoint": checkpoint,
                "phase": phase or "whole_prompt",
                "layer_index": values.get("layer_index"),
                "layer_type": values.get("layer_type"),
                "is_linear": values.get("is_linear"),
                "tensor_shape": values.get("tensor_shape"),
                "tensor_dtype": values.get("tensor_dtype"),
                "tensor_sum": values.get("tensor_sum"),
                "tensor_mean": values.get("tensor_mean"),
                "tensor_std": values.get("tensor_std"),
                "tensor_l2": values.get("tensor_l2"),
                "tensor_fingerprint": values.get("tensor_fingerprint"),
                "nan_count": values.get("nan_count"),
                "inf_count": values.get("inf_count"),
                "agrees_with_process_0": None,
                "max_abs_difference_if_available": None,
            }
            records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="qwen38-mlx-quantmatmul")
    args = parser.parse_args()
    root = Path(args.root)
    records: list[dict[str, Any]] = []
    for path in sorted((root / "forward-fingerprints").glob("process-*.json")):
        records.extend(flatten_file(path))
    for path in sorted((root / "forward-fingerprints").glob("chunked-process-*.json")):
        records.extend(flatten_file(path, phase_name="chunked"))

    baselines = {
        (item["prompt_hash"], item["phase"], item["checkpoint"]): item
        for item in records
        if item["process_index"] == 0
    }
    for item in records:
        baseline = baselines.get((item["prompt_hash"], item["phase"], item["checkpoint"]))
        if baseline is not None:
            item["agrees_with_process_0"] = item["tensor_fingerprint"] == baseline["tensor_fingerprint"]

    results_path = root / "results.json"
    for record in records:
        update_results(results_path, record, collection="forward_fingerprints")
    print(json.dumps({
        "records": len(records),
        "agreeing": sum(item["agrees_with_process_0"] is True for item in records),
        "disagreeing": sum(item["agrees_with_process_0"] is False for item in records),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
