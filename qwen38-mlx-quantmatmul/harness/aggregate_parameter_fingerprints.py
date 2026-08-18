#!/usr/bin/env python
"""Add compact fresh-process stored-parameter fingerprints to results.json."""

from __future__ import annotations

import json
from pathlib import Path

from common import update_results


def main() -> None:
    root = Path("qwen38-mlx-quantmatmul")
    results_path = root / "results.json"
    for path in sorted((root / "fresh-process").glob("parameter-fingerprint-p*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        compact = {}
        for label, values in payload["fingerprints"].items():
            compact[label] = {
                "path": values["path"],
                "weight_hash": values["weight_hash"],
                "scales_hash": values["scales_hash"],
                "biases_hash": values["biases_hash"],
            }
        update_results(
            results_path,
            {
                "test_id": payload["test_id"],
                "model": payload["model"],
                "mlx_lm_commit": payload["mlx_lm_commit"],
                "process_index": payload["process_index"],
                "fingerprints": compact,
                "all_selected_hashes_present": payload["all_selected_hashes_present"],
            },
            collection="parameter_fingerprints",
        )
    print(json.dumps({"processes": len(list((root / "fresh-process").glob("parameter-fingerprint-p*.json")))}, sort_keys=True))


if __name__ == "__main__":
    main()
