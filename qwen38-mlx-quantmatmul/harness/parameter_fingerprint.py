#!/usr/bin/env python
"""Fingerprint selected stored quantized parameters in one fresh process."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlx_lm import load

from common import DEFAULT_MODEL_PATH, MLX_LM_COMMIT, MODEL_ID, module_metadata, resolve_module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--manifest", default="qwen38-mlx-quantmatmul/module-manifest.json")
    parser.add_argument("--process-index", type=int, required=True)
    parser.add_argument("--output-dir", default="qwen38-mlx-quantmatmul/fresh-process")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    model, _ = load(args.model_path)
    fingerprints = {}
    for label, path in sorted(manifest["selected"].items()):
        module = resolve_module(model, path)
        fingerprints[label] = {
            "path": path,
            **module_metadata(module),
        }

    result = {
        "test_id": f"parameter-fingerprint-process-{args.process_index:02d}",
        "model": MODEL_ID,
        "mlx_lm_commit": MLX_LM_COMMIT,
        "process_index": args.process_index,
        "model_class": f"{type(model).__module__}.{type(model).__name__}",
        "fingerprints": fingerprints,
        "all_selected_hashes_present": all(
            item["weight_hash"] and item["scales_hash"] for item in fingerprints.values()
        ),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"parameter-fingerprint-p{args.process_index:02d}.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
