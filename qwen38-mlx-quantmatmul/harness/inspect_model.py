#!/usr/bin/env python
"""Inspect the actual Qwen3.8 checkpoint and selected quantized modules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlx_lm import load

from common import DEFAULT_MODEL_PATH, MODEL_ID, MLX_LM_COMMIT, module_metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", default="qwen38-mlx-quantmatmul")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load(args.model_path)
    named = dict(model.named_modules())
    quantized = {
        path: module_metadata(module)
        for path, module in named.items()
        if type(module).__name__ == "QuantizedLinear"
    }

    candidates = {
        "gated_delta_in_proj_qkv": "language_model.model.layers.0.linear_attn.in_proj_qkv",
        "gated_delta_in_proj_z": "language_model.model.layers.0.linear_attn.in_proj_z",
        "gated_delta_in_proj_a": "language_model.model.layers.0.linear_attn.in_proj_a",
        "gated_delta_in_proj_b": "language_model.model.layers.0.linear_attn.in_proj_b",
        "gated_delta_out_proj": "language_model.model.layers.0.linear_attn.out_proj",
        "full_attention_q_proj": "language_model.model.layers.3.self_attn.q_proj",
        "full_attention_k_proj": "language_model.model.layers.3.self_attn.k_proj",
        "full_attention_v_proj": "language_model.model.layers.3.self_attn.v_proj",
        "mlp_gate_proj": "language_model.model.layers.0.mlp.gate_proj",
        "mlp_up_proj": "language_model.model.layers.0.mlp.up_proj",
        "mlp_down_proj": "language_model.model.layers.0.mlp.down_proj",
        "lm_head": "language_model.lm_head",
    }
    selected = {}
    for label, path in candidates.items():
        if path in quantized:
            selected[label] = path
        else:
            matches = [candidate for candidate in quantized if candidate.endswith(path.split(".", 3)[-1])]
            if matches:
                selected[label] = matches[0]

    config_path = Path(args.model_path) / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else None
    manifest = {
        "model": MODEL_ID,
        "mlx_lm_commit": MLX_LM_COMMIT,
        "model_path": args.model_path,
        "model_class": f"{type(model).__module__}.{type(model).__name__}",
        "tokenizer_class": f"{type(tokenizer).__module__}.{type(tokenizer).__name__}",
        "model_args": getattr(model, "args", None),
        "config": config,
        "quantized_linear_count": len(quantized),
        "selected": selected,
        "modules": quantized,
    }
    (output_dir / "module-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "model_class": manifest["model_class"],
        "quantized_linear_count": len(quantized),
        "selected": selected,
        "selected_metadata": {label: quantized[path] for label, path in selected.items()},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
