#!/usr/bin/env python
"""Fingerprint evaluated Qwen3.8 hidden states after selected decoder layers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask, create_ssm_mask

from common import DEFAULT_MODEL_PATH, MLX_LM_COMMIT, MODEL_ID, materialize, module_metadata, resolve_module, write_json


def chat_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    elif hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    if encoded and isinstance(encoded[0], (list, tuple)):
        encoded = encoded[0]
    return [int(value) for value in encoded]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def token_hash(ids: list[int]) -> str:
    return sha256_bytes(np.asarray(ids, dtype=np.int32).tobytes(order="C"))


def tensor_fingerprint(value: mx.array) -> dict[str, Any]:
    """Evaluate and hash only a deterministic sample, not a tensor dump."""

    mx.eval(value)
    mx.synchronize()
    host = materialize(value)
    flat = host.reshape(-1)
    sample_count = min(4096, flat.size)
    if flat.size <= sample_count * 2:
        sample = flat
    else:
        stride = max(1, flat.size // sample_count)
        sample = np.concatenate((flat[:sample_count], flat[-sample_count:], flat[::stride][:sample_count]))
    return {
        "tensor_shape": list(map(int, value.shape)),
        "tensor_dtype": str(value.dtype),
        "tensor_sum": float(np.sum(host, dtype=np.float64)),
        "tensor_mean": float(np.mean(host, dtype=np.float64)),
        "tensor_std": float(np.std(host, dtype=np.float64)),
        "tensor_l2": float(np.linalg.norm(host.astype(np.float64))),
        "tensor_min": float(np.min(host)) if host.size else None,
        "tensor_max": float(np.max(host)) if host.size else None,
        "nan_count": int(np.isnan(host).sum()),
        "inf_count": int(np.isinf(host).sum()),
        "tensor_fingerprint": sha256_bytes(np.asarray(sample, dtype=np.float32).tobytes(order="C")),
    }


def run_forward(model: Any, ids: list[int], wanted_layers: set[int]) -> dict[str, dict[str, Any]]:
    text_model = model.model
    cache = model.language_model.make_cache()
    hidden = text_model.embed_tokens(mx.array([ids]))
    checkpoints: dict[str, dict[str, Any]] = {"embeddings": tensor_fingerprint(hidden)}

    fa_mask = None
    ssm_mask = None
    if text_model.fa_idx is not None:
        fa_mask = create_attention_mask(hidden, cache[text_model.fa_idx])
    if text_model.ssm_idx is not None:
        ssm_mask = create_ssm_mask(hidden, cache[text_model.ssm_idx])

    for index, (layer, layer_cache) in enumerate(zip(text_model.pipeline_layers, cache)):
        mask = ssm_mask if layer.is_linear else fa_mask
        hidden = layer(hidden, mask=mask, cache=layer_cache)
        if index in wanted_layers:
            checkpoints[f"layer_{index}"] = {
                "layer_index": index,
                "layer_type": type(layer).__name__,
                "is_linear": bool(layer.is_linear),
                **tensor_fingerprint(hidden),
            }

    normalized = text_model.norm(hidden)
    checkpoints["final_norm"] = tensor_fingerprint(normalized)
    return checkpoints


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--prompt", default="qwen38-mlx-threshold/prompts/0512.txt")
    parser.add_argument("--process-index", type=int, required=True)
    parser.add_argument("--layers", default="0,7,15,23,31,39,47,55,63")
    parser.add_argument("--manifest", default="qwen38-mlx-quantmatmul/module-manifest.json")
    parser.add_argument("--output-dir", default="qwen38-mlx-quantmatmul/forward-fingerprints")
    args = parser.parse_args()

    mx.set_default_device(mx.gpu)
    model, tokenizer = load(args.model_path)
    prompt_text = Path(args.prompt).read_text(encoding="utf-8")
    ids = chat_ids(tokenizer, prompt_text)
    wanted_layers = {int(value) for value in args.layers.split(",") if value}
    checkpoints = run_forward(model, ids, wanted_layers)

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    representative_hashes = {}
    for label, path in sorted(manifest["selected"].items()):
        representative_hashes[label] = {
            "path": path,
            **module_metadata(resolve_module(model, path)),
        }

    result = {
        "test_id": f"forward-fingerprint-process-{args.process_index:02d}",
        "model": MODEL_ID,
        "mlx_lm_commit": MLX_LM_COMMIT,
        "process_index": args.process_index,
        "prompt_file": args.prompt,
        "prompt_sha256": sha256_bytes(prompt_text.encode("utf-8")),
        "prompt_tokens": len(ids),
        "token_ids_hash": token_hash(ids),
        "layer_count": len(model.model.layers),
        "checkpoints": checkpoints,
        "representative_parameter_hashes": representative_hashes,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / f"process-{args.process_index:02d}.json", result)
    print(json.dumps({
        "test_id": result["test_id"],
        "process_index": args.process_index,
        "prompt_tokens": result["prompt_tokens"],
        "token_ids_hash": result["token_ids_hash"],
        "checkpoints": {key: value["tensor_fingerprint"] for key, value in checkpoints.items()},
    }, sort_keys=True))


if __name__ == "__main__":
    main()
