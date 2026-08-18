#!/usr/bin/env python
"""Fingerprint the exact mlx-lm prefill-then-final-token generation schedule."""

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

from common import (
    DEFAULT_MODEL_PATH,
    MLX_LM_COMMIT,
    MODEL_ID,
    materialize,
    module_metadata,
    resolve_module,
    update_results,
    write_json,
)
from forward_fingerprint import chat_ids, sha256_bytes, tensor_fingerprint, token_hash


def phase_forward(
    model: Any,
    ids: list[int],
    cache: list[Any],
    wanted_layers: set[int],
) -> tuple[dict[str, Any], mx.array]:
    """Copy Qwen3.5's text-model loop without evaluating hidden states early."""

    text_model = model.model
    inputs = mx.array([ids])
    hidden = text_model.embed_tokens(inputs)
    fa_mask = None
    ssm_mask = None
    if text_model.fa_idx is not None:
        fa_mask = create_attention_mask(hidden, cache[text_model.fa_idx])
    if text_model.ssm_idx is not None:
        ssm_mask = create_ssm_mask(hidden, cache[text_model.ssm_idx])

    tensors: dict[str, Any] = {"embeddings": hidden}
    for index, (layer, layer_cache) in enumerate(zip(text_model.pipeline_layers, cache)):
        mask = ssm_mask if layer.is_linear else fa_mask
        hidden = layer(hidden, mask=mask, cache=layer_cache)
        if index in wanted_layers:
            tensors[f"layer_{index}"] = {
                "layer_index": index,
                "layer_type": type(layer).__name__,
                "is_linear": bool(layer.is_linear),
                "tensor": hidden,
            }

    normalized = text_model.norm(hidden)
    tensors["final_norm"] = normalized
    return tensors, normalized


def evaluate_phase(tensors: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Fingerprint a phase after its normal cache synchronization point."""

    result: dict[str, dict[str, Any]] = {}
    for key, value in tensors.items():
        if isinstance(value, dict):
            result[key] = {
                key_name: item for key_name, item in value.items() if key_name != "tensor"
            }
            result[key].update(tensor_fingerprint(value["tensor"]))
        else:
            result[key] = tensor_fingerprint(value)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--prompt", default="qwen38-mlx-threshold/prompts/0512.txt")
    parser.add_argument("--process-index", type=int, required=True)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--layers", default="0,7,15,23,31,39,47,55,63")
    parser.add_argument("--manifest", default="qwen38-mlx-quantmatmul/module-manifest.json")
    parser.add_argument("--output-dir", default="qwen38-mlx-quantmatmul/forward-fingerprints")
    parser.add_argument("--results", default="qwen38-mlx-quantmatmul/results.json")
    args = parser.parse_args()

    mx.set_default_device(mx.gpu)
    model, tokenizer = load(args.model_path)
    prompt_text = Path(args.prompt).read_text(encoding="utf-8")
    ids = chat_ids(tokenizer, prompt_text)
    if len(ids) < 2:
        raise ValueError("The generation schedule requires at least two prompt tokens")
    wanted_layers = {int(value) for value in args.layers.split(",") if value}
    cache = model.language_model.make_cache()

    total = len(ids)
    prefill_count = min(args.prefill_step_size, total - 1)
    prefill_ids = ids[:prefill_count]
    remaining_ids = ids[prefill_count:]
    prefill_tensors, _ = phase_forward(model, prefill_ids, cache, wanted_layers)
    # This is the synchronization point used by mlx_lm.generate_step after each
    # prompt chunk.  The following clear is also part of that normal path.
    mx.eval([item.state for item in cache])
    mx.synchronize()
    prefill_fingerprints = evaluate_phase(prefill_tensors)
    mx.clear_cache()

    final_tensors, normalized = phase_forward(model, remaining_ids, cache, wanted_layers)
    logits = model.language_model.lm_head(normalized)
    last_logits = logits[:, -1, :]
    mx.eval(last_logits)
    mx.synchronize()
    decode_fingerprints = evaluate_phase(final_tensors)
    logits_host = materialize(last_logits)[0]
    top_ids = np.argsort(-logits_host.astype(np.float64))[:10]
    top10 = [
        {"token_id": int(token_id), "logit": float(logits_host[token_id])}
        for token_id in top_ids
    ]
    first_token_id = int(top_ids[0])
    try:
        first_token_text = tokenizer.decode([first_token_id])
    except Exception:
        first_token_text = None

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    representative_hashes = {}
    for label, path in sorted(manifest["selected"].items()):
        representative_hashes[label] = {
            "path": path,
            **module_metadata(resolve_module(model, path)),
        }

    result = {
        "test_id": f"chunked-forward-fingerprint-process-{args.process_index:02d}",
        "model": MODEL_ID,
        "mlx_lm_commit": MLX_LM_COMMIT,
        "process_index": args.process_index,
        "prompt_file": args.prompt,
        "prompt_sha256": sha256_bytes(prompt_text.encode("utf-8")),
        "prompt_tokens": len(ids),
        "token_ids_hash": token_hash(ids),
        "prefill_step_size": args.prefill_step_size,
        "prefill_tokens": len(prefill_ids),
        "final_decode_tokens": len(remaining_ids),
        "first_token_id": first_token_id,
        "first_token_text": first_token_text,
        "first_token_top10": top10,
        "prefill_checkpoints": prefill_fingerprints,
        "decode_checkpoints": decode_fingerprints,
        "representative_parameter_hashes": representative_hashes,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / f"chunked-process-{args.process_index:02d}.json", result)
    update_results(Path(args.results), result, collection="chunked_forward_fingerprints")
    print(json.dumps({
        "test_id": result["test_id"],
        "process_index": args.process_index,
        "prompt_tokens": result["prompt_tokens"],
        "prefill_tokens": result["prefill_tokens"],
        "final_decode_tokens": result["final_decode_tokens"],
        "first_token_id": result["first_token_id"],
        "first_token_text": result["first_token_text"],
        "top10": top10,
        "prefill_fingerprints": {key: value["tensor_fingerprint"] for key, value in prefill_fingerprints.items()},
        "decode_fingerprints": {key: value["tensor_fingerprint"] for key, value in decode_fingerprints.items()},
    }, sort_keys=True))


if __name__ == "__main__":
    main()
