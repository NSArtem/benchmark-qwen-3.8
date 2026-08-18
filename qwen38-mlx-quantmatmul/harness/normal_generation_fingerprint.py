#!/usr/bin/env python
"""Record the first greedy token from the pinned normal mlx-lm generation path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.generate import generate_step

from common import DEFAULT_MODEL_PATH, MLX_LM_COMMIT, MODEL_ID, materialize, module_metadata, resolve_module, update_results
from forward_fingerprint import chat_ids, sha256_bytes, token_hash


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--prompt", default="qwen38-mlx-threshold/prompts/0512.txt")
    parser.add_argument("--process-index", type=int, required=True)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--manifest", default="qwen38-mlx-quantmatmul/module-manifest.json")
    parser.add_argument("--results", default="qwen38-mlx-quantmatmul/results.json")
    parser.add_argument("--output-dir", default="qwen38-mlx-quantmatmul/outputs")
    args = parser.parse_args()

    mx.set_default_device(mx.gpu)
    model, tokenizer = load(args.model_path)
    prompt_text = Path(args.prompt).read_text(encoding="utf-8")
    ids = chat_ids(tokenizer, prompt_text)
    prompt = mx.array(ids)
    generator = generate_step(
        prompt,
        model,
        max_tokens=1,
        prefill_step_size=args.prefill_step_size,
        sampler=lambda logprobs: mx.argmax(logprobs, axis=-1),
    )
    token_array, logprobs = next(generator)
    mx.eval(token_array, logprobs)
    mx.synchronize()
    token_id = int(token_array if isinstance(token_array, (int, np.integer)) else token_array.item())
    logprobs_host = materialize(logprobs)
    top_ids = np.argsort(-logprobs_host.astype(np.float64))[:10]
    top10 = [
        {"token_id": int(token_id_value), "logprob": float(logprobs_host[token_id_value])}
        for token_id_value in top_ids
    ]
    try:
        token_text = tokenizer.decode([token_id])
    except Exception:
        token_text = None

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    qkv_path = manifest["selected"]["gated_delta_in_proj_qkv"]
    qkv_meta = module_metadata(resolve_module(model, qkv_path))
    result = {
        "test_id": f"normal-generation-first-token-{Path(args.prompt).stem}-process-{args.process_index:02d}",
        "model": MODEL_ID,
        "mlx_lm_commit": MLX_LM_COMMIT,
        "process_index": args.process_index,
        "prompt_file": args.prompt,
        "prompt_sha256": sha256_bytes(prompt_text.encode("utf-8")),
        "prompt_tokens": len(ids),
        "token_ids_hash": token_hash(ids),
        "prefill_step_size": args.prefill_step_size,
        "first_token_id": token_id,
        "first_token_text": token_text,
        "first_token_top10": top10,
        "qkv_weight_hash": qkv_meta["weight_hash"],
        "qkv_scales_hash": qkv_meta["scales_hash"],
        "qkv_biases_hash": qkv_meta["biases_hash"],
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"normal-generation-first-token-{Path(args.prompt).stem}-p{args.process_index:02d}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    update_results(Path(args.results), result, collection="normal_generation_first_token")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
