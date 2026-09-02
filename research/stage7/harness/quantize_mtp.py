#!/usr/bin/env python3
"""Create the single allowed source-backed Qwen3.8 MTP 4-bit artifact."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--bits", type=int, default=4)
    args = parser.parse_args()

    from mlx_vlm.speculative.drafters.qwen3_5_mtp import Qwen3_5MTPDraftModel
    from mlx_vlm.speculative.drafters.qwen3_5_mtp.config import Qwen3_5MTPConfig

    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    source_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    draft_config = Qwen3_5MTPConfig.from_dict(
        {"model_type": "qwen3_5_mtp", "text_config": source_config["text_config"]}
    )
    draft = Qwen3_5MTPDraftModel(draft_config)

    # The original HF source keeps the 15 tensors under ``mtp.*`` in the
    # final shard.  Published standalone MLX MTP sidecars contain the same
    # tensors under their draft-model keys and use ``model.safetensors``.
    # Accept both layouts so a source-backed MTP artifact can be recovered
    # without downloading the full target checkpoint.
    shard_path = source / "model-00018-of-00018.safetensors"
    if not shard_path.exists():
        shard_path = source / "model.safetensors"
    if not shard_path.exists():
        raise FileNotFoundError(
            f"no MTP safetensors file found under {source}"
        )
    shard = mx.load(str(shard_path))
    source_weights = {key: value for key, value in shard.items() if key.startswith("mtp.")}
    source_layout = "hf_mtp_prefixed"
    if not source_weights:
        source_weights = dict(shard)
        source_layout = "standalone_mlx_draft"
    if len(source_weights) != 15:
        raise RuntimeError(f"expected 15 MTP tensors, found {len(source_weights)}")
    source_payload_bytes = sum(int(value.nbytes) for value in source_weights.values())
    if source_layout == "hf_mtp_prefixed":
        source_weights = Qwen3_5MTPDraftModel.sanitize(None, source_weights)
    draft.load_weights(list(source_weights.items()))
    mx.eval(draft.parameters())
    del shard, source_weights
    gc.collect()
    mx.clear_cache()

    def linear_only(_path: str, module: nn.Module) -> bool:
        return isinstance(module, nn.Linear)

    nn.quantize(
        draft,
        group_size=args.group_size,
        bits=args.bits,
        mode="affine",
        class_predicate=linear_only,
    )
    mx.eval(draft.parameters())
    flat = dict(tree_flatten(draft.parameters()))
    if not flat or not any(key.endswith(".scales") for key in flat):
        raise RuntimeError("MTP quantization produced no quantized parameter keys")
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "model.safetensors"
    mx.save_safetensors(str(model_path), flat, metadata={"format": "mlx"})
    config = {
        "model_type": "qwen3_5_mtp",
        "text_config": source_config["text_config"],
        "block_size": int(draft_config.block_size),
        "tie_word_embeddings": bool(source_config["text_config"].get("tie_word_embeddings", False)),
        "quantization": {"bits": int(args.bits), "group_size": int(args.group_size), "mode": "affine"},
        "source_mtp_payload_bytes": source_payload_bytes,
        "source_tensor_count": 15,
    }
    write_json(output / "config.json", config)
    quantized_payload_bytes = sum(int(value.nbytes) for value in flat.values())
    write_json(
        output / "manifest.json",
        {
            "source_path": str(source),
            "source_weights_file": str(shard_path),
            "source_layout": source_layout,
            "source_mtp_payload_bytes": source_payload_bytes,
            "source_tensor_count": 15,
            "precision": f"{args.bits}-bit",
            "group_size": int(args.group_size),
            "mode": "affine",
            "quantized_parameter_bytes": quantized_payload_bytes,
            "filesystem_bytes": model_path.stat().st_size,
            "parameter_keys": sorted(flat),
        },
    )
    print(json.dumps({"output": str(output), "source_payload_bytes": source_payload_bytes, "quantized_parameter_bytes": quantized_payload_bytes, "filesystem_bytes": model_path.stat().st_size}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
