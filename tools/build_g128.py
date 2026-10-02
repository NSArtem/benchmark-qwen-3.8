#!/usr/bin/env python3
"""Build the production Qwen3.8-27B affine 3-bit / group-128 MLX checkpoint.

Bounded-memory, text-only conversion from the BF16 ``Qwen/Qwen3.8-27B``
source snapshot.  This is intentionally narrower than mlx-lm's general
converter: source tensors are loaded one transformer layer at a time,
sanitized with the pinned mlx-lm Qwen3.5-family implementation, quantized
(affine, 3 bits, group size 128), and flushed to ~512 MiB safetensor shards.
Peak MLX memory is about 3.3 GiB instead of the full 50+ GiB source.

Only ``lm_head.weight`` and ``model.language_model.*`` tensors are kept; the
vision tower and MTP (``mtp.*``) tensors are dropped.  Expected output: 851
tensors -> 498 quantized modules, 19 shards, 10,930,715,648 bytes
(10.180 GiB).

``--baseline`` must be an existing text-only MLX checkpoint of the same
architecture (for example ``lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly`` or a
previous g128 build).  Only its ``config.json``, index metadata, and
tokenizer/chat-template files are used; none of its weights are read.

``--self-test`` checks the conversion plan without loading model arrays.
The converter never changes the MLX wired-memory policy.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import re
import resource
import shutil
import struct
import subprocess
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_lm.utils import _get_classes


GROUP_SIZE = 128
BITS = 3
QUANTIZATION = {"group_size": GROUP_SIZE, "bits": BITS, "mode": "affine"}
DEFAULT_SHARD_BYTES = 512 * (1 << 20)
SENTINEL_FRAGMENT = "__qwen38_chunk_sentinel__"
LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.")


def read_safetensors_header(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("rb") as stream:
        raw_size = stream.read(8)
        if len(raw_size) != 8:
            raise ValueError(f"short safetensors header: {path}")
        header_size = struct.unpack("<Q", raw_size)[0]
        header = json.loads(stream.read(header_size))
    header.pop("__metadata__", None)
    return {
        name: {
            "shape": meta["shape"],
            "dtype": meta["dtype"],
            "data_bytes": meta["data_offsets"][1] - meta["data_offsets"][0],
        }
        for name, meta in header.items()
    }


def source_weight_map(source: Path) -> dict[str, str]:
    index_path = source / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"missing source index: {index_path}")
    with index_path.open() as stream:
        all_keys = json.load(stream)["weight_map"]
    kept = {
        key: filename
        for key, filename in all_keys.items()
        if key == "lm_head.weight" or key.startswith("model.language_model.")
    }
    if not kept:
        raise ValueError("source index has no Qwen3.8 text-only keys")
    return kept


def plan_chunks(weight_map: dict[str, str]) -> list[tuple[int | None, list[str]]]:
    by_layer: dict[int, list[str]] = {}
    non_layer: list[str] = []
    for key in sorted(weight_map):
        match = LAYER_RE.match(key)
        if match:
            by_layer.setdefault(int(match.group(1)), []).append(key)
        else:
            non_layer.append(key)

    chunks: list[tuple[int | None, list[str]]] = []
    if non_layer:
        chunks.append((None, non_layer))
    chunks.extend((layer, by_layer[layer]) for layer in sorted(by_layer))
    return chunks


def output_key(source_key: str) -> str:
    if source_key.startswith("model.language_model."):
        return source_key.replace(
            "model.language_model", "language_model.model", 1
        )
    return f"language_model.{source_key}"


def memory_snapshot(stage: str, elapsed_seconds: float) -> dict[str, Any]:
    active = int(mx.get_active_memory())
    cache = int(mx.get_cache_memory())
    peak = int(mx.get_peak_memory())
    return {
        "stage": stage,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "active_bytes": active,
        "cache_bytes": cache,
        "active_plus_cache_bytes": active + cache,
        "peak_bytes": peak,
        "active_gib": round(active / (1 << 30), 6),
        "cache_gib": round(cache / (1 << 30), 6),
        "active_plus_cache_gib": round((active + cache) / (1 << 30), 6),
        "peak_gib": round(peak / (1 << 30), 6),
        "max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }


def os_snapshot() -> dict[str, str]:
    result: dict[str, str] = {}
    for command in (("vm_stat",), ("memory_pressure", "-Q")):
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            result[command[0]] = f"unavailable: {exc}"
        else:
            result[command[0]] = (completed.stdout or completed.stderr).strip()
    return result


def append_metric(path: Path | None, record: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def build_model(config: dict[str, Any]):
    model_class, model_args_class = _get_classes(config)
    model = model_class(model_args_class.from_dict(config))
    return model


def quantization_keys(model) -> set[str]:
    keys = set()
    for path, module in model.named_modules():
        if hasattr(module, "to_quantized"):
            keys.add(f"{path}.weight")
    return keys


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def self_test(baseline: Path) -> dict[str, Any]:
    config = load_json(baseline / "config.json")
    model = build_model(config)
    qkeys = quantization_keys(model)
    if len(qkeys) != 498:
        raise AssertionError(f"expected 498 quantization keys, found {len(qkeys)}")

    expected_prefixes = {
        "language_model.model.layers.",
        "language_model.model.embed_tokens.weight",
        "language_model.lm_head.weight",
    }
    if not all(any(key.startswith(prefix) for prefix in expected_prefixes) for key in qkeys):
        raise AssertionError("unexpected quantization key outside Qwen3.8 text model")

    chunks = plan_chunks(
        {
            "lm_head.weight": "a.safetensors",
            "model.language_model.embed_tokens.weight": "a.safetensors",
            "model.language_model.layers.0.mlp.up_proj.weight": "a.safetensors",
            "model.language_model.layers.1.mlp.up_proj.weight": "b.safetensors",
            "model.visual.blocks.0.weight": "b.safetensors",
        }
    )
    if [layer for layer, _ in chunks] != [None, 0, 1]:
        raise AssertionError(f"unexpected chunk plan: {chunks}")

    mapped = {output_key(key) for key in ("lm_head.weight", "model.language_model.layers.0.mlp.up_proj.weight")}
    if mapped != {
        "language_model.lm_head.weight",
        "language_model.model.layers.0.mlp.up_proj.weight",
    }:
        raise AssertionError(f"unexpected sanitized key mapping: {mapped}")

    shifted = sanitize_chunk(
        model,
        {
            "model.language_model.layers.3.input_layernorm.weight": mx.ones(
                (4,), dtype=mx.bfloat16
            )
        },
    )
    mx.eval(*shifted.values())
    shifted_value = float(shifted["language_model.model.layers.3.input_layernorm.weight"][0])
    if abs(shifted_value - 2.0) > 0.01:
        raise AssertionError(f"global norm shift was not preserved: {shifted_value}")

    return {
        "model_type": config.get("model_type"),
        "architecture": config.get("architectures"),
        "quantization_key_count": len(qkeys),
        "text_only_quantization_keys": True,
        "chunk_plan_test": [layer for layer, _ in chunks],
        "global_norm_shift_test": shifted_value,
        "quantization": QUANTIZATION,
        "wired_limit_calls": 0,
    }


def sanitize_chunk(model, loaded: dict[str, Any]) -> dict[str, Any]:
    """Apply pinned Qwen3.5 sanitization while preserving its global norm rule.

    TextModel.sanitize shifts every norm when *any* unsanitized depthwise
    conv1d exists anywhere in the source. A one-layer chunk would otherwise
    miss that global condition for full-attention layers, so an inert
    conv1d-shaped sentinel is included for the duration of the pinned
    ``model.sanitize`` call and removed afterward.
    """

    sentinel_key = f"model.language_model.{SENTINEL_FRAGMENT}.conv1d.weight"
    loaded[sentinel_key] = mx.zeros((1, 1, 2), dtype=mx.float32)
    sanitized = model.sanitize(loaded)
    sanitized = {
        key: value
        for key, value in sanitized.items()
        if SENTINEL_FRAGMENT not in key
    }
    return sanitized


def source_headers(source: Path, weight_map: dict[str, str]) -> dict[str, dict[str, Any]]:
    headers: dict[str, dict[str, Any]] = {}
    for filename in sorted(set(weight_map.values())):
        for key, meta in read_safetensors_header(source / filename).items():
            if key in weight_map:
                headers[key] = meta
    return headers


def verify_source_plan(
    baseline: Path,
    source: Path,
    weight_map: dict[str, str],
    headers: dict[str, dict[str, Any]],
    model,
) -> dict[str, Any]:
    expected_output_keys = {output_key(key) for key in weight_map}
    qkeys = quantization_keys(model)
    mapped_qkeys = {output_key(key) for key in weight_map if key.endswith(".weight")}
    missing = sorted(qkeys - mapped_qkeys)
    if missing:
        raise ValueError(f"source is missing quantizable text weights: {missing[:5]}")
    if len(expected_output_keys) != 851:
        raise ValueError(f"expected 851 retained source tensors, found {len(expected_output_keys)}")
    if len(headers) != len(weight_map):
        raise ValueError("source index/header tensor count mismatch")

    baseline_index = load_json(baseline / "model.safetensors.index.json")
    return {
        "retained_source_tensors": len(weight_map),
        "retained_source_shards": len(set(weight_map.values())),
        "retained_source_bytes": sum(meta["data_bytes"] for meta in headers.values()),
        "quantization_keys": len(qkeys),
        "baseline_total_parameters": baseline_index["metadata"]["total_parameters"],
        "source_layers": len({match.group(1) for key in weight_map if (match := LAYER_RE.match(key))}),
    }


def flush_shard(
    output: Path,
    shard_index: int,
    shard_weights: dict[str, Any],
    output_map: dict[str, str],
) -> tuple[str, int]:
    if not shard_weights:
        return "", 0
    temp_name = f"model-{shard_index:05d}-of-XXXXX.safetensors"
    mx.save_safetensors(
        str(output / temp_name),
        dict(sorted(shard_weights.items())),
        metadata={"format": "mlx"},
    )
    for key in shard_weights:
        output_map[key] = temp_name
    size = sum(int(value.nbytes) for value in shard_weights.values())
    return temp_name, size


def copy_tokenizer_files(baseline: Path, output: Path) -> None:
    names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
        "special_tokens_map.json",
    )
    for name in names:
        source = baseline / name
        if source.exists():
            shutil.copy2(source, output / name)


def convert(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.source).resolve()
    baseline = Path(args.baseline).resolve()
    output = Path(args.output).resolve()
    metrics_path = Path(args.metrics).resolve() if args.metrics else None
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    if not source.is_dir() or not baseline.is_dir():
        raise FileNotFoundError("source and baseline must be directories")

    started = time.monotonic()
    output.mkdir(parents=True)
    if metrics_path is not None:
        metrics_path.unlink(missing_ok=True)
    append_metric(metrics_path, {"event": "start", "os": os_snapshot()})

    config = load_json(baseline / "config.json")
    model = build_model(config)
    qkeys = quantization_keys(model)
    if len(qkeys) != 498:
        raise ValueError(f"expected 498 quantization keys, found {len(qkeys)}")

    weight_map = source_weight_map(source)
    headers = source_headers(source, weight_map)
    plan = verify_source_plan(baseline, source, weight_map, headers, model)
    append_metric(metrics_path, {"event": "source_plan", **plan})

    chunks = plan_chunks(weight_map)
    target_dtype = mx.bfloat16
    cast_predicate = getattr(model, "cast_predicate", lambda _: True)
    quantized_modules = 0
    output_map: dict[str, str] = {}
    shard_weights: dict[str, Any] = {}
    shard_bytes = 0
    shard_index = 1
    total_size = 0
    saved_shards: list[str] = []

    for chunk_index, (layer_index, chunk_keys) in enumerate(chunks, start=1):
        loaded: dict[str, Any] = {}
        for filename in sorted({weight_map[key] for key in chunk_keys}):
            file_data = mx.load(str(source / filename))
            for key in chunk_keys:
                if weight_map[key] == filename:
                    loaded[key] = file_data[key]
            del file_data

        loaded = sanitize_chunk(model, loaded)
        expected_chunk_keys = {output_key(key) for key in chunk_keys}
        if expected_chunk_keys != set(loaded):
            missing = sorted(expected_chunk_keys - set(loaded))
            extra = sorted(set(loaded) - expected_chunk_keys)
            raise ValueError(f"sanitization key mismatch; missing={missing[:3]} extra={extra[:3]}")

        for key in sorted(list(loaded)):
            weight = loaded.pop(key)
            if cast_predicate(key) and mx.issubdtype(weight.dtype, mx.floating):
                weight = weight.astype(target_dtype)

            if key in qkeys:
                if weight.ndim < 2 or weight.shape[-1] % GROUP_SIZE:
                    raise ValueError(
                        f"invalid g128 quantization shape for {key}: {weight.shape}"
                    )
                quantized, scales, biases = mx.quantize(weight, **QUANTIZATION)
                mx.eval(quantized, scales, biases)
                quantized_modules += 1
                outputs = {
                    key: quantized,
                    key.replace(".weight", ".scales"): scales,
                    key.replace(".weight", ".biases"): biases,
                }
            else:
                mx.eval(weight)
                outputs = {key: weight}

            for output_key_name, value in outputs.items():
                shard_weights[output_key_name] = value
                shard_bytes += int(value.nbytes)
                total_size += int(value.nbytes)

            if shard_bytes >= args.shard_bytes:
                shard_name, _ = flush_shard(output, shard_index, shard_weights, output_map)
                saved_shards.append(shard_name)
                del shard_weights
                shard_weights = {}
                shard_bytes = 0
                shard_index += 1
                gc.collect()
                mx.clear_cache()

        del loaded
        gc.collect()
        mx.clear_cache()
        append_metric(
            metrics_path,
            {
                "event": "chunk",
                "chunk_index": chunk_index,
                "chunk_count": len(chunks),
                "layer_index": layer_index,
                "source_tensor_count": len(chunk_keys),
                **memory_snapshot(f"after_chunk_{chunk_index}", time.monotonic() - started),
            },
        )

    if shard_weights:
        shard_name, _ = flush_shard(output, shard_index, shard_weights, output_map)
        saved_shards.append(shard_name)
        del shard_weights
        gc.collect()
        mx.clear_cache()

    total_shards = len(saved_shards)
    if total_shards == 0:
        raise RuntimeError("conversion produced no output shards")
    rename_map: dict[str, str] = {}
    if total_shards == 1:
        old_name = saved_shards[0]
        new_name = "model.safetensors"
        (output / old_name).rename(output / new_name)
        rename_map[old_name] = new_name
    else:
        for index, old_name in enumerate(saved_shards, start=1):
            new_name = f"model-{index:05d}-of-{total_shards:05d}.safetensors"
            (output / old_name).rename(output / new_name)
            rename_map[old_name] = new_name
    output_map = {key: rename_map[name] for key, name in output_map.items()}

    output_config = copy.deepcopy(config)
    quantization = dict(QUANTIZATION)
    output_config["quantization"] = quantization
    output_config["quantization_config"] = copy.deepcopy(quantization)
    with (output / "config.json").open("w", encoding="utf-8") as stream:
        json.dump(dict(sorted(output_config.items())), stream, indent=4)

    baseline_index = load_json(baseline / "model.safetensors.index.json")
    index = {
        "metadata": {
            "total_size": total_size,
            "total_parameters": baseline_index["metadata"]["total_parameters"],
        },
        "weight_map": dict(sorted(output_map.items())),
    }
    with (output / "model.safetensors.index.json").open("w", encoding="utf-8") as stream:
        json.dump(index, stream, indent=4)
    copy_tokenizer_files(baseline, output)

    result = {
        "event": "complete",
        "output": str(output),
        "output_shards": total_shards,
        "output_bytes": total_size,
        "output_gib": round(total_size / (1 << 30), 6),
        "output_tensor_count": len(output_map),
        "output_parameter_count": index["metadata"]["total_parameters"],
        "quantization": QUANTIZATION,
        "quantized_modules": quantized_modules,
        **memory_snapshot("complete", time.monotonic() - started),
        "os": os_snapshot(),
    }
    append_metric(metrics_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Qwen3.8-27B affine 3-bit/g128 text-only MLX checkpoint",
    )
    parser.add_argument("--source", help="local BF16 Qwen/Qwen3.8-27B snapshot directory")
    parser.add_argument(
        "--baseline",
        required=True,
        help="existing text-only Qwen3.8 MLX checkpoint (config/tokenizer template)",
    )
    parser.add_argument("--output", help="new output directory (must not exist)")
    parser.add_argument("--metrics", help="optional progress JSONL path")
    parser.add_argument(
        "--shard-bytes",
        type=int,
        default=DEFAULT_SHARD_BYTES,
        help="output shard target; default 512 MiB",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="validate the conversion plan against --baseline without loading weights",
    )
    args = parser.parse_args()

    if args.self_test:
        print(json.dumps(self_test(Path(args.baseline)), indent=2, sort_keys=True))
        return
    if not args.source or not args.output:
        parser.error("--source and --output are required unless --self-test is used")
    print(json.dumps(convert(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
