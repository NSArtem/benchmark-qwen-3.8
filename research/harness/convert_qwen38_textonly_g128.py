#!/usr/bin/env python3
"""Bounded-memory Qwen3.8 text-only g128 conversion.

This is intentionally narrower than mlx-lm's general converter.  It handles
the exact Qwen/Qwen3.8-27B source layout and the existing text-only MLX model:
source tensors are loaded one transformer layer at a time, sanitized with the
pinned Qwen3.5 implementation, quantized, and flushed to small safetensor
shards. It supports the primary affine 3-bit g128 candidate and one explicitly
selected mixed_2_6 g128 candidate. Stage 4 additionally supports the pinned
mixed_3_4 control and a manifest-selected sensitivity_3_4 candidate. It must
run in the canonical MLX environment.

No model arrays are loaded by --self-test.  The converter never changes the
MLX wired-memory policy.
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


DEFAULT_GROUP_SIZE = 128
DEFAULT_BITS = 3
MIXED_LOW_BITS = 2
MIXED_HIGH_BITS = 6
MIXED_34_LOW_BITS = 3
MIXED_34_HIGH_BITS = 4
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


def mixed_2_6_params(path: str, num_layers: int = 64) -> dict[str, Any]:
    """Match pinned mlx-lm mixed_quant_predicate_builder('mixed_2_6')."""

    if "lm_head" in path:
        use_more_bits = True
    else:
        match = re.search(r"\.layers\.(\d+)\.", path)
        index = int(match.group(1)) if match else 0
        use_more_bits = (
            index < num_layers // 8
            or index >= 7 * num_layers // 8
            or (index - num_layers // 8) % 3 == 2
        )
    high = use_more_bits and (
        "v_proj" in path
        or "v_a_proj" in path
        or "v_b_proj" in path
        or "down_proj" in path
        or "lm_head" in path
    )
    return {
        "group_size": DEFAULT_GROUP_SIZE,
        "bits": MIXED_HIGH_BITS if high else MIXED_LOW_BITS,
        "mode": "affine",
    }


def mixed_3_4_params(path: str, num_layers: int = 64) -> dict[str, Any]:
    """Match pinned mlx-lm mixed_quant_predicate_builder('mixed_3_4')."""

    if "lm_head" in path:
        use_more_bits = True
    else:
        match = re.search(r"\.layers\.(\d+)\.", path)
        index = int(match.group(1)) if match else 0
        use_more_bits = (
            index < num_layers // 8
            or index >= 7 * num_layers // 8
            or (index - num_layers // 8) % 3 == 2
        )
    high = use_more_bits and (
        "v_proj" in path
        or "v_a_proj" in path
        or "v_b_proj" in path
        or "down_proj" in path
        or "lm_head" in path
    )
    return {
        "group_size": DEFAULT_GROUP_SIZE,
        "bits": MIXED_34_HIGH_BITS if high else MIXED_34_LOW_BITS,
        "mode": "affine",
    }


def quant_params(
    path: str,
    recipe: str,
    num_layers: int,
    promoted_paths: set[str] | None = None,
) -> dict[str, Any]:
    if recipe == "mixed_2_6":
        return mixed_2_6_params(path, num_layers)
    if recipe == "mixed_3_4_control":
        return mixed_3_4_params(path, num_layers)
    if recipe == "sensitivity_3_4":
        if promoted_paths is None:
            raise ValueError("sensitivity_3_4 requires --promotion-manifest")
        return {
            "group_size": DEFAULT_GROUP_SIZE,
            "bits": (
                MIXED_34_HIGH_BITS
                if path.removesuffix(".weight") in promoted_paths
                else MIXED_34_LOW_BITS
            ),
            "mode": "affine",
        }
    return {
        "group_size": DEFAULT_GROUP_SIZE,
        "bits": DEFAULT_BITS,
        "mode": "affine",
    }


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


def append_metric(path: Path, record: dict[str, Any]) -> None:
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


def load_promotion_manifest(path: Path, candidate_name: str | None) -> set[str]:
    value = load_json(path)
    if "selected_paths" in value:
        paths = value["selected_paths"]
    else:
        candidates = value.get("candidates", [])
        if candidate_name is None:
            if len(candidates) != 1:
                raise ValueError(
                    "promotion manifest contains multiple candidates; pass --candidate-name"
                )
            candidate = candidates[0]
        else:
            matching = [item for item in candidates if item.get("name") == candidate_name]
            if len(matching) != 1:
                raise ValueError(f"candidate not found in promotion manifest: {candidate_name}")
            candidate = matching[0]
        paths = candidate.get("selected_paths", [])
    if not isinstance(paths, list) or not all(isinstance(item, str) for item in paths):
        raise ValueError("promotion manifest selected_paths must be a list of strings")
    return set(paths)


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

    mixed_counts = {2: 0, 6: 0}
    for key in qkeys:
        mixed_counts[mixed_2_6_params(key)["bits"]] += 1
    if mixed_counts != {2: 457, 6: 41}:
        raise AssertionError(f"unexpected mixed_2_6 module counts: {mixed_counts}")
    mixed_34_counts = {3: 0, 4: 0}
    for key in qkeys:
        mixed_34_counts[mixed_3_4_params(key)["bits"]] += 1
    if mixed_34_counts != {3: 457, 4: 41}:
        raise AssertionError(f"unexpected mixed_3_4 module counts: {mixed_34_counts}")

    return {
        "model_type": config.get("model_type"),
        "architecture": config.get("architectures"),
        "quantization_key_count": len(qkeys),
        "text_only_quantization_keys": True,
        "chunk_plan_test": [layer for layer, _ in chunks],
        "global_norm_shift_test": shifted_value,
        "mixed_2_6_module_counts": mixed_counts,
        "mixed_3_4_module_counts": mixed_34_counts,
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
    metrics_path = Path(args.metrics).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    if not source.is_dir() or not baseline.is_dir():
        raise FileNotFoundError("source and baseline must be directories")

    started = time.monotonic()
    output.mkdir(parents=True)
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
    text_config = config.get("text_config", config)
    num_layers = int(text_config.get("num_hidden_layers", 64))
    promoted_paths: set[str] | None = None
    if args.quant_recipe == "sensitivity_3_4":
        if not args.promotion_manifest:
            raise ValueError("sensitivity_3_4 requires --promotion-manifest")
        promoted_paths = load_promotion_manifest(
            Path(args.promotion_manifest).resolve(), args.candidate_name
        )
        qmodule_paths = {key.removesuffix(".weight") for key in qkeys}
        if not promoted_paths <= qmodule_paths:
            missing = sorted(promoted_paths - qmodule_paths)
            raise ValueError(f"promotion manifest contains unknown module paths: {missing[:5]}")
    quantization_counts = {"2": 0, "3": 0, "4": 0, "6": 0}
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
                params = quant_params(key, args.quant_recipe, num_layers, promoted_paths)
                if weight.ndim < 2 or weight.shape[-1] % params["group_size"]:
                    raise ValueError(
                        f"invalid g128 quantization shape for {key}: {weight.shape}"
                    )
                quantized, scales, biases = mx.quantize(
                    weight,
                    group_size=params["group_size"],
                    bits=params["bits"],
                    mode=params["mode"],
                )
                mx.eval(quantized, scales, biases)
                quantization_counts[str(params["bits"])] += 1
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
    quantization = {
        "group_size": DEFAULT_GROUP_SIZE,
        "bits": DEFAULT_BITS,
        "mode": "affine",
    }
    if args.quant_recipe == "mixed_2_6":
        quantization["bits"] = MIXED_LOW_BITS
        quantization.update(
            {
                # mlx-lm's class_predicate receives module paths, not the
                # serialized `.weight` tensor paths used by qkeys.
                key.removesuffix(".weight"): quant_params(
                    key, args.quant_recipe, num_layers, promoted_paths
                )
                for key in sorted(qkeys)
            }
        )
    elif args.quant_recipe in {"mixed_3_4_control", "sensitivity_3_4"}:
        quantization.update(
            {
                key.removesuffix(".weight"): quant_params(
                    key, args.quant_recipe, num_layers, promoted_paths
                )
                for key in sorted(qkeys)
            }
        )
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
        "quant_recipe": args.quant_recipe,
        "quantization_counts": quantization_counts,
        **memory_snapshot("complete", time.monotonic() - started),
        "os": os_snapshot(),
    }
    append_metric(metrics_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", help="local Qwen/Qwen3.8-27B source snapshot")
    parser.add_argument("--baseline", required=True, help="local text-only MLX baseline")
    parser.add_argument("--output", help="new output directory")
    parser.add_argument(
        "--metrics",
        default="research/conversion/g128-progress.jsonl",
        help="progress JSONL path",
    )
    parser.add_argument(
        "--shard-bytes",
        type=int,
        default=DEFAULT_SHARD_BYTES,
        help="output shard target; default 512 MiB",
    )
    parser.add_argument(
        "--quant-recipe",
        choices=("affine_3_g128", "mixed_2_6", "mixed_3_4_control", "sensitivity_3_4"),
        default="affine_3_g128",
        help="g128 recipe; Stage 4 candidates use mixed_3_4_control or sensitivity_3_4",
    )
    parser.add_argument(
        "--promotion-manifest",
        help="JSON summary containing selected_paths for sensitivity_3_4",
    )
    parser.add_argument(
        "--candidate-name",
        help="candidate name inside --promotion-manifest when it contains multiple candidates",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        print(json.dumps(self_test(Path(args.baseline)), indent=2, sort_keys=True))
        return
    if not args.source or not args.output:
        parser.error("--source and --output are required unless --self-test is used")
    print(json.dumps(convert(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
