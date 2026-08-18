#!/usr/bin/env python3
"""Estimate MLX quantized checkpoint sizes from safetensor headers only.

No model weights are materialized by this tool.  It reconstructs the original
input width of each quantized linear from its current scale tensor and models
the pinned mlx-lm affine packing formula for alternate bits/group sizes.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
import subprocess
from pathlib import Path
from typing import Any


PINNED_MLX_LM_REVISION = "254d153fdeb6f150edd4fc5a54f9828638481fa8"
PINNED_CONVERT_SOURCE = "/Users/artem/.cache/uv/archive-v0/GD4ii1Q6IhKy2-zq/mlx_lm/convert.py"

DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}


def product(shape: list[int]) -> int:
    result = 1
    for value in shape:
        result *= value
    return result


def read_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        header_size_bytes = handle.read(8)
        if len(header_size_bytes) != 8:
            raise ValueError(f"short safetensor header: {path}")
        header_size = struct.unpack("<Q", header_size_bytes)[0]
        header = json.loads(handle.read(header_size))
    return header


def load_tensor_headers(model_path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    index = json.loads((model_path / "model.safetensors.index.json").read_text())
    tensor_headers: dict[str, dict[str, Any]] = {}
    for shard_name in sorted(set(index["weight_map"].values())):
        shard_path = model_path / shard_name
        header = read_header(shard_path)
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            start, end = entry["data_offsets"]
            dtype = entry["dtype"]
            tensor_headers[name] = {
                "shape": [int(v) for v in entry["shape"]],
                "dtype": dtype,
                "dtype_bytes": DTYPE_BYTES[dtype],
                "nbytes": int(end - start),
                "shard": shard_name,
            }
    return tensor_headers, index.get("metadata", {})


def quantized_modules(
    tensors: dict[str, dict[str, Any]], current_group_size: int, current_bits: int
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    modules: dict[str, dict[str, Any]] = {}
    components: set[str] = set()
    for name, weight in tensors.items():
        if not name.endswith(".weight"):
            continue
        prefix = name[: -len(".weight")]
        scale_name = f"{prefix}.scales"
        bias_name = f"{prefix}.biases"
        if scale_name not in tensors or bias_name not in tensors:
            continue
        scales = tensors[scale_name]
        if len(weight["shape"]) != 2 or len(scales["shape"]) < 2:
            raise ValueError(f"unexpected quantized tensor shapes for {prefix}")
        out_features = weight["shape"][0]
        group_count = scales["shape"][-1]
        in_features = group_count * current_group_size
        expected_packed = math.ceil(in_features * current_bits / 32)
        modules[prefix] = {
            "path": prefix,
            "out_features": out_features,
            "in_features": in_features,
            "group_count": group_count,
            "current_weight_shape": weight["shape"],
            "current_weight_nbytes": weight["nbytes"],
            "current_expected_packed_width": expected_packed,
            "current_scale_dtype": scales["dtype"],
            "current_scale_dtype_bytes": scales["dtype_bytes"],
            "current_bias_dtype": tensors[bias_name]["dtype"],
            "current_bias_dtype_bytes": tensors[bias_name]["dtype_bytes"],
            "current_scale_nbytes": scales["nbytes"],
            "current_bias_nbytes": tensors[bias_name]["nbytes"],
        }
        components.update({name, scale_name, bias_name})
    return modules, components


def layer_index(path: str) -> int:
    match = re.search(r"\.layers\.(\d+)\.", path)
    return int(match.group(1)) if match else 0


def mixed_high_bits(path: str, num_layers: int) -> bool:
    index = layer_index(path)
    use_more_bits = (
        index < num_layers // 8
        or index >= 7 * num_layers // 8
        or (index - num_layers // 8) % 3 == 2
    )
    if "v_proj" in path or "v_a_proj" in path or "v_b_proj" in path:
        return use_more_bits
    if "down_proj" in path:
        return use_more_bits
    if "lm_head" in path:
        return True
    return False


def candidate_layer_storage(
    module: dict[str, Any], bits: int, group_size: int, fallback_bytes: int
) -> tuple[int, dict[str, Any]]:
    out_features = module["out_features"]
    in_features = module["in_features"]
    if in_features % group_size != 0:
        fallback = out_features * in_features * fallback_bytes
        return fallback, {
            "quantized": False,
            "reason": "input dimension is not divisible by group size",
            "fallback_nbytes": fallback,
        }
    packed_width = math.ceil(in_features * bits / 32)
    group_count = in_features // group_size
    scale_nbytes = out_features * group_count * module["current_scale_dtype_bytes"]
    bias_nbytes = out_features * group_count * module["current_bias_dtype_bytes"]
    weight_nbytes = out_features * packed_width * 4
    return weight_nbytes + scale_nbytes + bias_nbytes, {
        "quantized": True,
        "bits": bits,
        "group_size": group_size,
        "packed_width": packed_width,
        "weight_nbytes": weight_nbytes,
        "scale_nbytes": scale_nbytes,
        "bias_nbytes": bias_nbytes,
    }


def candidate_storage(
    modules: dict[str, dict[str, Any]],
    nonquantized_bytes: int,
    bits: int,
    group_size: int,
    fallback_bytes: int,
    num_layers: int,
    mixed: bool = False,
) -> tuple[int, dict[str, Any]]:
    total = nonquantized_bytes
    low_count = 0
    high_count = 0
    skipped: list[str] = []
    layer_details: dict[str, Any] = {}
    for path, module in modules.items():
        layer_bits = 2 if mixed else bits
        if mixed and mixed_high_bits(path, num_layers):
            layer_bits = 6
            high_count += 1
        elif mixed:
            low_count += 1
        storage, details = candidate_layer_storage(
            module, layer_bits, group_size, fallback_bytes
        )
        if not details["quantized"]:
            skipped.append(path)
        total += storage
        layer_details[path] = {"bits": layer_bits, **details}
    return total, {
        "mixed": mixed,
        "low_bits": 2 if mixed else bits,
        "high_bits": 6 if mixed else None,
        "group_size": group_size,
        "high_bit_modules": high_count,
        "low_bit_modules": low_count,
        "skipped_modules": skipped,
        "layer_details": layer_details,
    }


def fit_context_curve(results_dir: Path) -> dict[str, Any]:
    points = []
    for path in sorted(results_dir.glob("results-*.json")):
        try:
            result = json.loads(path.read_text())
            if result.get("classification") != "COHERENT":
                continue
            if result.get("model_quantization") != {"mode": "affine", "bits": 3, "group_size": 64}:
                continue
            peak = max(
                float(snapshot["peak_gib"])
                for snapshot in result.get("memory_snapshots", [])
                if snapshot.get("peak_gib") is not None
            )
            points.append({"prompt_tokens": int(result["prompt_tokens"]), "peak_gib": peak})
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    if len(points) < 2:
        return {"points": points, "slope_gib_per_token": None, "intercept_gib": None}
    mean_x = sum(p["prompt_tokens"] for p in points) / len(points)
    mean_y = sum(p["peak_gib"] for p in points) / len(points)
    denominator = sum((p["prompt_tokens"] - mean_x) ** 2 for p in points)
    slope = sum(
        (p["prompt_tokens"] - mean_x) * (p["peak_gib"] - mean_y) for p in points
    ) / denominator
    intercept = mean_y - slope * mean_x
    return {
        "points": points,
        "slope_gib_per_token": slope,
        "intercept_gib": intercept,
        "reference_ceiling_gib": 16.0,
    }


def current_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except BaseException:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default="/Users/artem/.cache/huggingface/hub/models--lukaskremla--Qwen3.8-27B-3bit-MLX-TextOnly/snapshots/c98bba5926f51fec1c8d8737e577221673f524d7",
    )
    parser.add_argument("--results-dir", default="research/context-ladder")
    parser.add_argument("--output-json", default="research/quant-size/analysis.json")
    parser.add_argument("--output-md", default="research/quant-size/analysis.md")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    config = json.loads((model_path / "config.json").read_text())
    text_config = config.get("text_config", config)
    current_quant = config.get("quantization", config.get("quantization_config", {}))
    current_group = int(current_quant["group_size"])
    current_bits = int(current_quant["bits"])
    tensors, index_metadata = load_tensor_headers(model_path)
    modules, quant_components = quantized_modules(tensors, current_group, current_bits)
    nonquantized_bytes = sum(
        tensor["nbytes"] for name, tensor in tensors.items() if name not in quant_components
    )
    nonquantized_parameters = sum(
        product(tensor["shape"])
        for name, tensor in tensors.items()
        if name not in quant_components
    )
    total_parameters = int(index_metadata.get("total_parameters", 0))
    reconstructed_parameters = nonquantized_parameters + sum(
        m["out_features"] * m["in_features"] for m in modules.values()
    )
    saved_buffer_parameters = sum(
        product(tensor["shape"])
        for name, tensor in tensors.items()
        if name.endswith(".A_log") or name.endswith(".dt_bias")
    )
    reconstructed_trainable_parameters = reconstructed_parameters - saved_buffer_parameters
    fallback_bytes = 2 if text_config.get("dtype") in {"bfloat16", "float16"} else 2
    num_layers = int(text_config.get("num_hidden_layers", 0))

    actual_model_bytes = sum(tensor["nbytes"] for tensor in tensors.values())
    candidates: dict[str, Any] = {}
    candidate_specs = {
        "affine_3_g64": (3, 64, False),
        "affine_3_g128": (3, 128, False),
        "affine_2_g128": (2, 128, False),
        "mixed_2_6_g64": (2, 64, True),
        "mixed_2_6_g128": (2, 128, True),
    }
    for name, (bits, group_size, mixed) in candidate_specs.items():
        bytes_estimate, detail = candidate_storage(
            modules,
            nonquantized_bytes,
            bits,
            group_size,
            fallback_bytes,
            num_layers,
            mixed=mixed,
        )
        baseline = actual_model_bytes
        candidates[name] = {
            "stored_bytes": bytes_estimate,
            "stored_gib": bytes_estimate / 1024**3,
            "effective_bits_per_parameter": bytes_estimate * 8 / total_parameters,
            "savings_bytes_vs_actual": baseline - bytes_estimate,
            "savings_gib_vs_actual": (baseline - bytes_estimate) / 1024**3,
            "percent_saved_vs_actual": (baseline - bytes_estimate) / baseline * 100,
            "details": {key: value for key, value in detail.items() if key != "layer_details"},
        }

    baseline_formula = candidates["affine_3_g64"]["stored_bytes"]
    context_curve = fit_context_curve(Path(args.results_dir))
    slope = context_curve.get("slope_gib_per_token")
    if slope and slope > 0:
        ceiling = context_curve["reference_ceiling_gib"]
        for candidate in candidates.values():
            candidate["first_order_extra_tokens_to_16_gib"] = (
                candidate["savings_gib_vs_actual"] / slope
            )
            candidate["first_order_prompt_at_16_gib"] = (
                ceiling
                - context_curve["intercept_gib"]
                + candidate["savings_gib_vs_actual"]
            ) / slope

    result = {
        "experiment_id": "P4-QUANT-SIZES",
        "git_commit": current_commit(),
        "mlx_lm_revision": PINNED_MLX_LM_REVISION,
        "conversion_source": PINNED_CONVERT_SOURCE,
        "model_path": str(model_path),
        "config_architecture": config.get("architectures"),
        "text_config_summary": {
            key: text_config.get(key)
            for key in (
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "vocab_size",
                "layer_types",
                "tie_word_embeddings",
                "dtype",
            )
        },
        "current_quantization": current_quant,
        "safetensor_shards": sorted(set(tensor["shard"] for tensor in tensors.values())),
        "tensor_count": len(tensors),
        "quantized_module_count": len(modules),
        "quantized_component_count": len(quant_components),
        "actual_model_bytes": actual_model_bytes,
        "actual_model_gib": actual_model_bytes / 1024**3,
        "index_total_size_bytes": index_metadata.get("total_size"),
        "index_total_parameters": total_parameters,
        "reconstructed_parameter_count": reconstructed_parameters,
        "saved_buffer_parameter_count": saved_buffer_parameters,
        "reconstructed_trainable_parameter_count": reconstructed_trainable_parameters,
        "parameter_count_match": reconstructed_trainable_parameters == total_parameters,
        "nonquantized_bytes": nonquantized_bytes,
        "nonquantized_parameters": nonquantized_parameters,
        "fallback_unquantized_bytes_per_element": fallback_bytes,
        "baseline_formula_bytes": baseline_formula,
        "baseline_formula_error_bytes": baseline_formula - actual_model_bytes,
        "sample_modules": list(sorted(modules.values(), key=lambda x: x["path"]))[:12],
        "modules_not_divisible_by_128": [
            module["path"] for module in modules.values() if module["in_features"] % 128 != 0
        ],
        "candidates": candidates,
        "context_curve": context_curve,
        "notes": [
            "All estimates use safetensor headers; no weight array was loaded.",
            "Candidate quantization preserves the current Qwen3.5 model's observed quantizable-module set.",
            "Scale/bias dtype is taken per module from the current checkpoint; source conversion is expected to use bfloat16 weights per config.",
            "The index total_parameters excludes the saved A_log and dt_bias buffers; those 4,608 elements are included in stored bytes but not the effective-bpw denominator.",
            "The 16 GiB context intercept/slope extrapolation is first-order planning math, not a safety guarantee or quality result.",
        ],
    }
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Qwen3.8 quantization size analysis",
        "",
        "Header-only analysis; no new model was loaded or converted.",
        "",
        f"Current checkpoint bytes: **{actual_model_bytes:,}** ({actual_model_bytes / 1024**3:.3f} GiB); parameters: **{total_parameters:,}**.",
        f"Quantized modules: **{len(modules)}**; non-quantized storage: **{nonquantized_bytes / 1024**3:.3f} GiB**.",
        f"Reconstructed trainable parameter count matches index after excluding **{saved_buffer_parameters:,}** saved `A_log`/`dt_bias` buffer elements: **{reconstructed_trainable_parameters == total_parameters}**.",
        "",
        "| Candidate | Effective bpw | Stored GiB | Saved GiB | Saved % | First-order extra tokens to 16 GiB |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, candidate in candidates.items():
        lines.append(
            f"| {name} | {candidate['effective_bits_per_parameter']:.3f} | {candidate['stored_gib']:.3f} | {candidate['savings_gib_vs_actual']:.3f} | {candidate['percent_saved_vs_actual']:.2f}% | {candidate.get('first_order_extra_tokens_to_16_gib', '—'):.0f} |"
        )
    lines.extend(
        [
            "",
            f"Measured cold-curve fit: peak GiB ≈ {context_curve.get('intercept_gib', float('nan')):.3f} + {context_curve.get('slope_gib_per_token', float('nan')):.7f} × prompt tokens.",
            f"Modules not divisible by g128: {len(result['modules_not_divisible_by_128'])}.",
            "",
            "## Candidate ranking",
            "",
            "- `affine_3_g128`: lowest quality risk and same 3-bit runtime family; expected saving is metadata-only and therefore modest.",
            "- `mixed_2_6_g128`: built-in pinned predicate, larger saving than g128 affine, but mixed low-bit quality must be measured from original weights.",
            "- `mixed_2_6_g64`: same mixed policy without the small g128 metadata saving.",
            "- `affine_2_g128`: largest uniform saving among these candidates, but highest expected quality risk.",
            "",
            "These are storage estimates, not correctness or quality claims.",
        ]
    )
    output_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
