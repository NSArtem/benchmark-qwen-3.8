#!/usr/bin/env python3
"""Measure bounded, calibration-activated quantization sensitivity for Qwen3.8.

This is deliberately not a copy of mlx-lm's Dynamic Quantization estimator.
That estimator keeps a full teacher, a copied model, and a copied quantizable
layer set resident.  That is not a safe experiment for a 27B model on this
18 GB machine.

Instead, this harness:

* loads the already-quantized g128 model once;
* loads source tensors only for one converter chunk at a time;
* quantizes those source tensors to affine 4-bit/group-128;
* runs a small, fixed local calibration split through the g128 model; and
* measures the activation-weighted output difference between the resident
  g128 module and the temporary 4-bit module output.

The result is a sensitivity proxy, not an activation-aware teacher loss.  It
is suitable for ranking 3->4 promotions and for estimating their byte cost.
No wired-memory setting is changed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
import resource
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn


ROOT = Path(__file__).resolve().parents[3]
CONVERTER_DIR = ROOT / "research/harness"
if str(CONVERTER_DIR) not in sys.path:
    sys.path.insert(0, str(CONVERTER_DIR))

from convert_qwen38_textonly_g128 import (  # noqa: E402
    build_model,
    load_json,
    memory_snapshot,
    output_key,
    plan_chunks,
    quantization_keys,
    sanitize_chunk,
    source_headers,
    source_weight_map,
    verify_source_plan,
)


LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
DEFAULT_GROUP_SIZE = 128
BASELINE_BITS = 3
PROMOTED_BITS = 4


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalize_ids(value: Any) -> list[int]:
    if hasattr(value, "encodings") and value.encodings:
        return [int(item) for item in value.encodings[0].ids]
    if isinstance(value, dict):
        value = value["input_ids"]
    elif hasattr(value, "input_ids"):
        value = value.input_ids
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], (list, tuple)):
        value = value[0]
    return [int(item) for item in value]


def calibration_prompt(tokenizer: Any, prompt: str) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    from_text = normalize_ids(tokenizer.encode(rendered, add_special_tokens=False))
    direct = normalize_ids(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )
    if from_text != direct:
        raise RuntimeError("chat-template string/tokenized forms disagree")
    return from_text


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def layer_index(path: str) -> int | None:
    match = LAYER_RE.search(path)
    return int(match.group(1)) if match else None


def module_family(path: str) -> str:
    if "lm_head" in path:
        return "lm_head"
    if "embed_tokens" in path:
        return "embed_tokens"
    if ".linear_attn." in path:
        return "linear_attn"
    if ".self_attn." in path:
        return "self_attn"
    if ".mlp." in path:
        return "mlp"
    return "other"


def module_kind(path: str) -> str:
    return path.rsplit(".", 1)[-1]


def memory_guard(label: str, budget_gib: float) -> dict[str, Any]:
    snapshot = memory_snapshot(label, 0.0)
    if snapshot["active_plus_cache_gib"] > budget_gib:
        raise MemoryError(
            f"memory guard tripped at {label}: "
            f"{snapshot['active_plus_cache_gib']:.3f} GiB > {budget_gib:.3f} GiB"
        )
    return snapshot


class ProbeState:
    """State shared by temporary QuantizedLinear/Embedding call wrappers."""

    def __init__(self) -> None:
        self.active: dict[int, dict[str, Any]] = {}
        self.pending: list[tuple[str, Any, Any, int]] = []
        self.records: dict[str, dict[str, Any]] = {}

    def begin_case(self) -> None:
        self.pending = []

    def record_outputs(self, key: str, y3: Any, y4: Any) -> Any:
        y3_f = y3.astype(mx.float32)
        y4_f = y4.astype(mx.float32)
        diff = y4_f - y3_f
        sse_delta = mx.sum(diff * diff)
        sse_base = mx.sum(y3_f * y3_f)
        self.pending.append((key, sse_delta, sse_base, int(y3.size)))
        return y3

    def finish_case(self) -> None:
        if not self.pending:
            return
        values: list[Any] = []
        for _, sse_delta, sse_base, _ in self.pending:
            values.extend((sse_delta, sse_base))
        mx.eval(*values)
        for key, sse_delta, sse_base, output_elements in self.pending:
            record = self.records.setdefault(
                key,
                {
                    "calls": 0,
                    "output_elements": 0,
                    "sse_delta": 0.0,
                    "sse_base": 0.0,
                },
            )
            record["calls"] += 1
            record["output_elements"] += output_elements
            record["sse_delta"] += float(sse_delta.item())
            record["sse_base"] += float(sse_base.item())
        self.pending = []


def q4_linear_output(module: Any, x: Any, q4: dict[str, Any]) -> Any:
    result = mx.quantized_matmul(
        x,
        q4["weight"],
        scales=q4["scales"],
        biases=q4["biases"],
        transpose=True,
        group_size=DEFAULT_GROUP_SIZE,
        bits=PROMOTED_BITS,
        mode="affine",
    )
    bias = module.get("bias")
    if bias is not None:
        result = result + bias
    return result


def q4_embedding_output(q4: dict[str, Any], x: Any) -> Any:
    return mx.dequantize(
        q4["weight"][x],
        scales=q4["scales"][x],
        biases=q4["biases"][x],
        group_size=DEFAULT_GROUP_SIZE,
        bits=PROMOTED_BITS,
        mode="affine",
    )


def install_wrappers(state: ProbeState) -> tuple[Any, Any]:
    original_linear = nn.QuantizedLinear.__call__
    original_embedding = nn.QuantizedEmbedding.__call__

    def wrapped_linear(module: Any, x: Any) -> Any:
        entry = state.active.get(id(module))
        y3 = original_linear(module, x)
        if entry is None:
            return y3
        y4 = q4_linear_output(module, x, entry)
        return state.record_outputs(entry["path"], y3, y4)

    def wrapped_embedding(module: Any, x: Any) -> Any:
        entry = state.active.get(id(module))
        y3 = original_embedding(module, x)
        if entry is None:
            return y3
        y4 = q4_embedding_output(entry, x)
        return state.record_outputs(entry["path"], y3, y4)

    nn.QuantizedLinear.__call__ = wrapped_linear
    nn.QuantizedEmbedding.__call__ = wrapped_embedding
    return original_linear, original_embedding


def remove_wrappers(original_linear: Any, original_embedding: Any) -> None:
    nn.QuantizedLinear.__call__ = original_linear
    nn.QuantizedEmbedding.__call__ = original_embedding


def load_chunk(
    source: Path,
    weight_map: dict[str, str],
    model: Any,
    chunk_keys: list[str],
) -> dict[str, Any]:
    loaded: dict[str, Any] = {}
    for filename in sorted({weight_map[key] for key in chunk_keys}):
        file_data = mx.load(str(source / filename))
        for key in chunk_keys:
            if weight_map[key] == filename:
                loaded[key] = file_data[key]
        del file_data
    loaded = sanitize_chunk(model, loaded)
    expected = {output_key(key) for key in chunk_keys}
    if expected != set(loaded):
        missing = sorted(expected - set(loaded))
        extra = sorted(set(loaded) - expected)
        raise ValueError(f"sanitization key mismatch; missing={missing[:3]} extra={extra[:3]}")
    return loaded


def make_q4_map(
    loaded: dict[str, Any],
    qkeys: set[str],
    module_by_path: dict[str, Any],
) -> tuple[dict[int, dict[str, Any]], dict[str, dict[str, Any]]]:
    active: dict[int, dict[str, Any]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for key in sorted(list(loaded)):
        path = key.removesuffix(".weight")
        if key not in qkeys:
            loaded.pop(key)
            continue
        module = module_by_path.get(path)
        if module is None:
            raise ValueError(f"source quantization key has no model module: {path}")
        weight = loaded.pop(key)
        if not mx.issubdtype(weight.dtype, mx.floating):
            raise ValueError(f"non-floating quantization source tensor: {key}")
        weight = weight.astype(mx.bfloat16)
        if weight.ndim < 2 or weight.shape[-1] % DEFAULT_GROUP_SIZE:
            raise ValueError(f"invalid g128 shape for {key}: {weight.shape}")
        qweight, scales, biases = mx.quantize(
            weight,
            group_size=DEFAULT_GROUP_SIZE,
            bits=PROMOTED_BITS,
            mode="affine",
        )
        mx.eval(qweight, scales, biases)
        q4 = {"weight": qweight, "scales": scales, "biases": biases, "path": path}
        active[id(module)] = q4
        metadata[path] = {
            "source_key": key,
            "shape": list(weight.shape),
            "parameter_count": int(weight.size),
            "q3_payload_bytes": int(module.weight.nbytes + module.scales.nbytes + (module.biases.nbytes if module.biases is not None else 0)),
            "q4_payload_bytes": int(qweight.nbytes + scales.nbytes + biases.nbytes),
        }
        del weight, qweight, scales, biases
    return active, metadata


def finalize_records(
    state: ProbeState,
    metadata: dict[str, dict[str, Any]],
    calibration_token_count: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(metadata):
        item = dict(metadata[path])
        item.update(state.records.get(path, {}))
        item.setdefault("calls", 0)
        item.setdefault("output_elements", 0)
        item.setdefault("sse_delta", 0.0)
        item.setdefault("sse_base", 0.0)
        item["path"] = path
        item["layer_index"] = layer_index(path)
        item["family"] = module_family(path)
        item["module"] = module_kind(path)
        item["extra_payload_bytes"] = item["q4_payload_bytes"] - item["q3_payload_bytes"]
        item["calibration_token_count"] = calibration_token_count
        if item["output_elements"]:
            item["rms_output_delta"] = math.sqrt(item["sse_delta"] / item["output_elements"])
        else:
            item["rms_output_delta"] = None
        item["relative_rms_output_delta"] = math.sqrt(
            item["sse_delta"] / max(item["sse_base"], 1e-12)
        ) if item["sse_base"] > 0 else None
        item["delta_energy_per_parameter"] = (
            item["sse_delta"] / item["parameter_count"]
            if item["parameter_count"]
            else None
        )
        records.append(item)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--baseline", required=True, help="g128 MLX model")
    parser.add_argument("--calibration", required=True, help="frozen local JSONL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--case-ids", nargs="*", default=None)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument("--skip-non-layer", action="store_true")
    parser.add_argument("--memory-budget-gib", type=float, default=16.0)
    args = parser.parse_args()

    started = time.monotonic()
    source = Path(args.source).resolve()
    baseline = Path(args.baseline).resolve()
    calibration_path = Path(args.calibration).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    specs = read_jsonl(calibration_path)
    if args.case_ids:
        wanted = set(args.case_ids)
        specs = [spec for spec in specs if spec["case_id"] in wanted]
    if not specs:
        raise ValueError("no calibration cases selected")

    result: dict[str, Any] = {
        "experiment_id": "P3-BOUNDED-ACTIVATION-SENSITIVITY",
        "phase": 3,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": "unknown",
        "method": "activation_weighted_output_distortion_proxy",
        "method_limits": [
            "The reference path is the loaded g128 model, not a full-precision teacher.",
            "The proxy compares each g128 module output with a temporary affine 4-bit/g128 output.",
            "It ranks promotions and byte cost; it is not a learned end-to-end KL or perplexity sensitivity.",
        ],
        "source": str(source),
        "baseline": str(baseline),
        "calibration_path": str(calibration_path),
        "calibration_sha256": sha256_file(calibration_path),
        "calibration_case_ids": [spec["case_id"] for spec in specs],
        "calibration_cases": [],
        "max_tokens_per_case": args.max_tokens,
        "quantization": {
            "baseline_bits": BASELINE_BITS,
            "promoted_bits": PROMOTED_BITS,
            "group_size": DEFAULT_GROUP_SIZE,
            "mode": "affine",
        },
        "bounded_memory": {
            "memory_budget_gib": args.memory_budget_gib,
            "source_chunking": "existing converter plan_chunks plus sanitize_chunk",
            "full_source_model_resident": False,
            "full_teacher_model_resident": False,
        },
        "model": None,
        "source_plan": None,
        "memory": [],
        "records": [],
        "exception": None,
    }

    original_linear = original_embedding = None
    state = ProbeState()
    try:
        from mlx_lm.utils import load

        model, tokenizer = load(str(baseline))
        result["model"] = {
            "path": str(baseline),
            "config": load_json(baseline / "config.json"),
            "quantization_keys": None,
        }
        result["memory"].append(memory_guard("after_model_load", args.memory_budget_gib))

        schema_model = build_model(load_json(baseline / "config.json"))
        qkeys = quantization_keys(schema_model)
        result["model"]["quantization_keys"] = len(qkeys)
        module_by_path = {
            path: module
            for path, module in model.named_modules()
            if f"{path}.weight" in qkeys
        }
        if len(qkeys) != 498 or len(module_by_path) != 498:
            raise ValueError(
                f"expected 498 quantized modules, got "
                f"keys={len(qkeys)} modules={len(module_by_path)}"
            )

        weight_map = source_weight_map(source)
        headers = source_headers(source, weight_map)
        result["source_plan"] = verify_source_plan(
            baseline, source, weight_map, headers, schema_model
        )
        result["source_plan"]["source_header_sha256"] = sha256_file(source / "model.safetensors.index.json")

        calibration_inputs: list[dict[str, Any]] = []
        for spec in specs:
            token_ids = calibration_prompt(tokenizer, spec["prompt"])
            token_ids = token_ids[: args.max_tokens]
            if len(token_ids) < 2:
                raise ValueError(f"calibration case too short: {spec['case_id']}")
            calibration_inputs.append(
                {
                    "case_id": spec["case_id"],
                    "category": spec["category"],
                    "prompt_sha256": hashlib.sha256(spec["prompt"].encode()).hexdigest(),
                    "token_count": len(token_ids),
                    "token_ids_sha256": hashlib.sha256(json.dumps(token_ids, separators=(",", ":")).encode()).hexdigest(),
                    "token_ids": token_ids,
                }
            )
        result["calibration_cases"] = [
            {key: value for key, value in item.items() if key != "token_ids"}
            for item in calibration_inputs
        ]

        original_linear, original_embedding = install_wrappers(state)
        chunks = plan_chunks(weight_map)
        processed_chunks = 0
        processed_layers = 0
        records_by_path: dict[str, dict[str, Any]] = {}
        for chunk_index, (chunk_layer, chunk_keys) in enumerate(chunks, start=1):
            if chunk_layer is None and args.skip_non_layer:
                continue
            if chunk_layer is not None and args.max_layers is not None and chunk_layer >= args.max_layers:
                break
            chunk_started = time.monotonic()
            loaded = load_chunk(source, weight_map, schema_model, chunk_keys)
            active, metadata = make_q4_map(loaded, qkeys, module_by_path)
            del loaded
            state.active = {module_id: value for module_id, value in active.items()}
            del active
            gc.collect()
            mx.clear_cache()
            result["memory"].append(
                memory_guard(f"chunk_{chunk_index}_q4_ready", args.memory_budget_gib)
            )

            state.records = {}
            for calibration in calibration_inputs:
                state.begin_case()
                inputs = mx.array([calibration["token_ids"]])
                logits = model(inputs)
                pending_values = [value for _, value, _, _ in state.pending]
                pending_values.extend(value for _, _, value, _ in state.pending)
                mx.eval(logits, *pending_values)
                state.finish_case()
                del logits, inputs
                mx.clear_cache()
                memory_guard(
                    f"chunk_{chunk_index}_{calibration['case_id']}",
                    args.memory_budget_gib,
                )

            for path, metadata_item in metadata.items():
                prior = records_by_path.get(path)
                if prior is None:
                    records_by_path[path] = metadata_item
                else:
                    if prior != metadata_item:
                        raise ValueError(f"duplicate sensitivity metadata for {path}")
            chunk_records = finalize_records(
                state,
                metadata,
                sum(item["token_count"] for item in calibration_inputs),
            )
            for item in chunk_records:
                records_by_path[item["path"]] = item
            state.active = {}
            state.records = {}
            gc.collect()
            mx.clear_cache()
            result["memory"].append(
                memory_guard(f"chunk_{chunk_index}_released", args.memory_budget_gib)
            )
            processed_chunks += 1
            if chunk_layer is not None:
                processed_layers = max(processed_layers, chunk_layer + 1)
            print(
                json.dumps(
                    {
                        "chunk_index": chunk_index,
                        "layer_index": chunk_layer,
                        "module_count": len(metadata),
                        "elapsed_seconds": round(time.monotonic() - chunk_started, 3),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        result["records"] = sorted(records_by_path.values(), key=lambda item: item["path"])
        result["processed_chunks"] = processed_chunks
        result["processed_layers"] = processed_layers
        result["complete"] = len(result["records"]) == len(qkeys)
        result["memory"].append(memory_guard("complete", args.memory_budget_gib))
    except BaseException as exc:
        result["exception"] = f"{type(exc).__name__}: {exc}"
        result["complete"] = False
        result["memory"].append(memory_snapshot("exception", time.monotonic() - started))
        result["total_elapsed_seconds"] = round(time.monotonic() - started, 3)
        result["max_peak_gib"] = max(
            (item["peak_gib"] for item in result["memory"]), default=None
        )
        result["max_active_plus_cache_gib"] = max(
            (item["active_plus_cache_gib"] for item in result["memory"]),
            default=None,
        )
        output_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise
    finally:
        if original_linear is not None and original_embedding is not None:
            remove_wrappers(original_linear, original_embedding)
        state.active = {}
        gc.collect()
        mx.clear_cache()

    result["total_elapsed_seconds"] = round(time.monotonic() - started, 3)
    result["max_peak_gib"] = max((item["peak_gib"] for item in result["memory"]), default=None)
    result["max_active_plus_cache_gib"] = max(
        (item["active_plus_cache_gib"] for item in result["memory"]),
        default=None,
    )
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("complete", "processed_chunks", "processed_layers", "total_elapsed_seconds", "max_peak_gib", "max_active_plus_cache_gib")}, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
