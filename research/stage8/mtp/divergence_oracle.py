#!/usr/bin/env python3
"""Locate the first layer where Qwen3.8 chunk execution diverges.

The oracle runs the same predetermined token stream from one identical target
cache state in two ways: one token at a time, and one short sequence at a time.
It records per-layer hidden and cache state differences plus final-logit
differences.  It loads no MTP artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
STAGE7 = ROOT / "stage7/harness"
for path in (STAGE7,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_mtp_correctness as oracle  # noqa: E402
import run_mtp_performance as perf  # noqa: E402


DEFAULT_MODEL = "/Users/artem/models/Qwen3.8-27B-3bit-g128-MLX"
DEFAULT_TOKENS = [123, 456, 789]


def scalar(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def max_abs_diff(mx: Any, left: Any, right: Any) -> float | None:
    if left is None or right is None or tuple(left.shape) != tuple(right.shape):
        return None
    difference = mx.max(mx.abs(left.astype(mx.float32) - right.astype(mx.float32)))
    mx.eval(difference)
    return scalar(difference)


def equal_arrays(mx: Any, left: Any, right: Any) -> bool | None:
    difference = max_abs_diff(mx, left, right)
    return None if difference is None else difference == 0.0


def cache_state_arrays(cache: list[Any]) -> list[tuple[str, int, Any]]:
    from mlx_lm.models.cache import ArraysCache, KVCache

    arrays: list[tuple[str, int, Any]] = []
    for layer_index, entry in enumerate(cache):
        if isinstance(entry, ArraysCache):
            for state_index, value in enumerate(entry.cache):
                if value is not None:
                    arrays.append(("ArraysCache", layer_index * 10 + state_index, value))
        elif isinstance(entry, KVCache):
            if entry.keys is not None:
                arrays.append(("KVCache.keys", layer_index, entry.keys[..., : entry.offset, :]))
                arrays.append(("KVCache.values", layer_index, entry.values[..., : entry.offset, :]))
    return arrays


def cache_compare(mx: Any, left: list[Any], right: list[Any]) -> dict[str, Any]:
    from mlx_lm.models.cache import ArraysCache, KVCache

    left_by_layer = {index: entry for index, entry in enumerate(left)}
    right_by_layer = {index: entry for index, entry in enumerate(right)}
    layer_rows: list[dict[str, Any]] = []
    first: dict[str, Any] | None = None
    for index in range(max(len(left), len(right))):
        lentry = left_by_layer.get(index)
        rentry = right_by_layer.get(index)
        row: dict[str, Any] = {
            "layer_index": index,
            "left_type": type(lentry).__name__ if lentry is not None else None,
            "right_type": type(rentry).__name__ if rentry is not None else None,
            "cache_equal": True,
            "max_abs_diff": 0.0,
            "state_diffs": [],
        }
        if isinstance(lentry, ArraysCache) and isinstance(rentry, ArraysCache):
            for state_index, (lvalue, rvalue) in enumerate(zip(lentry.cache, rentry.cache)):
                difference = max_abs_diff(mx, lvalue, rvalue)
                equal = equal_arrays(mx, lvalue, rvalue)
                row["state_diffs"].append(
                    {"state_index": state_index, "max_abs_diff": difference, "equal": equal}
                )
                if equal is not True:
                    row["cache_equal"] = False
                if difference is not None:
                    row["max_abs_diff"] = max(row["max_abs_diff"], difference)
            if len(lentry.cache) != len(rentry.cache):
                row["cache_equal"] = False
        elif isinstance(lentry, KVCache) and isinstance(rentry, KVCache):
            row["left_offset"] = int(lentry.offset)
            row["right_offset"] = int(rentry.offset)
            if lentry.offset != rentry.offset:
                row["cache_equal"] = False
            for name, lvalue, rvalue in (
                ("keys", lentry.keys[..., : lentry.offset, :] if lentry.keys is not None else None, rentry.keys[..., : rentry.offset, :] if rentry.keys is not None else None),
                ("values", lentry.values[..., : lentry.offset, :] if lentry.values is not None else None, rentry.values[..., : rentry.offset, :] if rentry.values is not None else None),
            ):
                difference = max_abs_diff(mx, lvalue, rvalue)
                equal = equal_arrays(mx, lvalue, rvalue)
                row["state_diffs"].append(
                    {"state": name, "max_abs_diff": difference, "equal": equal}
                )
                if equal is not True:
                    row["cache_equal"] = False
                if difference is not None:
                    row["max_abs_diff"] = max(row["max_abs_diff"], difference)
        else:
            row["cache_equal"] = False
        if not row["cache_equal"] and first is None:
            first = {
                "layer_index": index,
                "cache_type": row["left_type"],
                "state_diffs": row["state_diffs"],
                "max_abs_diff": row["max_abs_diff"],
            }
        layer_rows.append(row)
    return {
        "cache_equal": first is None,
        "first_divergent": first,
        "layers": layer_rows,
    }


def traced_forward(
    mx: Any,
    qwen_module: Any,
    inner: Any,
    input_ids: Any,
    cache: list[Any],
) -> tuple[Any, list[Any]]:
    """Mirror Qwen3_5TextModel.__call__ while retaining every layer output."""

    hidden = inner.embed_tokens(input_ids)
    fa_mask = qwen_module.create_attention_mask(hidden, cache[inner.fa_idx])
    ssm_mask = qwen_module.create_ssm_mask(hidden, cache[inner.ssm_idx])
    layer_outputs: list[Any] = []
    for layer, layer_cache in zip(inner.layers, cache):
        mask = ssm_mask if layer.is_linear else fa_mask
        hidden = layer(hidden, mask=mask, cache=layer_cache)
        oracle.eval_cache(mx, cache, hidden)
        layer_outputs.append(hidden)
    hidden = inner.norm(hidden)
    oracle.eval_cache(mx, cache, hidden)
    return hidden, layer_outputs


def as_bsh(value: Any, sequence_length: int) -> Any:
    """Normalize MLX's occasional squeezed ``[B,H]``/``[S,H]`` result."""

    dimensions = len(tuple(value.shape))
    if dimensions == 3:
        return value
    if dimensions == 2:
        if sequence_length == 1:
            return value[:, None, :]
        if int(value.shape[0]) == sequence_length:
            return value[None, :, :]
    raise ValueError(
        f"cannot normalize hidden shape {tuple(value.shape)} for sequence length {sequence_length}"
    )


def hidden_compare(mx: Any, sequential: list[Any], batched: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    batched = as_bsh(batched, len(sequential))
    for index, seq in enumerate(sequential):
        seq_value = as_bsh(seq, 1)
        difference = max_abs_diff(mx, seq_value, batched[:, index : index + 1, :])
        rows.append({"position": index, "max_abs_diff": difference, "equal": difference == 0.0})
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    result: dict[str, Any] = {
        "experiment_id": args.experiment_id or "P5-QWEN38-FIRST-DIVERGENCE",
        "phase": 5,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model_path": str(Path(args.model).resolve()),
        "context": args.context,
        "prefix_tokens_requested": args.prefix_tokens,
        "candidate_token_ids": [int(item) for item in args.tokens],
        "candidate_token_sha256": hashlib.sha256(
            json.dumps([int(item) for item in args.tokens], separators=(",", ":")).encode()
        ).hexdigest(),
        "exception": None,
        "safety_stop_reason": None,
    }
    try:
        mx, load, _generate_step, _make_prompt_cache, _make_sampler, _ = (
            perf.scale_utils.oracle_utils.load_runtime()
        )
        model, tokenizer = load(args.model)
        prompt_ids, prompt_meta = perf.prompt_data(args.context, tokenizer)
        prompt_ids = prompt_ids[: args.prefix_tokens]
        result.update(prompt_meta)
        result["prefix_tokens_actual"] = len(prompt_ids)
        inner = model.language_model.model
        target_head = model.language_model.lm_head
        import mlx_lm.models.qwen3_5 as qwen_module

        prefix_cache = model.make_cache()
        prompt = mx.array(prompt_ids, dtype=mx.int32)[None, :]
        prefix_hidden = oracle.target_prefill_baseline_schedule(
            mx, inner, prompt, prefix_cache
        )
        mx.eval(prefix_hidden)
        initial_cache = oracle.copy_cache(mx, prefix_cache)
        del prefix_cache, prefix_hidden
        mx.clear_cache()

        sequential_cache = oracle.copy_cache(mx, initial_cache)
        sequential_outputs: list[Any] = []
        sequential_layers: list[list[Any]] = []
        sequential_cache_snapshots: list[list[Any]] = []
        for token_id in args.tokens:
            output, layers = traced_forward(
                mx,
                qwen_module,
                inner,
                mx.array([[int(token_id)]], dtype=mx.int32),
                sequential_cache,
            )
            sequential_outputs.append(output)
            sequential_layers.append(layers)
            sequential_cache_snapshots.append(oracle.copy_cache(mx, sequential_cache))

        per_length: list[dict[str, Any]] = []
        for length in range(1, len(args.tokens) + 1):
            batch_cache = oracle.copy_cache(mx, initial_cache)
            batch_input = mx.array([args.tokens[:length]], dtype=mx.int32)
            batch_output, batch_layers = traced_forward(
                mx, qwen_module, inner, batch_input, batch_cache
            )
            seq_output = mx.concatenate(
                [as_bsh(output, 1) for output in sequential_outputs[:length]],
                axis=1,
            )
            batch_output = as_bsh(batch_output, length)
            mx.eval(seq_output)
            layer_rows: list[dict[str, Any]] = []
            first_hidden: dict[str, Any] | None = None
            for layer_index, batch_layer in enumerate(batch_layers):
                rows = hidden_compare(
                    mx,
                    [sequential_layers[position][layer_index] for position in range(length)],
                    batch_layer,
                )
                layer_row = {
                    "layer_index": layer_index,
                    "layer_type": "linear_attention"
                    if inner.layers[layer_index].is_linear
                    else "full_attention",
                    "hidden": rows,
                    "hidden_equal": all(row["equal"] for row in rows),
                    "hidden_max_abs_diff": max(
                        [row["max_abs_diff"] or 0.0 for row in rows], default=0.0
                    ),
                }
                if not layer_row["hidden_equal"] and first_hidden is None:
                    first_hidden = {
                        "layer_index": layer_index,
                        "cache_type": type(batch_cache[layer_index]).__name__,
                        "layer_type": layer_row["layer_type"],
                        "position": next(
                            row["position"] for row in rows if not row["equal"]
                        ),
                        "max_abs_diff": layer_row["hidden_max_abs_diff"],
                    }
                layer_rows.append(layer_row)
            cache_result = cache_compare(
                mx, sequential_cache_snapshots[length - 1], batch_cache
            )
            seq_logits = target_head(seq_output)
            batch_logits = target_head(batch_output)
            mx.eval([seq_logits, batch_logits])
            logit_diff = max_abs_diff(mx, seq_logits, batch_logits)
            seq_top = oracle.ids(mx.argmax(seq_logits, axis=-1))
            batch_top = oracle.ids(mx.argmax(batch_logits, axis=-1))
            per_length.append(
                {
                    "length": length,
                    "first_hidden_divergent": first_hidden,
                    "hidden_layer_count": len(layer_rows),
                    "layers": layer_rows,
                    "cache": cache_result,
                    "logits_max_abs_diff": logit_diff,
                    "logits_equal": logit_diff == 0.0,
                    "sequential_top1": seq_top,
                    "batched_top1": batch_top,
                    "top1_equal": seq_top == batch_top,
                    "final_cache_offsets": [
                        int(entry.offset) if hasattr(entry, "offset") else None
                        for entry in batch_cache
                    ],
                }
            )
            del batch_cache, batch_output, batch_layers, seq_output, seq_logits, batch_logits
            mx.clear_cache()

        first_overall: dict[str, Any] | None = None
        for row in per_length:
            candidate = row["first_hidden_divergent"] or row["cache"]["first_divergent"]
            if candidate is not None:
                first_overall = {"length": row["length"], **candidate}
                break
        result["per_length"] = per_length
        result["first_divergence"] = first_overall
        result["sequential_final_cache_state"] = [
            {
                "layer_index": index,
                "cache_type": type(entry).__name__,
                "offset": int(entry.offset) if hasattr(entry, "offset") else None,
                "array_bytes": sum(int(value.nbytes) for _kind, _state, value in cache_state_arrays([entry])),
            }
            for index, entry in enumerate(sequential_cache)
        ]
    except BaseException as exc:
        result["exception"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = __import__("traceback").format_exc()
    result["elapsed_seconds"] = time.perf_counter() - started
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--context", choices=("short", "4k", "8k"), default="short")
    parser.add_argument("--prefix-tokens", type=int, default=64)
    parser.add_argument("--tokens", type=int, nargs="+", default=DEFAULT_TOKENS)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(args.tokens) < 2:
        parser.error("--tokens must contain at least two candidate tokens")
    result = run(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "experiment_id": result["experiment_id"],
        "exception": result["exception"],
        "first_divergence": result.get("first_divergence"),
        "per_length": [
            {
                "length": item["length"],
                "hidden_first": item["first_hidden_divergent"],
                "cache_equal": item["cache"]["cache_equal"],
                "logits_equal": item["logits_equal"],
                "top1_equal": item["top1_equal"],
            }
            for item in result.get("per_length", [])
        ],
    }, sort_keys=True))
    return 0 if result["exception"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
