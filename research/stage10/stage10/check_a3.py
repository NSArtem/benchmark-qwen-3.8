"""Run and record the two required A3 checks under the memory watchdog."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from .common import host_check, run_guarded, wait_for_preflight
from .source import SOURCE, build_model, load_sanitized_layer, load_source_tensors, output_key, read_source
from .layerwise import forward_loaded_model_hidden, score_hidden, score_logits


REPO_ROOT = Path(__file__).resolve().parents[3]
PRODUCTION = Path("/Users/artem/models/Qwen3.8-27B-3bit-g128-MLX")
RESULTS = REPO_ROOT / "benchmarks" / "results" / "stage10"


def _write_result(name: str, value: dict) -> None:
    path = RESULTS / "checks" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(value, indent=2))


def _check_a3_loop() -> dict:
    import mlx.core as mx
    from mlx_lm import load

    token_sets = np.load(RESULTS / "data" / "en_eval.npy", allow_pickle=False)[:2]
    if token_sets.shape != (2, 2048):
        raise ValueError(f"A3-loop expected 2x2048 tokens, received {token_sets.shape}")
    model, _tokenizer = load(str(PRODUCTION))
    model.eval()
    diffs = []
    matching = 0
    positions = 0
    for sequence in token_sets:
        tokens = sequence[None, :]
        inputs = mx.array(tokens, dtype=mx.int32)
        normal_logits = model(inputs)
        mx.eval(normal_logits)
        normal = score_logits(normal_logits, tokens, chunk_size=256)
        del normal_logits
        mx.clear_cache()

        hidden = forward_loaded_model_hidden(model, inputs)
        manual = score_hidden(
            hidden,
            model.language_model.lm_head,
            tokens,
            chunk_size=256,
        )
        diffs.append(float(np.max(np.abs(normal["token_logp"] - manual["token_logp"]))))
        matching += int(np.count_nonzero(normal["top1"] == manual["top1"]))
        positions += int(normal["top1"].size)
        del inputs, hidden, normal, manual
        mx.clear_cache()

    max_abs = max(diffs)
    agreement = matching / positions
    result = {
        "check": "A3-loop",
        "sequences": 2,
        "scored_positions": positions,
        "max_abs_logp_diff": max_abs,
        "argmax_agreement": agreement,
        "thresholds": {"max_abs_logp_diff": 1e-3, "argmax_agreement": 0.999},
        "status": "PASS" if max_abs <= 1e-3 and agreement >= 0.999 else "FAIL",
    }
    _write_result("a3_loop", result)
    if result["status"] != "PASS":
        raise RuntimeError(f"A3-loop check failed: {result}")
    return result


def _same_bytes(left, right) -> bool:
    import mlx.core as mx

    mx.eval(left, right)
    if left.dtype != right.dtype or left.shape != right.shape:
        return False
    # str(mx.bfloat16) is "mlx.core.bfloat16" and NumPy has no bfloat16, so
    # compare raw bytes for every dtype.
    lhs = np.asarray(left.view(mx.uint8))
    rhs = np.asarray(right.view(mx.uint8))
    return lhs.tobytes() == rhs.tobytes()


def _max_abs(left, right) -> float:
    import mlx.core as mx

    lhs_value = left.astype(mx.float32)
    rhs_value = right.astype(mx.float32)
    mx.eval(lhs_value, rhs_value)
    lhs = np.asarray(lhs_value)
    rhs = np.asarray(rhs_value)
    return float(np.max(np.abs(lhs - rhs)))


def _check_a3_sanitize() -> dict:
    import mlx.core as mx

    source_plan = read_source(SOURCE)
    sanitizer_model = build_model(source_plan["config"])
    production_index = json.loads((PRODUCTION / "model.safetensors.index.json").read_text())
    production_map = production_index["weight_map"]
    identical = 0
    total = 0
    diffs = []

    for layer_idx in (0, 3):
        sanitized = load_sanitized_layer(
            sanitizer_model,
            SOURCE,
            source_plan["weight_map"],
            layer_idx,
        )
        for key, source_weight in sorted(sanitized.items()):
            weight = source_weight
            cast_predicate = getattr(sanitizer_model, "cast_predicate", lambda _key: True)
            if cast_predicate(key) and mx.issubdtype(weight.dtype, mx.floating):
                weight = weight.astype(mx.bfloat16)
            scale_key = key.replace(".weight", ".scales")
            bias_key = key.replace(".weight", ".biases")
            quantized = key.endswith(".weight") and scale_key in production_map
            if quantized:
                qweight, scales, biases = mx.quantize(
                    weight, group_size=128, bits=3, mode="affine"
                )
                candidate_tensors = {
                    key: qweight,
                    scale_key: scales,
                    bias_key: biases,
                }
            else:
                candidate_tensors = {key: weight}
            missing = sorted(set(candidate_tensors) - set(production_map))
            if missing:
                raise KeyError(f"production checkpoint misses A3-sanitize tensors: {missing}")
            production_tensors = load_source_tensors(
                PRODUCTION, production_map, candidate_tensors.keys()
            )
            bad = []
            for tensor_key, candidate_tensor in candidate_tensors.items():
                total += 1
                if _same_bytes(candidate_tensor, production_tensors[tensor_key]):
                    identical += 1
                else:
                    bad.append(tensor_key)
            if bad:
                if quantized:
                    production_base = load_source_tensors(
                        PRODUCTION, production_map, (key, scale_key, bias_key)
                    )
                    candidate_dequant = mx.dequantize(
                        qweight, scales, biases, group_size=128, bits=3
                    )
                    production_dequant = mx.dequantize(
                        production_base[key],
                        production_base[scale_key],
                        production_base[bias_key],
                        group_size=128,
                        bits=3,
                    )
                    max_diff = _max_abs(candidate_dequant, production_dequant)
                else:
                    max_diff = _max_abs(weight, production_tensors[key])
                diffs.extend({"name": tensor_key, "max_abs_dequantized_diff": max_diff} for tensor_key in bad)
            del source_weight, weight, candidate_tensors, production_tensors
            if quantized:
                del qweight, scales, biases
            mx.clear_cache()
        del sanitized
        mx.clear_cache()

    result = {
        "check": "A3-sanitize",
        "layers": [0, 3],
        "byte_identical_tensors": identical,
        "tensor_count": total,
        "mismatches": diffs,
        "status": "PASS" if identical == total else "FAIL",
    }
    _write_result("a3_sanitize", result)
    if result["status"] != "PASS":
        raise RuntimeError(f"A3-sanitize check failed: {result}")
    return result


def _child(mode: str) -> None:
    if mode == "loop":
        _check_a3_loop()
    else:
        _check_a3_sanitize()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("loop", "sanitize"), required=True)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if args.child:
        _child(args.mode)
        return

    try:
        before = wait_for_preflight()
    except Exception as exc:
        print(f"preflight failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
    log_path = RESULTS / "logs" / f"a3_{args.mode}.log"
    command = [sys.executable, "-m", "stage10.check_a3", "--mode", args.mode, "--child"]
    exit_code, watchdog_fired = run_guarded(command, log_path)
    after = host_check()
    print(
        json.dumps(
            {
                "mode": args.mode,
                "exit_code": exit_code,
                "watchdog_fired": watchdog_fired,
                "free_pct_before": before.free_pct,
                "swap_mib_before": before.swap_used_mib,
                "free_pct_after": after.free_pct,
                "swap_mib_after": after.swap_used_mib,
                "log_path": str(log_path),
            },
            indent=2,
        )
    )
    if exit_code != 0 or watchdog_fired:
        raise SystemExit(exit_code or 1)


if __name__ == "__main__":
    main()
