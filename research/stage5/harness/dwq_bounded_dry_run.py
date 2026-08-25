#!/usr/bin/env python3
"""Run a small bounded DWQ-style distillation dry run.

This is deliberately a synthetic layer-sized experiment.  It exercises the
important part of DWQ—optimizing quantized scales/biases against fixed teacher
outputs—without ever loading a Qwen3.8 teacher and student together.  The
script also records the static residency calculation that gates a real 27B
run on the 18 GB machine.

It is not claimed to be an end-to-end replacement for mlx-lm's DWQ.  The
purpose is to distinguish "the optimization is numerically feasible" from
"the pinned whole-model implementation is safe to launch".
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import time
from pathlib import Path
from typing import Any

import mlx.core as mx


def gib(value: int | float) -> float:
    return float(value) / (1024**3)


def rss_bytes() -> int:
    # macOS reports ru_maxrss in bytes; retain a defensive branch for Linux.
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if value > 1024 * 1024 else value * 1024


def memory_snapshot(label: str) -> dict[str, Any]:
    active = int(mx.get_active_memory())
    cache = int(mx.get_cache_memory())
    peak = int(mx.get_peak_memory())
    return {
        "label": label,
        "active_bytes": active,
        "cache_bytes": cache,
        "active_plus_cache_bytes": active + cache,
        "peak_bytes": peak,
        "active_gib": round(gib(active), 6),
        "active_plus_cache_gib": round(gib(active + cache), 6),
        "peak_gib": round(gib(peak), 6),
        "rss_bytes": rss_bytes(),
    }


def scalar(value: Any) -> float:
    mx.eval(value)
    return float(value.item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/stage5/quant/dwq_bounded_dry_run.json"),
    )
    parser.add_argument("--steps", type=int, default=40)
    args = parser.parse_args()

    started = time.time()
    # MLX 0.32.1 only accepts the production group sizes here; use the
    # smallest supported size for a compact layer-sized exercise.
    group_size = 32
    bits = 3
    batch_count = 6
    input_width = 32
    output_width = 24

    # Fixed seeds make the dry run reproducible and keep it independent from
    # the held-out suite and all Stage 4/5 calibration data.
    mx.random.seed(20260825)
    teacher_weight = mx.random.normal((output_width, input_width), dtype=mx.float32)
    inputs = mx.random.normal(
        (batch_count, input_width), dtype=mx.float32
    )
    # A slightly softened teacher target resembles a local distillation target
    # without requiring a vocabulary-sized language-model head.
    teacher_logits = inputs @ teacher_weight.T
    mx.eval(teacher_weight, inputs, teacher_logits)

    qweight, scales, biases = mx.quantize(
        teacher_weight,
        group_size=group_size,
        bits=bits,
        mode="affine",
    )
    params = {
        "scales": scales.astype(mx.float32),
        "biases": biases.astype(mx.float32),
    }
    mx.eval(qweight, params)
    before = memory_snapshot("after_quantized_layer_setup")

    def loss_fn(current: dict[str, Any]) -> Any:
        student_logits = mx.quantized_matmul(
            inputs,
            qweight,
            scales=current["scales"],
            biases=current["biases"],
            transpose=True,
            group_size=group_size,
            bits=bits,
            mode="affine",
        )
        return mx.mean((student_logits - teacher_logits) ** 2)

    value_and_grad = mx.value_and_grad(loss_fn)
    initial_loss = scalar(loss_fn(params))
    losses = [initial_loss]
    learning_rate = 0.001
    for step in range(args.steps):
        loss, grads = value_and_grad(params)
        mx.eval(loss, grads)
        params = {
            name: params[name] - learning_rate * grads[name]
            for name in params
        }
        mx.eval(params)
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == args.steps:
            losses.append(scalar(loss_fn(params)))

    final_loss = scalar(loss_fn(params))
    after = memory_snapshot("after_bounded_scale_bias_optimization")

    source_payload_bytes = 55_562_855_904
    g128_payload_bytes = 10_930_715_648
    g64_payload_bytes = 11_771_132_928
    device_budget_bytes = 18 * 1024**3
    result = {
        "experiment_id": "P6-DWQ-BOUNDED-SYNTHETIC-DRY-RUN",
        "classification": "DRY_RUN_PASS_UNSAFE_FULL_MODEL",
        "algorithm_note": "synthetic layer-wise scale/bias distillation; not upstream whole-model DWQ",
        "canonical_environment": "envs/mlx-0.32.1",
        "mlx_version": "0.32.1",
        "seed": 20260825,
        "steps": args.steps,
        "input_shape": [batch_count, input_width],
        "layer_shape": [output_width, input_width],
        "group_size": group_size,
        "bits": bits,
        "initial_mse": initial_loss,
        "final_mse": final_loss,
        "relative_mse_reduction": (initial_loss - final_loss) / max(initial_loss, 1e-12),
        "loss_trace": losses,
        "memory_before": before,
        "memory_after": after,
        "static_residency": {
            "source_high_precision_payload_bytes": source_payload_bytes,
            "g128_student_payload_bytes": g128_payload_bytes,
            "g64_student_or_teacher_payload_bytes": g64_payload_bytes,
            "source_plus_g128_minimum_bytes": source_payload_bytes + g128_payload_bytes,
            "source_plus_g128_minimum_gib": gib(source_payload_bytes + g128_payload_bytes),
            "g64_plus_g128_minimum_bytes": g64_payload_bytes + g128_payload_bytes,
            "g64_plus_g128_minimum_gib": gib(g64_payload_bytes + g128_payload_bytes),
            "device_budget_bytes": device_budget_bytes,
            "device_budget_gib": gib(device_budget_bytes),
            "naive_source_teacher_plus_student_exceeds_budget": (
                source_payload_bytes + g128_payload_bytes > device_budget_bytes
            ),
            "activation_optimizer_and_cache_bytes_included": False,
        },
        "bounded_design_gate": {
            "synthetic_optimization_feasible": final_loss < initial_loss,
            "full_27b_dual_residency_safe": False,
            "next_real_path": "only a streamed layer-wise implementation with measured per-layer residency could be considered",
        },
        "wired_limit_calls": 0,
        "elapsed_seconds": time.time() - started,
        "output": str(args.output),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
