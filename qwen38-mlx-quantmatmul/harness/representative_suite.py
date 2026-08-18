#!/usr/bin/env python
"""Reduced fresh-process suite across representative real QuantizedLinear shapes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

from common import DEFAULT_MODEL_PATH, MODEL_ID, MLX_LM_COMMIT, update_results
from common import module_metadata, resolve_module
from quantized_linear_runner import dense_reference, layer_result, make_input, module_forward


DEFAULT_LABELS = (
    "gated_delta_in_proj_qkv",
    "gated_delta_in_proj_z",
    "gated_delta_in_proj_a",
    "gated_delta_in_proj_b",
    "gated_delta_out_proj",
    "full_attention_q_proj",
    "full_attention_k_proj",
    "full_attention_v_proj",
    "mlp_gate_proj",
    "mlp_up_proj",
    "mlp_down_proj",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--manifest", default="qwen38-mlx-quantmatmul/module-manifest.json")
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    parser.add_argument("--lengths", default="1,512")
    parser.add_argument("--process-index", type=int, required=True)
    parser.add_argument("--skip-reference-labels", default="")
    parser.add_argument("--seed", type=int, default=38027)
    parser.add_argument("--results", default="qwen38-mlx-quantmatmul/results.json")
    args = parser.parse_args()

    mx.set_default_device(mx.gpu)
    model, _ = load(args.model_path)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    labels = tuple(label for label in args.labels.split(",") if label)
    lengths = tuple(int(value) for value in args.lengths.split(",") if value)
    skip_reference = {label for label in args.skip_reference_labels.split(",") if label}
    results_path = Path(args.results)

    for label in labels:
        path = manifest["selected"][label]
        module = resolve_module(model, path)
        metadata = module_metadata(module)
        for sequence_length in lengths:
            # The label offset avoids accidental reuse of an input graph between
            # different matrix shapes while remaining identical across processes.
            seed = args.seed + sequence_length + sum(ord(char) for char in label)
            value = make_input(sequence_length, metadata["input_dims"], seed)
            output_hash, output_np = module_forward(module, value)
            reference_hash = None
            reference_np = None
            if label not in skip_reference:
                reference_hash, reference_np = dense_reference(module, value)
            result = layer_result(
                test_id=f"representative-{label}-p{args.process_index:02d}-t{sequence_length:04d}",
                module_path=path,
                layer_label=label,
                module=module,
                value=value,
                sequence_length=sequence_length,
                process_index=args.process_index,
                repeat_index=0,
                output_hash=output_hash,
                output_np=output_np,
                reference_hash=reference_hash,
                reference_np=reference_np,
                input_seed=seed,
            )
            result["reference_skipped"] = label in skip_reference
            result["test_family"] = "representative_real_layer_shapes"
            update_results(results_path, result, collection="representative_layer_tests")
            print(json.dumps(result, sort_keys=True))
            del value, output_np, reference_np
            mx.clear_cache()


if __name__ == "__main__":
    main()
