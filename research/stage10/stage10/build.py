"""Phase B candidate builds: RTN (rotated or not), GPTQ, and partial AWQ."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import guarded_module
from .models import write_json
from .reference import RESULTS

CANDIDATES = Path("/Users/artem/models-stage10")
BLOCK = 1024
GROUP_SIZE = 128


@dataclass(frozen=True)
class Recipe:
    layout: str
    rotate: bool
    method: str
    seed: int = 0


RECIPES = {
    "rtn3": Recipe("uniform3", False, "rtn"),
    "had3": Recipe("uniform3", True, "rtn"),
    "had3-seed1": Recipe("uniform3", True, "rtn", seed=1),
    "gptq3": Recipe("uniform3", False, "gptq"),
    "had3-gptq": Recipe("uniform3", True, "gptq"),
    "mixA-had-gptq": Recipe("mixA", True, "gptq"),
    "mixA-had-rtn": Recipe("mixA", True, "rtn"),
    "mixA-rtn": Recipe("mixA", False, "rtn"),
    "mixB-had-gptq": Recipe("mixB", True, "gptq"),
    "mixC-had-gptq": Recipe("mixC", True, "gptq"),
    "awq3": Recipe("uniform3", False, "awq"),
}
EXPERIMENTS = {
    "E02": "rtn3",
    "E03": "had3",
    "E04": "had3-seed1",
    "E06": "gptq3",
    "E07": "had3-gptq",
    "E08": "mixA-had-gptq",
    "E09": "mixA-had-rtn",
    "E10": "mixA-rtn",
    "E11": "mixB-had-gptq",
    "E12": "mixC-had-gptq",
    "E15": "awq3",
}


def layout_bits(layout: str, path: str) -> int:
    if layout == "uniform3":
        return 3
    if layout == "mixA":
        return 2 if ".mlp." in path else 3
    if layout == "mixB":
        return 2 if path.endswith(("mlp.gate_proj", "mlp.up_proj")) else 3
    if layout == "mixC":
        return 3 if path.endswith(("embed_tokens", "lm_head")) else 2
    raise ValueError(f"unknown layout {layout!r}")


def quantized_paths(model) -> list[str]:
    import mlx.nn as nn

    from .source import QUANTIZED_MODULES

    paths = sorted(
        path for path, module in model.named_modules() if isinstance(module, (nn.Linear, nn.Embedding))
    )
    if len(paths) != QUANTIZED_MODULES:
        raise ValueError(f"expected {QUANTIZED_MODULES} quantized modules, found {len(paths)}")
    return paths


def module_records(model, recipe: Recipe) -> list[dict[str, Any]]:
    from .candidate import module_record

    return [
        module_record(path, layout_bits(recipe.layout, path), BLOCK if recipe.rotate else 0, GROUP_SIZE)
        for path in quantized_paths(model)
    ]


def _store_tensor(writer, key: str, weight, records: dict[str, dict], seed: int, cast_predicate, quantizer=None):
    import mlx.core as mx

    from .candidate import quantize_module

    if cast_predicate(key) and mx.issubdtype(weight.dtype, mx.floating):
        weight = weight.astype(mx.bfloat16)
    path = key[: -len(".weight")] if key.endswith(".weight") else None
    record = records.get(path)
    if record is None:
        writer.add(key, weight)
        return
    for name, value in quantize_module(weight, record, seed, quantizer=quantizer).items():
        writer.add(f"{path}.{name}", value)


def build_rtn(exp_id: str, recipe_name: str, output: Path) -> dict:
    """Uniform/mixed RTN, rotated or not; one source layer at a time."""
    import mlx.core as mx

    from .candidate import PRODUCTION, CandidateWriter, candidate_config
    from .source import SOURCE, build_model, load_json, load_sanitized_tensors, plan_chunks, read_source

    recipe = RECIPES[recipe_name]
    plan = read_source(SOURCE)
    model = build_model(plan["config"])
    cast_predicate = getattr(model, "cast_predicate", lambda _key: True)
    records = module_records(model, recipe)
    by_path = {record["path"]: record for record in records}
    writer = CandidateWriter(output)
    for layer_idx, keys in plan_chunks(plan["weight_map"]):
        groups = [[key] for key in keys] if layer_idx is None else [keys]
        for group in groups:
            loaded = load_sanitized_tensors(model, plan["source"], plan["weight_map"], group)
            for key in sorted(loaded):
                _store_tensor(writer, key, loaded.pop(key), by_path, recipe.seed, cast_predicate)
            del loaded
            mx.clear_cache()
    config = candidate_config(load_json(PRODUCTION / "config.json"), exp_id, recipe_name, recipe.seed, records)
    writer.finish(config)
    return {}


def compare_with_production(output: Path) -> dict:
    """Count candidate tensors byte-identical to the production g128 tensors."""
    import mlx.core as mx
    import numpy as np

    from .candidate import PRODUCTION
    from .source import load_source_tensors

    prod_map = json.loads((PRODUCTION / "model.safetensors.index.json").read_text())["weight_map"]
    cand_map = json.loads((output / "model.safetensors.index.json").read_text())["weight_map"]
    identical, differing, missing = 0, [], sorted(set(prod_map) - set(cand_map))
    extra = sorted(set(cand_map) - set(prod_map))
    for key in sorted(set(prod_map) & set(cand_map)):
        left = load_source_tensors(PRODUCTION, prod_map, [key])[key]
        right = load_source_tensors(output, cand_map, [key])[key]
        same = (
            left.dtype == right.dtype
            and left.shape == right.shape
            and np.asarray(left.view(mx.uint8)).tobytes() == np.asarray(right.view(mx.uint8)).tobytes()
        )
        if same:
            identical += 1
        else:
            differing.append(key)
        del left, right
        mx.clear_cache()
    return {
        "production_tensors": len(prod_map),
        "candidate_tensors": len(cand_map),
        "byte_identical": identical,
        "differing": differing,
        "missing_in_candidate": missing,
        "extra_in_candidate": extra,
    }


def _child(args) -> None:
    import mlx.core as mx

    from .candidate import checkpoint_size

    recipe_name = EXPERIMENTS[args.exp]
    recipe = RECIPES[recipe_name]
    output = CANDIDATES / args.exp
    if args.compare_production:
        result = compare_with_production(output)
        write_json(RESULTS / args.exp / "production_compare.json", result)
        print(json.dumps({k: v for k, v in result.items() if k != "differing"}))
        return
    started = time.monotonic()
    mx.reset_peak_memory()
    if recipe.method == "rtn":
        params = build_rtn(args.exp, recipe_name, output)
    elif recipe.method == "gptq":
        from .gptq import build_gptq

        params = build_gptq(args.exp, recipe_name, output)
    elif recipe.method == "awq":
        from .awq import build_awq

        params = build_awq(args.exp, recipe_name, output)
    else:
        raise ValueError(recipe.method)
    build = {
        "wall_s": time.monotonic() - started,
        "peak_mlx_gib": mx.get_peak_memory() / 2**30,
        "params": {
            "layout": recipe.layout,
            "rotate": recipe.rotate,
            "method": recipe.method,
            "seed": recipe.seed,
            "group_size": GROUP_SIZE,
            "block": BLOCK if recipe.rotate else 0,
            "rotate_dtype": "float32",
            **params,
        },
    }
    write_json(RESULTS / args.exp / "build.json", build)
    write_json(RESULTS / args.exp / "checkpoint.json", checkpoint_size(output))
    print(json.dumps({k: v for k, v in build.items() if k != "params"}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", required=True, choices=sorted(EXPERIMENTS))
    parser.add_argument("--compare-production", action="store_true")
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if args.child:
        _child(args)
        return
    extra = ["--compare-production"] if args.compare_production else []
    name = "compare" if args.compare_production else "build"
    result = guarded_module(
        "stage10.build", ["--child", "--exp", args.exp, *extra], RESULTS / "logs" / f"{args.exp}_{name}.log"
    )
    write_json(RESULTS / args.exp / f"{name}_run.json", result)
    print(json.dumps(result, indent=2))
    if result["exit_code"] != 0 or result["safety_stop"]:
        sys.exit(result["exit_code"] or 1)


if __name__ == "__main__":
    main()
