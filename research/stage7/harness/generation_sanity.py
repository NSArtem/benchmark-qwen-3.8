#!/usr/bin/env python3
"""Greedy generation sanity on the frozen DWQ validation prompts.

This deliberately uses only the Stage 7 validation calibration set, never the
final held-out suites.  It is a small coherence/tool-format check for the
internal module-count gate.
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research/stage7/harness"))
from run_bounded_dwq import load_patch  # noqa: E402


def normalize_ids(value: Any) -> list[int]:
    if hasattr(value, "encodings") and value.encodings:
        return [int(item) for item in value.encodings[0].ids]
    if isinstance(value, dict):
        value = value["input_ids"]
    elif hasattr(value, "input_ids"):
        value = value.input_ids
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
        value = value[0]
    return [int(item) for item in value]


def prompt_ids(tokenizer: Any, prompt: str) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    from_text = normalize_ids(tokenizer.encode(rendered, add_special_tokens=False))
    direct = normalize_ids(tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    ))
    if from_text != direct:
        raise RuntimeError("chat-template forms disagree")
    return from_text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--patch", default=None)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    specs = [json.loads(line) for line in Path(args.calibration).read_text().splitlines() if line.strip()]
    started = time.perf_counter()
    result: dict[str, Any] = {
        "experiment_id": "P3-DWQ-GENERATION-SANITY",
        "phase": 3,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model_path": str(Path(args.model).resolve()),
        "patch_path": str(Path(args.patch).resolve()) if args.patch else None,
        "calibration_path": str(Path(args.calibration).resolve()),
        "case_count": len(specs),
        "max_tokens": args.max_tokens,
        "cases": [],
        "exception": None,
    }
    model = tokenizer = None
    try:
        from mlx_lm.generate import generate_step
        from mlx_lm.sample_utils import make_sampler
        from mlx_lm.utils import load

        model, tokenizer = load(args.model)
        modules = dict(model.named_modules())
        if args.patch:
            result["patched_modules"] = sorted(load_patch(Path(args.patch), modules))
        eos_ids = {int(value) for value in (tokenizer.eos_token_ids or [])}
        sampler = make_sampler(0.0, 1.0)
        for spec in specs:
            ids = prompt_ids(tokenizer, spec["prompt"])
            generated: list[int] = []
            finish_reason = "length"
            generator = generate_step(mx.array(ids), model, max_tokens=args.max_tokens, sampler=sampler)
            try:
                for token, _logprobs in generator:
                    token_id = int(token.item()) if hasattr(token, "item") else int(token)
                    generated.append(token_id)
                    if token_id in eos_ids:
                        finish_reason = "eos"
                        break
                    if len(generated) >= args.max_tokens:
                        break
            finally:
                generator.close()
            text = tokenizer.decode(generated, skip_special_tokens=True)
            result["cases"].append({
                "case_id": spec["case_id"],
                "category": spec.get("category"),
                "prompt_tokens": len(ids),
                "generated_tokens": len(generated),
                "finish_reason": finish_reason,
                "text": text,
            })
            del generator
            mx.clear_cache()
            gc.collect()
    except BaseException as exc:
        result["exception"] = f"{type(exc).__name__}: {exc}"
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        mx.clear_cache()
    result["peak_memory_bytes"] = int(mx.get_peak_memory())
    result["peak_memory_gib"] = result["peak_memory_bytes"] / (1024**3)
    result["rss_max_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    result["total_elapsed_seconds"] = time.perf_counter() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "case_count": len(result["cases"]), "exception": result["exception"], "peak_memory_gib": result["peak_memory_gib"]}, sort_keys=True))
    return 1 if result["exception"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
