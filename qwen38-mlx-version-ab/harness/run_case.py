#!/usr/bin/env python3
"""Run one fresh-process Qwen3.8 MLX version-A/B case."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm import load, stream_generate


MODEL_ID = "lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly"
MLX_LM_COMMIT = "254d153fdeb6f150edd4fc5a54f9828638481fa8"
MAX_OUTPUT_TOKENS = 128


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_ids(encoded: Any) -> list[int]:
    if isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    elif hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    if encoded and isinstance(encoded[0], (list, tuple)):
        encoded = encoded[0]
    return [int(value) for value in encoded]


def chat_tokens(tokenizer: Any, raw_prompt: str) -> list[int]:
    return normalize_ids(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": raw_prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )


def raw_tokens(tokenizer: Any, raw_prompt: str) -> list[int]:
    try:
        return normalize_ids(tokenizer.encode(raw_prompt, add_special_tokens=False))
    except TypeError:
        return normalize_ids(tokenizer.encode(raw_prompt))


def decode_token(tokenizer: Any, token_id: int) -> str:
    try:
        return str(tokenizer.decode([token_id]))
    except BaseException:
        return ""


def metadata_version(name: str) -> str:
    return metadata.version(name)


def memory_peak_gb() -> float | None:
    try:
        mx.synchronize()
        return float(mx.get_peak_memory()) / 1e9
    except BaseException:
        return None


def write_aggregate(path: Path, result: dict[str, Any]) -> None:
    if path.exists():
        aggregate = json.loads(path.read_text(encoding="utf-8"))
    else:
        aggregate = {
            "model": MODEL_ID,
            "mlx_lm_git_commit": MLX_LM_COMMIT,
            "tests": [],
        }
    tests = [item for item in aggregate.get("tests", []) if item.get("test_id") != result["test_id"]]
    tests.append(result)
    aggregate["tests"] = tests
    aggregate["last_updated_test"] = result["test_id"]
    path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_case(args: argparse.Namespace) -> None:
    prompt_path = Path(args.prompt_file)
    output_path = Path(args.output_file)
    log_json_path = Path(args.log_json_file)
    results_path = Path(args.results_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_json_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    prompt_bytes = prompt_path.read_bytes()
    raw_prompt = prompt_bytes.decode("utf-8")
    result: dict[str, Any] = {
        "test_id": args.test_id,
        "environment": args.environment,
        "python_version": platform.python_version(),
        "mlx_version": metadata_version("mlx"),
        "mlx_lm_metadata_version": metadata_version("mlx-lm"),
        "mlx_lm_git_commit": MLX_LM_COMMIT,
        "model": MODEL_ID,
        "model_path": args.model_path,
        "prompt_file": args.prompt_file,
        "prompt_sha256": sha256_bytes(prompt_bytes),
        "raw_prompt_tokens": None,
        "prompt_tokens": None,
        "repeat_index": args.repeat_index,
        "first_token_id": None,
        "first_token_text": None,
        "first_16_token_ids": None,
        "generated_tokens": None,
        "classification": None,
        "prompt_tps": None,
        "generation_tps": None,
        "peak_memory_gb": None,
        "finish_reason": None,
        "success": False,
        "failure_reason": None,
        "output_file": args.output_file,
        "log_file": args.log_file,
        "log_json_file": args.log_json_file,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "enable_thinking": False,
        "sampler": "mlx_lm.generate_step default argmax",
        "prefill_step_size": "default",
    }

    output_text = ""
    first_ids: list[int] = []
    first_token_id: int | None = None
    first_token_text: str | None = None
    response: Any = None
    error: BaseException | None = None
    started = time.perf_counter()

    try:
        expected = args.expected_mlx_version
        if result["mlx_version"] != expected:
            raise RuntimeError(f"expected mlx {expected}, found {result['mlx_version']}")
        print(f"test_id={args.test_id}")
        print(f"environment={args.environment}")
        print(f"python_version={result['python_version']}")
        print(f"python_executable={sys.executable}")
        print(f"mlx_version={result['mlx_version']}")
        print(f"mlx_lm_metadata_version={result['mlx_lm_metadata_version']}")
        print(f"mlx_lm_git_commit={MLX_LM_COMMIT}")
        print(f"model={MODEL_ID}")
        print(f"model_path={args.model_path}")
        print(f"prompt_file={args.prompt_file}")
        print(f"prompt_sha256={result['prompt_sha256']}")
        print("settings=enable_thinking_false,default_argmax_sampler,max_tokens_128,default_prefill,no_KV_quantization,no_activation_quantization,no_MTP,no_APC")

        model, tokenizer = load(args.model_path)
        raw_ids = raw_tokens(tokenizer, raw_prompt)
        prompt_ids = chat_tokens(tokenizer, raw_prompt)
        result["raw_prompt_tokens"] = len(raw_ids)
        result["prompt_tokens"] = len(prompt_ids)
        print(f"raw_prompt_tokens={len(raw_ids)}")
        print(f"prompt_tokens={len(prompt_ids)}")

        for current in stream_generate(
            model,
            tokenizer,
            mx.array(prompt_ids),
            max_tokens=MAX_OUTPUT_TOKENS,
        ):
            response = current
            output_text += str(current.text)
            token_id = int(current.token)
            if first_token_id is None:
                first_token_id = token_id
                first_token_text = decode_token(tokenizer, token_id)
            if len(first_ids) < 16:
                first_ids.append(token_id)

        result["success"] = True
        result["first_token_id"] = first_token_id
        result["first_token_text"] = first_token_text
        result["first_16_token_ids"] = first_ids
        result["generated_tokens"] = int(getattr(response, "generation_tokens", len(first_ids))) if response is not None else 0
        result["prompt_tps"] = float(response.prompt_tps) if response is not None else None
        result["generation_tps"] = float(response.generation_tps) if response is not None else None
        result["finish_reason"] = getattr(response, "finish_reason", None) if response is not None else None
        print(f"first_token_id={first_token_id}")
        print(f"first_token_text={first_token_text!r}")
        print(f"first_16_token_ids={first_ids}")
        print(f"generated_tokens={result['generated_tokens']}")
        print(f"finish_reason={result['finish_reason']}")
        print(f"prompt_tps={result['prompt_tps']}")
        print(f"generation_tps={result['generation_tps']}")
    except BaseException as exc:
        error = exc
        result["failure_reason"] = f"{type(exc).__name__}: {exc}"
        print(f"failure_reason={result['failure_reason']}")
        traceback.print_exc()
    finally:
        output_path.write_text(output_text, encoding="utf-8")
        result["first_token_id"] = first_token_id
        result["first_token_text"] = first_token_text
        result["first_16_token_ids"] = first_ids or None
        result["peak_memory_gb"] = memory_peak_gb()
        result["wall_seconds"] = time.perf_counter() - started
        if error is not None:
            result["success"] = False
        log_json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_aggregate(results_path, result)
        print(f"peak_memory_gb={result['peak_memory_gb']}")
        print(json.dumps(result, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-id", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--expected-mlx-version", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--log-json-file", required=True)
    parser.add_argument("--results-file", required=True)
    parser.add_argument("--repeat-index", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run_case(parse_args())
