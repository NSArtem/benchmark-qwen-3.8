#!/usr/bin/env python3
"""Localize the Qwen3.8 MLX 0.32.1 prefill OOM."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata as metadata
import json
import platform
import resource
import sys
import time
import traceback as traceback_module
from collections import Counter
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import load


MODEL_ID = "lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly"
MLX_LM_COMMIT = "254d153fdeb6f150edd4fc5a54f9828638481fa8"
PREFILL_STEP_SIZE = 2048

generate_module = importlib.import_module("mlx_lm.generate")
cache_module = importlib.import_module("mlx_lm.models.cache")
generation_stream = generate_module.generation_stream


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def gib(value: int | None) -> float | None:
    return value / (1024**3) if value is not None else None


def scalar_int(value: Any) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(value.item())
    except BaseException:
        values = value.tolist()
        if isinstance(values, list):
            while isinstance(values, list):
                values = values[0]
        return int(values)


def normalize_ids(encoded: Any) -> list[int]:
    if isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    elif hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    if encoded and isinstance(encoded[0], (list, tuple)):
        encoded = encoded[0]
    return [int(value) for value in encoded]


def chat_tokens(tokenizer: Any, raw_prompt: str) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": raw_prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return normalize_ids(encoded)


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


def device_record() -> dict[str, Any]:
    info = dict(mx.device_info())
    for key in ("memory_size", "max_recommended_working_set_size", "max_buffer_length"):
        if key in info:
            info[f"{key}_gib"] = gib(int(info[key]))
    return info


def rss_bytes() -> int | None:
    try:
        # macOS reports ru_maxrss in bytes.
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except BaseException:
        return None


def memory_snapshot(label: str, started: float, synchronize: bool = True) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "timestamp": time.time(),
        "elapsed_seconds": time.perf_counter() - started,
        "label": label,
        "synchronized": synchronize,
        "active_bytes": None,
        "active_gib": None,
        "cache_bytes": None,
        "cache_gib": None,
        "active_plus_cache_bytes": None,
        "active_plus_cache_gib": None,
        "peak_bytes": None,
        "peak_gib": None,
        "rss_max_bytes": rss_bytes(),
    }
    if synchronize:
        try:
            mx.synchronize()
        except BaseException as exc:
            snapshot["synchronize_error"] = f"{type(exc).__name__}: {exc}"
    for field, getter in (
        ("active_bytes", mx.get_active_memory),
        ("cache_bytes", mx.get_cache_memory),
        ("peak_bytes", mx.get_peak_memory),
    ):
        try:
            value = int(getter())
            snapshot[field] = value
            snapshot[field.replace("_bytes", "_gib")] = gib(value)
        except BaseException as exc:
            snapshot[f"{field}_error"] = f"{type(exc).__name__}: {exc}"
    if snapshot["active_bytes"] is not None and snapshot["cache_bytes"] is not None:
        total = snapshot["active_bytes"] + snapshot["cache_bytes"]
        snapshot["active_plus_cache_bytes"] = total
        snapshot["active_plus_cache_gib"] = gib(total)
    return snapshot


def model_nbytes(model: Any) -> int:
    total = 0
    for _, value in tree_flatten(model.parameters()):
        if hasattr(value, "nbytes"):
            total += int(value.nbytes)
    return total


def materialize_model(model: Any) -> None:
    values = [value for _, value in tree_flatten(model.parameters())]
    mx.eval(values)
    mx.synchronize()


def cache_summary(prompt_cache: list[Any]) -> dict[str, Any]:
    types = Counter(type(entry).__name__ for entry in prompt_cache)
    entries: list[dict[str, Any]] = []
    for index, entry in enumerate(prompt_cache):
        item: dict[str, Any] = {"index": index, "type": type(entry).__name__}
        nbytes = getattr(entry, "nbytes", None)
        if isinstance(nbytes, int):
            item["nbytes"] = nbytes
        state = getattr(entry, "state", None)
        if state is not None:
            item["state_type"] = type(state).__name__
            if hasattr(state, "nbytes"):
                try:
                    item["state_nbytes"] = int(state.nbytes)
                except BaseException:
                    pass
        entries.append(item)
    return {
        "count": len(prompt_cache),
        "type_counts": dict(types),
        "entries": entries,
    }


def append_result(results_path: Path, result: dict[str, Any]) -> None:
    if results_path.exists():
        aggregate = json.loads(results_path.read_text(encoding="utf-8"))
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
    results_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def set_wired_limit(result: dict[str, Any], enabled: bool) -> int | None:
    if not enabled:
        result["wired_limit_mode"] = "default"
        return None
    limit = int(mx.device_info()["max_recommended_working_set_size"])
    old = mx.set_wired_limit(limit)
    result["wired_limit_mode"] = "max_recommended_working_set_size"
    result["wired_limit_bytes"] = limit
    result["wired_limit_gib"] = gib(limit)
    result["wired_limit_previous_bytes"] = int(old) if old is not None else None
    return int(old) if old is not None else None


def set_cache_limit_zero(result: dict[str, Any], enabled: bool) -> int | None:
    if not enabled:
        result["cache_limit_bytes_or_mode"] = "default"
        return None
    old = mx.set_cache_limit(0)
    result["cache_limit_bytes_or_mode"] = 0
    result["cache_limit_previous_bytes"] = int(old) if old is not None else None
    return int(old) if old is not None else None


def restore_limits(old_wired: int | None, old_cache: int | None) -> None:
    if old_wired is not None:
        try:
            mx.set_wired_limit(old_wired)
        except BaseException:
            pass
    if old_cache is not None:
        try:
            mx.set_cache_limit(old_cache)
        except BaseException:
            pass


def run_case(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    prompt_path = Path(args.prompt_file)
    output_json_path = Path(args.log_json_file)
    results_path = Path(args.results_file)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_bytes = prompt_path.read_bytes()
    raw_prompt = prompt_bytes.decode("utf-8")
    result: dict[str, Any] = {
        "test_id": args.test_id,
        "environment": "mlx-0.32.1",
        "mlx_version": metadata.version("mlx"),
        "mlx_lm_metadata_version": metadata.version("mlx-lm"),
        "mlx_lm_git_commit": MLX_LM_COMMIT,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "model": MODEL_ID,
        "model_path": args.model_path,
        "prompt_file": args.prompt_file,
        "prompt_sha256": sha256_bytes(prompt_bytes),
        "raw_text_length": len(raw_prompt),
        "raw_prompt_tokens": None,
        "prompt_tokens": None,
        "prompt_first_16_token_ids": None,
        "prompt_last_16_token_ids": None,
        "execution_path": args.mode,
        "wired_limit_bytes": None,
        "wired_limit_gib": None,
        "wired_limit_previous_bytes": None,
        "cache_limit_bytes_or_mode": "default",
        "cache_limit_previous_bytes": None,
        "prefill_step_size": PREFILL_STEP_SIZE,
        "result": "ERROR",
        "exception_type": None,
        "exception_message": None,
        "traceback": None,
        "memory_snapshots": [],
        "device_info": device_record(),
        "first_token_id": None,
        "first_token_text": None,
        "cache_summary": None,
        "model_nbytes": None,
        "output_file": None,
        "log_file": args.log_file,
    }
    old_wired: int | None = None
    old_cache: int | None = None
    error: BaseException | None = None

    def snap(label: str, synchronize: bool = True) -> None:
        result["memory_snapshots"].append(memory_snapshot(label, started, synchronize))

    try:
        try:
            mx.reset_peak_memory()
        except BaseException:
            pass
        snap("process_start")
        model, tokenizer = load(args.model_path)
        snap("after_model_load")
        materialize_model(model)
        result["model_nbytes"] = model_nbytes(model)
        snap("after_model_materialization_eval")
        raw_ids = raw_tokens(tokenizer, raw_prompt)
        prompt_ids = chat_tokens(tokenizer, raw_prompt)
        result["raw_prompt_tokens"] = len(raw_ids)
        result["prompt_tokens"] = len(prompt_ids)
        result["prompt_first_16_token_ids"] = prompt_ids[:16]
        result["prompt_last_16_token_ids"] = prompt_ids[-16:]
        snap("after_prompt_tokenization")

        old_cache = set_cache_limit_zero(result, args.mode == "cache0")
        if args.mode in ("wired", "manual-wired"):
            old_wired = set_wired_limit(result, True)
            snap("after_wired_limit_set")

        if args.mode in ("baseline", "wired", "cache0"):
            generator = generate_module.generate_step(
                mx.array(prompt_ids),
                model,
                max_tokens=1,
                prefill_step_size=PREFILL_STEP_SIZE,
            )
            snap("before_consuming_generate_step")
            token, _ = next(generator)
            mx.eval(token)
            token_id = scalar_int(token)
            result["first_token_id"] = token_id
            result["first_token_text"] = decode_token(tokenizer, token_id)
            snap("after_first_generated_token")
            result["result"] = "PASS_FIRST_TOKEN"
        else:
            prompt_cache = cache_module.make_prompt_cache(model)
            result["cache_summary"] = cache_summary(prompt_cache)
            snap("S1_after_make_prompt_cache")
            prefill_tokens = mx.array(prompt_ids[:-1])
            snap("S2_before_prefill_model_call")
            with mx.stream(generation_stream):
                prefill_output = model(prefill_tokens[None], cache=prompt_cache)
            # Deliberately do not synchronize here: this is the graph-built point.
            snap("S3_after_graph_construction_before_eval", synchronize=False)
            snap("S4_immediately_before_cache_state_eval", synchronize=False)
            mx.eval([entry.state for entry in prompt_cache])
            snap("S5_after_cache_state_eval")
            result["result"] = "PASS_MANUAL_PREFILL"
    except BaseException as exc:
        error = exc
        result["result"] = "OOM" if "memory" in str(exc).lower() or "outofmemory" in str(exc).lower() else "ERROR"
        result["exception_type"] = type(exc).__name__
        result["exception_message"] = str(exc)
        result["traceback"] = traceback_module.format_exc()
        try:
            snap("exception", synchronize=False)
        except BaseException:
            pass
        print(result["traceback"], end="")
    finally:
        restore_limits(old_wired, old_cache)
        result["wall_seconds"] = time.perf_counter() - started
        result["rss_max_bytes"] = rss_bytes()
        output_json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        append_result(results_path, result)
        print(json.dumps({
            "test_id": args.test_id,
            "result": result["result"],
            "exception_type": result["exception_type"],
            "first_token_id": result["first_token_id"],
            "memory_snapshots": len(result["memory_snapshots"]),
        }, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-id", required=True)
    parser.add_argument("--mode", choices=("baseline", "wired", "cache0", "manual", "manual-wired"), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--log-json-file", required=True)
    parser.add_argument("--results-file", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run_case(parse_args())
