#!/usr/bin/env python3
"""Run one fresh-process MLX 0.32.1 direct-generate alignment case.

This runner intentionally does not call stream_generate(), set_wired_limit(),
set_cache_limit(), or any other tuning API.  The caller starts it once per
prompt/repeat so model loading and MLX state are fresh for every case.
"""

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
import traceback
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import load


MODEL_ID = "lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly"
MLX_LM_COMMIT = "254d153fdeb6f150edd4fc5a54f9828638481fa8"
PREFILL_STEP_SIZE = 2048
MAX_TOKENS = 128

generate_module = importlib.import_module("mlx_lm.generate")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def gib(value: int | None) -> float | None:
    return value / (1024**3) if value is not None else None


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


def decode(tokenizer: Any, token_ids: list[int]) -> str:
    return str(tokenizer.decode(token_ids))


def scalar_memory(getter: Any) -> int | None:
    try:
        return int(getter())
    except BaseException:
        return None


def rss_bytes() -> int | None:
    try:
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except BaseException:
        return None


def device_record() -> dict[str, Any]:
    info = dict(mx.device_info())
    for key in ("memory_size", "max_recommended_working_set_size", "max_buffer_length"):
        if key in info:
            info[f"{key}_gib"] = gib(int(info[key]))
    return info


def wired_state_record() -> dict[str, Any]:
    # Some MLX builds expose a getter; if not, the runner still records that it
    # never called the setter and leaves the default state untouched.
    getter = getattr(mx, "get_wired_limit", None)
    if getter is None:
        return {"getter_available": False, "value": None, "value_gib": None}
    try:
        value = int(getter())
        return {"getter_available": True, "value": value, "value_gib": gib(value)}
    except BaseException as exc:
        return {"getter_available": True, "value": None, "error": f"{type(exc).__name__}: {exc}"}


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
        value = scalar_memory(getter)
        snapshot[field] = value
        snapshot[field.replace("_bytes", "_gib")] = gib(value)
    if snapshot["active_bytes"] is not None and snapshot["cache_bytes"] is not None:
        total = snapshot["active_bytes"] + snapshot["cache_bytes"]
        snapshot["active_plus_cache_bytes"] = total
        snapshot["active_plus_cache_gib"] = gib(total)
    return snapshot


def materialize_model(model: Any) -> int:
    values = [value for _, value in tree_flatten(model.parameters())]
    mx.eval(values)
    mx.synchronize()
    return sum(int(value.nbytes) for value in values if hasattr(value, "nbytes"))


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    prompt_path = Path(args.prompt_file)
    token_path = prompt_path.with_suffix(".tokens.json")
    output_path = Path(args.output_file)
    result_path = Path(args.result_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    prompt_bytes = prompt_path.read_bytes()
    raw_prompt = prompt_bytes.decode("utf-8")
    expected = json.loads(token_path.read_text(encoding="utf-8"))
    result: dict[str, Any] = {
        "test_id": args.test_id,
        "target_prompt_tokens": int(expected["target_prompt_tokens"]),
        "actual_prompt_tokens": None,
        "prefill_tokens": None,
        "repeat_index": args.repeat_index,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "mlx_version": metadata.version("mlx"),
        "mlx_lm_metadata_version": metadata.version("mlx-lm"),
        "mlx_lm_git_commit": MLX_LM_COMMIT,
        "model": MODEL_ID,
        "model_path": args.model_path,
        "prompt_file": str(prompt_path),
        "prompt_sha256": sha256_bytes(prompt_bytes),
        "raw_text_length": len(raw_prompt),
        "raw_prompt_tokens": None,
        "prompt_first_16_token_ids": None,
        "prompt_last_16_token_ids": None,
        "first_token_id": None,
        "first_token_text": None,
        "first_16_token_ids": [],
        "generated_token_ids": [],
        "generated_token_ids_sha256": None,
        "generated_tokens": 0,
        "output_file": str(output_path),
        "log_file": args.log_file,
        "classification": None,
        "prompt_tps": None,
        "generation_tps": None,
        "peak_memory": None,
        "wired_state_before": None,
        "wired_state_after": None,
        "wired_limit_touched": False,
        "device_info": None,
        "model_nbytes": None,
        "memory_snapshots": [],
        "exception_type": None,
        "exception_message": None,
        "traceback": None,
        "finish_reason": None,
        "result": "ERROR",
    }

    def snap(label: str, synchronize: bool = True) -> None:
        result["memory_snapshots"].append(memory_snapshot(label, started, synchronize))

    try:
        if result["mlx_version"] != "0.32.1":
            raise RuntimeError(f"unexpected MLX version: {result['mlx_version']}")
        if result["mlx_lm_git_commit"] != MLX_LM_COMMIT:
            raise RuntimeError("unexpected mlx-lm commit constant")
        result["device_info"] = device_record()
        result["wired_state_before"] = wired_state_record()
        try:
            mx.reset_peak_memory()
        except BaseException:
            pass
        snap("process_start")

        model, tokenizer = load(args.model_path)
        snap("after_model_load")
        result["model_nbytes"] = materialize_model(model)
        snap("after_model_materialization_eval")

        raw_ids = raw_tokens(tokenizer, raw_prompt)
        prompt_ids = chat_tokens(tokenizer, raw_prompt)
        expected_ids = [int(value) for value in expected["tokens"]]
        if prompt_ids != expected_ids:
            raise RuntimeError("runner tokenization differs from saved prompt token IDs")
        if len(prompt_ids) != result["target_prompt_tokens"]:
            raise RuntimeError(
                f"expected {result['target_prompt_tokens']} prompt tokens, got {len(prompt_ids)}"
            )
        result["actual_prompt_tokens"] = len(prompt_ids)
        result["prefill_tokens"] = len(prompt_ids) - 1
        result["raw_prompt_tokens"] = len(raw_ids)
        result["prompt_first_16_token_ids"] = prompt_ids[:16]
        result["prompt_last_16_token_ids"] = prompt_ids[-16:]
        snap("after_prompt_tokenization")

        generation_started = time.perf_counter()
        generator = generate_module.generate_step(
            mx.array(prompt_ids),
            model,
            max_tokens=MAX_TOKENS,
            prefill_step_size=PREFILL_STEP_SIZE,
        )
        snap("immediately_before_consuming_generate_step")

        for token, _logprobs in generator:
            token_id = int(token)
            result["generated_token_ids"].append(token_id)
            if len(result["generated_token_ids"]) == 1:
                result["first_token_id"] = token_id
                result["first_token_text"] = decode(tokenizer, [token_id])
                snap("after_prefill_and_first_generated_token")
            if len(result["generated_token_ids"]) == 16:
                result["first_16_token_ids"] = list(result["generated_token_ids"])

        generation_seconds = time.perf_counter() - generation_started
        result["generated_tokens"] = len(result["generated_token_ids"])
        result["generation_tps"] = (
            result["generated_tokens"] / generation_seconds if generation_seconds > 0 else None
        )
        result["prompt_tps"] = None
        result["finish_reason"] = "max_tokens" if len(result["generated_token_ids"]) == MAX_TOKENS else "generator_stopped"
        completion = decode(tokenizer, result["generated_token_ids"])
        output_path.write_text(completion, encoding="utf-8")
        token_bytes = json.dumps(result["generated_token_ids"], separators=(",", ":")).encode()
        result["generated_token_ids_sha256"] = sha256_bytes(token_bytes)
        result["result"] = "PASS" if len(result["generated_token_ids"]) == MAX_TOKENS else "INCOMPLETE"
        snap("after_completion")
    except BaseException as exc:
        result["result"] = "OOM" if "memory" in str(exc).lower() or "outofmemory" in str(exc).lower() else "ERROR"
        result["exception_type"] = type(exc).__name__
        result["exception_message"] = str(exc)
        result["traceback"] = traceback.format_exc()
        try:
            snap("exception", synchronize=False)
        except BaseException:
            pass
        print(result["traceback"], end="", file=sys.stderr)
    finally:
        result["wired_state_after"] = wired_state_record()
        result["peak_memory"] = next(
            (snapshot.get("peak_bytes") for snapshot in reversed(result["memory_snapshots"]) if snapshot.get("peak_bytes") is not None),
            None,
        )
        result["wall_seconds"] = time.perf_counter() - started
        result["rss_max_bytes"] = rss_bytes()
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "test_id": result["test_id"],
                    "result": result["result"],
                    "prompt_tokens": result["actual_prompt_tokens"],
                    "generated_tokens": result["generated_tokens"],
                    "first_token_id": result["first_token_id"],
                    "classification": result["classification"],
                    "peak_memory_gib": gib(result["peak_memory"]),
                },
                sort_keys=True,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-id", required=True)
    parser.add_argument("--repeat-index", required=True, type=int)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--log-file", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
