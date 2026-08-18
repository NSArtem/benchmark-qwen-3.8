#!/usr/bin/env python3
"""Fresh-process cold probe through the repository's no-wired runtime."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import platform
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import load

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.runtime.qwen38_runtime import stream_generate_no_wired


MLX_LM_COMMIT = "254d153fdeb6f150edd4fc5a54f9828638481fa8"


def gib(value: int | None) -> float | None:
    return value / (1024**3) if value is not None else None


def scalar(getter: Any) -> int | None:
    try:
        return int(getter())
    except BaseException:
        return None


def normalize_ids(encoded: Any) -> list[int]:
    if isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    elif hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    if encoded and isinstance(encoded[0], (list, tuple)):
        encoded = encoded[0]
    return [int(value) for value in encoded]


def prompt_ids(tokenizer: Any, content: str) -> tuple[list[int], list[int]]:
    raw = normalize_ids(tokenizer.encode(content, add_special_tokens=False))
    chat = normalize_ids(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )
    return raw, chat


def snapshot(label: str, started: float) -> dict[str, Any]:
    try:
        mx.synchronize()
    except BaseException:
        pass
    active = scalar(mx.get_active_memory)
    cache = scalar(mx.get_cache_memory)
    peak = scalar(mx.get_peak_memory)
    total = active + cache if active is not None and cache is not None else None
    return {
        "label": label,
        "timestamp": time.time(),
        "elapsed_seconds": time.perf_counter() - started,
        "active_bytes": active,
        "active_gib": gib(active),
        "cache_bytes": cache,
        "cache_gib": gib(cache),
        "active_plus_cache_bytes": total,
        "active_plus_cache_gib": gib(total),
        "peak_bytes": peak,
        "peak_gib": gib(peak),
        "rss_max_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }


def os_snapshot() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, command in {
        "vm_stat": ["vm_stat"],
        "vm_swapusage": ["sysctl", "vm.swapusage"],
        "memory_pressure": ["memory_pressure", "-Q"],
    }.items():
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=10)
            result[name] = {
                "returncode": completed.returncode,
                "stdout": completed.stdout[-12000:],
                "stderr": completed.stderr[-4000:],
            }
        except BaseException as exc:
            result[name] = {"exception": f"{type(exc).__name__}: {exc}"}
    return result


def current_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        )
        return completed.stdout.strip()
    except BaseException:
        return None


def model_bytes(model: Any) -> int:
    values = [value for _, value in tree_flatten(model.parameters())]
    mx.eval(values)
    mx.synchronize()
    return sum(int(value.nbytes) for value in values if hasattr(value, "nbytes"))


def classify(text: str, token_ids: list[int], exception: str | None) -> str:
    if exception:
        return "OOM" if "Insufficient Memory" in exception or "OutOfMemory" in exception else "ERROR"
    stripped = text.strip()
    if not stripped or len(token_ids) < 16:
        return "BROKEN"
    if len(set(token_ids[-min(32, len(token_ids)) :])) <= 2:
        return "BROKEN"
    punctuation = sum(char in "!.,;:!?" for char in stripped)
    if punctuation > max(32, len(stripped) * 0.65):
        return "DEGRADED"
    return "COHERENT"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--token-file", default=None)
    parser.add_argument("--target-prompt-tokens", type=int, default=None)
    parser.add_argument("--quant-mode", default="affine")
    parser.add_argument("--quant-bits", type=int, default=3)
    parser.add_argument("--quant-group-size", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    args = parser.parse_args()

    started = time.perf_counter()
    print(
        "[qwen38-runtime] mlx-lm wired_limit wrapper disabled; "
        "MLX wired state untouched",
        file=sys.stderr,
        flush=True,
    )
    prompt_path = Path(args.prompt_file)
    token_path = Path(args.token_file) if args.token_file else prompt_path.with_suffix(".tokens.json")
    output_path = Path(args.output_file)
    result_path = Path(args.result_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_text = prompt_path.read_text(encoding="utf-8")
    prompt_bytes = prompt_path.read_bytes()
    expected = json.loads(token_path.read_text(encoding="utf-8")) if token_path.exists() else {}
    target_prompt_tokens = expected.get("target_prompt_tokens", args.target_prompt_tokens)
    result: dict[str, Any] = {
        "experiment_id": args.experiment_id or f"P1-G64-{target_prompt_tokens or 'custom'}",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": current_commit(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "mlx_version": metadata.version("mlx"),
        "mlx_lm_metadata_version": metadata.version("mlx-lm"),
        "mlx_lm_revision": MLX_LM_COMMIT,
        "model_path": args.model_path,
        "model_quantization": {
            "mode": args.quant_mode,
            "bits": args.quant_bits,
            "group_size": args.quant_group_size,
        },
        "prompt_path": str(prompt_path),
        "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
        "target_prompt_tokens": target_prompt_tokens,
        "prompt_tokens": None,
        "prefill_tokens": None,
        "generated_tokens": 0,
        "generated_token_hash": None,
        "classification": None,
        "prefill_step_size": args.prefill_step_size,
        "ttft_seconds": None,
        "prompt_tps": None,
        "decode_tps": None,
        "generation_seconds": None,
        "total_seconds": None,
        "memory_snapshots": [],
        "os_memory_before": os_snapshot(),
        "os_memory_after": None,
        "exception": None,
        "safety_stop_reason": None,
        "wired_policy": "no-wired helper; MLX wired state untouched",
        "wired_limit_touched": False,
    }
    try:
        if result["mlx_version"] != "0.32.1":
            raise RuntimeError(f"unexpected MLX version {result['mlx_version']}")
        result["memory_snapshots"].append(snapshot("process_start", started))
        model, tokenizer = load(args.model_path)
        result["memory_snapshots"].append(snapshot("after_model_load", started))
        result["model_nbytes"] = model_bytes(model)
        result["memory_snapshots"].append(snapshot("after_model_materialization", started))
        raw_ids, chat = prompt_ids(tokenizer, prompt_text)
        expected_ids = expected.get("tokens")
        if expected_ids is not None and chat != [int(value) for value in expected_ids]:
            raise RuntimeError("prompt tokenization differs from saved token IDs")
        result["prompt_tokens"] = len(chat)
        result["prefill_tokens"] = len(chat) - 1
        result["raw_prompt_tokens"] = len(raw_ids)
        result["memory_snapshots"].append(snapshot("after_prompt_tokenization", started))

        progress: list[dict[str, Any]] = []
        generation_started = time.perf_counter()

        def callback(processed: int, total: int) -> None:
            progress.append(
                {
                    "processed": processed,
                    "total": total,
                    "elapsed": time.perf_counter() - generation_started,
                }
            )

        generator = stream_generate_no_wired(
            model,
            tokenizer,
            mx.array(chat),
            max_tokens=args.max_tokens,
            prefill_step_size=args.prefill_step_size,
            prompt_progress_callback=callback,
        )
        result["memory_snapshots"].append(snapshot("before_generate_step", started))
        generated: list[int] = []
        first_token_time: float | None = None
        for response in generator:
            generated.append(int(response.token))
            if first_token_time is None:
                first_token_time = time.perf_counter()
                result["memory_snapshots"].append(snapshot("after_prefill_first_token", started))
        completed = time.perf_counter()
        result["memory_snapshots"].append(snapshot("after_completion", started))
        result["generated_tokens"] = len(generated)
        result["generated_token_ids"] = generated
        result["generated_token_hash"] = hashlib.sha256(
            json.dumps(generated, separators=(",", ":")).encode()
        ).hexdigest()
        result["progress_callbacks"] = progress
        if first_token_time is not None:
            result["ttft_seconds"] = first_token_time - generation_started
            prompt_done = next(
                (item["elapsed"] for item in progress if item["processed"] == item["total"]),
                None,
            )
            result["prompt_tps"] = (
                len(chat) / prompt_done if prompt_done and prompt_done > 0 else len(chat) / result["ttft_seconds"]
            )
            result["decode_tps"] = (
                (len(generated) - 1) / (completed - first_token_time) if len(generated) > 1 else None
            )
        result["generation_seconds"] = completed - generation_started
        result["total_seconds"] = time.perf_counter() - started
        decoded = tokenizer.decode(generated)
        result["classification"] = classify(decoded, generated, None)
        output_path.write_text(decoded, encoding="utf-8")
    except BaseException as exc:
        result["exception"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        result["classification"] = classify("", result.get("generated_token_ids", []), result["exception"])
        output_path.write_text("", encoding="utf-8")
    finally:
        result["os_memory_after"] = os_snapshot()
        result["total_seconds"] = time.perf_counter() - started
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True))
        if result["exception"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
