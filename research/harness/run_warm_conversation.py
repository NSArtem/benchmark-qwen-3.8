#!/usr/bin/env python3
"""Measure a persistent no-wired Qwen3.8 coding-agent conversation."""

from __future__ import annotations

import hashlib
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm import load

from research.runtime.qwen38_runtime import NoWiredChat, PINNED_MLX_LM_REVISION


MODEL_PATH = "/Users/artem/.cache/huggingface/hub/models--lukaskremla--Qwen3.8-27B-3bit-MLX-TextOnly/snapshots/c98bba5926f51fec1c8d8737e577221673f524d7"


def scalar(getter: Any) -> int | None:
    try:
        return int(getter())
    except BaseException:
        return None


def gib(value: int | None) -> float | None:
    return value / (1024**3) if value is not None else None


def current_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return completed.stdout.strip()
    except BaseException:
        return None


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


def memory_snapshot(label: str, started: float) -> dict[str, Any]:
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


def classify(text: str, token_ids: list[int]) -> str:
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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--prompt-file", default="research/context-ladder/prompts/prompt-4096.txt")
    parser.add_argument("--result-file", default="research/warm-cache/results-4k.json")
    parser.add_argument("--output-dir", default="research/warm-cache")
    parser.add_argument("--initial-max-tokens", type=int, default=128)
    parser.add_argument("--follow-up-max-tokens", type=int, default=64)
    args = parser.parse_args()

    started = time.perf_counter()
    output_dir = Path(args.output_dir)
    result_path = Path(args.result_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    initial_prompt_path = Path(args.prompt_file)
    initial_prompt = initial_prompt_path.read_text(encoding="utf-8")
    follow_ups = [
        "Your diagnosis is helpful. Which single invariant should the API enforce first, and where should that check live?",
        "Give a minimal TypeScript or SQL-level fix outline for that invariant, including the idempotency behavior on retry.",
        "What focused test would prevent this incident from regressing, and what failure should it assert?",
    ]
    queries = [initial_prompt, *follow_ups]
    result: dict[str, Any] = {
        "experiment_id": "P3-WARM4K",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": current_commit(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "mlx_version": __import__("importlib.metadata", fromlist=["version"]).version("mlx"),
        "mlx_lm_revision": PINNED_MLX_LM_REVISION,
        "model_path": args.model_path,
        "model_quantization": {"mode": "affine", "bits": 3, "group_size": 64},
        "prompt_path": str(initial_prompt_path),
        "prompt_sha256": hashlib.sha256(initial_prompt_path.read_bytes()).hexdigest(),
        "prefill_step_size": 2048,
        "kv_bits": None,
        "kv_group_size": 64,
        "quantized_kv_start": 0,
        "wired_policy": "no-wired helper; persistent prompt_cache; MLX wired state untouched",
        "turns": [],
        "exception": None,
        "safety_stop_reason": None,
    }

    try:
        model, tokenizer = load(args.model_path)
        result["memory_after_model_load"] = memory_snapshot("after_model_load", started)
        session = NoWiredChat(model, tokenizer, enable_thinking=False)
        for index, query in enumerate(queries, start=1):
            before = memory_snapshot(f"before_turn_{index}", started)
            max_tokens = args.initial_max_tokens if index == 1 else args.follow_up_max_tokens
            turn = session.turn(
                query,
                max_tokens=max_tokens,
                temperature=0.0,
                top_p=1.0,
                prefill_step_size=2048,
            )
            after = memory_snapshot(f"after_turn_{index}", started)
            output_path = output_dir / f"turn-{index}.txt"
            token_path = output_dir / f"turn-{index}.tokens.json"
            output_path.write_text(turn.text, encoding="utf-8")
            token_path.write_text(json.dumps(turn.token_ids, indent=2) + "\n", encoding="utf-8")
            result["turns"].append(
                {
                    "turn": index,
                    "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
                    "query_tokens_are_incremental": True,
                    "max_tokens": max_tokens,
                    "newly_processed_prompt_tokens": turn.prompt_tokens,
                    "generated_tokens": turn.generated_tokens,
                    "generated_token_hash": hashlib.sha256(
                        json.dumps(turn.token_ids, separators=(",", ":")).encode()
                    ).hexdigest(),
                    "classification": classify(turn.text, turn.token_ids),
                    "ttft_seconds": turn.ttft_seconds,
                    "prompt_tps": turn.prompt_tps,
                    "decode_tps": turn.generation_tps,
                    "generation_seconds": turn.generation_seconds,
                    "total_seconds": turn.total_seconds,
                    "memory_before": before,
                    "memory_after": after,
                    "os_memory_after": os_snapshot(),
                    "raw_output": str(output_path),
                    "raw_token_ids": str(token_path),
                }
            )
            if after.get("peak_gib") is not None and after["peak_gib"] >= 17.0:
                result["safety_stop_reason"] = "peak reached known 17+ GiB danger envelope"
                break
        result["os_memory_final"] = os_snapshot()
    except BaseException as exc:
        import traceback

        result["exception"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        result["total_elapsed_seconds"] = time.perf_counter() - started
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True))
        if result["exception"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
