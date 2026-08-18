#!/usr/bin/env python
"""Run one fresh-process Qwen3.8 Metal-or-ops GatedDeltaNet diagnostic."""

from __future__ import annotations

import argparse
import json
import re
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import mlx.core as mx

from mlx_lm.models import gated_delta as gd
from mlx_lm.models import qwen3_5 as q35
from mlx_lm import load, stream_generate


MODEL_ID = "lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly"
MLX_LM_COMMIT = "254d153fdeb6f150edd4fc5a54f9828638481fa8"
DEFAULT_ROOT = "qwen38-mlx-gated-delta-ab"


class HardTimeout(RuntimeError):
    pass


class Tee:
    def __init__(self, path: Path):
        self.path = path
        self.file = path.open("w", encoding="utf-8", buffering=1)
        self.console = sys.__stdout__

    def write(self, value: str) -> int:
        self.console.write(value)
        self.console.flush()
        self.file.write(value)
        self.file.flush()
        return len(value)

    def flush(self) -> None:
        self.console.flush()
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def command_output(*command: str) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def swap_used_mb() -> float | None:
    raw = command_output("sysctl", "-n", "vm.swapusage") or ""
    match = re.search(r"used\s*=\s*([0-9.]+)M", raw)
    return float(match.group(1)) if match else None


def memory_snapshot() -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    try:
        mx.synchronize()
    except BaseException:
        result["synchronize_error"] = 1
    for name, getter in (
        ("active_bytes", mx.get_active_memory),
        ("peak_bytes", mx.get_peak_memory),
        ("cache_bytes", mx.get_cache_memory),
    ):
        try:
            result[name] = int(getter())
        except BaseException:
            result[name] = None
    return result


def chat_tokens(tokenizer: Any, raw_prompt: str) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": raw_prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    elif hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    if encoded and isinstance(encoded[0], (list, tuple)):
        encoded = encoded[0]
    return [int(value) for value in encoded]


def decode_one(tokenizer: Any, token_id: int) -> str:
    try:
        return str(tokenizer.decode([token_id]))
    except BaseException:
        try:
            return str(tokenizer._tokenizer.decode([token_id]))
        except BaseException:
            return ""


def top_logprobs(logprobs: Any, limit: int = 10) -> list[dict[str, float | int]]:
    try:
        values = [float(value) for value in logprobs.tolist()]
    except BaseException:
        return []
    order = sorted(range(len(values)), key=lambda index: values[index], reverse=True)[:limit]
    return [{"token_id": index, "logprob": values[index]} for index in order]


def failure_text(error: BaseException | None) -> str | None:
    if error is None:
        return None
    return f"{type(error).__name__}: {error}"


def update_results(path: Path, result: dict[str, Any]) -> None:
    if path.exists():
        aggregate = json.loads(path.read_text(encoding="utf-8"))
    else:
        aggregate = {
            "model": MODEL_ID,
            "mlx_lm_commit": MLX_LM_COMMIT,
            "tests": [],
        }
    tests = [item for item in aggregate.get("tests", []) if item.get("test_id") != result["test_id"]]
    tests.append(result)
    aggregate["tests"] = tests
    aggregate["last_updated_test"] = result["test_id"]
    path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def install_mode(mode: str) -> tuple[bool, Any]:
    if mode == "metal":
        active = q35.gated_delta_update is gd.gated_delta_update
        if not active:
            raise RuntimeError("qwen3_5.gated_delta_update was unexpectedly patched before Metal mode")
        return False, None
    if mode != "ops":
        raise ValueError(f"unknown mode {mode}")

    original = gd.gated_delta_update

    def force_ops(*args: Any, **kwargs: Any) -> Any:
        force_ops.calls += 1
        kwargs["use_kernel"] = False
        return original(*args, **kwargs)

    force_ops.calls = 0
    q35.gated_delta_update = force_ops
    active = q35.gated_delta_update is force_ops and q35.gated_delta_update is not gd.gated_delta_update
    if not active:
        raise RuntimeError("failed to patch the symbol actually referenced by qwen3_5")
    return True, force_ops


def run_case(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root)
    logs = root / "logs"
    outputs = root / "outputs"
    logs.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"{args.name}.log"
    output_path = outputs / f"{args.name}.txt"
    result_path = logs / f"{args.name}.json"

    tee = Tee(log_path)
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = tee  # type: ignore[assignment]
    sys.stderr = tee  # type: ignore[assignment]

    started = time.perf_counter()
    inference_started: float | None = None
    prefill_started: float | None = None
    prefill_completed: float | None = None
    model_load_seconds: float | None = None
    model_memory: dict[str, Any] = {}
    run_memory: dict[str, Any] = {}
    before_swap = swap_used_mb()
    response: Any = None
    output_text = ""
    exception: BaseException | None = None
    first_token_id: int | None = None
    first_token_text = ""
    first_top_logprobs: list[dict[str, float | int]] = []
    first16: list[int] = []
    response_count = 0
    prompt_tokens: int | None = None
    ops_patch_active = False
    ops_wrapper: Any = None

    def alarm_handler(signum: int, frame: Any) -> None:
        raise HardTimeout(f"exceeded {args.timeout_seconds:.0f} seconds")

    timeout_set = False
    try:
        if hasattr(signal, "SIGALRM") and args.timeout_seconds > 0:
            signal.signal(signal.SIGALRM, alarm_handler)
            signal.setitimer(signal.ITIMER_REAL, args.timeout_seconds)
            timeout_set = True

        ops_patch_active, ops_wrapper = install_mode(args.mode)
        print(f"test_id={args.name}")
        print(f"model={args.model_id}")
        print(f"mlx_lm_commit={MLX_LM_COMMIT}")
        print(f"mode={args.mode}")
        print(f"qwen3_5_symbol_is_gd_symbol={q35.gated_delta_update is gd.gated_delta_update}")
        print(f"ops_patch_active={ops_patch_active}")
        print(f"model_path={args.model_path}")
        print(f"prompt_file={args.prompt_file}")
        print(f"max_output_tokens={args.max_tokens}")
        print("prefill_step_size=mlx_lm.generate_step_default(2048)")
        print("greedy=True")
        print("enable_thinking=False")
        print(f"iogpu_wired_limit={command_output('sysctl', 'iogpu.wired_limit_mb')}")
        print("no KV quantization, activation quantization, MTP, APC, or prompt cache")

        load_started = time.perf_counter()
        model, tokenizer = load(args.model_path)
        model_load_seconds = time.perf_counter() - load_started
        model_memory = memory_snapshot()
        raw_prompt = Path(args.prompt_file).read_text(encoding="utf-8")
        prompt_ids = chat_tokens(tokenizer, raw_prompt)
        prompt_tokens = len(prompt_ids)
        print(f"prompt_tokens={prompt_tokens}")
        print(f"model_load_seconds={model_load_seconds:.6f}")
        print(f"mlx_after_load={json.dumps(model_memory, sort_keys=True)}")

        inference_started = time.perf_counter()

        def progress(processed: int, total: int) -> None:
            nonlocal prefill_started, prefill_completed
            now = time.perf_counter()
            if prefill_started is None and processed > 0:
                prefill_started = now
            if processed >= total:
                prefill_completed = now
            print(f"prefill_progress={processed}/{total}")

        for current in stream_generate(
            model,
            tokenizer,
            mx.array(prompt_ids),
            max_tokens=args.max_tokens,
            prompt_progress_callback=progress,
        ):
            response_count += 1
            response = current
            output_text += current.text
            token_id = int(current.token)
            if first_token_id is None:
                first_token_id = token_id
                first_token_text = decode_one(tokenizer, token_id)
                first_top_logprobs = top_logprobs(current.logprobs)
            if len(first16) < 16:
                first16.append(token_id)
        print(f"generation_finish_reason={response.finish_reason if response else None}")
        print(f"generation_tokens={response.generation_tokens if response else 0}")
        print(f"first_token_id={first_token_id}")
        print(f"first_token_text={first_token_text!r}")
        print(f"first_16_token_ids={first16}")
        print(f"first_token_top_logprobs={json.dumps(first_top_logprobs, sort_keys=True)}")
        if ops_wrapper is not None:
            print(f"ops_wrapper_call_count={ops_wrapper.calls}")
    except BaseException as error:
        exception = error
        print(f"benchmark_exception={failure_text(error)}")
        traceback.print_exc()
    finally:
        if timeout_set:
            signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            run_memory = memory_snapshot()
        except BaseException:
            run_memory = {}
        after_swap = swap_used_mb()
        wall_seconds = time.perf_counter() - started
        output_path.write_text(output_text, encoding="utf-8")
        generation_tokens = int(response.generation_tokens) if response is not None else None
        prompt_tps = float(response.prompt_tps) if response is not None else None
        generation_tps = float(response.generation_tps) if response is not None else None
        callback_prefill_seconds = (
            prefill_completed - prefill_started
            if prefill_started is not None and prefill_completed is not None
            else None
        )
        result: dict[str, Any] = {
            "test_id": args.name,
            "model": args.model_id,
            "mlx_lm_commit": MLX_LM_COMMIT,
            "model_path": args.model_path,
            "implementation": args.mode,
            "prompt_file": args.prompt_file,
            "target_prompt_tokens": args.target_prompt_tokens,
            "actual_prompt_tokens": prompt_tokens,
            "prefill_step_size": 2048,
            "max_output_tokens": args.max_tokens,
            "first_token_id": first_token_id,
            "first_token_text": first_token_text or None,
            "first_token_top_logprobs": first_top_logprobs or None,
            "first_16_token_ids": first16 or None,
            "generated_tokens": generation_tokens,
            "response_count": response_count,
            "output_classification": None,
            "prompt_tps": prompt_tps,
            "generation_tps": generation_tps,
            "callback_prefill_seconds": callback_prefill_seconds,
            "callback_prefill_tps": (
                prompt_tokens / callback_prefill_seconds
                if prompt_tokens and callback_prefill_seconds and callback_prefill_seconds > 0
                else None
            ),
            "peak_memory_gb": (
                (run_memory.get("peak_bytes") or model_memory.get("peak_bytes")) / 1e9
                if (run_memory.get("peak_bytes") or model_memory.get("peak_bytes")) is not None
                else None
            ),
            "mlx_after_load": model_memory,
            "mlx_after_run": run_memory,
            "swap_before_mb": before_swap,
            "swap_after_mb": after_swap,
            "swap_delta_mb": after_swap - before_swap if after_swap is not None and before_swap is not None else None,
            "finish_reason": response.finish_reason if response is not None else None,
            "success": exception is None,
            "failure_reason": failure_text(exception),
            "wall_seconds": wall_seconds,
            "model_load_seconds": model_load_seconds,
            "output_file": str(output_path),
            "log_file": str(log_path),
            "result_file": str(result_path),
            "repeat_index": args.repeat_index,
            "ops_patch_active": ops_patch_active,
            "ops_wrapper_call_count": ops_wrapper.calls if ops_wrapper is not None else 0,
            "notes": args.notes,
        }
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        update_results(root / "results.json", result)
        print(json.dumps({
            "test_id": args.name,
            "implementation": args.mode,
            "actual_prompt_tokens": prompt_tokens,
            "generated_tokens": generation_tokens,
            "first_token_id": first_token_id,
            "prompt_tps": prompt_tps,
            "generation_tps": generation_tps,
            "peak_memory_gb": result["peak_memory_gb"],
            "ops_patch_active": ops_patch_active,
            "ops_wrapper_call_count": result["ops_wrapper_call_count"],
            "success": exception is None,
            "failure_reason": failure_text(exception),
        }, sort_keys=True))
        sys.stdout, sys.stderr = old_stdout, old_stderr
        tee.close()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--mode", choices=("metal", "ops"), required=True)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--target-prompt-tokens", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--repeat-index", type=int, required=True)
    parser.add_argument("--notes", default=None)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    return parser.parse_args()


if __name__ == "__main__":
    run_case(parse_args())
