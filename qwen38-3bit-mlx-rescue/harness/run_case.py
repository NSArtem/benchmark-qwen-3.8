#!/usr/bin/env python3
"""Minimal public mlx-lm runner for one bounded rescue configuration.

The monitor is telemetry only.  It never stops for swap, page-outs, low free
memory, or pressure; the only automatic stop is the explicit hard timeout.
"""

from __future__ import annotations

import argparse
import json
import re
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm import load, stream_generate


MODEL = "lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly"
DEFAULT_MAX_TOKENS = 128
DEFAULT_MEMORY_INTERVAL = 2.0


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
        completed = subprocess.run(
            list(command), check=False, capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout if completed.returncode == 0 else None


def swap_snapshot() -> dict[str, float | None]:
    raw = command_output("sysctl", "-n", "vm.swapusage") or ""
    result: dict[str, float | None] = {
        "swap_total_mb": None,
        "swap_used_mb": None,
        "swap_free_mb": None,
    }
    for key, label in (
        ("swap_total_mb", "total"),
        ("swap_used_mb", "used"),
        ("swap_free_mb", "free"),
    ):
        match = re.search(rf"{label}\s*=\s*([0-9.]+)M", raw)
        if match:
            result[key] = float(match.group(1))
    return result


def vm_stat_snapshot() -> dict[str, int | None]:
    raw = command_output("vm_stat") or ""
    page_size_match = re.search(r"page size of (\d+) bytes", raw)
    fields = {
        "Pages free": "pages_free",
        "Pages active": "pages_active",
        "Pages inactive": "pages_inactive",
        "Pages speculative": "pages_speculative",
        "Pages wired down": "pages_wired_down",
        "Pages stored in compressor": "pages_stored_in_compressor",
        "Pageins": "pageins",
        "Pageouts": "pageouts",
        "Swapins": "swapins",
        "Swapouts": "swapouts",
    }
    result: dict[str, int | None] = {
        "page_size_bytes": int(page_size_match.group(1)) if page_size_match else None
    }
    for source, destination in fields.items():
        match = re.search(rf"^\s*{re.escape(source)}:\s+(\d+)\.?\s*$", raw, re.MULTILINE)
        result[destination] = int(match.group(1)) if match else None
    return result


def system_snapshot() -> dict[str, Any]:
    return {
        "timestamp": time.time(),
        "monotonic": time.monotonic(),
        **swap_snapshot(),
        **vm_stat_snapshot(),
    }


def delta(before: dict[str, Any], after: dict[str, Any], key: str) -> int | None:
    old = before.get(key)
    new = after.get(key)
    if isinstance(old, int) and isinstance(new, int):
        return max(new - old, 0)
    return None


class MemoryMonitor:
    def __init__(self, path: Path, interval: float):
        self.path = path
        self.interval = interval
        self.stop_event = threading.Event()
        self.before = system_snapshot()
        self.samples: list[dict[str, Any]] = []
        path.write_text(
            json.dumps({"event": "monitor_start", "memory": self.before}, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        self.thread = threading.Thread(target=self._run, name="mlx-rescue-memory", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        previous = self.before
        previous_time = time.monotonic()
        while not self.stop_event.wait(self.interval):
            current = system_snapshot()
            now = time.monotonic()
            elapsed = max(now - previous_time, 1e-6)
            sample = {
                "event": "sample",
                "elapsed_seconds": round(elapsed, 3),
                "memory": current,
                "pageouts_per_second": self._rate(previous, current, "pageouts", elapsed),
                "swapouts_per_second": self._rate(previous, current, "swapouts", elapsed),
            }
            self.samples.append(sample)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(sample, sort_keys=True) + "\n")
            previous = current
            previous_time = now

    @staticmethod
    def _rate(before: dict[str, Any], after: dict[str, Any], key: str, elapsed: float) -> float | None:
        change = delta(before, after, key)
        return round(change / elapsed, 1) if change is not None else None

    def stop(self) -> dict[str, Any]:
        self.stop_event.set()
        self.thread.join(timeout=max(10.0, self.interval + 2.0))
        after = system_snapshot()
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": "monitor_stop", "memory": after}, sort_keys=True) + "\n")
        swap_values = [
            sample["memory"].get("swap_used_mb")
            for sample in self.samples
            if isinstance(sample["memory"].get("swap_used_mb"), (int, float))
        ]
        before_swap = self.before.get("swap_used_mb")
        after_swap = after.get("swap_used_mb")
        swap_candidates = [
            value
            for value in [before_swap, after_swap, *swap_values]
            if isinstance(value, (int, float))
        ]
        return {
            "before": self.before,
            "after": after,
            "sample_count": len(self.samples),
            "swap_peak_mb": max(swap_candidates, default=None),
            "swap_delta_mb": (
                after_swap - before_swap
                if isinstance(after_swap, (int, float)) and isinstance(before_swap, (int, float))
                else None
            ),
            "pageins_delta": delta(self.before, after, "pageins"),
            "pageouts_delta": delta(self.before, after, "pageouts"),
            "swapins_delta": delta(self.before, after, "swapins"),
            "swapouts_delta": delta(self.before, after, "swapouts"),
            "paging_observed": bool(
                delta(self.before, after, "pageouts") or delta(self.before, after, "swapouts")
            ),
            "policy": "telemetry only; no memory or paging watchdog",
        }


def mlx_memory_snapshot(synchronize: bool = True) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    if synchronize:
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


def format_chat_prompt(tokenizer: Any, raw_prompt: str) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": raw_prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    if encoded and isinstance(encoded[0], (list, tuple)):
        encoded = encoded[0]
    return [int(token) for token in encoded]


def classify_failure(error: BaseException | None) -> str | None:
    if error is None:
        return None
    message = f"{type(error).__name__}: {error}"
    lower = message.lower()
    if "insufficient memory" in lower or "out of memory" in lower or "oom" in lower:
        return "Metal OOM / insufficient memory: " + message
    if isinstance(error, HardTimeout):
        return "hard wall-clock timeout: " + message
    return message


def update_aggregate(path: Path, result: dict[str, Any]) -> None:
    if path.exists():
        aggregate = json.loads(path.read_text(encoding="utf-8"))
    else:
        aggregate = {"model": result["model"], "runtime": result["runtime"], "cases": []}
    cases = [case for case in aggregate.get("cases", []) if case.get("case") != result.get("case")]
    cases.append(result)
    aggregate["cases"] = cases
    aggregate["last_updated_case"] = result.get("case")
    path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_case(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root)
    logs = root / "logs"
    outputs = root / "outputs"
    logs.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"{args.name}.log"
    memory_path = logs / f"{args.name}-memory.jsonl"
    case_path = logs / f"{args.name}.json"
    output_path = outputs / f"{args.name}.txt"

    tee = Tee(log_path)
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = tee  # type: ignore[assignment]
    sys.stderr = tee  # type: ignore[assignment]
    monitor = MemoryMonitor(memory_path, args.memory_interval)
    started = time.perf_counter()
    inference_started: float | None = None
    first_response_time: float | None = None
    last_response_time: float | None = None
    prefill_started: float | None = None
    prefill_completed: float | None = None
    prompt_tokens: list[int] = []
    output_text = ""
    response: Any = None
    responses_seen = 0
    emitted_tokens = 0
    load_seconds: float | None = None
    exception: BaseException | None = None
    old_cache_limit: Any = None
    cache_limit_changed = False
    mlx_after_load: dict[str, Any] = {}
    mlx_after_run: dict[str, Any] = {}
    memory_summary: dict[str, Any] = {}
    timeout_set = False

    def alarm_handler(signum: int, frame: Any) -> None:
        raise HardTimeout(f"exceeded {args.timeout_seconds:.0f} seconds")

    try:
        if hasattr(signal, "SIGALRM") and args.timeout_seconds > 0:
            signal.signal(signal.SIGALRM, alarm_handler)
            signal.setitimer(signal.ITIMER_REAL, args.timeout_seconds)
            timeout_set = True

        monitor.start()
        print(f"case={args.name}")
        print(f"model={args.model}")
        print(f"prompt_file={args.prompt_file}")
        print(f"max_tokens={args.max_tokens}")
        print(f"prefill_step_size={args.prefill_step_size}")
        print(f"cache_limit={args.cache_limit}")
        print(f"quantize_activations={args.quantize_activations}")
        print(f"kv_bits={args.kv_bits}")
        print(f"kv_group_size={args.kv_group_size}")
        print(f"quantized_kv_start={args.quantized_kv_start}")
        print(f"wired_limit_check={command_output('sysctl', 'iogpu.wired_limit_mb')}")
        print("memory policy: telemetry only; no low-memory or swap watchdog")

        if args.cache_limit is not None:
            old_cache_limit = mx.set_cache_limit(args.cache_limit)
            cache_limit_changed = True
            print(f"old_cache_limit={old_cache_limit}")

        load_started = time.perf_counter()
        model_config = {"quantize_activations": True} if args.quantize_activations else None
        model, tokenizer = load(args.model, model_config=model_config)
        load_seconds = time.perf_counter() - load_started
        mlx_after_load = mlx_memory_snapshot()
        print(f"model_load_seconds={load_seconds:.6f}")
        print(f"mlx_after_load={json.dumps(mlx_after_load, sort_keys=True)}")

        raw_prompt = Path(args.prompt_file).read_text(encoding="utf-8")
        prompt_tokens = format_chat_prompt(tokenizer, raw_prompt)
        print(f"prompt_tokens={len(prompt_tokens)}")
        inference_started = time.perf_counter()

        def progress(processed: int, total: int) -> None:
            nonlocal prefill_started, prefill_completed
            now = time.perf_counter()
            if prefill_started is None:
                prefill_started = now
            if processed >= total:
                prefill_completed = now
            print(f"prefill_progress={processed}/{total}")

        generation_kwargs: dict[str, Any] = {"prompt_progress_callback": progress}
        if args.prefill_step_size is not None:
            generation_kwargs["prefill_step_size"] = args.prefill_step_size
        if args.kv_bits is not None:
            generation_kwargs.update(
                {
                    "kv_bits": args.kv_bits,
                    "kv_group_size": args.kv_group_size,
                    "quantized_kv_start": args.quantized_kv_start,
                }
            )

        for current in stream_generate(
            model,
            tokenizer,
            mx.array(prompt_tokens),
            max_tokens=args.max_tokens,
            **generation_kwargs,
        ):
            now = time.perf_counter()
            responses_seen += 1
            if first_response_time is None:
                first_response_time = now
                print(f"first_response_after_seconds={now - inference_started:.6f}")
            last_response_time = now
            output_text += current.text
            if current.finish_reason is None:
                emitted_tokens += 1
            response = current
        print(f"generation_finish_reason={response.finish_reason if response else None}")
        print(f"generation_tokens={response.generation_tokens if response else 0}")
        print(f"emitted_tokens={emitted_tokens}")
    except BaseException as error:
        exception = error
        print(f"benchmark_exception={classify_failure(error)}")
        traceback.print_exc()
    finally:
        if timeout_set:
            signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            mlx_after_run = mlx_memory_snapshot()
        except BaseException:
            mlx_after_run = {}
        memory_summary = monitor.stop()
        wall_seconds = time.perf_counter() - started
        output_path.write_text(output_text, encoding="utf-8")
        finish_reason = response.finish_reason if response is not None else None
        generation_tokens = int(response.generation_tokens) if response is not None else emitted_tokens
        reported_prefill_tps = float(response.prompt_tps) if response is not None else None
        reported_decode_tps = float(response.generation_tps) if response is not None else None
        prefill_seconds = (
            len(prompt_tokens) / reported_prefill_tps
            if reported_prefill_tps and reported_prefill_tps > 0
            else None
        )
        callback_prefill_seconds = (
            prefill_completed - prefill_started
            if prefill_started is not None and prefill_completed is not None
            else None
        )
        ttft_seconds = (
            first_response_time - inference_started
            if first_response_time is not None and inference_started is not None
            else None
        )
        inference_wall_seconds = (
            last_response_time - inference_started
            if last_response_time is not None and inference_started is not None
            else None
        )
        result: dict[str, Any] = {
            "case": args.name,
            "model": args.model,
            "runtime": "mlx-lm public load/stream_generate",
            "configuration": {
                "prefill_step_size": args.prefill_step_size,
                "cache_limit": args.cache_limit,
                "quantize_activations": args.quantize_activations,
                "kv_bits": args.kv_bits,
                "kv_group_size": args.kv_group_size,
                "quantized_kv_start": args.quantized_kv_start,
                "max_tokens": args.max_tokens,
                "greedy": True,
                "enable_thinking": False,
            },
            "prompt_file": args.prompt_file,
            "prompt_tokens": len(prompt_tokens),
            "requested_output_tokens": args.max_tokens,
            "generation_tokens": generation_tokens,
            "emitted_tokens": emitted_tokens,
            "responses_seen": responses_seen,
            "completed_without_exception": exception is None,
            "coherent": None,
            "verdict": None,
            "finish_reason": finish_reason,
            "load_seconds": load_seconds,
            "prefill_seconds": prefill_seconds,
            "prefill_tps": reported_prefill_tps,
            "callback_prefill_seconds": callback_prefill_seconds,
            "callback_prefill_tps": (
                len(prompt_tokens) / callback_prefill_seconds
                if callback_prefill_seconds and callback_prefill_seconds > 0
                else None
            ),
            "ttft_seconds": ttft_seconds,
            "decode_tps": reported_decode_tps,
            "decode_seconds": (
                generation_tokens / reported_decode_tps
                if reported_decode_tps and reported_decode_tps > 0
                else None
            ),
            "inference_wall_seconds": inference_wall_seconds,
            "wall_seconds": wall_seconds,
            "mlx_after_load": mlx_after_load,
            "mlx_after_run": mlx_after_run,
            "mlx_peak_bytes": mlx_after_run.get("peak_bytes") or mlx_after_load.get("peak_bytes"),
            "swap_before_mb": memory_summary.get("before", {}).get("swap_used_mb"),
            "swap_peak_mb": memory_summary.get("swap_peak_mb"),
            "swap_after_mb": memory_summary.get("after", {}).get("swap_used_mb"),
            "swap_delta_mb": memory_summary.get("swap_delta_mb"),
            "pageouts_delta": memory_summary.get("pageouts_delta"),
            "swapouts_delta": memory_summary.get("swapouts_delta"),
            "memory_summary": memory_summary,
            "failure_reason": classify_failure(exception),
            "output_path": str(output_path),
            "memory_path": str(memory_path),
            "log_path": str(log_path),
        }
        case_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        update_aggregate(root / "results.json", result)
        print(json.dumps({
            key: result.get(key)
            for key in (
                "case", "completed_without_exception", "prompt_tokens", "generation_tokens",
                "prefill_tps", "decode_tps", "mlx_peak_bytes", "swap_delta_mb", "failure_reason",
            )
        }, sort_keys=True))
        if cache_limit_changed:
            try:
                mx.set_cache_limit(old_cache_limit)
            except BaseException:
                pass
        sys.stdout, sys.stderr = old_stdout, old_stderr
        tee.close()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--root", default="qwen38-3bit-mlx-rescue")
    parser.add_argument("--prompt-file", default="qwen38-3bit-mlx-rescue/prompt-16k.txt")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--prefill-step-size", type=int, default=None)
    parser.add_argument("--cache-limit", type=int, default=None)
    parser.add_argument("--quantize-activations", action="store_true")
    parser.add_argument("--kv-bits", type=int, default=None)
    parser.add_argument("--kv-group-size", type=int, default=64)
    parser.add_argument("--quantized-kv-start", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--memory-interval", type=float, default=2.0)
    return parser.parse_args()


if __name__ == "__main__":
    run_case(parse_args())
