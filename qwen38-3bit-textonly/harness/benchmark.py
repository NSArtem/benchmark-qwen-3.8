#!/usr/bin/env python3
"""Native mlx-lm benchmark runner for Qwen3.8 27B 3-bit TextOnly.

This runner deliberately uses the public ``mlx_lm.load`` and
``mlx_lm.stream_generate`` APIs.  The memory sampler is telemetry only: it
does not stop on low free memory, swap growth, page-outs, or pressure levels.
The only automatic stop is the explicit wall-clock timeout (and normal Python
exceptions such as a Metal OOM).
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
DEFAULT_MAX_TOKENS = 256
DEFAULT_MEMORY_INTERVAL = 2.0


class HardTimeout(RuntimeError):
    pass


class Tee:
    def __init__(self, path: Path):
        self._path = path
        self._file = path.open("w", encoding="utf-8", buffering=1)
        self._console = sys.__stdout__

    def write(self, value: str) -> int:
        self._console.write(value)
        self._console.flush()
        self._file.write(value)
        self._file.flush()
        return len(value)

    def flush(self) -> None:
        self._console.flush()
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def command_output(*command: str) -> str | None:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def vm_stat_snapshot() -> dict[str, Any]:
    raw = command_output("vm_stat")
    if raw is None:
        return {}
    page_size_match = re.search(r"page size of (\d+) bytes", raw)
    fields = {
        "Pages free": "pages_free",
        "Pages active": "pages_active",
        "Pages inactive": "pages_inactive",
        "Pages speculative": "pages_speculative",
        "Pages throttled": "pages_throttled",
        "Pages wired down": "pages_wired_down",
        "Pages purgeable": "pages_purgeable",
        "Pages stored in compressor": "pages_stored_in_compressor",
        "Pages occupied by compressor": "pages_occupied_by_compressor",
        "Pageins": "pageins",
        "Pageouts": "pageouts",
        "Swapins": "swapins",
        "Swapouts": "swapouts",
    }
    result: dict[str, Any] = {
        "page_size_bytes": int(page_size_match.group(1)) if page_size_match else None,
    }
    for source, destination in fields.items():
        match = re.search(rf"^\s*{re.escape(source)}:\s+(\d+)\.?\s*$", raw, re.MULTILINE)
        if match:
            result[destination] = int(match.group(1))
    return result


def swap_snapshot() -> dict[str, float | None]:
    raw = command_output("sysctl", "-n", "vm.swapusage")
    result: dict[str, float | None] = {
        "swap_total_mb": None,
        "swap_used_mb": None,
        "swap_free_mb": None,
    }
    if raw:
        for key, label in (
            ("swap_total_mb", "total"),
            ("swap_used_mb", "used"),
            ("swap_free_mb", "free"),
        ):
            match = re.search(rf"{label}\s*=\s*([0-9.]+)M", raw)
            if match:
                result[key] = float(match.group(1))
    return result


def free_percent_snapshot() -> float | None:
    raw = command_output("memory_pressure", "-Q")
    if raw is None:
        return None
    match = re.search(r"free percentage:\s*([0-9.]+)%", raw, re.IGNORECASE)
    return float(match.group(1)) if match else None


def system_snapshot() -> dict[str, Any]:
    return {
        "timestamp": time.time(),
        "monotonic": time.monotonic(),
        **swap_snapshot(),
        "free_percent": free_percent_snapshot(),
        **vm_stat_snapshot(),
    }


def counter_delta(before: dict[str, Any], after: dict[str, Any], key: str) -> int | None:
    old = before.get(key)
    new = after.get(key)
    if isinstance(old, int) and isinstance(new, int):
        return max(new - old, 0)
    return None


class MemoryMonitor:
    """Record system memory activity without making policy decisions."""

    def __init__(self, path: Path, interval_seconds: float):
        self.path = path
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.samples: list[dict[str, Any]] = []
        self.before = system_snapshot()
        self.after: dict[str, Any] | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="qwen38-memory-monitor",
            daemon=True,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"event": "monitor_start", "memory": self.before}, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )

    def _write(self, value: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, sort_keys=True) + "\n")

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        previous = self.before
        previous_time = time.monotonic()
        while not self.stop_event.wait(self.interval_seconds):
            current = system_snapshot()
            now = time.monotonic()
            elapsed = max(now - previous_time, 1e-6)
            sample = {
                "event": "sample",
                "elapsed_seconds": round(elapsed, 3),
                "memory": current,
                "pageins_per_second": self._rate(previous, current, "pageins", elapsed),
                "pageouts_per_second": self._rate(previous, current, "pageouts", elapsed),
                "swapins_per_second": self._rate(previous, current, "swapins", elapsed),
                "swapouts_per_second": self._rate(previous, current, "swapouts", elapsed),
            }
            with self.lock:
                self.samples.append(sample)
            self._write(sample)
            previous = current
            previous_time = now

    @staticmethod
    def _rate(
        before: dict[str, Any], after: dict[str, Any], key: str, elapsed: float
    ) -> float | None:
        delta = counter_delta(before, after, key)
        return round(delta / elapsed, 1) if delta is not None else None

    def stop(self) -> dict[str, Any]:
        self.stop_event.set()
        self._thread.join(timeout=max(10.0, self.interval_seconds + 2.0))
        self.after = system_snapshot()
        self._write({"event": "monitor_stop", "memory": self.after})
        with self.lock:
            samples = list(self.samples)
        swap_values = [
            sample["memory"].get("swap_used_mb")
            for sample in samples
            if isinstance(sample["memory"].get("swap_used_mb"), (int, float))
        ]
        free_values = [
            sample["memory"].get("free_percent")
            for sample in samples
            if isinstance(sample["memory"].get("free_percent"), (int, float))
        ]
        pageout_rates = [
            sample.get("pageouts_per_second")
            for sample in samples
            if isinstance(sample.get("pageouts_per_second"), (int, float))
        ]
        swapout_rates = [
            sample.get("swapouts_per_second")
            for sample in samples
            if isinstance(sample.get("swapouts_per_second"), (int, float))
        ]
        peak_swap = max(
            [self.before.get("swap_used_mb"), self.after.get("swap_used_mb"), *swap_values],
            default=None,
        )
        return {
            "before": self.before,
            "after": self.after,
            "sample_count": len(samples),
            "swap_peak_mb": peak_swap,
            "free_percent_min_sampled": min(free_values) if free_values else None,
            "max_pageouts_per_second": max(pageout_rates) if pageout_rates else None,
            "max_swapouts_per_second": max(swapout_rates) if swapout_rates else None,
            "pageins_delta": counter_delta(self.before, self.after, "pageins"),
            "pageouts_delta": counter_delta(self.before, self.after, "pageouts"),
            "swapins_delta": counter_delta(self.before, self.after, "swapins"),
            "swapouts_delta": counter_delta(self.before, self.after, "swapouts"),
            "paging_observed": bool(pageout_rates or swapout_rates),
            "real_thrashing": None,
            "thrashing_note": (
                "Telemetry only; real thrashing requires sustained adjacent heavy paging "
                "with a clear loss of generation progress and is assessed after the run."
            ),
        }


def format_prompt(tokenizer: Any, raw_prompt: str) -> list[int]:
    if getattr(tokenizer, "has_chat_template", False):
        tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": raw_prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    else:
        tokens = tokenizer.encode(raw_prompt)
    if isinstance(tokens, dict):
        tokens = tokens["input_ids"]
    return [int(token) for token in tokens]


def mlx_peak_bytes() -> int | None:
    try:
        mx.synchronize()
        return int(mx.get_peak_memory())
    except Exception:
        return None


def classify_failure(error: BaseException) -> str:
    message = f"{type(error).__name__}: {error}"
    lower = message.lower()
    if "insufficient memory" in lower or "out of memory" in lower or "oom" in lower:
        return "Metal OOM / insufficient memory: " + message
    if isinstance(error, HardTimeout):
        return "hard wall-clock timeout: " + message
    return message


def run_case(args: argparse.Namespace) -> dict[str, Any]:
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    name = args.name
    log_path = result_dir / f"{name}.log"
    output_path = result_dir / f"{name}-output.txt"
    memory_path = result_dir / f"{name}-memory.jsonl"
    result_path = result_dir / f"{name}.json"

    tee = Tee(log_path)
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = tee  # type: ignore[assignment]
    sys.stderr = tee  # type: ignore[assignment]
    monitor = MemoryMonitor(memory_path, args.memory_interval)
    started = time.perf_counter()
    prompt_tokens: list[int] = []
    output_text = ""
    response = None
    first_response_time: float | None = None
    last_response_time: float | None = None
    prefill_callback_start: float | None = None
    prefill_last_chunk: float | None = None
    prefill_complete: float | None = None
    load_seconds: float | None = None
    inference_started: float | None = None
    exception: BaseException | None = None
    timeout_was_set = False

    def alarm_handler(signum: int, frame: Any) -> None:
        raise HardTimeout(f"exceeded {args.timeout_seconds:.0f} seconds")

    try:
        if hasattr(signal, "SIGALRM") and args.timeout_seconds > 0:
            signal.signal(signal.SIGALRM, alarm_handler)
            signal.setitimer(signal.ITIMER_REAL, args.timeout_seconds)
            timeout_was_set = True

        monitor.start()
        print(f"case={name}")
        print(f"model={args.model}")
        print(f"prompt_file={args.prompt_file}")
        print(f"max_tokens={args.max_tokens}")
        print("memory policy: telemetry only; no low-memory or swap watchdog")

        load_started = time.perf_counter()
        model, tokenizer = load(args.model)
        mx.synchronize()
        load_seconds = time.perf_counter() - load_started
        print(f"model_load_seconds={load_seconds:.6f}")
        print(f"mlx_peak_after_load_bytes={mlx_peak_bytes()}")

        raw_prompt = Path(args.prompt_file).read_text(encoding="utf-8")
        prompt_tokens = format_prompt(tokenizer, raw_prompt)
        print(f"prompt_tokens={len(prompt_tokens)}")
        inference_started = time.perf_counter()

        def progress(processed: int, total: int) -> None:
            nonlocal prefill_callback_start, prefill_last_chunk, prefill_complete
            now = time.perf_counter()
            if prefill_callback_start is None:
                prefill_callback_start = now
            if processed < total:
                prefill_last_chunk = now
            if processed >= total:
                prefill_complete = now
            print(f"prefill_progress={processed}/{total}")

        for current in stream_generate(
            model,
            tokenizer,
            mx.array(prompt_tokens),
            max_tokens=args.max_tokens,
            prompt_progress_callback=progress,
        ):
            now = time.perf_counter()
            if first_response_time is None:
                first_response_time = now
                print(f"first_response_after_seconds={now - inference_started:.6f}")
            last_response_time = now
            output_text += current.text
            response = current
        print(f"generation_finish_reason={response.finish_reason if response else None}")
        print(f"output_tokens={response.generation_tokens if response else 0}")
    except BaseException as error:  # Persist a useful record even for Metal errors/timeouts.
        exception = error
        print(f"benchmark_exception={classify_failure(error)}")
        traceback.print_exc()
    finally:
        if timeout_was_set:
            signal.setitimer(signal.ITIMER_REAL, 0)
        memory_summary = monitor.stop()
        wall_seconds = time.perf_counter() - started
        output_path.write_text(output_text, encoding="utf-8")
        peak_bytes = mlx_peak_bytes()
        output_tokens = int(response.generation_tokens) if response is not None else 0
        finish_reason = response.finish_reason if response is not None else None
        prefill_seconds = None
        prefill_tps = None
        if prefill_callback_start is not None and prefill_last_chunk is not None:
            prefill_seconds = max(prefill_last_chunk - prefill_callback_start, 0.0)
            prefill_tps = (max(len(prompt_tokens) - 1, 0) / prefill_seconds) if prefill_seconds else None
        ttft_seconds = (
            first_response_time - inference_started
            if first_response_time is not None and inference_started is not None
            else None
        )
        decode_seconds = None
        decode_tps = None
        reported_prompt_tps = None
        if response is not None:
            reported_prompt_tps = float(response.prompt_tps)
            decode_tps = float(response.generation_tps)
            decode_seconds = output_tokens / decode_tps if decode_tps > 0 else None
        inference_wall = (
            last_response_time - inference_started
            if last_response_time is not None and inference_started is not None
            else None
        )
        before_swap = memory_summary["before"].get("swap_used_mb")
        after_swap = memory_summary["after"].get("swap_used_mb")
        result: dict[str, Any] = {
            "model": args.model,
            "runtime": "mlx-lm public load/stream_generate",
            "case": name,
            "context_target": args.context_target,
            "prompt_file": args.prompt_file,
            "prompt_tokens": len(prompt_tokens),
            "output_tokens": output_tokens,
            "success": exception is None and output_tokens >= args.minimum_output_tokens,
            "coherent": None,
            "coherence_note": None,
            "eos": finish_reason == "stop" if response is not None else None,
            "finish_reason": finish_reason,
            "load_seconds": load_seconds,
            "prefill_seconds": prefill_seconds,
            "prefill_tps": prefill_tps,
            "mlx_reported_prompt_tps": reported_prompt_tps,
            "ttft_seconds": ttft_seconds,
            "decode_seconds": decode_seconds,
            "decode_tps": decode_tps,
            "decode_wall_seconds": inference_wall - ttft_seconds if inference_wall is not None and ttft_seconds is not None else None,
            "wall_seconds": wall_seconds,
            "inference_wall_seconds": inference_wall,
            "mlx_peak_bytes": peak_bytes,
            "swap_before_mb": before_swap,
            "swap_peak_mb": memory_summary.get("swap_peak_mb"),
            "swap_after_mb": after_swap,
            "swap_delta_mb": after_swap - before_swap if isinstance(after_swap, (int, float)) and isinstance(before_swap, (int, float)) else None,
            "pageins_delta": memory_summary.get("pageins_delta"),
            "pageouts_delta": memory_summary.get("pageouts_delta"),
            "swapins_delta": memory_summary.get("swapins_delta"),
            "swapouts_delta": memory_summary.get("swapouts_delta"),
            "free_percent_min_sampled": memory_summary.get("free_percent_min_sampled"),
            "max_pageouts_per_second": memory_summary.get("max_pageouts_per_second"),
            "max_swapouts_per_second": memory_summary.get("max_swapouts_per_second"),
            "paging_observed": memory_summary.get("paging_observed"),
            "real_thrashing": None,
            "machine_responsive": None,
            "failure_reason": classify_failure(exception) if exception else None,
            "output_path": str(output_path),
            "memory_path": str(memory_path),
            "log_path": str(log_path),
            "memory_summary": memory_summary,
        }
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({k: result[k] for k in (
            "success", "prompt_tokens", "output_tokens", "prefill_tps", "ttft_seconds",
            "decode_tps", "mlx_peak_bytes", "swap_delta_mb", "failure_reason",
        )}, sort_keys=True))
        sys.stdout, sys.stderr = old_stdout, old_stderr
        tee.close()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--context-target", default=None)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--minimum-output-tokens", type=int, default=128)
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--memory-interval", type=float, default=DEFAULT_MEMORY_INTERVAL)
    return parser.parse_args()


if __name__ == "__main__":
    run_case(parse_args())
