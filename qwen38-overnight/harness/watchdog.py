"""Run one benchmark child with a conservative macOS memory watchdog.

The child is placed in its own process group.  Any termination performed here
is sent only to that group, which contains the subprocess started by this
watchdog and its descendants, never unrelated applications.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from memory_utils import memory_summary, snapshot, with_rates


POLICY: dict[str, Any] = {
    "sample_interval_seconds": 1.0,
    "emergency_free_percent": 8.0,
    "emergency_consecutive_samples": 3,
    "low_free_percent": 12.0,
    "paging_free_percent": 20.0,
    "meaningful_pageouts_per_second": 25.0,
    "meaningful_swapouts_per_second": 500.0,
    "low_free_paging_consecutive_samples": 3,
    "critical_swapouts_per_second": 5000.0,
    "critical_paging_consecutive_samples": 2,
    "paging_stall_seconds": 25.0,
    "default_wall_timeout_seconds": 1200.0,
    "rationale": (
        "Swap growth alone is not an abort condition.  Abort only when free memory "
        "is persistently low and page-outs/swap-outs show active pressure, when "
        "free memory is emergency-low, when the child reports Metal OOM, or when "
        "progress stalls during worsening pressure."
    ),
}

OOM_PATTERNS = (
    re.compile(r"insufficient memory", re.IGNORECASE),
    re.compile(r"kIOGPUCommandBufferCallbackErrorOutOfMemory", re.IGNORECASE),
    re.compile(r"out of memory", re.IGNORECASE),
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def append_jsonl(handle: Any, value: Any) -> None:
    handle.write(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n")
    handle.flush()


def terminate_group(process: subprocess.Popen[Any], reason: str) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 10.0
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.25)
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--memory-log", type=Path, required=True)
    parser.add_argument("--child-log", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--child-result", type=Path, required=True)
    parser.add_argument("--progress-file", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=POLICY["default_wall_timeout_seconds"])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("watchdog requires a command after --")

    args.memory_log.parent.mkdir(parents=True, exist_ok=True)
    args.child_log.parent.mkdir(parents=True, exist_ok=True)
    args.progress_file.parent.mkdir(parents=True, exist_ok=True)
    args.memory_log.write_text("", encoding="utf-8")
    args.progress_file.touch()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    started = time.monotonic()
    process: subprocess.Popen[Any]
    with args.child_log.open("a", encoding="utf-8") as child_log, args.memory_log.open(
        "a", encoding="utf-8"
    ) as memory_log:
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[2],
            env=env,
            stdout=child_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        append_jsonl(
            memory_log,
            {
                "event": "watchdog_start",
                "label": args.label,
                "timestamp": time.time(),
                "child_pid": process.pid,
                "command": command,
                "policy": POLICY,
            },
        )
        samples: list[dict[str, Any]] = []
        previous = snapshot()
        append_jsonl(
            memory_log,
            {"event": "sample", "phase": "startup", "memory": previous, "rates": {}},
        )
        samples.append({"event": "sample", "phase": "startup", "memory": previous, "rates": {}})
        emergency_streak = 0
        paging_streak = 0
        critical_streak = 0
        watchdog_reason: str | None = None
        last_progress_mtime = args.progress_file.stat().st_mtime
        last_log_size = 0
        next_sample = time.monotonic()
        while process.poll() is None:
            now = time.monotonic()
            if now < next_sample:
                time.sleep(min(next_sample - now, 0.25))
                continue
            next_sample = time.monotonic() + float(POLICY["sample_interval_seconds"])
            current = snapshot()
            elapsed = max(current["timestamp"] - previous["timestamp"], 1e-6)
            rates = with_rates(previous, current, elapsed)
            try:
                progress_mtime = args.progress_file.stat().st_mtime
                if progress_mtime > last_progress_mtime:
                    last_progress_mtime = progress_mtime
            except OSError:
                progress_mtime = last_progress_mtime

            try:
                log_size = args.child_log.stat().st_size
                log_tail = args.child_log.read_text(encoding="utf-8", errors="replace")[-12000:]
            except OSError:
                log_size = last_log_size
                log_tail = ""
            if log_size > last_log_size:
                last_log_size = log_size
            child_reports_oom = any(pattern.search(log_tail) for pattern in OOM_PATTERNS)

            free = current.get("free_percent")
            pageouts = rates.get("pageouts_per_second") or 0.0
            swapouts = rates.get("swapouts_per_second") or 0.0
            emergency = isinstance(free, (int, float)) and free <= POLICY["emergency_free_percent"]
            paging = (
                isinstance(free, (int, float))
                and free <= POLICY["paging_free_percent"]
                and (pageouts >= POLICY["meaningful_pageouts_per_second"] or swapouts >= POLICY["meaningful_swapouts_per_second"])
            )
            low_free_paging = (
                isinstance(free, (int, float))
                and free <= POLICY["low_free_percent"]
                and (pageouts >= POLICY["meaningful_pageouts_per_second"] or swapouts >= POLICY["meaningful_swapouts_per_second"])
            )
            critical = (
                isinstance(free, (int, float))
                and free <= POLICY["paging_free_percent"]
                and swapouts >= POLICY["critical_swapouts_per_second"]
            )
            emergency_streak = emergency_streak + 1 if emergency else 0
            paging_streak = paging_streak + 1 if low_free_paging else 0
            critical_streak = critical_streak + 1 if critical else 0
            stalled = (
                time.time() - last_progress_mtime >= POLICY["paging_stall_seconds"]
                and isinstance(free, (int, float))
                and free <= POLICY["paging_free_percent"]
                and paging_streak >= 2
            )
            sample = {
                "event": "sample",
                "phase": "running",
                "timestamp": time.time(),
                "memory": current,
                "rates": rates,
                "signals": {
                    "emergency": emergency,
                    "paging": paging,
                    "low_free_paging": low_free_paging,
                    "critical": critical,
                    "stalled_with_pressure": stalled,
                    "child_reports_metal_oom": child_reports_oom,
                },
            }
            append_jsonl(memory_log, sample)
            samples.append(sample)
            previous = current

            if child_reports_oom:
                watchdog_reason = "child_reported_metal_oom"
            elif emergency_streak >= POLICY["emergency_consecutive_samples"]:
                watchdog_reason = "emergency_low_free_memory"
            elif paging_streak >= POLICY["low_free_paging_consecutive_samples"]:
                watchdog_reason = "low_free_memory_with_sustained_paging"
            elif critical_streak >= POLICY["critical_paging_consecutive_samples"]:
                watchdog_reason = "critical_swapout_rate_with_pressure"
            elif stalled:
                watchdog_reason = "progress_stalled_during_memory_pressure"
            if watchdog_reason:
                append_jsonl(
                    memory_log,
                    {
                        "event": "watchdog_abort",
                        "timestamp": time.time(),
                        "reason": watchdog_reason,
                        "memory": current,
                        "rates": rates,
                    },
                )
                terminate_group(process, watchdog_reason)
                break
            if time.monotonic() - started >= args.timeout:
                watchdog_reason = "wall_clock_timeout"
                append_jsonl(
                    memory_log,
                    {"event": "watchdog_abort", "timestamp": time.time(), "reason": watchdog_reason},
                )
                terminate_group(process, watchdog_reason)
                break

        return_code = process.wait()
        after = snapshot()
        append_jsonl(
            memory_log,
            {
                "event": "watchdog_stop",
                "timestamp": time.time(),
                "return_code": return_code,
                "watchdog_abort": watchdog_reason is not None,
                "watchdog_reason": watchdog_reason,
                "memory_after": after,
            },
        )

    summary = memory_summary(samples)
    summary["before"] = samples[0]["memory"] if samples else None
    summary["after"] = after
    if summary.get("swap_before_mb") is None and samples:
        summary["swap_before_mb"] = samples[0]["memory"].get("swap", {}).get("used_mb")
    if summary.get("swap_after_mb") is None:
        summary["swap_after_mb"] = after.get("swap", {}).get("used_mb")
    result = {
        "label": args.label,
        "command": command,
        "child_pid": process.pid,
        "child_returncode": return_code,
        "watchdog_abort": watchdog_reason is not None,
        "watchdog_reason": watchdog_reason,
        "wall_seconds": time.monotonic() - started,
        "policy": POLICY,
        "memory_summary": summary,
        "memory_log": str(args.memory_log),
        "child_log": str(args.child_log),
        "child_result": str(args.child_result),
    }
    write_json(args.result_json, result)
    if not args.child_result.exists():
        write_json(
            args.child_result,
            {
                "status": "watchdog_aborted" if watchdog_reason else "child_failed_without_result",
                "watchdog_abort": watchdog_reason is not None,
                "watchdog_reason": watchdog_reason,
                "error": "The child did not persist a result before it exited.",
            },
        )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if return_code == 0 and watchdog_reason is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
