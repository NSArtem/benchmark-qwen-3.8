"""Small macOS memory telemetry helpers used by the overnight harness.

This module deliberately uses only built-in macOS commands.  It does not import
MLX and it never changes system state.
"""

from __future__ import annotations

import re
import subprocess
import time
from typing import Any


def capture(command: list[str], timeout: float = 3.0) -> str:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _counter(raw: str, label: str) -> int | None:
    match = re.search(rf"^\s*{re.escape(label)}:\s+(\d+)", raw, re.MULTILINE)
    return int(match.group(1)) if match else None


def vm_stat_snapshot() -> dict[str, int | None]:
    raw = capture(["vm_stat"])
    page_size_match = re.search(r"page size of (\d+) bytes", raw)
    return {
        "page_size": int(page_size_match.group(1)) if page_size_match else 4096,
        "pageins": _counter(raw, "Pageins"),
        "pageouts": _counter(raw, "Pageouts"),
        "swapins": _counter(raw, "Swapins"),
        "swapouts": _counter(raw, "Swapouts"),
        "compressor_pages": _counter(raw, "Pages stored in compressor"),
    }


def swap_snapshot() -> dict[str, float | None]:
    raw = capture(["sysctl", "-n", "vm.swapusage"])
    values: dict[str, float | None] = {
        "total_mb": None,
        "used_mb": None,
        "free_mb": None,
    }
    for key in values:
        match = re.search(rf"\b{key.removesuffix('_mb')}\s*=\s*([0-9.]+)M", raw)
        if match:
            values[key] = float(match.group(1))
    return values


def free_percent() -> float | None:
    raw = capture(["memory_pressure", "-Q"])
    match = re.search(r"free percentage:\s*([0-9.]+)%", raw, re.IGNORECASE)
    return float(match.group(1)) if match else None


def snapshot() -> dict[str, Any]:
    return {
        "timestamp": time.time(),
        "free_percent": free_percent(),
        "swap": swap_snapshot(),
        **vm_stat_snapshot(),
    }


def rate(previous: dict[str, Any], current: dict[str, Any], key: str, elapsed: float) -> float | None:
    old = previous.get(key)
    new = current.get(key)
    if not isinstance(old, int) or not isinstance(new, int):
        return None
    return max(new - old, 0) / max(elapsed, 1e-6)


def with_rates(previous: dict[str, Any], current: dict[str, Any], elapsed: float) -> dict[str, Any]:
    return {
        "pageins_per_second": rate(previous, current, "pageins", elapsed),
        "pageouts_per_second": rate(previous, current, "pageouts", elapsed),
        "swapins_per_second": rate(previous, current, "swapins", elapsed),
        "swapouts_per_second": rate(previous, current, "swapouts", elapsed),
    }


def numeric_delta(before: dict[str, Any], after: dict[str, Any], key: str) -> float | int | None:
    old = before.get(key)
    new = after.get(key)
    if isinstance(old, (int, float)) and isinstance(new, (int, float)):
        return new - old
    return None


def memory_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {
            "sample_count": 0,
            "min_free_percent": None,
            "swap_before_mb": None,
            "swap_peak_mb": None,
            "swap_after_mb": None,
            "pageout_delta": None,
            "swapout_delta": None,
            "max_pageouts_per_second": None,
            "max_swapouts_per_second": None,
            "paging_assessment": "telemetry_unavailable",
        }
    memories = [item.get("memory", {}) for item in samples]
    free_values = [
        value for value in (m.get("free_percent") for m in memories)
        if isinstance(value, (int, float))
    ]
    swap_values = [
        value for value in (m.get("swap", {}).get("used_mb") for m in memories)
        if isinstance(value, (int, float))
    ]
    pageout_rates = [
        value for value in (item.get("rates", {}).get("pageouts_per_second") for item in samples)
        if isinstance(value, (int, float))
    ]
    swapout_rates = [
        value for value in (item.get("rates", {}).get("swapouts_per_second") for item in samples)
        if isinstance(value, (int, float))
    ]
    first = memories[0]
    last = memories[-1]
    pageout_delta = numeric_delta(first, last, "pageouts")
    swapout_delta = numeric_delta(first, last, "swapouts")
    meaningful_paging_samples = 0
    severe_samples = 0
    for item in samples:
        memory = item.get("memory", {})
        free = memory.get("free_percent")
        rates = item.get("rates", {})
        pageouts = rates.get("pageouts_per_second") or 0
        swapouts = rates.get("swapouts_per_second") or 0
        if isinstance(free, (int, float)) and free <= 20 and (pageouts >= 25 or swapouts >= 500):
            meaningful_paging_samples += 1
        if isinstance(free, (int, float)) and free <= 12 and (pageouts >= 25 or swapouts >= 500):
            severe_samples += 1
    if severe_samples >= 3:
        assessment = "severe_sustained_paging"
    elif meaningful_paging_samples >= 3:
        assessment = "meaningful_sustained_paging"
    else:
        assessment = "no_meaningful_sustained_paging"
    return {
        "sample_count": len(samples),
        "min_free_percent": min(free_values) if free_values else None,
        "swap_before_mb": swap_values[0] if swap_values else None,
        "swap_peak_mb": max(swap_values) if swap_values else None,
        "swap_after_mb": swap_values[-1] if swap_values else None,
        "pageout_delta": pageout_delta,
        "swapout_delta": swapout_delta,
        "max_pageouts_per_second": max(pageout_rates) if pageout_rates else None,
        "max_swapouts_per_second": max(swapout_rates) if swapout_rates else None,
        "paging_assessment": assessment,
    }
