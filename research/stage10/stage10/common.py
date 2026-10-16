"""Host preflight and watchdog-managed subprocess helpers."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
WATCHDOG = REPO_ROOT / "tools" / "memory_watchdog.sh"


class HostNotReady(RuntimeError):
    """Raised when free memory or swap use is outside the task thresholds."""


@dataclass(frozen=True)
class HostState:
    free_pct: float
    swap_used_mib: float
    memory_pressure: str
    swapusage: str


def _run_readonly(command: Sequence[str]) -> str:
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def _swap_mib(output: str) -> float:
    match = re.search(r"\bused\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*([KMG])", output, re.I)
    if match is None:
        raise ValueError(f"could not parse used swap from sysctl output: {output!r}")
    value = float(match.group(1))
    unit = match.group(2).upper()
    return value / 1024.0 if unit == "K" else value * 1024.0 if unit == "G" else value


def host_check() -> HostState:
    """Return memory-pressure free percentage and swap use in MiB."""
    pressure = _run_readonly(["memory_pressure", "-Q"])
    swap = _run_readonly(["sysctl", "vm.swapusage"])
    match = re.search(r"free percentage\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%", pressure, re.I)
    if match is None:
        raise ValueError(f"could not parse free percentage from memory_pressure output: {pressure!r}")
    return HostState(float(match.group(1)), _swap_mib(swap), pressure, swap)


def preflight() -> HostState:
    """Require at least 40% free memory and at most 4096 MiB swap used."""
    state = host_check()
    if state.free_pct < 40 or state.swap_used_mib > 4096:
        raise HostNotReady(
            f"host preflight failed: free_pct={state.free_pct:g}, "
            f"swap_used_mib={state.swap_used_mib:g}; required free_pct>=40 "
            "and swap_used_mib<=4096"
        )
    return state


def wait_for_preflight(*, interval_s: int = 900, max_wait_s: int = 3600) -> HostState:
    """Retry a blocked preflight every 15 minutes for up to one hour."""
    started = time.monotonic()
    while True:
        try:
            return preflight()
        except HostNotReady:
            elapsed = time.monotonic() - started
            remaining = max_wait_s - elapsed
            if remaining <= 0:
                raise
            time.sleep(min(interval_s, remaining))


def run_guarded(cmd: Sequence[str], log_path: str | Path) -> tuple[int, bool]:
    """Run one command with the repository memory watchdog attached.

    The log contains the command, combined child output, and watchdog samples.
    Returns ``(exit_code, watchdog_fired)``. Commands must be argument arrays;
    shell parsing is intentionally not used.
    """
    if isinstance(cmd, (str, bytes)) or not cmd:
        raise TypeError("cmd must be a non-empty sequence of arguments")
    if not WATCHDOG.is_file():
        raise FileNotFoundError(f"memory watchdog not found: {WATCHDOG}")
    log = Path(log_path).expanduser().resolve()
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        stream.write(f"command: {shlex.join([os.fspath(part) for part in cmd])}\n")
        stream.flush()

    with log.open("ab") as output:
        process = subprocess.Popen(
            [os.fspath(part) for part in cmd],
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        watchdog = subprocess.Popen(
            ["/bin/zsh", str(WATCHDOG), str(process.pid), str(log)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        started = time.monotonic()
        exit_code = process.wait()
        watchdog.wait()
        elapsed = time.monotonic() - started

    log.write_text(
        log.read_text(encoding="utf-8") + f"\nelapsed_s: {elapsed:.3f}\n",
        encoding="utf-8",
    )
    watchdog_fired = "SAFETY STOP" in log.read_text(encoding="utf-8", errors="replace")
    return exit_code, watchdog_fired


def watchdog_min_free(log_path: str | Path) -> float | None:
    """Lowest ``free=N%`` sample the watchdog wrote to ``log_path``."""
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    values = [float(value) for value in re.findall(r"\bfree=([0-9]+(?:\.[0-9]+)?)%", text)]
    return min(values) if values else None


def guarded_module(module: str, args: Sequence[str], log_path: str | Path) -> dict:
    """Wait for preflight, run ``python -m module args...`` under the watchdog."""
    import sys

    before = wait_for_preflight()
    started = time.monotonic()
    exit_code, fired = run_guarded([sys.executable, "-m", module, *args], log_path)
    wall = time.monotonic() - started
    after = host_check()
    return {
        "exit_code": exit_code,
        "safety_stop": fired,
        "wall_s": wall,
        "free_pct_before": before.free_pct,
        "swap_mib_before": before.swap_used_mib,
        "free_pct_after": after.free_pct,
        "swap_mib_after": after.swap_used_mib,
        "free_pct_min": watchdog_min_free(log_path),
        "log_path": str(log_path),
    }
