"""Wait for post-run memory pressure to return to a reasonable idle state."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from memory_utils import snapshot, with_rates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--max-wait", type=float, default=120.0)
    parser.add_argument("--settle-seconds", type=float, default=10.0)
    args = parser.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text("", encoding="utf-8")
    samples = []
    previous = snapshot()
    healthy_streak = 0
    started = time.monotonic()
    with args.log.open("a", encoding="utf-8") as handle:
        while time.monotonic() - started < args.max_wait:
            current = snapshot()
            elapsed = max(current["timestamp"] - previous["timestamp"], 1e-6)
            rates = with_rates(previous, current, elapsed)
            free = current.get("free_percent")
            swapouts = rates.get("swapouts_per_second") or 0.0
            healthy = isinstance(free, (int, float)) and free >= 35.0 and swapouts < 500.0
            healthy_streak = healthy_streak + 1 if healthy else 0
            record = {"timestamp": time.time(), "memory": current, "rates": rates, "healthy": healthy}
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            samples.append(record)
            previous = current
            if healthy_streak >= max(1, int(args.settle_seconds)):
                break
            time.sleep(1.0)
    final = samples[-1] if samples else None
    summary = {
        "waited_seconds": time.monotonic() - started,
        "sample_count": len(samples),
        "healthy": healthy_streak >= max(1, int(args.settle_seconds)),
        "final": final,
    }
    args.log.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
