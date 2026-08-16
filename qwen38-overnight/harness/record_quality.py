"""Add a quality-smoke result to results.json and regenerate results.md."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--supervisor", type=Path, required=True)
    args = parser.parse_args()
    registry = read_json(args.registry)
    if not isinstance(registry, dict):
        registry = {"schema_version": 1, "runs": [], "quality": []}
    registry.setdefault("quality", [])
    result = read_json(args.result)
    supervisor = read_json(args.supervisor)
    if not isinstance(result, dict):
        result = {"status": "missing_result", "error": str(args.result)}
    if not isinstance(supervisor, dict):
        supervisor = {"watchdog_abort": True, "watchdog_reason": "missing_supervisor_result"}
    result["supervisor"] = supervisor
    result["recorded_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    result["candidate"] = args.candidate
    registry["quality"] = [item for item in registry["quality"] if item.get("candidate") != args.candidate]
    registry["quality"].append(result)
    registry["quality"].sort(key=lambda item: item.get("candidate", ""))
    args.registry.write_text(json.dumps(registry, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"candidate": args.candidate, "status": result.get("status"), "passed": result.get("passed")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
