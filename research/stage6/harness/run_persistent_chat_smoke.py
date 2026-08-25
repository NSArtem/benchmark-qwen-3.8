#!/usr/bin/env python3
"""Run a bounded exact-suffix chat smoke against one model.

The first user turn is a flattened Stage 5 long-context prompt. Four short
follow-ups then exercise canonical transcript reconstruction, exact suffix
accounting, cache growth, and reset-free multi-turn behavior. This harness
uses the checked-in production chat implementation and writes only measured
runtime records.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.runtime.qwen38_chat import ExactSuffixChat, _memory_snapshot, _require_mlx


def os_snapshot() -> dict[str, str]:
    result: dict[str, str] = {}
    for name, command in (
        ("memory_pressure", ["memory_pressure", "-Q"]),
        ("vm_swapusage", ["sysctl", "vm.swapusage"]),
    ):
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=10)
            result[name] = (completed.stdout + completed.stderr).strip()
        except BaseException as exc:
            result[name] = f"{type(exc).__name__}: {exc}"
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--followup-count", type=int, default=4)
    parser.add_argument("--preserve-newlines", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    if not args.preserve_newlines:
        prompt = prompt.replace("\n", " ")
    prompt = prompt.strip()
    followups = [
        "State the single most important constraint from the previous task in one sentence.",
        "Give a two-item checklist for the next safe engineering action.",
        "Which context scale did the supplied task use? Answer briefly.",
        "Finish with one sentence naming the safest next action.",
    ]
    if args.followup_count < 0 or args.followup_count > len(followups):
        raise ValueError(f"followup-count must be between 0 and {len(followups)}")
    queries = [prompt, *followups[: args.followup_count]]
    result: dict[str, object] = {
        "experiment_id": f"P2-{args.label.upper()}-PERSISTENT-CHAT-{len(queries)}TURN",
        "phase": 2,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model_label": args.label,
        "model_path": str(Path(args.model).resolve()),
        "prompt_file": str(Path(args.prompt_file)),
        "prompt_sha256": __import__("hashlib").sha256(prompt.encode()).hexdigest(),
        "max_tokens": args.max_tokens,
        "followup_count": args.followup_count,
        "preserve_newlines": args.preserve_newlines,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "os_memory_before": os_snapshot(),
        "memory_before": None,
        "turns": [],
        "exception": None,
        "safety_stop_reason": None,
    }
    try:
        mx, load, _, _, _ = _require_mlx()
        model, tokenizer = load(args.model)
        result["memory_before"] = _memory_snapshot()
        session = ExactSuffixChat(model, tokenizer)
        for index, query in enumerate(queries, start=1):
            turn_started = time.perf_counter()
            turn = session.turn(
                query,
                max_tokens=args.max_tokens,
                temperature=0.0,
                top_p=1.0,
                prefill_step_size=2048,
            )
            record = asdict(turn)
            record["turn"] = index
            record["query_tokens_utf8"] = len(query.encode("utf-8"))
            record["elapsed_wall_seconds"] = time.perf_counter() - turn_started
            result["turns"].append(record)  # type: ignore[union-attr]
        result["stats_before_reset"] = session.stats()
        result["memory_after"] = _memory_snapshot()
        session.reset()
        result["stats_after_reset"] = session.stats()
        result["rss_max_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        result["os_memory_after"] = os_snapshot()
    except BaseException as exc:
        result["exception"] = f"{type(exc).__name__}: {exc}"
        result["os_memory_after"] = os_snapshot()
    result["total_elapsed_seconds"] = time.perf_counter() - started
    result["turn_count"] = len(result["turns"])  # type: ignore[arg-type]
    result["all_exact_suffix_checks_passed"] = bool(
        result["exception"] is None
        and len(result["turns"]) == len(queries)  # type: ignore[arg-type]
        and all(
            int(turn["cache_offset_after"]) == int(turn["logical_context_tokens"])
            and int(turn["processed_suffix_tokens"]) >= 0
            for turn in result["turns"]  # type: ignore[union-attr]
        )
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 1 if result["exception"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
