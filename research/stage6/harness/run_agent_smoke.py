#!/usr/bin/env python3
"""Run a serialized, coding-agent-style OpenAI-compatible backend smoke."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def request_json(base: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    url = base.rstrip("/") + path
    if payload is None:
        request = urllib.request.Request(url, method="GET")
    else:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
        )
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {path}: {detail}") from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--session-id", default="stage6-agent-smoke")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tokens", type=int, default=48)
    args = parser.parse_args()

    started = time.perf_counter()
    result: dict[str, Any] = {
        "experiment_id": "P10-G128-SERIALIZED-AGENT-27B-SMOKE",
        "phase": 10,
        "base_url": args.base_url,
        "session_id": args.session_id,
        "max_tokens": args.max_tokens,
        "serialized_expected": True,
        "health": None,
        "models": None,
        "metrics_before": None,
        "exchanges": [],
        "reset": None,
        "metrics_after": None,
        "exception": None,
    }
    transcript: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": "You are a careful coding agent. Keep answers concise and state assumptions explicitly.",
        },
        {
            "role": "user",
            "content": "Inspect the following task: add a safe read_file helper that rejects paths outside the repository. What should the first implementation step be?",
        },
    ]

    prompts = [
        "List the security invariants this helper must preserve.",
        "Write compact pseudocode for the path validation branch.",
        "Name two tests that distinguish path traversal from a valid repository path.",
        "After the tool result below, identify one bug risk and one mitigation.",
        "Draft a minimal code-review checklist for this change.",
        "Explain how a symlink could bypass a naive prefix check.",
        "Give a regression test for a symlink escape and state the expected result.",
        "Review the proposed approach for portability on macOS and Linux.",
        "State the exact failure behavior the helper should expose to its caller.",
        "Summarize the implementation plan in three ordered steps.",
    ]
    tool_injection_at = {3, 6}

    try:
        result["health"] = request_json(args.base_url, "/health")
        result["models"] = request_json(args.base_url, "/v1/models")
        result["metrics_before"] = request_json(args.base_url, "/metrics")
        for index, prompt in enumerate(prompts, start=1):
            transcript.append({"role": "user", "content": prompt})
            if index in tool_injection_at:
                transcript.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": f"call_stage6_{index}",
                                "type": "function",
                                "function": {"name": "read_file", "arguments": "{\"path\":\"src/helper.py\"}"},
                            }
                        ],
                    }
                )
                transcript.append(
                    {
                        "role": "tool",
                        "tool_call_id": f"call_stage6_{index}",
                        "content": "def read_file(path):\n    return Path(path).read_text()\n",
                    }
                )
            request_started = time.perf_counter()
            response = request_json(
                args.base_url,
                "/v1/chat/completions",
                {
                    "model": "qwen38-local",
                    "session_id": args.session_id,
                    "messages": transcript,
                    "max_tokens": args.max_tokens,
                    "mode": "greedy",
                    "stream": False,
                },
            )
            choice = response["choices"][0]
            message = choice["message"]
            assistant_text = str(message.get("content") or "")
            transcript.append({"role": "assistant", "content": assistant_text})
            result["exchanges"].append(
                {
                    "exchange": index,
                    "request_seconds": time.perf_counter() - request_started,
                    "message_count_sent": len(transcript) - 1,
                    "tool_injected": index in tool_injection_at,
                    "assistant_text": assistant_text,
                    "finish_reason": choice.get("finish_reason"),
                    "usage": response.get("usage"),
                    "metrics": response.get("x_qwen38_metrics"),
                    "session_id": response.get("session_id"),
                }
            )
        result["reset"] = request_json(
            args.base_url,
            f"/v1/sessions/{args.session_id}/reset",
            {},
        )
        result["metrics_after"] = request_json(args.base_url, "/metrics")
        result["classification"] = (
            "PASS"
            if len(result["exchanges"]) == len(prompts)
            and all(item["session_id"] == args.session_id for item in result["exchanges"])
            and result["reset"].get("reset") is True
            and result["metrics_after"].get("active_session", {}).get("message_count") == 0
            else "FAIL"
        )
    except BaseException as exc:
        result["exception"] = f"{type(exc).__name__}: {exc}"
        result["classification"] = "ERROR"
    result["total_elapsed_seconds"] = time.perf_counter() - started
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0 if result["classification"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
