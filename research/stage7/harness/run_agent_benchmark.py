#!/usr/bin/env python3
"""Run deterministic coding-agent tasks through the serialized Qwen3.8 API."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import resource
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
SERVER = ROOT / "research/runtime/qwen38_agent_server.py"
FIXTURE_ROOT = ROOT / "research/stage7/agent/fixtures"
DEFAULT_MODEL = "/Users/artem/models/Qwen3.8-27B-3bit-g128-MLX"
TOOL_RE = re.compile(r"\bTOOL_CALL\s+([A-Za-z_][A-Za-z0-9_]*)\s+([^\s`]+)")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def request_json(
    base_url: str,
    path: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 900.0,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + path
    if payload is None:
        request = urllib.request.Request(url, method="GET")
    else:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
        )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {path}: {detail}") from exc


def wait_for_health(base_url: str, process: subprocess.Popen[bytes], timeout: float) -> dict[str, Any]:
    deadline = time.perf_counter() + timeout
    last_error: str | None = None
    while time.perf_counter() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"agent server exited with code {process.returncode}")
        try:
            response = request_json(base_url, "/health", timeout=5.0)
            if response.get("status") == "ready":
                return response
        except BaseException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(1.0)
    raise TimeoutError(f"agent server did not become ready: {last_error}")


def fixture(name: str) -> str:
    return (FIXTURE_ROOT / name).read_text(encoding="utf-8")


def valid_tool_call(text: str, expected_name: str, expected_path: str) -> bool:
    matches = TOOL_RE.findall(text)
    return any(name == expected_name and path == expected_path for name, path in matches)


def extract_json_object(text: str) -> dict[str, Any] | None:
    candidates = [text.strip()]
    candidates.extend(re.findall(r"\{.*?\}", text, flags=re.DOTALL))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def task_specs() -> list[dict[str, Any]]:
    system = (
        "You are a careful coding agent operating on a read-only repository snapshot. "
        "Do not claim file contents until a tool result is supplied. The only tool is "
        "read_file. Request it exactly as TOOL_CALL read_file <path>, with no markdown "
        "around the call. After the tool result, reason from the supplied text and "
        "propose a minimal, testable fix. Keep each response under 120 tokens, "
        "avoid repeating earlier explanations, and include only the essential patch "
        "or plan when asked."
    )
    return [
        {
            "task_id": "A-mutable-default",
            "system": system,
            "initial_user": (
                "Inspect this small Python repository. First request buggy_cache.py "
                "using the tool protocol, then diagnose the failing instance-isolation test."
            ),
            "tools": [
                ("read_file", "buggy_cache.py", fixture("buggy_cache.py")),
                ("read_file", "test_buggy_cache.py", fixture("test_buggy_cache.py")),
            ],
            "followups": [
                (
                    "The tool returned the file. The test in test_buggy_cache.py fails "
                    "because two instances share state. Identify the root cause and "
                    "give the minimal patch, including the exact default-argument behavior."
                )
            ],
            "success_terms": [
                ["mutable", "default"],
                ["none", "list"],
                ["instance", "share"],
            ],
        },
        {
            "task_id": "B-idempotent-recovery",
            "system": system,
            "initial_user": (
                "A worker delivery retry caused a duplicate mutation. First request "
                "retry.py using the tool protocol; do not guess its contents."
            ),
            "tools": [("read_file", "retry.py", fixture("retry.py"))],
            "followups": [
                (
                    "Initial hypothesis: simply retry store[event['job_id']] = state one "
                    "more time. That hypothesis is wrong. The test output is:\n"
                    + fixture("test_retry_failure.txt")
                    + "\nRevise the hypothesis and propose the corrected idempotent algorithm. "
                    "Explain what the second delivery must return and when the duplicate "
                    "check occurs."
                )
            ],
            "success_terms": [
                ["duplicate", "id"],
                ["before", "apply"],
                ["original", "return"],
            ],
        },
        {
            "task_id": "C-monotonic-events",
            "system": system,
            "initial_user": (
                "Diagnose a job event ordering bug. First request state.py using the "
                "tool protocol. We will inspect a second file afterward."
            ),
            "tools": [
                ("read_file", "state.py", fixture("state.py")),
                ("read_file", "events.py", fixture("events.py")),
            ],
            "followups": [
                "The state.py tool result is above. Now request events.py with the tool protocol so you can check the publisher boundary.",
                (
                    "The second tool result is above. The test output is:\n"
                    + fixture("test_state_failure.txt")
                    + "\nGive a concise multi-file patch plan. Preserve event sequence ordering, "
                    "prevent a terminal job from becoming active again, and explain the "
                    "late-event guard."
                ),
            ],
            "success_terms": [
                ["sequence", "order"],
                ["terminal", "running"],
                ["reject", "late"],
            ],
        },
        {
            "task_id": "D-structured-result",
            "system": system,
            "initial_user": (
                "Inspect the response formatter. First request formatter.py using the "
                "tool protocol, then wait for the test result."
            ),
            "tools": [
                ("read_file", "formatter.py", fixture("formatter.py")),
                (
                    "read_file",
                    "tests/test_formatter.py",
                    fixture("test_formatter_failure.txt"),
                ),
            ],
            "followups": [
                (
                    "The test output is:\n"
                    + fixture("test_formatter_failure.txt")
                    + "\nReturn JSON only with exactly these keys: action, path, reason. "
                    "Set path to formatter.py and describe the smallest fix."
                )
            ],
            "structured_only": True,
            "success_terms": [["formatter.py", "ok"]],
        },
    ]


def assess_task(spec: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    assistant_text = "\n".join(
        event["assistant_text"] for event in events if event.get("assistant_text")
    )
    lower = assistant_text.lower()
    checks = [all(term.lower() in lower for term in group) for group in spec["success_terms"]]
    tool_events = [event for event in events if event.get("tool_expected")]
    valid_tools = sum(1 for event in tool_events if event.get("tool_valid"))
    invalid_tools = sum(1 for event in tool_events if event.get("tool_expected") and not event.get("tool_valid"))
    structured = False
    structured_value: dict[str, Any] | None = None
    if spec.get("structured_only"):
        final_text = next(
            (event["assistant_text"] for event in reversed(events) if event.get("assistant_text")),
            "",
        )
        structured_value = extract_json_object(final_text)
        structured = (
            structured_value is not None
            and set(structured_value) == {"action", "path", "reason"}
            and "formatter.py" in str(structured_value.get("path", ""))
            and bool(str(structured_value.get("action", "")).strip())
            and bool(str(structured_value.get("reason", "")).strip())
        )
    task_success = (all(checks) and (structured if spec.get("structured_only") else True))
    partial_score = sum(checks) / len(checks) if checks else 0.0
    if spec.get("structured_only") and structured:
        partial_score = max(partial_score, 1.0)
    ttft = [
        event["metrics"].get("ttft_seconds")
        for event in events
        if event.get("metrics") and isinstance(event["metrics"].get("ttft_seconds"), (int, float))
    ]
    warm_ttft = ttft[1:] if len(ttft) > 1 else []
    decode = [
        event["metrics"].get("decode_tps")
        for event in events
        if event.get("metrics") and isinstance(event["metrics"].get("decode_tps"), (int, float))
    ]
    contexts = [
        event["metrics"].get("logical_context_tokens")
        for event in events
        if event.get("metrics") and isinstance(event["metrics"].get("logical_context_tokens"), int)
    ]
    return {
        "task_id": spec["task_id"],
        "task_success": task_success,
        "partial_score": partial_score,
        "checks": checks,
        "structured_valid": structured,
        "structured_value": structured_value,
        "assistant_turns": len(events),
        "valid_tool_calls": valid_tools,
        "invalid_tool_calls": invalid_tools,
        "warm_ttft_seconds": warm_ttft,
        "median_warm_ttft_seconds": statistics.median(warm_ttft) if warm_ttft else None,
        "max_warm_ttft_seconds": max(warm_ttft) if warm_ttft else None,
        "decode_tps": decode,
        "median_decode_tps": statistics.median(decode) if decode else None,
        "final_logical_context_tokens": max(contexts) if contexts else None,
        "assistant_text": assistant_text,
    }


def run_task(base_url: str, spec: dict[str, Any], session_id: str, max_tokens: int) -> dict[str, Any]:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": spec["system"]},
        {"role": "user", "content": spec["initial_user"]},
    ]
    events: list[dict[str, Any]] = []
    expected_tools = list(spec["tools"])
    allowed_tools = {(name, path): content for name, path, content in expected_tools}
    # Task C deliberately inserts its second file request after an explicit
    # user instruction; A and D permit the model to request a test file
    # naturally after inspecting the implementation file.
    auto_tools = allowed_tools if not spec["task_id"].startswith("C-") else {
        (expected_tools[0][0], expected_tools[0][1]): expected_tools[0][2]
    }

    def complete(kind: str, tool_expected: tuple[str, str] | None = None) -> str:
        nonlocal messages
        started = time.perf_counter()
        response = request_json(
            base_url,
            "/v1/chat/completions",
            {
                "model": "qwen38-stage7-g128",
                "session_id": session_id,
                "messages": messages,
                "max_tokens": max_tokens,
                "mode": "greedy",
                "stream": False,
            },
        )
        elapsed = time.perf_counter() - started
        message = response["choices"][0]["message"]
        text = str(message.get("content") or "")
        tool_valid = None
        detected = TOOL_RE.findall(text)
        if tool_expected is not None:
            tool_valid = valid_tool_call(text, *tool_expected)
        elif detected:
            tool_valid = any((name, path) in auto_tools for name, path in detected)
        event = {
            "kind": kind,
            "messages_sent": len(messages),
            "request_seconds": elapsed,
            "assistant_text": text,
            "finish_reason": response["choices"][0].get("finish_reason"),
            "usage": response.get("usage"),
            "metrics": response.get("x_qwen38_metrics"),
            "session_id": response.get("session_id"),
            "tool_expected": tool_expected is not None or bool(detected),
            "tool_valid": tool_valid,
            "detected_tool_calls": [
                {"name": name, "path": path} for name, path in detected
            ],
        }
        events.append(event)
        messages.append({"role": "assistant", "content": text})
        return text

    def inject_tool(name: str, path: str, content: str, depth: int = 0) -> None:
        nonlocal messages
        messages.append({
            "role": "tool",
            "content": f"read_file {path} returned:\n{content}",
        })
        # A model may request a related test file after seeing the first tool
        # result. Follow that request only when it names a frozen fixture;
        # unknown requests remain visible as invalid tool-use evidence.
        response_text = complete("tool_result")
        if depth >= 2:
            return
        for requested_name, requested_path in TOOL_RE.findall(response_text)[:1]:
            requested_content = auto_tools.get((requested_name, requested_path))
            if requested_content is not None:
                inject_tool(
                    requested_name,
                    requested_path,
                    requested_content,
                    depth + 1,
                )

    def maybe_inject_requested_tool(response_text: str) -> None:
        for requested_name, requested_path in TOOL_RE.findall(response_text)[:1]:
            requested_content = auto_tools.get((requested_name, requested_path))
            if requested_content is not None:
                inject_tool(requested_name, requested_path, requested_content)
                break

    tool_index = 0
    first_tool = expected_tools[tool_index]
    complete("assistant", (first_tool[0], first_tool[1]))
    inject_tool(*first_tool)
    tool_index += 1

    for followup in spec["followups"]:
        messages.append({"role": "user", "content": followup})
        if tool_index < len(expected_tools) and spec["task_id"].startswith("C-") and tool_index == 1:
            second_tool = expected_tools[tool_index]
            complete("assistant", (second_tool[0], second_tool[1]))
            inject_tool(*second_tool)
            tool_index += 1
        else:
            response_text = complete("assistant")
            maybe_inject_requested_tool(response_text)

    metrics = request_json(base_url, "/metrics")
    assessment = assess_task(spec, events)
    assessment.update(
        {
            "session_id": session_id,
            "events": events,
            "transcript": messages,
            "server_metrics_before_reset": metrics,
        }
    )
    try:
        assessment["reset"] = request_json(
            base_url, f"/v1/sessions/{session_id}/reset", {}
        )
    except BaseException as exc:
        assessment["reset_error"] = f"{type(exc).__name__}: {exc}"
    return assessment


def run(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    output = Path(args.output).resolve()
    server_log = output.with_suffix(".server.log")
    result: dict[str, Any] = {
        "experiment_id": "P11-G128-SERIALIZED-CODING-AGENT",
        "phase": 11,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": None,
        "model_name": "g128",
        "model_path": str(Path(args.model).resolve()),
        "server_path": str(SERVER),
        "base_url": args.base_url,
        "max_tokens": args.max_tokens,
        "fixture_root": str(FIXTURE_ROOT),
        "fixture_hashes": {
            path.name: sha256_file(path)
            for path in sorted(FIXTURE_ROOT.iterdir())
            if path.is_file()
        },
        "health": None,
        "models": None,
        "tasks": [],
        "exception": None,
        "safety_stop_reason": None,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    process: subprocess.Popen[bytes] | None = None
    log_handle = None
    try:
        try:
            result["git_commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip()
        except BaseException:
            result["git_commit"] = None
        log_handle = server_log.open("wb")
        process = subprocess.Popen(
            [
                sys.executable,
                str(SERVER),
                "--model",
                args.model,
                "--host",
                args.host,
                "--port",
                str(args.port),
                "--model-name",
                "qwen38-stage7-g128",
                "--mode",
                "greedy",
                "--max-tokens",
                str(args.max_tokens),
            ],
            cwd=ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        result["health"] = wait_for_health(args.base_url, process, args.server_timeout)
        result["models"] = request_json(args.base_url, "/v1/models")
        for index, spec in enumerate(task_specs(), start=1):
            session_id = f"stage7-agent-{index}-{spec['task_id']}"
            task_started = time.perf_counter()
            task = run_task(args.base_url, spec, session_id, args.max_tokens)
            task["elapsed_seconds"] = time.perf_counter() - task_started
            result["tasks"].append(task)
            print(
                json.dumps(
                    {
                        "task_id": task["task_id"],
                        "success": task["task_success"],
                        "partial_score": task["partial_score"],
                        "valid_tool_calls": task["valid_tool_calls"],
                        "invalid_tool_calls": task["invalid_tool_calls"],
                        "turns": task["assistant_turns"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        all_events = [event for task in result["tasks"] for event in task["events"]]
        warm_ttft = [
            value
            for task in result["tasks"]
            for value in task["warm_ttft_seconds"]
            if isinstance(value, (int, float))
        ]
        decode = [
            value
            for task in result["tasks"]
            for value in task["decode_tps"]
            if isinstance(value, (int, float))
        ]
        result["summary"] = {
            "task_count": len(result["tasks"]),
            "tasks_passed": sum(bool(task["task_success"]) for task in result["tasks"]),
            "partial_score_mean": statistics.mean(
                task["partial_score"] for task in result["tasks"]
            ),
            "total_assistant_turns": len(all_events),
            "valid_tool_calls": sum(task["valid_tool_calls"] for task in result["tasks"]),
            "invalid_tool_calls": sum(task["invalid_tool_calls"] for task in result["tasks"]),
            "protocol_request_errors": 0,
            "median_warm_ttft_seconds": statistics.median(warm_ttft) if warm_ttft else None,
            "max_warm_ttft_seconds": max(warm_ttft) if warm_ttft else None,
            "median_decode_tps": statistics.median(decode) if decode else None,
            "reset_passes": sum(
                task.get("reset", {}).get("reset") is True for task in result["tasks"]
            ),
            "serialized": True,
        }
        result["classification"] = (
            "PASS_AGENT_TASKS_WITH_PARTIAL_MODEL_TOOL_USE"
            if result["summary"]["tasks_passed"] == len(result["tasks"])
            else "COMPLETED_AGENT_TASKS_WITH_MODEL_QUALITY_FAILURES"
        )
    except BaseException as exc:
        result["exception"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = __import__("traceback").format_exc()
        result["classification"] = "AGENT_BENCHMARK_ERROR"
    finally:
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        if log_handle is not None:
            log_handle.close()
        result["server_log"] = str(server_log)
        result["total_elapsed_seconds"] = time.perf_counter() - started
        result["rss_max_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write_json(output, result)
        print(
            json.dumps(
                {
                    "classification": result.get("classification"),
                    "summary": result.get("summary"),
                    "exception": result.get("exception"),
                    "total_elapsed_seconds": result["total_elapsed_seconds"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return 0 if result.get("classification", "").startswith("PASS_") else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--server-timeout", type=float, default=900.0)
    parser.add_argument("--output", required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
