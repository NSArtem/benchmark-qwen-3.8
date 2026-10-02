#!/usr/bin/env python3
"""Long serialized coding-agent integration benchmark for ``qwen38.server``.

Starts the real server with the g128 model, then drives one ~10K-token
session through file inspection, a long tool result, a wrong hypothesis,
failing-test revisions, and three forced ``finish_reason=length`` turns that
must continue by exact cache reuse.

The harness uses only deterministic local fixture contents. It never supplies
an expected tool result after an invalid or unknown tool request; such a turn
is retained as a protocol-quality failure in the raw transcript.

Run from the repository root (loads the 27B model; do not run alongside
another heavy MLX process)::

    uv run python benchmarks/long_agent.py --output /tmp/long-agent.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "benchmarks/fixtures"
MODEL_NAME = "qwen38-g128"
DEFAULT_MODEL = "/Users/artem/models/Qwen3.8-27B-3bit-g128-MLX"
TOOL_RE = __import__("re").compile(
    r"\bTOOL_CALL\s+([A-Za-z_][A-Za-z0-9_]*)\s+([^\s`]+)"
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def fixture(name: str) -> str:
    return (FIXTURE_ROOT / name).read_text(encoding="utf-8")


def long_fixture_payload() -> str:
    """Build a deterministic long tool result from local repository fixtures."""

    readme = fixture("README.md")
    state = fixture("state.py")
    events = fixture("events.py")
    blocks: list[str] = []
    for index in range(24):
        blocks.append(
            f"=== dependency-audit-block-{index:02d} ===\n"
            f"source: README.md, state.py, events.py\n"
            f"{readme}\n"
            f"--- state.py snapshot ---\n{state}\n"
            f"--- events.py snapshot ---\n{events}\n"
        )
    return "\n".join(blocks)


def fixture_map() -> dict[str, str]:
    long_payload = long_fixture_payload()
    return {
        "buggy_cache.py": fixture("buggy_cache.py"),
        "test_buggy_cache.py": fixture("test_buggy_cache.py"),
        "tests/test_buggy_cache.py": fixture("test_buggy_cache.py"),
        "retry.py": fixture("retry.py"),
        "test_retry_failure.txt": fixture("test_retry_failure.txt"),
        "tests/test_retry.py": fixture("test_retry_failure.txt"),
        "state.py": fixture("state.py"),
        "events.py": fixture("events.py"),
        "test_state_failure.txt": fixture("test_state_failure.txt"),
        "tests/test_state.py": fixture("test_state_failure.txt"),
        "tests/test_events.py": fixture("events.py"),
        "formatter.py": fixture("formatter.py"),
        "tests/test_formatter.py": fixture("test_formatter_failure.txt"),
        "long_dependency_audit.txt": long_payload,
        "lantern_dependency_audit.txt": long_payload,
    }


def request_json(
    base_url: str,
    path: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 1200.0,
) -> tuple[int, dict[str, Any] | None, str | None]:
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
            return response.status, json.loads(response.read().decode("utf-8")), None
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        try:
            payload_value = json.loads(detail)
        except json.JSONDecodeError:
            payload_value = None
        return exc.code, payload_value, detail
    except BaseException as exc:
        return 0, None, f"{type(exc).__name__}: {exc}"


def wait_for_health(base_url: str, process: subprocess.Popen[bytes], timeout: float) -> dict[str, Any]:
    deadline = time.perf_counter() + timeout
    last_error: str | None = None
    while time.perf_counter() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"agent server exited with code {process.returncode}")
        status, payload, error = request_json(base_url, "/health", timeout=5.0)
        if status == 200 and payload and payload.get("status") == "ready":
            return payload
        last_error = error or f"HTTP {status}"
        time.sleep(1.0)
    raise TimeoutError(f"agent server did not become ready: {last_error}")


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def system_prompt() -> str:
    return (
        "You are a careful coding agent in the fictional Lantern monorepo. "
        "The repository snapshot is read-only. The only tool is read_file. "
        "When a file is needed, emit exactly one plain-text request of the form "
        "TOOL_CALL read_file path, with no invented contents and no markdown "
        "fence around the request. After a tool result, reason from it and do "
        "not request another file unless the user explicitly asks for one. Wait for the tool result before reasoning "
        "about that file. Preserve these early constraints throughout the "
        "session: use the smallest testable patch, preserve event ordering, "
        "protect idempotency, keep terminal state terminal, and do not claim "
        "an inspection that did not happen. This is a serialized session; "
        "if an earlier assistant response was capped by max_tokens, continue "
        "from its exact stopping point rather than restarting or repeating it."
    )


class SessionRunner:
    def __init__(self, base_url: str, session_id: str, max_tokens: int) -> None:
        self.base_url = base_url
        self.session_id = session_id
        self.default_max_tokens = max_tokens
        self.fixtures = fixture_map()
        self.messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt()}
        ]
        self.events: list[dict[str, Any]] = []
        self.invalid_tool_calls = 0
        self.missing_tool_requests = 0
        self.expected_tool_requests = 0
        self.injected_tool_results = 0
        self.http_errors = 0
        self.unknown_tool_paths: list[str] = []
        self.invalid_tool_details: list[dict[str, Any]] = []
        self.truncation_recoveries: list[dict[str, Any]] = []

    def _tool_matches(self, text: str) -> list[tuple[str, str]]:
        return [(name, path) for name, path in TOOL_RE.findall(text)]

    def _valid_matches(self, matches: list[tuple[str, str]]) -> bool:
        return bool(matches) and all(name == "read_file" and path in self.fixtures for name, path in matches)

    def complete(
        self,
        label: str,
        *,
        user_text: str | None = None,
        max_tokens: int | None = None,
        expected_tool_path: str | None = None,
        forced_truncation: bool = False,
        recovery_after_truncation: bool = False,
    ) -> tuple[str | None, dict[str, Any]]:
        if user_text is not None:
            self.messages.append({"role": "user", "content": user_text})
        budget = int(max_tokens or self.default_max_tokens)
        started = time.perf_counter()
        status, payload, error = request_json(
            self.base_url,
            "/v1/chat/completions",
            {
                "model": MODEL_NAME,
                "session_id": self.session_id,
                "messages": self.messages,
                "max_tokens": budget,
                "mode": "greedy",
                "stream": False,
            },
        )
        elapsed = time.perf_counter() - started
        event: dict[str, Any] = {
            "turn_index": len(self.events) + 1,
            "label": label,
            "messages_sent": len(self.messages),
            "request_seconds": elapsed,
            "request_status": status,
            "request_error": error,
            "requested_max_tokens": budget,
            "forced_truncation_expected": forced_truncation,
            "recovery_after_truncation": recovery_after_truncation,
            "assistant_text": "",
            "finish_reason": None,
            "usage": None,
            "metrics": None,
            "tool_matches": [],
            "tool_expected": expected_tool_path is not None,
            "expected_tool_path": expected_tool_path,
            "tool_valid": None,
            "tool_results_injected": [],
        }
        self.events.append(event)
        if status != 200 or not payload:
            self.http_errors += 1
            event["tool_valid"] = False if expected_tool_path else None
            return None, event

        choice = (payload.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        text = str(message.get("content") or "")
        finish_reason = choice.get("finish_reason")
        matches = self._tool_matches(text)
        known = self._valid_matches(matches)
        expected_satisfied = (
            expected_tool_path is None
            or expected_tool_path in [path for _name, path in matches]
        )
        event.update(
            {
                "assistant_text": text,
                "finish_reason": finish_reason,
                "usage": payload.get("usage"),
                "metrics": payload.get("x_qwen38_metrics"),
                "tool_matches": [
                    {"name": name, "path": path} for name, path in matches
                ],
                "tool_valid": (known and expected_satisfied) if matches else (False if expected_tool_path else None),
            }
        )
        if expected_tool_path is not None:
            self.expected_tool_requests += 1
            if not expected_satisfied:
                self.missing_tool_requests += 1
        if matches and not known:
            self.invalid_tool_calls += sum(
                not (name == "read_file" and path in self.fixtures)
                for name, path in matches
            )
            for name, path in matches:
                if name != "read_file" or path not in self.fixtures:
                    self.unknown_tool_paths.append(path)
                    detail = {"label": label, "name": name, "path": path}
                    self.invalid_tool_details.append(detail)
        self.messages.append({"role": "assistant", "content": text})
        return text, event

    def service_tools(self, text: str | None, *, max_depth: int = 1) -> None:
        """Supply results only when every requested call is known and valid."""

        current = text
        for depth in range(max_depth):
            if not current:
                return
            matches = self._tool_matches(current)
            if not matches:
                return
            if not self._valid_matches(matches):
                # Deliberately do not append any tool result for this response.
                return
            for name, path in matches:
                content = self.fixtures[path]
                self.messages.append(
                    {
                        "role": "tool",
                        "content": f"read_file {path} returned:\n{content}",
                    }
                )
                self.injected_tool_results += 1
            current, event = self.complete(
                f"tool_result_{depth + 1}",
                max_tokens=self.default_max_tokens,
            )
            event["tool_results_injected"] = [path for _name, path in matches]

    def expected_turn(
        self,
        label: str,
        prompt: str,
        path: str,
        *,
        max_tokens: int | None = None,
    ) -> str | None:
        exact_prompt = (
            f"Output exactly one line and nothing else: TOOL_CALL read_file {path}. "
            "After the tool result, do not request another file unless explicitly asked."
        )
        text, event = self.complete(
            label,
            user_text=exact_prompt,
            max_tokens=max_tokens,
            expected_tool_path=path,
        )
        # A model may emit an incomplete ``TOOL`` fragment at a long context
        # boundary. Give it one explicit protocol retry, without supplying a
        # result for the incomplete response.
        matches = self._tool_matches(text or "")
        if (
            event.get("request_status") == 200
            and path not in [item_path for _name, item_path in matches]
        ):
            text, _event = self.complete(
                f"{label}_protocol_retry",
                user_text=(
                    f"Output exactly one line and nothing else: TOOL_CALL read_file {path}. "
                    "Do not answer or invent file contents."
                ),
                max_tokens=16,
                expected_tool_path=path,
            )
        self.service_tools(text)
        return text

    def normal_turn(
        self,
        label: str,
        prompt: str,
        *,
        max_tokens: int | None = None,
    ) -> str | None:
        text, _event = self.complete(label, user_text=prompt, max_tokens=max_tokens)
        self.service_tools(text)
        return text

    def force_truncation(self, number: int, context: str) -> dict[str, Any]:
        prompt = (
            f"This is deliberate truncation checkpoint {number}. {context} "
            f"Output a single unclosed implementation ledger beginning exactly "
            f"with TRUNCATION-{number} ITEM-000. Generate hundreds of short "
            "numbered items, retain the early Lantern constraints, and do not "
            "provide a final item, conclusion, summary, or tool request; the "
            "response must continue past the max_tokens limit."
        )
        first_event: dict[str, Any] | None = None
        attempts = 0
        text: str | None = None
        while attempts < 2:
            attempts += 1
            text, first_event = self.complete(
                f"forced_truncation_{number}_attempt_{attempts}",
                user_text=prompt,
                max_tokens=8 if attempts == 1 else 1,
                forced_truncation=True,
            )
            if first_event.get("finish_reason") == "length":
                break
            prompt = (
                f"Checkpoint {number} retry: emit a continuous stream of numbered "
                f"ledger items beginning TRUNCATION-{number} ITEM-000. Do not "
                "stop voluntarily, close the ledger, summarize, or request a tool."
            )
        forced = bool(first_event and first_event.get("finish_reason") == "length")
        recovery_text: str | None = None
        recovery_event: dict[str, Any] | None = None
        if first_event and first_event.get("request_status") == 200:
            recovery_text, recovery_event = self.complete(
                f"truncation_{number}_recovery",
                user_text=(
                    f"The previous assistant response was intentionally stopped by "
                    f"max_tokens at checkpoint {number}. Continue the same numbered "
                    "checklist from the next item without restarting, repeating, or "
                    "summarizing. Preserve every earlier constraint and do not use a tool."
                ),
                max_tokens=96,
                recovery_after_truncation=forced,
            )
            self.service_tools(recovery_text)
        recovery_ok = bool(
            forced
            and recovery_event
            and recovery_event.get("request_status") == 200
            and recovery_event.get("metrics", {}).get("cache_reuse_mode")
            in {"exact_reuse", "fresh_fallback"}
            and recovery_event.get("request_status") != 409
        )
        record = {
            "checkpoint": number,
            "attempts": attempts,
            "forced": forced,
            "forced_turn_label": first_event.get("label") if first_event else None,
            "forced_finish_reason": first_event.get("finish_reason") if first_event else None,
            "recovery_turn_label": recovery_event.get("label") if recovery_event else None,
            "recovery_finish_reason": recovery_event.get("finish_reason") if recovery_event else None,
            "recovery_ok": recovery_ok,
            "recovery_cache_reuse_mode": (
                recovery_event.get("metrics", {}).get("cache_reuse_mode")
                if recovery_event
                else None
            ),
            "forced_text": text,
            "recovery_text": recovery_text,
        }
        self.truncation_recoveries.append(record)
        return record

    def summary(self) -> dict[str, Any]:
        successful = [
            item for item in self.truncation_recoveries if item.get("recovery_ok")
        ]
        metrics = [
            event.get("metrics")
            for event in self.events
            if event.get("request_status") == 200 and event.get("metrics")
        ]
        ttft = [
            float(item["ttft_seconds"])
            for item in metrics
            if isinstance(item.get("ttft_seconds"), (int, float))
        ]
        warm_ttft = ttft[1:] if len(ttft) > 1 else []
        decode = [
            float(item["decode_tps"])
            for item in metrics
            if isinstance(item.get("decode_tps"), (int, float))
        ]
        contexts = [
            int(item["logical_context_tokens"])
            for item in metrics
            if isinstance(item.get("logical_context_tokens"), int)
        ]
        cache_reuse = sum(item.get("cache_reuse_mode") == "exact_reuse" for item in metrics)
        fresh_fallback = sum(item.get("cache_reuse_mode") == "fresh_fallback" for item in metrics)
        final_text = "\n".join(event.get("assistant_text", "") for event in self.events)
        lower = final_text.lower()
        required_facts = {
            "mutable_default": "mutable" in lower and "default" in lower,
            "idempotency": "idempot" in lower,
            "ordering": "sequence" in lower or "ordering" in lower,
            "terminal_guard": "terminal" in lower,
            "formatter": "formatter.py" in lower,
            "continuation_constraints": (
                "constraint" in lower
                or "read-only" in lower
                or "supplied" in lower
            ),
        }
        final_assistant = next(
            (
                event.get("assistant_text", "")
                for event in reversed(self.events)
                if event.get("assistant_text")
            ),
            "",
        )
        task_success = bool(
            self.http_errors == 0
            and self.invalid_tool_calls == 0
            and self.missing_tool_requests == 0
            and len(successful) >= 3
            and len(contexts) > 0
            and max(contexts) >= 8000
            and all(required_facts.values())
            and all(item.get("recovery_ok") for item in self.truncation_recoveries if item.get("forced"))
        )
        return {
            "task_success": task_success,
            "assistant_turns": len(self.events),
            "http_errors": self.http_errors,
            "http_409_count": sum(event.get("request_status") == 409 for event in self.events),
            "valid_tool_calls": self.injected_tool_results,
            "invalid_tool_calls": self.invalid_tool_calls,
            "missing_expected_tool_requests": self.missing_tool_requests,
            "unknown_tool_paths": self.unknown_tool_paths,
            "forced_truncations": sum(item.get("forced") for item in self.truncation_recoveries),
            "truncation_recoveries": len(successful),
            "truncation_recovery_records": self.truncation_recoveries,
            "cache_exact_reuse_requests": cache_reuse,
            "cache_fresh_fallback_requests": fresh_fallback,
            "cache_reuse_vs_fallback": {
                "exact_reuse": cache_reuse,
                "fresh_fallback": fresh_fallback,
            },
            "turns_with_metrics": len(metrics),
            "max_logical_context_tokens": max(contexts, default=None),
            "final_logical_context_tokens": contexts[-1] if contexts else None,
            "warm_ttft_seconds": warm_ttft,
            "median_warm_ttft_seconds": statistics.median(warm_ttft) if warm_ttft else None,
            "p95_warm_ttft_seconds": percentile(warm_ttft, 0.95),
            "max_warm_ttft_seconds": max(warm_ttft) if warm_ttft else None,
            "decode_tps": decode,
            "median_decode_tps": statistics.median(decode) if decode else None,
            "p95_decode_tps": percentile(decode, 0.95),
            "required_facts": required_facts,
            "final_assistant_text": final_assistant,
            "injected_tool_after_invalid": False,
        }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output = Path(args.output).resolve()
    server_log = output.with_suffix(".server.log")
    fixtures = fixture_map()
    result: dict[str, Any] = {
        "benchmark": "long-serialized-coding-agent",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model_name": "g128",
        "model_path": str(Path(args.model).resolve()),
        "server_module": "qwen38.server",
        "base_url": args.base_url,
        "session_id": args.session_id,
        "max_tokens_default": args.max_tokens,
        "fixture_root": str(FIXTURE_ROOT),
        "fixture_hashes": {
            path.name: sha256_file(path)
            for path in sorted(FIXTURE_ROOT.iterdir())
            if path.is_file()
        },
        "long_fixture_sha256": sha256_bytes(fixtures["long_dependency_audit.txt"].encode()),
        "health": None,
        "tasks": [],
        "exception": None,
        "safety_stop_reason": None,
    }
    process: subprocess.Popen[bytes] | None = None
    log_handle = None
    runner: SessionRunner | None = None
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        log_handle = server_log.open("wb")
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "qwen38.server",
                "--model",
                args.model,
                "--host",
                args.host,
                "--port",
                str(args.port),
                "--model-name",
                MODEL_NAME,
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
        status, models, error = request_json(args.base_url, "/v1/models", timeout=20.0)
        result["models"] = models
        if status != 200:
            raise RuntimeError(f"model endpoint failed: HTTP {status}: {error}")

        runner = SessionRunner(args.base_url, args.session_id, args.max_tokens)
        runner.expected_turn(
            "inspect_buggy_cache",
            "Start by requesting buggy_cache.py with the exact tool protocol. Do not guess its contents.",
            "buggy_cache.py",
            max_tokens=128,
        )
        runner.normal_turn(
            "initial_buggy_cache_diagnosis",
            "The implementation file is supplied above. Diagnose the instance-isolation failure and state the minimal testable fix. Keep the early read-only and no-invented-contents constraints visible, but do not request another file yet.",
            max_tokens=128,
        )
        runner.expected_turn(
            "inspect_buggy_cache_test",
            "Now request tests/test_buggy_cache.py exactly with the tool protocol so the failing assertion can be checked.",
            "tests/test_buggy_cache.py",
            max_tokens=96,
        )
        runner.normal_turn(
            "buggy_cache_revision",
            "Use the test result to revise the diagnosis. Explain why the mutable default leaks between instances and give the smallest patch. Do not claim any file not supplied.",
            max_tokens=128,
        )
        runner.expected_turn(
            "inspect_retry",
            "Next inspect retry.py. Request it with TOOL_CALL read_file retry.py and wait for the result.",
            "retry.py",
            max_tokens=96,
        )
        runner.normal_turn(
            "retry_wrong_hypothesis",
            "The failing test output is:\n"
            + fixture("test_retry_failure.txt")
            + "\nInitial hypothesis: simply retrying the store assignment is enough. That hypothesis is wrong. Revise it using the supplied source and test, explain the duplicate-before-apply guard, and state what the second delivery returns.",
            max_tokens=160,
        )
        runner.expected_turn(
            "inspect_state",
            "Now inspect state.py using the exact tool protocol. Preserve the earlier idempotency constraint.",
            "state.py",
            max_tokens=96,
        )
        runner.expected_turn(
            "inspect_events",
            "Request events.py next so the publisher boundary can be checked against state.py.",
            "events.py",
            max_tokens=96,
        )
        runner.normal_turn(
            "state_failing_test_revision",
            "The state test output is:\n"
            + fixture("test_state_failure.txt")
            + "\nGive a concise multi-file patch plan. Preserve event sequence ordering, prevent a terminal job from becoming active again, reject the late event, and remember the original idempotency requirement.",
            max_tokens=160,
        )
        runner.expected_turn(
            "inspect_formatter",
            "Inspect formatter.py with the tool protocol before proposing its envelope change.",
            "formatter.py",
            max_tokens=96,
        )
        runner.normal_turn(
            "formatter_test_revision",
            "The formatter test result is:\n"
            + fixture("test_formatter_failure.txt")
            + "\nReturn a concise JSON-oriented fix plan. Preserve the request identifier and the smallest-patch rule.",
            max_tokens=128,
        )
        runner.expected_turn(
            "inspect_long_audit",
            "The next dependency is long_dependency_audit.txt. Request it exactly; it is a long deterministic tool result. Do not summarize before it arrives.",
            "long_dependency_audit.txt",
            max_tokens=96,
        )
        runner.normal_turn(
            "long_audit_synthesis",
            "Using the supplied long audit and the earlier fixture results, state the current root-cause map in a compact way. Explicitly retain the early constraints, the wrong retry hypothesis and its correction, the failing-test revision, and the terminal-state guard. Do not request a tool.",
            max_tokens=160,
        )
        runner.force_truncation(
            1,
            "Reconcile all four fixture issues and the long audit before the first checkpoint.",
        )
        runner.normal_turn(
            "post_truncation_review_1",
            "The first capped checklist has been continued. Without restarting it, give the next minimal implementation decision and mention why the earlier constraints still apply.",
            max_tokens=128,
        )
        runner.force_truncation(
            2,
            "Review the revised idempotency and terminal-event decisions against the supplied evidence.",
        )
        runner.normal_turn(
            "post_truncation_review_2",
            "Continue the serialized review after checkpoint two. Reconcile any remaining contradiction between the initial wrong hypothesis and the failing test result; do not invent new files.",
            max_tokens=128,
        )
        runner.force_truncation(
            3,
            "Prepare the final implementation checklist while preserving all early constraints and the long audit dependencies.",
        )
        runner.normal_turn(
            "final_solution_summary",
            "Reply with exactly six short bullet lines and no tool request. Include one line each for: the mutable default fix; idempotent duplicate handling; event sequence ordering; the terminal-state guard; the formatter.py envelope correction; and the words `read-only constraints preserved` plus `only supplied tool results used`.",
            max_tokens=128,
        )

        status, metrics, error = request_json(args.base_url, "/metrics", timeout=30.0)
        result["server_metrics_before_reset"] = metrics if status == 200 else {"status": status, "error": error}
        result["server_metrics_status"] = status
        if runner is not None:
            result["summary"] = runner.summary()
            result["events"] = runner.events
            result["transcript"] = runner.messages
        result["classification"] = (
            "PASS_LONG_SERIALIZED_AGENT"
            if result.get("summary", {}).get("task_success")
            else "LONG_AGENT_COMPLETED_WITH_QUALITY_OR_PROTOCOL_FAILURES"
        )
    except KeyboardInterrupt:
        result["exception"] = "KeyboardInterrupt"
        result["safety_stop_reason"] = "interrupted by operator"
        result["classification"] = "LONG_AGENT_SAFETY_STOP"
    except BaseException as exc:
        result["exception"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = __import__("traceback").format_exc()
        result["classification"] = "LONG_AGENT_BENCHMARK_ERROR"
        if runner is not None:
            result["summary"] = runner.summary()
            result["events"] = runner.events
            result["transcript"] = runner.messages
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
        result["server_exit_code"] = process.returncode if process is not None else None
        result["total_elapsed_seconds"] = time.perf_counter() - started
        result["rss_max_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write_json(output, result)
        print(
            json.dumps(
                {
                    "classification": result.get("classification"),
                    "exception": result.get("exception"),
                    "summary": result.get("summary"),
                    "total_elapsed_seconds": result["total_elapsed_seconds"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return 0 if result.get("classification") == "PASS_LONG_SERIALIZED_AGENT" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Long serialized coding-agent benchmark")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--base-url", default="http://127.0.0.1:8766")
    parser.add_argument("--session-id", default="long-agent")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--server-timeout", type=float, default=1200.0)
    parser.add_argument("--output", required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
