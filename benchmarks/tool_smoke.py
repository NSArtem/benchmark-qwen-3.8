#!/usr/bin/env python3
"""Short real-model OpenAI tool-calling smoke for ``qwen38-server``.

Starts the real server with the g128 model and drives one session through
native ``tools`` / ``tool_calls`` / ``tool`` round trips over SSE streaming:

1. a task that requires ``read_file``;
2. the tool result, which must be used in the answer (exact cache reuse);
3. a task that requires ``run_command`` (the command really runs in a
   temporary directory, but only if it is the expected one);
4. its result;
5. a non-streaming text turn;
6. a changed tool schema, which must fall back to a fresh prefill;
7. a request to read two files, to observe multi-call turns.

Tool results come only from executing the call the model actually made.
An unexpected call is recorded as invalid and is never answered with an
expected result.

Loads the 27B model; do not run alongside another heavy MLX process::

    uv run python benchmarks/tool_smoke.py --output benchmarks/results/tool-smoke.json
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "/Users/artem/models/Qwen3.8-27B-3bit-g128-MLX"
MODEL_NAME = "qwen38-local"

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file from the working directory.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Relative file path"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command in the working directory and return stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "Shell command"}},
                "required": ["command"],
            },
        },
    },
]
EXTRA_TOOL = {
    "type": "function",
    "function": {
        "name": "list_dir",
        "description": "List files in a directory.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    },
}

FILES = {
    "notes.txt": "Release checklist\nThe launch code is AMBER-42.\nOwner: Lantern team\n",
    "hello.py": "print('hello from smoke', 6 + 1)\n",
    "a.txt": "alpha=1\n",
    "b.txt": "beta=2\n",
}


def post_stream(base_url: str, body: dict[str, Any], timeout: float = 900.0) -> dict[str, Any]:
    payload = json.dumps({**body, "stream": True, "stream_options": {"include_usage": True}}).encode()
    request = urllib.request.Request(
        base_url + "/v1/chat/completions", data=payload, method="POST",
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
    )
    started = time.perf_counter()
    first_content: float | None = None
    content = ""
    calls: dict[int, dict[str, Any]] = {}
    finish = None
    usage = None
    metrics = None
    chunks = 0
    heartbeats = 0
    error = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            for raw_line in response:
                line = raw_line.decode("utf-8").rstrip("\n")
                if line.startswith(":"):
                    heartbeats += 1
                    continue
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if "error" in event:
                    error = event["error"]
                    break
                chunks += 1
                if event.get("usage"):
                    usage = event["usage"]
                    metrics = event.get("x_qwen38_metrics")
                for choice in event.get("choices", []):
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        first_content = first_content or time.perf_counter()
                        content += delta["content"]
                    for call in delta.get("tool_calls") or []:
                        slot = calls.setdefault(call["index"], {"id": None, "name": "", "arguments": ""})
                        slot["id"] = call.get("id") or slot["id"]
                        slot["name"] += (call.get("function") or {}).get("name") or ""
                        slot["arguments"] += (call.get("function") or {}).get("arguments") or ""
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
    except urllib.error.HTTPError as exc:
        return {"status": exc.code, "error": exc.read().decode("utf-8", "replace")}
    return {
        "status": status,
        "content": content,
        "tool_calls": [calls[i] for i in sorted(calls)],
        "finish_reason": finish,
        "usage": usage,
        "metrics": metrics,
        "chunks": chunks,
        "heartbeats": heartbeats,
        "error": error,
        "seconds": time.perf_counter() - started,
    }


def post_json(base_url: str, path: str, body: dict[str, Any] | None = None, timeout: float = 900.0) -> tuple[int, Any]:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        base_url + path, data=data, method="GET" if body is None else "POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


class Session:
    def __init__(self, base_url: str, workdir: Path, max_tokens: int) -> None:
        self.base_url = base_url
        self.workdir = workdir
        self.max_tokens = max_tokens
        self.messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": "You are a coding assistant working in a small project directory. "
                "Use the provided tools to inspect files and run commands; never guess file contents.",
            }
        ]
        self.tools = TOOLS
        self.steps: list[dict[str, Any]] = []

    def assistant_turn(self, label: str, *, stream: bool = True, tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        tools = tools or self.tools
        body = {"model": MODEL_NAME, "messages": self.messages, "tools": tools, "max_completion_tokens": self.max_tokens}
        if stream:
            result = post_stream(self.base_url, body)
        else:
            started = time.perf_counter()
            status, payload = post_json(self.base_url, "/v1/chat/completions", body)
            result = {"status": status, "seconds": time.perf_counter() - started}
            if status == 200:
                choice = payload["choices"][0]
                result.update(
                    {
                        "content": choice["message"].get("content") or "",
                        "tool_calls": [
                            {"id": c["id"], "name": c["function"]["name"], "arguments": c["function"]["arguments"]}
                            for c in choice["message"].get("tool_calls") or []
                        ],
                        "finish_reason": choice["finish_reason"],
                        "usage": payload["usage"],
                        "metrics": payload["x_qwen38_metrics"],
                    }
                )
            else:
                result["error"] = payload
        result["label"] = label
        result["stream"] = stream
        self.steps.append(result)
        if result.get("status") == 200 and not result.get("error"):
            message: dict[str, Any] = {"role": "assistant", "content": result["content"] or None}
            if result["tool_calls"]:
                # Replay like an OpenAI client: parse then re-serialize arguments.
                message["tool_calls"] = [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(json.loads(call["arguments"]), separators=(",", ":")),
                        },
                    }
                    for call in result["tool_calls"]
                ]
            self.messages.append(message)
        return result

    def execute_tools(self, result: dict[str, Any], expected: dict[str, Any]) -> list[dict[str, Any]]:
        """Run each call the model made, if it matches the expected tool/argument."""

        records = []
        for call in result.get("tool_calls") or []:
            args = json.loads(call["arguments"])
            valid = call["name"] == expected["name"] and expected["check"](args)
            output = "error: unexpected tool call"
            if valid and call["name"] == "read_file":
                output = (self.workdir / args["path"]).read_text()
            elif valid and call["name"] == "run_command":
                completed = subprocess.run(
                    shlex.split(args["command"]), cwd=self.workdir, capture_output=True, text=True, timeout=30
                )
                output = completed.stdout + completed.stderr
            records.append({"name": call["name"], "arguments": args, "valid": valid, "output": output})
            self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": output})
        return records


def run(args: argparse.Namespace) -> dict[str, Any]:
    workdir = Path(tempfile.mkdtemp(prefix="qwen38-tool-smoke-"))
    for name, text in FILES.items():
        (workdir / name).write_text(text)
    base_url = f"http://127.0.0.1:{args.port}"
    log = Path(args.output).with_suffix(".server.log")
    metrics_log = Path(args.output).with_suffix(".requests.jsonl")
    metrics_log.unlink(missing_ok=True)
    server = subprocess.Popen(
        [sys.executable, "-m", "qwen38.server", "--model", args.model, "--port", str(args.port),
         "--model-name", MODEL_NAME, "--metrics-log", str(metrics_log)],
        stdout=log.open("w"), stderr=subprocess.STDOUT, cwd=ROOT,
    )
    result: dict[str, Any] = {"model": args.model, "workdir": str(workdir), "checks": {}}
    try:
        deadline = time.time() + 300
        while time.time() < deadline:
            if server.poll() is not None:
                raise RuntimeError(f"server exited with {server.returncode}")
            try:
                if post_json(base_url, "/health", timeout=5)[0] == 200:
                    break
            except OSError:
                pass
            time.sleep(1)
        session = Session(base_url, workdir, args.max_tokens)
        checks = result["checks"]

        session.messages.append({"role": "user", "content": "What is the launch code written in notes.txt?"})
        t1 = session.assistant_turn("read_request")
        checks["t1_tool_calls"] = t1.get("finish_reason") == "tool_calls" and bool(t1.get("tool_calls"))
        t1["executed"] = session.execute_tools(t1, {"name": "read_file", "check": lambda a: a.get("path") == "notes.txt"})
        checks["t1_valid_read"] = bool(t1["executed"]) and all(r["valid"] for r in t1["executed"])

        t2 = session.assistant_turn("read_answer")
        checks["t2_uses_result"] = "AMBER-42" in (t2.get("content") or "")
        checks["t2_exact_reuse"] = (t2.get("metrics") or {}).get("cache_reuse_mode") == "exact_reuse"

        session.messages.append({"role": "user", "content": "Run `python3 hello.py` and tell me exactly what it prints."})
        t3 = session.assistant_turn("command_request")
        checks["t3_tool_calls"] = t3.get("finish_reason") == "tool_calls" and bool(t3.get("tool_calls"))
        checks["t3_exact_reuse"] = (t3.get("metrics") or {}).get("cache_reuse_mode") == "exact_reuse"
        t3["executed"] = session.execute_tools(
            t3, {"name": "run_command", "check": lambda a: "python3 hello.py" in str(a.get("command", ""))}
        )
        checks["t3_valid_command"] = bool(t3["executed"]) and all(r["valid"] for r in t3["executed"])

        t4 = session.assistant_turn("command_answer")
        checks["t4_uses_result"] = "hello from smoke 7" in (t4.get("content") or "")
        checks["t4_exact_reuse"] = (t4.get("metrics") or {}).get("cache_reuse_mode") == "exact_reuse"

        session.messages.append({"role": "user", "content": "In one sentence, which tools did you use so far?"})
        t5 = session.assistant_turn("non_stream_text", stream=False)
        checks["t5_text_stop"] = t5.get("finish_reason") == "stop" and not t5.get("tool_calls")
        checks["t5_exact_reuse"] = (t5.get("metrics") or {}).get("cache_reuse_mode") == "exact_reuse"

        session.tools = TOOLS + [EXTRA_TOOL]
        session.messages.append({"role": "user", "content": "Reply with just the word OK."})
        t6 = session.assistant_turn("changed_tools")
        checks["t6_fresh_fallback"] = (t6.get("metrics") or {}).get("cache_fallback_reason") == "tool_set_changed"
        checks["t6_ok"] = t6.get("status") == 200 and t6.get("finish_reason") in {"stop", "tool_calls"}

        session.messages.append({"role": "user", "content": "Read both a.txt and b.txt, then give me the sum of their values."})
        t7 = session.assistant_turn("multi_read_request")
        t7["executed"] = session.execute_tools(t7, {"name": "read_file", "check": lambda a: a.get("path") in {"a.txt", "b.txt"}})
        checks["t7_valid_reads"] = bool(t7["executed"]) and all(r["valid"] for r in t7["executed"])
        reads = {r["arguments"].get("path") for r in t7["executed"]}
        depth = 0
        while reads != {"a.txt", "b.txt"} and depth < 2 and session.steps[-1].get("tool_calls"):
            follow = session.assistant_turn(f"multi_read_follow_{depth}")
            follow["executed"] = session.execute_tools(follow, {"name": "read_file", "check": lambda a: a.get("path") in {"a.txt", "b.txt"}})
            reads |= {r["arguments"].get("path") for r in follow["executed"]}
            depth += 1
        if session.steps[-1].get("tool_calls"):
            session.assistant_turn("multi_read_answer")
        checks["t7_answer_3"] = "3" in (session.steps[-1].get("content") or "")
        result["t7_calls_in_first_turn"] = len(t7.get("tool_calls") or [])

        status, metrics = post_json(base_url, "/metrics")
        result["server_metrics"] = metrics
        checks["no_409_or_5xx"] = metrics["http_5xx"] == 0 and metrics["active_session"]["cache_conflict_error"] == 0
        checks["no_http_errors"] = all(step.get("status") == 200 and not step.get("error") for step in session.steps)
        checks["no_stale_replay"] = not any((s.get("metrics") or {}).get("replay_detected") for s in session.steps)
        checks["no_tool_parse_failures"] = metrics["tool_parse_failures"] == 0
        result["steps"] = session.steps
        result["transcript"] = session.messages
        result["classification"] = "PASS_TOOL_SMOKE" if all(checks.values()) else "FAIL_TOOL_SMOKE"
    finally:
        server.terminate()
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            server.kill()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--port", type=int, default=8031)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--output", default=str(ROOT / "benchmarks/results/tool-smoke.json"))
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result = run(args)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"classification": result.get("classification"), "checks": result.get("checks")}, indent=2))
    for step in result.get("steps", []):
        m = step.get("metrics") or {}
        print(
            f"{step['label']:<22} finish={step.get('finish_reason')!s:<10} calls={[(c['name'], c['arguments']) for c in step.get('tool_calls') or []]} "
            f"mode={m.get('cache_reuse_mode')} suffix={m.get('processed_suffix_tokens')} ctx={m.get('logical_context_tokens')} "
            f"ttft={m.get('ttft_seconds') or 0:.2f}s decode={m.get('decode_tps') or 0:.1f} content={(step.get('content') or '')[:90]!r}"
        )
    return 0 if result.get("classification") == "PASS_TOOL_SMOKE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
