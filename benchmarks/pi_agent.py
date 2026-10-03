#!/usr/bin/env python3
"""Real Pi coding-agent benchmark against ``qwen38-server``.

Pi itself owns the agent loop and executes its built-in tools (read, bash,
edit, write).  This runner only:

1. starts the server (or uses ``--base-url``);
2. writes a disposable Pi agent directory from ``integrations/pi/*.example.json``
   (standard ``models.json`` + compaction settings, no extensions);
3. copies each fixture repo from ``benchmarks/pi_fixtures/<task>/repo`` into a
   temporary git repository and runs ``pi --mode json`` with the task
   prompt(s) (later prompts continue the same Pi session with ``-c``);
4. evaluates the result with the fixture's hidden check, which Pi never sees,
   and verifies protected files are unchanged;
5. joins Pi's JSONL events with the server's per-request metrics log.

Nothing is injected into Pi's conversation.  Results go to
``benchmarks/results/pi-stage9.jsonl`` (one ``request`` record per model call,
one ``task`` record per task) plus a summary JSON.

Loads the 27B model; do not run alongside another heavy MLX process::

    uv run python benchmarks/pi_agent.py
    uv run python benchmarks/pi_agent.py --tasks single_bug --output /tmp/x.jsonl
    uv run python benchmarks/pi_agent.py --verify-fixtures   # no model, no Pi
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "benchmarks" / "pi_fixtures"
DEFAULT_MODEL = "/Users/artem/models/Qwen3.8-27B-3bit-g128-MLX"
TASK_ORDER = [
    "single_bug",
    "multi_file_bug",
    "wrong_hypothesis",
    "search_read",
    "long_output",
    "constraint",
    "long_session",
]
INVALID_TOOL_PREFIXES = ("Validation failed for tool", "Tool ")


def sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def load_task(task_id: str) -> dict[str, Any]:
    task = json.loads((FIXTURES / task_id / "task.json").read_text())
    task["id"] = task_id
    task["check"] = [arg.replace("{fixture}", str(FIXTURES / task_id)) for arg in task["check"]]
    return task


def prepare_repo(task_id: str, parent: Path) -> Path:
    repo = parent / task_id
    suffix = 1
    while repo.exists():  # the same task may be listed more than once
        suffix += 1
        repo = parent / f"{task_id}.{suffix}"
    shutil.copytree(FIXTURES / task_id / "repo", repo)
    env = {**os.environ, "GIT_AUTHOR_NAME": "bench", "GIT_AUTHOR_EMAIL": "bench@localhost",
           "GIT_COMMITTER_NAME": "bench", "GIT_COMMITTER_EMAIL": "bench@localhost"}
    for command in (["git", "init", "-q"], ["git", "add", "-A"], ["git", "commit", "-q", "-m", "fixture"]):
        subprocess.run(command, cwd=repo, check=True, env=env)
    return repo


def run_check(task: dict[str, Any], repo: Path) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(task["check"], cwd=repo, capture_output=True, text=True, timeout=180)
        output = (completed.stdout + completed.stderr)[-2000:]
        code = completed.returncode
    except subprocess.TimeoutExpired:
        output, code = "check timed out", -1
    return {"passed": code == 0, "exit_code": code, "output_tail": output, "seconds": time.perf_counter() - started}


def verify_fixtures(task_ids: list[str]) -> int:
    """Every fixture must fail its hidden check before any agent work."""

    failures = 0
    with tempfile.TemporaryDirectory(prefix="pi-fixture-verify-") as tmp:
        for task_id in task_ids:
            task = load_task(task_id)
            repo = prepare_repo(task_id, Path(tmp))
            result = run_check(task, repo)
            ok = not result["passed"]
            failures += not ok
            print(f"{task_id:<18} baseline check fails as expected: {ok}")
    return 1 if failures else 0


def write_agent_dir(agent_dir: Path, base_url: str) -> None:
    agent_dir.mkdir(parents=True, exist_ok=True)
    models = json.loads((ROOT / "integrations/pi/models.example.json").read_text())
    models["providers"]["qwen38-local"]["baseUrl"] = base_url.rstrip("/") + "/v1"
    (agent_dir / "models.json").write_text(json.dumps(models, indent=2))
    shutil.copy(ROOT / "integrations/pi/settings.example.json", agent_dir / "settings.json")


def pi_version() -> str:
    try:
        return subprocess.run(["pi", "--version"], capture_output=True, text=True, timeout=30).stdout.strip()
    except OSError as exc:
        return f"unavailable: {exc}"


def get_json(url: str, timeout: float = 30.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def summarize_pi_events(path: Path) -> dict[str, Any]:
    turns = 0
    stop_reasons: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    executions: list[dict[str, Any]] = []
    errors: list[str] = []
    retries = 0
    compactions = 0
    final_text = ""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = event.get("type")
        if kind == "turn_end":
            turns += 1
        elif kind == "message_end" and (event.get("message") or {}).get("role") == "assistant":
            message = event["message"]
            stop_reasons.append(message.get("stopReason"))
            if message.get("errorMessage"):
                errors.append(message["errorMessage"])
            for block in message.get("content") or []:
                if block.get("type") == "toolCall":
                    tool_calls.append({"id": block.get("id"), "name": block.get("name"), "arguments": block.get("arguments")})
                elif block.get("type") == "text" and block.get("text", "").strip():
                    final_text = block["text"]
        elif kind == "tool_execution_end":
            content = (event.get("result") or {}).get("content") or []
            text = "".join(part.get("text", "") for part in content if isinstance(part, dict))
            executions.append({
                "id": event.get("toolCallId"),
                "name": event.get("toolName"),
                "is_error": bool(event.get("isError")),
                "invalid": bool(event.get("isError")) and text.startswith(INVALID_TOOL_PREFIXES),
                "error_text": text[:300] if event.get("isError") else None,
            })
        elif kind == "agent_end" and event.get("willRetry"):
            retries += 1
        elif kind and "compaction" in kind and kind.endswith("start"):
            compactions += 1
    return {
        "turns": turns,
        "stop_reasons": stop_reasons,
        "tool_calls": tool_calls,
        "executions": executions,
        "errors": errors,
        "retries": retries,
        "compactions": compactions,
        "final_text": final_text[-600:],
    }


def run_task(task: dict[str, Any], args: argparse.Namespace, work: Path, base_url: str, metrics_log: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    repo = prepare_repo(task["id"], work / "repos")
    protected = {name: sha256(repo / name) for name in task.get("protected", [])}
    agent_dir = work / "agent"
    session_dir = work / "sessions" / repo.name
    session_dir.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "PI_CODING_AGENT_DIR": str(agent_dir),
        "PI_CODING_AGENT_SESSION_DIR": str(session_dir),
        "PI_OFFLINE": "1",
        "PI_TELEMETRY": "0",
        "PI_SKIP_VERSION_CHECK": "1",
    }
    log_offset = len(metrics_log.read_text().splitlines()) if metrics_log.exists() else 0
    started = time.perf_counter()
    prompt_runs = []
    for index, prompt in enumerate(task["prompts"]):
        events_path = work / "events" / f"{repo.name}.{index}.jsonl"
        events_path.parent.mkdir(parents=True, exist_ok=True)
        command = ["pi", "--mode", "json", "--provider", "qwen38-local", "--model", "qwen38-local"]
        if index:
            command.append("-c")
        command.append(prompt)
        prompt_started = time.perf_counter()
        timed_out = False
        with events_path.open("w") as stdout, (events_path.with_suffix(".stderr")).open("w") as stderr:
            process = subprocess.Popen(command, cwd=repo, env=env, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
            try:
                exit_code = process.wait(timeout=args.task_timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                process.terminate()
                try:
                    exit_code = process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    exit_code = process.wait()
        summary = summarize_pi_events(events_path)
        summary.update({"prompt_index": index, "exit_code": exit_code, "timed_out": timed_out,
                        "seconds": time.perf_counter() - prompt_started})
        prompt_runs.append(summary)
        print(f"  [{task['id']}#{index}] exit={exit_code} turns={summary['turns']} tools={[c['name'] for c in summary['tool_calls']]} "
              f"stops={summary['stop_reasons']} errors={summary['errors'][:2]} {summary['seconds']:.0f}s", flush=True)
        if timed_out:
            break
    wall = time.perf_counter() - started
    check = run_check(task, repo)
    protected_ok = all(sha256(repo / name) == digest for name, digest in protected.items())
    diff = subprocess.run(["git", "diff", "--stat"], cwd=repo, capture_output=True, text=True).stdout
    records = []
    if metrics_log.exists():
        for line in metrics_log.read_text().splitlines()[log_offset:]:
            record = json.loads(line)
            record = {key: record.get(key) for key in (
                "http_status", "stream", "message_count", "last_role", "prompt_tokens", "processed_suffix_tokens",
                "cached_tokens", "generated_tokens", "logical_context_tokens", "cache_reuse_mode",
                "cache_fallback_reason", "finish_reason", "tool_calls", "tool_parse_error", "unknown_tools",
                "argument_type_mismatches", "ttft_seconds", "decode_tps", "wall_seconds", "replay_detected", "repeated_tool_call",
                "cancelled", "error", "peak_memory_bytes",
            )}
            records.append({"type": "request", "task_id": task["id"], "turn": len(records) + 1, **record})
    executions = [e for run in prompt_runs for e in run["executions"]]
    calls = [c for run in prompt_runs for c in run["tool_calls"]]
    statuses = [r["http_status"] for r in records]
    task_record = {
        "type": "task",
        "task_id": task["id"],
        "category": task["category"],
        "prompts": len(task["prompts"]),
        "success": check["passed"] and protected_ok and not any(run["timed_out"] for run in prompt_runs),
        "tests_pass": check["passed"],
        "protected_files_unchanged": protected_ok,
        "check": check,
        "pi_turns": sum(run["turns"] for run in prompt_runs),
        "model_requests": len(records),
        "tool_calls": len(calls),
        "tool_calls_by_name": {name: sum(c["name"] == name for c in calls) for name in sorted({c["name"] for c in calls})},
        "valid_tool_calls": sum(not e["invalid"] for e in executions),
        "invalid_tool_calls": sum(e["invalid"] for e in executions),
        # Error results from valid calls: failing test commands, edit text not found, etc.
        "tool_execution_errors": sum(e["is_error"] and not e["invalid"] for e in executions),
        "tool_error_samples": [e["error_text"] for e in executions if e["is_error"]][:5],
        "tool_parse_failures": sum(bool(r["tool_parse_error"]) for r in records),
        "protocol_errors": sum(s != 200 for s in statuses) + sum(bool(r.get("error")) and s == 200 for r, s in zip(records, statuses)),
        "pi_errors": [err for run in prompt_runs for err in run["errors"]],
        "pi_retries": sum(run["retries"] for run in prompt_runs),
        "stale_replay_count": sum(bool(r["replay_detected"]) for r in records),
        "repeated_tool_calls": sum(bool(r.get("repeated_tool_call")) for r in records),
        "exact_reuse_count": sum(r["cache_reuse_mode"] == "exact_reuse" for r in records),
        "fresh_fallback_count": sum(r["cache_reuse_mode"] == "fresh_fallback" for r in records),
        "fresh_prefill_count": sum(r["cache_reuse_mode"] == "fresh_prefill" for r in records),
        "length_finishes": sum(r["finish_reason"] == "length" for r in records),
        "max_logical_context": max([r["logical_context_tokens"] or 0 for r in records] or [0]),
        "max_prompt_tokens": max([r["prompt_tokens"] or 0 for r in records] or [0]),
        "output_tokens": sum(r["generated_tokens"] or 0 for r in records),
        "median_warm_ttft": statistics.median([r["ttft_seconds"] for r in records if r["cache_reuse_mode"] == "exact_reuse" and r["ttft_seconds"]] or [0]),
        "max_ttft": max([r["ttft_seconds"] or 0 for r in records] or [0]),
        "median_decode_tps": statistics.median([r["decode_tps"] for r in records if r["decode_tps"] and (r["generated_tokens"] or 0) >= 16] or [0]),
        "wall_seconds": wall,
        "timed_out": any(run["timed_out"] for run in prompt_runs),
        "pi_exit_codes": [run["exit_code"] for run in prompt_runs],
        "final_text": prompt_runs[-1]["final_text"] if prompt_runs else "",
        "diff_stat": diff[-1500:],
        "failure_category": None,
    }
    return task_record, records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--port", type=int, default=8032)
    parser.add_argument("--base-url", default=None, help="use an already running server")
    parser.add_argument("--tasks", default=",".join(TASK_ORDER))
    parser.add_argument("--task-timeout", type=float, default=1800.0)
    parser.add_argument("--output", default=str(ROOT / "benchmarks/results/pi-stage9.jsonl"))
    parser.add_argument("--keep-work", action="store_true")
    parser.add_argument("--verify-fixtures", action="store_true")
    parser.add_argument("--server-arg", action="append", default=[], help="extra qwen38-server argument")
    args = parser.parse_args()
    task_ids = [t for t in args.tasks.split(",") if t]
    if args.verify_fixtures:
        return verify_fixtures(task_ids)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="qwen38-pi-bench-"))
    metrics_log = output.with_suffix(".server-requests.jsonl")
    base_url = args.base_url or f"http://127.0.0.1:{args.port}"
    server = None
    if args.base_url is None:
        metrics_log.unlink(missing_ok=True)
        server_log = output.with_suffix(".server.log").open("w")
        server = subprocess.Popen(
            [sys.executable, "-m", "qwen38.server", "--model", args.model, "--port", str(args.port),
             "--model-name", "qwen38-local", "--metrics-log", str(metrics_log), *args.server_arg],
            cwd=ROOT, stdout=server_log, stderr=subprocess.STDOUT,
        )
        deadline = time.time() + 300
        while True:
            if server.poll() is not None:
                raise SystemExit(f"server exited with {server.returncode}")
            try:
                if get_json(base_url + "/health", timeout=5).get("status") == "ready":
                    break
            except OSError:
                pass
            if time.time() > deadline:
                raise SystemExit("server did not become ready")
            time.sleep(1)
    write_agent_dir(work / "agent", base_url)
    header = {"type": "run", "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "pi_version": pi_version(),
              "model": args.model, "tasks": task_ids, "work": str(work),
              "server_args": args.server_arg, "settings": json.loads((ROOT / "integrations/pi/settings.example.json").read_text())}
    print(json.dumps(header), flush=True)
    tasks_out = []
    try:
        with output.open("w") as out:
            out.write(json.dumps(header) + "\n")
            for task_id in task_ids:
                task = load_task(task_id)
                print(f"== {task_id}: {task['category']}", flush=True)
                task_record, records = run_task(task, args, work, base_url, metrics_log)
                for record in records:
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.write(json.dumps(task_record, ensure_ascii=False) + "\n")
                out.flush()
                tasks_out.append(task_record)
                print(f"   -> success={task_record['success']} turns={task_record['pi_turns']} requests={task_record['model_requests']} "
                      f"tools={task_record['tool_calls']} invalid={task_record['invalid_tool_calls']} exec_err={task_record['tool_execution_errors']} "
                      f"reuse={task_record['exact_reuse_count']} fresh={task_record['fresh_fallback_count'] + task_record['fresh_prefill_count']} "
                      f"maxctx={task_record['max_logical_context']} wall={task_record['wall_seconds']:.0f}s", flush=True)
        if server is not None:
            header["server_metrics"] = get_json(base_url + "/metrics")
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                server.kill()
        if not args.keep_work:
            shutil.rmtree(work / "repos", ignore_errors=True)
    summary = {
        "run": header,
        "tasks": len(tasks_out),
        "successes": sum(t["success"] for t in tasks_out),
        "tool_calls": sum(t["tool_calls"] for t in tasks_out),
        "invalid_tool_calls": sum(t["invalid_tool_calls"] for t in tasks_out),
        "tool_execution_errors": sum(t["tool_execution_errors"] for t in tasks_out),
        "tool_parse_failures": sum(t["tool_parse_failures"] for t in tasks_out),
        "protocol_errors": sum(t["protocol_errors"] for t in tasks_out),
        "stale_replay": sum(t["stale_replay_count"] for t in tasks_out),
        "exact_reuse": sum(t["exact_reuse_count"] for t in tasks_out),
        "fresh": sum(t["fresh_fallback_count"] + t["fresh_prefill_count"] for t in tasks_out),
        "max_logical_context": max([t["max_logical_context"] for t in tasks_out] or [0]),
        "wall_seconds": sum(t["wall_seconds"] for t in tasks_out),
        "per_task": {t["task_id"]: {k: t[k] for k in ("success", "pi_turns", "tool_calls", "invalid_tool_calls", "max_logical_context", "wall_seconds")} for t in tasks_out},
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
