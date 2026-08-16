"""Run a bounded two-turn APC test against one mlx-vlm server child."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import httpx

from memory_utils import memory_summary, snapshot, with_rates
from watchdog import OOM_PATTERNS, POLICY


TURN1_USER = (
    "Inspect the Lantern incident described above. Give a concise diagnosis of the two most likely root causes "
    "and name the specific files/functions to change. Keep it to about 100-140 tokens; do not provide the full patch yet."
)
TURN2_USER = (
    "Now give the minimal patch strategy and identify the tests that should be updated. Keep the answer concise."
)
MAX_TOKENS = 128
TENANT_ID = "qwen38-overnight"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def append_jsonl(handle: Any, value: Any) -> None:
    handle.write(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n")
    handle.flush()


def terminate_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 10.0
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.25)
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def response_json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except (ValueError, json.JSONDecodeError):
        return {"_non_json_body": response.text}


def number(value: Any) -> float | int | None:
    if isinstance(value, bool):
        return None
    return value if isinstance(value, (int, float)) else None


def first_number(*values: Any) -> float | int | None:
    for value in values:
        parsed = number(value)
        if parsed is not None:
            return parsed
    return None


def latest_metrics(value: Any) -> dict[str, Any]:
    if isinstance(value, dict) and isinstance(value.get("latest"), dict):
        return value["latest"]
    return value if isinstance(value, dict) else {}


def assistant_text(value: Any) -> str:
    if not isinstance(value, dict) or not isinstance(value.get("choices"), list) or not value["choices"]:
        return ""
    message = value["choices"][0].get("message", {})
    content = message.get("content", "") if isinstance(message, dict) else ""
    return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)


def wait_health(client: httpx.Client, timeout: float, abort: threading.Event) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        if abort.is_set():
            raise RuntimeError("APC server watchdog aborted during startup")
        try:
            response = client.get("/health")
            if response.status_code == 200:
                value = response_json(response)
                if isinstance(value, dict):
                    last = value
                    if value.get("apc_enabled") is True:
                        return value
        except httpx.HTTPError:
            pass
        time.sleep(2.0)
    raise TimeoutError(f"APC server did not become ready; last health={last}")


def request_turn(
    client: httpx.Client,
    label: str,
    model_id: str,
    messages: list[dict[str, str]],
    output_path: Path,
    health: dict[str, Any],
    abort: threading.Event,
) -> dict[str, Any]:
    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "stream": False,
        "enable_thinking": False,
    }
    started = time.perf_counter()
    memory_before = snapshot()
    request_error: str | None = None
    response: httpx.Response | None = None
    try:
        response = client.post(
            "/v1/chat/completions",
            headers={"X-APC-Tenant": TENANT_ID},
            json=payload,
        )
        api_value = response_json(response)
    except httpx.HTTPError as exc:
        api_value = {"error": str(exc)}
        request_error = str(exc)
    wall = time.perf_counter() - started
    memory_after = snapshot()
    try:
        metrics_payload = response_json(client.get("/metrics"))
        cache_payload = response_json(client.get("/v1/cache/stats"))
    except httpx.HTTPError as exc:
        metrics_payload = {"error": str(exc)}
        cache_payload = {"error": str(exc)}
    latest = latest_metrics(metrics_payload)
    usage = api_value.get("usage", {}) if isinstance(api_value, dict) else {}
    details = usage.get("prompt_tokens_details", {}) if isinstance(usage, dict) else {}
    timings = api_value.get("timings", {}) if isinstance(api_value, dict) else {}
    if not isinstance(timings, dict):
        timings = {}
    prompt_tokens = number(usage.get("prompt_tokens") if isinstance(usage, dict) else None)
    cached_tokens = number(details.get("cached_tokens") if isinstance(details, dict) else None)
    completion_tokens = number(usage.get("completion_tokens") if isinstance(usage, dict) else None)
    hit = float(cached_tokens) / float(prompt_tokens) if cached_tokens is not None and prompt_tokens else None
    measurement = {
        "label": label,
        "http_status": response.status_code if response is not None else None,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "cache_hit_fraction": hit,
        "completion_tokens": completion_tokens,
        "wall_seconds": wall,
        "ttft_seconds": first_number(latest.get("ttft_s"), timings.get("ttft_s")),
        "prefill_seconds": first_number(latest.get("prompt_eval_time_s"), (timings.get("prompt_ms", 0) / 1000 if isinstance(timings.get("prompt_ms"), (int, float)) else None)),
        "prefill_tok_s": first_number(latest.get("prefill_tok_s"), timings.get("prompt_per_second")),
        "decode_tok_s": first_number(timings.get("predicted_per_second"), latest.get("decode_tok_s")),
        "peak_memory_gb": first_number(timings.get("peak_memory"), latest.get("peak_memory_gb")),
        "draft_kind": timings.get("draft_kind", latest.get("draft_kind")),
        "draft_rounds": first_number(timings.get("draft_rounds"), latest.get("draft_rounds")),
        "drafted_tokens": first_number(timings.get("draft_n"), latest.get("draft_n")),
        "accepted_tokens": first_number(timings.get("draft_n_accepted"), latest.get("draft_n_accepted")),
        "request_error": request_error,
        "assistant_text_sha256": digest(assistant_text(api_value)),
    }
    accepted = measurement["accepted_tokens"]
    drafted = measurement["drafted_tokens"]
    measurement["acceptance_rate"] = float(accepted) / float(drafted) if accepted is not None and drafted not in (None, 0) else None
    record = {
        "turn": label,
        "request": payload,
        "api_response": api_value,
        "client_measurements": measurement,
        "server_metrics_latest": latest,
        "server_metrics_response": metrics_payload,
        "server_cache_stats": cache_payload,
        "health_before": health,
        "memory_before": memory_before,
        "memory_after": memory_after,
    }
    write_json(output_path, record)
    if request_error or response is None or response.status_code >= 400:
        raise RuntimeError(f"{label} failed; response saved to {output_path}")
    return {**record, "assistant_text": assistant_text(api_value)}


def monitor_server(
    process: subprocess.Popen[Any],
    memory_log: Path,
    child_log: Path,
    abort: threading.Event,
    state: dict[str, Any],
) -> None:
    memory_log.parent.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = []
    previous = snapshot()
    emergency_streak = 0
    paging_streak = 0
    critical_streak = 0
    with memory_log.open("w", encoding="utf-8") as handle:
        append_jsonl(handle, {"event": "watchdog_start", "policy": POLICY, "memory": previous})
        while not abort.is_set() and process.poll() is None:
            time.sleep(1.0)
            current = snapshot()
            elapsed = max(current["timestamp"] - previous["timestamp"], 1e-6)
            rates = with_rates(previous, current, elapsed)
            try:
                tail = child_log.read_text(encoding="utf-8", errors="replace")[-12000:]
            except OSError:
                tail = ""
            oom = any(pattern.search(tail) for pattern in OOM_PATTERNS)
            free = current.get("free_percent")
            pageouts = rates.get("pageouts_per_second") or 0.0
            swapouts = rates.get("swapouts_per_second") or 0.0
            emergency = isinstance(free, (int, float)) and free <= POLICY["emergency_free_percent"]
            low_paging = isinstance(free, (int, float)) and free <= POLICY["low_free_percent"] and (pageouts >= POLICY["meaningful_pageouts_per_second"] or swapouts >= POLICY["meaningful_swapouts_per_second"])
            critical = isinstance(free, (int, float)) and free <= POLICY["paging_free_percent"] and swapouts >= POLICY["critical_swapouts_per_second"]
            emergency_streak = emergency_streak + 1 if emergency else 0
            paging_streak = paging_streak + 1 if low_paging else 0
            critical_streak = critical_streak + 1 if critical else 0
            record = {"event": "sample", "memory": current, "rates": rates, "signals": {"oom": oom, "emergency": emergency, "low_paging": low_paging, "critical": critical}}
            append_jsonl(handle, record)
            samples.append(record)
            previous = current
            reason = None
            if oom:
                reason = "child_reported_metal_oom"
            elif emergency_streak >= POLICY["emergency_consecutive_samples"]:
                reason = "emergency_low_free_memory"
            elif paging_streak >= POLICY["low_free_paging_consecutive_samples"]:
                reason = "low_free_memory_with_sustained_paging"
            elif critical_streak >= POLICY["critical_paging_consecutive_samples"]:
                reason = "critical_swapout_rate_with_pressure"
            if reason:
                state["watchdog_reason"] = reason
                abort.set()
                terminate_group(process)
                break
        state["memory_summary"] = memory_summary(samples)
        state["memory_summary"]["before"] = samples[0]["memory"] if samples else previous
        state["memory_summary"]["after"] = snapshot()
        append_jsonl(handle, {"event": "watchdog_stop", "reason": state.get("watchdog_reason"), "memory_after": state["memory_summary"]["after"]})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model")
    parser.add_argument("--mode", choices=("target", "mtp"), required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--supervisor-json", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8108)
    parser.add_argument("--timeout", type=float, default=1500.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    server_log = args.output_dir / "apc-server.log"
    memory_log = args.output_dir / "memory-apc.log"
    turn1_path = args.output_dir / "turn1.json"
    turn2_path = args.output_dir / "turn2.json"
    prefix = args.prompt.read_text(encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "APC_ENABLED": "1",
            "APC_TRACE": "1",
            "APC_DISK_PATH": str(args.output_dir / "apc-cache"),
            "APC_DISK_MAX_GB": "1",
            "UV_CACHE_DIR": ".uv-cache",
            "PYTHONUNBUFFERED": "1",
        }
    )
    command = [
        "uv",
        "run",
        "mlx_vlm.server",
        "--model",
        args.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--prefill-step-size",
        str(PREFILL_STEP_SIZE),
        "--log-level",
        "INFO",
    ]
    if args.mode == "mtp":
        if not args.draft_model:
            raise SystemExit("MTP APC requires --draft-model")
        command.extend(["--draft-model", args.draft_model, "--draft-kind", "mtp"])

    process = None
    state: dict[str, Any] = {"watchdog_reason": None, "memory_summary": {}}
    abort = threading.Event()
    monitor_thread: threading.Thread | None = None
    started = time.monotonic()
    try:
        with server_log.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[2], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        monitor_thread = threading.Thread(target=monitor_server, args=(process, memory_log, server_log, abort, state), daemon=True)
        monitor_thread.start()
        timeout = httpx.Timeout(900.0, connect=5.0, write=30.0, pool=30.0)
        with httpx.Client(base_url=f"http://127.0.0.1:{args.port}", timeout=timeout) as client:
            health1 = wait_health(client, min(args.timeout, 900.0), abort)
            messages1 = [{"role": "system", "content": prefix}, {"role": "user", "content": TURN1_USER}]
            first = request_turn(client, "turn1_cold", args.model, messages1, turn1_path, health1, abort)
            health2_value = response_json(client.get("/health"))
            messages2 = messages1 + [{"role": "assistant", "content": first["assistant_text"]}, {"role": "user", "content": TURN2_USER}]
            second = request_turn(client, "turn2_warm", args.model, messages2, turn2_path, health2_value if isinstance(health2_value, dict) else {}, abort)
        status = "complete"
        summary = {
            "status": status,
            "candidate_mode": args.mode,
            "model_id": args.model,
            "draft_model": args.draft_model if args.mode == "mtp" else None,
            "prompt_file": str(args.prompt),
            "prompt_sha256": digest(prefix),
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
            "turn1": first["client_measurements"],
            "turn2": second["client_measurements"],
            "server_command": command,
            "wall_seconds": time.monotonic() - started,
            "memory": state.get("memory_summary"),
            "watchdog_abort": state.get("watchdog_reason") is not None,
            "watchdog_reason": state.get("watchdog_reason"),
        }
    except Exception as exc:
        status = "failed_or_aborted"
        summary = {
            "status": status,
            "candidate_mode": args.mode,
            "model_id": args.model,
            "draft_model": args.draft_model if args.mode == "mtp" else None,
            "prompt_file": str(args.prompt),
            "prompt_sha256": digest(prefix),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "server_command": command,
            "wall_seconds": time.monotonic() - started,
            "memory": state.get("memory_summary"),
            "watchdog_abort": state.get("watchdog_reason") is not None,
            "watchdog_reason": state.get("watchdog_reason"),
        }
    finally:
        abort.set()
        if process is not None:
            terminate_group(process)
            process.wait()
        if monitor_thread is not None:
            monitor_thread.join(timeout=12.0)
        summary.setdefault("memory", state.get("memory_summary"))
        summary["watchdog_abort"] = state.get("watchdog_reason") is not None
        summary["watchdog_reason"] = state.get("watchdog_reason")
        write_json(args.result_json, summary)
        supervisor = {
            "label": f"{args.model}:{args.mode}:apc",
            "command": command,
            "child_returncode": process.returncode if process is not None else None,
            "watchdog_abort": state.get("watchdog_reason") is not None,
            "watchdog_reason": state.get("watchdog_reason"),
            "memory_summary": state.get("memory_summary"),
            "memory_log": str(memory_log),
            "child_log": str(server_log),
        }
        write_json(args.supervisor_json, supervisor)
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False), flush=True)
    return 0 if status == "complete" and not summary.get("watchdog_abort") else 2


if __name__ == "__main__":
    raise SystemExit(main())
