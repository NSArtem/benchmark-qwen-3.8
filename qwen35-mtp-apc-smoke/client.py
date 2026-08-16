#!/usr/bin/env python3
"""Run the cold/warm APC smoke test against an mlx_vlm OpenAI server."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx


TARGET_MODEL = "mlx-community/Qwen3.5-9B-4bit"
TENANT_ID = "qwen35-apc-smoke"
MAX_TOKENS = 192
TEMPERATURE = 0.0
BASE_HEADERS = {"X-APC-Tenant": TENANT_ID}

TURN1_USER = (
    "Inspect the Lantern incident described above. Give a concise diagnosis of the "
    "two most likely root causes and name the specific files/functions to change. "
    "Keep it to about 150-200 tokens; do not provide the full patch yet."
)
TURN2_USER = (
    "Good. Now give the minimal patch strategy and identify the tests that should be "
    "updated. Keep the answer concise."
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_capture(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def system_memory_snapshot() -> dict[str, Any]:
    """Take a small before/after snapshot; this is not a continuous monitor."""

    swap = run_capture(["sysctl", "-n", "vm.swapusage"])
    pressure = run_capture(["memory_pressure", "-Q"])
    vm_stat = run_capture(["vm_stat"])
    pageouts = None
    swapouts = None
    if vm_stat:
        pageouts_match = re.search(r"Pageouts:\s+(\d+)", vm_stat)
        swapouts_match = re.search(r"Swapouts:\s+(\d+)", vm_stat)
        if pageouts_match:
            pageouts = int(pageouts_match.group(1))
        if swapouts_match:
            swapouts = int(swapouts_match.group(1))
    free_percent = None
    if pressure:
        free_match = re.search(r"free percentage:\s*([0-9.]+)%", pressure, re.IGNORECASE)
        if free_match:
            free_percent = float(free_match.group(1))
    return {
        "swapusage": swap,
        "memory_pressure_free_percent": free_percent,
        "pageouts": pageouts,
        "swapouts": swapouts,
    }


def json_response(response: httpx.Response) -> Any:
    try:
        return response.json()
    except (ValueError, json.JSONDecodeError):
        return {"_non_json_body": response.text}


def get_json(client: httpx.Client, path: str) -> Any:
    response = client.get(path)
    response.raise_for_status()
    return json_response(response)


def wait_for_health(client: httpx.Client, timeout_seconds: float = 900.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_health: dict[str, Any] = {}
    while time.monotonic() < deadline:
        try:
            response = client.get("/health")
            if response.status_code == 200:
                value = json_response(response)
                if isinstance(value, dict):
                    last_health = value
                    if value.get("apc_enabled") is True:
                        return value
        except httpx.HTTPError:
            pass
        time.sleep(2.0)
    raise TimeoutError(f"server did not become APC-ready; last health={last_health}")


def latest_metrics(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    latest = value.get("latest")
    if isinstance(latest, dict):
        return latest
    return value


def number(value: Any) -> float | int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    return None


def first_number(*values: Any) -> float | int | None:
    for value in values:
        parsed = number(value)
        if parsed is not None:
            return parsed
    return None


def extract_assistant_text(api_response: dict[str, Any]) -> str:
    choices = api_response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content", "") if isinstance(message, dict) else ""
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False)


def request_turn(
    *,
    label: str,
    messages: list[dict[str, str]],
    output_path: Path,
    client: httpx.Client,
    health_before: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "model": TARGET_MODEL,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": 1.0,
        "seed": 0,
        "stream": False,
        "enable_thinking": False,
    }
    memory_before = system_memory_snapshot()
    started = time.perf_counter()
    response: httpx.Response | None = None
    api_response: Any
    request_error: str | None = None
    try:
        response = client.post("/v1/chat/completions", headers=BASE_HEADERS, json=payload)
        api_response = json_response(response)
    except httpx.HTTPError as exc:
        api_response = {"error": str(exc)}
        request_error = str(exc)
    wall_seconds = time.perf_counter() - started
    memory_after = system_memory_snapshot()

    metrics_payload: Any = None
    cache_stats: Any = None
    metrics_error: str | None = None
    try:
        metrics_payload = get_json(client, "/metrics")
        cache_stats = get_json(client, "/v1/cache/stats")
    except httpx.HTTPError as exc:
        metrics_error = str(exc)

    latest = latest_metrics(metrics_payload)
    usage = api_response.get("usage", {}) if isinstance(api_response, dict) else {}
    prompt_details = usage.get("prompt_tokens_details", {}) if isinstance(usage, dict) else {}
    timings = api_response.get("timings", {}) if isinstance(api_response, dict) else {}
    if not isinstance(timings, dict):
        timings = {}
    prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
    cached_tokens = prompt_details.get("cached_tokens") if isinstance(prompt_details, dict) else None
    completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
    prompt_tokens = number(prompt_tokens)
    cached_tokens = number(cached_tokens)
    completion_tokens = number(completion_tokens)
    cache_hit_fraction = (
        float(cached_tokens) / float(prompt_tokens)
        if cached_tokens is not None and prompt_tokens not in (None, 0)
        else None
    )
    prompt_seconds = first_number(
        latest.get("prompt_eval_time_s"),
        (float(timings["prompt_ms"]) / 1000.0 if number(timings.get("prompt_ms")) is not None else None),
    )
    ttft_seconds = first_number(latest.get("ttft_s"))
    decode_tok_s = first_number(
        timings.get("predicted_per_second"),
        latest.get("decode_tok_s"),
    )
    prefill_tok_s = first_number(
        latest.get("prefill_tok_s"),
        timings.get("prompt_per_second"),
    )
    peak_memory_gb = first_number(
        timings.get("peak_memory"),
        latest.get("peak_memory_gb"),
    )
    draft_rounds = first_number(timings.get("draft_rounds"), latest.get("draft_rounds"))
    drafted_tokens = first_number(timings.get("draft_n"), latest.get("draft_n"))
    accepted_tokens = first_number(
        timings.get("draft_n_accepted"), latest.get("draft_n_accepted")
    )
    acceptance_rate = (
        float(accepted_tokens) / float(drafted_tokens)
        if accepted_tokens is not None and drafted_tokens not in (None, 0)
        else None
    )
    assistant_text = extract_assistant_text(api_response) if isinstance(api_response, dict) else ""
    measurements = {
        "label": label,
        "http_status": response.status_code if response is not None else None,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "cache_hit_fraction": cache_hit_fraction,
        "completion_tokens": completion_tokens,
        "wall_seconds": wall_seconds,
        "ttft_seconds": ttft_seconds,
        "prefill_seconds": prompt_seconds,
        "prefill_tok_s": prefill_tok_s,
        "decode_tok_s": decode_tok_s,
        "peak_memory_gb": peak_memory_gb,
        "draft_kind": timings.get("draft_kind", latest.get("draft_kind")),
        "draft_rounds": draft_rounds,
        "drafted_tokens": drafted_tokens,
        "accepted_tokens": accepted_tokens,
        "acceptance_rate": acceptance_rate,
        "assistant_text_chars": len(assistant_text),
        "assistant_text_sha256": hashlib.sha256(assistant_text.encode("utf-8")).hexdigest(),
        "request_error": request_error,
        "metrics_error": metrics_error,
    }
    record = {
        "turn": label,
        "request": payload,
        "api_response": api_response,
        "client_measurements": measurements,
        "server_metrics_latest": latest,
        "server_metrics_response": metrics_payload,
        "server_cache_stats": cache_stats,
        "health_before": health_before,
        "memory_before": memory_before,
        "memory_after": memory_after,
    }
    write_json(output_path, record)
    if request_error or (response is not None and response.status_code >= 400):
        raise RuntimeError(f"{label} failed; response saved to {output_path}")
    return {**record, "assistant_text": assistant_text}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8099")
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--turn1-output", type=Path, required=True)
    parser.add_argument("--turn2-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prefix = args.prompt.read_text(encoding="utf-8")
    common = [{"role": "system", "content": prefix}]
    messages1 = common + [{"role": "user", "content": TURN1_USER}]
    timeout = httpx.Timeout(900.0, connect=5.0, write=30.0, pool=30.0)
    with httpx.Client(base_url=args.base_url.rstrip("/"), timeout=timeout) as client:
        health1 = wait_for_health(client)
        turn1 = request_turn(
            label="turn1_cold",
            messages=messages1,
            output_path=args.turn1_output,
            client=client,
            health_before=health1,
        )
        assistant_text = turn1["assistant_text"]
        messages2 = messages1 + [
            {"role": "assistant", "content": assistant_text},
            {"role": "user", "content": TURN2_USER},
        ]
        health2 = get_json(client, "/health")
        turn2 = request_turn(
            label="turn2_warm",
            messages=messages2,
            output_path=args.turn2_output,
            client=client,
            health_before=health2,
        )

    summary = {
        "model": TARGET_MODEL,
        "draft_model": "mlx-community/Qwen3.5-9B-MTP-4bit",
        "draft_kind": "mtp",
        "prompt_file": str(args.prompt),
        "prefix_chars": len(prefix),
        "prefix_bytes": len(prefix.encode("utf-8")),
        "prefix_sha256": hashlib.sha256(prefix.encode("utf-8")).hexdigest(),
        "request_config": {
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": 1.0,
            "seed": 0,
            "enable_thinking": False,
            "stream": False,
            "tenant_header": TENANT_ID,
        },
        "health_ready": health1,
        "turn1": turn1["client_measurements"],
        "turn2": turn2["client_measurements"],
        "turn2_prompt_messages": len(messages2),
    }
    write_json(args.summary_output, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
