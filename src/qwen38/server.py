#!/usr/bin/env python3
"""Serialized OpenAI-compatible Qwen3.8 agent backend.

This server deliberately supports one active session and one inference at a
time.  It reuses the validated exact-suffix cache mechanics from
``qwen38.chat`` and accepts generic text content for system, user,
assistant, and tool messages.  It does not attempt to parse provider-specific
tool-call objects; callers may persist tool-call/result payloads as text.

The ``--self-test`` path exercises protocol normalization without importing
MLX or loading a model, which is useful on a memory-constrained machine.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Sequence

from qwen38.chat import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_PREFILL_STEP_SIZE,
    HistoryTurn,
    _cache_offset,
    _memory_snapshot,
    _require_mlx,
    canonical_prompt,
    exact_suffix,
    eos_token_ids,
    generate_tokens,
    normalize_cache_boundary,
    reconcile_canonical_history,
    response_replay_detected,
)


ALLOWED_ROLES = {"system", "user", "assistant", "tool"}
MAX_BODY_BYTES = 8 * 1024 * 1024


def content_to_text(content: Any, *, tool_calls: Any = None) -> str:
    """Normalize OpenAI text or content-part payloads to template-safe text."""

    if isinstance(content, str):
        return content
    if content is None and tool_calls is not None:
        return json.dumps(tool_calls, sort_keys=True, separators=(",", ":"))
    if content is None:
        return ""
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(json.dumps(item, sort_keys=True, separators=(",", ":")))
        return "".join(parts)
    return str(content)


def normalize_messages(raw_messages: Any) -> list[dict[str, str]]:
    """Validate and normalize the generic message surface exposed by the API."""

    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError("messages must be a non-empty list")
    normalized: list[dict[str, str]] = []
    for index, raw in enumerate(raw_messages):
        if not isinstance(raw, dict):
            raise ValueError(f"messages[{index}] must be an object")
        role = raw.get("role")
        if role not in ALLOWED_ROLES:
            raise ValueError(
                f"messages[{index}].role must be one of {sorted(ALLOWED_ROLES)}"
            )
        text = content_to_text(raw.get("content"), tool_calls=raw.get("tool_calls"))
        if not text and role in {"user", "tool"}:
            raise ValueError(f"messages[{index}] has empty {role} content")
        normalized.append({"role": role, "content": text})
    return normalized


def appended_messages(
    previous: Sequence[dict[str, str]],
    incoming: Sequence[dict[str, str]],
) -> list[dict[str, str]]:
    """Require the request transcript to extend the serialized session."""

    previous_list = list(previous)
    incoming_list = list(incoming)
    if not previous_list:
        return incoming_list
    if incoming_list[: len(previous_list)] != previous_list:
        raise RuntimeError(
            "request messages do not preserve the server session transcript; reset the session"
        )
    suffix = incoming_list[len(previous_list) :]
    if not suffix:
        raise ValueError("request must append a new user or tool message")
    if suffix[-1]["role"] == "assistant":
        raise ValueError("request must end with a user or tool message")
    return incoming_list


def protocol_self_test() -> dict[str, Any]:
    raw = [
        {"role": "system", "content": "Answer tersely."},
        {"role": "user", "content": [{"type": "text", "text": "Inspect file."}]},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_1", "function": {"name": "read_file"}}],
        },
        {"role": "tool", "content": "contents: hello.py"},
    ]
    first = normalize_messages(raw)
    extended = appended_messages(first, first + [{"role": "user", "content": "Summarize."}])
    assert extended[-1] == {"role": "user", "content": "Summarize."}
    try:
        appended_messages(first, [{"role": "user", "content": "stale"}])
    except RuntimeError:
        pass
    else:
        raise AssertionError("stale transcript was accepted")
    return {
        "classification": "PROTOCOL_SELF_TEST_PASS",
        "normalized_message_count": len(first),
        "roles": [item["role"] for item in first],
        "tool_call_content_is_text": first[2]["content"].startswith("[{")
        or first[2]["content"].startswith("{"),
        "extension_count": len(extended),
    }


@dataclass
class GenerationRecord:
    prompt_tokens: int
    generated_ids: list[int]
    text: str
    finish_reason: str = "length"


@dataclass
class AgentSession:
    model: Any
    tokenizer: Any
    session_id: str
    system_prompt: str | None = None
    enable_thinking: bool = False
    max_kv_size: int | None = None
    messages: list[dict[str, str]] = field(default_factory=list)
    cached_token_ids: list[int] = field(default_factory=list)
    records: list[GenerationRecord] = field(default_factory=list)
    cache_exact_reuse: int = 0
    cache_fresh_fallback: int = 0
    cache_conflict_error: int = 0
    truncation_continuations: int = 0

    def __post_init__(self) -> None:
        _, _, _, make_prompt_cache, _ = _require_mlx()
        self._max_kv_size = self.max_kv_size
        self.prompt_cache = make_prompt_cache(self.model, self.max_kv_size)

    def reset(self) -> None:
        self._clear_runtime_state()

    def stats(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "message_count": len(self.messages),
            "generation_count": len(self.records),
            "logical_context_tokens": len(self.cached_token_ids),
            "cached_prefix_tokens": len(self.cached_token_ids),
            "cache_layers": len(self.prompt_cache),
            "cache_exact_reuse": self.cache_exact_reuse,
            "cache_fresh_fallback": self.cache_fresh_fallback,
            "cache_conflict_error": self.cache_conflict_error,
            "truncation_continuations": self.truncation_continuations,
        }

    def _clear_runtime_state(self) -> None:
        """Discard transcript/cache state while retaining cumulative counters."""

        _, _, _, make_prompt_cache, _ = _require_mlx()
        self.messages.clear()
        self.cached_token_ids.clear()
        self.records.clear()
        self.prompt_cache = make_prompt_cache(self.model, self._max_kv_size)

    def _with_server_system(self, incoming: list[dict[str, str]]) -> list[dict[str, str]]:
        if self.system_prompt is None or incoming[0]["role"] == "system":
            return incoming
        return [{"role": "system", "content": self.system_prompt}, *incoming]

    def complete(
        self,
        incoming: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        kv_bits: int | None,
        kv_group_size: int,
        quantized_kv_start: int,
        prefill_step_size: int,
    ) -> dict[str, Any]:
        _, _, _, _, make_sampler = _require_mlx()
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not incoming:
            raise ValueError("request must contain at least one message")
        if incoming[-1]["role"] not in {"user", "tool"}:
            raise ValueError("request must end with a user or tool message")
        if not incoming[-1]["content"].strip():
            raise ValueError("last message content is empty")
        # Apply the configured system message before prefix validation on every
        # request.  This lets an OpenAI client omit the server-owned system
        # message while still preserving the exact cached transcript.
        messages = self._with_server_system(incoming)
        cache_mode = "fresh_prefill"
        if self.messages:
            try:
                messages = appended_messages(self.messages, messages)
                cache_mode = "exact_reuse"
                if self.records and self.records[-1].finish_reason == "length":
                    self.truncation_continuations += 1
            except RuntimeError:
                # The transcript is syntactically valid but no longer extends
                # the authoritative session.  It is safe to rebuild; it is
                # not safe to feed a guessed suffix into the old cache.
                self._clear_runtime_state()
                self.cache_fresh_fallback += 1
                cache_mode = "fresh_fallback"

        def fresh_fallback() -> int:
            nonlocal cache_mode
            self._clear_runtime_state()
            self.cache_fresh_fallback += 1
            cache_mode = "fresh_fallback"
            return _cache_offset(self.prompt_cache)

        try:
            actual_before = _cache_offset(self.prompt_cache)
        except RuntimeError:
            actual_before = fresh_fallback()
        if actual_before != len(self.cached_token_ids):
            if cache_mode == "exact_reuse":
                actual_before = fresh_fallback()
            else:
                raise RuntimeError(
                    f"cache offset {actual_before} != logical prefix {len(self.cached_token_ids)}"
                )

        canonical = canonical_prompt(
            self.tokenizer,
            messages,
            enable_thinking=self.enable_thinking,
        )
        history = [
            HistoryTurn("", record.prompt_tokens, record.generated_ids, record.text)
            for record in self.records
        ]
        eos_ids = eos_token_ids(self.tokenizer)
        if cache_mode == "exact_reuse":
            try:
                canonical = reconcile_canonical_history(canonical, history, eos_ids)
            except RuntimeError:
                actual_before = fresh_fallback()
                canonical = canonical_prompt(
                    self.tokenizer,
                    messages,
                    enable_thinking=self.enable_thinking,
                )
        else:
            canonical = normalize_cache_boundary(canonical, history, eos_ids)
        full_token_ids = canonical["token_ids"]
        try:
            suffix_ids = exact_suffix(full_token_ids, self.cached_token_ids)
        except RuntimeError:
            if cache_mode != "exact_reuse":
                raise
            actual_before = fresh_fallback()
            canonical = canonical_prompt(
                self.tokenizer,
                messages,
                enable_thinking=self.enable_thinking,
            )
            full_token_ids = canonical["token_ids"]
            suffix_ids = exact_suffix(full_token_ids, self.cached_token_ids)
        cached_before = list(self.cached_token_ids)
        lcp = 0
        while lcp < min(len(cached_before), len(full_token_ids)) and cached_before[lcp] == full_token_ids[lcp]:
            lcp += 1
        if cache_mode == "exact_reuse":
            self.cache_exact_reuse += 1
        result = generate_tokens(
            self.model,
            self.prompt_cache,
            suffix_ids,
            max_tokens=max_tokens,
            sampler=make_sampler(temperature, top_p),
            eos_ids=eos_ids,
            prefill_step_size=prefill_step_size,
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
            quantized_kv_start=quantized_kv_start,
        )
        generated_ids = result.token_ids
        finish_reason = "length" if result.stop == "length" else "stop"
        started, completed = result.started, result.completed
        cache_after = _cache_offset(self.prompt_cache)
        expected_after = len(full_token_ids) + len(generated_ids)
        if cache_after != expected_after:
            raise RuntimeError(
                f"cache after-generation offset {cache_after} != expected {expected_after}"
            )
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        ttft = result.ttft_seconds
        decode_tps = result.decode_tps
        prompt_tps = len(suffix_ids) / ttft if ttft else None
        memory = _memory_snapshot()
        replay_detected = any(
            response_replay_detected(generated_ids, record.generated_ids)
            for record in self.records
        )
        self.messages = messages + [{"role": "assistant", "content": text}]
        self.cached_token_ids = list(full_token_ids) + list(generated_ids)
        self.records.append(
            GenerationRecord(
                len(full_token_ids),
                list(generated_ids),
                text,
                finish_reason,
            )
        )
        return {
            "text": text,
            "finish_reason": finish_reason,
            "prompt_tokens": len(full_token_ids),
            "processed_suffix_tokens": len(suffix_ids),
            "generated_tokens": len(generated_ids),
            "ttft_seconds": ttft,
            "prompt_tps": prompt_tps,
            "decode_tps": decode_tps,
            "total_seconds": completed - started,
            "cache_offset_before": actual_before,
            "cache_offset_after": cache_after,
            "logical_context_tokens": len(self.cached_token_ids),
            "replay_detected": replay_detected,
            "cache_reuse_mode": cache_mode,
            "cache_exact_reuse": cache_mode == "exact_reuse",
            "cache_fresh_fallback": cache_mode == "fresh_fallback",
            "cache_lcp_tokens_before_request": lcp,
            "cache_cached_prefix_tokens_before_request": len(cached_before),
            "canonical_cache_boundary_adjustments": canonical.get(
                "cache_boundary_adjustments", []
            ),
            **memory,
        }


class AgentHTTPServer(HTTPServer):
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], **kwargs: Any) -> None:
        super().__init__(address, AgentRequestHandler)
        self.model = kwargs["model"]
        self.tokenizer = kwargs["tokenizer"]
        self.model_name = kwargs["model_name"]
        self.system_prompt = kwargs.get("system_prompt")
        self.enable_thinking = kwargs.get("enable_thinking", False)
        self.max_kv_size = kwargs.get("max_kv_size")
        self.default_max_tokens = kwargs.get("default_max_tokens", DEFAULT_MAX_TOKENS)
        self.default_mode = kwargs.get("default_mode", "greedy")
        self.default_temperature = kwargs.get("temperature")
        self.default_top_p = kwargs.get("top_p")
        self.prefill_step_size = kwargs.get("prefill_step_size", DEFAULT_PREFILL_STEP_SIZE)
        self.kv_bits = kwargs.get("kv_bits")
        self.kv_group_size = kwargs.get("kv_group_size", 64)
        self.quantized_kv_start = kwargs.get("quantized_kv_start", 0)
        self.session: AgentSession | None = None
        self.session_lock = threading.Lock()
        self.request_count = 0

    def get_session(self, requested_id: str | None) -> AgentSession:
        session_id = requested_id or (self.session.session_id if self.session else "default")
        if self.session is None or self.session.session_id != session_id:
            self.session = AgentSession(
                self.model,
                self.tokenizer,
                session_id=session_id,
                system_prompt=self.system_prompt,
                enable_thinking=self.enable_thinking,
                max_kv_size=self.max_kv_size,
            )
        return self.session

    def reset_session(self, requested_id: str | None) -> dict[str, Any]:
        with self.session_lock:
            if self.session is not None and (
                requested_id is None or self.session.session_id == requested_id
            ):
                self.session.reset()
                return self.session.stats()
            return {"session_id": requested_id or "default", "reset": True, "message_count": 0}


class AgentRequestHandler(BaseHTTPRequestHandler):
    server: AgentHTTPServer

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write("[qwen38-agent] " + (format % args) + "\n")

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str, error_type: str = "invalid_request_error") -> None:
        self._send_json(
            status,
            {"error": {"message": message, "type": error_type}},
        )

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > MAX_BODY_BYTES:
            raise ValueError("invalid Content-Length")
        raw = self.rfile.read(length)
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def _session_id(self, body: dict[str, Any] | None = None) -> str | None:
        if body is not None and body.get("session_id") is not None:
            return str(body["session_id"])
        value = self.headers.get("X-Qwen38-Session-Id")
        return value or None

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health" or self.path == "/v1/models":
            self._send_json(
                200,
                {
                    "object": "list" if self.path == "/v1/models" else "health",
                    "data": [{"id": self.server.model_name}] if self.path == "/v1/models" else None,
                    "status": "ready",
                },
            )
            return
        if self.path == "/metrics":
            session_stats = self.server.session.stats() if self.server.session else None
            self._send_json(
                200,
                {
                    "request_count": self.server.request_count,
                    "serialized": True,
                    "active_session": session_stats,
                },
            )
            return
        self._error(404, "unknown endpoint", "not_found")

    def do_POST(self) -> None:  # noqa: N802
        if self.path.startswith("/v1/sessions/") and self.path.endswith("/reset"):
            session_id = self.path[len("/v1/sessions/") : -len("/reset")].strip("/")
            self._send_json(200, {"session_id": session_id, "reset": True, "stats": self.server.reset_session(session_id)})
            return
        if self.path != "/v1/chat/completions":
            self._error(404, "unknown endpoint", "not_found")
            return
        try:
            body = self._body()
            if body.get("stream", False):
                raise ValueError("streaming responses are not implemented; use stream=false")
            incoming = normalize_messages(body.get("messages"))
            session_id = self._session_id(body)
            mode = str(body.get("mode", self.server.default_mode))
            if mode not in {"greedy", "sample"}:
                raise ValueError("mode must be greedy or sample")
            temperature = body.get("temperature", self.server.default_temperature)
            if temperature is None:
                temperature = 0.0 if mode == "greedy" else 0.7
            top_p = body.get("top_p", self.server.default_top_p)
            if top_p is None:
                top_p = 1.0 if mode == "greedy" else 0.9
            max_tokens = int(body.get("max_tokens", self.server.default_max_tokens))
            with self.server.session_lock:
                session = self.server.get_session(session_id)
                metrics = session.complete(
                    incoming,
                    max_tokens=max_tokens,
                    temperature=float(temperature),
                    top_p=float(top_p),
                    kv_bits=self.server.kv_bits,
                    kv_group_size=self.server.kv_group_size,
                    quantized_kv_start=self.server.quantized_kv_start,
                    prefill_step_size=self.server.prefill_step_size,
                )
                self.server.request_count += 1
            response_id = "chatcmpl-" + uuid.uuid4().hex[:20]
            self._send_json(
                200,
                {
                    "id": response_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": self.server.model_name,
                    "session_id": session.session_id,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": metrics.pop("text")},
                            "finish_reason": metrics["finish_reason"],
                        }
                    ],
                    "usage": {
                        "prompt_tokens": metrics["prompt_tokens"],
                        "completion_tokens": metrics["generated_tokens"],
                        "total_tokens": metrics["prompt_tokens"] + metrics["generated_tokens"],
                    },
                    "x_qwen38_metrics": metrics,
                },
            )
        except ValueError as exc:
            self._error(400, str(exc))
        except (RuntimeError, KeyError) as exc:
            if self.server.session is not None:
                with self.server.session_lock:
                    self.server.session.cache_conflict_error += 1
                    self.server.session.reset()
            self._error(409, str(exc), "session_state_error")
        except BaseException as exc:  # keep the local service alive after inference errors
            if self.server.session is not None:
                with self.server.session_lock:
                    self.server.session.reset()
            self._error(500, f"inference failed: {type(exc).__name__}: {exc}", "server_error")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Serialized OpenAI-compatible Qwen3.8 agent backend")
    p.add_argument("--model", help="MLX model directory")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model-name", default="qwen38-local")
    p.add_argument("--system-prompt", default=None)
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--mode", choices=("greedy", "sample"), default="greedy")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--prefill-step-size", type=int, default=DEFAULT_PREFILL_STEP_SIZE)
    p.add_argument("--max-kv-size", type=int, default=None)
    p.add_argument("--kv-bits", type=int, choices=(4, 8), default=None)
    p.add_argument("--kv-group-size", type=int, default=64)
    p.add_argument("--quantized-kv-start", type=int, default=0)
    p.add_argument("--self-test", action="store_true", help="run protocol-only tests without MLX/model load")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.self_test:
        print(json.dumps(protocol_self_test(), indent=2, sort_keys=True))
        return 0
    if not args.model:
        parser().error("--model is required unless --self-test is used")
    print("[qwen38-agent] loading model; one serialized session; wired-memory settings untouched", file=sys.stderr, flush=True)
    _, mlx_load, _, _, _ = _require_mlx()
    model, tokenizer = mlx_load(args.model)
    server = AgentHTTPServer(
        (args.host, args.port),
        model=model,
        tokenizer=tokenizer,
        model_name=args.model_name,
        system_prompt=args.system_prompt,
        enable_thinking=args.enable_thinking,
        max_kv_size=args.max_kv_size,
        default_max_tokens=args.max_tokens,
        default_mode=args.mode,
        temperature=args.temperature,
        top_p=args.top_p,
        prefill_step_size=args.prefill_step_size,
        kv_bits=args.kv_bits,
        kv_group_size=args.kv_group_size,
        quantized_kv_start=args.quantized_kv_start,
    )
    print(f"[qwen38-agent] listening on http://{args.host}:{args.port}", file=sys.stderr, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if server.session is not None:
            server.session.reset()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
