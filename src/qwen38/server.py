#!/usr/bin/env python3
"""Serialized OpenAI-compatible Qwen3.8 agent backend.

This server deliberately supports one active session and one inference at a
time.  It reuses the validated exact-suffix cache mechanics from
``qwen38.chat`` and implements the OpenAI Chat Completions surface a coding
agent needs: structured ``tools`` / assistant ``tool_calls`` / ``tool``
results rendered through the Qwen3.8 chat template, native tool-call parsing
of the model's output, and SSE streaming.

Generated token IDs remain the cache truth.  The parsed API message is a view
of them; a replayed assistant message is only used to recognize which stored
generation it stands for (see ``qwen38.protocol``).

The ``--self-test`` path exercises the protocol layer without importing MLX
or loading a model, which is useful on a memory-constrained machine.
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
from typing import Any, Callable, Sequence

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
    reconcile_canonical_history,
    response_replay_detected,
)
from qwen38.protocol import (
    CONTEXT_OVERFLOW_CODE,
    ParsedAssistant,
    ProtocolError,
    StreamAssembler,
    chunk_object,
    context_overflow_message,
    extends_transcript,
    normalize_messages,
    parse_assistant_text,
    parse_chat_request,
    sse_event,
    tool_call_deltas,
    tools_identity,
    usage_object,
)


MAX_BODY_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_CONTEXT_TOKENS = 16384
HEARTBEAT_SECONDS = 10.0

__all__ = ["AgentHTTPServer", "AgentSession", "ContextOverflow", "normalize_messages"]


class ContextOverflow(Exception):
    def __init__(self, limit: int, requested: int) -> None:
        super().__init__(context_overflow_message(limit, requested))
        self.limit = limit
        self.requested = requested


class ClientDisconnected(Exception):
    """The HTTP client went away while a response was being streamed."""


@dataclass
class GenerationRecord:
    prompt_tokens: int
    generated_ids: list[int]
    text: str
    finish_reason: str = "length"


@dataclass
class _Plan:
    messages: list[dict[str, Any]]
    canonical: dict[str, Any]
    cache_mode: str  # fresh_prefill | exact_reuse | fresh_fallback
    fallback_reason: str | None = None


@dataclass
class AgentSession:
    model: Any
    tokenizer: Any
    session_id: str
    system_prompt: str | None = None
    enable_thinking: bool = False
    max_kv_size: int | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    tools_key: str | None = None
    cached_token_ids: list[int] = field(default_factory=list)
    records: list[GenerationRecord] = field(default_factory=list)
    cache_exact_reuse: int = 0
    cache_fresh_fallback: int = 0
    cache_conflict_error: int = 0
    truncation_continuations: int = 0
    invalidations: int = 0

    def __post_init__(self) -> None:
        _, _, _, make_prompt_cache, _ = _require_mlx()
        self._max_kv_size = self.max_kv_size
        self.prompt_cache = make_prompt_cache(self.model, self.max_kv_size)

    def reset(self) -> None:
        self._clear_runtime_state()

    def invalidate(self) -> None:
        """Drop state that may no longer match what the client holds."""

        self.invalidations += 1
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
            "invalidations": self.invalidations,
        }

    def _clear_runtime_state(self) -> None:
        """Discard transcript/cache state while retaining cumulative counters."""

        _, _, _, make_prompt_cache, _ = _require_mlx()
        self.messages.clear()
        self.tools_key = None
        self.cached_token_ids.clear()
        self.records.clear()
        self.prompt_cache = make_prompt_cache(self.model, self._max_kv_size)

    def _with_server_system(self, incoming: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.system_prompt is None or incoming[0]["role"] == "system":
            return incoming
        return [{"role": "system", "content": self.system_prompt}, *incoming]

    def _render(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> dict[str, Any]:
        try:
            return canonical_prompt(
                self.tokenizer,
                messages,
                enable_thinking=self.enable_thinking,
                tools=tools,
            )
        except RuntimeError:
            raise
        except Exception as exc:  # jinja raise_exception / malformed structure
            raise ProtocolError(f"chat template rejected the conversation: {exc}") from exc

    def _plan(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tools_key: str | None,
    ) -> _Plan:
        """Decide exact reuse vs fresh prefill without touching any state."""

        if not self.messages:
            return _Plan(messages, self._render(messages, tools), "fresh_prefill")
        reason: str | None = None
        if tools_key != self.tools_key:
            reason = "tool_set_changed"
        elif not extends_transcript(self.messages, messages):
            reason = "transcript_does_not_extend_session"
        if reason is None:
            # Render the stored copies of already-served messages, so the
            # assistant spans come from this server's own API view.  They are
            # then replaced by the authoritative generated token IDs.
            render_messages = list(self.messages) + list(messages[len(self.messages) :])
            try:
                actual = _cache_offset(self.prompt_cache)
                if actual != len(self.cached_token_ids):
                    raise RuntimeError(
                        f"cache offset {actual} != logical prefix {len(self.cached_token_ids)}"
                    )
                canonical = self._render(render_messages, tools)
                history = [
                    HistoryTurn("", record.prompt_tokens, record.generated_ids, record.text)
                    for record in self.records
                ]
                canonical = reconcile_canonical_history(
                    canonical, history, eos_token_ids(self.tokenizer)
                )
                exact_suffix(canonical["token_ids"], self.cached_token_ids)
                return _Plan(render_messages, canonical, "exact_reuse")
            except RuntimeError as exc:
                reason = f"token_prefix_mismatch: {exc}"
        return _Plan(messages, self._render(messages, tools), "fresh_fallback", reason)

    def complete(
        self,
        incoming: list[dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        kv_bits: int | None,
        kv_group_size: int,
        quantized_kv_start: int,
        prefill_step_size: int,
        tools: list[dict[str, Any]] | None = None,
        max_context_tokens: int | None = None,
        before_generate: Callable[[dict[str, Any]], None] | None = None,
        on_delta: Callable[[str], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """Serve one assistant turn.

        Validation, prompt rendering and the context check happen before any
        session state changes.  ``before_generate(info)`` runs right before
        the model is touched (the HTTP layer sends stream headers there).
        ``on_delta`` receives final content text as it is generated.  If
        ``should_stop`` returns True, or any error occurs after generation
        starts, the session is discarded: the cache then holds tokens the
        client never accepted.
        """

        _, _, _, _, make_sampler = _require_mlx()
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not incoming:
            raise ValueError("request must contain at least one message")
        if incoming[-1]["role"] not in {"user", "tool"}:
            raise ValueError("request must end with a user or tool message")
        if incoming[-1]["role"] == "user" and not (incoming[-1].get("content") or "").strip():
            raise ValueError("last message content is empty")
        tools = tools or None
        tools_key = tools_identity(tools)
        messages = self._with_server_system(incoming)
        plan = self._plan(messages, tools, tools_key)
        prompt_tokens = len(plan.canonical["token_ids"])
        if max_context_tokens:
            if prompt_tokens >= max_context_tokens:
                raise ContextOverflow(max_context_tokens, prompt_tokens)
            max_tokens = min(max_tokens, max_context_tokens - prompt_tokens)
        if before_generate is not None:
            before_generate({"prompt_tokens": prompt_tokens, "cache_mode": plan.cache_mode})

        cache_mode = plan.cache_mode
        if cache_mode == "fresh_fallback":
            self._clear_runtime_state()
            self.cache_fresh_fallback += 1
        actual_before = _cache_offset(self.prompt_cache)
        if actual_before != len(self.cached_token_ids):
            raise RuntimeError(
                f"cache offset {actual_before} != logical prefix {len(self.cached_token_ids)}"
            )
        full_token_ids = plan.canonical["token_ids"]
        suffix_ids = exact_suffix(full_token_ids, self.cached_token_ids)
        cached_before = len(self.cached_token_ids)
        if cache_mode == "exact_reuse":
            self.cache_exact_reuse += 1
            if self.records and self.records[-1].finish_reason == "length":
                self.truncation_continuations += 1

        assembler = StreamAssembler(tools_active=bool(tools)) if on_delta else None

        def on_token(ids: list[int]) -> None:
            if should_stop is not None and should_stop():
                raise ClientDisconnected("client disconnected during generation")
            if assembler is not None:
                delta = assembler.feed(self.tokenizer.decode(ids, skip_special_tokens=True))
                if delta:
                    on_delta(delta)

        try:
            result = generate_tokens(
                self.model,
                self.prompt_cache,
                suffix_ids,
                max_tokens=max_tokens,
                sampler=make_sampler(temperature, top_p),
                eos_ids=eos_token_ids(self.tokenizer),
                prefill_step_size=prefill_step_size,
                kv_bits=kv_bits,
                kv_group_size=kv_group_size,
                quantized_kv_start=quantized_kv_start,
                on_token=on_token,
            )
        except BaseException:
            # The cache advanced past the committed ledger; it cannot be
            # trimmed (recurrent state), so it must be rebuilt next time.
            self.invalidate()
            raise
        generated_ids = result.token_ids
        cache_after = _cache_offset(self.prompt_cache)
        expected_after = len(full_token_ids) + len(generated_ids)
        if cache_after != expected_after:
            raise RuntimeError(
                f"cache after-generation offset {cache_after} != expected {expected_after}"
            )
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        parsed = parse_assistant_text(text, tools)
        if parsed.tool_calls:
            finish_reason = "tool_calls"
        else:
            finish_reason = "length" if result.stop == "length" else "stop"
        content_tail = assembler.finish(parsed) if assembler is not None else ""
        ttft = result.ttft_seconds
        prompt_tps = len(suffix_ids) / ttft if ttft else None
        memory = _memory_snapshot()
        replay_detected = any(
            response_replay_detected(generated_ids, record.generated_ids)
            for record in self.records
        )
        self.messages = list(plan.messages) + [parsed.stored_message()]
        self.tools_key = tools_key
        self.cached_token_ids = list(full_token_ids) + list(generated_ids)
        self.records.append(
            GenerationRecord(len(full_token_ids), list(generated_ids), text, finish_reason)
        )
        return {
            "text": text,
            "message": parsed.message(),
            "content_tail": content_tail,
            "tool_calls": parsed.tool_calls,
            "tool_parse_error": parsed.parse_error,
            "unknown_tools": parsed.unknown_tools,
            "argument_type_mismatches": parsed.argument_type_mismatches,
            "finish_reason": finish_reason,
            "generation_stop": result.stop,
            "prompt_tokens": len(full_token_ids),
            "processed_suffix_tokens": len(suffix_ids),
            "cached_tokens": cached_before,
            "generated_tokens": len(generated_ids),
            "max_tokens_effective": max_tokens,
            "tools_in_prompt": len(tools or ()),
            "ttft_seconds": ttft,
            "prompt_tps": prompt_tps,
            "decode_tps": result.decode_tps,
            "total_seconds": result.completed - result.started,
            "cache_offset_before": actual_before,
            "cache_offset_after": cache_after,
            "logical_context_tokens": len(self.cached_token_ids),
            "replay_detected": replay_detected,
            "cache_reuse_mode": cache_mode,
            "cache_exact_reuse": cache_mode == "exact_reuse",
            "cache_fresh_fallback": cache_mode == "fresh_fallback",
            "cache_fallback_reason": plan.fallback_reason,
            "cache_cached_prefix_tokens_before_request": cached_before,
            "canonical_cache_boundary_adjustments": plan.canonical.get(
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
        self.max_context_tokens = kwargs.get("max_context_tokens", DEFAULT_MAX_CONTEXT_TOKENS)
        self.default_mode = kwargs.get("default_mode", "greedy")
        self.default_temperature = kwargs.get("temperature")
        self.default_top_p = kwargs.get("top_p")
        self.prefill_step_size = kwargs.get("prefill_step_size", DEFAULT_PREFILL_STEP_SIZE)
        self.kv_bits = kwargs.get("kv_bits")
        self.kv_group_size = kwargs.get("kv_group_size", 64)
        self.quantized_kv_start = kwargs.get("quantized_kv_start", 0)
        self.heartbeat_seconds = kwargs.get("heartbeat_seconds", HEARTBEAT_SECONDS)
        self.metrics_log = kwargs.get("metrics_log")
        self.session: AgentSession | None = None
        self.session_lock = threading.Lock()
        self.request_count = 0
        self.counters: dict[str, int] = {
            "stream_requests": 0,
            "tool_call_turns": 0,
            "tool_calls_emitted": 0,
            "tool_parse_failures": 0,
            "unknown_tool_calls": 0,
            "argument_type_mismatches": 0,
            "cancelled_requests": 0,
            "context_overflow_rejections": 0,
            "http_4xx": 0,
            "http_5xx": 0,
            "stream_errors": 0,
        }

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

    def log_metrics(self, record: dict[str, Any]) -> None:
        if not self.metrics_log:
            return
        with open(self.metrics_log, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


class _StreamWriter:
    """Serialized SSE writes with heartbeat comments and disconnect tracking."""

    def __init__(self, handler: BaseHTTPRequestHandler, heartbeat_seconds: float) -> None:
        self.handler = handler
        self.lock = threading.Lock()
        self.disconnected = False
        self.last_write = time.monotonic()
        self._stop = threading.Event()
        self._heartbeat_seconds = heartbeat_seconds
        self._thread: threading.Thread | None = None

    def write(self, data: bytes) -> None:
        with self.lock:
            if self.disconnected:
                raise ClientDisconnected("client disconnected")
            try:
                self.handler.wfile.write(data)
                self.handler.wfile.flush()
            except OSError as exc:
                self.disconnected = True
                raise ClientDisconnected(f"client disconnected: {type(exc).__name__}") from exc
            self.last_write = time.monotonic()

    def start_heartbeat(self) -> None:
        def run() -> None:
            # SSE comments keep the client's idle timer alive during long
            # prefills; they also detect a closed connection early.
            while not self._stop.wait(1.0):
                if time.monotonic() - self.last_write >= self._heartbeat_seconds:
                    try:
                        self.write(b": keep-alive\n\n")
                    except ClientDisconnected:
                        return

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop_heartbeat(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)


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

    def _error(
        self,
        status: int,
        message: str,
        error_type: str = "invalid_request_error",
        code: str | None = None,
        param: str | None = None,
    ) -> None:
        if 400 <= status < 500:
            self.server.counters["http_4xx"] += 1
        elif status >= 500:
            self.server.counters["http_5xx"] += 1
        self._send_json(
            status,
            {"error": {"message": message, "type": error_type, "param": param, "code": code}},
        )

    def _body(self) -> Any:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > MAX_BODY_BYTES:
            raise ProtocolError("invalid Content-Length")
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProtocolError(f"request body is not valid JSON: {exc}") from exc

    def _session_id(self, body_session: str | None) -> str | None:
        if body_session is not None:
            return body_session
        value = self.headers.get("X-Qwen38-Session-Id")
        return value or None

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/")
        if path == "/v1/models":
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.server.model_name,
                            "object": "model",
                            "created": 0,
                            "owned_by": "qwen38",
                        }
                    ],
                    "status": "ready",
                },
            )
            return
        if path == "/health":
            self._send_json(200, {"object": "health", "data": None, "status": "ready"})
            return
        if path == "/metrics":
            session_stats = self.server.session.stats() if self.server.session else None
            self._send_json(
                200,
                {
                    "request_count": self.server.request_count,
                    "serialized": True,
                    **self.server.counters,
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
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._error(404, "unknown endpoint", "not_found")
            return
        self._chat_completions()

    def _sampling(self, request: Any) -> tuple[float, float, int]:
        mode = request.mode or self.server.default_mode
        temperature = request.temperature
        if temperature is None:
            temperature = self.server.default_temperature
        if temperature is None:
            temperature = 0.0 if mode == "greedy" else 0.7
        top_p = request.top_p
        if top_p is None:
            top_p = self.server.default_top_p
        if top_p is None:
            top_p = 1.0 if mode == "greedy" else 0.9
        max_tokens = request.max_tokens or self.server.default_max_tokens
        return float(temperature), float(top_p), int(max_tokens)

    def _chat_completions(self) -> None:
        server = self.server
        started = time.perf_counter()
        completion_id = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())
        log: dict[str, Any] = {
            "ts": time.time(),
            "request_index": server.request_count,
            "completion_id": completion_id,
        }
        writer: _StreamWriter | None = None
        headers_sent = False
        try:
            request = parse_chat_request(self._body())
            temperature, top_p, max_tokens = self._sampling(request)
            log.update(
                {
                    "stream": request.stream,
                    "message_count": len(request.messages),
                    "last_role": request.messages[-1]["role"],
                    "tools_declared": len(request.tools or ()),
                    "tool_choice": request.tool_choice,
                    "max_tokens_requested": max_tokens,
                }
            )

            def before_generate(_info: dict[str, Any]) -> None:
                nonlocal writer, headers_sent
                if not request.stream:
                    return
                # Headers go out before prefill so a long cold prefill does
                # not trip the client's response-header timeout.
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                headers_sent = True
                writer = _StreamWriter(self, server.heartbeat_seconds)
                writer.write(
                    sse_event(
                        chunk_object(
                            completion_id, created, server.model_name,
                            {"role": "assistant", "content": ""},
                        )
                    )
                )
                writer.start_heartbeat()

            def on_delta(text: str) -> None:
                assert writer is not None
                writer.write(
                    sse_event(chunk_object(completion_id, created, server.model_name, {"content": text}))
                )

            with server.session_lock:
                session = server.get_session(self._session_id(request.session_id))
                try:
                    metrics = session.complete(
                        request.messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        kv_bits=server.kv_bits,
                        kv_group_size=server.kv_group_size,
                        quantized_kv_start=server.quantized_kv_start,
                        prefill_step_size=server.prefill_step_size,
                        tools=request.effective_tools,
                        max_context_tokens=server.max_context_tokens or None,
                        before_generate=before_generate,
                        on_delta=on_delta if request.stream else None,
                        should_stop=(lambda: writer is not None and writer.disconnected)
                        if request.stream
                        else None,
                    )
                finally:
                    if writer is not None:
                        writer.stop_heartbeat()
                server.request_count += 1
                if request.stream:
                    try:
                        self._finish_stream(writer, request, completion_id, created, metrics)
                    except ClientDisconnected:
                        # The turn was committed but the client may not have
                        # received all of it: never reuse it.
                        session.invalidate()
                        raise
            self._count_tools(metrics)
            log.update(self._metrics_log_fields(metrics))
            log["http_status"] = 200
            if not request.stream:
                self._send_json(
                    200,
                    {
                        "id": completion_id,
                        "object": "chat.completion",
                        "created": created,
                        "model": server.model_name,
                        "session_id": session.session_id,
                        "choices": [
                            {
                                "index": 0,
                                "message": metrics["message"],
                                "logprobs": None,
                                "finish_reason": metrics["finish_reason"],
                            }
                        ],
                        "usage": usage_object(
                            metrics["prompt_tokens"], metrics["generated_tokens"], metrics["cached_tokens"]
                        ),
                        "x_qwen38_metrics": _public_metrics(metrics),
                    },
                )
        except ClientDisconnected as exc:
            server.counters["cancelled_requests"] += 1
            log.update({"http_status": 200, "cancelled": True, "error": str(exc)})
            self.close_connection = True
        except ContextOverflow as exc:
            server.counters["context_overflow_rejections"] += 1
            log.update({"http_status": 400, "error": str(exc), "prompt_tokens": exc.requested})
            self._error(400, str(exc), code=CONTEXT_OVERFLOW_CODE, param="messages")
        except (ProtocolError, ValueError) as exc:
            log.update({"http_status": 400, "error": str(exc)})
            if headers_sent:
                self._stream_error(writer, str(exc), "invalid_request_error")
            else:
                self._error(400, str(exc), param=getattr(exc, "param", None), code=getattr(exc, "code", None))
        except (RuntimeError, KeyError) as exc:
            log.update({"http_status": 409, "error": str(exc)})
            if server.session is not None:
                server.session.cache_conflict_error += 1
                server.session.reset()
            if headers_sent:
                self._stream_error(writer, str(exc), "session_state_error")
            else:
                self._error(409, str(exc), "session_state_error")
        except BaseException as exc:  # keep the local service alive after inference errors
            log.update({"http_status": 500, "error": f"{type(exc).__name__}: {exc}"})
            if server.session is not None:
                server.session.reset()
            message = f"inference failed: {type(exc).__name__}: {exc}"
            if headers_sent:
                self._stream_error(writer, message, "server_error")
            else:
                self._error(500, message, "server_error")
        finally:
            log["wall_seconds"] = time.perf_counter() - started
            try:
                server.log_metrics(log)
            except OSError:
                pass

    def _finish_stream(
        self,
        writer: _StreamWriter | None,
        request: Any,
        completion_id: str,
        created: int,
        metrics: dict[str, Any],
    ) -> None:
        assert writer is not None
        name = self.server.model_name
        if metrics["content_tail"]:
            writer.write(
                sse_event(chunk_object(completion_id, created, name, {"content": metrics["content_tail"]}))
            )
        for delta in tool_call_deltas(metrics["tool_calls"]):
            writer.write(sse_event(chunk_object(completion_id, created, name, delta)))
        writer.write(
            sse_event(chunk_object(completion_id, created, name, {}, metrics["finish_reason"]))
        )
        if request.include_usage:
            final = chunk_object(completion_id, created, name, None)
            final["usage"] = usage_object(
                metrics["prompt_tokens"], metrics["generated_tokens"], metrics["cached_tokens"]
            )
            final["x_qwen38_metrics"] = _public_metrics(metrics)
            writer.write(sse_event(final))
        writer.write(sse_event("[DONE]"))
        self.close_connection = True

    def _stream_error(self, writer: _StreamWriter | None, message: str, error_type: str) -> None:
        self.server.counters["stream_errors"] += 1
        self.close_connection = True
        if writer is None:
            return
        try:
            writer.write(sse_event({"error": {"message": message, "type": error_type}}))
        except ClientDisconnected:
            pass

    def _count_tools(self, metrics: dict[str, Any]) -> None:
        counters = self.server.counters
        if metrics["tool_calls"]:
            counters["tool_call_turns"] += 1
            counters["tool_calls_emitted"] += len(metrics["tool_calls"])
        if metrics["tool_parse_error"]:
            counters["tool_parse_failures"] += 1
        counters["unknown_tool_calls"] += len(metrics["unknown_tools"])
        counters["argument_type_mismatches"] += len(metrics["argument_type_mismatches"])

    def _metrics_log_fields(self, metrics: dict[str, Any]) -> dict[str, Any]:
        keys = (
            "finish_reason", "generation_stop", "prompt_tokens", "processed_suffix_tokens",
            "cached_tokens", "generated_tokens", "logical_context_tokens", "cache_reuse_mode",
            "cache_fallback_reason", "ttft_seconds", "decode_tps", "prompt_tps", "total_seconds",
            "replay_detected", "tool_parse_error", "unknown_tools", "argument_type_mismatches",
            "peak_memory_bytes", "active_plus_cache_bytes", "tools_in_prompt",
        )
        fields = {key: metrics.get(key) for key in keys}
        fields["tool_calls"] = [
            {"name": call["function"]["name"], "arguments": call["function"]["arguments"]}
            for call in metrics["tool_calls"]
        ]
        if self.server.session is not None:
            fields["session"] = self.server.session.stats()
        fields["counters"] = dict(self.server.counters)
        return fields


def _public_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    hidden = {"text", "message", "content_tail", "tool_calls"}
    return {key: value for key, value in metrics.items() if key not in hidden}


def protocol_self_test() -> dict[str, Any]:
    """Exercise the protocol layer without MLX, a tokenizer or a model."""

    from qwen38.protocol import parse_assistant_text as parse

    tools = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
                    "required": ["path"],
                },
            },
        }
    ]
    request = parse_chat_request(
        {
            "messages": [
                {"role": "system", "content": "Answer tersely."},
                {"role": "user", "content": [{"type": "text", "text": "Inspect file."}]},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "call_1", "type": "function",
                         "function": {"name": "read", "arguments": "{\"path\":\"a.py\"}"}}
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "print('hi')"},
            ],
            "tools": tools,
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_completion_tokens": 64,
        }
    )
    assert request.messages[2]["tool_calls"][0]["function"]["arguments"] == {"path": "a.py"}
    replay = [*request.messages[:2], {**request.messages[2], "content": ""}]
    assert extends_transcript(replay, request.messages)
    assert not extends_transcript(request.messages[:2], [request.messages[0], {"role": "user", "content": "x"}])
    generated = (
        "Reading it.\n\n<tool_call>\n<function=read>\n<parameter=path>\nsrc/café.py\n"
        "</parameter>\n<parameter=limit>\n20\n</parameter>\n</function>\n</tool_call>"
    )
    parsed = parse(generated, tools, new_id=lambda: "call_x")
    assert parsed.content == "Reading it."
    assert parsed.tool_calls[0]["function"]["arguments"] == {"path": "src/café.py", "limit": 20}
    malformed = parse("<tool_call>\n<function=read>\n<parameter=path>\nx", tools)
    assert malformed.parse_error and not malformed.tool_calls
    assembler = StreamAssembler(tools_active=True)
    streamed = "".join(assembler.feed(generated[:end]) for end in range(1, len(generated) + 1))
    streamed += assembler.finish(parsed)
    assert streamed == parsed.content
    return {
        "classification": "PROTOCOL_SELF_TEST_PASS",
        "normalized_message_count": len(request.messages),
        "roles": [item["role"] for item in request.messages],
        "structured_tool_calls": True,
        "parsed_tool_call": parsed.message()["tool_calls"][0]["function"],
        "malformed_tool_call_rejected": True,
        "stream_reassembles_content": True,
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Serialized OpenAI-compatible Qwen3.8 agent backend")
    p.add_argument("--model", help="MLX model directory")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model-name", default="qwen38-local")
    p.add_argument("--system-prompt", default=None)
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="default when a request sets no limit")
    p.add_argument(
        "--max-context-tokens",
        type=int,
        default=DEFAULT_MAX_CONTEXT_TOKENS,
        help="prompt+completion cap; larger prompts get a context_length_exceeded error (0 disables)",
    )
    p.add_argument("--mode", choices=("greedy", "sample"), default="greedy")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--prefill-step-size", type=int, default=DEFAULT_PREFILL_STEP_SIZE)
    p.add_argument("--max-kv-size", type=int, default=None)
    p.add_argument("--kv-bits", type=int, choices=(4, 8), default=None)
    p.add_argument("--kv-group-size", type=int, default=64)
    p.add_argument("--quantized-kv-start", type=int, default=0)
    p.add_argument("--metrics-log", default=None, help="append one JSON line per chat request")
    p.add_argument("--self-test", action="store_true", help="run protocol-only tests without MLX/model load")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.self_test:
        print(json.dumps(protocol_self_test(), indent=2, sort_keys=True, ensure_ascii=False))
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
        max_context_tokens=args.max_context_tokens,
        default_mode=args.mode,
        temperature=args.temperature,
        top_p=args.top_p,
        prefill_step_size=args.prefill_step_size,
        kv_bits=args.kv_bits,
        kv_group_size=args.kv_group_size,
        quantized_kv_start=args.quantized_kv_start,
        metrics_log=args.metrics_log,
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
