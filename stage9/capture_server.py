#!/usr/bin/env python3
"""Temporary Stage 9 request-capture endpoint (no MLX, no model).

Records every request (headers with credentials redacted + JSON body) to a
JSONL file and answers with tiny, standards-shaped OpenAI chat completion
streams:

- last message is a user message containing ``CAPTURE_TOOL`` and tools are
  declared -> one streamed ``read`` tool call (path ``hello.txt``);
- last message is a user message containing ``CAPTURE_TWO`` -> two streamed
  tool calls in one turn;
- last message is a user message containing ``CAPTURE_SLOW`` -> a slow text
  stream, used to observe client cancellation;
- last message is a tool result -> a short streamed text answer;
- otherwise -> streamed text ``pong``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

LOG_PATH = "capture.jsonl"


def _chunk(cid: str, delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
    return {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "qwen38-local",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[capture] " + (fmt % args) + "\n")

    def _record(self, body: Any) -> None:
        headers = {
            k: ("<redacted>" if k.lower() in {"authorization", "api-key"} else v)
            for k, v in self.headers.items()
        }
        with open(LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {"t": time.time(), "method": self.command, "path": self.path, "headers": headers, "body": body},
                    ensure_ascii=False,
                )
                + "\n"
            )

    def _json(self, status: int, payload: Any) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        self._record(None)
        if self.path.rstrip("/").endswith("/models"):
            self._json(200, {"object": "list", "data": [{"id": "qwen38-local", "object": "model", "owned_by": "local"}]})
        else:
            self._json(404, {"error": {"message": "unknown", "type": "not_found"}})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw or b"null")
        except json.JSONDecodeError:
            body = {"_raw": raw.decode("utf-8", "replace")}
        self._record(body)
        if not self.path.endswith("/chat/completions"):
            self._json(404, {"error": {"message": "unknown", "type": "not_found"}})
            return
        messages = body.get("messages") or []
        last = messages[-1] if messages else {}
        text = last.get("content") if isinstance(last.get("content"), str) else json.dumps(last.get("content"))
        cid = "chatcmpl-capture%d" % int(time.time() * 1000)
        events: list[dict[str, Any]] = []
        delay = 0.0
        finish = "stop"
        if last.get("role") == "user" and "CAPTURE_TWO" in text and body.get("tools"):
            events.append(_chunk(cid, {"role": "assistant", "content": "Reading both."}))
            for index, path in enumerate(("hello.txt", "world.txt")):
                events.append(_chunk(cid, {"tool_calls": [{"index": index, "id": f"call_cap{index}", "type": "function", "function": {"name": "read", "arguments": ""}}]}))
                events.append(_chunk(cid, {"tool_calls": [{"index": index, "function": {"arguments": '{"path": '}}]}))
                events.append(_chunk(cid, {"tool_calls": [{"index": index, "function": {"arguments": json.dumps(path) + "}"}}]}))
            finish = "tool_calls"
        elif last.get("role") == "user" and "CAPTURE_TOOL" in text and body.get("tools"):
            events.append(_chunk(cid, {"role": "assistant", "content": None, "tool_calls": [{"index": 0, "id": "call_cap0", "type": "function", "function": {"name": "read", "arguments": ""}}]}))
            events.append(_chunk(cid, {"tool_calls": [{"index": 0, "function": {"arguments": '{"path": "hel'}}]}))
            events.append(_chunk(cid, {"tool_calls": [{"index": 0, "function": {"arguments": 'lo.txt"}'}}]}))
            finish = "tool_calls"
        elif last.get("role") == "user" and "CAPTURE_SLOW" in text:
            events.append(_chunk(cid, {"role": "assistant", "content": ""}))
            for i in range(60):
                events.append(_chunk(cid, {"content": f"tick{i} "}))
            delay = 0.5
        elif last.get("role") == "tool":
            events.append(_chunk(cid, {"role": "assistant", "content": "The file says: "}))
            events.append(_chunk(cid, {"content": "café \U0001f600 done."}))
        else:
            events.append(_chunk(cid, {"role": "assistant", "content": "po"}))
            events.append(_chunk(cid, {"content": "ng"}))
        events.append(_chunk(cid, {}, finish))
        usage = {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            for event in events:
                self.wfile.write(b"data: " + json.dumps(event, ensure_ascii=False).encode() + b"\n\n")
                self.wfile.flush()
                if delay:
                    time.sleep(delay)
            if (body.get("stream_options") or {}).get("include_usage"):
                tail = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()), "model": "qwen38-local", "choices": [], "usage": usage}
                self.wfile.write(b"data: " + json.dumps(tail).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError) as exc:
            with open(LOG_PATH, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"t": time.time(), "event": "client_disconnect", "error": type(exc).__name__}) + "\n")
        self.close_connection = True


def main() -> int:
    global LOG_PATH
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--log", default="capture.jsonl")
    args = parser.parse_args()
    LOG_PATH = args.log
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    sys.stderr.write(f"[capture] listening on {args.port}, logging to {LOG_PATH}\n")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
