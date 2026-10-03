"""HTTP/SSE tests for /v1/chat/completions with the real tokenizer and a fake generator."""

from __future__ import annotations

import http.client
import json
import re
import socket
import threading
import time
import unittest
from typing import Any

import qwen38.server as agent_server

from fake_mlx import FakeMLX, FakeModel, encode, load_tokenizer, requires_tokenizer

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a file",
            "parameters": {"type": "object", "required": ["path"], "properties": {"path": {"type": "string"}, "limit": {"type": "number"}}},
        },
    }
]

# Pi 1.0.0 classifies this as context overflow (pi-ai utils/overflow.js).
PI_OVERFLOW_RE = re.compile(r"maximum context length is \d+ tokens", re.I)


def call_text(name: str, **params: Any) -> str:
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


def parse_sse(raw: bytes) -> tuple[list[Any], list[str]]:
    events: list[Any] = []
    comments: list[str] = []
    for block in raw.decode("utf-8").split("\n\n"):
        if not block.strip():
            continue
        if block.startswith(":"):
            comments.append(block)
            continue
        assert block.startswith("data: "), block
        data = block[len("data: ") :]
        events.append(data if data == "[DONE]" else json.loads(data))
    return events, comments


def assemble(events: list[Any]) -> dict[str, Any]:
    """Assemble chunks the way an OpenAI client (and Pi) does."""

    content = ""
    calls: dict[int, dict[str, Any]] = {}
    finish = None
    usage = None
    ids = set()
    for event in events:
        if event == "[DONE]":
            continue
        ids.add(event["id"])
        assert event["object"] == "chat.completion.chunk"
        if event.get("usage"):
            usage = event["usage"]
        for choice in event["choices"]:
            delta = choice["delta"]
            content += delta.get("content") or ""
            for call in delta.get("tool_calls") or []:
                slot = calls.setdefault(call["index"], {"id": None, "name": "", "arguments": ""})
                slot["id"] = call.get("id") or slot["id"]
                function = call.get("function") or {}
                slot["name"] += function.get("name") or ""
                slot["arguments"] += function.get("arguments") or ""
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    assert len(ids) == 1
    return {"content": content, "tool_calls": [calls[i] for i in sorted(calls)], "finish_reason": finish, "usage": usage}


@requires_tokenizer
class StreamingHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tok = load_tokenizer()

    def setUp(self) -> None:
        self.fake = FakeMLX()
        self.fake.install()
        self.model = FakeModel()
        self.server = agent_server.AgentHTTPServer(
            ("127.0.0.1", 0),
            model=self.model,
            tokenizer=self.tok,
            model_name="qwen38-test",
            default_max_tokens=256,
            max_context_tokens=4096,
            heartbeat_seconds=0.5,
        )
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.fake.uninstall()

    def queue(self, text: str, eos: bool = True) -> None:
        self.model.responses.append(encode(self.tok, text, eos))

    def post(self, body: dict[str, Any]) -> tuple[int, dict[str, str], bytes]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        payload = json.dumps(body).encode()
        conn.request("POST", "/v1/chat/completions", payload, {"Content-Type": "application/json"})
        response = conn.getresponse()
        data = response.read()
        headers = {k.lower(): v for k, v in response.getheaders()}
        conn.close()
        return response.status, headers, data

    def stream(self, body: dict[str, Any]) -> tuple[list[Any], list[str], dict[str, str]]:
        status, headers, data = self.post({**body, "stream": True, "stream_options": {"include_usage": True}})
        self.assertEqual(status, 200, data)
        self.assertTrue(headers["content-type"].startswith("text/event-stream"))
        events, comments = parse_sse(data)
        return events, comments, headers

    def metrics(self) -> dict[str, Any]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request("GET", "/metrics")
        value = json.loads(conn.getresponse().read())
        conn.close()
        return value

    def test_models_endpoint(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request("GET", "/v1/models")
        body = json.loads(conn.getresponse().read())
        self.assertEqual(body["object"], "list")
        self.assertEqual(body["data"][0]["id"], "qwen38-test")
        self.assertEqual(body["data"][0]["object"], "model")

    def test_streamed_text_matches_non_streamed(self) -> None:
        text = "Hello café \U0001f600!\nSecond line.  "
        self.queue(text)
        status, _, data = self.post({"messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 200)
        plain = json.loads(data)
        self.queue(text)
        events, _, _ = self.stream({"messages": [{"role": "user", "content": "hi"}], "session_id": "other"})
        self.assertEqual(events[0]["choices"][0]["delta"], {"role": "assistant", "content": ""})
        self.assertEqual(events[-1], "[DONE]")
        self.assertEqual(events[-2]["choices"], [])
        streamed = assemble(events)
        self.assertEqual(streamed["content"], plain["choices"][0]["message"]["content"])
        self.assertEqual(streamed["content"], text)
        self.assertEqual(streamed["finish_reason"], "stop")
        self.assertEqual(streamed["usage"]["completion_tokens"], plain["usage"]["completion_tokens"])
        content_events = [e for e in events[:-1] if e["choices"] and e["choices"][0]["delta"].get("content")]
        self.assertGreater(len(content_events), 3)
        self.assertTrue(all("�" not in e["choices"][0]["delta"]["content"] for e in content_events))

    def test_streamed_tool_call_round_trip(self) -> None:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": [{"type": "text", "text": "read a.py"}]},
        ]
        self.queue("Reading.\n\n" + call_text("read", path="a.py", limit=5))
        events, _, _ = self.stream({"messages": messages, "tools": TOOLS})
        first = assemble(events)
        self.assertEqual(first["finish_reason"], "tool_calls")
        self.assertEqual(first["content"], "Reading.")
        self.assertEqual(len(first["tool_calls"]), 1)
        call = first["tool_calls"][0]
        self.assertTrue(call["id"].startswith("call_"))
        self.assertEqual(call["name"], "read")
        self.assertEqual(json.loads(call["arguments"]), {"path": "a.py", "limit": 5})
        self.assertEqual(first["usage"]["prompt_tokens_details"]["cached_tokens"], 0)

        # Replay exactly as Pi does: re-serialized arguments, tool result with id.
        messages += [
            {"role": "assistant", "content": first["content"], "tool_calls": [
                {"id": call["id"], "type": "function", "function": {"name": "read", "arguments": json.dumps(json.loads(call["arguments"]), separators=(",", ":"))}}
            ]},
            {"role": "tool", "content": "print('a')\n", "tool_call_id": call["id"]},
        ]
        self.queue("a.py prints a.")
        events, _, _ = self.stream({"messages": messages, "tools": TOOLS})
        second = assemble(events)
        self.assertEqual(second["finish_reason"], "stop")
        self.assertEqual(second["content"], "a.py prints a.")
        final_metrics = events[-2]["x_qwen38_metrics"]
        self.assertEqual(final_metrics["cache_reuse_mode"], "exact_reuse")
        self.assertGreater(second["usage"]["prompt_tokens_details"]["cached_tokens"], 0)
        counters = self.metrics()
        self.assertEqual((counters["tool_call_turns"], counters["tool_calls_emitted"], counters["http_4xx"]), (1, 1, 0))
        self.assertEqual(counters["active_session"]["cache_exact_reuse"], 1)

    def test_heartbeat_during_slow_prefill(self) -> None:
        self.queue("late")
        self.model.prefill_delay = 1.6
        events, comments, _ = self.stream({"messages": [{"role": "user", "content": "x"}]})
        self.assertTrue(any(c.startswith(": keep-alive") for c in comments))
        self.assertEqual(assemble(events)["content"], "late")

    def test_context_overflow_error_is_recognizable(self) -> None:
        big = "word " * 5000
        status, _, data = self.post({"messages": [{"role": "user", "content": big}], "stream": True})
        self.assertEqual(status, 400)
        error = json.loads(data)["error"]
        self.assertEqual(error["code"], "context_length_exceeded")
        self.assertRegex(error["message"], PI_OVERFLOW_RE)
        self.assertEqual(self.metrics()["context_overflow_rejections"], 1)

    def test_invalid_requests_get_400_before_streaming(self) -> None:
        for body in (
            {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]},
            {"messages": [{"role": "user", "content": "x"}], "tools": TOOLS, "tool_choice": "required"},
            {"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]},
        ):
            status, headers, data = self.post({**body, "stream": True})
            self.assertEqual(status, 400, data)
            self.assertTrue(headers["content-type"].startswith("application/json"))
            self.assertIn("message", json.loads(data)["error"])

    def test_error_after_headers_is_an_sse_error_event(self) -> None:
        self.queue("this will fail mid way through")
        self.model.fail_after = 2
        events, _, _ = self.stream({"messages": [{"role": "user", "content": "x"}]})
        self.assertIn("error", events[-1])
        self.assertEqual(events[-1]["error"]["type"], "server_error")
        self.assertNotIn("[DONE]", events)
        self.assertEqual(self.metrics()["active_session"]["logical_context_tokens"], 0)

    def test_client_disconnect_stops_generation_and_discards_cache(self) -> None:
        self.queue("Good answer.")
        status, _, data = self.post({"messages": [{"role": "user", "content": "first"}]})
        first = json.loads(data)["choices"][0]["message"]
        self.model.responses.append(encode(self.tok, "tick " * 400))
        self.model.token_delay = 0.01
        body = json.dumps({
            "messages": [{"role": "user", "content": "first"}, first, {"role": "user", "content": "go"}],
            "stream": True,
        }).encode()
        sock = socket.create_connection(("127.0.0.1", self.port), timeout=10)
        sock.sendall(
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
            + f"Content-Length: {len(body)}\r\n\r\n".encode() + body
        )
        received = b""
        while received.count(b"tick") < 3:
            received += sock.recv(4096)
        sock.close()
        deadline = time.time() + 10
        while time.time() < deadline and self.metrics()["cancelled_requests"] == 0:
            time.sleep(0.05)
        counters = self.metrics()
        self.assertEqual(counters["cancelled_requests"], 1)
        self.assertLess(self.model.tokens_yielded, 300)
        self.assertEqual(counters["active_session"]["logical_context_tokens"], 0)
        self.assertEqual(counters["active_session"]["invalidations"], 1)
        # The client never got the partial turn; the next request is rebuilt.
        self.model.token_delay = 0.0
        self.queue("Recovered.")
        status, _, data = self.post({"messages": [{"role": "user", "content": "first"}, first, {"role": "user", "content": "go"}]})
        self.assertEqual(status, 200)
        reply = json.loads(data)
        self.assertEqual(reply["x_qwen38_metrics"]["cache_reuse_mode"], "fresh_prefill")
        self.assertEqual(reply["choices"][0]["message"]["content"], "Recovered.")

    def test_length_truncated_stream_then_continuation(self) -> None:
        messages = [{"role": "user", "content": "list"}]
        self.model.responses.append(encode(self.tok, "Step one:\nStep two:\n", eos=False))
        events, _, _ = self.stream({"messages": messages, "max_completion_tokens": 4})
        first = assemble(events)
        self.assertEqual(first["finish_reason"], "length")
        messages += [{"role": "assistant", "content": first["content"]}, {"role": "user", "content": "continue"}]
        self.queue("Step three.")
        events, _, _ = self.stream({"messages": messages})
        self.assertEqual(events[-2]["x_qwen38_metrics"]["cache_reuse_mode"], "exact_reuse")
        self.assertEqual(self.metrics()["active_session"]["truncation_continuations"], 1)


if __name__ == "__main__":
    unittest.main()
