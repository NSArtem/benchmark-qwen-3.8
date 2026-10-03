"""Pure OpenAI protocol tests: no MLX, no tokenizer, no model."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from qwen38.protocol import (
    ProtocolError,
    StreamAssembler,
    convert_parameter,
    extends_transcript,
    messages_equal,
    parse_assistant_text,
    parse_chat_request,
    tool_call_deltas,
    tools_identity,
)
from qwen38.server import protocol_self_test

FIXTURE = Path(__file__).parent / "fixtures" / "pi-1.0.0-requests.json"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "required": ["path"],
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "number"},
                    "limit": {"type": "integer"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "edits": {"type": "array"},
                    "dry_run": {"type": "boolean"},
                    "note": {"type": ["string", "null"]},
                },
            },
        },
    },
]


def _ids():
    counter = iter(range(1000))
    return lambda: f"call_{next(counter)}"


def call_text(name: str, **params: str) -> str:
    body = "".join(f"<parameter={key}>\n{value}\n</parameter>\n" for key, value in params.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


class RequestParsingTests(unittest.TestCase):
    def test_pi_fixture_request_parses(self) -> None:
        body = json.loads(FIXTURE.read_text())["final_request"]
        request = parse_chat_request(body)
        self.assertTrue(request.stream)
        self.assertTrue(request.include_usage)
        self.assertEqual(request.max_tokens, 4096)
        self.assertEqual(request.tool_choice, "auto")
        self.assertEqual([t["function"]["name"] for t in request.tools], ["read", "bash", "edit", "write"])
        roles = [m["role"] for m in request.messages]
        self.assertEqual(roles[0], "system")
        self.assertEqual(roles[-1], "tool")
        user = request.messages[1]
        self.assertIsInstance(user["content"], str)
        calls = [m for m in request.messages if m.get("tool_calls")]
        self.assertEqual(len(calls[0]["tool_calls"]), 2)
        self.assertEqual(calls[0]["tool_calls"][0]["function"]["arguments"], {"path": "hello.txt"})
        first_tool = next(m for m in request.messages if m["role"] == "tool")
        self.assertEqual(first_tool["tool_call_id"], "call_cap0")

    def test_backward_compatible_text_request(self) -> None:
        request = parse_chat_request(
            {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8, "session_id": 7}
        )
        self.assertFalse(request.stream)
        self.assertIsNone(request.tools)
        self.assertEqual(request.max_tokens, 8)
        self.assertEqual(request.session_id, "7")

    def test_max_completion_tokens_preferred(self) -> None:
        request = parse_chat_request(
            {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8, "max_completion_tokens": 32}
        )
        self.assertEqual(request.max_tokens, 32)

    def test_developer_role_is_system(self) -> None:
        request = parse_chat_request(
            {"messages": [{"role": "developer", "content": "be brief"}, {"role": "user", "content": "x"}]}
        )
        self.assertEqual(request.messages[0]["role"], "system")

    def test_harmless_unknown_fields_ignored(self) -> None:
        parse_chat_request(
            {
                "messages": [{"role": "user", "content": "x"}],
                "store": False,
                "metadata": {"a": 1},
                "user": "u",
                "seed": 3,
                "parallel_tool_calls": True,
                "response_format": {"type": "text"},
                "stop": None,
            }
        )

    def test_rejections(self) -> None:
        user = [{"role": "user", "content": "x"}]
        cases = {
            "image part": {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:"}}]}]},
            "empty messages": {"messages": []},
            "bad role": {"messages": [{"role": "robot", "content": "x"}]},
            "late system": {"messages": [*user, {"role": "system", "content": "s"}]},
            "empty user": {"messages": [{"role": "user", "content": "  "}]},
            "tool_choice required": {"messages": user, "tools": TOOLS, "tool_choice": "required"},
            "named tool_choice": {"messages": user, "tools": TOOLS, "tool_choice": {"type": "function", "function": {"name": "read"}}},
            "non-function tool": {"messages": user, "tools": [{"type": "custom", "custom": {"name": "x"}}]},
            "duplicate tool": {"messages": user, "tools": [TOOLS[0], TOOLS[0]]},
            "bad tool name": {"messages": user, "tools": [{"type": "function", "function": {"name": "a b"}}]},
            "bad parameters": {"messages": user, "tools": [{"type": "function", "function": {"name": "a", "parameters": "x"}}]},
            "bad arguments json": {
                "messages": [*user, {"role": "assistant", "content": None, "tool_calls": [{"id": "c", "type": "function", "function": {"name": "read", "arguments": "{bad"}}]}, {"role": "tool", "content": "r", "tool_call_id": "c"}]
            },
            "array arguments": {
                "messages": [*user, {"role": "assistant", "tool_calls": [{"id": "c", "function": {"name": "read", "arguments": "[1]"}}]}, {"role": "tool", "content": "r"}]
            },
            "n=2": {"messages": user, "n": 2},
            "stop": {"messages": user, "stop": ["\n"]},
            "json schema format": {"messages": user, "response_format": {"type": "json_object"}},
            "stream not bool": {"messages": user, "stream": "yes"},
            "zero max tokens": {"messages": user, "max_completion_tokens": 0},
            "not an object": [],
        }
        for name, body in cases.items():
            with self.subTest(name), self.assertRaises(ProtocolError):
                parse_chat_request(body)

    def test_tool_choice_none_hides_tools(self) -> None:
        request = parse_chat_request(
            {"messages": [{"role": "user", "content": "x"}], "tools": TOOLS, "tool_choice": "none"}
        )
        self.assertIsNone(request.effective_tools)
        auto = parse_chat_request({"messages": [{"role": "user", "content": "x"}], "tools": TOOLS})
        self.assertEqual(auto.effective_tools, TOOLS)

    def test_empty_tools_means_no_tools(self) -> None:
        request = parse_chat_request({"messages": [{"role": "user", "content": "x"}], "tools": []})
        self.assertIsNone(request.tools)

    def test_tools_identity_tracks_order_and_content(self) -> None:
        self.assertIsNone(tools_identity(None))
        self.assertNotEqual(tools_identity(TOOLS), tools_identity(list(reversed(TOOLS))))
        changed = json.loads(json.dumps(TOOLS))
        changed[0]["function"]["description"] = "Read a file!"
        self.assertNotEqual(tools_identity(TOOLS), tools_identity(changed))
        self.assertEqual(tools_identity(TOOLS), tools_identity(json.loads(json.dumps(TOOLS))))


class TranscriptIdentityTests(unittest.TestCase):
    def _assistant(self, arguments: str, content=None, call_id="call_1") -> dict:
        return parse_chat_request(
            {
                "messages": [
                    {"role": "user", "content": "x"},
                    {"role": "assistant", "content": content, "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "read", "arguments": arguments}}]},
                    {"role": "tool", "content": "r", "tool_call_id": call_id},
                ]
            }
        ).messages[1]

    def test_reserialized_arguments_are_equal(self) -> None:
        self.assertTrue(
            messages_equal(self._assistant('{"path": "a", "limit": 3}'), self._assistant('{"limit":3,"path":"a"}'))
        )

    def test_null_and_empty_content_are_equal(self) -> None:
        self.assertTrue(messages_equal(self._assistant("{}", None), self._assistant("{}", "")))
        self.assertTrue(messages_equal(self._assistant("{}", "Reading."), self._assistant("{}", "Reading.\n\n")))

    def test_changed_arguments_name_or_id_differ(self) -> None:
        base = self._assistant('{"path": "a"}')
        self.assertFalse(messages_equal(base, self._assistant('{"path": "b"}')))
        self.assertFalse(messages_equal(base, self._assistant('{"path": "a"}', call_id="call_2")))
        self.assertFalse(messages_equal(base, {"role": "assistant", "content": ""}))

    def test_tool_result_identity_includes_call_id(self) -> None:
        self.assertFalse(
            messages_equal(
                {"role": "tool", "content": "r", "tool_call_id": "a"},
                {"role": "tool", "content": "r", "tool_call_id": "b"},
            )
        )

    def test_extends_requires_strict_extension(self) -> None:
        stored = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
        self.assertTrue(extends_transcript(stored, [*stored, {"role": "user", "content": "c"}]))
        self.assertFalse(extends_transcript(stored, stored))
        self.assertFalse(extends_transcript(stored, [stored[0], {"role": "assistant", "content": "B"}, {"role": "user", "content": "c"}]))
        self.assertFalse(extends_transcript(stored, [{"role": "user", "content": "c"}]))


class ToolCallParsingTests(unittest.TestCase):
    def test_plain_text_answer(self) -> None:
        parsed = parse_assistant_text("The answer is 4.\n", TOOLS)
        self.assertEqual(parsed.content, "The answer is 4.\n")
        self.assertEqual(parsed.tool_calls, [])
        self.assertIsNone(parsed.parse_error)
        self.assertNotIn("tool_calls", parsed.message())

    def test_single_call_with_types(self) -> None:
        text = "I'll read it.\n\n" + call_text("read", path="src/a.py", offset="10", limit="20")
        parsed = parse_assistant_text(text, TOOLS, new_id=_ids())
        self.assertEqual(parsed.content, "I'll read it.")
        self.assertEqual(
            parsed.tool_calls,
            [{"id": "call_0", "type": "function", "function": {"name": "read", "arguments": {"path": "src/a.py", "offset": 10, "limit": 20}}}],
        )
        message = parsed.message()
        self.assertEqual(json.loads(message["tool_calls"][0]["function"]["arguments"])["limit"], 20)
        self.assertIsInstance(message["tool_calls"][0]["function"]["arguments"], str)

    def test_tool_only_turn_has_null_content(self) -> None:
        parsed = parse_assistant_text(call_text("read", path="a"), TOOLS)
        self.assertIsNone(parsed.message()["content"])
        self.assertEqual(parsed.stored_message()["content"], "")

    def test_multiple_calls_and_unique_ids(self) -> None:
        text = call_text("read", path="a") + "\n" + call_text("read", path="b") + "\n"
        parsed = parse_assistant_text(text, TOOLS)
        self.assertEqual([c["function"]["arguments"]["path"] for c in parsed.tool_calls], ["a", "b"])
        ids = [c["id"] for c in parsed.tool_calls]
        self.assertEqual(len(set(ids)), 2)
        self.assertTrue(all(i.startswith("call_") for i in ids))

    def test_multiline_unicode_and_escapes(self) -> None:
        content = 'def f():\n    return "café \\n \U0001f600"\n\n'
        text = call_text("edit", path="x.py", edits=json.dumps([{"oldText": "a\\b", "newText": content}]))
        parsed = parse_assistant_text(text, TOOLS)
        self.assertIsNone(parsed.parse_error)
        edits = parsed.tool_calls[0]["function"]["arguments"]["edits"]
        self.assertEqual(edits, [{"oldText": "a\\b", "newText": content}])
        # Strings keep exact interior/trailing newlines (one template newline removed).
        parsed = parse_assistant_text(call_text("read", path="line1\nline2\n"), TOOLS)
        self.assertEqual(parsed.tool_calls[0]["function"]["arguments"]["path"], "line1\nline2\n")

    def test_value_containing_closing_tag(self) -> None:
        tricky = "x = '</parameter>' # not the end"
        parsed = parse_assistant_text(call_text("edit", path="p", note=tricky), TOOLS)
        self.assertEqual(parsed.tool_calls[0]["function"]["arguments"]["note"], tricky)

    def test_nested_json_object(self) -> None:
        tools = [{"type": "function", "function": {"name": "cfg", "parameters": {"properties": {"obj": {"type": "object"}}}}}]
        parsed = parse_assistant_text(call_text("cfg", obj='{"a": {"b": [1, 2, {"c": null}]}}'), tools)
        self.assertEqual(parsed.tool_calls[0]["function"]["arguments"]["obj"], {"a": {"b": [1, 2, {"c": None}]}})

    def test_type_conversions(self) -> None:
        self.assertEqual(convert_parameter("true", {"type": "boolean"}), (True, True))
        self.assertEqual(convert_parameter("False", {"type": "boolean"}), (False, True))
        self.assertEqual(convert_parameter("null", {"type": ["string", "null"]}), (None, True))
        self.assertEqual(convert_parameter("null", {"type": "string"}), ("null", True))
        self.assertEqual(convert_parameter("3.0", {"type": "integer"}), (3, True))
        self.assertEqual(convert_parameter("1.5", {"type": "number"}), (1.5, True))
        self.assertEqual(convert_parameter("007", {"type": "string"}), ("007", True))
        self.assertEqual(convert_parameter("abc", {"type": "integer"}), ("abc", False))
        self.assertEqual(convert_parameter("[1]", {"type": "object"}), ("[1]", False))
        self.assertEqual(convert_parameter("x", {"enum": ["x", "y"]}), ("x", True))
        self.assertEqual(convert_parameter("42", None), ("42", True))

    def test_type_mismatch_is_reported_not_fatal(self) -> None:
        parsed = parse_assistant_text(call_text("read", path="a", limit="ten"), TOOLS)
        self.assertEqual(parsed.tool_calls[0]["function"]["arguments"]["limit"], "ten")
        self.assertEqual(parsed.argument_type_mismatches, ["read.limit"])

    def test_unknown_tool_is_reported(self) -> None:
        parsed = parse_assistant_text(call_text("grep", pattern="x"), TOOLS)
        self.assertEqual(parsed.tool_calls[0]["function"]["name"], "grep")
        self.assertEqual(parsed.unknown_tools, ["grep"])

    def test_malformed_and_truncated_calls_stay_text(self) -> None:
        full = call_text("read", path="a.py")
        cases = {
            "truncated": full[:-10],
            "cut in value": "<tool_call>\n<function=read>\n<parameter=path>\nsrc/",
            "no function": "<tool_call>\nread a.py\n</tool_call>",
            "missing close": full.replace("</tool_call>", ""),
            "text after call": full + "\nDone, now waiting.",
            "bad name": full.replace("<function=read>", "<function=re ad>"),
            "duplicate param": call_text("read", path="a").replace("</function>", "<parameter=path>\nb\n</parameter>\n</function>"),
            "empty": "<tool_call>",
        }
        for name, text in cases.items():
            with self.subTest(name):
                parsed = parse_assistant_text("Let me look.\n" + text, TOOLS)
                self.assertEqual(parsed.tool_calls, [])
                self.assertTrue(parsed.parse_error)
                self.assertEqual(parsed.content, "Let me look.\n" + text)

    def test_no_parsing_without_tools(self) -> None:
        text = call_text("read", path="a")
        parsed = parse_assistant_text(text, None)
        self.assertEqual(parsed.content, text)
        self.assertEqual(parsed.tool_calls, [])
        self.assertIsNone(parsed.parse_error)


class StreamAssemblyTests(unittest.TestCase):
    def _stream(self, text: str, tools_active: bool, steps: list[int] | None = None) -> tuple[str, list[str]]:
        assembler = StreamAssembler(tools_active)
        ends = steps or list(range(1, len(text) + 1))
        deltas = [assembler.feed(text[:end]) for end in ends]
        parsed = parse_assistant_text(text, TOOLS if tools_active else None)
        deltas.append(assembler.finish(parsed))
        joined = "".join(deltas)
        self.assertEqual(joined, parsed.content or "")
        return joined, [d for d in deltas if d]

    def test_text_stream_equals_final_content(self) -> None:
        for tools_active in (True, False):
            with self.subTest(tools_active=tools_active):
                joined, deltas = self._stream("Hello world.\n\nSecond line  \n", tools_active)
                self.assertEqual(joined, "Hello world.\n\nSecond line  \n")
                self.assertGreater(len(deltas), 2)

    def test_tool_call_markup_never_streamed_as_content(self) -> None:
        text = "Checking.\n\n" + call_text("read", path="a.py")
        joined, deltas = self._stream(text, True)
        self.assertEqual(joined, "Checking.")
        self.assertFalse(any("<" in d for d in deltas))

    def test_partial_marker_prefix_is_held(self) -> None:
        assembler = StreamAssembler(True)
        self.assertEqual(assembler.feed("ok <tool"), "ok")
        self.assertEqual(assembler.feed("ok <toolbox"), " <toolbox")

    def test_incomplete_utf8_is_held(self) -> None:
        assembler = StreamAssembler(False)
        self.assertEqual(assembler.feed("caf�"), "caf")
        self.assertEqual(assembler.feed("café!"), "é!")

    def test_malformed_call_flushes_raw_text(self) -> None:
        text = "Trying.\n<tool_call>\n<function=read>\n<parameter=path>\nsrc/"
        joined, _ = self._stream(text, True)
        self.assertEqual(joined, text)

    def test_tool_call_deltas_reassemble(self) -> None:
        parsed = parse_assistant_text(
            call_text("read", path="東京" * 300) + call_text("read", path="b"), TOOLS
        )
        deltas = tool_call_deltas(parsed.tool_calls, fragment_chars=64)
        assembled: dict[int, dict] = {}
        for delta in deltas:
            for item in delta["tool_calls"]:
                slot = assembled.setdefault(item["index"], {"id": None, "name": None, "args": ""})
                slot["id"] = item.get("id", slot["id"])
                slot["name"] = (item.get("function") or {}).get("name") or slot["name"]
                slot["args"] += (item.get("function") or {}).get("arguments", "")
        self.assertEqual(len(assembled), 2)
        for index, call in enumerate(parsed.tool_calls):
            self.assertEqual(assembled[index]["id"], call["id"])
            self.assertEqual(assembled[index]["name"], "read")
            self.assertEqual(json.loads(assembled[index]["args"]), call["function"]["arguments"])
        self.assertGreater(len(deltas), 4)


class SelfTest(unittest.TestCase):
    def test_protocol_self_test(self) -> None:
        self.assertEqual(protocol_self_test()["classification"], "PROTOCOL_SELF_TEST_PASS")


if __name__ == "__main__":
    unittest.main()
