"""Tokenizer-level tool-calling tests: real Qwen3.8 template, fake generator.

These pin down the model's native tool format as the installed chat template
defines it, prove the server's parser inverts that rendering, and check cache
identity when tools take part in the prompt.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

import qwen38.chat as chat
from qwen38.protocol import parse_assistant_text, parse_chat_request
from qwen38.server import AgentSession, ContextOverflow

from fake_mlx import FakeMLX, FakeModel, complete, encode, load_tokenizer, requires_tokenizer

PI_FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "pi-1.0.0-requests.json").read_text())
PI_TOOLS = PI_FIXTURE["final_request"]["tools"]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}, "timeout": {"type": "number"}},
                "required": ["command"],
            },
        },
    },
]


def call_text(name: str, **params: Any) -> str:
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


@requires_tokenizer
class NativeTemplateFormatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tok = load_tokenizer()

    def render(self, messages: list[dict[str, Any]], tools: Any = None, **kw: Any) -> str:
        kwargs = {"tools": tools} if tools else {}
        return self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False, **kwargs, **kw
        )

    def test_tool_markers_are_single_non_special_tokens(self) -> None:
        for marker, expected in (("<tool_call>", 248058), ("</tool_call>", 248059), ("<tool_response>", 248066), ("</tool_response>", 248067)):
            ids = chat.normalize_token_ids(self.tok.encode(marker, add_special_tokens=False))
            self.assertEqual(ids, [expected])
            self.assertEqual(self.tok.decode(ids, skip_special_tokens=True), marker)
        self.assertEqual(chat.eos_token_ids(self.tok), {248046})

    def test_tools_render_in_system_block_in_order(self) -> None:
        messages = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "hi"}]
        rendered = self.render(messages, TOOLS)
        self.assertTrue(rendered.startswith("<|im_start|>system\n# Tools\n"))
        first = rendered.index(json.dumps(TOOLS[0], ensure_ascii=False))
        second = rendered.index(json.dumps(TOOLS[1], ensure_ascii=False))
        self.assertLess(first, second)
        self.assertIn("</IMPORTANT>\n\nSYS<|im_end|>", rendered)
        self.assertTrue(rendered.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
        self.assertNotIn("# Tools", self.render(messages))

    def test_assistant_tool_calls_and_grouped_results(self) -> None:
        messages = [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": "Reading.",
                "tool_calls": [
                    {"id": "call_a", "type": "function", "function": {"name": "read_file", "arguments": {"path": "a"}}},
                    {"id": "call_b", "type": "function", "function": {"name": "run_command", "arguments": {"command": "ls", "timeout": 5}}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_a", "content": "A"},
            {"role": "tool", "tool_call_id": "call_b", "content": "B"},
        ]
        rendered = self.render(messages, TOOLS)
        expected = (
            "<|im_start|>assistant\n<think>\n\n</think>\n\nReading.\n\n"
            + call_text("read_file", path="a")
            + "\n"
            + call_text("run_command", command="ls", timeout=5)
            + "<|im_end|>\n<|im_start|>user\n<tool_response>\nA\n</tool_response>\n"
            "<tool_response>\nB\n</tool_response><|im_end|>\n"
        )
        self.assertIn(expected, rendered)
        # Tool call IDs are not part of the prompt: they belong to the API layer.
        self.assertNotIn("call_a", rendered)

    def test_template_requires_mapping_arguments(self) -> None:
        # Gap in the source template: OpenAI sends arguments as a JSON string.
        # The server parses it to a mapping before rendering.
        messages = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a"}'}}]},
            {"role": "tool", "tool_call_id": "c", "content": "A"},
        ]
        with self.assertRaises(Exception):
            self.render(messages, TOOLS)
        normalized = parse_chat_request({"messages": messages, "tools": TOOLS}).messages
        self.assertIn(call_text("read_file", path="a"), self.render(normalized, TOOLS))

    def test_parser_inverts_template_rendering(self) -> None:
        arguments = {
            "path": "src/caf\u00e9 \u6771\u4eac.py",
            "edits": [
                {"oldText": "def f():\n    return 1\n", "newText": "def f():\n    return \"\\n\" + '\U0001f600'\n"},
            ],
        }
        messages = [
            {"role": "user", "content": "edit"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "x", "type": "function", "function": {"name": "edit", "arguments": arguments}}]},
            {"role": "tool", "tool_call_id": "x", "content": "ok"},
        ]
        rendered = self.render(messages, PI_TOOLS)
        start = rendered.index("<tool_call>", rendered.index("<|im_start|>assistant\n"))
        end = rendered.index("<|im_end|>", start)
        parsed = parse_assistant_text(rendered[start:end], PI_TOOLS)
        self.assertIsNone(parsed.parse_error)
        self.assertEqual(parsed.tool_calls[0]["function"]["arguments"], arguments)

    def test_preserve_thinking_default_keeps_assistant_prefix_stable(self) -> None:
        # Every assistant turn re-renders with the same empty think block the
        # generation prompt used, so later requests stay exact extensions.
        first = self.render([{"role": "user", "content": "a"}], TOOLS)
        second = self.render(
            [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
                {"role": "user", "content": "c"},
            ],
            TOOLS,
        )
        self.assertTrue(second.startswith(first))

    def test_thinking_mode_changes_prompt(self) -> None:
        messages = [{"role": "user", "content": "a"}]
        on = self.tok.apply_chat_template(messages, tools=TOOLS, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        self.assertIn("Reasoning effort is set to xhigh", on)
        self.assertTrue(on.endswith("<|im_start|>assistant\n<think>\n"))


@requires_tokenizer
class ToolSessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tok = load_tokenizer()

    def setUp(self) -> None:
        self.fake = FakeMLX()
        self.fake.install()
        self.model = FakeModel()
        self.session = AgentSession(self.model, self.tok, "s")

    def tearDown(self) -> None:
        self.fake.uninstall()

    def queue(self, text: str, eos: bool = True) -> None:
        self.model.responses.append(encode(self.tok, text, eos))

    def client_view(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """What an OpenAI client stores and replays (arguments re-serialized)."""

        message = json.loads(json.dumps(metrics["message"]))
        for call in message.get("tool_calls") or []:
            call["function"]["arguments"] = json.dumps(
                json.loads(call["function"]["arguments"]), separators=(",", ":")
            )
        return message

    def replay(self, raw_messages: list[dict[str, Any]], tools: Any = TOOLS) -> list[dict[str, Any]]:
        return parse_chat_request({"messages": raw_messages, "tools": tools}).messages

    def test_tool_round_trip_reuses_exact_cache(self) -> None:
        raw = [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Show README.md"}]
        self.queue("I'll read it.\n\n" + call_text("read_file", path="README.md"))
        first = complete(self.session, self.replay(raw), tools=TOOLS)
        self.assertEqual(first["finish_reason"], "tool_calls")
        self.assertEqual(first["cache_reuse_mode"], "fresh_prefill")
        call = first["message"]["tool_calls"][0]
        self.assertEqual(json.loads(call["function"]["arguments"]), {"path": "README.md"})
        self.assertEqual(first["message"]["content"], "I'll read it.")

        raw += [self.client_view(first), {"role": "tool", "tool_call_id": call["id"], "content": "# Demo\n"}]
        self.queue("The README is titled Demo.")
        second = complete(self.session, self.replay(raw), tools=TOOLS)
        self.assertEqual(second["cache_reuse_mode"], "exact_reuse")
        self.assertEqual(second["finish_reason"], "stop")
        self.assertEqual(second["cached_tokens"], first["logical_context_tokens"])
        # Only the tool-result suffix was prefilled.
        self.assertLess(second["processed_suffix_tokens"], 40)
        self.assertEqual(self.model.prompts[1], self.model.prompts[1][: second["processed_suffix_tokens"]])

        raw += [self.client_view(second), {"role": "user", "content": "Now run the tests."}]
        self.queue(call_text("run_command", command="pytest -q", timeout=60) + "\n" + call_text("read_file", path="log.txt"))
        third = complete(self.session, self.replay(raw), tools=TOOLS)
        self.assertEqual(third["cache_reuse_mode"], "exact_reuse")
        self.assertEqual(len(third["tool_calls"]), 2)
        self.assertIsNone(third["message"]["content"])
        args = json.loads(third["message"]["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(args, {"command": "pytest -q", "timeout": 60})

        ids = [c["id"] for c in third["message"]["tool_calls"]]
        raw += [self.client_view(third)] + [
            {"role": "tool", "tool_call_id": i, "content": f"result {n}"} for n, i in enumerate(ids)
        ]
        self.queue("All tests pass.")
        fourth = complete(self.session, self.replay(raw), tools=TOOLS)
        self.assertEqual(fourth["cache_reuse_mode"], "exact_reuse")
        stats = self.session.stats()
        self.assertEqual((stats["cache_exact_reuse"], stats["cache_fresh_fallback"]), (3, 0))
        self.assertFalse(any(m["replay_detected"] for m in (first, second, third, fourth)))
        # The ledger equals a fresh canonical render reconciled with generated IDs.
        self.assertEqual(stats["logical_context_tokens"], fourth["cache_offset_after"])

    def test_reformatted_tool_json_keeps_exact_reuse_over_many_turns(self) -> None:
        # The model's own JSON spelling differs from the template's tojson
        # re-render, so each assistant span re-renders with a different
        # length.  Every later turn must still reuse the cache exactly.
        tools = [{"type": "function", "function": {"name": "edit", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "edits": {"type": "array"}}}}}]
        raw: list[dict[str, Any]] = [{"role": "user", "content": "Make three edits."}]
        for index in range(4):
            spelled = '[\n  {\n    "oldText": "a%d",\n    "newText": "b%d"\n  }\n]' % (index, index)
            self.queue(f"Edit {index}.\n\n" + call_text("edit", path="f.py", edits=spelled))
            result = complete(self.session, self.replay(raw, tools), tools=tools)
            self.assertEqual(result["finish_reason"], "tool_calls")
            if index:
                self.assertEqual(result["cache_reuse_mode"], "exact_reuse", result["cache_fallback_reason"])
                self.assertEqual(result["processed_suffix_tokens"], result["prompt_tokens"] - result["cached_tokens"])
            adjustments = result["canonical_cache_boundary_adjustments"]
            if index >= 2:
                self.assertTrue(any(a.get("old_length") != a.get("new_length") for a in adjustments))
            call_id = result["message"]["tool_calls"][0]["id"]
            raw += [self.client_view(result), {"role": "tool", "tool_call_id": call_id, "content": f"edited {index}"}]
        self.assertEqual(self.session.stats()["cache_fresh_fallback"], 0)

    def _first_tool_turn(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        raw = [{"role": "user", "content": "Show README.md"}]
        self.queue(call_text("read_file", path="README.md"))
        first = complete(self.session, self.replay(raw), tools=TOOLS)
        call_id = first["message"]["tool_calls"][0]["id"]
        raw += [self.client_view(first), {"role": "tool", "tool_call_id": call_id, "content": "# Demo"}]
        return raw, first

    def test_repeated_tool_call_is_not_stale_replay(self) -> None:
        raw = [{"role": "user", "content": "Run the tests, fix, and rerun."}]
        command = call_text("run_command", command="python3 -m unittest -v")
        self.queue(command)
        first = complete(self.session, self.replay(raw), tools=TOOLS)
        raw += [self.client_view(first), {"role": "tool", "tool_call_id": first["message"]["tool_calls"][0]["id"], "content": "FAILED"}]
        self.queue(command)
        second = complete(self.session, self.replay(raw), tools=TOOLS)
        self.assertTrue(second["repeated_tool_call"])
        self.assertFalse(second["replay_detected"])
        raw += [self.client_view(second), {"role": "tool", "tool_call_id": second["message"]["tool_calls"][0]["id"], "content": "FAILED"}]
        self.queue("The same long prose answer, repeated verbatim twice.")
        complete(self.session, self.replay(raw), tools=TOOLS)
        raw += [{"role": "assistant", "content": "The same long prose answer, repeated verbatim twice."}, {"role": "user", "content": "again"}]
        self.queue("The same long prose answer, repeated verbatim twice.")
        prose = complete(self.session, self.replay(raw), tools=TOOLS)
        self.assertTrue(prose["replay_detected"])

    def test_changed_tool_schema_forces_fresh_prefill(self) -> None:
        raw, _ = self._first_tool_turn()
        changed = json.loads(json.dumps(TOOLS))
        changed[1]["function"]["description"] = "Run a shell command safely"
        self.queue("ok")
        result = complete(self.session, self.replay(raw, changed), tools=changed)
        self.assertEqual(result["cache_reuse_mode"], "fresh_fallback")
        self.assertEqual(result["cache_fallback_reason"], "tool_set_changed")
        self.assertEqual(result["cached_tokens"], 0)
        self.assertEqual(result["processed_suffix_tokens"], result["prompt_tokens"])

    def test_reordered_tools_force_fresh_prefill(self) -> None:
        raw, _ = self._first_tool_turn()
        reordered = list(reversed(TOOLS))
        self.queue("ok")
        result = complete(self.session, self.replay(raw, reordered), tools=reordered)
        self.assertEqual(result["cache_reuse_mode"], "fresh_fallback")

    def test_dropping_tools_forces_fresh_prefill(self) -> None:
        raw, _ = self._first_tool_turn()
        self.queue("ok")
        result = complete(self.session, self.replay(raw, None), tools=None)
        self.assertEqual(result["cache_reuse_mode"], "fresh_fallback")

    def test_edited_tool_call_is_not_reused(self) -> None:
        raw, _ = self._first_tool_turn()
        raw[1]["tool_calls"][0]["function"]["arguments"] = '{"path":"OTHER.md"}'
        self.queue("ok")
        result = complete(self.session, self.replay(raw), tools=TOOLS)
        self.assertEqual(result["cache_reuse_mode"], "fresh_fallback")
        self.assertEqual(result["cache_fallback_reason"], "transcript_does_not_extend_session")

    def test_truncated_tool_call_is_text_and_continuable(self) -> None:
        raw = [{"role": "user", "content": "Show README.md"}]
        text = call_text("read_file", path="README.md")
        ids = encode(self.tok, text, eos=False)
        self.model.responses.append(ids)
        first = complete(self.session, self.replay(raw), max_tokens=len(ids) - 3, tools=TOOLS)
        self.assertEqual(first["finish_reason"], "length")
        self.assertEqual(first["tool_calls"], [])
        self.assertTrue(first["tool_parse_error"])
        self.assertTrue(first["message"]["content"].startswith("<tool_call>"))
        raw += [self.client_view(first), {"role": "user", "content": "continue"}]
        self.queue("README.md")
        second = complete(self.session, self.replay(raw), tools=TOOLS)
        self.assertEqual(second["cache_reuse_mode"], "exact_reuse")
        self.assertEqual(self.session.stats()["truncation_continuations"], 1)

    def test_malformed_tool_syntax_is_text_with_stop(self) -> None:
        raw = [{"role": "user", "content": "Show README.md"}]
        self.queue("<tool_call>\nread_file README.md\n</tool_call>")
        result = complete(self.session, self.replay(raw), tools=TOOLS)
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(result["tool_calls"], [])
        self.assertTrue(result["tool_parse_error"])
        self.assertIn("read_file README.md", result["message"]["content"])

    def test_context_overflow_leaves_session_intact(self) -> None:
        raw, first = self._first_tool_turn()
        with self.assertRaises(ContextOverflow) as caught:
            complete(self.session, self.replay(raw), tools=TOOLS, max_context_tokens=first["prompt_tokens"])
        self.assertIn("maximum context length is", str(caught.exception))
        self.assertEqual(len(self.session.messages), 2)
        self.queue("ok")
        result = complete(self.session, self.replay(raw), tools=TOOLS, max_context_tokens=100_000)
        self.assertEqual(result["cache_reuse_mode"], "exact_reuse")

    def test_generation_clamped_to_context_cap(self) -> None:
        raw = [{"role": "user", "content": "count"}]
        self.queue("one two three four five six seven eight", eos=False)
        prompt = len(chat.canonical_prompt(self.tok, self.replay(raw), tools=TOOLS)["token_ids"])
        result = complete(self.session, self.replay(raw), tools=TOOLS, max_context_tokens=prompt + 3)
        self.assertEqual(result["generated_tokens"], 3)
        self.assertEqual(result["finish_reason"], "length")

    def test_cancellation_discards_cache(self) -> None:
        raw, _ = self._first_tool_turn()
        self.queue("a long answer that will be cut by the client")
        seen: list[str] = []
        with self.assertRaises(Exception):
            complete(
                self.session, self.replay(raw), tools=TOOLS,
                on_delta=seen.append, should_stop=lambda: len(seen) >= 2,
            )
        stats = self.session.stats()
        self.assertEqual((stats["message_count"], stats["logical_context_tokens"], stats["invalidations"]), (0, 0, 1))
        self.queue("fresh answer")
        result = complete(self.session, self.replay(raw), tools=TOOLS)
        self.assertEqual(result["cache_reuse_mode"], "fresh_prefill")
        self.assertEqual(result["processed_suffix_tokens"], result["prompt_tokens"])

    def test_generation_failure_discards_cache(self) -> None:
        raw, _ = self._first_tool_turn()
        self.queue("partial output then crash")
        self.model.fail_after = 2
        with self.assertRaises(MemoryError):
            complete(self.session, self.replay(raw), tools=TOOLS)
        self.assertEqual(self.session.stats()["logical_context_tokens"], 0)

    def test_streamed_deltas_match_final_content_unicode(self) -> None:
        raw = [{"role": "user", "content": "emoji"}]
        text = "Caf\u00e9 \U0001f469\u200d\U0001f4bb \u6771\u4eac \U00010348 done.\n\n" + call_text("read_file", path="\u00e9.txt")
        self.queue(text)
        deltas: list[str] = []
        result = complete(self.session, self.replay(raw), tools=TOOLS, on_delta=deltas.append)
        self.assertTrue(all("\ufffd" not in d for d in deltas))
        self.assertEqual("".join(deltas) + result["content_tail"], result["message"]["content"])
        self.assertEqual(result["message"]["content"], text[: text.index("\n\n<tool_call>")])
        self.assertEqual(json.loads(result["message"]["tool_calls"][0]["function"]["arguments"]), {"path": "\u00e9.txt"})


if __name__ == "__main__":
    unittest.main()
