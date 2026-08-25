"""Protocol-only tests for the serialized Qwen3.8 agent backend."""

import unittest

from research.runtime.qwen38_agent_server import (
    appended_messages,
    normalize_messages,
    protocol_self_test,
)


class AgentProtocolTests(unittest.TestCase):
    def test_tool_messages_are_text_normalized(self) -> None:
        messages = normalize_messages(
            [
                {"role": "user", "content": "read"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"function": {"name": "read_file"}}],
                },
                {"role": "tool", "content": "file contents"},
            ]
        )
        self.assertEqual([m["role"] for m in messages], ["user", "assistant", "tool"])
        self.assertIn("read_file", messages[1]["content"])

    def test_stale_transcript_is_rejected(self) -> None:
        previous = [{"role": "user", "content": "one"}]
        with self.assertRaises(RuntimeError):
            appended_messages(previous, [{"role": "user", "content": "other"}])

    def test_protocol_self_test(self) -> None:
        self.assertEqual(protocol_self_test()["classification"], "PROTOCOL_SELF_TEST_PASS")


if __name__ == "__main__":
    unittest.main()
