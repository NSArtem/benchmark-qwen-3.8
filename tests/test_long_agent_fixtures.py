"""The long-agent benchmark's fixtures and tool protocol load without a model."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


BENCHMARK = Path(__file__).resolve().parents[1] / "benchmarks" / "long_agent.py"


def _load_benchmark():
    spec = importlib.util.spec_from_file_location("long_agent", BENCHMARK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LongAgentFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bench = _load_benchmark()

    def test_every_fixture_file_is_served(self) -> None:
        fixtures = self.bench.fixture_map()
        served = set(fixtures.values())
        for path in self.bench.FIXTURE_ROOT.iterdir():
            if path.name != "README.md":
                self.assertIn(path.read_text(encoding="utf-8"), served, path.name)

    def test_long_tool_result_is_long(self) -> None:
        payload = self.bench.fixture_map()["long_dependency_audit.txt"]
        self.assertGreater(len(payload), 20_000)

    def test_tool_request_protocol(self) -> None:
        self.assertEqual(
            self.bench.TOOL_RE.findall("TOOL_CALL read_file tests/test_state.py"),
            [("read_file", "tests/test_state.py")],
        )


class HarnessInjectionTests(unittest.TestCase):
    """An expected tool request the model misses twice is completed at most once."""

    PATH = "long_dependency_audit.txt"

    def setUp(self) -> None:
        self.bench = _load_benchmark()
        self.replies: list[str] = []
        self.sent: list[list[dict]] = []

        def fake_request_json(_base_url, _path, payload, **_kwargs):
            self.sent.append([dict(m) for m in payload["messages"]])
            text = self.replies.pop(0)
            return 200, {
                "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                "usage": {},
                "x_qwen38_metrics": {},
            }, None

        self.bench.request_json = fake_request_json
        self.runner = self.bench.SessionRunner("http://unused", "s", 32)

    def test_cut_off_request_is_completed_once(self) -> None:
        self.replies = ["TOOL_CALL read_file long_dependency", "TOOL", "synthesis"]
        self.runner.expected_turn("inspect", "", self.PATH)
        self.assertEqual(self.replies, [])
        self.assertTrue(self.sent[-1][-1]["content"].startswith(f"read_file {self.PATH} returned:\n"))
        summary = self.runner.summary()
        self.assertTrue(summary["injected_tool_after_invalid"])
        self.assertEqual(summary["invalid_tool_calls"], 1)
        self.assertEqual(summary["recovered_invalid_tool_calls"], 1)
        self.assertEqual(summary["missing_expected_tool_requests"], 2)
        self.assertEqual(summary["recovered_missing_tool_requests"], 2)
        self.assertEqual(summary["valid_tool_calls"], 0)

        # A second miss is not completed and stays unrecovered.
        self.replies = ["TOOL", "TOOL"]
        self.runner.expected_turn("inspect_again", "", self.PATH)
        summary = self.runner.summary()
        self.assertEqual(len(summary["harness_tool_injections"]), 1)
        self.assertEqual(summary["missing_expected_tool_requests"], 4)
        self.assertEqual(summary["recovered_missing_tool_requests"], 2)
        self.assertFalse(summary["task_success"])

    def test_valid_request_for_another_file_is_not_overridden(self) -> None:
        self.replies = ["TOOL_CALL read_file retry.py", "TOOL_CALL read_file retry.py", "answer"]
        self.runner.expected_turn("inspect", "", self.PATH)
        summary = self.runner.summary()
        self.assertFalse(summary["injected_tool_after_invalid"])
        self.assertEqual(summary["valid_tool_calls"], 1)
        self.assertEqual(summary["recovered_missing_tool_requests"], 0)
        self.assertTrue(self.sent[-1][-1]["content"].startswith("read_file retry.py returned:\n"))


if __name__ == "__main__":
    unittest.main()
