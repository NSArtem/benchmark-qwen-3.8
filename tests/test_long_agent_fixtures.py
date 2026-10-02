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


if __name__ == "__main__":
    unittest.main()
