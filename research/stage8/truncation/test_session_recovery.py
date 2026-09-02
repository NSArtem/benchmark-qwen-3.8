"""Lightweight session tests using the real tokenizer and a fake generator.

These tests exercise cache bookkeeping and recovery without loading Qwen3.8
weights or allocating Metal memory.
"""

from __future__ import annotations

import unittest
from typing import Any

from transformers import AutoTokenizer

import research.runtime.qwen38_agent_server as agent_server
import research.runtime.qwen38_chat as chat
from research.runtime.qwen38_agent_server import AgentSession


MODEL = "/Users/artem/models/Qwen3.8-27B-3bit-g128-MLX"


class _Scalar:
    def __init__(self, value: int) -> None:
        self.value = int(value)

    def item(self) -> int:
        return self.value


class _Cache:
    def __init__(self) -> None:
        self.offset = 0


class _Core:
    def array(self, value: Any) -> list[int]:
        return [int(item) for item in value]

    def synchronize(self) -> None:
        return None

    def get_active_memory(self) -> int:
        return 0

    def get_cache_memory(self) -> int:
        return 0

    def get_peak_memory(self) -> int:
        return 0


class _Model:
    def __init__(self, responses: list[list[int]]) -> None:
        self.responses = list(responses)


class SessionRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)

    def setUp(self) -> None:
        self.core = _Core()

        def make_prompt_cache(_model: Any, _max_kv_size: int | None) -> list[_Cache]:
            return [_Cache()]

        def generate_step(
            prompt: list[int],
            model: _Model,
            *,
            max_tokens: int,
            prompt_cache: list[_Cache],
            **_kwargs: Any,
        ):
            response = list(model.responses.pop(0))[:max_tokens]

            def iterator():
                prompt_cache[0].offset += len(prompt)
                for token_id in response:
                    prompt_cache[0].offset += 1
                    yield _Scalar(token_id), None

            return iterator()

        self.model = _Model(
            [
                [9008, 239, 102, 373],
                [68, 1257],
                [2487, 26555],
                [14311, 248046],
                [71, 72],
            ]
        )
        self.fake_mlx = (
            self.core,
            None,
            generate_step,
            make_prompt_cache,
            lambda _temperature, _top_p: object(),
        )
        self.old_agent_require = agent_server._require_mlx
        self.old_chat_require = chat._require_mlx
        agent_server._require_mlx = lambda: self.fake_mlx
        chat._require_mlx = lambda: self.fake_mlx

    def tearDown(self) -> None:
        agent_server._require_mlx = self.old_agent_require
        chat._require_mlx = self.old_chat_require

    def _session(self) -> AgentSession:
        return AgentSession(self.model, self.tokenizer, "test-session")

    def _complete(self, session: AgentSession, messages: list[dict[str, str]], max_tokens: int = 4) -> dict[str, Any]:
        return session.complete(
            messages,
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            kv_bits=None,
            kv_group_size=64,
            quantized_kv_start=0,
            prefill_step_size=128,
        )

    def test_length_truncation_and_two_followups_reuse_exact_state(self) -> None:
        session = self._session()
        first = self._complete(session, [{"role": "user", "content": "emit a marker"}])
        self.assertEqual(first["finish_reason"], "length")
        second_messages = [*session.messages, {"role": "user", "content": "continue"}]
        second = self._complete(session, second_messages, max_tokens=2)
        self.assertEqual(second["cache_reuse_mode"], "exact_reuse")
        self.assertTrue(second["cache_exact_reuse"])
        self.assertEqual(session.stats()["truncation_continuations"], 1)
        third_messages = [*session.messages, {"role": "user", "content": "continue again"}]
        third = self._complete(session, third_messages, max_tokens=2)
        self.assertEqual(third["cache_reuse_mode"], "exact_reuse")
        self.assertEqual(session.stats()["truncation_continuations"], 2)
        self.assertEqual(session.stats()["cache_fresh_fallback"], 0)

    def test_eos_turn_continues_and_reset_discards_cache(self) -> None:
        session = self._session()
        self.model.responses = [[14311, 248046], [71, 72], [73, 74]]
        first = self._complete(session, [{"role": "user", "content": "finish"}], max_tokens=8)
        self.assertEqual(first["finish_reason"], "stop")
        continuation = self._complete(
            session,
            [*session.messages, {"role": "user", "content": "next"}],
            max_tokens=2,
        )
        self.assertEqual(continuation["cache_reuse_mode"], "exact_reuse")
        session.reset()
        fresh = self._complete(session, [{"role": "user", "content": "after reset"}], max_tokens=2)
        self.assertEqual(fresh["cache_reuse_mode"], "fresh_prefill")
        self.assertEqual(session.stats()["logical_context_tokens"], fresh["logical_context_tokens"])

    def test_valid_history_edit_uses_fresh_fallback(self) -> None:
        session = self._session()
        self._complete(session, [{"role": "user", "content": "original"}], max_tokens=2)
        edited = self._complete(
            session,
            [
                {"role": "user", "content": "edited"},
                {"role": "assistant", "content": "stored externally"},
                {"role": "user", "content": "continue from edited history"},
            ],
            max_tokens=2,
        )
        self.assertEqual(edited["cache_reuse_mode"], "fresh_fallback")
        self.assertTrue(edited["cache_fresh_fallback"])
        self.assertEqual(session.stats()["cache_fresh_fallback"], 1)
        self.assertEqual(session.stats()["cache_conflict_error"], 0)


if __name__ == "__main__":
    unittest.main()
