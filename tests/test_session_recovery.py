"""Session recovery tests using the real tokenizer and a fake generator.

These tests exercise cache bookkeeping and truncation recovery without loading
Qwen3.8 weights or allocating Metal memory.  The token sequences below are
real Qwen3.8 regressions: a length-truncated turn ending inside a partial
UTF-8 sequence (``[9008, 239, 102, 373]`` decodes to an emoji followed by
U+FFFD and re-encodes as ``[9008, 239, 102, 5513]``) must still continue by
exact reuse.

The tokenizer is read from the local model directory (``QWEN38_MODEL``,
default below); the tests are skipped when it is absent.
"""

from __future__ import annotations

import json
import os
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import qwen38.chat as chat
import qwen38.server as agent_server
from qwen38.server import AgentSession


MODEL = os.environ.get("QWEN38_MODEL", "/Users/artem/models/Qwen3.8-27B-3bit-g128-MLX")


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


requires_tokenizer = unittest.skipUnless(
    (Path(MODEL) / "tokenizer.json").is_file(),
    f"Qwen3.8 tokenizer not found at {MODEL}; set QWEN38_MODEL",
)

# Length budgets at and around power-of-two boundaries.
LENGTH_BUDGETS = [1, 2, 3, 31, 32, 63, 64, 127, 128, 255, 256]

# Prose, punctuation, indentation, a code fence, multi-byte Unicode (including
# a ZWJ emoji sequence), and an unterminated code span.
STRESS_TEXT = (
    "normal prose; punctuation, indentation, and code: \n"
    "```python\n"
    "def merge(items):\n"
    "    return [x for x in items]\n"
    "```\n"
    "Unicode boundaries: caf\u00e9 \u2014 \u6771\u4eac \U0001f469\u200d\U0001f4bb \U00010348.\n"
    "partial-looking text: `unfinished"
)


def _load_tokenizer() -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL, local_files_only=True)


@requires_tokenizer
class TranscriptReconciliationTests(unittest.TestCase):
    """Decoded assistant text cannot always reproduce the generated tokens.

    The cached stream is ``first_prompt + generated_ids``.  The next request
    renders the decoded assistant text through the chat template; after
    reconciliation with the authoritative IDs, the cached stream must be an
    exact prefix of the new canonical prompt.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = _load_tokenizer()
        cls.eos_ids = chat.eos_token_ids(cls.tokenizer)

    def _encode(self, text: str) -> list[int]:
        return chat.normalize_token_ids(self.tokenizer.encode(text, add_special_tokens=False))

    def _continuation(self, generated: list[int]) -> tuple[list[int], dict[str, Any]]:
        first = [{"role": "user", "content": "Produce a mixed prose/code/Unicode response."}]
        first_prompt = chat.canonical_prompt(self.tokenizer, first)
        cached = list(first_prompt["token_ids"]) + generated
        history = [
            chat.HistoryTurn(
                user_text=first[0]["content"],
                prompt_token_count=first_prompt["token_count"],
                generated_ids=generated,
                assistant_text=self.tokenizer.decode(generated, skip_special_tokens=True),
            )
        ]
        following = [
            *first,
            {"role": "assistant", "content": history[0].assistant_text},
            {"role": "user", "content": "Continue from the exact previous response."},
        ]
        raw = chat.canonical_prompt(self.tokenizer, following)
        return cached, {
            "raw": raw,
            "reconciled": chat.reconcile_canonical_history(raw, history, self.eos_ids),
        }

    def _assert_exact_continuation(self, generated: list[int]) -> dict[str, Any]:
        cached, prompts = self._continuation(generated)
        suffix = chat.exact_suffix(prompts["reconciled"]["token_ids"], cached)
        self.assertTrue(suffix, "continuation must add the next user turn")
        return prompts

    def test_length_budgets_over_stress_text(self) -> None:
        stream = self._encode(STRESS_TEXT)
        for budget in LENGTH_BUDGETS:
            with self.subTest(budget=budget):
                generated = (stream * (budget // len(stream) + 1))[:budget]
                self._assert_exact_continuation(generated)

    def test_trailing_newline_trimmed_by_template(self) -> None:
        # Original HTTP 409 cause: the assistant template trims content, so a
        # length-truncated turn ending in "\n" re-renders with <|im_end|> where
        # the cache holds the newline token.
        generated = self._encode("First step:\n")
        self.assertEqual(self.tokenizer.decode(generated[-1:]), "\n")
        cached, prompts = self._continuation(generated)
        with self.assertRaises(RuntimeError):
            chat.exact_suffix(prompts["raw"]["token_ids"], cached)
        self._assert_exact_continuation(generated)

    def test_partial_utf8_sequence(self) -> None:
        generated = [9008, 239, 102, 373]
        decoded = self.tokenizer.decode(generated, skip_special_tokens=True)
        self.assertIn("\ufffd", decoded)
        self.assertNotEqual(self._encode(decoded), generated)
        self._assert_exact_continuation(generated)

    def test_eos_terminated_turn(self) -> None:
        eos = sorted(self.eos_ids)[0]
        self._assert_exact_continuation(self._encode("complete") + [eos])


@requires_tokenizer
class SessionRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = _load_tokenizer()

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

    def test_trailing_newline_length_turn_reuses_exact_state(self) -> None:
        session = self._session()
        newline_turn = chat.normalize_token_ids(
            self.tokenizer.encode("Step one:\n", add_special_tokens=False)
        )
        self.model.responses = [newline_turn, [71, 72]]
        first = self._complete(
            session,
            [{"role": "user", "content": "start a list"}],
            max_tokens=len(newline_turn),
        )
        self.assertEqual(first["finish_reason"], "length")
        second = self._complete(
            session,
            [*session.messages, {"role": "user", "content": "continue"}],
            max_tokens=2,
        )
        self.assertEqual(second["cache_reuse_mode"], "exact_reuse")
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



@requires_tokenizer
class HttpRoundTripTests(unittest.TestCase):
    """Drive the real HTTP handler with the fake generator."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = _load_tokenizer()

    def setUp(self) -> None:
        SessionRecoveryTests.setUp(self)  # type: ignore[arg-type]
        self.server = agent_server.AgentHTTPServer(
            ("127.0.0.1", 0),
            model=self.model,
            tokenizer=self.tokenizer,
            model_name="qwen38-test",
            default_max_tokens=4,
        )
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        SessionRecoveryTests.tearDown(self)  # type: ignore[arg-type]

    def _request(self, path: str, body: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            self.base + path,
            data=data,
            method="GET" if body is None else "POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_truncated_turn_continues_over_http(self) -> None:
        self.assertEqual(self._request("/health")[0], 200)
        messages = [{"role": "user", "content": "emit a marker"}]
        status, first = self._request(
            "/v1/chat/completions", {"messages": messages, "session_id": "s"}
        )
        self.assertEqual(status, 200)
        choice = first["choices"][0]
        self.assertEqual(choice["finish_reason"], "length")
        messages += [choice["message"], {"role": "tool", "content": "tool output"}]
        status, second = self._request(
            "/v1/chat/completions", {"messages": messages, "session_id": "s", "max_tokens": 2}
        )
        self.assertEqual(status, 200, second)
        self.assertEqual(second["x_qwen38_metrics"]["cache_reuse_mode"], "exact_reuse")
        status, metrics = self._request("/metrics")
        self.assertEqual(metrics["active_session"]["truncation_continuations"], 1)
        self.assertEqual(metrics["active_session"]["cache_conflict_error"], 0)
        status, reset = self._request("/v1/sessions/s/reset", {})
        self.assertEqual((status, reset["reset"]), (200, True))


if __name__ == "__main__":
    unittest.main()
