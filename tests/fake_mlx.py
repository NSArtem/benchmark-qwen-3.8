"""Fake MLX runtime for session/HTTP tests: real tokenizer, scripted tokens.

The fake ``generate_step`` advances a one-layer fake cache exactly like the
real hybrid cache is checked: by ``len(prompt)`` on prefill and by one per
generated token.  Responses are queued token-ID lists.
"""

from __future__ import annotations

import os
import time
import unittest
from pathlib import Path
from typing import Any

import qwen38.chat as chat
import qwen38.server as agent_server

MODEL = os.environ.get("QWEN38_MODEL", "/Users/artem/models/Qwen3.8-27B-3bit-g128-MLX")

requires_tokenizer = unittest.skipUnless(
    (Path(MODEL) / "tokenizer.json").is_file(),
    f"Qwen3.8 tokenizer not found at {MODEL}; set QWEN38_MODEL",
)

_TOKENIZER: Any = None


def load_tokenizer() -> Any:
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    return _TOKENIZER


class _Scalar:
    def __init__(self, value: int) -> None:
        self.value = int(value)

    def item(self) -> int:
        return self.value


class FakeCache:
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


class FakeModel:
    def __init__(self, responses: list[list[int]] | None = None) -> None:
        self.responses: list[list[int]] = list(responses or [])
        self.prompts: list[list[int]] = []
        self.token_delay = 0.0
        self.prefill_delay = 0.0
        self.fail_after: int | None = None
        self.tokens_yielded = 0


class FakeMLX:
    """Install/uninstall the fake runtime into ``qwen38.chat`` and ``qwen38.server``."""

    def __init__(self) -> None:
        self.core = _Core()

        def make_prompt_cache(_model: Any, _max_kv_size: int | None) -> list[FakeCache]:
            return [FakeCache()]

        def generate_step(prompt: list[int], model: FakeModel, *, max_tokens: int, prompt_cache: list[FakeCache], **_kw: Any):
            response = list(model.responses.pop(0))[:max_tokens]
            model.prompts.append(list(prompt))

            def iterator():
                if model.prefill_delay:
                    time.sleep(model.prefill_delay)
                prompt_cache[0].offset += len(prompt)
                for index, token_id in enumerate(response):
                    if model.fail_after is not None and index >= model.fail_after:
                        raise MemoryError("fake Metal allocation failure")
                    if model.token_delay:
                        time.sleep(model.token_delay)
                    prompt_cache[0].offset += 1
                    model.tokens_yielded += 1
                    yield _Scalar(token_id), None

            return iterator()

        self.runtime = (self.core, None, generate_step, make_prompt_cache, lambda _t, _p: object())

    def install(self) -> None:
        self._old = (chat._require_mlx, agent_server._require_mlx)
        chat._require_mlx = lambda: self.runtime
        agent_server._require_mlx = lambda: self.runtime

    def uninstall(self) -> None:
        chat._require_mlx, agent_server._require_mlx = self._old


def encode(tokenizer: Any, text: str, eos: bool = True) -> list[int]:
    ids = chat.normalize_token_ids(tokenizer.encode(text, add_special_tokens=False))
    return ids + [int(tokenizer.eos_token_id)] if eos else ids


def complete(session: Any, messages: list[dict[str, Any]], max_tokens: int = 256, **kwargs: Any) -> dict[str, Any]:
    return session.complete(
        messages,
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        kv_bits=None,
        kv_group_size=64,
        quantized_kv_start=0,
        prefill_step_size=128,
        **kwargs,
    )
