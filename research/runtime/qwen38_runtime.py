#!/usr/bin/env python3
"""Small MLX 0.32.1 runtime with mlx-lm's wired-limit policy disabled.

This module deliberately calls the pinned ``generate_step`` API directly.  It
does not modify mlx, mlx-lm, or the process-wide wired limit.  The chat session
keeps one prompt cache and follows the pinned mlx-lm chat behavior: each turn
passes only the new user message while the cache carries prior turns.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import pathlib
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Generator, Optional, Union

import mlx.core as mx

from mlx_lm import load
from mlx_lm.generate import (
    GenerationResponse,
    generate_step,
    speculative_generate_step,
)
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper


PINNED_MLX_LM_REVISION = "254d153fdeb6f150edd4fc5a54f9828638481fa8"
STARTUP_MESSAGE = (
    "[qwen38-runtime] mlx-lm wired_limit wrapper disabled; "
    "MLX wired state untouched"
)


def stream_generate_no_wired(
    model: Any,
    tokenizer: Any,
    prompt: Union[str, mx.array, list[int]],
    max_tokens: int = 256,
    draft_model: Optional[Any] = None,
    **kwargs: Any,
) -> Generator[GenerationResponse, None, None]:
    """Yield stock-style responses without entering mlx-lm's wired context."""

    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    if not isinstance(prompt, mx.array):
        if isinstance(prompt, str):
            add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(
                tokenizer.bos_token
            )
            prompt = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        prompt = mx.array(prompt)

    detokenizer = tokenizer.detokenizer
    kwargs["max_tokens"] = max_tokens

    if draft_model is None:
        kwargs.pop("num_draft_tokens", None)
        token_generator = generate_step(prompt, model, **kwargs)
        token_generator = (
            (token, logprobs, False) for token, logprobs in token_generator
        )
    else:
        kwargs.pop("max_kv_size", None)
        kwargs.pop("prompt_progress_callback", None)
        token_generator = speculative_generate_step(
            prompt, model, draft_model, **kwargs
        )

    tic = time.perf_counter()
    prompt_tps: Optional[float] = None
    generation_tps: Optional[float] = None
    token = None
    logprobs = None
    from_draft = False
    for n, (token, logprobs, from_draft) in enumerate(token_generator):
        if n == 0:
            prompt_time = time.perf_counter() - tic
            prompt_tps = prompt.size / prompt_time
            tic = time.perf_counter()
        if token in tokenizer.eos_token_ids:
            break

        detokenizer.add_token(token)
        if (n + 1) == max_tokens:
            break

        generation_tps = (n + 1) / (time.perf_counter() - tic)
        yield GenerationResponse(
            text=detokenizer.last_segment,
            token=token,
            logprobs=logprobs,
            from_draft=from_draft,
            prompt_tokens=prompt.size,
            prompt_tps=prompt_tps,
            generation_tokens=n + 1,
            generation_tps=generation_tps,
            peak_memory=mx.get_peak_memory() / 1e9,
            finish_reason=None,
        )

    if token is None or logprobs is None or prompt_tps is None:
        return
    detokenizer.finalize()
    generation_tps = (n + 1) / (time.perf_counter() - tic)
    yield GenerationResponse(
        text=detokenizer.last_segment,
        token=token,
        logprobs=logprobs,
        from_draft=from_draft,
        prompt_tokens=prompt.size,
        prompt_tps=prompt_tps,
        generation_tokens=n + 1,
        generation_tps=generation_tps,
        peak_memory=mx.get_peak_memory() / 1e9,
        finish_reason="stop" if token in tokenizer.eos_token_ids else "length",
    )


def generate_no_wired(*args: Any, **kwargs: Any) -> Generator[GenerationResponse, None, None]:
    """Named one-shot entry point; equivalent to ``stream_generate_no_wired``."""

    yield from stream_generate_no_wired(*args, **kwargs)


@dataclass
class ChatTurn:
    text: str
    prompt_tokens: int
    prompt_tps: Optional[float]
    generated_tokens: int
    generation_tps: Optional[float]
    peak_memory_gb: Optional[float]
    ttft_seconds: Optional[float]
    generation_seconds: Optional[float]
    total_seconds: float
    token_ids: list[int]


class NoWiredChat:
    """Persistent-cache chat session using the pinned mlx-lm turn protocol."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        system_prompt: Optional[str] = None,
        max_kv_size: Optional[int] = None,
        enable_thinking: bool = False,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.enable_thinking = enable_thinking
        self.max_kv_size = max_kv_size
        self.prompt_cache = make_prompt_cache(model, max_kv_size)

    def reset(self) -> None:
        self.prompt_cache = make_prompt_cache(self.model, self.max_kv_size)

    def turn(
        self,
        query: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        kv_bits: Optional[int] = None,
        kv_group_size: int = 64,
        quantized_kv_start: int = 0,
        prefill_step_size: int = 2048,
    ) -> ChatTurn:
        messages = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": query})
        prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        sampler = make_sampler(temperature, top_p)
        started = time.perf_counter()
        first_token_time: Optional[float] = None
        completed_responses = []
        token_ids: list[int] = []
        for response in stream_generate_no_wired(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            prompt_cache=self.prompt_cache,
            prefill_step_size=prefill_step_size,
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
            quantized_kv_start=quantized_kv_start,
        ):
            if first_token_time is None:
                first_token_time = time.perf_counter()
            completed_responses.append(response)
            token_ids.append(int(response.token))
        completed = time.perf_counter()
        if not completed_responses:
            return ChatTurn("", len(prompt), None, 0, None, None, None, None, completed - started, [])
        last = completed_responses[-1]
        return ChatTurn(
            text="".join(response.text for response in completed_responses),
            prompt_tokens=last.prompt_tokens,
            prompt_tps=last.prompt_tps,
            generated_tokens=last.generation_tokens,
            generation_tps=last.generation_tps,
            peak_memory_gb=last.peak_memory,
            ttft_seconds=(first_token_time - started if first_token_time is not None else None),
            generation_seconds=(completed - first_token_time if first_token_time is not None else None),
            total_seconds=completed - started,
            token_ids=token_ids,
        )


def _chat_prompt(tokenizer: Any, content: str, enable_thinking: bool) -> Any:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def _sampler(args: argparse.Namespace) -> Callable[[mx.array], mx.array]:
    return make_sampler(args.temperature, args.top_p)


def _read_prompt(args: argparse.Namespace) -> str:
    if args.prompt is not None:
        return args.prompt
    return pathlib.Path(args.prompt_file).read_text(encoding="utf-8")


def _add_generation_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--max-kv-size", type=int, default=None)
    parser.add_argument("--kv-bits", type=int, default=None)
    parser.add_argument("--kv-group-size", type=int, default=64)
    parser.add_argument("--quantized-kv-start", type=int, default=0)


def _add_prompt_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt")
    group.add_argument("--prompt-file")
    parser.add_argument("--enable-thinking", action="store_true")


def _info(model_path: str) -> None:
    config_path = pathlib.Path(model_path) / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "python_version": sys.version.split()[0],
                "mlx_version": metadata.version("mlx"),
                "mlx_lm_metadata_version": metadata.version("mlx-lm"),
                "mlx_lm_revision": PINNED_MLX_LM_REVISION,
                "model_path": str(pathlib.Path(model_path).resolve()),
                "model_quantization": config.get(
                    "quantization", config.get("quantization_config")
                ),
                "wired_policy": "disabled wrapper; MLX wired state untouched",
            },
            indent=2,
        )
    )


def _main_generate(args: argparse.Namespace) -> None:
    print(STARTUP_MESSAGE, file=sys.stderr, flush=True)
    model, tokenizer = load(args.model)
    content = _read_prompt(args)
    prompt = _chat_prompt(tokenizer, content, args.enable_thinking)
    responses = list(
        generate_no_wired(
            model,
            tokenizer,
            prompt,
            max_tokens=args.max_tokens,
            sampler=_sampler(args),
            max_kv_size=args.max_kv_size,
            prefill_step_size=args.prefill_step_size,
            kv_bits=args.kv_bits,
            kv_group_size=args.kv_group_size,
            quantized_kv_start=args.quantized_kv_start,
        )
    )
    if not responses:
        return
    print("".join(response.text for response in responses))
    last = responses[-1]
    print(
        json.dumps(
            {
                "prompt_tokens": last.prompt_tokens,
                "prompt_tps": last.prompt_tps,
                "generated_tokens": last.generation_tokens,
                "generation_tps": last.generation_tps,
                "peak_memory_gb_decimal": last.peak_memory,
            },
            sort_keys=True,
        ),
        file=sys.stderr,
    )


def _main_chat(args: argparse.Namespace) -> None:
    print(STARTUP_MESSAGE, file=sys.stderr, flush=True)
    model, tokenizer = load(args.model)
    session = NoWiredChat(
        model,
        tokenizer,
        system_prompt=args.system_prompt,
        max_kv_size=args.max_kv_size,
        enable_thinking=args.enable_thinking,
    )
    print("qwen38 no-wired chat; enter /quit or /reset", file=sys.stderr)
    for line in sys.stdin:
        query = line.rstrip("\n")
        if query == "/quit":
            break
        if query == "/reset":
            session.reset()
            print("[chat reset]", file=sys.stderr, flush=True)
            continue
        turn = session.turn(
            query,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            kv_bits=args.kv_bits,
            kv_group_size=args.kv_group_size,
            quantized_kv_start=args.quantized_kv_start,
            prefill_step_size=args.prefill_step_size,
        )
        print(turn.text, flush=True)
        print(
            json.dumps(
                {key: value for key, value in turn.__dict__.items() if key != "token_ids"},
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.8 MLX no-wired runtime")
    subparsers = parser.add_subparsers(dest="command", required=True)

    info = subparsers.add_parser("info")
    info.add_argument("--model", required=True)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--model", required=True)
    _add_prompt_options(generate)
    _add_generation_options(generate)

    chat = subparsers.add_parser("chat")
    chat.add_argument("--model", required=True)
    chat.add_argument("--system-prompt", default=None)
    chat.add_argument("--enable-thinking", action="store_true")
    _add_generation_options(chat)

    args = parser.parse_args()
    if args.command == "info":
        _info(args.model)
    elif args.command == "generate":
        _main_generate(args)
    else:
        _main_chat(args)


if __name__ == "__main__":
    main()
