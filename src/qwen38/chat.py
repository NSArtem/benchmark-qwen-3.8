#!/usr/bin/env python3
"""Single-user Qwen3.8 chat with validated exact-suffix cache reuse.

The model weights remain resident for the lifetime of one process.  The
logical conversation is rendered with the canonical tokenizer chat template on
every turn, while only the suffix after the measured cache offset is passed to
``generate_step``.  If the cache offset or token prefix is not exactly what
the logical transcript predicts, this runtime raises instead of silently
replaying or corrupting state.

This is intentionally a CLI, not a multi-user server.  It never changes MLX
or macOS wired-memory settings.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


mx: Any = None
load: Any = None
generate_step: Any = None
make_prompt_cache: Any = None
make_sampler: Any = None


def _require_mlx() -> tuple[Any, Any, Any, Any, Any]:
    """Load MLX lazily so token/state regression tests need no Metal device."""

    global mx, load, generate_step, make_prompt_cache, make_sampler
    if mx is None:
        import mlx.core as mlx_core
        from mlx_lm import load as mlx_load
        from mlx_lm.generate import generate_step as mlx_generate_step
        from mlx_lm.models.cache import make_prompt_cache as mlx_make_prompt_cache
        from mlx_lm.sample_utils import make_sampler as mlx_make_sampler

        mx = mlx_core
        load = mlx_load
        generate_step = mlx_generate_step
        make_prompt_cache = mlx_make_prompt_cache
        make_sampler = mlx_make_sampler
    return mx, load, generate_step, make_prompt_cache, make_sampler


PINNED_MLX_LM_REVISION = "254d153fdeb6f150edd4fc5a54f9828638481fa8"
DEFAULT_PREFILL_STEP_SIZE = 2048
DEFAULT_MAX_TOKENS = 512
MAX_ASSISTANT_VARIANTS = 4096


def normalize_token_ids(value: Any) -> list[int]:
    """Convert tokenizer output variants to one flat integer list."""

    if hasattr(value, "encodings") and value.encodings:
        return [int(item) for item in value.encodings[0].ids]
    if isinstance(value, dict):
        value = value["input_ids"]
    elif hasattr(value, "input_ids"):
        value = value.input_ids
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, (list, tuple)) and value and isinstance(
        value[0], (list, tuple)
    ):
        value = value[0]
    return [int(item) for item in value]


def eos_token_ids(tokenizer: Any) -> set[int]:
    """Return EOS IDs across HF and mlx-lm tokenizer wrapper variants."""

    value = getattr(tokenizer, "eos_token_ids", None)
    if value is None:
        value = getattr(tokenizer, "eos_token_id", None)
    if value is None:
        return set()
    if isinstance(value, int):
        return {int(value)}
    return {int(item) for item in value}


def canonical_prompt(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    enable_thinking: bool = False,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Render and cross-check the canonical Qwen chat-template token stream.

    ``tools`` are rendered by the template into the system block, so they are
    part of the prompt identity.
    """

    template_kwargs: dict[str, Any] = {"enable_thinking": enable_thinking}
    if tools:
        template_kwargs["tools"] = tools
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )
    encoded_from_text = normalize_token_ids(
        tokenizer.encode(rendered, add_special_tokens=False)
    )
    encoded_direct = normalize_token_ids(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            **template_kwargs,
        )
    )
    if encoded_from_text != encoded_direct:
        raise RuntimeError("chat-template string/tokenized forms disagree")
    return {
        "rendered_text": rendered,
        "token_ids": encoded_from_text,
        "token_count": len(encoded_from_text),
    }


def exact_suffix(full_token_ids: Sequence[int], cached_token_ids: Sequence[int]) -> list[int]:
    """Return the only safe suffix when the cached stream is an exact prefix."""

    full = list(full_token_ids)
    cached = list(cached_token_ids)
    if len(cached) > len(full) or full[: len(cached)] != cached:
        raise RuntimeError("logical transcript is not an exact prefix of the cache")
    return full[len(cached) :]


def response_replay_detected(current_token_ids: Sequence[int], previous_token_ids: Sequence[int]) -> bool:
    """Detect a response that is an exact prefix replay of an older response."""

    current = list(current_token_ids)
    previous = list(previous_token_ids)
    # A short identical answer (for example, "OK") is not evidence of stale
    # cache replay. Require a meaningful token stream before raising the
    # diagnostic flag; exact-suffix accounting remains the correctness gate.
    return len(current) >= 8 and current == previous[: len(current)]


@dataclass
class HistoryTurn:
    user_text: str
    prompt_token_count: int
    generated_ids: list[int]
    assistant_text: str


def reconcile_canonical_history(
    canonical: dict[str, Any],
    history: Sequence[HistoryTurn],
    eos_ids: set[int],
) -> dict[str, Any]:
    """Restore authoritative generated IDs inside a rendered transcript.

    The API-facing assistant string is deliberately decoded text.  That text
    is not always a lossless representation of the generated token stream:
    the Qwen template trims assistant content, and a byte-pair stream may be
    decoded with a replacement character when a length boundary cuts through
    a partial UTF-8 sequence.  The cache, however, already contains the exact
    IDs.  Each ``HistoryTurn.prompt_token_count`` identifies the beginning of
    its generated region in that exact stream, so replace the corresponding
    rendered assistant-content region with the stored IDs before asking for a
    suffix.

    This function is intentionally token-only.  It does not infer equality
    from decoded text and it keeps one template EOS boundary when a generated
    stream itself ended in EOS.  ``cache_boundary_adjustments`` is diagnostic
    evidence, not a correctness input.
    """

    token_ids = list(canonical["token_ids"])
    adjustments: list[dict[str, Any]] = list(
        canonical.get("cache_boundary_adjustments", [])
    )
    for turn_index, turn in enumerate(history):
        generated = [int(value) for value in turn.generated_ids]
        if not generated:
            continue
        # ``token_ids`` is edited in place, so everything before this turn is
        # already in authoritative (cache) coordinates, which is also what
        # ``prompt_token_count`` measures.  No offset correction applies.
        start = int(turn.prompt_token_count)
        if start < 0 or start > len(token_ids):
            raise RuntimeError(
                f"history turn {turn_index} prompt boundary {start} is outside the canonical stream"
            )

        boundary = next(
            (index for index in range(start, len(token_ids)) if int(token_ids[index]) in eos_ids),
            None,
        )
        if boundary is None:
            raise RuntimeError(
                f"history turn {turn_index} has no assistant template boundary after token {start}"
            )

        generated_ends_eos = int(generated[-1]) in eos_ids
        replacement = generated[:-1] if generated_ends_eos else generated
        old_segment = token_ids[start:boundary]
        if old_segment != replacement:
            token_ids[start:boundary] = replacement
            # The template's close token moved with the splice.
            boundary = start + len(replacement)
            adjustments.append(
                {
                    "turn_index": turn_index,
                    "start": start,
                    "old_length": len(old_segment),
                    "new_length": len(replacement),
                    "reason": "authoritative_generated_ids_replace_rendered_assistant_content",
                }
            )

        # The template supplies the assistant-close token.  If generation
        # ended in EOS, that one token is the generated boundary as well.  A
        # template that emitted an additional identical EOS is normalized to a
        # single token so the cached stream stays an exact prefix.
        if generated_ends_eos:
            if boundary >= len(token_ids):
                raise RuntimeError(
                    f"history turn {turn_index} lost its assistant EOS boundary"
                )
            if int(token_ids[boundary]) != int(generated[-1]):
                token_ids[boundary] = int(generated[-1])
                adjustments.append(
                    {
                        "turn_index": turn_index,
                        "index": boundary,
                        "reason": "authoritative_generated_EOS_replaced_template_boundary",
                    }
                )
            if (
                boundary + 1 < len(token_ids)
                and int(token_ids[boundary + 1]) == int(generated[-1])
            ):
                del token_ids[boundary + 1]
                adjustments.append(
                    {
                        "turn_index": turn_index,
                        "removed_index": boundary + 1,
                        "removed_token_id": int(generated[-1]),
                        "reason": "duplicate_template_EOS_after_authoritative_generation",
                    }
                )

    adjusted = dict(canonical)
    adjusted["token_ids"] = token_ids
    adjusted["token_count"] = len(token_ids)
    adjusted["cache_boundary_adjustments"] = adjustments
    return adjusted


class TokenLedger:
    """Small token-only state machine used by the production session and tests."""

    def __init__(self) -> None:
        self.cached_token_ids: list[int] = []

    def suffix_for(self, full_token_ids: Sequence[int]) -> list[int]:
        return exact_suffix(full_token_ids, self.cached_token_ids)

    def commit(
        self,
        prompt_token_ids: Sequence[int],
        generated_token_ids: Sequence[int],
    ) -> None:
        self.cached_token_ids = list(prompt_token_ids) + list(generated_token_ids)

    def reset(self) -> None:
        self.cached_token_ids.clear()


def _cache_offset(cache: Sequence[Any]) -> int:
    """Read one consistent logical offset from all hybrid cache layers."""

    offsets: list[int] = []
    for entry in cache:
        try:
            value = getattr(entry, "offset")
            value = value() if callable(value) else value
            offsets.append(int(value))
        except BaseException:
            continue
    unique = sorted(set(offsets))
    if not unique:
        raise RuntimeError("cache offset unavailable; refusing unsafe suffix reuse")
    if len(unique) != 1:
        raise RuntimeError(f"cache layer offsets disagree: {unique}")
    return unique[0]


def _memory_snapshot() -> dict[str, int | None]:
    mlx_core, _, _, _, _ = _require_mlx()
    try:
        mlx_core.synchronize()
    except BaseException:
        pass
    values: dict[str, int | None] = {}
    for name, getter in (
        ("active_memory_bytes", mlx_core.get_active_memory),
        ("cache_memory_bytes", mlx_core.get_cache_memory),
        ("peak_memory_bytes", mlx_core.get_peak_memory),
    ):
        try:
            values[name] = int(getter())
        except BaseException:
            values[name] = None
    if values["active_memory_bytes"] is not None and values["cache_memory_bytes"] is not None:
        values["active_plus_cache_bytes"] = (
            values["active_memory_bytes"] + values["cache_memory_bytes"]
        )
    else:
        values["active_plus_cache_bytes"] = None
    return values


@dataclass
class GeneratedTokens:
    """Raw outcome of one direct ``generate_step`` run."""

    token_ids: list[int]
    stop: str  # "eos", "length", or "generator_stop"
    started: float
    first_token_time: float | None
    completed: float

    @property
    def ttft_seconds(self) -> float | None:
        if self.first_token_time is None:
            return None
        return self.first_token_time - self.started

    @property
    def generation_seconds(self) -> float | None:
        if self.first_token_time is None:
            return None
        return self.completed - self.first_token_time

    @property
    def decode_tps(self) -> float | None:
        seconds = self.generation_seconds
        return max(len(self.token_ids) - 1, 0) / seconds if seconds else None


def generate_tokens(
    model: Any,
    prompt_cache: Sequence[Any],
    suffix_ids: Sequence[int],
    *,
    max_tokens: int,
    sampler: Any,
    eos_ids: set[int],
    prefill_step_size: int = DEFAULT_PREFILL_STEP_SIZE,
    kv_bits: int | None = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
    on_token: Any = None,
) -> GeneratedTokens:
    """Feed ``suffix_ids`` into the persistent cache and decode new tokens.

    This is the single generation loop shared by the CLI and the server. It
    calls pinned ``generate_step`` directly (never the wired-limit wrapper).
    ``on_token(token_ids)`` is called after every generated token with the
    full list so far; an exception from it stops generation and propagates,
    leaving the cache advanced past the committed ledger (callers must then
    discard the cache).
    """

    mlx_core, _, mlx_generate_step, _, _ = _require_mlx()
    started = time.perf_counter()
    generator = mlx_generate_step(
        mlx_core.array(list(suffix_ids)),
        model,
        max_tokens=max_tokens,
        sampler=sampler,
        prompt_cache=prompt_cache,
        prefill_step_size=prefill_step_size,
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
        quantized_kv_start=quantized_kv_start,
    )
    generated_ids: list[int] = []
    first_token_time: float | None = None
    stop = "length"
    try:
        while len(generated_ids) < max_tokens:
            try:
                token, _logprobs = next(generator)
            except StopIteration:
                stop = "generator_stop"
                break
            if first_token_time is None:
                first_token_time = time.perf_counter()
            token_id = int(token.item()) if hasattr(token, "item") else int(token)
            generated_ids.append(token_id)
            if on_token is not None:
                on_token(generated_ids)
            if token_id in eos_ids:
                stop = "eos"
                break
    finally:
        generator.close()
    completed = time.perf_counter()
    try:
        mlx_core.synchronize()
    except BaseException:
        pass
    return GeneratedTokens(generated_ids, stop, started, first_token_time, completed)


@dataclass
class ExactSuffixTurn:
    text: str
    prompt_tokens: int
    processed_suffix_tokens: int
    generated_tokens: int
    finish_reason: str
    ttft_seconds: float | None
    prompt_tps: float | None
    decode_tps: float | None
    generation_seconds: float | None
    total_seconds: float
    cache_offset_before: int
    cache_offset_after: int
    logical_context_tokens: int
    replay_detected: bool
    token_ids: list[int]
    active_memory_bytes: int | None
    cache_memory_bytes: int | None
    active_plus_cache_bytes: int | None
    peak_memory_bytes: int | None


class ExactSuffixChat:
    """One logical conversation backed by one validated persistent cache."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        system_prompt: str | None = None,
        max_kv_size: int | None = None,
        enable_thinking: bool = False,
    ) -> None:
        _, _, _, mlx_make_prompt_cache, _ = _require_mlx()
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.max_kv_size = max_kv_size
        self.enable_thinking = enable_thinking
        self._max_kv_size = max_kv_size
        self.kv_bits: int | None = None
        self.ledger = TokenLedger()
        self.history: list[HistoryTurn] = []
        self.prompt_cache = mlx_make_prompt_cache(model, max_kv_size)

    @property
    def logical_context_tokens(self) -> int:
        return len(self.ledger.cached_token_ids)

    def reset(self) -> None:
        """Drop both logical transcript and all hybrid cache state."""

        _, _, _, mlx_make_prompt_cache, _ = _require_mlx()
        self.ledger.reset()
        self.history.clear()
        self.prompt_cache = mlx_make_prompt_cache(self.model, self._max_kv_size)

    def stats(self) -> dict[str, Any]:
        return {
            "turns": len(self.history),
            "logical_context_tokens": self.logical_context_tokens,
            "cached_prefix_tokens": self.logical_context_tokens,
            "kv_bits": self.kv_bits,
            "cache_layers": len(self.prompt_cache),
        }

    def _base_messages(self) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        return messages

    def _messages_for_candidate(
        self,
        query: str,
        assistant_texts: Sequence[str],
    ) -> list[dict[str, str]]:
        messages = self._base_messages()
        for history, assistant_text in zip(self.history, assistant_texts):
            messages.append({"role": "user", "content": history.user_text})
            messages.append({"role": "assistant", "content": assistant_text})
        messages.append({"role": "user", "content": query})
        return messages

    def _assistant_variants(self, turn: HistoryTurn) -> list[str]:
        variants: list[str] = []
        for skip_special_tokens in (True, False):
            text = self.tokenizer.decode(
                turn.generated_ids,
                skip_special_tokens=skip_special_tokens,
            )
            if text not in variants:
                variants.append(text)
        return variants

    def _select_canonical(
        self,
        query: str,
    ) -> tuple[dict[str, Any], list[str], int]:
        actual_before = _cache_offset(self.prompt_cache)
        expected_before = len(self.ledger.cached_token_ids)
        if actual_before != expected_before:
            raise RuntimeError(
                f"cache offset {actual_before} != logical prefix {expected_before}"
            )

        if not self.history:
            assistant_texts: list[str] = []
            candidates: Iterable[Sequence[str]] = (assistant_texts,)
        else:
            primary = tuple(turn.assistant_text for turn in self.history)
            variant_lists = [self._assistant_variants(turn) for turn in self.history]
            candidates_list: list[Sequence[str]] = [primary]
            for candidate in itertools.product(*variant_lists):
                if tuple(candidate) != primary:
                    candidates_list.append(candidate)
                if len(candidates_list) >= MAX_ASSISTANT_VARIANTS:
                    break
            if len(candidates_list) >= MAX_ASSISTANT_VARIANTS and not any(
                tuple(candidate) == primary for candidate in candidates_list
            ):
                raise RuntimeError("too many assistant token-boundary variants")
            candidates = candidates_list

        eos_ids = eos_token_ids(self.tokenizer)
        for assistant_texts in candidates:
            messages = self._messages_for_candidate(query, assistant_texts)
            candidate = canonical_prompt(
                self.tokenizer,
                messages,
                enable_thinking=self.enable_thinking,
            )
            candidate = reconcile_canonical_history(candidate, self.history, eos_ids)
            token_ids = candidate["token_ids"]
            if token_ids[:actual_before] == self.ledger.cached_token_ids:
                for history, selected_text in zip(self.history, assistant_texts):
                    history.assistant_text = selected_text
                return candidate, list(assistant_texts), actual_before
        raise RuntimeError(
            "no canonical transcript candidate begins with the measured cache stream"
        )

    def turn(
        self,
        query: str,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
        top_p: float = 1.0,
        kv_bits: int | None = None,
        kv_group_size: int = 64,
        quantized_kv_start: int = 0,
        prefill_step_size: int = DEFAULT_PREFILL_STEP_SIZE,
    ) -> ExactSuffixTurn:
        _, _, _, _, mlx_make_sampler = _require_mlx()
        if not query.strip():
            raise ValueError("empty user messages are not sent to the model")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.kv_bits = kv_bits
        canonical, _, cache_offset_before = self._select_canonical(query)
        suffix_ids = exact_suffix(canonical["token_ids"], self.ledger.cached_token_ids)
        result = generate_tokens(
            self.model,
            self.prompt_cache,
            suffix_ids,
            max_tokens=max_tokens,
            sampler=mlx_make_sampler(temperature, top_p),
            eos_ids=eos_token_ids(self.tokenizer),
            prefill_step_size=prefill_step_size,
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
            quantized_kv_start=quantized_kv_start,
        )
        generated_ids = result.token_ids
        finish_reason = result.stop
        started, completed = result.started, result.completed
        cache_offset_after = _cache_offset(self.prompt_cache)
        expected_after = len(canonical["token_ids"]) + len(generated_ids)
        if cache_offset_after != expected_after:
            raise RuntimeError(
                f"cache after-generation offset {cache_offset_after} != expected {expected_after}"
            )
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        ttft = result.ttft_seconds
        generation_seconds = result.generation_seconds
        decode_tps = result.decode_tps
        prompt_tps = len(suffix_ids) / ttft if ttft else None
        replay_was_detected = any(
            response_replay_detected(generated_ids, previous.generated_ids)
            for previous in self.history
        )
        memory = _memory_snapshot()
        self.ledger.commit(canonical["token_ids"], generated_ids)
        assistant_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )
        self.history.append(
            HistoryTurn(
                user_text=query,
                prompt_token_count=len(canonical["token_ids"]),
                generated_ids=list(generated_ids),
                assistant_text=assistant_text,
            )
        )
        return ExactSuffixTurn(
            text=text,
            prompt_tokens=len(canonical["token_ids"]),
            processed_suffix_tokens=len(suffix_ids),
            generated_tokens=len(generated_ids),
            finish_reason=finish_reason,
            ttft_seconds=ttft,
            prompt_tps=prompt_tps,
            decode_tps=decode_tps,
            generation_seconds=generation_seconds,
            total_seconds=completed - started,
            cache_offset_before=cache_offset_before,
            cache_offset_after=cache_offset_after,
            logical_context_tokens=len(self.ledger.cached_token_ids),
            replay_detected=replay_was_detected,
            token_ids=generated_ids,
            active_memory_bytes=memory["active_memory_bytes"],
            cache_memory_bytes=memory["cache_memory_bytes"],
            active_plus_cache_bytes=memory["active_plus_cache_bytes"],
            peak_memory_bytes=memory["peak_memory_bytes"],
        )


def _sampling_defaults(args: argparse.Namespace) -> tuple[float, float]:
    if args.temperature is not None:
        temperature = args.temperature
    else:
        temperature = 0.0 if args.mode == "greedy" else 0.7
    if args.top_p is not None:
        top_p = args.top_p
    else:
        top_p = 1.0 if args.mode == "greedy" else 0.9
    return temperature, top_p


def _metric_line(turn: ExactSuffixTurn) -> str:
    def gib(value: int | None) -> str:
        return "n/a" if value is None else f"{value / (1024**3):.3f}"

    def seconds(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    def rate(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.2f}"

    return (
        "[metrics] "
        f"context={turn.logical_context_tokens} "
        f"suffix={turn.processed_suffix_tokens} "
        f"ttft={seconds(turn.ttft_seconds)}s "
        f"decode={rate(turn.decode_tps)} tok/s "
        f"active+cache={gib(turn.active_plus_cache_bytes)} GiB "
        f"peak={gib(turn.peak_memory_bytes)} GiB "
        f"finish={turn.finish_reason} replay={turn.replay_detected}"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Single-user Qwen3.8 exact-suffix persistent chat"
    )
    parser.add_argument("--model", required=True, help="MLX model directory")
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--mode", choices=("greedy", "sample"), default="greedy")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--prefill-step-size", type=int, default=DEFAULT_PREFILL_STEP_SIZE)
    parser.add_argument("--max-kv-size", type=int, default=None)
    parser.add_argument("--kv-bits", type=int, choices=(4, 8), default=None)
    parser.add_argument("--kv-group-size", type=int, default=64)
    parser.add_argument("--quantized-kv-start", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    temperature, top_p = _sampling_defaults(args)
    print(
        "[qwen38-chat] loading model; wired-memory settings untouched",
        file=sys.stderr,
        flush=True,
    )
    _, mlx_load, _, _, _ = _require_mlx()
    model, tokenizer = mlx_load(args.model)
    session = ExactSuffixChat(
        model,
        tokenizer,
        system_prompt=args.system_prompt,
        max_kv_size=args.max_kv_size,
        enable_thinking=args.enable_thinking,
    )
    print(
        "[qwen38-chat] ready; /reset clears transcript/cache, /stats reports context, /quit exits",
        file=sys.stderr,
        flush=True,
    )
    try:
        for line in sys.stdin:
            query = line.rstrip("\r\n")
            if query == "/quit" or query == "/exit":
                break
            if query == "/reset":
                session.reset()
                print("[qwen38-chat] reset", file=sys.stderr, flush=True)
                continue
            if query in ("/stats", "/context"):
                print(
                    "[qwen38-chat] "
                    + json.dumps(session.stats(), sort_keys=True),
                    file=sys.stderr,
                    flush=True,
                )
                continue
            if not query.strip():
                continue
            try:
                turn = session.turn(
                    query,
                    max_tokens=args.max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    kv_bits=args.kv_bits,
                    kv_group_size=args.kv_group_size,
                    quantized_kv_start=args.quantized_kv_start,
                    prefill_step_size=args.prefill_step_size,
                )
            except KeyboardInterrupt:
                session.reset()
                print(
                    "[qwen38-chat] interrupted; session reset for safety",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            except BaseException as exc:
                session.reset()
                print(
                    f"[qwen38-chat] generation failed ({type(exc).__name__}: {exc}); session reset",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            print(turn.text, flush=True)
            print(_metric_line(turn), file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        session.reset()
        print("[qwen38-chat] interrupted; session reset", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
