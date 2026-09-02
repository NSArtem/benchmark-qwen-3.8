#!/usr/bin/env python3
"""Run one or more short real-g128 continuation probes.

The script loads the target once, records the exact first-turn state, and then
attempts the next serialized request against the pre-fix session logic.  It
does not alter production code and never loads an MTP artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from mlx_lm import load

from research.runtime.qwen38_agent_server import AgentSession
from research.runtime.qwen38_chat import (
    HistoryTurn,
    canonical_prompt,
    exact_suffix,
    normalize_cache_boundary,
    normalize_token_ids,
    reconcile_canonical_history,
    _cache_offset,
)
from research.stage8.truncation.reproduce_truncation import eos_ids, token_context


DEFAULT_MODEL = "/Users/artem/models/Qwen3.8-27B-3bit-g128-MLX"
DEFAULT_PROMPT = (
    "Write a long Python code block that demonstrates a streaming parser. "
    "Include comments, indentation, and enough detail to reach the response "
    "length boundary; do not stop early."
)


def first_value_mismatch(left: list[int], right: list[int]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if int(a) != int(b):
            return index
    return None


def common_prefix(left: list[int], right: list[int]) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def jsonable_turn(session: AgentSession, result: dict[str, Any], followup: str) -> dict[str, Any]:
    record = session.records[-1]
    incoming = [*session.messages, {"role": "user", "content": followup}]
    raw = canonical_prompt(session.tokenizer, incoming, enable_thinking=session.enable_thinking)
    history = [
        HistoryTurn("", item.prompt_tokens, item.generated_ids, item.text)
        for item in session.records
    ]
    raw_adjusted = normalize_cache_boundary(raw, history, eos_ids(session.tokenizer))
    repaired = reconcile_canonical_history(
        raw,
        history,
        eos_ids(session.tokenizer),
    )
    cached = list(session.cached_token_ids)
    raw_reconstructed = list(raw_adjusted["token_ids"])
    reconstructed = list(repaired["token_ids"])
    mismatch = first_value_mismatch(cached, raw_reconstructed)
    if mismatch is None and len(cached) != len(raw_reconstructed):
        mismatch = min(len(cached), len(raw_reconstructed))
    lcp = common_prefix(cached, raw_reconstructed)
    exact_prefix = len(cached) <= len(raw_reconstructed) and cached == raw_reconstructed[: len(cached)]
    try:
        suffix = exact_suffix(raw_reconstructed, cached)
        old_error = None
        old_status = 200
    except RuntimeError as exc:
        suffix = None
        old_error = str(exc)
        old_status = 409
    prompt_count = int(result["prompt_tokens"])
    generated = list(record.generated_ids)
    generated_reencoded = normalize_token_ids(
        session.tokenizer.encode(record.text, add_special_tokens=False)
    )
    return {
        "finish_reason": result["finish_reason"],
        "first_request_metrics": result,
        "generated_token_ids": generated,
        "decoded_response_text": record.text,
        "reencoded_response_token_ids": generated_reencoded,
        "decode_reencode_identity": generated == generated_reencoded,
        "first_prompt_token_count": prompt_count,
        "cache_offset_after_first_request": _cache_offset(session.prompt_cache),
        "cache_represented_prefix_length": len(cached),
        "cached_token_ids": cached,
        "canonical_next_token_ids": reconstructed,
        "canonical_next_token_count": len(reconstructed),
        "raw_canonical_next_token_ids": raw_reconstructed,
        "raw_canonical_next_token_count": len(raw_reconstructed),
        "repaired_canonical_next_token_count": len(reconstructed),
        "post_fix_exact_prefix": len(cached) <= len(reconstructed)
        and cached == reconstructed[: len(cached)],
        "cache_boundary_adjustments": raw_adjusted.get(
            "cache_boundary_adjustments", []
        ),
        "longest_common_prefix_tokens": lcp,
        "first_mismatch_index": mismatch if not exact_prefix else None,
        "first_mismatch_index_in_generated_stream": (
            mismatch - prompt_count
            if mismatch is not None and mismatch >= prompt_count and not exact_prefix
            else None
        ),
        "prefix_relation": "exact_prefix" if exact_prefix else "divergent",
        "old_server_http_status": old_status,
        "old_server_error": old_error,
        "suffix_if_reused": suffix,
        "boundary_tokens": token_context(session.tokenizer, cached, prompt_count, radius=5),
        "mismatch_cached_tokens": (
            token_context(session.tokenizer, cached, mismatch, radius=5)
            if mismatch is not None and mismatch < len(cached)
            else []
        ),
        "mismatch_canonical_tokens": (
            token_context(session.tokenizer, reconstructed, mismatch, radius=5)
            if mismatch is not None and mismatch < len(reconstructed)
            else []
        ),
    }


def run_probe(model: Any, tokenizer: Any, prompt: str, max_tokens: int, followup: str) -> dict[str, Any]:
    session = AgentSession(model, tokenizer, session_id=f"stage8-real-{max_tokens}-{time.time_ns()}")
    first_messages = [{"role": "user", "content": prompt}]
    started = time.perf_counter()
    first = session.complete(
        first_messages,
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        kv_bits=None,
        kv_group_size=64,
        quantized_kv_start=0,
        prefill_step_size=2048,
    )
    evidence = jsonable_turn(session, first, followup)
    incoming = [*session.messages, {"role": "user", "content": followup}]
    continuation_error = None
    continuation_status = 200
    continuation = None
    try:
        continuation = session.complete(
            incoming,
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            kv_bits=None,
            kv_group_size=64,
            quantized_kv_start=0,
            prefill_step_size=2048,
        )
    except BaseException as exc:
        continuation_status = 409 if isinstance(exc, (RuntimeError, KeyError)) else 500
        continuation_error = f"{type(exc).__name__}: {exc}"
    return {
        "probe_id": f"real-g128-max-{max_tokens}",
        "prompt": prompt,
        "followup": followup,
        "max_tokens": max_tokens,
        "elapsed_seconds": time.perf_counter() - started,
        "pre_fix_projection": evidence,
        "continuation_http_status_projection": continuation_status,
        "continuation_error": continuation_error,
        "continuation_metrics": continuation,
        "session_stats_after_probe": session.stats(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--followup", default="Now summarize the parser invariant in one sentence.")
    parser.add_argument("--max-tokens", type=int, nargs="+", default=[256])
    args = parser.parse_args()

    model, tokenizer = load(args.model)
    probes: list[dict[str, Any]] = []
    for budget in args.max_tokens:
        probes.append(run_probe(model, tokenizer, args.prompt, budget, args.followup))
    result = {
        "experiment_id": "P1-REAL-G128-TRUNCATION-REPRODUCTION",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model_path": str(Path(args.model).resolve()),
        "tokenizer_class": type(tokenizer).__name__,
        "probes": probes,
        "summary": {
            "probe_count": len(probes),
            "length_probes": sum(p["pre_fix_projection"]["finish_reason"] == "length" for p in probes),
            "projected_409s": sum(p["continuation_http_status_projection"] == 409 for p in probes),
            "actual_continuation_409s": sum(p["continuation_http_status_projection"] == 409 for p in probes),
            "decode_reencode_nonidentity": sum(
                not p["pre_fix_projection"]["decode_reencode_identity"] for p in probes
            ),
            "max_logical_context_tokens": max(
                (p["session_stats_after_probe"].get("logical_context_tokens", 0) for p in probes),
                default=0,
            ),
        },
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
