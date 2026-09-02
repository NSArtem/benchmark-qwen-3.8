#!/usr/bin/env python3
"""Token-level reproduction of the pre-Stage-8 truncated-session failure.

This harness loads only the local tokenizer.  It deliberately does not load
Qwen3.8 weights.  A generated token stream is treated as authoritative cache
state, then the old server representation (decoded assistant text rendered
back through the chat template) is compared with that cached stream.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from transformers import AutoTokenizer

from research.runtime.qwen38_chat import (
    HistoryTurn,
    canonical_prompt,
    exact_suffix,
    normalize_cache_boundary,
    normalize_token_ids,
    reconcile_canonical_history,
)
from research.runtime.qwen38_agent_server import appended_messages


DEFAULT_MODEL = "/Users/artem/models/Qwen3.8-27B-3bit-g128-MLX"
BOUNDARY_BUDGETS = [1, 2, 3, 31, 32, 63, 64, 127, 128, 255, 256]


def eos_ids(tokenizer: Any) -> set[int]:
    value = getattr(tokenizer, "eos_token_ids", None)
    if value is None:
        value = getattr(tokenizer, "eos_token_id", None)
    if value is None:
        return set()
    if isinstance(value, int):
        return {int(value)}
    return {int(item) for item in value}


def first_value_mismatch(left: Sequence[int], right: Sequence[int]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if int(a) != int(b):
            return index
    return None


def token_context(tokenizer: Any, ids: Sequence[int], index: int, radius: int = 3) -> list[dict[str, Any]]:
    start = max(0, index - radius)
    end = min(len(ids), index + radius + 1)
    values: list[dict[str, Any]] = []
    for position in range(start, end):
        token_id = int(ids[position])
        values.append(
            {
                "index": position,
                "id": token_id,
                "token": str(tokenizer.convert_ids_to_tokens(token_id)),
                "decoded_single": tokenizer.decode([token_id], skip_special_tokens=True),
            }
        )
    return values


def canonical_tokens(tokenizer: Any, messages: list[dict[str, str]]) -> dict[str, Any]:
    return canonical_prompt(tokenizer, messages, enable_thinking=False)


def run_case(
    tokenizer: Any,
    *,
    case_id: str,
    prompt_text: str,
    generated_ids: Sequence[int],
    termination: str,
    followup: str,
) -> dict[str, Any]:
    first_messages = [{"role": "user", "content": prompt_text}]
    first_prompt = canonical_tokens(tokenizer, first_messages)
    generated = [int(value) for value in generated_ids]
    decoded = tokenizer.decode(generated, skip_special_tokens=True)
    reencoded = normalize_token_ids(
        tokenizer.encode(decoded, add_special_tokens=False)
    )
    cached = list(first_prompt["token_ids"]) + generated
    history = [
        HistoryTurn(
            user_text=prompt_text,
            prompt_token_count=len(first_prompt["token_ids"]),
            generated_ids=generated,
            assistant_text=decoded,
        )
    ]
    next_messages = [
        *first_messages,
        {"role": "assistant", "content": decoded},
        {"role": "user", "content": followup},
    ]
    next_prompt_raw = canonical_tokens(tokenizer, next_messages)
    raw_reconstructed = list(next_prompt_raw["token_ids"])
    next_prompt = reconcile_canonical_history(
        next_prompt_raw,
        history,
        eos_ids(tokenizer),
    )
    reconstructed = list(next_prompt["token_ids"])
    raw_mismatch = first_value_mismatch(cached, raw_reconstructed)
    if raw_mismatch is None and len(cached) != len(raw_reconstructed):
        raw_mismatch = min(len(cached), len(raw_reconstructed))
    mismatch = first_value_mismatch(cached, reconstructed)
    if mismatch is None and len(cached) != len(reconstructed):
        mismatch = min(len(cached), len(reconstructed))
    lcp = 0
    while lcp < min(len(cached), len(reconstructed)) and cached[lcp] == reconstructed[lcp]:
        lcp += 1
    raw_exact_prefix = len(cached) <= len(raw_reconstructed) and cached == raw_reconstructed[: len(cached)]
    repaired_exact_prefix = len(cached) <= len(reconstructed) and cached == reconstructed[: len(cached)]
    prefix_relation = "exact_prefix" if raw_exact_prefix else "divergent"
    try:
        suffix = exact_suffix(raw_reconstructed, cached)
        old_error = None
        old_status = 200
    except RuntimeError as exc:
        suffix = None
        old_error = str(exc)
        old_status = 409
    boundary_index = len(first_prompt["token_ids"])
    mismatch_index_in_generation = (
        mismatch - boundary_index if mismatch is not None and mismatch >= boundary_index else None
    )
    return {
        "case_id": case_id,
        "termination": termination,
        "prompt_text": prompt_text,
        "prompt_sha256": hashlib.sha256(prompt_text.encode()).hexdigest(),
        "followup": followup,
        "generated_token_ids": generated,
        "decoded_text": decoded,
        "reencoded_token_ids": reencoded,
        "decode_reencode_identity": generated == reencoded,
        "first_prompt_token_ids": list(first_prompt["token_ids"]),
        "first_prompt_token_count": len(first_prompt["token_ids"]),
        "cached_token_ids": cached,
        "canonical_next_token_ids": reconstructed,
        "canonical_next_token_count": len(reconstructed),
        "raw_canonical_next_token_ids": raw_reconstructed,
        "raw_canonical_next_token_count": len(raw_reconstructed),
        "cache_represented_prefix_length": len(cached),
        "cache_offset_before_next_request": len(cached),
        "longest_common_prefix_tokens": lcp,
        "raw_longest_common_prefix_tokens": min(
            next(
                (index for index, (left, right) in enumerate(zip(cached, raw_reconstructed)) if left != right),
                min(len(cached), len(raw_reconstructed)),
            ),
            min(len(cached), len(raw_reconstructed)),
        ),
        "raw_first_mismatch_index": raw_mismatch,
        "raw_first_mismatch_index_in_generated_stream": (
            raw_mismatch - boundary_index
            if raw_mismatch is not None and raw_mismatch >= boundary_index
            else None
        ),
        "first_mismatch_index": raw_mismatch if not raw_exact_prefix else None,
        "first_mismatch_index_in_generated_stream": (
            raw_mismatch - boundary_index
            if raw_mismatch is not None and not raw_exact_prefix
            else None
        ),
        "boundary_index": boundary_index,
        "boundary_tokens": token_context(tokenizer, cached, boundary_index, radius=4),
        "mismatch_cached_tokens": (
            token_context(tokenizer, cached, raw_mismatch, radius=4)
            if raw_mismatch is not None and raw_mismatch < len(cached)
            else []
        ),
        "mismatch_canonical_tokens": (
            token_context(tokenizer, raw_reconstructed, raw_mismatch, radius=4)
            if raw_mismatch is not None and raw_mismatch < len(raw_reconstructed)
            else []
        ),
        "cache_boundary_adjustments": next_prompt.get("cache_boundary_adjustments", []),
        "prefix_relation": prefix_relation,
        "old_server_http_status": old_status,
        "old_server_error": old_error,
        "old_server_suffix_if_reused": suffix,
        "message_history_prefix_check": appended_messages(
            first_messages + [{"role": "assistant", "content": decoded}],
            next_messages,
        )
        == next_messages,
        "chat_boundary_involved": bool(
            next_prompt.get("cache_boundary_adjustments")
            or (
                raw_mismatch is not None
                and raw_mismatch >= boundary_index
                and raw_mismatch < len(raw_reconstructed)
                and raw_reconstructed[raw_mismatch] in eos_ids(tokenizer)
            )
        ),
        "post_fix_exact_prefix": repaired_exact_prefix,
    }


def stress_stream(tokenizer: Any) -> list[int]:
    """A deterministic stream with ordinary, code, whitespace, and Unicode."""

    text = (
        "normal prose; punctuation, indentation, and code: \n"
        "```python\n"
        "def merge(items):\n"
        "    return [x for x in items]\n"
        "```\n"
        "Unicode boundaries: café — 東京 👩‍💻 𐍈.\n"
        "partial-looking text: `unfinished"
    )
    return normalize_token_ids(tokenizer.encode(text, add_special_tokens=False))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    generated_stress = stress_stream(tokenizer)
    cases: list[dict[str, Any]] = []
    cases.append(
        run_case(
            tokenizer,
            case_id="explicit-partial-emoji",
            prompt_text="Emit one Unicode marker and stop at the configured length boundary.",
            generated_ids=[9008, 239, 102, 373],
            termination="length",
            followup="Now continue with one concise sentence.",
        )
    )
    for budget in BOUNDARY_BUDGETS:
        ids = (generated_stress * ((budget // max(len(generated_stress), 1)) + 1))[:budget]
        cases.append(
            run_case(
                tokenizer,
                case_id=f"stress-budget-{budget}",
                prompt_text="Produce a mixed prose/code/Unicode response.",
                generated_ids=ids,
                termination="length",
                followup="Continue from the exact previous response and report the next step.",
            )
        )
    # EOS is represented separately: this controls the known boundary-EOS
    # normalization path and proves it is not the same as the partial-byte case.
    eos_id = sorted(eos_ids(tokenizer) or {248046})[0]
    eos_prefix = normalize_token_ids(tokenizer.encode("complete", add_special_tokens=False))
    cases.append(
        run_case(
            tokenizer,
            case_id="explicit-eos",
            prompt_text="Return a complete answer.",
            generated_ids=eos_prefix + [eos_id],
            termination="eos",
            followup="Continue with a new user request.",
        )
    )
    output = {
        "experiment_id": "P1-TRUNCATION-TOKEN-ROOT-CAUSE-ORACLE",
        "model_path": str(Path(args.model).resolve()),
        "tokenizer_class": type(tokenizer).__name__,
        "eos_token_ids": sorted(eos_ids(tokenizer)),
        "budgets": BOUNDARY_BUDGETS,
        "cases": cases,
        "summary": {
            "case_count": len(cases),
            "old_409_cases": sum(case["old_server_http_status"] == 409 for case in cases),
            "decode_reencode_nonidentity_cases": sum(
                not case["decode_reencode_identity"] for case in cases
            ),
            "explicit_partial_case_first_mismatch": cases[0]["first_mismatch_index"],
            "explicit_partial_case_generation_mismatch": cases[0][
                "first_mismatch_index_in_generated_stream"
            ],
            "eos_case_status": cases[-1]["old_server_http_status"],
        },
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
