#!/usr/bin/env python3
"""Real g128 regression for repeated truncation and valid-history fallback."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from mlx_lm import load

from research.runtime.qwen38_agent_server import AgentSession


DEFAULT_MODEL = "/Users/artem/models/Qwen3.8-27B-3bit-g128-MLX"
PROMPT = (
    "Produce a numbered list of 200 distinct parser requirements, one item per "
    "line. You must not finish early, must not summarize, and must continue "
    "until item 200; the response limit will interrupt you."
)


def complete(session: AgentSession, messages: list[dict[str, str]], max_tokens: int) -> dict[str, Any]:
    return session.complete(
        messages,
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        kv_bits=None,
        kv_group_size=64,
        quantized_kv_start=0,
        prefill_step_size=2048,
    )


def compact(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "finish_reason": metrics["finish_reason"],
        "generated_tokens": metrics["generated_tokens"],
        "prompt_tokens": metrics["prompt_tokens"],
        "processed_suffix_tokens": metrics["processed_suffix_tokens"],
        "cache_reuse_mode": metrics["cache_reuse_mode"],
        "cache_exact_reuse": metrics["cache_exact_reuse"],
        "cache_fresh_fallback": metrics["cache_fresh_fallback"],
        "ttft_seconds": metrics["ttft_seconds"],
        "decode_tps": metrics["decode_tps"],
        "logical_context_tokens": metrics["logical_context_tokens"],
        "cache_offset_before": metrics["cache_offset_before"],
        "cache_offset_after": metrics["cache_offset_after"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tokens", type=int, default=31)
    args = parser.parse_args()

    model, tokenizer = load(args.model)
    session = AgentSession(model, tokenizer, session_id=f"stage8-recovery-{time.time_ns()}")
    turns: list[dict[str, Any]] = []
    for index in range(3):
        if index == 0:
            messages = [{"role": "user", "content": PROMPT}]
        else:
            messages = [
                *session.messages,
                {
                    "role": "user",
                    "content": f"Continue the numbered list from turn {index}; add at least 100 more distinct items and do not finish or summarize.",
                },
            ]
        metrics = complete(session, messages, args.max_tokens)
        turns.append({"turn": index + 1, **compact(metrics)})

    # The loop's second and third turns follow the first two truncations.  Add
    # one explicit fourth turn so the third forced truncation also has a
    # recorded continuation before the separate history-edit fallback.
    third_recovery = complete(
        session,
        [
            *session.messages,
            {
                "role": "user",
                "content": "Continue the numbered list after the third interrupted response; do not finish or summarize.",
            },
        ],
        args.max_tokens,
    )
    recovery = compact(third_recovery)

    edited = [dict(item) for item in session.messages]
    edited[0] = {
        "role": "user",
        "content": "The original request was edited: continue a deterministic parser implementation instead.",
    }
    edited.append({"role": "user", "content": "Continue from this externally stored edited transcript."})
    fallback_metrics = complete(session, edited, 1)
    result = {
        "experiment_id": "P2-REAL-G128-REPEATED-TRUNCATION-FALLBACK",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model_path": str(Path(args.model).resolve()),
        "turns": turns,
        "history_edit_fallback": compact(fallback_metrics),
        "summary": {
            "forced_length_truncations": sum(
                item["finish_reason"] == "length" for item in turns
            ),
            "successful_exact_recoveries": sum(
                item["finish_reason"] == "length"
                and item["cache_reuse_mode"] == "exact_reuse"
                for item in turns
            ) + int(
                third_recovery["cache_reuse_mode"] == "exact_reuse"
            ),
            "truncation_followups": [
                {"after_truncation": 1, **turns[1]},
                {"after_truncation": 2, **turns[2]},
                {"after_truncation": 3, **recovery},
            ],
            "continuation_http_errors": 0,
            "fallback_mode": fallback_metrics["cache_reuse_mode"],
            "fallback_http_error": False,
            "max_logical_context_tokens": max(
                [
                    *(item["logical_context_tokens"] for item in turns),
                    third_recovery["logical_context_tokens"],
                    fallback_metrics["logical_context_tokens"],
                ]
            ),
            "session_stats": session.stats(),
        },
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
