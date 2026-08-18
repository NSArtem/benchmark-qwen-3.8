#!/usr/bin/env python3
"""Validate the user-facing full-transcript Qwen3.8 chat-safe path.

The model is loaded once, but the runtime creates a fresh hybrid cache for
each logical turn.  This script saves prompt/output token IDs and compares the
safe output against saved Mode-B exact-token and Mode-C full-transcript fresh
oracle hashes from the protocol harness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.night2.harness.protocol_cache_harness import (
    DEFAULT_FOLLOWUPS,
    PINNED_MLX_LM_REVISION,
    append_jsonl,
    chat_tokens,
    classify_output,
    current_commit,
    load_runtime,
    memory_snapshot,
    model_metadata,
    os_snapshot,
    read_json,
    text_hash,
    write_json,
)


DEFAULT_MODEL = "/Users/artem/.cache/huggingface/hub/models--lukaskremla--Qwen3.8-27B-3bit-MLX-TextOnly/snapshots/c98bba5926f51fec1c8d8737e577221673f524d7"


def exact_hash(value: list[int]) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def build_messages(
    queries: list[str], answers: list[str], current_query: str
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for query, answer in zip(queries, answers):
        messages.append({"role": "user", "content": query})
        messages.append({"role": "assistant", "content": answer})
    messages.append({"role": "user", "content": current_query})
    return messages


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument(
        "--prompt-file", default="research/context-ladder/prompts/prompt-4096.txt"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--oracle-aggregate", default=None)
    parser.add_argument("--initial-max-tokens", type=int, default=128)
    parser.add_argument("--follow-up-max-tokens", type=int, default=64)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--stop-peak-gib", type=float, default=16.0)
    parser.add_argument("--follow-up", action="append", default=[])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty raw artifact directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = Path(args.prompt_file).resolve()
    initial_prompt = prompt_path.read_text(encoding="utf-8")
    followups = args.follow_up if args.follow_up else DEFAULT_FOLLOWUPS
    if len(followups) < 2:
        raise SystemExit("at least two follow-ups are required")
    queries = [initial_prompt, *followups]
    prompt_sha256 = hashlib.sha256(prompt_path.read_bytes()).hexdigest()

    oracle_relations: dict[int, dict[str, Any]] = {}
    if args.oracle_aggregate:
        aggregate = read_json(Path(args.oracle_aggregate))
        oracle_relations = {
            int(item["turn"]): item for item in aggregate.get("relations", [])
        }

    result: dict[str, Any] = {
        "experiment_id": args.experiment_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": current_commit(),
        "protocol_mode": "chat_safe_full_transcript_reprefill",
        "prompt_path": str(prompt_path),
        "prompt_sha256": prompt_sha256,
        "follow_up_prompts": followups,
        "prefill_step_size": args.prefill_step_size,
        "initial_max_tokens": args.initial_max_tokens,
        "follow_up_max_tokens": args.follow_up_max_tokens,
        "model": model_metadata(args.model_path),
        "turns": [],
        "comparisons": [],
        "exception": None,
        "safety_stop_reason": None,
    }

    started = time.perf_counter()
    answers: list[str] = []
    answer_ids: list[list[int]] = []
    try:
        mx, load, _generate_step, _make_prompt_cache, _make_sampler, _stream = load_runtime()
        from research.runtime.qwen38_runtime import NoWiredTranscriptChat

        model, tokenizer = load(args.model_path)
        result["memory_after_model_load"] = memory_snapshot(
            mx, "after_model_load", started
        )
        session = NoWiredTranscriptChat(model, tokenizer, enable_thinking=False)

        for index, query in enumerate(queries, start=1):
            prompt_messages = build_messages(queries[: index - 1], answers, query)
            prompt_ids = chat_tokens(tokenizer, prompt_messages)
            prompt_path_raw = output_dir / f"turn-{index}.prompt.tokens.json"
            prompt_path_raw.write_text(
                json.dumps(prompt_ids, indent=2) + "\n", encoding="utf-8"
            )
            max_tokens = (
                args.initial_max_tokens
                if index == 1
                else args.follow_up_max_tokens
            )
            turn = session.turn(
                query,
                max_tokens=max_tokens,
                temperature=0.0,
                top_p=1.0,
                prefill_step_size=args.prefill_step_size,
            )
            after = memory_snapshot(mx, f"after_turn_{index}", started)
            generated_ids = [int(value) for value in turn.token_ids]
            output_path = output_dir / f"turn-{index}.txt"
            token_path = output_dir / f"turn-{index}.tokens.json"
            output_path.write_text(turn.text, encoding="utf-8")
            token_path.write_text(
                json.dumps(generated_ids, indent=2) + "\n", encoding="utf-8"
            )
            classification, checks = classify_output(
                turn.text, generated_ids, index, answer_ids
            )
            relation = oracle_relations.get(index, {})
            comparison = {
                "turn": index,
                "safe_generated_token_hash": exact_hash(generated_ids),
                "exact_token_oracle_hash": relation.get("exact_token_oracle_hash"),
                "full_transcript_oracle_hash": relation.get(
                    "full_transcript_oracle_hash"
                ),
                "safe_vs_exact_token_equal": (
                    relation.get("exact_token_oracle_hash") == exact_hash(generated_ids)
                    if relation.get("exact_token_oracle_hash") is not None
                    else None
                ),
                "safe_vs_full_transcript_equal": (
                    relation.get("full_transcript_oracle_hash") == exact_hash(generated_ids)
                    if relation.get("full_transcript_oracle_hash") is not None
                    else None
                ),
                "oracle_relation_before_safe": relation.get("classification"),
            }
            result["comparisons"].append(comparison)
            result["turns"].append(
                {
                    "turn": index,
                    "query_text": query,
                    "query_sha256": text_hash(query),
                    "prompt_token_ids": prompt_ids,
                    "prompt_sha256": exact_hash(prompt_ids),
                    "prompt_tokens": len(prompt_ids),
                    "generated_token_ids": generated_ids,
                    "generated_token_hash": exact_hash(generated_ids),
                    "generated_tokens": len(generated_ids),
                    "raw_text": turn.text,
                    "decoded_text": turn.text,
                    "classification": classification,
                    "replay_detected": checks["overlap"]["replay_detected"],
                    "previous_turn_prefix_overlap": checks["overlap"][
                        "max_prefix_overlap_fraction"
                    ],
                    "overlap_metrics": checks["overlap"],
                    "semantic_checks": checks["semantic"],
                    "ttft_seconds": turn.ttft_seconds,
                    "prompt_tps": turn.prompt_tps,
                    "decode_tps": turn.generation_tps,
                    "generation_seconds": turn.generation_seconds,
                    "total_seconds": turn.total_seconds,
                    "reported_peak_memory_gb_decimal": turn.peak_memory_gb,
                    "memory_after_generation": after,
                    "os_memory_after_generation": os_snapshot(),
                    "raw_output_path": str(output_path),
                    "raw_token_ids_path": str(token_path),
                    "raw_prompt_token_ids_path": str(prompt_path_raw),
                }
            )
            answers.append(turn.text)
            answer_ids.append(generated_ids)
            if after.get("peak_gib") is not None and after["peak_gib"] >= args.stop_peak_gib:
                result["safety_stop_reason"] = (
                    f"peak {after['peak_gib']:.3f} GiB reached configured unattended stop "
                    f"{args.stop_peak_gib:.3f} GiB"
                )
                break
        result["os_memory_final"] = os_snapshot()
        comparisons_with_oracles = [
            item
            for item in result["comparisons"]
            if item["safe_vs_full_transcript_equal"] is not None
        ]
        result["overall_full_transcript_match"] = bool(
            comparisons_with_oracles
            and all(item["safe_vs_full_transcript_equal"] for item in comparisons_with_oracles)
        )
        result["overall_classification"] = (
            "PASS"
            if result["overall_full_transcript_match"]
            and not result["safety_stop_reason"]
            and all(item["classification"] == "COHERENT" for item in result["turns"])
            else "UNRESOLVED"
        )
    except BaseException as exc:
        import traceback

        result["exception"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        result["total_elapsed_seconds"] = time.perf_counter() - started
        result_path = output_dir / "safe.json"
        write_json(result_path, result)
        entries = []
        for turn in result.get("turns", []):
            comparison = next(
                item for item in result["comparisons"] if item["turn"] == turn["turn"]
            )
            entries.append(
                {
                    "timestamp": result["timestamp"],
                    "experiment_id": f"{args.experiment_id}-T{turn['turn']}",
                    "git_commit": result["git_commit"],
                    "python_version": result["model"].get("python_version"),
                    "mlx_version": result["model"].get("mlx_version"),
                    "mlx_lm_revision": PINNED_MLX_LM_REVISION,
                    "model_path": args.model_path,
                    "quantization": result["model"].get("quantization"),
                    "protocol_mode": "chat_safe_full_transcript_reprefill",
                    "prompt_path": str(prompt_path),
                    "prompt_sha256": prompt_sha256,
                    "prompt_tokens": turn.get("prompt_tokens"),
                    "incremental_prompt_tokens": None,
                    "generated_tokens": turn.get("generated_tokens"),
                    "generated_token_hash": turn.get("generated_token_hash"),
                    "previous_turn_prefix_overlap": turn.get(
                        "previous_turn_prefix_overlap"
                    ),
                    "replay_detected": turn.get("replay_detected"),
                    "classification": turn.get("classification"),
                    "exact_token_oracle_hash": comparison.get("exact_token_oracle_hash"),
                    "full_transcript_oracle_hash": comparison.get(
                        "full_transcript_oracle_hash"
                    ),
                    "oracle_relation": (
                        "PASS_FULL_TRANSCRIPT"
                        if comparison.get("safe_vs_full_transcript_equal")
                        else "DIFFERS_FROM_FULL_TRANSCRIPT"
                    ),
                    "ttft_seconds": turn.get("ttft_seconds"),
                    "prompt_tps": turn.get("prompt_tps"),
                    "decode_tps": turn.get("decode_tps"),
                    "total_seconds": turn.get("total_seconds"),
                    "active_gib": turn.get("memory_after_generation", {}).get("active_gib"),
                    "cache_gib": turn.get("memory_after_generation", {}).get("cache_gib"),
                    "active_plus_cache_gib": turn.get("memory_after_generation", {}).get(
                        "active_plus_cache_gib"
                    ),
                    "peak_gib": turn.get("memory_after_generation", {}).get("peak_gib"),
                    "os_memory_metrics": turn.get("os_memory_after_generation"),
                    "cache_layer_summary": None,
                    "exception": result.get("exception"),
                    "safety_stop_reason": result.get("safety_stop_reason"),
                    "raw_artifact": str(result_path),
                }
            )
        if entries:
            append_jsonl(REPO_ROOT / "research/night2/results.jsonl", entries)
        print(
            json.dumps(
                {
                    "safe_result": str(result_path),
                    "overall_classification": result.get("overall_classification"),
                    "overall_full_transcript_match": result.get(
                        "overall_full_transcript_match"
                    ),
                    "exception": result.get("exception"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return 1 if result.get("exception") else 0


if __name__ == "__main__":
    raise SystemExit(main())

