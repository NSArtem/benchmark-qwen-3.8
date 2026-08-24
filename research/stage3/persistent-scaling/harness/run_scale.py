#!/usr/bin/env python3
"""Bounded exact-suffix persistent-chat scaling for Qwen3.8 g128.

This harness is deliberately separate from the trusted full-transcript chat
helper. It keeps one prompt cache for a logical conversation, renders the full
canonical transcript at every turn, and supplies only the suffix not already
represented by the measured cache offset. It records enough token and cache
metadata to reject a malformed suffix instead of silently continuing.

The Phase 1 oracle helper is imported only for its direct-generate_step,
memory, and compact cache-diagnostic primitives. It never calls a wired-limit
helper.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
ORACLE_HARNESS_DIR = (
    REPO_ROOT / "research/stage3/cache-boundary-oracle/harness"
)
if str(ORACLE_HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(ORACLE_HARNESS_DIR))

import run_oracle as oracle_utils


PINNED_MLX_LM_REVISION = "254d153fdeb6f150edd4fc5a54f9828638481fa8"
DEFAULT_MODEL = "/Users/artem/models/Qwen3.8-27B-3bit-g128-MLX"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "research/stage3/persistent-scaling"
INITIAL_MAX_TOKENS = 128
FOLLOW_UP_MAX_TOKENS = 64
PREFILL_STEP_SIZE = 2048
STOP_PEAK_GIB = 15.6

CONTEXTS: dict[str, dict[str, Any]] = {
    "8k": {
        "prompt_file": REPO_ROOT / "research/context-ladder/prompts/prompt-8192.txt",
        "target_prompt_tokens": 8192,
        "oracle_turn": 2,
    },
    "12k": {
        "prompt_file": REPO_ROOT / "research/context-ladder/prompts/prompt-12288.txt",
        "target_prompt_tokens": 12288,
        "oracle_turn": None,
    },
    "16k": {
        "prompt_file": REPO_ROOT / "qwen38-mtp-smoke/coding-prompt.txt",
        "target_prompt_tokens": 16767,
        "oracle_turn": 5,
    },
}

FOLLOW_UPS = [
    "Answer ONLY which single database invariant should be enforced first. Name the invariant and the enforcement boundary.",
    "Turn that invariant into one focused regression test. State the setup, exact assertion, and failure it catches.",
    "Using the initial repository prompt, identify the component where this invariant can be violated. Explain the failure path in two sentences.",
    "Give one concise implementation recommendation that preserves the first invariant and the idempotency requirement. Do not introduce a second database.",
]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def exact_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_ids(value: Any) -> list[int]:
    if hasattr(value, "encodings") and value.encodings:
        return [int(item) for item in value.encodings[0].ids]
    if isinstance(value, dict):
        value = value["input_ids"]
    elif hasattr(value, "input_ids"):
        value = value.input_ids
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], (list, tuple)):
        value = value[0]
    return [int(item) for item in value]


def canonical_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> dict[str, Any]:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    from_text = normalize_ids(tokenizer.encode(rendered, add_special_tokens=False))
    direct = normalize_ids(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )
    if from_text != direct:
        raise RuntimeError("chat-template string/tokenized forms disagree")
    return {
        "rendered_text": rendered,
        "token_ids": from_text,
        "token_count": len(from_text),
        "token_sha256": exact_hash(from_text),
    }


def normalize_canonical_for_cache_boundary(
    canonical: dict[str, Any],
    previous_turns: list[dict[str, Any]],
    previous_answer_ids: list[list[int]],
    eos_ids: set[int],
) -> dict[str, Any]:
    """Remove template-added EOS tokens already present in raw cache output.

    `generate_step` yields the EOS/control token and the prompt cache retains
    it. Qwen's chat template also emits `<|im_end|>` after an assistant
    message. For an assistant turn that already ended in EOS, retaining both
    would make the next exact suffix contain a duplicate boundary token. The
    adjustment is made only when the raw generated segment is present at the
    measured turn boundary and the following candidate token is the same EOS.
    """

    token_ids = list(canonical["token_ids"])
    adjustments: list[dict[str, Any]] = []
    for turn, generated in zip(previous_turns, previous_answer_ids):
        if not generated or int(generated[-1]) not in eos_ids:
            continue
        start = int(turn["prompt_token_count"])
        end = start + len(generated)
        if (
            end < len(token_ids)
            and token_ids[start:end] == generated
            and token_ids[end] == int(generated[-1])
        ):
            del token_ids[end]
            adjustments.append(
                {
                    "turn": int(turn["turn"]),
                    "removed_token_id": int(generated[-1]),
                    "removed_index": end,
                    "reason": "template_added_EOS_already_in_cached_generation",
                }
            )
    adjusted = dict(canonical)
    adjusted["token_ids"] = token_ids
    adjusted["token_count"] = len(token_ids)
    adjusted["token_sha256"] = exact_hash(token_ids)
    adjusted["cache_boundary_adjustments"] = adjustments
    return adjusted


def lcp(left: list[int], right: list[int]) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def cache_offset(diag: dict[str, Any] | None) -> int | None:
    if not diag:
        return None
    value = diag.get("logical_prefix_token_count")
    return int(value) if value is not None else None


def stage_json(stage: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in stage.items()
        if key != "first_logprob_values"
    }


def output_checks(
    text: str,
    ids: list[int],
    previous_answers: list[list[int]],
    turn: int,
) -> dict[str, Any]:
    previous_metrics: list[dict[str, Any]] = []
    for previous_index, previous in enumerate(previous_answers, start=1):
        common = lcp(ids, previous)
        previous_metrics.append(
            {
                "previous_turn": previous_index,
                "previous_tokens": len(previous),
                "longest_common_prefix_tokens": common,
                "prefix_overlap_fraction": common / len(ids) if ids else 0.0,
                "exact_prefix_replay": bool(ids and ids == previous[: len(ids)]),
            }
        )
    max_overlap = max(
        (item["prefix_overlap_fraction"] for item in previous_metrics),
        default=0.0,
    )
    lower = text.lower()
    terms_by_turn = {
        2: ["invariant", "enforce"],
        3: ["regression", "assert"],
        4: ["component"],
        5: ["invariant", "recommend"],
    }
    terms = terms_by_turn.get(turn, [])
    pathological_tail = False
    if len(ids) >= 24:
        pathological_tail = len(set(ids[-min(32, len(ids)) :])) <= 2
    replay = bool(ids and max_overlap >= 0.90)
    classification = (
        "BROKEN"
        if not text.strip() or len(ids) < 8 or replay or pathological_tail
        else "COHERENT"
    )
    return {
        "classification": classification,
        "replay_detected": replay,
        "max_prefix_overlap_fraction": max_overlap,
        "previous_turns": previous_metrics,
        "term_presence": {term: term in lower for term in terms},
        "supporting_terms_all_present": all(term in lower for term in terms),
        "pathological_low_entropy_tail": pathological_tail,
        "text_nonempty": bool(text.strip()),
    }


def build_message_candidates(
    initial_text: str,
    queries: list[str],
    assistant_candidates: list[list[str]],
) -> list[dict[str, Any]]:
    """Try both special-token decode variants for each prior answer."""

    candidates: list[dict[str, Any]] = []
    combinations = itertools.product(*assistant_candidates) if assistant_candidates else [()]
    for assistant_texts in combinations:
        messages: list[dict[str, str]] = [
            {"role": "user", "content": initial_text}
        ]
        for assistant_index, answer in enumerate(assistant_texts):
            messages.append({"role": "assistant", "content": answer})
            if assistant_index < len(queries):
                messages.append(
                    {"role": "user", "content": queries[assistant_index]}
                )
        candidates.append(
            {
                "messages": messages,
                "assistant_texts": list(assistant_texts),
            }
        )
    return candidates


def append_stage3_jsonl(entries: list[dict[str, Any]]) -> None:
    path = REPO_ROOT / "research/stage3/results.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


def context_output_dir(output_root: Path, context: str) -> Path:
    return output_root / context


def persistent_run(args: argparse.Namespace) -> int:
    context = args.context
    spec = CONTEXTS[context]
    output_dir = context_output_dir(Path(args.output_root), context)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    prompt_path = Path(spec["prompt_file"]).resolve()
    initial_text = prompt_path.read_text(encoding="utf-8")
    queries = [initial_text, *FOLLOW_UPS]
    result: dict[str, Any] = {
        "experiment_id": f"{getattr(args, 'experiment_prefix', 'P3-G128-PERSISTENT-SUFFIX')}-{context.upper()}",
        "phase": getattr(args, "phase", 3),
        "context_label": context,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": oracle_utils.current_commit(),
        "model_path": args.model,
        "prompt_path": str(prompt_path),
        "prompt_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
        "target_prompt_tokens": spec["target_prompt_tokens"],
        "follow_up_prompts": FOLLOW_UPS,
        "initial_max_tokens": args.initial_max_tokens,
        "follow_up_max_tokens": args.follow_up_max_tokens,
        "prefill_step_size": PREFILL_STEP_SIZE,
        "stop_peak_gib": args.stop_peak_gib,
        "kv_bits": args.kv_bits,
        "kv_group_size": args.kv_group_size,
        "quantized_kv_start": args.quantized_kv_start,
        "protocol_mode": "persistent_exact_canonical_suffix",
        "model": None,
        "turns": [],
        "exception": None,
        "safety_stop_reason": None,
        "overall_classification": None,
    }
    answers: list[str] = []
    answer_ids: list[list[int]] = []
    assistant_candidates: list[list[str]] = []
    append_entries: list[dict[str, Any]] = []
    try:
        mx, load, generate_step, make_prompt_cache, make_sampler, _ = oracle_utils.load_runtime()
        result["model"] = oracle_utils.model_metadata(args.model)
        result["os_memory_before"] = oracle_utils.os_snapshot()
        model, tokenizer = load(args.model)
        eos_ids = {int(value) for value in (tokenizer.eos_token_ids or [])}
        result["memory_after_model_load"] = oracle_utils.memory_snapshot(
            mx, "after_model_load", started
        )
        cache = make_prompt_cache(model)
        expected_cache_ids: list[int] = []
        for turn_index, query in enumerate(queries, start=1):
            max_tokens = (
                args.initial_max_tokens
                if turn_index == 1
                else args.follow_up_max_tokens
            )
            if turn_index == 1:
                canonical = canonical_prompt(
                    tokenizer,
                    [{"role": "user", "content": initial_text}],
                )
                selected_assistant_texts: list[str] = []
                selected_candidate_index = 0
            else:
                candidate_records = build_message_candidates(
                    initial_text,
                    queries[1 : turn_index - 1],
                    assistant_candidates[: turn_index - 1],
                )
                selected = None
                measured_before_for_selection = cache_offset(
                    {"logical_prefix_token_count": len(expected_cache_ids)}
                )
                # The real offset is read from the new stage's before snapshot
                # below. The logical expected count is the conservative first
                # selector; candidates are checked again after measurement.
                for candidate_index, candidate in enumerate(candidate_records):
                    candidate_canonical = canonical_prompt(tokenizer, candidate["messages"] + [{"role": "user", "content": query}])
                    candidate_canonical = normalize_canonical_for_cache_boundary(
                        candidate_canonical,
                        result["turns"],
                        answer_ids,
                        eos_ids,
                    )
                    if candidate_canonical["token_ids"][: len(expected_cache_ids)] == expected_cache_ids:
                        selected = (candidate_index, candidate, candidate_canonical)
                        break
                if selected is None:
                    raise RuntimeError(
                        f"{context} turn {turn_index}: no canonical transcript candidate begins with logical cached stream"
                    )
                selected_candidate_index, selected_record, canonical = selected
                selected_assistant_texts = selected_record["assistant_texts"]

            measured_before_expected = len(expected_cache_ids)
            if turn_index == 1:
                suffix_ids = canonical["token_ids"]
            else:
                # The cache diagnostic is available only after generation_stage
                # starts. The expected logical stream is exact by construction;
                # use its length here and verify the actual offset immediately
                # after the stage's before snapshot is captured.
                suffix_ids = canonical["token_ids"][measured_before_expected:]
            before_offset = cache_offset(
                {
                    "logical_prefix_token_count": (
                        0 if turn_index == 1 else measured_before_expected
                    )
                }
            )
            stage = oracle_utils.generate_stage(
                mx,
                generate_step,
                make_sampler,
                tokenizer,
                model,
                suffix_ids,
                cache,
                max_tokens,
                f"{context}:turn_{turn_index}",
                started,
                kv_bits=args.kv_bits,
                kv_group_size=args.kv_group_size,
                quantized_kv_start=args.quantized_kv_start,
            )
            actual_before = cache_offset(stage["cache_diagnostics"][0])
            actual_after = cache_offset(stage["cache_diagnostics"][-1])
            if turn_index == 1:
                measured_before_expected = 0
                before_offset = 0
            if actual_before != measured_before_expected:
                raise RuntimeError(
                    f"{context} turn {turn_index}: actual cache offset {actual_before} "
                    f"does not equal expected {measured_before_expected}"
                )
            generated_ids = [int(value) for value in stage["generated_token_ids"]]
            generated_text = stage["decoded_output"]
            if turn_index == 1:
                prefix_exact = True
                logical_equation = len(suffix_ids) == len(canonical["token_ids"])
            else:
                prefix_exact = canonical["token_ids"][:actual_before] == expected_cache_ids
                logical_equation = (
                    prefix_exact and actual_before + len(suffix_ids) == len(canonical["token_ids"])
                )
            expected_after_ids = canonical["token_ids"] + generated_ids
            after_length_matches = actual_after == len(expected_after_ids)
            checks = output_checks(generated_text, generated_ids, answer_ids, turn_index)
            state_path = output_dir / "state_signatures" / f"turn-{turn_index}.json"
            write_json(
                state_path,
                {
                    "context": context,
                    "turn": turn_index,
                    "cache_diagnostics": stage["cache_diagnostics"],
                },
            )
            prompt_token_path = output_dir / "tokens" / f"turn-{turn_index}.canonical-prompt.json"
            suffix_token_path = output_dir / "tokens" / f"turn-{turn_index}.exact-suffix.json"
            generated_token_path = output_dir / "tokens" / f"turn-{turn_index}.generated.json"
            write_json(
                prompt_token_path,
                {
                    "token_ids": canonical["token_ids"],
                    "token_count": len(canonical["token_ids"]),
                    "sha256": exact_hash(canonical["token_ids"]),
                    "rendered_text_sha256": text_hash(canonical["rendered_text"]),
                },
            )
            write_json(
                suffix_token_path,
                {
                    "token_ids": suffix_ids,
                    "token_count": len(suffix_ids),
                    "sha256": exact_hash(suffix_ids),
                    "first_32_tokens": suffix_ids[:32],
                    "last_32_tokens": suffix_ids[-32:],
                },
            )
            write_json(
                generated_token_path,
                {
                    "token_ids": generated_ids,
                    "token_count": len(generated_ids),
                    "sha256": exact_hash(generated_ids),
                },
            )
            output_path = output_dir / "outputs" / f"turn-{turn_index}.txt"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(generated_text, encoding="utf-8")
            turn_record = {
                "turn": turn_index,
                "user_query": query,
                "user_query_sha256": text_hash(query),
                "prompt_token_count": len(canonical["token_ids"]),
                "prompt_token_sha256": exact_hash(canonical["token_ids"]),
                "canonical_boundary_adjustments": canonical.get(
                    "cache_boundary_adjustments", []
                ),
                "processed_suffix_tokens": len(suffix_ids),
                "suffix_token_sha256": exact_hash(suffix_ids),
                "generated_token_ids": generated_ids,
                "generated_token_count": len(generated_ids),
                "generated_token_hash": exact_hash(generated_ids),
                "assistant_text": generated_text,
                "assistant_text_sha256": text_hash(generated_text),
                "cache_offset_before": actual_before,
                "cache_offset_after": actual_after,
                "expected_cache_prefix_tokens": measured_before_expected,
                "expected_cache_after_tokens": len(expected_after_ids),
                "canonical_prefix_exact_at_boundary": prefix_exact,
                "logical_equation_holds": logical_equation,
                "cache_after_length_matches": after_length_matches,
                "canonical_candidate_index": selected_candidate_index,
                "canonical_candidate_count": 1 if turn_index == 1 else len(candidate_records),
                "state_signature_after": stage["cache_diagnostics"][-1].get("state_signature_sha256"),
                "state_signature_path": str(state_path),
                "classification": checks["classification"],
                "replay_detected": checks["replay_detected"],
                "output_checks": checks,
                "ttft_seconds": stage["ttft_seconds"],
                "prompt_tps": stage["prompt_tps"],
                "decode_tps": stage["decode_tps"],
                "generation_seconds": stage["generation_seconds"],
                "total_seconds": stage["total_seconds"],
                "memory_after_generation": stage["memory_after_generation"],
                "raw_output_path": str(output_path),
                "raw_canonical_prompt_path": str(prompt_token_path),
                "raw_suffix_path": str(suffix_token_path),
                "raw_generated_token_path": str(generated_token_path),
            }
            result["turns"].append(turn_record)
            answers.append(generated_text)
            answer_ids.append(generated_ids)
            assistant_candidates.append(
                [
                    tokenizer.decode(generated_ids, skip_special_tokens=True),
                    tokenizer.decode(generated_ids, skip_special_tokens=False),
                ]
            )
            expected_cache_ids = expected_after_ids
            if not logical_equation or not after_length_matches:
                result["safety_stop_reason"] = (
                    f"protocol accounting failed at turn {turn_index}; stopped before next turn"
                )
                break
            if checks["classification"] == "BROKEN":
                broken_count = sum(
                    item.get("classification") == "BROKEN" for item in result["turns"]
                )
                if broken_count >= 2:
                    result["safety_stop_reason"] = (
                        f"persistent output BROKEN twice at context {context}"
                    )
                    break
            peak_gib = stage["memory_after_generation"].get("peak_gib")
            if peak_gib is not None and peak_gib >= args.stop_peak_gib:
                result["safety_stop_reason"] = (
                    f"peak {peak_gib:.3f} GiB reached configured unattended stop "
                    f"{args.stop_peak_gib:.3f} GiB"
                )
                break
        result["os_memory_after"] = oracle_utils.os_snapshot()
        result["total_logical_tokens_after_last_turn"] = (
            result["turns"][-1]["cache_offset_after"] if result["turns"] else None
        )
        result["overall_classification"] = (
            "PASS"
            if len(result["turns"]) == len(queries)
            and all(item["classification"] == "COHERENT" for item in result["turns"])
            and all(
                item["logical_equation_holds"] and item["cache_after_length_matches"]
                for item in result["turns"]
            )
            and not result["safety_stop_reason"]
            else "STOPPED_OR_UNRESOLVED"
        )
    except BaseException as exc:
        result["exception"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        message = str(exc).lower()
        if "out of memory" in message or "metal" in message or "gpu" in message:
            result["safety_stop_reason"] = "runtime_exception_requires_manual_safety_review"
        if result["overall_classification"] is None:
            result["overall_classification"] = "ERROR"
    finally:
        result["total_elapsed_seconds"] = time.perf_counter() - started
        write_json(output_dir / "persistent.json", result)
        for turn in result.get("turns", []):
            memory = turn.get("memory_after_generation", {})
            append_entries.append(
                {
                    "timestamp": result["timestamp"],
                    "experiment_id": f"{result['experiment_id']}-T{turn['turn']}",
                    "phase": result["phase"],
                    "git_commit": result["git_commit"],
                    "python_version": (result.get("model") or {}).get("python_version"),
                    "mlx_version": (result.get("model") or {}).get("mlx_version"),
                    "mlx_lm_revision": PINNED_MLX_LM_REVISION,
                    "model_name": "Qwen3.8-27B-3bit-g128",
                    "model_path": args.model,
                    "quant_bits": 3,
                    "quant_group_size": 128,
                    "execution_mode": (
                        "persistent_exact_canonical_suffix_kv8"
                        if args.kv_bits is not None
                        else "persistent_exact_canonical_suffix"
                    ),
                    "persistent_cache": True,
                    "kv_bits": args.kv_bits,
                    "kv_group_size": args.kv_group_size,
                    "quantized_kv_start": args.quantized_kv_start,
                    "prompt_path": str(prompt_path),
                    "prompt_sha256": result["prompt_sha256"],
                    "total_logical_tokens": turn["prompt_token_count"],
                    "processed_suffix_tokens": turn["processed_suffix_tokens"],
                    "generated_tokens": turn["generated_token_count"],
                    "generated_token_hash": turn["generated_token_hash"],
                    "classification": turn["classification"],
                    "replay_detected": turn["replay_detected"],
                    "structured_output_valid": None,
                    "ttft_seconds": turn["ttft_seconds"],
                    "prompt_tps": turn["prompt_tps"],
                    "decode_tps": turn["decode_tps"],
                    "total_seconds": turn["total_seconds"],
                    "active_memory_bytes": memory.get("active_bytes"),
                    "cache_memory_bytes": memory.get("cache_bytes"),
                    "active_plus_cache_bytes": memory.get("active_plus_cache_bytes"),
                    "peak_memory_bytes": memory.get("peak_bytes"),
                    "rss_bytes": memory.get("rss_max_bytes"),
                    "cache_state_signature": turn["state_signature_after"],
                    "exception": result.get("exception"),
                    "safety_stop_reason": result.get("safety_stop_reason"),
                    "protocol_accounting": {
                        "canonical_prefix_exact_at_boundary": turn["canonical_prefix_exact_at_boundary"],
                        "logical_equation_holds": turn["logical_equation_holds"],
                        "cache_after_length_matches": turn["cache_after_length_matches"],
                    },
                    "raw_artifact": str(output_dir / "persistent.json"),
                }
            )
        if append_entries:
            append_stage3_jsonl(append_entries)
        print(
            json.dumps(
                {
                    "context": context,
                    "experiment_id": result["experiment_id"],
                    "overall_classification": result.get("overall_classification"),
                    "turns": len(result.get("turns", [])),
                    "exception": result.get("exception"),
                    "safety_stop_reason": result.get("safety_stop_reason"),
                    "elapsed_seconds": result["total_elapsed_seconds"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return 1 if result.get("exception") else 0


def oracle_run(args: argparse.Namespace) -> int:
    context = args.context
    spec = CONTEXTS[context]
    output_dir = context_output_dir(Path(args.output_root), context)
    persistent_path = output_dir / "persistent.json"
    oracle_label = f"oracle-turn-{args.oracle_turn}"
    if args.oracle_suffix:
        oracle_label += f"-{args.oracle_suffix}"
    result_path = output_dir / f"{oracle_label}.json"
    started = time.perf_counter()
    result: dict[str, Any] = {
        "experiment_id": (
            f"P3-G128-ORACLE-{context.upper()}-T{args.oracle_turn}"
            + (f"-{args.oracle_suffix.upper()}" if args.oracle_suffix else "")
        ),
        "phase": 3,
        "context_label": context,
        "oracle_turn": args.oracle_turn,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": oracle_utils.current_commit(),
        "model_path": args.model,
        "protocol_mode": "fresh_full_transcript_semantic_oracle",
        "exception": None,
        "safety_stop_reason": None,
    }
    try:
        persistent = read_json(persistent_path)
        turns = persistent["turns"]
        turn_index = int(args.oracle_turn)
        if turn_index < 2 or turn_index > len(turns):
            raise RuntimeError(
                f"oracle turn {turn_index} unavailable; persistent run has {len(turns)} turns"
            )
        prompt_path = Path(persistent["prompt_path"])
        initial_text = prompt_path.read_text(encoding="utf-8")
        messages: list[dict[str, str]] = [{"role": "user", "content": initial_text}]
        for prior_index, prior in enumerate(turns[: turn_index - 1]):
            messages.append({"role": "assistant", "content": prior["assistant_text"]})
            if prior_index + 1 < turn_index - 1:
                messages.append(
                    {
                        "role": "user",
                        "content": turns[prior_index + 1]["user_query"],
                    }
                )
        current = turns[turn_index - 1]
        messages.append({"role": "user", "content": current["user_query"]})
        mx, load, generate_step, make_prompt_cache, make_sampler, _ = oracle_utils.load_runtime()
        result["model"] = oracle_utils.model_metadata(args.model)
        result["prompt_path"] = str(prompt_path)
        result["prompt_sha256"] = persistent["prompt_sha256"]
        result["os_memory_before"] = oracle_utils.os_snapshot()
        model, tokenizer = load(args.model)
        result["memory_after_model_load"] = oracle_utils.memory_snapshot(
            mx, "after_model_load", started
        )
        canonical = canonical_prompt(tokenizer, messages)
        result["persistent_prompt_token_sha256"] = current["prompt_token_sha256"]
        result["full_transcript_prompt_token_sha256"] = exact_hash(
            canonical["token_ids"]
        )
        result["persistent_prompt_token_equal_full_transcript"] = (
            result["persistent_prompt_token_sha256"]
            == result["full_transcript_prompt_token_sha256"]
        )
        cache = make_prompt_cache(model)
        stage = oracle_utils.generate_stage(
            mx,
            generate_step,
            make_sampler,
            tokenizer,
            model,
            canonical["token_ids"],
            cache,
            int(current["generated_token_count"]),
            f"{context}:oracle_turn_{turn_index}",
            started,
        )
        previous_ids = [item["generated_token_ids"] for item in turns[: turn_index - 1]]
        checks = output_checks(stage["decoded_output"], stage["generated_token_ids"], previous_ids, turn_index)
        persistent_ids = current["generated_token_ids"]
        result["prompt_tokens"] = len(canonical["token_ids"])
        result["prompt_token_sha256"] = exact_hash(canonical["token_ids"])
        result["generated_token_ids"] = stage["generated_token_ids"]
        result["generated_tokens"] = stage["generated_tokens"]
        result["generated_token_hash"] = exact_hash(stage["generated_token_ids"])
        result["decoded_output"] = stage["decoded_output"]
        result["classification"] = checks["classification"]
        result["replay_detected"] = checks["replay_detected"]
        result["output_checks"] = checks
        result["ttft_seconds"] = stage["ttft_seconds"]
        result["prompt_tps"] = stage["prompt_tps"]
        result["decode_tps"] = stage["decode_tps"]
        result["total_seconds"] = stage["total_seconds"]
        result["memory_after_generation"] = stage["memory_after_generation"]
        result["cache_state_signature"] = stage["cache_diagnostics"][-1].get("state_signature_sha256")
        result["persistent_output_hash"] = exact_hash(persistent_ids)
        result["persistent_vs_oracle_token_equal"] = persistent_ids == stage["generated_token_ids"]
        result["persistent_vs_oracle_token_lcp"] = lcp(persistent_ids, stage["generated_token_ids"])
        result["semantic_oracle_pass"] = (
            current["classification"] == "COHERENT"
            and checks["classification"] == "COHERENT"
            and not current["replay_detected"]
            and not checks["replay_detected"]
        )
        result["os_memory_after"] = oracle_utils.os_snapshot()
        write_json(output_dir / "state_signatures" / f"{oracle_label}.json", {
            "context": context,
            "turn": turn_index,
            "cache_diagnostics": stage["cache_diagnostics"],
        })
        write_json(output_dir / "tokens" / f"{oracle_label}.json", {
            "token_ids": stage["generated_token_ids"],
            "token_count": stage["generated_tokens"],
            "sha256": result["generated_token_hash"],
        })
        (output_dir / "outputs" / f"{oracle_label}.txt").write_text(
            stage["decoded_output"], encoding="utf-8"
        )
    except BaseException as exc:
        result["exception"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        result["overall_classification"] = "ERROR"
    finally:
        result["total_elapsed_seconds"] = time.perf_counter() - started
        write_json(result_path, result)
        if result.get("exception") is None:
            memory = result.get("memory_after_generation", {})
            append_stage3_jsonl(
                [
                    {
                        "timestamp": result["timestamp"],
                        "experiment_id": result["experiment_id"],
                        "phase": 3,
                        "git_commit": result["git_commit"],
                        "python_version": result["model"]["python_version"],
                        "mlx_version": result["model"]["mlx_version"],
                        "mlx_lm_revision": PINNED_MLX_LM_REVISION,
                        "model_name": "Qwen3.8-27B-3bit-g128",
                        "model_path": args.model,
                        "quant_bits": 3,
                        "quant_group_size": 128,
                        "execution_mode": "fresh_full_transcript_semantic_oracle",
                        "persistent_cache": False,
                        "kv_bits": None,
                        "prompt_path": result.get("prompt_path"),
                        "prompt_sha256": result.get("prompt_sha256"),
                        "total_logical_tokens": result.get("prompt_tokens"),
                        "processed_suffix_tokens": None,
                        "generated_tokens": result.get("generated_tokens"),
                        "generated_token_hash": result.get("generated_token_hash"),
                        "classification": result.get("classification"),
                        "replay_detected": result.get("replay_detected"),
                        "structured_output_valid": None,
                        "ttft_seconds": result.get("ttft_seconds"),
                        "prompt_tps": result.get("prompt_tps"),
                        "decode_tps": result.get("decode_tps"),
                        "total_seconds": result.get("total_seconds"),
                        "active_memory_bytes": memory.get("active_bytes"),
                        "cache_memory_bytes": memory.get("cache_bytes"),
                        "active_plus_cache_bytes": memory.get("active_plus_cache_bytes"),
                        "peak_memory_bytes": memory.get("peak_bytes"),
                        "rss_bytes": memory.get("rss_max_bytes"),
                        "cache_state_signature": result.get("cache_state_signature"),
                        "exception": None,
                        "safety_stop_reason": None,
                        "semantic_oracle_pass": result.get("semantic_oracle_pass"),
                        "raw_artifact": str(result_path),
                    }
                ]
            )
        print(
            json.dumps(
                {
                    "context": context,
                    "oracle_turn": args.oracle_turn,
                    "semantic_oracle_pass": result.get("semantic_oracle_pass"),
                    "exception": result.get("exception"),
                    "elapsed_seconds": result["total_elapsed_seconds"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return 1 if result.get("exception") else 0


def combine_context(output_root: Path, context: str) -> dict[str, Any]:
    output_dir = context_output_dir(output_root, context)
    persistent = read_json(output_dir / "persistent.json")
    oracles = []
    for path in sorted(output_dir.glob("oracle-turn-*.json")):
        oracles.append(read_json(path))
    summary = {
        "context": context,
        "target_prompt_tokens": CONTEXTS[context]["target_prompt_tokens"],
        "prompt_path": persistent.get("prompt_path"),
        "overall_classification": persistent.get("overall_classification"),
        "safety_stop_reason": persistent.get("safety_stop_reason"),
        "exception": persistent.get("exception"),
        "turns_completed": len(persistent.get("turns", [])),
        "turns": [
            {
                "turn": item["turn"],
                "logical_prompt_tokens": item["prompt_token_count"],
                "suffix_tokens": item["processed_suffix_tokens"],
                "ttft_seconds": item["ttft_seconds"],
                "decode_tps": item["decode_tps"],
                "generated_tokens": item["generated_token_count"],
                "peak_gib": item["memory_after_generation"].get("peak_gib"),
                "active_plus_cache_gib": item["memory_after_generation"].get("active_plus_cache_gib"),
                "classification": item["classification"],
                "replay_detected": item["replay_detected"],
                "protocol_ok": item["logical_equation_holds"] and item["cache_after_length_matches"],
            }
            for item in persistent.get("turns", [])
        ],
        "oracles": [
            {
                "turn": item.get("oracle_turn"),
                "prompt_tokens": item.get("prompt_tokens"),
                "ttft_seconds": item.get("ttft_seconds"),
                "decode_tps": item.get("decode_tps"),
                "classification": item.get("classification"),
                "semantic_oracle_pass": item.get("semantic_oracle_pass"),
                "persistent_vs_oracle_token_lcp": item.get("persistent_vs_oracle_token_lcp"),
            }
            for item in oracles
        ],
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def run_all(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    order = ["8k", "12k", "16k"]
    context_summaries: list[dict[str, Any]] = []
    overall_returncode = 0
    for context in order:
        output_dir = context_output_dir(output_root, context)
        if output_dir.exists() and any(output_dir.iterdir()):
            raise SystemExit(f"refusing to overwrite non-empty context directory: {output_dir}")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--mode",
            "persistent",
            "--context",
            context,
            "--model",
            args.model,
            "--output-root",
            str(output_root),
            "--initial-max-tokens",
            str(args.initial_max_tokens),
            "--follow-up-max-tokens",
            str(args.follow_up_max_tokens),
            "--stop-peak-gib",
            str(args.stop_peak_gib),
        ]
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "logs").mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(command, capture_output=True, text=True)
        (output_dir / "logs/persistent.stdout.log").write_text(completed.stdout, encoding="utf-8")
        (output_dir / "logs/persistent.stderr.log").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            overall_returncode = completed.returncode
        if not (output_dir / "persistent.json").exists():
            break
        persistent_summary = combine_context(output_root, context)
        oracle_turn = CONTEXTS[context].get("oracle_turn")
        if (
            oracle_turn is not None
            and persistent_summary["turns_completed"] >= oracle_turn
            and persistent_summary["overall_classification"] == "PASS"
        ):
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--mode",
                "oracle",
                "--context",
                context,
                "--oracle-turn",
                str(oracle_turn),
                "--model",
                args.model,
                "--output-root",
                str(output_root),
            ]
            completed = subprocess.run(command, capture_output=True, text=True)
            (output_dir / "logs/oracle.stdout.log").write_text(completed.stdout, encoding="utf-8")
            (output_dir / "logs/oracle.stderr.log").write_text(completed.stderr, encoding="utf-8")
            if completed.returncode != 0:
                overall_returncode = completed.returncode
        context_summaries.append(combine_context(output_root, context))
        if persistent_summary["overall_classification"] != "PASS":
            # Do not scale upward after a protocol, correctness, or safety
            # failure. The independent quality phase remains available.
            break
    summary = {
        "experiment_id": "P3-G128-PERSISTENT-SUFFIX-SCALING",
        "phase": 3,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": oracle_utils.current_commit(),
        "model_path": args.model,
        "contexts": context_summaries,
        "returncode": overall_returncode,
        "outcome": "PASS" if context_summaries and all(item["overall_classification"] == "PASS" for item in context_summaries) else "STOPPED_OR_UNRESOLVED",
    }
    write_json(output_root / "scaling-summary.json", summary)
    lines = [
        "# Phase 3 exact-suffix persistent scaling",
        "",
        f"Overall outcome: **{summary['outcome']}**",
        "",
        "| context | turns | warm TTFT (s) | decode (tok/s) | peak (GiB) | correctness |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for item in context_summaries:
        turns = item.get("turns", [])
        warm = [turn["ttft_seconds"] for turn in turns[1:] if turn.get("ttft_seconds") is not None]
        decodes = [turn["decode_tps"] for turn in turns[1:] if turn.get("decode_tps") is not None]
        peaks = [turn["peak_gib"] for turn in turns if turn.get("peak_gib") is not None]
        lines.append(
            f"| {item['context']} | {item['turns_completed']} | "
            f"{(f'{min(warm):.3f}–{max(warm):.3f}' if warm else 'n/a')} | "
            f"{(f'{sum(decodes)/len(decodes):.3f}' if decodes else 'n/a')} | "
            f"{(f'{max(peaks):.3f}' if peaks else 'n/a')} | {item['overall_classification']} |"
        )
        for oracle in item.get("oracles", []):
            lines.append(
                f"\nFresh oracle at turn {oracle.get('turn')}: "
                f"{oracle.get('semantic_oracle_pass')} (TTFT {oracle.get('ttft_seconds')}, "
                f"decode {oracle.get('decode_tps')})."
            )
    (output_root / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return overall_returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("run", "persistent", "oracle", "combine"), required=True)
    parser.add_argument("--context", choices=sorted(CONTEXTS), default=None)
    parser.add_argument("--oracle-turn", type=int, default=None)
    parser.add_argument("--oracle-suffix", default="")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--initial-max-tokens", type=int, default=INITIAL_MAX_TOKENS)
    parser.add_argument("--follow-up-max-tokens", type=int, default=FOLLOW_UP_MAX_TOKENS)
    parser.add_argument("--stop-peak-gib", type=float, default=STOP_PEAK_GIB)
    parser.add_argument("--phase", type=int, default=3)
    parser.add_argument(
        "--experiment-prefix",
        default="P3-G128-PERSISTENT-SUFFIX",
    )
    parser.add_argument("--kv-bits", type=int, default=None)
    parser.add_argument("--kv-group-size", type=int, default=64)
    parser.add_argument("--quantized-kv-start", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "run":
        raise SystemExit(run_all(args))
    if args.context is None:
        raise SystemExit("--context is required for persistent/oracle/combine modes")
    if args.mode == "persistent":
        raise SystemExit(persistent_run(args))
    if args.mode == "oracle":
        if args.oracle_turn is None:
            raise SystemExit("--oracle-turn is required for oracle mode")
        raise SystemExit(oracle_run(args))
    print(json.dumps(combine_context(Path(args.output_root), args.context), sort_keys=True))


if __name__ == "__main__":
    main()
