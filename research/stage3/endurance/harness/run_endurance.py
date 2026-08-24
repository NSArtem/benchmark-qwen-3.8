#!/usr/bin/env python3
"""Bounded 8K persistent-chat endurance run for Qwen3.8 g128.

This uses the validated Phase 3 exact-canonical-suffix protocol in one
persistent process. It intentionally avoids a Cartesian product of decoded
assistant-token variants: the EOS-aware variant is tried first, followed by
the two uniform decode policies, and every candidate is checked against the
measured cached token prefix before it is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
SCALING_HARNESS_DIR = REPO_ROOT / "research/stage3/persistent-scaling/harness"
if str(SCALING_HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(SCALING_HARNESS_DIR))

import run_scale as scale_utils


DEFAULT_MODEL = "/Users/artem/models/Qwen3.8-27B-3bit-g128-MLX"
DEFAULT_PROMPT = REPO_ROOT / "research/context-ladder/prompts/prompt-8192.txt"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "research/stage3/endurance-final"
INITIAL_MAX_TOKENS = 128
FOLLOW_UP_MAX_TOKENS = 80
DEFAULT_TURNS = 25
PREFILL_STEP_SIZE = 2048
STOP_PEAK_GIB = 15.4


def follow_up_prompts() -> list[dict[str, Any]]:
    """Twenty-four linked follow-ups, producing 25 assistant turns total."""

    prompts = [
        (
            "Remember the first invariant you identified from the initial Lantern repository context. "
            "Restate it in one sentence, name its database enforcement boundary, and end with marker ENDURANCE-T02."
        ),
        (
            "Revisit your previous answer and revise it if needed: explain why the idempotency-key race "
            "in the initial API transaction matters more or less than the outbox atomicity race. Mention turn 2 and end with ENDURANCE-T03."
        ),
        (
            "From the original repository map, name the exact component where the terminal-job regression can "
            "be introduced. Give a two-sentence failure path and end with ENDURANCE-T04."
        ),
        (
            "Compare your current component recommendation with the invariant you named on turn 2. Keep the "
            "stronger constraint, reject the weaker one, and state one focused test. End with ENDURANCE-T05."
        ),
        (
            "Look back at the initial repository's worker lease design. Explain how a stale lease holder could "
            "publish completion and state the smallest guard that prevents it. End with ENDURANCE-T06."
        ),
        (
            "Use the initial job state machine and turn 2 invariant together: describe the transaction ordering "
            "for a successful completion, including sequence allocation and the outbox row. End with ENDURANCE-T07."
        ),
        (
            "Identify the exact interleaving between a timed-out POST retry and the worker finishing that causes "
            "the incident. Contrast the API transaction with the worker transaction. End with ENDURANCE-T08."
        ),
        (
            "Audit the SSE replay/live handoff from the initial context. Give one way a browser can see a gap or "
            "regression and one minimal reducer or cursor rule to prevent it. End with ENDURANCE-T09."
        ),
        (
            "Write the setup and assertion for one race test covering concurrent identical POST requests. Keep it "
            "focused on the existing PostgreSQL idempotency table and end with ENDURANCE-T10."
        ),
        (
            "Write the setup and assertion for one race test covering a lost worker lease attempting completion. "
            "Reference the lease id and job row condition, then end with ENDURANCE-T11."
        ),
        (
            "Challenge your earlier recommendation: does moving Redis enqueue inside the database transaction "
            "actually solve the durable outbox problem? State the correct minimal interpretation and end with ENDURANCE-T12."
        ),
        (
            "Recall the tenant-boundary requirement from the initial prompt. Name the query or reducer boundary "
            "that must carry tenant identity and one test that catches a cross-tenant read. End with ENDURANCE-T13."
        ),
        (
            "Compare your current recommendation with the answer from turn 2 and the race test from turn 10. "
            "List one thing that must remain in the minimal fix and end with ENDURANCE-T14."
        ),
        (
            "Give a concise SQL-oriented correction for monotonic event sequence allocation while a job row is "
            "locked. Include the projection update and outbox insert boundary. End with ENDURANCE-T15."
        ),
        (
            "Explain how the browser should treat a delayed terminal event after a newer terminal event has been "
            "applied. Use sequence comparison and do not invent a new messaging platform. End with ENDURANCE-T16."
        ),
        (
            "From the initial files, identify the smallest worker method change needed so progress and completion "
            "events cannot bypass the job projection transaction. End with ENDURANCE-T17."
        ),
        (
            "Review the outbox publisher behavior described in the repository. State whether duplicate delivery is "
            "handled at publication, consumption, or both, and give one idempotent rule. End with ENDURANCE-T18."
        ),
        (
            "State a production rollout order for the invariant, idempotency, lease, and SSE fixes you have named. "
            "Keep it to four steps and end with ENDURANCE-T19."
        ),
        (
            "Design one failure-injection test that crashes between durable commit and queue publication. Explain "
            "what must be observable after restart and end with ENDURANCE-T20."
        ),
        (
            "Audit the accumulated conversation for an unresolved behavior that belongs in a separate follow-up, "
            "not the minimal fix. Cite the initial task's scope and end with ENDURANCE-T21."
        ),
        (
            "Prioritize the three most valuable assertions for a regression suite covering the incident. Compare "
            "them with the test you proposed on turn 10 and end with ENDURANCE-T22."
        ),
        (
            "Give one concise final implementation recommendation that preserves the first invariant, idempotency, "
            "lease ownership, and ordered SSE replay. End with ENDURANCE-T23."
        ),
        (
            "Perform a stale-context audit: answer only with the component and invariant that still matter most from "
            "the initial prompt, and explain why the current turn is not a replay. End with ENDURANCE-T24."
        ),
        (
            "Summarize the final patch boundary in five short bullets, explicitly distinguishing the minimal fix from "
            "follow-up work. Preserve the initial repository facts and end with ENDURANCE-T25."
        ),
    ]
    context_guard = (
        "Keep the answer grounded in the initial Lantern repository excerpts, "
        "preserve the existing file boundaries, and do not introduce a new "
        "messaging platform or database."
    )
    return [
        {
            "turn": index + 2,
            "content": f"{content} {context_guard}",
            "anchor": f"ENDURANCE-T{index + 2:02d}",
        }
        for index, content in enumerate(prompts)
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def endurance_checks(
    text: str,
    ids: list[int],
    previous_answers: list[list[int]],
    query: str,
    anchor: str,
) -> dict[str, Any]:
    previous_metrics: list[dict[str, Any]] = []
    for index, previous in enumerate(previous_answers, start=1):
        common = scale_utils.lcp(ids, previous)
        previous_metrics.append(
            {
                "previous_turn": index,
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
    pathological_tail = bool(
        len(ids) >= 24 and len(set(ids[-min(32, len(ids)) :])) <= 2
    )
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
        "query_sha256": text_hash(query),
        "anchor": anchor,
        "anchor_present": anchor.lower() in text.lower(),
        "pathological_low_entropy_tail": pathological_tail,
        "text_nonempty": bool(text.strip()),
    }


def canonical_for_turn(
    tokenizer: Any,
    initial_text: str,
    current_query: str,
    prior_queries: list[dict[str, Any]],
    previous_turns: list[dict[str, Any]],
    answer_ids: list[list[int]],
    expected_cache_ids: list[int],
    eos_ids: set[int],
) -> tuple[dict[str, Any], str, list[str]]:
    """Build and validate the full canonical prompt for the next turn."""

    policies = ("eos_aware", "all_special_preserving", "all_skip_special")
    for policy in policies:
        assistant_texts: list[str] = []
        for generated in answer_ids:
            if policy == "eos_aware":
                skip_special = not generated or int(generated[-1]) not in eos_ids
            elif policy == "all_special_preserving":
                skip_special = False
            else:
                skip_special = True
            assistant_texts.append(
                tokenizer.decode(generated, skip_special_tokens=skip_special)
            )
        messages: list[dict[str, str]] = [
            {"role": "user", "content": initial_text}
        ]
        for index, assistant_text in enumerate(assistant_texts):
            messages.append({"role": "assistant", "content": assistant_text})
            if index < len(prior_queries):
                messages.append(
                    {"role": "user", "content": prior_queries[index]["content"]}
                )
        messages.append({"role": "user", "content": current_query})
        canonical = scale_utils.canonical_prompt(tokenizer, messages)
        canonical = scale_utils.normalize_canonical_for_cache_boundary(
            canonical, previous_turns, answer_ids, eos_ids
        )
        if canonical["token_ids"][: len(expected_cache_ids)] == expected_cache_ids:
            return canonical, policy, assistant_texts
    raise RuntimeError(
        "no endurance transcript candidate began with the measured cache stream"
    )


def append_jsonl(entries: list[dict[str, Any]]) -> None:
    path = REPO_ROOT / "research/stage3/results.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


def run(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    output_root.joinpath("outputs").mkdir(exist_ok=True)
    output_root.joinpath("tokens").mkdir(exist_ok=True)
    output_root.joinpath("state_signatures").mkdir(exist_ok=True)
    output_root.joinpath("logs").mkdir(exist_ok=True)
    started = time.perf_counter()
    prompt_path = Path(args.prompt).resolve()
    initial_text = prompt_path.read_text(encoding="utf-8")
    all_followups = follow_up_prompts()[: max(args.turns - 1, 0)]
    queries: list[dict[str, Any]] = [
        {"turn": 1, "content": initial_text, "anchor": None},
        *all_followups,
    ]
    result: dict[str, Any] = {
        "experiment_id": "P4-G128-PERSISTENT-ENDURANCE-8K",
        "phase": 4,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": scale_utils.oracle_utils.current_commit(),
        "model_path": args.model,
        "prompt_path": str(prompt_path),
        "prompt_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
        "requested_turns": args.turns,
        "completed_turns": 0,
        "initial_max_tokens": args.initial_max_tokens,
        "follow_up_max_tokens": args.follow_up_max_tokens,
        "prefill_step_size": PREFILL_STEP_SIZE,
        "stop_peak_gib": args.stop_peak_gib,
        "protocol_mode": "persistent_exact_canonical_suffix_eos_aware",
        "model": None,
        "turns": [],
        "periodic_os_snapshots": [],
        "exception": None,
        "safety_stop_reason": None,
        "overall_classification": None,
    }
    expected_cache_ids: list[int] = []
    answer_ids: list[list[int]] = []
    answer_texts: list[str] = []
    append_entries: list[dict[str, Any]] = []
    try:
        mx, load, generate_step, make_prompt_cache, make_sampler, _ = (
            scale_utils.oracle_utils.load_runtime()
        )
        result["model"] = scale_utils.oracle_utils.model_metadata(args.model)
        result["os_memory_before"] = scale_utils.oracle_utils.os_snapshot()
        model, tokenizer = load(args.model)
        eos_ids = {int(value) for value in (tokenizer.eos_token_ids or [])}
        result["memory_after_model_load"] = scale_utils.oracle_utils.memory_snapshot(
            mx, "phase4_after_model_load", started
        )
        cache = make_prompt_cache(model)
        for turn_index, query_record in enumerate(queries, start=1):
            current_query = query_record["content"]
            max_tokens = (
                args.initial_max_tokens
                if turn_index == 1
                else args.follow_up_max_tokens
            )
            candidate_policy = "initial"
            candidate_assistant_texts: list[str] = []
            if turn_index == 1:
                canonical = scale_utils.canonical_prompt(
                    tokenizer, [{"role": "user", "content": initial_text}]
                )
            else:
                canonical, candidate_policy, candidate_assistant_texts = canonical_for_turn(
                    tokenizer,
                    initial_text,
                    current_query,
                    all_followups[: turn_index - 2],
                    result["turns"],
                    answer_ids,
                    expected_cache_ids,
                    eos_ids,
                )
            suffix_ids = (
                canonical["token_ids"]
                if turn_index == 1
                else canonical["token_ids"][len(expected_cache_ids) :]
            )
            stage = scale_utils.oracle_utils.generate_stage(
                mx,
                generate_step,
                make_sampler,
                tokenizer,
                model,
                suffix_ids,
                cache,
                max_tokens,
                f"phase4:turn_{turn_index}",
                started,
            )
            actual_before = scale_utils.cache_offset(stage["cache_diagnostics"][0])
            actual_after = scale_utils.cache_offset(stage["cache_diagnostics"][-1])
            generated_ids = [int(value) for value in stage["generated_token_ids"]]
            generated_text = stage["decoded_output"]
            expected_after_ids = canonical["token_ids"] + generated_ids
            prefix_exact = (
                turn_index == 1
                or canonical["token_ids"][: int(actual_before or 0)]
                == expected_cache_ids
            )
            logical_equation = (
                actual_before == len(expected_cache_ids)
                and actual_before + len(suffix_ids) == len(canonical["token_ids"])
                and prefix_exact
            )
            after_length_matches = actual_after == len(expected_after_ids)
            checks = endurance_checks(
                generated_text,
                generated_ids,
                answer_ids,
                current_query,
                query_record.get("anchor") or "INITIAL",
            )
            state_path = output_root / "state_signatures" / f"turn-{turn_index}.json"
            write_json(
                state_path,
                {
                    "turn": turn_index,
                    "cache_diagnostics": stage["cache_diagnostics"],
                },
            )
            prompt_token_path = output_root / "tokens" / f"turn-{turn_index}.canonical-prompt.json"
            suffix_token_path = output_root / "tokens" / f"turn-{turn_index}.exact-suffix.json"
            generated_token_path = output_root / "tokens" / f"turn-{turn_index}.generated.json"
            write_json(
                prompt_token_path,
                {
                    "token_ids": canonical["token_ids"],
                    "token_count": len(canonical["token_ids"]),
                    "sha256": scale_utils.exact_hash(canonical["token_ids"]),
                },
            )
            write_json(
                suffix_token_path,
                {
                    "token_ids": suffix_ids,
                    "token_count": len(suffix_ids),
                    "sha256": scale_utils.exact_hash(suffix_ids),
                },
            )
            write_json(
                generated_token_path,
                {
                    "token_ids": generated_ids,
                    "token_count": len(generated_ids),
                    "sha256": scale_utils.exact_hash(generated_ids),
                },
            )
            output_path = output_root / "outputs" / f"turn-{turn_index}.txt"
            output_path.write_text(generated_text, encoding="utf-8")
            memory = stage["memory_after_generation"]
            turn_record = {
                "turn": turn_index,
                "user_query": current_query,
                "user_query_sha256": text_hash(current_query),
                "anchor": query_record.get("anchor"),
                "prompt_token_count": len(canonical["token_ids"]),
                "prompt_token_sha256": scale_utils.exact_hash(canonical["token_ids"]),
                "processed_suffix_tokens": len(suffix_ids),
                "suffix_token_sha256": scale_utils.exact_hash(suffix_ids),
                "generated_token_ids": generated_ids,
                "generated_token_count": len(generated_ids),
                "generated_token_hash": scale_utils.exact_hash(generated_ids),
                "assistant_text": generated_text,
                "assistant_text_sha256": text_hash(generated_text),
                "cache_offset_before": actual_before,
                "cache_offset_after": actual_after,
                "expected_cache_prefix_tokens": len(expected_cache_ids),
                "expected_cache_after_tokens": len(expected_after_ids),
                "canonical_prefix_exact_at_boundary": prefix_exact,
                "logical_equation_holds": logical_equation,
                "cache_after_length_matches": after_length_matches,
                "candidate_policy": candidate_policy,
                "canonical_boundary_adjustments": canonical.get(
                    "cache_boundary_adjustments", []
                ),
                "state_signature_after": stage["cache_diagnostics"][-1].get(
                    "state_signature_sha256"
                ),
                "state_signature_path": str(state_path),
                "classification": checks["classification"],
                "replay_detected": checks["replay_detected"],
                "output_checks": checks,
                "ttft_seconds": stage["ttft_seconds"],
                "prompt_tps": stage["prompt_tps"],
                "decode_tps": stage["decode_tps"],
                "generation_seconds": stage["generation_seconds"],
                "total_seconds": stage["total_seconds"],
                "memory_after_generation": memory,
                "raw_output_path": str(output_path),
                "raw_canonical_prompt_path": str(prompt_token_path),
                "raw_suffix_path": str(suffix_token_path),
                "raw_generated_token_path": str(generated_token_path),
            }
            result["turns"].append(turn_record)
            result["completed_turns"] = len(result["turns"])
            answer_ids.append(generated_ids)
            answer_texts.append(generated_text)
            expected_cache_ids = expected_after_ids
            if turn_index == 1 or turn_index % 5 == 0:
                result["periodic_os_snapshots"].append(
                    {
                        "turn": turn_index,
                        "snapshot": scale_utils.oracle_utils.os_snapshot(),
                    }
                )
            if not logical_equation or not after_length_matches:
                result["safety_stop_reason"] = (
                    f"protocol accounting failed at endurance turn {turn_index}"
                )
                break
            if checks["classification"] == "BROKEN":
                result["safety_stop_reason"] = (
                    f"first BROKEN endurance output at turn {turn_index}"
                )
                break
            peak_gib = memory.get("peak_gib")
            active_plus_cache_gib = memory.get("active_plus_cache_gib")
            if peak_gib is not None and peak_gib >= args.stop_peak_gib:
                result["safety_stop_reason"] = (
                    f"peak {peak_gib:.3f} GiB reached unattended stop "
                    f"{args.stop_peak_gib:.3f} GiB"
                )
                break
            if (
                active_plus_cache_gib is not None
                and active_plus_cache_gib >= args.stop_active_plus_cache_gib
            ):
                result["safety_stop_reason"] = (
                    f"active plus cache {active_plus_cache_gib:.3f} GiB reached "
                    f"unattended stop {args.stop_active_plus_cache_gib:.3f} GiB"
                )
                break
        result["os_memory_after"] = scale_utils.oracle_utils.os_snapshot()
        result["final_logical_tokens"] = (
            result["turns"][-1]["cache_offset_after"] if result["turns"] else None
        )
        result["overall_classification"] = (
            "PASS"
            if result["completed_turns"] == len(queries)
            and not result["safety_stop_reason"]
            and all(
                turn["classification"] == "COHERENT"
                and turn["logical_equation_holds"]
                and turn["cache_after_length_matches"]
                and not turn["replay_detected"]
                for turn in result["turns"]
            )
            else "STOPPED_OR_UNRESOLVED"
        )
    except BaseException as exc:
        result["exception"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        if result["overall_classification"] is None:
            result["overall_classification"] = "ERROR"
    finally:
        result["total_elapsed_seconds"] = time.perf_counter() - started
        write_json(output_root / "endurance.json", result)
        for turn in result["turns"]:
            memory = turn.get("memory_after_generation", {})
            append_entries.append(
                {
                    "timestamp": result["timestamp"],
                    "experiment_id": f"{result['experiment_id']}-T{turn['turn']}",
                    "phase": 4,
                    "git_commit": result["git_commit"],
                    "python_version": (result.get("model") or {}).get("python_version"),
                    "mlx_version": (result.get("model") or {}).get("mlx_version"),
                    "mlx_lm_revision": scale_utils.PINNED_MLX_LM_REVISION,
                    "model_name": "Qwen3.8-27B-3bit-g128",
                    "model_path": args.model,
                    "quant_bits": 3,
                    "quant_group_size": 128,
                    "execution_mode": "persistent_exact_canonical_suffix_endurance",
                    "persistent_cache": True,
                    "kv_bits": None,
                    "turn": turn["turn"],
                    "total_logical_tokens": turn["prompt_token_count"],
                    "processed_suffix_tokens": turn["processed_suffix_tokens"],
                    "generated_tokens": turn["generated_token_count"],
                    "generated_token_hash": turn["generated_token_hash"],
                    "classification": turn["classification"],
                    "replay_detected": turn["replay_detected"],
                    "ttft_seconds": turn["ttft_seconds"],
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
                        "canonical_prefix_exact_at_boundary": turn[
                            "canonical_prefix_exact_at_boundary"
                        ],
                        "logical_equation_holds": turn["logical_equation_holds"],
                        "cache_after_length_matches": turn["cache_after_length_matches"],
                    },
                    "raw_artifact": str(output_root / "endurance.json"),
                }
            )
        if append_entries:
            append_jsonl(append_entries)
        summary = {
            "experiment_id": result["experiment_id"],
            "overall_classification": result["overall_classification"],
            "completed_turns": result["completed_turns"],
            "final_logical_tokens": result.get("final_logical_tokens"),
            "safety_stop_reason": result.get("safety_stop_reason"),
            "exception": result.get("exception"),
            "total_elapsed_seconds": result["total_elapsed_seconds"],
        }
        (output_root / "logs" / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, sort_keys=True), flush=True)
    return 1 if result.get("exception") else 0


def write_results_markdown(output_root: Path, result: dict[str, Any]) -> None:
    turns = result.get("turns", [])
    lines = [
        "# Phase 4 persistent-chat endurance",
        "",
        f"Overall outcome: **{result.get('overall_classification')}**",
        "",
        "| turn | logical tokens | suffix | TTFT (s) | decode (tok/s) | active+cache (GiB) | peak (GiB) | classification |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for turn in turns:
        memory = turn.get("memory_after_generation", {})
        lines.append(
            f"| {turn['turn']} | {turn['prompt_token_count']} | "
            f"{turn['processed_suffix_tokens']} | {turn['ttft_seconds']:.3f} | "
            f"{turn['decode_tps']:.3f} | "
            f"{memory.get('active_plus_cache_gib', float('nan')):.3f} | "
            f"{memory.get('peak_gib', float('nan')):.3f} | "
            f"{turn['classification']} |"
        )
    output_root.joinpath("results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=str(DEFAULT_PROMPT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--turns", type=int, default=DEFAULT_TURNS)
    parser.add_argument("--initial-max-tokens", type=int, default=INITIAL_MAX_TOKENS)
    parser.add_argument("--follow-up-max-tokens", type=int, default=FOLLOW_UP_MAX_TOKENS)
    parser.add_argument("--stop-peak-gib", type=float, default=STOP_PEAK_GIB)
    parser.add_argument("--stop-active-plus-cache-gib", type=float, default=15.0)
    args = parser.parse_args()
    code = run(args)
    result = json.loads((Path(args.output_root) / "endurance.json").read_text())
    write_results_markdown(Path(args.output_root), result)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
