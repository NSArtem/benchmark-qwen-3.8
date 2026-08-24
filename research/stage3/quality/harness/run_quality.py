#!/usr/bin/env python3
"""Deterministic diverse g64-vs-g128 quality suite for Qwen3.8.

Each quantization runs in its own fresh process. Cases use fresh prompt caches
within that process, and all inference goes through the pinned direct
`generate_step` path. The suite records raw token IDs/text plus objective
checks; the parent compares the two model result directories without loading a
model.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import subprocess
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


G64_MODEL = "/Users/artem/.cache/huggingface/hub/models--lukaskremla--Qwen3.8-27B-3bit-MLX-TextOnly/snapshots/c98bba5926f51fec1c8d8737e577221673f524d7"
G128_MODEL = "/Users/artem/models/Qwen3.8-27B-3bit-g128-MLX"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "research/stage3/quality-final"
MAX_TOKENS = 64
PREFILL_STEP_SIZE = 2048


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def case(
    case_id: str,
    category: str,
    prompt: str,
    *,
    required_terms: list[str] | None = None,
    forbidden_terms: list[str] | None = None,
    line_prefixes: list[str] | None = None,
    max_chars: int | None = None,
    json_keys: list[str] | None = None,
    json_types: dict[str, type] | None = None,
    expected_json_values: dict[str, Any] | None = None,
    max_tokens: int = MAX_TOKENS,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "category": category,
        "prompt": prompt,
        "required_terms": required_terms or [],
        "forbidden_terms": forbidden_terms or [],
        "line_prefixes": line_prefixes,
        "max_chars": max_chars,
        "json_keys": json_keys,
        "json_types": {
            key: value.__name__ for key, value in (json_types or {}).items()
        },
        "expected_json_values": expected_json_values or {},
        "max_tokens": max_tokens,
    }


def lantern_code_cases() -> list[dict[str, Any]]:
    return [
        case(
            "coding-01",
            "coding",
            """Review this TypeScript transaction sketch:
            `SELECT idempotency_keys WHERE tenant_id=$1 AND key=$2; INSERT jobs; INSERT idempotency_keys; COMMIT`.
            Two identical requests can run concurrently. Diagnose the race and give the smallest database-level fix.
            Keep the answer under 100 words.""",
            required_terms=["unique", "idempotency", "transaction"],
        ),
        case(
            "coding-02",
            "coding",
            "A reducer applies events whenever `event.sequence > job.version`, but it accepts a delayed running event after a succeeded event. State the invariant and the exact guard needed for terminal jobs. Keep it concise.",
            required_terms=["terminal", "sequence", "ignore"],
        ),
        case(
            "coding-03",
            "coding",
            "An API updates a job row and commits, then appends its outbox row in a second transaction. Explain the atomicity failure and name the transaction boundary that must contain both operations.",
            required_terms=["atomic", "same", "transaction"],
        ),
        case(
            "coding-04",
            "coding",
            "A worker checks its Redis lease before rendering but not immediately before publishing success. Give the minimal completion predicate using job state and lease identity, and say what happens when it fails.",
            required_terms=["lease", "running", "success"],
        ),
        case(
            "coding-05",
            "coding",
            "An SSE route replays PostgreSQL rows and then subscribes to Redis. A disconnect can occur between those steps. Describe a cursor-based fix using Last-Event-ID and monotonic sequence ordering.",
            required_terms=["last-event-id", "sequence", "replay"],
        ),
        case(
            "coding-06",
            "coding",
            "A repository method selects a job by `id` only even though every route has a tenantId. Identify the security bug and show the required query predicate in words or SQL.",
            required_terms=["tenant", "id", "predicate"],
        ),
        case(
            "coding-07",
            "coding",
            "The jobs table stores `version` and `next_event_sequence`; two workers can update the same job. Explain why a row lock is needed and how it preserves event ordering.",
            required_terms=["lock", "sequence", "order"],
        ),
        case(
            "coding-08",
            "coding",
            "A retry with the same idempotency key but a different request body finds an existing record. State the correct response and why returning the original body is not enough.",
            required_terms=["conflict", "hash", "idempotency"],
        ),
        case(
            "coding-09",
            "coding",
            "Plan one focused integration test for the timeout-after-commit race: two identical POSTs, one delayed worker, and an outbox assertion. List the setup and two assertions.",
            required_terms=["two", "job", "outbox"],
        ),
        case(
            "coding-10",
            "coding",
            "Give a minimal production patch order for this incident: idempotency uniqueness, atomic projection plus outbox, lease ownership, and browser event ordering. Use four numbered items.",
            required_terms=["idempotency", "outbox", "lease", "browser"],
        ),
    ]


def instruction_cases() -> list[dict[str, Any]]:
    return [
        case(
            "instruction-01",
            "instruction",
            "Output exactly three lines and nothing else. Use the prefixes `A:`, `B:`, and `C:`. A must be `cache`, B must be `suffix`, and C must be `oracle`.",
            line_prefixes=["A:", "B:", "C:"],
            max_chars=80,
        ),
        case(
            "instruction-02",
            "instruction",
            "Return exactly the ordered words `correctness`, `stability`, `context`, `latency` separated by ` > `. Do not add punctuation or explanation.",
            required_terms=["correctness > stability > context > latency"],
            max_chars=90,
        ),
        case(
            "instruction-03",
            "instruction",
            "Answer in one sentence of at most 20 words. Mention both `g128` and `normal KV`; do not mention any alternative quantization.",
            required_terms=["g128", "normal kv"],
            forbidden_terms=["2-bit", "dwq", "draft"],
            max_chars=180,
        ),
        case(
            "instruction-04",
            "instruction",
            "Output exactly two bullet lines, each beginning with `- `. The first must contain `PASS`; the second must contain `STOP`. No heading.",
            required_terms=["pass", "stop"],
            line_prefixes=["- ", "- "],
            max_chars=100,
        ),
        case(
            "instruction-05",
            "instruction",
            "Reply with the single uppercase word `COHERENT`. No whitespace-separated extra words, punctuation, or explanation.",
            required_terms=["coherent"],
            max_chars=30,
        ),
        case(
            "instruction-06",
            "instruction",
            "Give a two-item numbered list. Item 1 must say `measure`; item 2 must say `preserve`. Do not use a code fence.",
            required_terms=["measure", "preserve"],
            max_chars=120,
        ),
    ]


def structured_cases() -> list[dict[str, Any]]:
    return [
        case(
            "structured-01",
            "structured",
            'Return JSON only with exactly these keys: `"state"` (string) and `"safe"` (boolean). Set state to "queued" and safe to true.',
            json_keys=["state", "safe"],
            json_types={"state": str, "safe": bool},
            expected_json_values={"state": "queued", "safe": True},
        ),
        case(
            "structured-02",
            "structured",
            'Return JSON only with keys `"sequence"` (integer), `"terminal"` (boolean), and `"event"` (string). Use 4, false, and "job.started".',
            json_keys=["sequence", "terminal", "event"],
            json_types={"sequence": int, "terminal": bool, "event": str},
            expected_json_values={"sequence": 4, "terminal": False, "event": "job.started"},
        ),
        case(
            "structured-03",
            "structured",
            'Return one JSON object with exactly `"tenant"`, `"job"`, and `"retryable"`. Values must be "violet", "j-17", and false.',
            json_keys=["tenant", "job", "retryable"],
            json_types={"tenant": str, "job": str, "retryable": bool},
            expected_json_values={"tenant": "violet", "job": "j-17", "retryable": False},
        ),
        case(
            "structured-04",
            "structured",
            'Return JSON only: `{"action": string, "count": integer, "valid": boolean}`. Use action "replay", count 3, valid true.',
            json_keys=["action", "count", "valid"],
            json_types={"action": str, "count": int, "valid": bool},
            expected_json_values={"action": "replay", "count": 3, "valid": True},
        ),
        case(
            "structured-05",
            "structured",
            'Return JSON only with exactly `"first"` and `"second"` string keys. Set first to "invariant" and second to "lease".',
            json_keys=["first", "second"],
            json_types={"first": str, "second": str},
            expected_json_values={"first": "invariant", "second": "lease"},
        ),
        case(
            "structured-06",
            "structured",
            'Return JSON only with `"ordered"` as an array of exactly three strings and `"ok"` as true. The array must be ["queued", "running", "succeeded"].',
            json_keys=["ordered", "ok"],
            expected_json_values={"ordered": ["queued", "running", "succeeded"], "ok": True},
        ),
    ]


def reasoning_cases() -> list[dict[str, Any]]:
    return [
        case(
            "reasoning-01",
            "reasoning",
            "A queue has 12 jobs. Four complete, three are cancelled, and two new jobs arrive. How many remain nonterminal? Give the integer and a one-sentence calculation.",
            required_terms=["7"],
        ),
        case(
            "reasoning-02",
            "reasoning",
            "Three events have sequence numbers 2, 4, and 3 and arrive in that order. If the reducer starts at version 1 and ignores stale events, what final version should it show? Give the integer.",
            required_terms=["4"],
        ),
        case(
            "reasoning-03",
            "reasoning",
            "A retry is safe only when both the tenant and request hash match. Tenant matches but the hash differs. Is the retry accepted or rejected? Answer with one of those two words and why.",
            required_terms=["rejected"],
        ),
        case(
            "reasoning-04",
            "reasoning",
            "A lease lasts 30 seconds and is renewed at 10 and 20 seconds. A completion arrives at 31 seconds without a successful renewal after 20. Should it publish? Answer yes or no and explain.",
            required_terms=["no"],
        ),
        case(
            "reasoning-05",
            "reasoning",
            "A durable replay sends sequences 5 and 6, then live delivery sends 6 and 7. How many distinct events should the browser apply? Give the integer and the rule.",
            required_terms=["3", "sequence"],
        ),
        case(
            "reasoning-06",
            "reasoning",
            "A state machine allows queued -> running -> succeeded, and cancellation from queued or running. Can succeeded -> running occur? Answer yes or no and cite terminality.",
            required_terms=["no", "terminal"],
        ),
    ]


def long_context_text() -> str:
    rows: list[str] = []
    for index in range(160):
        owner = ["Ada", "Borin", "Cleo", "Dara"][index % 4]
        shard = ["amber", "blue", "copper", "drift"][index % 4]
        rows.append(
            f"Record {index:03d}: owner={owner}; shard={shard}; retry_window={20 + index % 31}; "
            f"status={'active' if index % 5 else 'paused'}; checksum=R{index:03d}X; note=ordinary operational metadata."
        )
    rows[14] = "Record 014: owner=Iris; shard=amber; retry_window=37; status=active; checksum=A7C9; note=priority handoff after audit."
    rows[63] = "Record 063: owner=Noor; shard=violet; retry_window=44; status=paused; checksum=Violet-63; note=tenant isolation review."
    rows[119] = "Record 119: owner=Uma; shard=copper; retry_window=28; status=active; checksum=U119Z; note=never retry after terminal state."
    rows[142] = "Record 142: owner=Rafi; shard=drift; retry_window=51; status=active; checksum=DRIFT-142; note=last-event cursor is sequence-based."
    return "LONG CONTEXT LEDGER\n" + "\n".join(rows)


def retrieval_cases() -> list[dict[str, Any]]:
    context = long_context_text()
    return [
        case(
            "retrieval-01",
            "retrieval",
            f"{context}\n\nWhat are the owner, retry window, and checksum for Record 014? Include all three exact values.",
            required_terms=["iris", "37", "a7c9"],
            max_tokens=48,
        ),
        case(
            "retrieval-02",
            "retrieval",
            f"{context}\n\nWhich shard and status belong to Record 063, and what does its note concern? Answer with those details.",
            required_terms=["violet", "paused", "tenant"],
            max_tokens=48,
        ),
        case(
            "retrieval-03",
            "retrieval",
            f"{context}\n\nRetrieve Record 119 and state its exact note, then explain in one sentence why that constraint matters.",
            required_terms=["never retry", "terminal"],
            max_tokens=56,
        ),
        case(
            "retrieval-04",
            "retrieval",
            f"{context}\n\nCompare Record 014 with Record 142: give their shards and explain which one mentions a last-event cursor. Do not use other record numbers.",
            required_terms=["amber", "drift", "last-event"],
            max_tokens=56,
        ),
        case(
            "retrieval-05",
            "retrieval",
            f"{context}\n\nThe early constraint is in Record 119 and the later cursor fact is in Record 142. State both checksums and both constraints without inventing a new record.",
            required_terms=["u119z", "drift-142", "terminal", "cursor"],
            max_tokens=64,
        ),
    ]


def prose_cases() -> list[dict[str, Any]]:
    return [
        case(
            "prose-01",
            "prose",
            "Continue a short scene in exactly 3 sentences. The characters are Mira and Sol; the brass key is in the observatory; keep the mood calm and do not add a weapon.",
            required_terms=["mira", "sol", "observatory", "key"],
            forbidden_terms=["weapon", "sword", "gun"],
        ),
        case(
            "prose-02",
            "prose",
            "Summarize this continuity state in two sentences: Niko promised to return the blue map to Tessa before dusk, and the bridge is closed. Do not invent a destination.",
            required_terms=["niko", "tessa", "blue map", "bridge"],
            forbidden_terms=["castle", "mountain"],
        ),
        case(
            "prose-03",
            "prose",
            "Write a 4-line dialogue between Eda and Jun. Eda knows the garden bell is broken; Jun proposes waiting by the east gate. Preserve both facts and no others.",
            required_terms=["eda", "jun", "garden bell", "east gate"],
        ),
        case(
            "prose-04",
            "prose",
            "Give a concise continuity-safe recap: Orin carries a sealed letter, Vale distrusts the river road, and the next scene begins at noon. Keep all three facts.",
            required_terms=["orin", "vale", "sealed letter", "river road", "noon"],
        ),
        case(
            "prose-05",
            "prose",
            "Continue in two sentences: Lysa repaired the lantern but left its blue glass unchanged; Pera is waiting in the west room. Do not change the color or location.",
            required_terms=["lysa", "pera", "blue", "west room"],
            forbidden_terms=["red lantern", "east room"],
        ),
    ]


def quality_cases() -> list[dict[str, Any]]:
    cases = (
        lantern_code_cases()
        + instruction_cases()
        + structured_cases()
        + reasoning_cases()
        + retrieval_cases()
        + prose_cases()
    )
    if len(cases) != len({item["case_id"] for item in cases}):
        raise RuntimeError("quality case IDs are not unique")
    return cases


def parse_json_output(text: str) -> tuple[Any | None, bool]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.I)
        candidate = re.sub(r"\s*```$", "", candidate)
    decoder = json.JSONDecoder()
    for index, char in enumerate(candidate):
        if char not in "{[":
            continue
        try:
            value, end = decoder.raw_decode(candidate[index:])
        except json.JSONDecodeError:
            continue
        trailing = candidate[index + end :].strip()
        return value, not trailing
    return None, False


def repetition_check(token_ids: list[int]) -> dict[str, Any]:
    for ngram_size in (8, 6, 4):
        counts: dict[tuple[int, ...], int] = {}
        for index in range(0, max(len(token_ids) - ngram_size + 1, 0)):
            ngram = tuple(token_ids[index : index + ngram_size])
            counts[ngram] = counts.get(ngram, 0) + 1
        repeated = max(counts.values(), default=0)
        if repeated >= 3:
            return {
                "repetition_detected": True,
                "repeated_ngram_size": ngram_size,
                "repeated_ngram_count": repeated,
            }
    return {
        "repetition_detected": False,
        "repeated_ngram_size": None,
        "repeated_ngram_count": 0,
    }


def evaluate_case(case_data: dict[str, Any], text: str, token_ids: list[int]) -> dict[str, Any]:
    lower = text.lower()
    required = case_data["required_terms"]
    forbidden = case_data["forbidden_terms"]
    required_ok = {term: term.lower() in lower for term in required}
    forbidden_ok = {term: term.lower() not in lower for term in forbidden}
    line_prefixes = case_data.get("line_prefixes")
    lines = [line for line in text.strip().splitlines() if line.strip()]
    line_shape_ok = True
    if line_prefixes is not None:
        line_shape_ok = len(lines) == len(line_prefixes) and all(
            line.startswith(prefix) for line, prefix in zip(lines, line_prefixes)
        )
    max_chars = case_data.get("max_chars")
    max_chars_ok = max_chars is None or len(text.strip()) <= max_chars
    parsed, strict_json = parse_json_output(text) if case_data.get("json_keys") else (None, False)
    json_shape_ok = True
    json_values_ok = True
    if case_data.get("json_keys"):
        json_shape_ok = (
            isinstance(parsed, dict)
            and set(parsed) == set(case_data["json_keys"])
            and strict_json
        )
        if json_shape_ok:
            for key, type_name in case_data["json_types"].items():
                expected_type = {"str": str, "int": int, "bool": bool, "list": list}[type_name]
                if type(parsed[key]) is not expected_type:
                    json_shape_ok = False
            for key, expected in case_data["expected_json_values"].items():
                if parsed.get(key) != expected:
                    json_values_ok = False
    repetition = repetition_check(token_ids)
    objective_pass = bool(
        text.strip()
        and len(token_ids) >= 4
        and all(required_ok.values())
        and all(forbidden_ok.values())
        and line_shape_ok
        and max_chars_ok
        and json_shape_ok
        and json_values_ok
        and not repetition["repetition_detected"]
    )
    return {
        "objective_pass": objective_pass,
        "required_terms": required_ok,
        "forbidden_terms_absent": forbidden_ok,
        "line_shape_ok": line_shape_ok,
        "line_count": len(lines),
        "max_chars_ok": max_chars_ok,
        "json_parse_ok": strict_json if case_data.get("json_keys") else None,
        "json_shape_ok": json_shape_ok if case_data.get("json_keys") else None,
        "json_values_ok": json_values_ok if case_data.get("json_keys") else None,
        "parsed_json": parsed if case_data.get("json_keys") and strict_json else None,
        **repetition,
    }


def run_model(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "outputs").mkdir(exist_ok=True)
    (output_root / "tokens").mkdir(exist_ok=True)
    (output_root / "state_signatures").mkdir(exist_ok=True)
    (output_root / "logs").mkdir(exist_ok=True)
    cases = quality_cases()
    started = time.perf_counter()
    result: dict[str, Any] = {
        "experiment_id": f"P5-{args.label.upper()}-DIVERSE-QUALITY",
        "phase": 5,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": scale_utils.oracle_utils.current_commit(),
        "model_label": args.label,
        "model_path": args.model,
        "model": None,
        "case_count": len(cases),
        "cases": [],
        "exception": None,
        "overall_classification": None,
    }
    append_entries: list[dict[str, Any]] = []
    try:
        mx, load, generate_step, make_prompt_cache, make_sampler, _ = (
            scale_utils.oracle_utils.load_runtime()
        )
        result["model"] = scale_utils.oracle_utils.model_metadata(args.model)
        result["os_memory_before"] = scale_utils.oracle_utils.os_snapshot()
        model, tokenizer = load(args.model)
        result["memory_after_model_load"] = scale_utils.oracle_utils.memory_snapshot(
            mx, f"quality_{args.label}_after_model_load", started
        )
        for case_data in cases:
            case_started = time.perf_counter()
            case_result: dict[str, Any] = {
                "case_id": case_data["case_id"],
                "category": case_data["category"],
                "prompt_sha256": sha256_text(case_data["prompt"]),
                "max_tokens": case_data["max_tokens"],
                "exception": None,
            }
            try:
                canonical = scale_utils.canonical_prompt(
                    tokenizer, [{"role": "user", "content": case_data["prompt"]}]
                )
                cache = make_prompt_cache(model)
                stage = scale_utils.oracle_utils.generate_stage(
                    mx,
                    generate_step,
                    make_sampler,
                    tokenizer,
                    model,
                    canonical["token_ids"],
                    cache,
                    case_data["max_tokens"],
                    f"quality:{args.label}:{case_data['case_id']}",
                    started,
                )
                generated_ids = [int(value) for value in stage["generated_token_ids"]]
                generated_text = stage["decoded_output"]
                checks = evaluate_case(case_data, generated_text, generated_ids)
                state_path = output_root / "state_signatures" / f"{case_data['case_id']}.json"
                write_json(
                    state_path,
                    {
                        "case_id": case_data["case_id"],
                        "cache_diagnostics": stage["cache_diagnostics"],
                    },
                )
                output_path = output_root / "outputs" / f"{case_data['case_id']}.txt"
                output_path.write_text(generated_text, encoding="utf-8")
                token_path = output_root / "tokens" / f"{case_data['case_id']}.json"
                write_json(
                    token_path,
                    {
                        "token_ids": generated_ids,
                        "token_count": len(generated_ids),
                        "sha256": scale_utils.exact_hash(generated_ids),
                    },
                )
                case_result.update(
                    {
                        "prompt_tokens": len(canonical["token_ids"]),
                        "prompt_token_sha256": scale_utils.exact_hash(canonical["token_ids"]),
                        "generated_token_ids": generated_ids,
                        "generated_tokens": len(generated_ids),
                        "generated_token_hash": scale_utils.exact_hash(generated_ids),
                        "decoded_output": generated_text,
                        "decoded_output_sha256": sha256_text(generated_text),
                        "checks": checks,
                        "first_logprob_summary": stage["first_logprob_summary"],
                        "finish_reason": stage["finish_reason"],
                        "ttft_seconds": stage["ttft_seconds"],
                        "prompt_tps": stage["prompt_tps"],
                        "decode_tps": stage["decode_tps"],
                        "total_seconds": stage["total_seconds"],
                        "memory_after_generation": stage["memory_after_generation"],
                        "cache_state_signature": stage["cache_diagnostics"][-1].get(
                            "state_signature_sha256"
                        ),
                        "state_signature_path": str(state_path),
                        "raw_output_path": str(output_path),
                        "raw_token_path": str(token_path),
                    }
                )
                del cache
                scale_utils.oracle_utils.safe_clear(mx)
            except BaseException as exc:
                case_result["exception"] = f"{type(exc).__name__}: {exc}"
                case_result["traceback"] = traceback.format_exc()
            case_result["elapsed_seconds"] = time.perf_counter() - case_started
            result["cases"].append(case_result)
        result["os_memory_after"] = scale_utils.oracle_utils.os_snapshot()
        result["overall_classification"] = (
            "PASS"
            if len(result["cases"]) == len(cases)
            and all(item.get("exception") is None for item in result["cases"])
            else "PARTIAL_OR_ERROR"
        )
    except BaseException as exc:
        result["exception"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        result["overall_classification"] = "ERROR"
    finally:
        result["total_elapsed_seconds"] = time.perf_counter() - started
        write_json(output_root / "model.json", result)
        for item in result["cases"]:
            memory = item.get("memory_after_generation", {})
            checks = item.get("checks", {})
            append_entries.append(
                {
                    "timestamp": result["timestamp"],
                    "experiment_id": f"{result['experiment_id']}-{item['case_id']}",
                    "phase": 5,
                    "git_commit": result["git_commit"],
                    "model_label": args.label,
                    "model_name": f"Qwen3.8-27B-3bit-{args.label}",
                    "model_path": args.model,
                    "quant_bits": 3,
                    "quant_group_size": 64 if args.label == "g64" else 128,
                    "execution_mode": "fresh_case_cache_quality_suite",
                    "persistent_cache": False,
                    "kv_bits": None,
                    "case_id": item["case_id"],
                    "category": item["category"],
                    "prompt_tokens": item.get("prompt_tokens"),
                    "generated_tokens": item.get("generated_tokens"),
                    "generated_token_hash": item.get("generated_token_hash"),
                    "classification": (
                        "COHERENT" if checks.get("objective_pass") else "OBJECTIVE_FAIL"
                    )
                    if item.get("exception") is None
                    else "ERROR",
                    "objective_pass": checks.get("objective_pass"),
                    "structured_output_valid": checks.get("json_parse_ok"),
                    "repetition_detected": checks.get("repetition_detected"),
                    "ttft_seconds": item.get("ttft_seconds"),
                    "decode_tps": item.get("decode_tps"),
                    "total_seconds": item.get("total_seconds"),
                    "peak_memory_bytes": memory.get("peak_bytes"),
                    "active_plus_cache_bytes": memory.get("active_plus_cache_bytes"),
                    "exception": item.get("exception"),
                    "raw_artifact": str(output_root / "model.json"),
                }
            )
        with (REPO_ROOT / "research/stage3/results.jsonl").open("a", encoding="utf-8") as handle:
            for entry in append_entries:
                handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
        print(
            json.dumps(
                {
                    "experiment_id": result["experiment_id"],
                    "model_label": args.label,
                    "case_count": len(result["cases"]),
                    "overall_classification": result["overall_classification"],
                    "exception": result.get("exception"),
                    "elapsed_seconds": result["total_elapsed_seconds"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return 1 if result.get("exception") else 0


def lcp(left: list[int], right: list[int]) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def load_result(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compare_models(output_root: Path) -> dict[str, Any]:
    g64 = load_result(output_root / "g64" / "model.json")
    g128 = load_result(output_root / "g128" / "model.json")
    by64 = {item["case_id"]: item for item in g64["cases"]}
    by128 = {item["case_id"]: item for item in g128["cases"]}
    rows: list[dict[str, Any]] = []
    for case_data in quality_cases():
        case_id = case_data["case_id"]
        left = by64[case_id]
        right = by128[case_id]
        left_ids = left.get("generated_token_ids", [])
        right_ids = right.get("generated_token_ids", [])
        common = lcp(left_ids, right_ids)
        rows.append(
            {
                "case_id": case_id,
                "category": case_data["category"],
                "g64_objective_pass": left.get("checks", {}).get("objective_pass"),
                "g128_objective_pass": right.get("checks", {}).get("objective_pass"),
                "g64_generated_tokens": left.get("generated_tokens"),
                "g128_generated_tokens": right.get("generated_tokens"),
                "g64_output_sha256": left.get("decoded_output_sha256"),
                "g128_output_sha256": right.get("decoded_output_sha256"),
                "first_divergence_token": common
                if left_ids != right_ids
                else None,
                "token_lcp": common,
                "token_equal": left_ids == right_ids,
                "g64_repetition": left.get("checks", {}).get("repetition_detected"),
                "g128_repetition": right.get("checks", {}).get("repetition_detected"),
                "g64_json_valid": left.get("checks", {}).get("json_parse_ok"),
                "g128_json_valid": right.get("checks", {}).get("json_parse_ok"),
                "g64_exception": left.get("exception"),
                "g128_exception": right.get("exception"),
            }
        )
    categories: dict[str, dict[str, Any]] = {}
    for category in sorted({row["category"] for row in rows}):
        subset = [row for row in rows if row["category"] == category]
        pass64 = sum(bool(row["g64_objective_pass"]) for row in subset)
        pass128 = sum(bool(row["g128_objective_pass"]) for row in subset)
        categories[category] = {
            "cases": len(subset),
            "g64_objective_pass": pass64,
            "g128_objective_pass": pass128,
            "g64_rate": pass64 / len(subset),
            "g128_rate": pass128 / len(subset),
            "delta_g128_minus_g64": (pass128 - pass64) / len(subset),
            "token_equal_count": sum(row["token_equal"] for row in subset),
            "mean_token_lcp": sum(row["token_lcp"] for row in subset) / len(subset),
        }
    total = len(rows)
    pass64 = sum(bool(row["g64_objective_pass"]) for row in rows)
    pass128 = sum(bool(row["g128_objective_pass"]) for row in rows)
    category_deltas = [item["delta_g128_minus_g64"] for item in categories.values()]
    rate64 = pass64 / total
    rate128 = pass128 / total
    if rate128 >= rate64 - 0.05 and min(category_deltas, default=0.0) >= -0.15:
        verdict = "DEFAULT"
    elif rate128 >= rate64 - 0.20 and min(category_deltas, default=0.0) >= -0.35:
        verdict = "PERFORMANCE/LONG-CONTEXT MODE"
    else:
        verdict = "EXPERIMENTAL ONLY"
    comparison = {
        "experiment_id": "P5-G64-VS-G128-DIVERSE-QUALITY",
        "phase": 5,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": scale_utils.oracle_utils.current_commit(),
        "case_count": total,
        "categories": categories,
        "g64_total_objective_pass": pass64,
        "g128_total_objective_pass": pass128,
        "g64_total_rate": rate64,
        "g128_total_rate": rate128,
        "objective_rate_delta_g128_minus_g64": rate128 - rate64,
        "quality_verdict_for_g128": verdict,
        "rows": rows,
        "teacher_forced_reference": {
            "source": "research/night2/quality/P9-G64-nll-full4096-chunk256.json and P9-G128-nll-full4096-chunk256.json",
            "source_commit": "4e10a8c09db8a3f2f51d9421cb9d1b71abcf564e",
            "evaluated_targets": 4095,
            "context_tokens": 4096,
            "g64_average_nll_nats": 4.138705738705739,
            "g128_average_nll_nats": 5.833455433455433,
            "g128_minus_g64_nll_nats": 1.694749694749694,
            "interpretation": "prior teacher-forced evidence; supplementary to this diverse generation suite",
        },
    }
    write_json(output_root / "comparison.json", comparison)
    lines = [
        "# Phase 5 g64 vs g128 diverse quality",
        "",
        f"g128 verdict: **{verdict}**",
        "",
        f"Objective pass rate: g64 {pass64}/{total} ({rate64:.1%}); g128 {pass128}/{total} ({rate128:.1%}).",
        "",
        "| category | cases | g64 pass | g128 pass | delta | mean token LCP |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for category, summary in categories.items():
        lines.append(
            f"| {category} | {summary['cases']} | {summary['g64_objective_pass']} | "
            f"{summary['g128_objective_pass']} | {summary['delta_g128_minus_g64']:+.1%} | "
            f"{summary['mean_token_lcp']:.1f} |"
        )
    lines.extend(
        [
            "",
            "Prior teacher-forced 4K reference: g64 mean NLL 4.1387 nats; "
            "g128 5.8335 nats; delta +1.6947 nats. This is supplementary, "
            "not the sole quality verdict.",
        ]
    )
    (output_root / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (REPO_ROOT / "research/stage3/results.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "timestamp": comparison["timestamp"],
                    "experiment_id": comparison["experiment_id"],
                    "phase": 5,
                    "git_commit": comparison["git_commit"],
                    "case_count": total,
                    "g64_total_objective_pass": pass64,
                    "g128_total_objective_pass": pass128,
                    "g64_total_rate": rate64,
                    "g128_total_rate": rate128,
                    "objective_rate_delta_g128_minus_g64": rate128 - rate64,
                    "quality_verdict_for_g128": verdict,
                    "raw_artifact": str(output_root / "comparison.json"),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )
    return comparison


def run_all(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cases = quality_cases()
    if len(cases) < 30 or len(cases) > 50:
        raise SystemExit(f"unexpected quality suite size: {len(cases)}")
    for label, model in (("g64", G64_MODEL), ("g128", G128_MODEL)):
        model_root = output_root / label
        if model_root.exists() and any(model_root.iterdir()):
            raise SystemExit(f"refusing to overwrite non-empty quality directory: {model_root}")
        model_root.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--mode",
            "model",
            "--label",
            label,
            "--model",
            model,
            "--output-root",
            str(model_root),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        (model_root / "logs" / "launcher.stdout.log").write_text(
            completed.stdout, encoding="utf-8"
        )
        (model_root / "logs" / "launcher.stderr.log").write_text(
            completed.stderr, encoding="utf-8"
        )
        if completed.returncode != 0 or not (model_root / "model.json").exists():
            return completed.returncode or 1
    comparison = compare_models(output_root)
    print(
        json.dumps(
            {
                "experiment_id": comparison["experiment_id"],
                "case_count": comparison["case_count"],
                "g64_total_rate": comparison["g64_total_rate"],
                "g128_total_rate": comparison["g128_total_rate"],
                "quality_verdict_for_g128": comparison["quality_verdict_for_g128"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("run", "model", "compare"), required=True)
    parser.add_argument("--label", choices=("g64", "g128"), default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()
    if args.mode == "run":
        raise SystemExit(run_all(args))
    if args.mode == "model":
        if args.label is None or args.model is None:
            raise SystemExit("--label and --model are required for model mode")
        raise SystemExit(run_model(args))
    compare_models(Path(args.output_root))


if __name__ == "__main__":
    main()
