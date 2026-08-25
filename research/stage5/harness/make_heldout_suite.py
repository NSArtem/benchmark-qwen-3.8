#!/usr/bin/env python3
"""Create and freeze the completely held-out Stage 5 quality suite.

The cases and expected checks are source-controlled here so the generated JSONL
is deterministic.  They are intentionally separate from every Stage 4 suite
and are not imported by calibration, sensitivity, or candidate-selection code.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "research/stage5/heldout"
STAGE4_EVAL_SHA = "f24e3983efbd898084852c3fde7b0f824604bc6a63349de5f32efeaa36ca3de0"
STAGE4_CAL_SHA = "e856f39622c48f29f913d65d55f18d4ff9fb9bca16b0b23736361347472b9558"


def case(
    case_id: str,
    category: str,
    prompt: str,
    *,
    score_kind: str = "objective",
    max_tokens: int = 256,
    required_groups: list[list[str]] | None = None,
    forbidden_terms: list[str] | None = None,
    line_prefixes: list[str] | None = None,
    exact_lines: list[str] | None = None,
    max_chars: int | None = None,
    structured: dict[str, Any] | None = None,
    answer_regex: str | None = None,
    teacher_target: str | None = None,
    rubric: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "category": category,
        "prompt": prompt,
        "score_kind": score_kind,
        "max_tokens": max_tokens,
        "required_groups": required_groups or [],
        "forbidden_terms": forbidden_terms or [],
        "line_prefixes": line_prefixes,
        "exact_lines": exact_lines,
        "max_chars": max_chars,
        "structured": structured,
        "answer_regex": answer_regex,
        "teacher_target": teacher_target,
        "rubric": rubric or [],
    }


def coding_cases() -> list[dict[str, Any]]:
    return [
        case(
            "h5-coding-01",
            "coding",
            "An API stores an idempotency row containing tenant_id, route, key, request_sha256, and the serialized response. Two retries may differ only in JSON whitespace. Give the database uniqueness rule, the canonicalization rule for hashing, and the behavior for a matching versus changed body.",
            max_tokens=384,
            required_groups=[
                ["unique", "unique index", "constraint"],
                ["tenant", "route", "scope"],
                ["canonical", "normalize", "normalized"],
                ["hash", "sha"],
                ["same", "replay", "original"],
                ["conflict", "reject", "mismatch", "different"],
            ],
            teacher_target="Enforce uniqueness on the tenant, route, and idempotency key. Hash a canonical JSON representation so whitespace does not change the request identity. A matching hash returns the stored response; a different hash returns a conflict.",
        ),
        case(
            "h5-coding-02",
            "coding",
            "A worker claims a job with lease_epoch = 12 and later tries to complete it. Write the essential conditional UPDATE predicate that fences a worker after a newer claimant has epoch 13. Explain what zero affected rows means and what the worker must do.",
            max_tokens=384,
            required_groups=[
                ["lease"],
                ["epoch", "generation", "token"],
                ["where", "conditional", "predicate"],
                ["zero", "0", "affected", "row"],
                ["reject", "abort", "do not", "not publish", "stop"],
            ],
        ),
        case(
            "h5-coding-03",
            "coding",
            "An outbox publisher marks a message sent before contacting the downstream service. The process crashes after the mark but before the call. Propose the smallest at-least-once correction, including an idempotent consumer or deduplication key.",
            max_tokens=384,
            required_groups=[
                ["outbox"],
                ["crash", "failure", "before"],
                ["at-least-once", "retry", "after"],
                ["idempotent", "dedup", "duplicate"],
                ["mark", "ack", "sent", "claim"],
            ],
        ),
        case(
            "h5-coding-04",
            "coding",
            "A browser reducer receives job.started sequence 20, job.succeeded sequence 22, then a duplicate job.started sequence 20 and a late job.running sequence 21. State the ordering invariant and the terminal-state guard; say how each late event is handled.",
            max_tokens=384,
            required_groups=[
                ["sequence", "order", "monotonic"],
                ["terminal", "succeeded"],
                ["ignore", "drop", "discard", "reject"],
                ["duplicate", "late", "stale"],
                ["running", "started"],
            ],
        ),
        case(
            "h5-coding-05",
            "coding",
            "A multi-tenant repository has `SELECT * FROM jobs WHERE id = $1`, and callers pass tenant_id separately. Give the corrected predicate and one security regression assertion using two tenants with the same job id.",
            max_tokens=384,
            required_groups=[
                ["tenant"],
                ["where", "and", "predicate"],
                ["id"],
                ["same", "match", "both"],
                ["not", "cannot", "never", "forbid", "isolation"],
                ["assert", "test", "expect"],
            ],
        ),
        case(
            "h5-coding-06",
            "coding",
            "An SSE server takes a durable snapshot at cursor 40 and subscribes to live events afterward. Event 41 can be published between those operations. Give one minimal ordering fix and explain the overlap check that guarantees delivery without showing an event twice.",
            max_tokens=384,
            required_groups=[
                ["cursor", "sequence", "watermark"],
                ["subscribe", "subscription", "before"],
                ["second read", "reread", "overlap", "replay"],
                ["dedup", "duplicate", "de-dup"],
                ["41", "event"],
            ],
        ),
    ]


def instruction_cases() -> list[dict[str, Any]]:
    return [
        case(
            "h5-instruction-01",
            "instruction",
            "Output exactly four lines and nothing else. Use these exact lines: ROLE: evaluator; CACHE: exact-suffix; SCOPE: single-user; RESULT: READY.",
            max_tokens=128,
            line_prefixes=["ROLE:", "CACHE:", "SCOPE:", "RESULT:"],
            exact_lines=["ROLE: evaluator", "CACHE: exact-suffix", "SCOPE: single-user", "RESULT: READY"],
            max_chars=140,
            teacher_target="ROLE: evaluator\nCACHE: exact-suffix\nSCOPE: single-user\nRESULT: READY",
        ),
        case(
            "h5-instruction-02",
            "instruction",
            "Output exactly three bullet lines, with no heading or explanation. Put KEEP first, DROP second, and RESET third.",
            max_tokens=128,
            line_prefixes=["- ", "- ", "- "],
            required_groups=[["keep"], ["drop"], ["reset"]],
            max_chars=100,
        ),
        case(
            "h5-instruction-03",
            "instruction",
            "Answer in exactly one sentence of no more than 16 words. Include the terms mixed34 and 16K. Do not mention g128, g64, KV8, or DWQ.",
            max_tokens=128,
            required_groups=[["mixed34"], ["16k"]],
            forbidden_terms=["g128", "g64", "kv8", "dwq"],
            max_chars=180,
        ),
        case(
            "h5-instruction-04",
            "instruction",
            "Return only this exact uppercase word, with no punctuation or explanation: HOLDOUT_OK",
            max_tokens=64,
            exact_lines=["HOLDOUT_OK"],
            max_chars=30,
        ),
    ]


def structured_cases() -> list[dict[str, Any]]:
    return [
        case(
            "h5-structured-01",
            "structured",
            'Return JSON only with exactly these keys: "route" (string), "retry" (boolean), and "limit" (integer). Use "events", false, and 12.',
            max_tokens=160,
            structured={
                "keys": ["route", "retry", "limit"],
                "types": {"route": "str", "retry": "bool", "limit": "int"},
                "values": {"route": "events", "retry": False, "limit": 12},
            },
            teacher_target='{"route":"events","retry":false,"limit":12}',
        ),
        case(
            "h5-structured-02",
            "structured",
            'Return one JSON object with exactly "tenant" (string), "sequence" (integer), and "terminal" (boolean). Use "cobalt", 17, and true.',
            max_tokens=160,
            structured={
                "keys": ["tenant", "sequence", "terminal"],
                "types": {"tenant": "str", "sequence": "int", "terminal": "bool"},
                "values": {"tenant": "cobalt", "sequence": 17, "terminal": True},
            },
        ),
        case(
            "h5-structured-03",
            "structured",
            'Return JSON only with exactly "items" (array), "count" (integer), and "valid" (boolean). Use ["alpha", "beta"], 2, and true.',
            max_tokens=160,
            structured={
                "keys": ["items", "count", "valid"],
                "types": {"items": "list", "count": "int", "valid": "bool"},
                "values": {"items": ["alpha", "beta"], "count": 2, "valid": True},
            },
        ),
        case(
            "h5-structured-04",
            "structured",
            'Return JSON only with exactly "lease_epoch" (integer), "owner" (string), and "active" (boolean). Use 4, "worker-2", and false.',
            max_tokens=160,
            structured={
                "keys": ["lease_epoch", "owner", "active"],
                "types": {"lease_epoch": "int", "owner": "str", "active": "bool"},
                "values": {"lease_epoch": 4, "owner": "worker-2", "active": False},
            },
        ),
    ]


def reasoning_cases() -> list[dict[str, Any]]:
    return [
        case(
            "h5-reasoning-01",
            "reasoning",
            "A queue starts with 19 jobs. Six succeed, four are cancelled, and three new jobs arrive. How many jobs are nonterminal now? End with `Final: <integer>`.",
            max_tokens=256,
            answer_regex=r"final\s*:\s*12\b",
            teacher_target="19 - 6 - 4 + 3 = 12 nonterminal jobs. Final: 12",
        ),
        case(
            "h5-reasoning-02",
            "reasoning",
            "A display starts at version 3 and accepts a version only when it is greater than the current value. It receives 4, 2, 9, and 7. What final version is displayed? End with `Final: <integer>`.",
            max_tokens=256,
            answer_regex=r"final\s*:\s*9\b",
        ),
        case(
            "h5-reasoning-03",
            "reasoning",
            "A batch contains 8 groups of 14 records plus 3 independent records. How many records are there in total? End with `Final: <integer>`.",
            max_tokens=256,
            answer_regex=r"final\s*:\s*115\b",
        ),
        case(
            "h5-reasoning-04",
            "reasoning",
            "A client receives event sequences [11, 12, 11, 14, 13, 14]. It applies each distinct sequence once. How many events are applied? End with `Final: <integer>`.",
            max_tokens=256,
            answer_regex=r"final\s*:\s*4\b",
        ),
    ]


def ledger_context() -> str:
    rows: list[str] = []
    owners = ["Anya", "Basil", "Cora", "Dion", "Esha", "Farid", "Gita", "Hugo"]
    zones = ["coral", "flint", "mint", "navy", "ochre", "plum", "rose", "slate"]
    for index in range(96):
        rows.append(
            f"Entry {index:03d}: owner={owners[index % len(owners)]}; "
            f"zone={zones[index % len(zones)]}; window={18 + (index * 11) % 37}; "
            f"state={'ready' if index % 4 else 'paused'}; digest=E{index:03d}Q; note=ordinary queue metadata."
        )
    rows[14] = "Entry 014: owner=Kira; zone=orchid; window=33; state=ready; digest=ORC-14; note=priority handoff after audit."
    rows[57] = "Entry 057: owner=Malik; zone=teal; window=46; state=paused; digest=TEAL-57; note=tenant boundary review."
    rows[93] = "Entry 093: owner=Veda; zone=iron; window=27; state=ready; digest=IRON-93; note=cursor resumes after reconnect."
    return "FROZEN SUPPORT LEDGER\n" + "\n".join(rows)


def retrieval_cases() -> list[dict[str, Any]]:
    context = ledger_context()
    return [
        case(
            "h5-retrieval-01",
            "retrieval",
            f"{context}\n\nRetrieve Entry 014. State its owner, zone, window, and digest exactly.",
            max_tokens=256,
            required_groups=[["kira"], ["orchid"], ["33"], ["orc-14"]],
            teacher_target="Entry 014 is owned by Kira, in zone orchid, with window 33 and digest ORC-14.",
        ),
        case(
            "h5-retrieval-02",
            "retrieval",
            f"{context}\n\nFor Entry 057, give the exact zone and state, then state what its note concerns.",
            max_tokens=256,
            required_groups=[["teal"], ["paused"], ["tenant", "boundary"]],
        ),
        case(
            "h5-retrieval-03",
            "retrieval",
            f"{context}\n\nCompare Entry 014 and Entry 093. Give both zones, both digests, and identify which entry mentions cursor resumption.",
            max_tokens=256,
            required_groups=[["orchid"], ["iron"], ["orc-14"], ["iron-93"], ["cursor", "resum"]],
        ),
    ]


def prose_cases() -> list[dict[str, Any]]:
    return [
        case(
            "h5-prose-01",
            "prose",
            "Continue a quiet scene in exactly three sentences. Elena and Pavel are in the map room, and the red notebook is on the north desk. Do not add a weapon or move the notebook.",
            score_kind="semi-objective",
            max_tokens=256,
            required_groups=[["elena"], ["pavel"], ["map room", "map-room", "maproom"], ["red notebook", "notebook"], ["north desk", "north"]],
            forbidden_terms=["weapon", "gun", "knife", "sword", "south desk", "east desk", "west desk"],
            teacher_target="Elena and Pavel studied the map in the quiet room. The red notebook remained on the north desk. They continued their calm review.",
            rubric=["Preserves the two named people and the map-room setting.", "Keeps the red notebook on the north desk.", "Uses exactly three sentences in a calm continuation."],
        ),
        case(
            "h5-prose-02",
            "prose",
            "Write exactly two neutral status sentences. The migration window is Friday 03:00 UTC, the backup region is Dublin, and the operator is Niko. Do not change any of those facts.",
            score_kind="semi-objective",
            max_tokens=192,
            required_groups=[["friday"], ["03:00", "3:00"], ["utc"], ["dublin"], ["niko"]],
            forbidden_terms=["saturday", "london", "madrid", "monday"],
            rubric=["Preserves the time, timezone, region, and operator.", "Uses exactly two neutral sentences."],
        ),
        case(
            "h5-prose-03",
            "prose",
            "In exactly three sentences, summarize a handoff: Rowan approved the checksum, the blue folder remains in cabinet 2, and the next review is at noon. Keep the tone factual and do not invent a different folder or reviewer.",
            score_kind="semi-objective",
            max_tokens=224,
            required_groups=[["rowan"], ["checksum"], ["blue folder"], ["cabinet 2", "cabinet"], ["noon"]],
            forbidden_terms=["red folder", "cabinet 3", "midnight", "evening"],
            rubric=["Preserves the named approver, folder, cabinet, and review time.", "Uses exactly three factual sentences."],
        ),
    ]


def build_cases() -> list[dict[str, Any]]:
    return coding_cases() + instruction_cases() + structured_cases() + reasoning_cases() + retrieval_cases() + prose_cases()


def main() -> None:
    cases = build_cases()
    ids = [item["case_id"] for item in cases]
    categories = Counter(item["category"] for item in cases)
    expected_categories = {"coding": 6, "instruction": 4, "structured": 4, "reasoning": 4, "retrieval": 3, "prose": 3}
    if len(cases) != 24 or len(set(ids)) != len(ids) or dict(categories) != expected_categories:
        raise AssertionError(f"held-out shape invalid: count={len(cases)} categories={dict(categories)}")
    if not all(item["case_id"].startswith("h5-") for item in cases):
        raise AssertionError("held-out IDs must be Stage 5-only")
    if any(not item["prompt"].strip() for item in cases):
        raise AssertionError("empty prompt")

    OUT.mkdir(parents=True, exist_ok=True)
    suite_path = OUT / "heldout_v1.jsonl"
    payload = "".join(json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for item in cases)
    suite_path.write_text(payload, encoding="utf-8")
    suite_sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    teacher_ids = [item["case_id"] for item in cases if item.get("teacher_target")]
    manifest = {
        "suite": "Qwen3.8 Stage 5 Completely Held-Out Quality Suite",
        "version": 1,
        "local_only": True,
        "case_count": len(cases),
        "categories": dict(sorted(categories.items())),
        "case_ids": ids,
        "suite_path": str(suite_path),
        "suite_sha256": suite_sha,
        "teacher_forced_case_ids": teacher_ids,
        "teacher_forced_count": len(teacher_ids),
        "stage4_calibration_sha256": STAGE4_CAL_SHA,
        "stage4_evaluation_sha256": STAGE4_EVAL_SHA,
        "selection_policy": "Not used for Stage 5 calibration, sensitivity, allocation, mini-tournament selection, or candidate rejection.",
        "manual_verification": {
            "status": "passed",
            "objective_arithmetic": {"h5-reasoning-01": 12, "h5-reasoning-02": 9, "h5-reasoning-03": 115, "h5-reasoning-04": 4},
            "structured_values_checked": True,
            "retrieval_anchor_entries_checked": [14, 57, 93],
            "prose_named_facts_checked": True,
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
