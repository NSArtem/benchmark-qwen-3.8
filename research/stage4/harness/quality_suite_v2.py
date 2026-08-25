#!/usr/bin/env python3
"""Build the frozen local Qwen3.8 Quality Suite V2.

The suite is deliberately data-first: this module is the reproducible source
for the checked-in calibration/evaluation JSONL files.  No live web data is
used.  Objective checks are encoded as structured metadata so the model runner
can report them separately from semi-objective continuity checks and the blind
pairwise review rubric.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "research/stage4"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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
            "coding-01",
            "coding",
            """Two identical POST requests for the same tenant, idempotency key, and body can run concurrently. The current code SELECTs the key, then inserts the job and key if no row was found. Diagnose the race and give the smallest database-level fix. Explain how a retry returns the original result.""",
            max_tokens=384,
            required_groups=[
                ["unique constraint", "unique index", "unique key", "on conflict"],
                ["idempotency"],
                ["transaction", "atomic", "same commit", "upsert", "conflict", "unique constraint"],
                ["original", "existing", "replay"],
            ],
            teacher_target="Add a unique constraint on tenant_id plus idempotency_key and perform the insert atomically with ON CONFLICT. Store and return the original result for a matching body; reject a different body hash.",
        ),
        case(
            "coding-02",
            "coding",
            """A reducer has already applied terminal event sequence 6 = succeeded. A delayed event sequence 7 says running. State the invariant and the guard that prevents a terminal job from becoming active again. Include what the reducer should do with this event.""",
            max_tokens=384,
            required_groups=[
                ["terminal"],
                ["monotonic", "sequence", "version", "state"],
                ["ignore", "reject", "discard", "drop"],
                ["running", "active"],
            ],
        ),
        case(
            "coding-03",
            "coding",
            """An API updates a job state and commits. It then inserts the corresponding outbox row in a second transaction. Explain the atomicity failure and state the transaction boundary that must contain both operations. Mention the failure symptom.""",
            max_tokens=384,
            required_groups=[
                ["atomic", "atomicity"],
                ["outbox"],
                ["same transaction", "single transaction", "one transaction"],
                ["commit", "crash", "missing", "inconsistent"],
            ],
        ),
        case(
            "coding-04",
            "coding",
            """Worker A's lease expires and Worker B acquires the same job with a new lease token. Worker A finishes rendering and tries to publish success. Give a minimal completion predicate and the database/Redis mechanism that fences Worker A.""",
            max_tokens=384,
            required_groups=[
                ["lease"],
                ["token", "owner", "fencing"],
                ["compare", "conditional", "where", "check"],
                ["reject", "do not publish", "ignore", "fail"],
            ],
        ),
        case(
            "coding-05",
            "coding",
            """An SSE endpoint reads durable PostgreSQL events and then subscribes to Redis Streams. A disconnect can happen between those steps. Describe a cursor-based fix using Last-Event-ID, a monotonic sequence, and overlap de-duplication.""",
            max_tokens=384,
            required_groups=[
                ["last-event-id", "last event id", "cursor"],
                ["sequence", "monotonic", "ordering"],
                ["dedup", "de-dup", "duplicate", "overlap"],
                ["replay", "resume", "catch up", "gap", "delivery"],
            ],
        ),
        case(
            "coding-06",
            "coding",
            """A repository method selects a job by id only, although every request supplies tenantId. Identify the security defect and write the required query predicate in words or SQL. State what the test must prove.""",
            max_tokens=384,
            required_groups=[
                ["tenant"],
                ["where", "predicate", "AND"],
                ["id"],
                ["isolation", "cross-tenant", "not read", "forbidden"],
            ],
        ),
        case(
            "coding-07",
            "coding",
            """Design one focused integration test for a timeout-after-commit retry. Send two concurrent POSTs with the same idempotency key, delay the worker, and inspect the database. Give the setup and at least two exact assertions. Also assert that exactly one outbox event exists.""",
            max_tokens=512,
            required_groups=[
                ["two", "concurrent", "parallel"],
                ["same", "idempotency"],
                ["one job", "one row", "exactly one", "single job", "no duplicate"],
                ["outbox"],
                ["assert", "expect", "verify"],
            ],
        ),
        case(
            "coding-08",
            "coding",
            """Write a minimal regression-test plan for a stale event arriving after a terminal state. Include the event sequences, the terminal-state invariant, and the assertion that the final state is not reopened.""",
            max_tokens=384,
            required_groups=[
                ["terminal"],
                ["stale", "late", "delayed"],
                ["sequence", "version"],
                ["assert", "expect", "verify"],
                ["not", "never", "does not", "no "],
            ],
        ),
    ]


def instruction_cases() -> list[dict[str, Any]]:
    return [
        case(
            "instruction-01",
            "instruction",
            "Output exactly four lines and nothing else. Use these prefixes and values: MODE: greedy; CHECK: schema; LIMIT: one; END: DONE.",
            max_tokens=128,
            line_prefixes=["MODE:", "CHECK:", "LIMIT:", "END:"],
            exact_lines=["MODE: greedy", "CHECK: schema", "LIMIT: one", "END: DONE"],
            max_chars=100,
            teacher_target="MODE: greedy\nCHECK: schema\nLIMIT: one\nEND: DONE",
        ),
        case(
            "instruction-02",
            "instruction",
            "Output exactly three bullet lines, with no heading or explanation. The lines must contain PASS, REJECT, and RESET in that order.",
            max_tokens=128,
            line_prefixes=["- ", "- ", "- "],
            required_groups=[["pass"], ["reject"], ["reset"]],
            max_chars=100,
        ),
        case(
            "instruction-03",
            "instruction",
            "Answer in exactly one sentence of no more than 18 words. Include the terms g128 and normal KV. Do not mention any other quantization or cache mode.",
            max_tokens=128,
            required_groups=[["g128"], ["normal kv"]],
            forbidden_terms=["g64", "kv8", "2-bit", "4-bit", "dwq", "awq", "gptq", "mixed"],
            max_chars=180,
        ),
        case(
            "instruction-04",
            "instruction",
            "Return only this exact uppercase word, with no punctuation or explanation: COHERENT",
            max_tokens=64,
            exact_lines=["COHERENT"],
            max_chars=20,
        ),
    ]


def structured_cases() -> list[dict[str, Any]]:
    return [
        case(
            "structured-01",
            "structured",
            'Return JSON only with exactly these keys: "state" (string), "safe" (boolean), and "attempts" (integer). Use "queued", true, and 1.',
            max_tokens=160,
            structured={
                "keys": ["state", "safe", "attempts"],
                "types": {"state": "str", "safe": "bool", "attempts": "int"},
                "values": {"state": "queued", "safe": True, "attempts": 1},
            },
            teacher_target='{"state":"queued","safe":true,"attempts":1}',
        ),
        case(
            "structured-02",
            "structured",
            'Return one JSON object with exactly "tenant" (string), "job" (string), and "retryable" (boolean). Use "violet", "j-17", and false.',
            max_tokens=160,
            structured={
                "keys": ["tenant", "job", "retryable"],
                "types": {"tenant": "str", "job": "str", "retryable": "bool"},
                "values": {"tenant": "violet", "job": "j-17", "retryable": False},
            },
        ),
        case(
            "structured-03",
            "structured",
            'Return JSON only with exactly "sequence" (integer), "terminal" (boolean), and "event" (string). Use 4, false, and "job.started".',
            max_tokens=160,
            structured={
                "keys": ["sequence", "terminal", "event"],
                "types": {"sequence": "int", "terminal": "bool", "event": "str"},
                "values": {"sequence": 4, "terminal": False, "event": "job.started"},
            },
        ),
        case(
            "structured-04",
            "structured",
            'Return JSON only with exactly "ordered" (array of strings) and "ok" (boolean). Use ["queued", "running", "succeeded"] and true.',
            max_tokens=160,
            structured={
                "keys": ["ordered", "ok"],
                "types": {"ordered": "list", "ok": "bool"},
                "values": {"ordered": ["queued", "running", "succeeded"], "ok": True},
            },
        ),
    ]


def reasoning_cases() -> list[dict[str, Any]]:
    return [
        case(
            "reasoning-01",
            "reasoning",
            "At the start there are 12 queued jobs. Four complete, three are cancelled, and two new jobs arrive. How many jobs are nonterminal now? End with `Final: <integer>`.",
            max_tokens=256,
            answer_regex=r"final\s*:\s*7\b",
            teacher_target="There are 12 - 4 - 3 + 2 = 7 nonterminal jobs. Final: 7",
        ),
        case(
            "reasoning-02",
            "reasoning",
            "A reducer starts at version 1 and receives event versions 2, 5, and 4 in that order. It applies a version only when it is greater than the current version. What final version is displayed? End with `Final: <integer>`.",
            max_tokens=256,
            answer_regex=r"final\s*:\s*5\b",
        ),
        case(
            "reasoning-03",
            "reasoning",
            "A durable replay sends event sequences 8 and 9, then live delivery sends 9 and 10. With sequence-based de-duplication, how many distinct events should the client apply? End with `Final: <integer>`.",
            max_tokens=256,
            answer_regex=r"final\s*:\s*3\b",
        ),
        case(
            "reasoning-04",
            "reasoning",
            "A batch contains 7 groups of 18 records plus 5 independent records. How many records are there in total? End with `Final: <integer>`.",
            max_tokens=256,
            answer_regex=r"final\s*:\s*131\b",
        ),
    ]


def ledger_context() -> str:
    rows: list[str] = []
    owners = ["Ada", "Borin", "Cleo", "Dara", "Eli", "Faye"]
    shards = ["amber", "blue", "copper", "drift", "jade", "silver"]
    # 140 records are approximately 4–5K tokenizer tokens on the local
    # template: long enough to test distant retrieval, short enough to keep a
    # full two-model suite practical on the 18 GB machine.
    for index in range(140):
        rows.append(
            f"Record {index:03d}: owner={owners[index % len(owners)]}; "
            f"shard={shards[index % len(shards)]}; retry_window={20 + (index * 7) % 41}; "
            f"status={'active' if index % 5 else 'paused'}; checksum=R{index:03d}X; "
            "note=ordinary operational metadata."
        )
    rows[17] = "Record 017: owner=Iris; shard=amber; retry_window=37; status=active; checksum=A7C9; note=priority handoff after audit."
    rows[88] = "Record 088: owner=Noor; shard=violet; retry_window=44; status=paused; checksum=VIOLET-88; note=tenant isolation review."
    rows[119] = "Record 119: owner=Uma; shard=copper; retry_window=28; status=active; checksum=U119Z; note=never retry after terminal state."
    rows[139] = "Record 139: owner=Rafi; shard=drift; retry_window=51; status=active; checksum=DRIFT-139; note=last-event cursor is sequence-based."
    return "LOCAL OPERATIONS LEDGER\n" + "\n".join(rows)


def retrieval_cases() -> list[dict[str, Any]]:
    context = ledger_context()
    return [
        case(
            "retrieval-01",
            "retrieval",
            f"{context}\n\nRetrieve Record 017. State its owner, shard, retry window, and checksum exactly.",
            max_tokens=256,
            required_groups=[["iris"], ["amber"], ["37"], ["a7c9"]],
            teacher_target="Record 017 is owned by Iris, on shard amber, with retry window 37 and checksum A7C9.",
        ),
        case(
            "retrieval-02",
            "retrieval",
            f"{context}\n\nFor Record 088, give the exact shard and status, then state what its note concerns.",
            max_tokens=256,
            required_groups=[["violet"], ["paused"], ["tenant", "isolation"]],
        ),
        case(
            "retrieval-03",
            "retrieval",
            f"{context}\n\nRetrieve Record 119 and state its exact checksum and its note. Do not substitute a nearby record.",
            max_tokens=256,
            required_groups=[["u119z"], ["never retry", "retry"], ["terminal"]],
        ),
        case(
            "retrieval-04",
            "retrieval",
            f"{context}\n\nCompare Record 017 and Record 139. Give both shards, both checksums, and identify which record mentions the last-event cursor.",
            max_tokens=256,
            required_groups=[["amber"], ["drift"], ["a7c9"], ["drift-139"], ["last-event", "cursor"]],
        ),
    ]


def prose_cases() -> list[dict[str, Any]]:
    rubric = [
        "Preserves every supplied named fact.",
        "Avoids contradiction or invented critical location/object facts.",
        "Follows the requested tone and sentence/line shape.",
    ]
    return [
        case(
            "prose-01",
            "prose",
            "Continue a calm scene in exactly three sentences. Mira and Sol are in the observatory, and the brass key is on the east sill. Do not add a weapon or change the key's location.",
            score_kind="semi-objective",
            max_tokens=256,
            required_groups=[["mira"], ["sol"], ["observatory"], ["brass key", "key"], ["east sill"]],
            forbidden_terms=["weapon", "sword", "gun", "west sill"],
            teacher_target="Mira and Sol stayed in the observatory, where the brass key rested on the east sill. The calm room held their attention as they planned the next step. Sol checked the window while Mira kept the key in sight.",
            rubric=rubric,
        ),
        case(
            "prose-02",
            "prose",
            "Summarize this continuity state in two sentences: Niko promised to return the blue map to Tessa before dusk, and the bridge is closed. Do not invent a destination or change the color.",
            score_kind="semi-objective",
            max_tokens=256,
            required_groups=[["niko"], ["tessa"], ["blue map", "map"], ["bridge"], ["closed"]],
            forbidden_terms=["red map", "castle", "mountain"],
            rubric=rubric,
        ),
        case(
            "prose-03",
            "prose",
            "Write a four-line dialogue between Eda and Jun. Eda knows the garden bell is broken; Jun proposes waiting by the east gate. Preserve both facts and do not add a new destination.",
            score_kind="semi-objective",
            max_tokens=256,
            required_groups=[["eda"], ["jun"], ["garden bell", "bell"], ["broken"], ["east gate"]],
            forbidden_terms=["west gate", "north gate", "castle"],
            rubric=rubric,
        ),
        case(
            "prose-04",
            "prose",
            "Give a concise continuity-safe recap in two sentences: Orin carries a sealed letter, Vale distrusts the river road, and the next scene begins at noon. Keep all three facts.",
            score_kind="semi-objective",
            max_tokens=256,
            required_groups=[["orin"], ["sealed letter", "letter"], ["vale"], ["river road"], ["noon"]],
            forbidden_terms=["trusts the river", "morning", "midnight"],
            rubric=rubric,
        ),
    ]


def calibration_cases() -> list[dict[str, Any]]:
    return [
        case(
            "cal-coding-01",
            "coding",
            "Two requests with the same idempotency key race. Name the database uniqueness and atomicity requirements in a short answer.",
            max_tokens=256,
            required_groups=[["unique"], ["idempotency"], ["atomic", "transaction"]],
        ),
        case(
            "cal-coding-02",
            "coding",
            "A stale running event follows succeeded. State the terminal-state invariant and the action for the stale event.",
            max_tokens=256,
            required_groups=[["terminal"], ["sequence", "version"], ["ignore", "reject", "discard"]],
        ),
        case(
            "cal-instruction-01",
            "instruction",
            "Output exactly two lines: FIRST: alpha and SECOND: beta. Do not add anything else.",
            max_tokens=96,
            line_prefixes=["FIRST:", "SECOND:"],
            exact_lines=["FIRST: alpha", "SECOND: beta"],
            max_chars=50,
        ),
        case(
            "cal-instruction-02",
            "instruction",
            "Answer with exactly one word: READY. No punctuation.",
            max_tokens=64,
            exact_lines=["READY"],
            max_chars=20,
        ),
        case(
            "cal-structured-01",
            "structured",
            'Return JSON only: {"ok": true, "count": 2}. Use exactly those keys and types.',
            max_tokens=128,
            structured={"keys": ["ok", "count"], "types": {"ok": "bool", "count": "int"}, "values": {"ok": True, "count": 2}},
        ),
        case(
            "cal-structured-02",
            "structured",
            'Return JSON only with exactly "name" as "Ada" and "tags" as ["a", "b"].',
            max_tokens=128,
            structured={"keys": ["name", "tags"], "types": {"name": "str", "tags": "list"}, "values": {"name": "Ada", "tags": ["a", "b"]}},
        ),
        case(
            "cal-reasoning-01",
            "reasoning",
            "A reducer applies 3 then ignores 2 because it is stale. What final version remains? End with `Final: <integer>`.",
            max_tokens=160,
            answer_regex=r"final\s*:\s*3\b",
        ),
        case(
            "cal-reasoning-02",
            "reasoning",
            "A worker has 5 tasks and completes 2. How many remain? End with `Final: <integer>`.",
            max_tokens=160,
            answer_regex=r"final\s*:\s*3\b",
        ),
        case(
            "cal-retrieval-01",
            "retrieval",
            "LOCAL NOTE: owner=Iris; code=A7C9; status=active. What are the owner and code?",
            max_tokens=128,
            required_groups=[["iris"], ["a7c9"]],
        ),
        case(
            "cal-retrieval-02",
            "retrieval",
            "EARLY FACT: shard=amber. LATE FACT: checksum=U149Z. State both exact values.",
            max_tokens=128,
            required_groups=[["amber"], ["u149z"]],
        ),
        case(
            "cal-prose-01",
            "prose",
            "In two sentences, preserve these facts: Lysa repaired the lantern; its glass is blue; Pera waits in the west room.",
            score_kind="semi-objective",
            max_tokens=192,
            required_groups=[["lysa"], ["repaired"], ["blue"], ["pera"], ["west room"]],
            forbidden_terms=["red", "east room"],
            rubric=["Preserves supplied facts", "Does not invent a changed color or location"],
        ),
        case(
            "cal-prose-02",
            "prose",
            "Write one calm sentence containing Mira, the observatory, and the brass key.",
            score_kind="semi-objective",
            max_tokens=128,
            required_groups=[["mira"], ["observatory"], ["brass key", "key"]],
            rubric=["Contains named facts", "Calm continuity"],
        ),
    ]


def all_cases() -> list[dict[str, Any]]:
    cases = calibration_cases() + coding_cases() + instruction_cases() + structured_cases() + reasoning_cases() + retrieval_cases() + prose_cases()
    for item in cases:
        if item["category"] == "coding":
            item["prompt"] += "\nKeep the answer concise (at most 220 words) and finish all requested checks."
            item["max_tokens"] = max(item["max_tokens"], 512)
        elif item["category"] == "reasoning":
            item["prompt"] += "\nBe concise (at most 100 words) and include the requested final answer marker."
            item["max_tokens"] = max(item["max_tokens"], 384)
    ids = [item["case_id"] for item in cases]
    if len(cases) != len(set(ids)):
        raise RuntimeError("duplicate Quality Suite V2 case id")
    return cases


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")
    return sha256_bytes(payload.encode("utf-8"))


def main() -> None:
    cases = all_cases()
    calibration = [item for item in cases if item["case_id"].startswith("cal-")]
    evaluation = [item for item in cases if not item["case_id"].startswith("cal-")]
    cal_path = OUT / "calibration" / "calibration_v2.jsonl"
    eval_path = OUT / "evaluation" / "evaluation_v2.jsonl"
    cal_hash = write_jsonl(cal_path, calibration)
    eval_hash = write_jsonl(eval_path, evaluation)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = None
    manifest = {
        "suite": "Qwen3.8 Quality Suite V2",
        "version": 2,
        "source": "research/stage4/harness/quality_suite_v2.py",
        "local_only": True,
        "calibration_count": len(calibration),
        "evaluation_count": len(evaluation),
        "total_count": len(cases),
        "categories": {category: sum(item["category"] == category for item in cases) for category in sorted({item["category"] for item in cases})},
        "calibration_sha256": cal_hash,
        "evaluation_sha256": eval_hash,
        "git_commit_at_freeze": commit,
        "calibration_case_ids": [item["case_id"] for item in calibration],
        "evaluation_case_ids": [item["case_id"] for item in evaluation],
        "teacher_forced_case_ids": [item["case_id"] for item in evaluation if item.get("teacher_target")],
        "output_budgets": {category: sorted({item["max_tokens"] for item in cases if item["category"] == category}) for category in sorted({item["category"] for item in cases})},
    }
    (OUT / "evaluation" / "suite_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
