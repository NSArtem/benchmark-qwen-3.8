#!/usr/bin/env python3
"""Freeze targeted Stage 6 held-out budget corrections.

Stage 5's raw outputs show two kinds of truncation artifact: verbose coding
answers stopping at the 384-token cap, and arithmetic/state answers stopping
at the 256-token cap before the required ``Final: ...`` marker.  This script
changes only those generation budgets, keeps the v1 scorer and prompts
byte-for-byte otherwise, and records the correction plus both suite hashes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_V1 = ROOT / "research/stage5/heldout/heldout_v1.jsonl"
DEFAULT_OUT = ROOT / "research/stage6/heldout/heldout_v2.jsonl"
DEFAULT_MANIFEST = ROOT / "research/stage6/heldout/heldout_v2_manifest.json"

CORRECTIONS = {
    **{
        f"h5-coding-{index:02d}": {
            "old_max_tokens": 384,
            "new_max_tokens": 1024,
            "old_rule": "verbose coding answer had a 384-token generation budget",
            "reason_invalid": "Stage 5 raw outputs for this coding family included length-finished answers; affected outputs must not be treated as clean semantic failures",
            "new_rule": "same prompt, scorer, and semantic answer; only max_tokens is 1024",
        }
        for index in range(1, 7)
    },
    **{
        f"h5-reasoning-{index:02d}": {
            "old_max_tokens": 256,
            "new_max_tokens": 1024,
            "old_rule": "arithmetic/state answer had a 256-token generation budget",
            "reason_invalid": "Stage 5 raw outputs included length-finished reasoning answers before the required final sentinel",
            "new_rule": "same prompt, answer_regex, and expected answer; only max_tokens is 1024",
        }
        for index in (1, 2, 4)
    },
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def canonical_payload(items: list[dict]) -> str:
    return "".join(json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for item in items)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_V1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    original = read_jsonl(args.input)
    corrected = copy.deepcopy(original)
    by_id = {item["case_id"]: item for item in corrected}
    if set(by_id) != {item["case_id"] for item in original} or len(corrected) != 24:
        raise AssertionError("unexpected heldout_v1 shape")

    for case_id, correction in CORRECTIONS.items():
        item = by_id[case_id]
        if item["max_tokens"] != correction["old_max_tokens"]:
            raise AssertionError(f"{case_id}: unexpected old budget {item['max_tokens']}")
        item["max_tokens"] = correction["new_max_tokens"]

    original_payload = args.input.read_text(encoding="utf-8")
    corrected_payload = canonical_payload(corrected)
    original_hash = hashlib.sha256(original_payload.encode("utf-8")).hexdigest()
    corrected_hash = hashlib.sha256(corrected_payload.encode("utf-8")).hexdigest()
    if original_hash != "6fe7bd7da6732e9057a75764292b18b29c3262a96eb39391b700f038f641851b":
        raise AssertionError(f"unexpected heldout_v1 hash: {original_hash}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(corrected_payload, encoding="utf-8")
    manifest = {
        "suite": "Qwen3.8 Stage 6 corrected held-out quality suite",
        "version": 2,
        "source_suite": str(args.input),
        "source_sha256": original_hash,
        "suite_path": str(args.output),
        "suite_sha256": corrected_hash,
        "case_count": len(corrected),
        "semantic_answers_changed": False,
        "scorer_changed": False,
        "corrections": CORRECTIONS,
        "unchanged_case_count": len(corrected) - len(CORRECTIONS),
        "selection_policy": "final-selection evidence only; not used for calibration, sensitivity, or allocation",
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
