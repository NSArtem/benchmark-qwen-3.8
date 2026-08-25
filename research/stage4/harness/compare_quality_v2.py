#!/usr/bin/env python3
"""Compare the preserved g64/g128 Quality Suite V2 runs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def lcp(left: list[int], right: list[int]) -> int:
    length = 0
    for a, b in zip(left, right):
        if a != b:
            break
        length += 1
    return length


def model_summary(label: str, final_root: Path, raw_root: Path, teacher_path: Path) -> dict[str, Any]:
    final = read(final_root / "model.json")
    raw = read(raw_root / "model.json")
    teacher = read(teacher_path)
    cases = final["cases"]
    by_category: dict[str, dict[str, Any]] = {}
    for category in sorted({case["category"] for case in cases}):
        subset = [case for case in cases if case["category"] == category]
        objective = [case for case in subset if case["checks"].get("objective_pass") is not None]
        semi = [case for case in subset if case["checks"].get("semi_objective_pass") is not None]
        by_category[category] = {
            "cases": len(subset),
            "objective_pass": sum(bool(case["checks"].get("objective_pass")) for case in objective),
            "objective_total": len(objective),
            "semi_objective_pass": sum(bool(case["checks"].get("semi_objective_pass")) for case in semi),
            "semi_objective_total": len(semi),
        }
    lcp_rows = []
    for case in cases:
        case_id = case["case_id"]
        token_path = raw_root / "tokens" / f"{case_id}.json"
        token_ids = [int(value) for value in read(token_path)["token_ids"]]
        lcp_rows.append({"case_id": case_id, "generated_tokens": len(token_ids), "token_ids": token_ids})
    return {
        "label": label,
        "model": final.get("model"),
        "suite_sha256": final.get("suite_sha256"),
        "objective_pass": sum(bool(case["checks"].get("objective_pass")) for case in cases),
        "objective_total": sum(case["checks"].get("objective_pass") is not None for case in cases),
        "semi_objective_pass": sum(bool(case["checks"].get("semi_objective_pass")) for case in cases),
        "semi_objective_total": sum(case["checks"].get("semi_objective_pass") is not None for case in cases),
        "objective_failures": [case["case_id"] for case in cases if case["checks"].get("objective_pass") is False],
        "category_summary": by_category,
        "finish_reasons": {reason: sum(case.get("finish_reason") == reason for case in cases) for reason in sorted({case.get("finish_reason") for case in cases})},
        "early_eos_count": sum(bool(case.get("early_eos")) for case in cases),
        "generated_tokens_mean": statistics.mean(case.get("generated_tokens", 0) for case in cases),
        "decode_tps_mean": statistics.mean(case.get("decode_tps", 0.0) for case in cases),
        "decode_tps_median": statistics.median(case.get("decode_tps", 0.0) for case in cases),
        "peak_gib_max": max((case.get("memory_after_generation") or {}).get("peak_gib", 0.0) for case in cases),
        "raw_total_elapsed_seconds": raw.get("total_elapsed_seconds"),
        "teacher_forced": {
            "case_count": len(teacher.get("cases", [])),
            "mean_nll_nats": teacher.get("mean_nll_nats"),
            "weighted_mean_nll_nats": teacher.get("weighted_mean_nll_nats"),
            "mean_top1_accuracy": teacher.get("mean_top1_accuracy"),
            "weighted_top1_accuracy": teacher.get("weighted_top1_accuracy"),
            "total_target_tokens": teacher.get("total_target_tokens"),
        },
        "_lcp_rows": lcp_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g64-root", type=Path, required=True)
    parser.add_argument("--g128-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    g64 = model_summary("g64", args.g64_root / "final", args.g64_root, args.g64_root / "teacher-forced-v2.json")
    g128 = model_summary("g128", args.g128_root / "final", args.g128_root, args.g128_root / "teacher-forced-v2.json")
    rows = []
    for left, right in zip(g64.pop("_lcp_rows"), g128.pop("_lcp_rows")):
        rows.append({"case_id": left["case_id"], "g64_generated_tokens": left["generated_tokens"], "g128_generated_tokens": right["generated_tokens"], "token_lcp": lcp(left["token_ids"], right["token_ids"]), "token_equal": left["token_ids"] == right["token_ids"]})
    result = {
        "experiment_id": "P1-G64-VS-G128-QUALITY-V2",
        "phase": 1,
        "suite_sha256": g64["suite_sha256"],
        "g64": g64,
        "g128": g128,
        "case_alignment": rows,
        "token_equal_count": sum(row["token_equal"] for row in rows),
        "mean_token_lcp": statistics.mean(row["token_lcp"] for row in rows),
        "objective_rate_delta_g128_minus_g64": g128["objective_pass"] / g128["objective_total"] - g64["objective_pass"] / g64["objective_total"],
        "teacher_forced_weighted_nll_delta_g128_minus_g64": g128["teacher_forced"]["weighted_mean_nll_nats"] - g64["teacher_forced"]["weighted_mean_nll_nats"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
