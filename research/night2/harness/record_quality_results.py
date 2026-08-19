#!/usr/bin/env python3
"""Append saved Phase-8 quality measurements to the Night-2 JSONL ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from protocol_cache_harness import append_jsonl, current_commit


def quality_entry(path: Path, result: dict[str, Any]) -> dict[str, Any]:
    model = result.get("model") or {}
    source = result.get("source_token_record") or {}
    memory = result.get("memory_after_eval") or result.get("memory_after_release") or {}
    exception = result.get("exception")
    return {
        "timestamp": result.get("timestamp"),
        "experiment_id": f"P9-QUALITY-{path.stem}",
        "git_commit": result.get("git_commit") or current_commit(),
        "python_version": model.get("python_version"),
        "mlx_version": model.get("mlx_version"),
        "mlx_lm_revision": result.get("mlx_lm_revision"),
        "model_path": result.get("model_path"),
        "quantization": model.get("quantization"),
        "protocol_mode": result.get("protocol_mode"),
        "quality_metric": "teacher_forced_average_nll",
        "prompt_path": source.get("prompt_file"),
        "prompt_sha256": source.get("prompt_sha256"),
        "prompt_tokens": source.get("actual_prompt_tokens"),
        "token_offset": result.get("token_offset"),
        "tokens_used": result.get("tokens_used"),
        "evaluated_targets": result.get("evaluated_targets"),
        "token_sha256": result.get("token_sha256"),
        "average_nll_nats": result.get("average_nll_nats"),
        "perplexity": result.get("perplexity"),
        "average_target_logprob_nats": result.get("average_target_logprob_nats"),
        "top1_accuracy": result.get("top1_accuracy"),
        "active_gib": memory.get("active_gib"),
        "cache_gib": memory.get("cache_gib"),
        "active_plus_cache_gib": memory.get("active_plus_cache_gib"),
        "peak_gib": memory.get("peak_gib"),
        "os_memory_metrics": result.get("os_memory_after_eval"),
        "classification": "HARNESS_BUG" if exception else "QUALITY_MEASURED",
        "exception": exception,
        "raw_artifact": str(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", default="research/night2/results.jsonl")
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()

    ledger = Path(args.ledger)
    existing_ids: set[str] = set()
    if ledger.exists():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            try:
                existing_ids.add(str(json.loads(line).get("experiment_id")))
            except json.JSONDecodeError:
                continue

    entries: list[dict[str, Any]] = []
    valid_by_file: dict[str, dict[str, Any]] = {}
    for path in args.inputs:
        result = json.loads(path.read_text(encoding="utf-8"))
        entry = quality_entry(path, result)
        if entry["experiment_id"] not in existing_ids:
            entries.append(entry)
        if not result.get("exception"):
            valid_by_file[path.name] = result

    g64 = valid_by_file.get("P9-G64-nll-full4096-chunk256.json")
    g128 = valid_by_file.get("P9-G128-nll-full4096-chunk256.json")
    comparison_id = "P9-QUALITY-COMPARISON-FULL4096-CHUNK256"
    if g64 and g128 and comparison_id not in existing_ids:
        g64_nll = float(g64["average_nll_nats"])
        g128_nll = float(g128["average_nll_nats"])
        entries.append(
            {
                "timestamp": g128.get("timestamp"),
                "experiment_id": comparison_id,
                "git_commit": g128.get("git_commit") or current_commit(),
                "mlx_version": (g128.get("model") or {}).get("mlx_version"),
                "mlx_lm_revision": g128.get("mlx_lm_revision"),
                "quality_metric": "teacher_forced_average_nll",
                "corpus_prompt_path": (g128.get("source_token_record") or {}).get("prompt_file"),
                "corpus_prompt_sha256": (g128.get("source_token_record") or {}).get("prompt_sha256"),
                "corpus_tokens": g128.get("tokens_used"),
                "evaluated_targets": g128.get("evaluated_targets"),
                "g64_raw_artifact": "research/night2/quality/P9-G64-nll-full4096-chunk256.json",
                "g128_raw_artifact": "research/night2/quality/P9-G128-nll-full4096-chunk256.json",
                "g64_average_nll_nats": g64_nll,
                "g128_average_nll_nats": g128_nll,
                "g128_minus_g64_nll_nats": g128_nll - g64_nll,
                "g64_perplexity": g64["perplexity"],
                "g128_perplexity": g128["perplexity"],
                "g128_to_g64_perplexity_ratio": float(g128["perplexity"]) / float(g64["perplexity"]),
                "g64_top1_accuracy": g64["top1_accuracy"],
                "g128_top1_accuracy": g128["top1_accuracy"],
                "g128_minus_g64_top1_accuracy": float(g128["top1_accuracy"]) - float(g64["top1_accuracy"]),
                "classification": "MATERIAL_ON_MEASURED_CORPUS",
                "interpretation": "Content-dependent quality delta is material on the measured 4096-token coding prompt; not a universal quality claim.",
            }
        )

    if entries:
        append_jsonl(ledger, entries)
    print(json.dumps({"appended": len(entries), "experiment_ids": [e["experiment_id"] for e in entries]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
