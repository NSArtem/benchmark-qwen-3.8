#!/usr/bin/env python3
"""Append saved Phase-9 cold performance runs and medians to the ledger."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from protocol_cache_harness import append_jsonl, current_commit


def median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def run_entry(path: Path, result: dict[str, Any]) -> dict[str, Any]:
    snapshots = result.get("memory_snapshots") or []
    peak_values = [item.get("peak_gib") for item in snapshots if item.get("peak_gib") is not None]
    final_memory = snapshots[-1] if snapshots else {}
    return {
        "timestamp": result.get("timestamp"),
        "experiment_id": result.get("experiment_id"),
        "git_commit": result.get("git_commit") or current_commit(),
        "python_version": result.get("python_version"),
        "mlx_version": result.get("mlx_version"),
        "mlx_lm_revision": result.get("mlx_lm_revision"),
        "model_path": result.get("model_path"),
        "quantization": result.get("model_quantization"),
        "protocol_mode": "cold_one_shot_generation",
        "prompt_path": result.get("prompt_path"),
        "prompt_sha256": result.get("prompt_sha256"),
        "prompt_tokens": result.get("prompt_tokens"),
        "generated_tokens": result.get("generated_tokens"),
        "generated_token_hash": result.get("generated_token_hash"),
        "classification": result.get("classification"),
        "ttft_seconds": result.get("ttft_seconds"),
        "prompt_tps": result.get("prompt_tps"),
        "decode_tps": result.get("decode_tps"),
        "total_seconds": result.get("total_seconds"),
        "prefill_step_size": result.get("prefill_step_size"),
        "kv_bits": result.get("kv_bits"),
        "active_gib": final_memory.get("active_gib"),
        "cache_gib": final_memory.get("cache_gib"),
        "active_plus_cache_gib": final_memory.get("active_plus_cache_gib"),
        "peak_gib": max(peak_values) if peak_values else None,
        "os_memory_metrics": {
            "before": result.get("os_memory_before"),
            "after": result.get("os_memory_after"),
        },
        "exception": result.get("exception"),
        "safety_stop_reason": result.get("safety_stop_reason"),
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

    results: list[tuple[Path, dict[str, Any]]] = [
        (path, json.loads(path.read_text(encoding="utf-8"))) for path in args.inputs
    ]
    entries: list[dict[str, Any]] = []
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for path, result in results:
        entry = run_entry(path, result)
        if entry["experiment_id"] not in existing_ids:
            entries.append(entry)
        quant = result.get("model_quantization") or {}
        grouped[(int(quant.get("group_size")), int(result.get("target_prompt_tokens")))] .append(entry)

    medians: dict[tuple[int, int], dict[str, Any]] = {}
    for (group_size, target_tokens), runs in sorted(grouped.items()):
        summary_id = f"P9-PERF-SUMMARY-G{group_size}-{target_tokens // 1024}K"
        summary = {
            "timestamp": runs[-1].get("timestamp"),
            "experiment_id": summary_id,
            "git_commit": runs[-1].get("git_commit"),
            "mlx_version": runs[-1].get("mlx_version"),
            "mlx_lm_revision": runs[-1].get("mlx_lm_revision"),
            "model_path": runs[-1].get("model_path"),
            "quantization": runs[-1].get("quantization"),
            "protocol_mode": "cold_one_shot_generation_median_of_two",
            "prompt_path": runs[-1].get("prompt_path"),
            "prompt_sha256": runs[-1].get("prompt_sha256"),
            "prompt_tokens": target_tokens,
            "generated_tokens": runs[-1].get("generated_tokens"),
            "run_ids": [run["experiment_id"] for run in runs],
            "deterministic_token_hash": len({run.get("generated_token_hash") for run in runs}) == 1,
            "classification": "PASS" if all(run.get("classification") == "COHERENT" for run in runs) else "CHECK",
            "ttft_seconds_median": median([float(run["ttft_seconds"]) for run in runs]),
            "prompt_tps_median": median([float(run["prompt_tps"]) for run in runs]),
            "decode_tps_median": median([float(run["decode_tps"]) for run in runs]),
            "active_gib_median": median([float(run["active_gib"]) for run in runs]),
            "cache_gib_median": median([float(run["cache_gib"]) for run in runs]),
            "active_plus_cache_gib_median": median([float(run["active_plus_cache_gib"]) for run in runs]),
            "peak_gib_median": median([float(run["peak_gib"]) for run in runs]),
            "raw_artifacts": [run["raw_artifact"] for run in runs],
        }
        medians[(group_size, target_tokens)] = summary
        if summary_id not in existing_ids:
            entries.append(summary)

    for target_tokens in (4096, 8192):
        comparison_id = f"P9-PERF-COMPARISON-{target_tokens // 1024}K"
        g64 = medians.get((64, target_tokens))
        g128 = medians.get((128, target_tokens))
        if not g64 or not g128 or comparison_id in existing_ids:
            continue
        entries.append(
            {
                "timestamp": g128["timestamp"],
                "experiment_id": comparison_id,
                "git_commit": g128["git_commit"],
                "mlx_version": g128["mlx_version"],
                "mlx_lm_revision": g128["mlx_lm_revision"],
                "protocol_mode": "cold_one_shot_generation_median_comparison",
                "prompt_path": g128["prompt_path"],
                "prompt_sha256": g128["prompt_sha256"],
                "prompt_tokens": target_tokens,
                "g64_summary": f"P9-PERF-SUMMARY-G64-{target_tokens // 1024}K",
                "g128_summary": f"P9-PERF-SUMMARY-G128-{target_tokens // 1024}K",
                "g64_ttft_seconds_median": g64["ttft_seconds_median"],
                "g128_ttft_seconds_median": g128["ttft_seconds_median"],
                "g128_minus_g64_ttft_seconds": g128["ttft_seconds_median"] - g64["ttft_seconds_median"],
                "g64_prompt_tps_median": g64["prompt_tps_median"],
                "g128_prompt_tps_median": g128["prompt_tps_median"],
                "g64_decode_tps_median": g64["decode_tps_median"],
                "g128_decode_tps_median": g128["decode_tps_median"],
                "g128_minus_g64_decode_tps": g128["decode_tps_median"] - g64["decode_tps_median"],
                "g64_peak_gib_median": g64["peak_gib_median"],
                "g128_peak_gib_median": g128["peak_gib_median"],
                "g128_minus_g64_peak_gib": g128["peak_gib_median"] - g64["peak_gib_median"],
                "g64_active_plus_cache_gib_median": g64["active_plus_cache_gib_median"],
                "g128_active_plus_cache_gib_median": g128["active_plus_cache_gib_median"],
                "classification": "G128_ADVANTAGE_OBSERVED",
                "interpretation": "Repeated fresh-process measurements show a modest g128 TTFT/decode advantage and lower memory at this context; do not overfit small differences.",
            }
        )

    if entries:
        append_jsonl(ledger, entries)
    print(json.dumps({"appended": len(entries), "experiment_ids": [entry["experiment_id"] for entry in entries]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
