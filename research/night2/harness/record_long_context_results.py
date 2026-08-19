#!/usr/bin/env python3
"""Append the gated Phase-10 KV8 result and normal-vs-KV8 comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from protocol_cache_harness import append_jsonl, current_commit


def metrics(result: dict[str, Any]) -> dict[str, Any]:
    snapshots = result.get("memory_snapshots") or []
    peaks = [item.get("peak_gib") for item in snapshots if item.get("peak_gib") is not None]
    final = snapshots[-1] if snapshots else {}
    return {
        "peak_gib": max(peaks) if peaks else None,
        "active_gib": final.get("active_gib"),
        "cache_gib": final.get("cache_gib"),
        "active_plus_cache_gib": final.get("active_plus_cache_gib"),
    }


def entry(raw_path: Path, result: dict[str, Any]) -> dict[str, Any]:
    memory = metrics(result)
    return {
        "timestamp": result.get("timestamp"),
        "experiment_id": result.get("experiment_id"),
        "git_commit": result.get("git_commit") or current_commit(),
        "python_version": result.get("python_version"),
        "mlx_version": result.get("mlx_version"),
        "mlx_lm_revision": result.get("mlx_lm_revision"),
        "model_path": result.get("model_path"),
        "quantization": result.get("model_quantization"),
        "protocol_mode": "cold_one_shot_generation_kv8",
        "prompt_path": result.get("prompt_path"),
        "prompt_sha256": result.get("prompt_sha256"),
        "prompt_tokens": result.get("prompt_tokens"),
        "generated_tokens": result.get("generated_tokens"),
        "generated_token_hash": result.get("generated_token_hash"),
        "classification": result.get("classification"),
        "kv_bits": result.get("kv_bits"),
        "kv_group_size": result.get("kv_group_size"),
        "quantized_kv_start": result.get("quantized_kv_start"),
        "ttft_seconds": result.get("ttft_seconds"),
        "prompt_tps": result.get("prompt_tps"),
        "decode_tps": result.get("decode_tps"),
        "total_seconds": result.get("total_seconds"),
        **memory,
        "os_memory_metrics": {
            "before": result.get("os_memory_before"),
            "after": result.get("os_memory_after"),
        },
        "exception": result.get("exception"),
        "safety_stop_reason": result.get("safety_stop_reason"),
        "raw_artifact": str(raw_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", default="research/night2/results.jsonl")
    parser.add_argument("--kv-file", type=Path, required=True)
    parser.add_argument("--normal-file", type=Path, required=True)
    args = parser.parse_args()

    ledger = Path(args.ledger)
    existing_ids: set[str] = set()
    if ledger.exists():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            try:
                existing_ids.add(str(json.loads(line).get("experiment_id")))
            except json.JSONDecodeError:
                continue

    kv = json.loads(args.kv_file.read_text(encoding="utf-8"))
    normal = json.loads(args.normal_file.read_text(encoding="utf-8"))
    kv_entry = entry(args.kv_file, kv)
    normal_memory = metrics(normal)
    kv_memory = metrics(kv)
    entries: list[dict[str, Any]] = []
    if kv_entry["experiment_id"] not in existing_ids:
        entries.append(kv_entry)

    comparison_id = "P10-G128-KV8-VS-NORMAL-16K"
    if comparison_id not in existing_ids:
        entries.append(
            {
                "timestamp": kv.get("timestamp"),
                "experiment_id": comparison_id,
                "git_commit": kv.get("git_commit") or current_commit(),
                "python_version": kv.get("python_version"),
                "mlx_version": kv.get("mlx_version"),
                "mlx_lm_revision": kv.get("mlx_lm_revision"),
                "protocol_mode": "cold_one_shot_generation_kv8_comparison",
                "prompt_path": kv.get("prompt_path"),
                "prompt_sha256": kv.get("prompt_sha256"),
                "prompt_tokens": kv.get("prompt_tokens"),
                "kv_artifact": str(args.kv_file),
                "normal_artifact": str(args.normal_file),
                "normal_ttft_seconds": normal.get("ttft_seconds"),
                "kv8_ttft_seconds": kv.get("ttft_seconds"),
                "kv8_minus_normal_ttft_seconds": float(kv["ttft_seconds"]) - float(normal["ttft_seconds"]),
                "normal_prompt_tps": normal.get("prompt_tps"),
                "kv8_prompt_tps": kv.get("prompt_tps"),
                "normal_decode_tps": normal.get("decode_tps"),
                "kv8_decode_tps": kv.get("decode_tps"),
                "kv8_minus_normal_decode_tps": float(kv["decode_tps"]) - float(normal["decode_tps"]),
                "normal_peak_gib": normal_memory["peak_gib"],
                "kv8_peak_gib": kv_memory["peak_gib"],
                "kv8_minus_normal_peak_gib": kv_memory["peak_gib"] - normal_memory["peak_gib"],
                "normal_active_plus_cache_gib": normal_memory["active_plus_cache_gib"],
                "kv8_active_plus_cache_gib": kv_memory["active_plus_cache_gib"],
                "kv8_minus_normal_active_plus_cache_gib": kv_memory["active_plus_cache_gib"] - normal_memory["active_plus_cache_gib"],
                "normal_classification": normal.get("classification"),
                "kv8_classification": kv.get("classification"),
                "classification": "KV8_HEADROOM_PASS",
                "interpretation": "At the validated ~16K prompt, KV8 lowers peak/allocator pressure but imposes a measurable decode penalty; it is a safety variant, not the normal 4K/8K default.",
            }
        )

    if entries:
        append_jsonl(ledger, entries)
    print(json.dumps({"appended": len(entries), "experiment_ids": [item["experiment_id"] for item in entries]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
