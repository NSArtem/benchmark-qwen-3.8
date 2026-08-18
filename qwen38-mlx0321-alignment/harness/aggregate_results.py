#!/usr/bin/env python3
"""Finalize and aggregate the completed fresh-process alignment runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


TARGETS = (512, 513, 640, 641, 768, 769)
REPEATS = (1, 2, 3)


def gib(value: int | None) -> float | None:
    return value / (1024**3) if value is not None else None


def snapshot(result: dict[str, Any], label: str) -> dict[str, Any] | None:
    return next((item for item in result["memory_snapshots"] if item.get("label") == label), None)


def memory_view(result: dict[str, Any], label: str) -> dict[str, Any] | None:
    item = snapshot(result, label)
    if item is None:
        return None
    return {
        "active_bytes": item.get("active_bytes"),
        "cache_bytes": item.get("cache_bytes"),
        "active_plus_cache_bytes": item.get("active_plus_cache_bytes"),
        "peak_bytes": item.get("peak_bytes"),
        "active_gib": item.get("active_gib"),
        "cache_gib": item.get("cache_gib"),
        "active_plus_cache_gib": item.get("active_plus_cache_gib"),
        "peak_gib": item.get("peak_gib"),
    }


def determinism(items: list[dict[str, Any]]) -> dict[str, Any]:
    first = [item.get("first_token_id") for item in items]
    first16 = [tuple(item.get("first_16_token_ids") or []) for item in items]
    full = [item.get("generated_token_ids_sha256") for item in items]
    classes = [item.get("classification") for item in items]
    return {
        "same_first_token": len(set(first)) == 1,
        "same_first_16_tokens": len(set(first16)) == 1,
        "same_complete_token_sequence": len(set(full)) == 1,
        "same_classification": len(set(classes)) == 1,
        "first_token_ids": first,
        "first_16_token_ids": [list(value) for value in first16],
        "generated_token_id_hashes": full,
        "classifications": classes,
    }


def pair_summary(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    left_d = determinism(left)
    right_d = determinism(right)
    return {
        "unaligned": {
            "final_prompt_tokens": left[0]["actual_prompt_tokens"],
            "main_prefill_tokens": left[0]["prefill_tokens"],
            "classification_counts": dict(Counter(item["classification"] for item in left)),
            "determinism": left_d,
            "memory_run1": {
                "after_prefill_first_token": memory_view(left[0], "after_prefill_and_first_generated_token"),
                "after_completion": memory_view(left[0], "after_completion"),
                "peak_bytes": left[0].get("peak_memory"),
                "peak_gib": gib(left[0].get("peak_memory")),
            },
        },
        "aligned": {
            "final_prompt_tokens": right[0]["actual_prompt_tokens"],
            "main_prefill_tokens": right[0]["prefill_tokens"],
            "classification_counts": dict(Counter(item["classification"] for item in right)),
            "determinism": right_d,
            "memory_run1": {
                "after_prefill_first_token": memory_view(right[0], "after_prefill_and_first_generated_token"),
                "after_completion": memory_view(right[0], "after_completion"),
                "peak_bytes": right[0].get("peak_memory"),
                "peak_gib": gib(right[0].get("peak_memory")),
            },
        },
        "same_first_token_across_pair": len({item["first_token_id"] for item in left + right}) == 1,
        "same_first_16_across_pair": len({tuple(item["first_16_token_ids"]) for item in left + right}) == 1,
        "same_complete_sequence_across_pair": len({item["generated_token_ids_sha256"] for item in left + right}) == 1,
        "peak_delta_aligned_minus_unaligned_gib": gib(right[0].get("peak_memory")) - gib(left[0].get("peak_memory")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="qwen38-mlx0321-alignment")
    parser.add_argument("--classification", default="COHERENT", choices=("COHERENT", "DEGRADED", "BROKEN"))
    args = parser.parse_args()
    root = Path(args.root)
    log_dir = root / "logs"
    tests: list[dict[str, Any]] = []

    for target in TARGETS:
        for repeat in REPEATS:
            path = log_dir / f"{target}-r{repeat}.json"
            if not path.exists():
                raise FileNotFoundError(path)
            item = json.loads(path.read_text(encoding="utf-8"))
            if item.get("result") != "PASS" or item.get("generated_tokens") != 128:
                raise RuntimeError(f"run did not complete 128 tokens: {path}")
            item["classification"] = args.classification
            path.write_text(json.dumps(item, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            tests.append(item)

    by_target = {target: [item for item in tests if item["actual_prompt_tokens"] == target] for target in TARGETS}
    length_summaries = []
    for target in TARGETS:
        items = by_target[target]
        d = determinism(items)
        length_summaries.append(
            {
                "final_prompt_tokens": target,
                "main_prefill_tokens": target - 1,
                "run_count": len(items),
                "classification_counts": dict(Counter(item["classification"] for item in items)),
                "determinism": d,
                "peak_memory_gib": [gib(item.get("peak_memory")) for item in items],
                "after_prefill_memory_gib": [memory_view(item, "after_prefill_and_first_generated_token") for item in items],
                "after_completion_memory_gib": [memory_view(item, "after_completion") for item in items],
                "generation_tps": [item.get("generation_tps") for item in items],
            }
        )

    pairs = {
        "512_vs_513": pair_summary(by_target[512], by_target[513]),
        "640_vs_641": pair_summary(by_target[640], by_target[641]),
        "768_vs_769": pair_summary(by_target[768], by_target[769]),
    }

    manifest = json.loads((root / "prompt-manifest.json").read_text(encoding="utf-8"))
    aggregate = {
        "experiment": "Qwen3.8 MLX 0.32.1 prompt-prefill alignment",
        "environment": "envs/mlx-0.32.1",
        "model": "lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly",
        "mlx_version": "0.32.1",
        "mlx_lm_git_commit": "254d153fdeb6f150edd4fc5a54f9828638481fa8",
        "prefill_step_size": 2048,
        "max_tokens": 128,
        "wired_limit_policy": "untouched; direct generate_step default state",
        "cache_policy": "default; no cache-limit modification",
        "prompt_manifest_sha256": hashlib.sha256((root / "prompt-manifest.json").read_bytes()).hexdigest(),
        "prompt_manifest": manifest,
        "tests": tests,
        "length_summaries": length_summaries,
        "adjacent_pairs": pairs,
        "run_count": len(tests),
        "classification_counts_all": dict(Counter(item["classification"] for item in tests)),
        "oom_count": sum(item.get("result") == "OOM" for item in tests),
    }
    (root / "results.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"run_count": len(tests), "classification_counts": aggregate["classification_counts_all"], "oom_count": aggregate["oom_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
