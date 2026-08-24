#!/usr/bin/env python3
"""Compare normal and 8-bit attention KV on the validated 16K suffix chat.

The two configurations run in separate fresh child processes, sequentially,
with the same g128 model, prompt, follow-up questions, direct generate_step
path, and exact-suffix transcript protocol.  This keeps allocator history out
of the comparison while exercising real persistent chat in both modes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
SCALE_HARNESS = REPO_ROOT / "research/stage3/persistent-scaling/harness/run_scale.py"
MODEL = "/Users/artem/models/Qwen3.8-27B-3bit-g128-MLX"
PROMPT_PATH = REPO_ROOT / "qwen38-mtp-smoke/coding-prompt.txt"
PINNED_REVISION = "254d153fdeb6f150edd4fc5a54f9828638481fa8"
CONTEXT = "16k"
STOP_PEAK_GIB = 15.6


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_child(
    *,
    label: str,
    output_root: Path,
    kv_bits: int | None,
    model: str,
    stop_peak_gib: float,
) -> dict[str, Any]:
    child_root = output_root / label
    child_root.mkdir(parents=True, exist_ok=True)
    prefix = "P6-G128-KV8" if kv_bits is not None else "P6-G128-NORMAL-KV"
    command = [
        sys.executable,
        str(SCALE_HARNESS),
        "--mode",
        "persistent",
        "--context",
        CONTEXT,
        "--model",
        model,
        "--output-root",
        str(child_root),
        "--initial-max-tokens",
        "128",
        "--follow-up-max-tokens",
        "64",
        "--stop-peak-gib",
        str(stop_peak_gib),
        "--phase",
        "6",
        "--experiment-prefix",
        prefix,
        "--kv-group-size",
        "64",
        "--quantized-kv-start",
        "0",
    ]
    if kv_bits is not None:
        command.extend(["--kv-bits", str(kv_bits)])
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True)
    (child_root / "launcher.stdout.log").write_text(
        completed.stdout, encoding="utf-8"
    )
    (child_root / "launcher.stderr.log").write_text(
        completed.stderr, encoding="utf-8"
    )
    result_path = child_root / CONTEXT / "persistent.json"
    result: dict[str, Any] = {
        "label": label,
        "kv_bits": kv_bits,
        "kv_group_size": 64,
        "quantized_kv_start": 0,
        "command": command,
        "launcher_returncode": completed.returncode,
        "launcher_elapsed_seconds": time.perf_counter() - started,
        "raw_result_path": str(result_path),
        "stdout_path": str(child_root / "launcher.stdout.log"),
        "stderr_path": str(child_root / "launcher.stderr.log"),
    }
    if result_path.exists():
        result["persistent"] = json.loads(result_path.read_text(encoding="utf-8"))
    else:
        result["persistent"] = None
    return result


def turns_summary(persistent: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not persistent:
        return []
    rows = []
    for turn in persistent.get("turns", []):
        memory = turn.get("memory_after_generation", {})
        rows.append(
            {
                "turn": turn.get("turn"),
                "prompt_tokens": turn.get("prompt_token_count"),
                "suffix_tokens": turn.get("processed_suffix_tokens"),
                "generated_tokens": turn.get("generated_token_count"),
                "ttft_seconds": turn.get("ttft_seconds"),
                "decode_tps": turn.get("decode_tps"),
                "active_plus_cache_gib": memory.get("active_plus_cache_gib"),
                "peak_gib": memory.get("peak_gib"),
                "classification": turn.get("classification"),
                "replay_detected": turn.get("replay_detected"),
                "logical_equation_holds": turn.get("logical_equation_holds"),
                "cache_after_length_matches": turn.get("cache_after_length_matches"),
                "cache_offset_before": turn.get("cache_offset_before"),
                "cache_offset_after": turn.get("cache_offset_after"),
                "state_signature_after": turn.get("state_signature_after"),
            }
        )
    return rows


def numeric(values: list[Any]) -> list[float]:
    return [float(value) for value in values if value is not None]


def mode_summary(child: dict[str, Any]) -> dict[str, Any]:
    persistent = child.get("persistent") or {}
    turns = turns_summary(persistent)
    warm = turns[1:]
    peaks = numeric([item["peak_gib"] for item in turns])
    active_cache = numeric([item["active_plus_cache_gib"] for item in turns])
    warm_ttft = numeric([item["ttft_seconds"] for item in warm])
    warm_decode = numeric([item["decode_tps"] for item in warm])
    protocol_ok = all(
        item["logical_equation_holds"] and item["cache_after_length_matches"]
        for item in turns
    )
    coherent = all(item["classification"] == "COHERENT" for item in turns)
    no_replay = all(not item["replay_detected"] for item in turns)
    return {
        "label": child["label"],
        "kv_bits": child["kv_bits"],
        "kv_group_size": child["kv_group_size"],
        "quantized_kv_start": child["quantized_kv_start"],
        "launcher_returncode": child["launcher_returncode"],
        "overall_classification": persistent.get("overall_classification"),
        "safety_stop_reason": persistent.get("safety_stop_reason"),
        "exception": persistent.get("exception"),
        "turns_completed": len(turns),
        "final_logical_offset": persistent.get("total_logical_tokens_after_last_turn"),
        "cold_ttft_seconds": turns[0]["ttft_seconds"] if turns else None,
        "warm_ttft_median_seconds": median(warm_ttft) if warm_ttft else None,
        "warm_ttft_max_seconds": max(warm_ttft) if warm_ttft else None,
        "warm_decode_mean_tps": sum(warm_decode) / len(warm_decode)
        if warm_decode
        else None,
        "warm_decode_min_tps": min(warm_decode) if warm_decode else None,
        "peak_gib": max(peaks) if peaks else None,
        "active_plus_cache_peak_gib": max(active_cache) if active_cache else None,
        "all_turns_coherent": coherent,
        "no_replay": no_replay,
        "protocol_accounting_pass": protocol_ok,
        "correctness_pass": bool(turns and coherent and no_replay and protocol_ok),
        "turns": turns,
        "os_memory_before": persistent.get("os_memory_before"),
        "os_memory_after": persistent.get("os_memory_after"),
        "model": persistent.get("model"),
        "prompt_path": persistent.get("prompt_path"),
        "prompt_sha256": persistent.get("prompt_sha256"),
        "raw_artifact": child["raw_result_path"],
        "git_commit": persistent.get("git_commit"),
    }


def append_result(entry: dict[str, Any]) -> None:
    path = REPO_ROOT / "research/stage3/results.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


def compare(
    output_root: Path,
    normal_child: dict[str, Any],
    kv8_child: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    normal = mode_summary(normal_child)
    kv8 = mode_summary(kv8_child)
    normal_peak = normal.get("peak_gib")
    kv8_peak = kv8.get("peak_gib")
    normal_decode = normal.get("warm_decode_mean_tps")
    kv8_decode = kv8.get("warm_decode_mean_tps")
    memory_saving = (
        normal_peak - kv8_peak
        if normal_peak is not None and kv8_peak is not None
        else None
    )
    decode_ratio = (
        kv8_decode / normal_decode
        if kv8_decode is not None and normal_decode
        else None
    )
    result: dict[str, Any] = {
        "experiment_id": "P6-G128-NORMAL-KV-VS-KV8-PERSISTENT-16K",
        "phase": 6,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": normal.get("git_commit"),
        "mlx_lm_revision": PINNED_REVISION,
        "model_path": model,
        "prompt_path": str(PROMPT_PATH.resolve()),
        "prompt_sha256": sha256_file(PROMPT_PATH),
        "conversation_shape": "same 16K-class prompt and five exact-suffix turns; separate fresh process per mode",
        "normal_kv": normal,
        "kv8": kv8,
        "comparison": {
            "peak_saving_gib": memory_saving,
            "warm_decode_ratio_kv8_over_normal": decode_ratio,
            "warm_decode_penalty_fraction": 1.0 - decode_ratio
            if decode_ratio is not None
            else None,
            "turn_count_equal": normal["turns_completed"] == kv8["turns_completed"],
            "prompt_hash_equal": normal.get("prompt_sha256") == kv8.get("prompt_sha256"),
            "both_correctness_pass": normal["correctness_pass"] and kv8["correctness_pass"],
        },
        "decision": None,
        "raw_output_root": str(output_root),
    }
    if not normal["correctness_pass"] or not kv8["correctness_pass"]:
        result["decision"] = "STOPPED_OR_UNRESOLVED_CORRECTNESS"
    elif memory_saving is not None and decode_ratio is not None:
        if memory_saving >= 0.40 and decode_ratio >= 0.85:
            result["decision"] = "SELECTABLE_LONG_CONTEXT_MODE"
        else:
            result["decision"] = "REJECT_DEFAULT_SMALL_SAVING_OR_SLOW"
    else:
        result["decision"] = "MEASUREMENT_INCOMPLETE"
    return result


def write_report(result: dict[str, Any], path: Path) -> None:
    def fmt(value: Any, digits: int = 3) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, bool):
            return str(value)
        return f"{float(value):.{digits}f}"

    normal = result["normal_kv"]
    kv8 = result["kv8"]
    comparison = result["comparison"]
    lines = [
        "# Phase 6 g128 normal KV versus KV8 persistent 16K comparison",
        "",
        f"Decision: **{result['decision']}**",
        "",
        "Both modes used separate fresh child processes and the same 16K-class",
        "prompt/follow-up shape under the validated exact-suffix protocol.",
        "",
        "| mode | turns | cold TTFT (s) | warm TTFT median (s) | warm decode mean (tok/s) | peak (GiB) | active+cache peak (GiB) | correctness |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for mode in (normal, kv8):
        lines.append(
            f"| {mode['label']} | {mode['turns_completed']} | "
            f"{fmt(mode['cold_ttft_seconds'])} | "
            f"{fmt(mode['warm_ttft_median_seconds'])} | "
            f"{fmt(mode['warm_decode_mean_tps'])} | "
            f"{fmt(mode['peak_gib'])} | "
            f"{fmt(mode['active_plus_cache_peak_gib'])} | "
            f"{mode['correctness_pass']} |"
        )
    lines.extend(
        [
            "",
            f"Peak saving (normal minus KV8): {fmt(comparison['peak_saving_gib'])} GiB.",
            f"KV8 warm decode ratio: {fmt(comparison['warm_decode_ratio_kv8_over_normal'], 4)}; "
            f"penalty: {fmt((comparison['warm_decode_penalty_fraction'] * 100) if comparison['warm_decode_penalty_fraction'] is not None else None, 1)}%.",
            "",
            "| mode | turn | prompt | suffix | TTFT (s) | decode (tok/s) | peak (GiB) | classification | protocol |",
            "|---|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for mode in (normal, kv8):
        for turn in mode["turns"]:
            lines.append(
                f"| {mode['label']} | {turn['turn']} | {turn['prompt_tokens']} | "
                f"{turn['suffix_tokens']} | {fmt(turn['ttft_seconds'])} | "
                f"{fmt(turn['decode_tps'])} | {fmt(turn['peak_gib'])} | "
                f"{turn['classification']} | "
                f"{turn['logical_equation_holds'] and turn['cache_after_length_matches']} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(REPO_ROOT / "research/stage3/kv8-final"))
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--stop-peak-gib", type=float, default=STOP_PEAK_GIB)
    args = parser.parse_args()
    output_root = Path(args.output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    if not PROMPT_PATH.exists():
        raise SystemExit(f"missing prompt: {PROMPT_PATH}")

    # Never overlap these children: each owns a 27B model and Metal allocator.
    normal_child = run_child(
        label="normal",
        output_root=output_root,
        kv_bits=None,
        model=args.model,
        stop_peak_gib=args.stop_peak_gib,
    )
    kv8_child = run_child(
        label="kv8",
        output_root=output_root,
        kv_bits=8,
        model=args.model,
        stop_peak_gib=args.stop_peak_gib,
    )
    result = compare(output_root, normal_child, kv8_child, args.model)
    write_json(output_root / "comparison.json", result)
    write_report(result, output_root / "results.md")
    append_result(
        {
            "timestamp": result["timestamp"],
            "experiment_id": result["experiment_id"],
            "phase": 6,
            "git_commit": result["git_commit"],
            "python_version": result["normal_kv"].get("model", {}).get("python_version"),
            "mlx_version": result["normal_kv"].get("model", {}).get("mlx_version"),
            "mlx_lm_revision": PINNED_REVISION,
            "model_name": "Qwen3.8-27B-3bit-g128",
            "model_path": args.model,
            "quant_bits": 3,
            "quant_group_size": 128,
            "execution_mode": "persistent_exact_canonical_suffix_comparison",
            "persistent_cache": True,
            "kv_bits": 8,
            "kv_group_size": 64,
            "quantized_kv_start": 0,
            "prompt_path": str(PROMPT_PATH.resolve()),
            "prompt_sha256": result["prompt_sha256"],
            "total_logical_tokens": result["kv8"].get("final_logical_offset"),
            "processed_suffix_tokens": None,
            "generated_tokens": None,
            "generated_token_hash": None,
            "classification": result["decision"],
            "replay_detected": not (
                result["normal_kv"]["no_replay"] and result["kv8"]["no_replay"]
            ),
            "structured_output_valid": None,
            "ttft_seconds": result["kv8"].get("warm_ttft_median_seconds"),
            "prompt_tps": None,
            "decode_tps": result["kv8"].get("warm_decode_mean_tps"),
            "total_seconds": None,
            "active_memory_bytes": None,
            "cache_memory_bytes": None,
            "active_plus_cache_bytes": None,
            "peak_memory_bytes": None,
            "rss_bytes": None,
            "cache_state_signature": None,
            "exception": {
                "normal": result["normal_kv"].get("exception"),
                "kv8": result["kv8"].get("exception"),
            },
            "safety_stop_reason": {
                "normal": result["normal_kv"].get("safety_stop_reason"),
                "kv8": result["kv8"].get("safety_stop_reason"),
            },
            "comparison": result["comparison"],
            "raw_artifact": str(output_root / "comparison.json"),
        }
    )
    print(
        json.dumps(
            {
                "experiment_id": result["experiment_id"],
                "decision": result["decision"],
                "normal": {
                    "turns": result["normal_kv"]["turns_completed"],
                    "peak_gib": result["normal_kv"]["peak_gib"],
                    "warm_ttft_median": result["normal_kv"]["warm_ttft_median_seconds"],
                    "warm_decode": result["normal_kv"]["warm_decode_mean_tps"],
                    "correctness": result["normal_kv"]["correctness_pass"],
                },
                "kv8": {
                    "turns": result["kv8"]["turns_completed"],
                    "peak_gib": result["kv8"]["peak_gib"],
                    "warm_ttft_median": result["kv8"]["warm_ttft_median_seconds"],
                    "warm_decode": result["kv8"]["warm_decode_mean_tps"],
                    "correctness": result["kv8"]["correctness_pass"],
                },
                "peak_saving_gib": result["comparison"]["peak_saving_gib"],
                "decode_penalty_fraction": result["comparison"]["warm_decode_penalty_fraction"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if result["decision"] != "MEASUREMENT_INCOMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
