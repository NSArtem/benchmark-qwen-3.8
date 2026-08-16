"""Persist one attempted stage into the machine-readable registry."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def nested(value: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def normalize(result: dict[str, Any], supervisor: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    measured = result.get("measured", {}) if isinstance(result.get("measured"), dict) else {}
    memory = supervisor.get("memory_summary", {}) if isinstance(supervisor, dict) else {}
    draft = result.get("speculative") if isinstance(result.get("speculative"), dict) else {}
    draft_config = result.get("draft_configuration") if isinstance(result.get("draft_configuration"), dict) else {}
    text_status = metadata.get("text_only_status") if isinstance(metadata, dict) else {}
    return {
        "model_id": result.get("model_id") or metadata.get("model_id"),
        "revision": metadata.get("revision"),
        "quantization": metadata.get("quantization_config"),
        "reported_download_size_bytes": metadata.get("reported_download_size_bytes"),
        "reported_download_size_gb_decimal": metadata.get("reported_download_size_gb_decimal"),
        "text_only": text_status.get("text_only") if isinstance(text_status, dict) else None,
        "input_tokens": result.get("input_tokens") or measured.get("prompt_tokens"),
        "output_tokens": result.get("output_tokens") or measured.get("generation_tokens"),
        "target_only_success": result.get("status") == "complete" and result.get("mode") == "target",
        "prefill_time_seconds": measured.get("prefill_seconds"),
        "prefill_tok_s": measured.get("prompt_tps"),
        "ttft_seconds": measured.get("time_to_first_generated_token_seconds"),
        "decode_tok_s": measured.get("generation_tps"),
        "wall_seconds": measured.get("wall_seconds"),
        "mlx_peak_memory_gb": measured.get("peak_mlx_memory_gb"),
        "minimum_free_memory_percent": memory.get("min_free_percent"),
        "swap_before_mb": memory.get("swap_before_mb"),
        "swap_peak_mb": memory.get("swap_peak_mb"),
        "swap_after_mb": memory.get("swap_after_mb"),
        "pageout_delta": memory.get("pageout_delta"),
        "swapout_delta": memory.get("swapout_delta"),
        "paging_assessment": memory.get("paging_assessment"),
        "watchdog_abort": supervisor.get("watchdog_abort"),
        "watchdog_reason": supervisor.get("watchdog_reason"),
        "mtp_model": result.get("draft_model"),
        "mtp_block_size": draft.get("block_size") or draft_config.get("block_size"),
        "drafted_tokens": draft.get("drafted_tokens"),
        "accepted_draft_tokens": draft.get("accepted_draft_tokens"),
        "acceptance_rate": draft.get("acceptance_rate"),
        "effective_tokens_per_round": draft.get("mean_accepted_tokens_per_round_including_bonus"),
        "mtp_decode_tok_s": measured.get("generation_tps") if result.get("mode") == "mtp" else None,
        "apc_cached_tokens": None,
        "apc_cache_hit_ratio": None,
        "apc_cold_ttft_seconds": None,
        "apc_warm_ttft_seconds": None,
        "quality_smoke_results": None,
        "final_status": result.get("status"),
        "final_verdict": None,
    }


def refresh_derived(registry: dict[str, Any]) -> None:
    runs = registry.get("runs", [])
    by_key = {(run.get("candidate"), run.get("stage")): run for run in runs}
    candidates = sorted({run.get("candidate") for run in runs if run.get("candidate")})
    for candidate in candidates:
        target = by_key.get((candidate, "target-practical"))
        mtp = by_key.get((candidate, "mtp-practical"))
        if target and mtp:
            target_rate = target.get("metrics", {}).get("decode_tok_s")
            mtp_rate = mtp.get("metrics", {}).get("decode_tok_s")
            if isinstance(target_rate, (int, float)) and isinstance(mtp_rate, (int, float)) and target_rate > 0:
                mtp["metrics"]["mtp_multiplier"] = mtp_rate / target_rate
                target["metrics"]["mtp_multiplier"] = None
        for run in (target, mtp):
            if run and run.get("metrics", {}).get("final_verdict") is None:
                status = run.get("metrics", {}).get("final_status")
                run["metrics"]["final_verdict"] = "completed" if status == "complete" else status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--supervisor", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args()
    registry = read_json(args.registry)
    if not isinstance(registry, dict):
        registry = {
            "schema_version": 1,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "runs": [],
            "quality": [],
        }
    registry.setdefault("runs", [])
    registry.setdefault("quality", [])
    result = read_json(args.result)
    supervisor = read_json(args.supervisor)
    metadata = read_json(args.metadata) if args.metadata else {}
    if not isinstance(result, dict):
        result = {"status": "missing_result", "error": f"Could not read {args.result}"}
    if not isinstance(supervisor, dict):
        supervisor = {"watchdog_abort": True, "watchdog_reason": "missing_supervisor_result"}
    if not isinstance(metadata, dict):
        metadata = {}
    run = {
        "candidate": args.candidate,
        "stage": args.stage,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "result_path": str(args.result),
        "supervisor_path": str(args.supervisor),
        "metadata_path": str(args.metadata) if args.metadata else None,
        "metrics": normalize(result, supervisor, metadata),
        "result": result,
        "supervisor": supervisor,
    }
    key = (args.candidate, args.stage)
    registry["runs"] = [
        old for old in registry["runs"] if (old.get("candidate"), old.get("stage")) != key
    ]
    registry["runs"].append(run)
    registry["runs"].sort(key=lambda item: (item.get("candidate", ""), item.get("stage", "")))
    refresh_derived(registry)
    args.registry.parent.mkdir(parents=True, exist_ok=True)
    args.registry.write_text(json.dumps(registry, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"candidate": args.candidate, "stage": args.stage, "status": result.get("status")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
