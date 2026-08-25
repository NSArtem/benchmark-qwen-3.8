#!/usr/bin/env python3
"""Focused control study for the Stage 3 turn-17 early EOS.

Arms are run in separate processes so each has an independent MLX peak:
one-shot full transcript, fresh segmented prefix, and sequential persistent
reproduction using the saved Stage 3 exact suffixes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SCALE_DIR = ROOT / "research/stage3/persistent-scaling/harness"
if str(SCALE_DIR) not in sys.path:
    sys.path.insert(0, str(SCALE_DIR))
import run_scale as scale_utils  # noqa: E402


ENDURANCE_ROOT = ROOT / "research/stage3/endurance-final"
ENDURANCE_JSON = ENDURANCE_ROOT / "endurance.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_ids(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def compact_diag(diag: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": diag.get("label"),
        "logical_prefix_token_count": diag.get("logical_prefix_token_count"),
        "offsets": diag.get("offsets"),
        "offsets_consistent": diag.get("offsets_consistent"),
        "layer_count": diag.get("layer_count"),
        "state_signature_sha256": diag.get("state_signature_sha256"),
        "class_counts": diag.get("class_counts"),
    }


def stage_record(stage: dict[str, Any], eos_ids: set[int], expected_ids: list[int] | None = None) -> dict[str, Any]:
    first_values = stage.get("first_logprob_values")
    eos_details: list[dict[str, Any]] = []
    first_vector_hash = None
    if first_values is not None:
        values = np.asarray(first_values, dtype=np.float32).reshape(-1)
        first_vector_hash = hashlib.sha256(values.tobytes()).hexdigest()
        for token_id in sorted(eos_ids):
            if 0 <= token_id < values.size:
                logprob = float(values[token_id])
                eos_details.append(
                    {
                        "token_id": token_id,
                        "logprob": logprob,
                        "probability": math.exp(logprob),
                        "rank": int(1 + np.count_nonzero(values > logprob)),
                    }
                )
    generated_ids = [int(value) for value in stage["generated_token_ids"]]
    record = {
        "prompt_tokens": stage.get("prompt_tokens"),
        "generated_token_ids": generated_ids,
        "generated_token_count": len(generated_ids),
        "generated_token_hash": sha256_ids(generated_ids),
        "decoded_output": stage.get("decoded_output"),
        "finish_reason": stage.get("finish_reason"),
        "eos_token_ids": sorted(eos_ids),
        "first_token_id": generated_ids[0] if generated_ids else None,
        "first_logprob_summary": stage.get("first_logprob_summary"),
        "first_vector_hash": first_vector_hash,
        "eos_first_token_details": eos_details,
        "ttft_seconds": stage.get("ttft_seconds"),
        "prompt_tps": stage.get("prompt_tps"),
        "decode_tps": stage.get("decode_tps"),
        "generation_seconds": stage.get("generation_seconds"),
        "total_seconds": stage.get("total_seconds"),
        "memory_after_generation": stage.get("memory_after_generation"),
        "cache_before": compact_diag(stage["cache_diagnostics"][0]),
        "cache_after_first_yield": compact_diag(stage["cache_diagnostics"][-2]),
        "cache_after": compact_diag(stage["cache_diagnostics"][-1]),
    }
    if expected_ids is not None:
        record["expected_generated_token_hash"] = sha256_ids(expected_ids)
        record["expected_generated_token_count"] = len(expected_ids)
        record["matches_saved_generation"] = generated_ids == expected_ids
    return record


def prefill_cache(mx: Any, model: Any, cache: Any, token_ids: list[int], chunk_size: int = 2048) -> dict[str, Any]:
    started = time.perf_counter()
    chunks = 0
    for start in range(0, len(token_ids), chunk_size):
        stop = min(start + chunk_size, len(token_ids))
        inputs = mx.array([token_ids[start:stop]])
        logits = model(inputs, cache=cache)
        mx.eval(logits)
        del logits, inputs
        chunks += 1
    try:
        mx.synchronize()
    except BaseException:
        pass
    return {"token_count": len(token_ids), "chunk_size": chunk_size, "chunk_count": chunks, "elapsed_seconds": time.perf_counter() - started}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--arm", choices=("one_shot", "segmented", "persistent"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tokens", type=int, default=80)
    args = parser.parse_args()
    started = time.perf_counter()
    endurance = read_json(ENDURANCE_JSON)
    turn17 = endurance["turns"][16]
    canonical_ids = [int(value) for value in read_json(ENDURANCE_ROOT / "tokens/turn-17.canonical-prompt.json")["token_ids"]]
    suffix_ids = [int(value) for value in read_json(ENDURANCE_ROOT / "tokens/turn-17.exact-suffix.json")["token_ids"]]
    prefix_count = int(turn17["cache_offset_before"])
    prefix_ids = canonical_ids[:prefix_count]
    if canonical_ids[prefix_count:] != suffix_ids:
        raise RuntimeError("saved turn-17 canonical prompt does not split at the saved cache offset")
    result: dict[str, Any] = {
        "experiment_id": f"P2-EARLY-EOS-{args.label.upper()}-{args.arm.upper()}",
        "phase": 2,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": scale_utils.oracle_utils.current_commit(),
        "model_label": args.label,
        "model_path": str(Path(args.model).resolve()),
        "arm": args.arm,
        "max_tokens": args.max_tokens,
        "protocol_reference": str(ENDURANCE_JSON),
        "turn17_user_query": turn17["user_query"],
        "turn17_prompt_tokens": len(canonical_ids),
        "turn17_cache_offset_before": prefix_count,
        "turn17_suffix_tokens": len(suffix_ids),
        "canonical_prompt_sha256": sha256_ids(canonical_ids),
        "prefix_sha256": sha256_ids(prefix_ids),
        "suffix_sha256": sha256_ids(suffix_ids),
        "model": None,
        "result": None,
        "exception": None,
    }
    try:
        mx, load, generate_step, make_prompt_cache, make_sampler, _metadata = scale_utils.oracle_utils.load_runtime()
        result["model"] = scale_utils.oracle_utils.model_metadata(args.model)
        model, tokenizer = load(args.model)
        eos_ids = {int(value) for value in (tokenizer.eos_token_ids or [])}
        result["memory_after_model_load"] = scale_utils.oracle_utils.memory_snapshot(mx, f"stage4_early_eos_{args.arm}_after_model_load", started)
        cache = make_prompt_cache(model)
        if args.arm == "one_shot":
            stage = scale_utils.oracle_utils.generate_stage(mx, generate_step, make_sampler, tokenizer, model, canonical_ids, cache, args.max_tokens, "phase2:one_shot", started)
            result["result"] = stage_record(stage, eos_ids)
            result["result"]["expected_cache_before"] = 0
            result["result"]["expected_cache_after"] = len(canonical_ids) + len(stage["generated_token_ids"])
        elif args.arm == "segmented":
            prefill = prefill_cache(mx, model, cache, prefix_ids)
            stage = scale_utils.oracle_utils.generate_stage(mx, generate_step, make_sampler, tokenizer, model, suffix_ids, cache, args.max_tokens, "phase2:segmented", started)
            result["result"] = stage_record(stage, eos_ids)
            result["result"]["prefill"] = prefill
            result["result"]["expected_cache_before"] = prefix_count
            result["result"]["expected_cache_after"] = len(canonical_ids) + len(stage["generated_token_ids"])
        else:
            prior_turns: list[dict[str, Any]] = []
            for turn_index in range(1, 17):
                canonical_path = ENDURANCE_ROOT / f"tokens/turn-{turn_index}.canonical-prompt.json"
                suffix_path = ENDURANCE_ROOT / f"tokens/turn-{turn_index}.exact-suffix.json"
                generated_path = ENDURANCE_ROOT / f"tokens/turn-{turn_index}.generated.json"
                prompt_ids = [int(value) for value in read_json(canonical_path)["token_ids"]]
                prompt_suffix = prompt_ids if turn_index == 1 else [int(value) for value in read_json(suffix_path)["token_ids"]]
                expected_ids = [int(value) for value in read_json(generated_path)["token_ids"]]
                stage = scale_utils.oracle_utils.generate_stage(mx, generate_step, make_sampler, tokenizer, model, prompt_suffix, cache, max(len(expected_ids), 1), f"phase2:persistent:turn_{turn_index}", started)
                prior_turns.append({
                    "turn": turn_index,
                    "expected_prompt_tokens": len(prompt_ids),
                    "processed_suffix_tokens": len(prompt_suffix),
                    "cache_before": compact_diag(stage["cache_diagnostics"][0]),
                    "cache_after": compact_diag(stage["cache_diagnostics"][-1]),
                    "expected_generated_token_hash": sha256_ids(expected_ids),
                    "actual_generated_token_hash": sha256_ids([int(value) for value in stage["generated_token_ids"]]),
                    "generated_token_count": len(stage["generated_token_ids"]),
                    "matches_saved_generation": [int(value) for value in stage["generated_token_ids"]] == expected_ids,
                })
                stage.pop("first_logprob_values", None)
                scale_utils.oracle_utils.safe_clear(mx)
            stage = scale_utils.oracle_utils.generate_stage(mx, generate_step, make_sampler, tokenizer, model, suffix_ids, cache, args.max_tokens, "phase2:persistent:turn_17", started)
            result["result"] = stage_record(stage, eos_ids)
            result["result"]["expected_cache_before"] = prefix_count
            result["result"]["expected_cache_after"] = len(canonical_ids) + len(stage["generated_token_ids"])
            result["prior_turns"] = prior_turns
            result["prior_generation_match_count"] = sum(item["matches_saved_generation"] for item in prior_turns)
            result["prior_generation_count"] = len(prior_turns)
        result["result"]["cache_before_matches_expected"] = result["result"]["cache_before"].get("logical_prefix_token_count") == result["result"].get("expected_cache_before")
        result["result"]["cache_after_matches_expected"] = result["result"]["cache_after"].get("logical_prefix_token_count") == result["result"].get("expected_cache_after")
        output_path = Path(args.output).with_suffix(".txt")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result["result"]["decoded_output"], encoding="utf-8")
        result["result"]["raw_output_path"] = str(output_path)
        result["memory_after_generation"] = result["result"].get("memory_after_generation")
        result["overall_classification"] = "EARLY_EOS_REPRODUCED" if result["result"]["generated_token_count"] <= 8 and result["result"]["finish_reason"] == "eos" else "COMPLETED_OR_NOT_EARLY_EOS"
    except BaseException as exc:
        result["exception"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        result["overall_classification"] = "ERROR"
    result["total_elapsed_seconds"] = time.perf_counter() - started
    write_json(Path(args.output), result)
    print(json.dumps({key: value for key, value in result.items() if key not in {"result", "prior_turns"}}, indent=2, sort_keys=True), flush=True)
    print(json.dumps({"experiment_id": result["experiment_id"], "overall_classification": result.get("overall_classification"), "generated_tokens": (result.get("result") or {}).get("generated_token_count"), "finish_reason": (result.get("result") or {}).get("finish_reason"), "eos_first_token_details": (result.get("result") or {}).get("eos_first_token_details"), "cache_before_matches_expected": (result.get("result") or {}).get("cache_before_matches_expected"), "cache_after_matches_expected": (result.get("result") or {}).get("cache_after_matches_expected")}, sort_keys=True), flush=True)
    return 1 if result.get("exception") else 0


if __name__ == "__main__":
    raise SystemExit(main())
