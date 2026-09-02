#!/usr/bin/env python3
"""Fresh-process headline benchmark for the Stage 8 transactional verifier.

Baseline and transactional invocations are intentionally separate. The
transactional mode checks token identity against a baseline JSON artifact but
does not run the expensive independent cache-state oracle; that oracle is
covered by ``run_transactional.py``.
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
STAGE7 = ROOT / "stage7/harness"
STAGE3 = ROOT / "stage3/persistent-scaling/harness"
STAGE8_MTP = ROOT / "stage8/mtp"
for path in (STAGE7, STAGE3, STAGE8_MTP):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_mtp_performance as perf  # noqa: E402
from run_mtp_correctness import exact_hash, load_mtp_draft  # noqa: E402
from transactional_verifier import transactional_mtp_greedy  # noqa: E402


DEFAULT_MODEL = "/Users/artem/models/Qwen3.8-27B-3bit-g128-MLX"
DEFAULT_SOURCE = "/private/tmp/qwen38-mtp-bf16-stage8"
DEFAULT_MTP_WEIGHTS = "/private/tmp/qwen38-mtp-4bit-g128-stage8/model.safetensors"


def memory_snapshot(mx: Any, label: str) -> dict[str, Any]:
    mx.synchronize()

    def get(name: str) -> int | None:
        try:
            return int(getattr(mx, name)())
        except BaseException:
            return None

    return {
        "label": label,
        "active_memory_bytes": get("get_active_memory"),
        "cache_memory_bytes": get("get_cache_memory"),
        "peak_memory_bytes": get("get_peak_memory"),
        "rss_max_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }


def reset_peak_memory(mx: Any) -> None:
    try:
        mx.reset_peak_memory()
    except BaseException:
        pass


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    result: dict[str, Any] = {
        "experiment_id": args.experiment_id
        or f"P10-TRANSACTIONAL-{args.context.upper()}-{args.mode.upper()}",
        "phase": 10,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "mode": args.mode,
        "verification_mode": (
            "generate_step" if args.mode == "baseline" else "sequential_recurrent_transaction_shared_kv"
        ),
        "forward_mode": "sequential" if args.mode == "transactional" else None,
        "model_path": str(Path(args.model).resolve()),
        "source_path": str(Path(args.source).resolve()) if args.mode == "transactional" else None,
        "mtp_weights": str(Path(args.mtp_weights).resolve()) if args.mode == "transactional" else None,
        "context": args.context,
        "max_tokens": args.max_tokens,
        "exception": None,
        "safety_stop_reason": None,
    }
    try:
        mx, load, generate_step, make_prompt_cache, make_sampler, _ = (
            perf.scale_utils.oracle_utils.load_runtime()
        )
        model, tokenizer = load(args.model)
        prompt_ids, prompt_meta = perf.prompt_data(args.context, tokenizer)
        result.update(prompt_meta)
        result["memory_after_model"] = memory_snapshot(mx, "after_model_load")

        if args.mode == "baseline":
            reset_peak_memory(mx)
            baseline = perf.baseline_run(
                mx,
                generate_step,
                make_prompt_cache,
                make_sampler,
                model,
                tokenizer,
                prompt_ids,
                args.max_tokens,
                started,
            )
            result["output"] = {
                "token_ids": baseline["generated_token_ids"],
                "token_hash": baseline["generated_token_hash"],
                "text": baseline["decoded_output"],
            }
            result["metrics"] = {
                key: baseline.get(key)
                for key in (
                    "ttft_seconds",
                    "decode_seconds",
                    "prompt_tps",
                    "decode_tps",
                    "generated_tokens",
                    "finish_reason",
                )
            }
            result["memory_after_generation"] = memory_snapshot(
                mx, "after_baseline_generation"
            )
            result["exact_match"] = True
            result["classification"] = "P10_BASELINE_REFERENCE"
        else:
            reset_peak_memory(mx)
            draft, draft_meta = load_mtp_draft(
                mx,
                Path(args.source),
                model,
                Path(args.mtp_weights).resolve(),
            )
            result["mtp"] = draft_meta
            result["memory_after_mtp_weights"] = memory_snapshot(
                mx, "after_mtp_weights"
            )
            reset_peak_memory(mx)
            result["memory_before_generation"] = memory_snapshot(
                mx, "before_transactional_generation"
            )
            transactional = transactional_mtp_greedy(
                mx,
                model,
                tokenizer,
                draft,
                prompt_ids,
                args.max_tokens,
                forward_mode="sequential",
            )
            transactional.pop("target_cache", None)
            result["transactional"] = transactional
            output_ids = [int(item) for item in transactional["token_ids"]]
            result["output"] = {
                "token_ids": output_ids,
                "token_hash": exact_hash(output_ids),
                "text": transactional["text"],
            }
            timing = transactional["timing"]
            generated_decode_tokens = max(len(output_ids) - 1, 0)
            result["metrics"] = {
                "ttft_seconds": (
                    timing["target_prefill_seconds"]
                    + timing["first_logits_seconds"]
                    + timing["draft_prefill_seconds"]
                ),
                "decode_seconds": timing["decode_seconds"],
                "decode_tps": (
                    generated_decode_tokens / timing["decode_seconds"]
                    if timing["decode_seconds"] > 0
                    else None
                ),
                "generated_tokens": len(output_ids),
                "finish_reason": transactional["finish_reason"],
            }
            result["memory_after_generation"] = memory_snapshot(
                mx, "after_transactional_generation"
            )
            if args.baseline_json:
                baseline = json.loads(Path(args.baseline_json).read_text(encoding="utf-8"))
                expected = [int(item) for item in baseline["output"]["token_ids"]]
                result["baseline_reference_path"] = str(
                    Path(args.baseline_json).resolve()
                )
                result["baseline_reference_token_hash"] = baseline["output"][
                    "token_hash"
                ]
                result["exact_match"] = expected == output_ids
            else:
                result["exact_match"] = None
            result["classification"] = (
                "P10_TRANSACTIONAL_EXACT" if result["exact_match"] is True else "P10_TRANSACTIONAL_MISMATCH"
            )
        result["rss_max_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except BaseException as exc:
        result["exception"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = __import__("traceback").format_exc()
        result["classification"] = "P10_ERROR"
    result["elapsed_seconds"] = time.perf_counter() - started
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline", "transactional"), required=True)
    parser.add_argument("--context", choices=sorted(perf.CONTEXTS), required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--mtp-weights", default=DEFAULT_MTP_WEIGHTS)
    parser.add_argument("--baseline-json", default=None)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "experiment_id": result["experiment_id"],
                "classification": result.get("classification"),
                "exception": result.get("exception"),
                "context": result.get("context"),
                "prompt_tokens": result.get("prompt_tokens"),
                "exact_match": result.get("exact_match"),
                "metrics": result.get("metrics"),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if result.get("exception") is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
