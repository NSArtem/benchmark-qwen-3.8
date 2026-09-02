#!/usr/bin/env python3
"""Validate the Stage 8 transactional verifier against baseline state."""

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
STAGE8_MTP = ROOT / "stage8/mtp"
for path in (STAGE7, STAGE8_MTP):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_mtp_correctness as oracle  # noqa: E402
import run_mtp_performance as perf  # noqa: E402
from divergence_oracle import cache_compare  # noqa: E402
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    result: dict[str, Any] = {
        "experiment_id": args.experiment_id or f"P8-TRANSACTIONAL-{args.context.upper()}",
        "phase": 8,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "verification_mode": "sequential_recurrent_transaction_shared_kv",
        "forward_mode": args.forward_mode,
        "model_path": str(Path(args.model).resolve()),
        "source_path": str(Path(args.source).resolve()),
        "mtp_weights": str(Path(args.mtp_weights).resolve()),
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

        baseline = oracle.baseline_greedy(
            mx,
            generate_step,
            make_sampler,
            make_prompt_cache,
            model,
            tokenizer,
            prompt_ids,
            args.max_tokens,
        )
        result["baseline"] = {
            "token_ids": baseline["token_ids"],
            "token_hash": baseline["token_hash"],
            "finish_reason": baseline["finish_reason"],
        }
        gc.collect()
        mx.clear_cache()
        draft, draft_meta = oracle.load_mtp_draft(
            mx,
            Path(args.source),
            model,
            Path(args.mtp_weights),
        )
        result["mtp"] = draft_meta
        result["memory_after_mtp_weights"] = memory_snapshot(mx, "after_mtp_weights")
        transactional = transactional_mtp_greedy(
            mx,
            model,
            tokenizer,
            draft,
            prompt_ids,
            args.max_tokens,
            forward_mode=args.forward_mode,
        )
        target_cache = transactional.pop("target_cache")
        result["transactional"] = transactional
        result["memory_after_transactional"] = memory_snapshot(
            mx, "after_transactional_generation"
        )
        result["exact_tokens"] = baseline["token_ids"] == transactional["token_ids"]

        # Independent sequential state oracle: prefill the same prompt, then
        # process exactly the target inputs that the transaction committed.
        inner = model.language_model.model
        reference_cache = model.make_cache()
        prompt = mx.array(prompt_ids, dtype=mx.int32)[None, :]
        oracle.target_prefill_baseline_schedule(mx, inner, prompt, reference_cache)
        for token_id in transactional["committed_target_inputs"]:
            hidden = inner(mx.array([[int(token_id)]], dtype=mx.int32), cache=reference_cache)
            oracle.eval_cache(mx, reference_cache, hidden)
        result["state_oracle"] = cache_compare(mx, reference_cache, target_cache)
        result["state_exact"] = result["state_oracle"]["cache_equal"]
        result["state_first_divergent"] = result["state_oracle"]["first_divergent"]
        result["memory_after_state_oracle"] = memory_snapshot(
            mx, "after_state_oracle"
        )
        result["classification"] = (
            "TRANSACTIONAL_EXACT_STATE_PASS"
            if result["exact_tokens"] and result["state_exact"]
            else "TRANSACTIONAL_EXACTNESS_FAIL"
        )
    except BaseException as exc:
        result["exception"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = __import__("traceback").format_exc()
    result["elapsed_seconds"] = time.perf_counter() - started
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--mtp-weights", default=DEFAULT_MTP_WEIGHTS)
    parser.add_argument("--context", choices=("short", "4k", "8k"), default="short")
    parser.add_argument(
        "--forward-mode", choices=("sequential", "layerwise"), default="sequential"
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "experiment_id": result["experiment_id"],
        "exception": result["exception"],
        "classification": result.get("classification"),
        "exact_tokens": result.get("exact_tokens"),
        "state_exact": result.get("state_exact"),
        "first_divergent": result.get("state_first_divergent"),
    }, sort_keys=True))
    return 0 if result["exception"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
