#!/usr/bin/env python3
"""Diagnostic profile of the unchanged Stage 7 sequential MTP oracle.

This script deliberately runs the Stage 7 snapshot/sequential-verify/replay
algorithm.  The only new behavior is opt-in timing, counters, and synchronized
memory checkpoints; headline benchmark code does not use this profile mode.
"""

from __future__ import annotations

import argparse
import collections
import gc
import hashlib
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
STAGE7 = ROOT / "stage7/harness"
SCALE = ROOT / "stage3/persistent-scaling/harness"
for path in (STAGE7, SCALE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_mtp_correctness as oracle  # noqa: E402
import run_mtp_performance as perf  # noqa: E402


DEFAULT_MODEL = "/Users/artem/models/Qwen3.8-27B-3bit-g128-MLX"
DEFAULT_SOURCE = "/private/tmp/qwen38-mtp-bf16-stage8"
DEFAULT_MTP_WEIGHTS = "/private/tmp/qwen38-mtp-4bit-g128-stage8/model.safetensors"


class DiagnosticProfile:
    """Collect nested spans and explicit synchronized cache checkpoints."""

    def __init__(self, mx: Any) -> None:
        self.mx = mx
        self.counts: collections.Counter[str] = collections.Counter()
        self.durations: collections.Counter[str] = collections.Counter()
        self._starts: dict[str, list[float]] = collections.defaultdict(list)
        self.spans: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.checkpoints: list[dict[str, Any]] = []
        self.diagnostic_sync_seconds = 0.0
        self.diagnostic_sync_calls = 0

    def count(self, name: str, amount: int = 1) -> None:
        self.counts[name] += int(amount)

    def event(self, name: str, **payload: Any) -> None:
        self.events.append({"event": name, **payload})

    def start(self, name: str) -> None:
        self._starts[name].append(time.perf_counter())

    def stop(self, name: str, **payload: Any) -> None:
        starts = self._starts.get(name)
        if not starts:
            self.event("unmatched_span_stop", name=name, **payload)
            return
        seconds = time.perf_counter() - starts.pop()
        self.durations[name] += seconds
        self.spans.append({"name": name, "seconds": seconds, **payload})

    @staticmethod
    def _array_bytes(value: Any) -> int:
        try:
            return int(value.nbytes)
        except BaseException:
            return 0

    def _cache_detail(self, cache: list[Any]) -> dict[str, int]:
        from mlx_lm.models.cache import ArraysCache, KVCache

        recurrent = 0
        attention = 0
        entries = 0
        for entry in cache:
            if entry is None:
                continue
            entries += 1
            if isinstance(entry, ArraysCache):
                recurrent += sum(self._array_bytes(item) for item in entry.cache if item is not None)
                recurrent += self._array_bytes(getattr(entry, "left_padding", None))
                recurrent += self._array_bytes(getattr(entry, "lengths", None))
            elif isinstance(entry, KVCache):
                attention += self._array_bytes(getattr(entry, "keys", None))
                attention += self._array_bytes(getattr(entry, "values", None))
        return {
            "cache_entries": entries,
            "recurrent_cache_bytes": recurrent,
            "attention_cache_bytes": attention,
            "total_cache_array_bytes": recurrent + attention,
        }

    def cache_checkpoint(self, label: str, cache: list[Any]) -> None:
        sync_started = time.perf_counter()
        self.mx.synchronize()
        sync_seconds = time.perf_counter() - sync_started
        self.diagnostic_sync_seconds += sync_seconds
        self.diagnostic_sync_calls += 1

        def getter(name: str) -> int | None:
            try:
                return int(getattr(self.mx, name)())
            except BaseException:
                return None

        detail = self._cache_detail(cache)
        self.checkpoints.append(
            {
                "label": label,
                "timestamp": time.perf_counter(),
                "active_memory_bytes": getter("get_active_memory"),
                "cache_memory_bytes": getter("get_cache_memory"),
                "peak_memory_bytes": getter("get_peak_memory"),
                "rss_max_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                **detail,
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "counts": dict(self.counts),
            "durations_seconds": dict(self.durations),
            "spans": self.spans,
            "events": self.events,
            "memory_checkpoints": self.checkpoints,
            "diagnostic_sync_calls": self.diagnostic_sync_calls,
            "diagnostic_sync_seconds": self.diagnostic_sync_seconds,
        }


def memory_snapshot(mx: Any, label: str) -> dict[str, Any]:
    mx.synchronize()

    def getter(name: str) -> int | None:
        try:
            return int(getattr(mx, name)())
        except BaseException:
            return None

    return {
        "label": label,
        "active_memory_bytes": getter("get_active_memory"),
        "cache_memory_bytes": getter("get_cache_memory"),
        "peak_memory_bytes": getter("get_peak_memory"),
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
        or f"P4-MTP-DIAGNOSTIC-{args.context.upper()}",
        "phase": 4,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "mode": "diagnostic_profile",
        "verification_mode": "sequential_snapshot_replay",
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
        result["memory_before_model"] = memory_snapshot(mx, "before_model_load")
        model, tokenizer = load(args.model)
        result["memory_after_model"] = memory_snapshot(mx, "after_model_load")
        prompt_ids, prompt_meta = perf.prompt_data(args.context, tokenizer)
        result.update(prompt_meta)

        reset_peak_memory(mx)
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
            "token_hash": baseline["token_hash"],
            "generated_tokens": len(baseline["token_ids"]),
            "finish_reason": baseline["finish_reason"],
        }
        gc.collect()
        mx.clear_cache()
        result["memory_after_baseline_cleanup"] = memory_snapshot(
            mx, "after_baseline_cleanup"
        )
        result["baseline_peak_since_reset"] = result[
            "memory_after_baseline_cleanup"
        ]["peak_memory_bytes"]

        profile = DiagnosticProfile(mx)
        reset_peak_memory(mx)
        result["memory_before_mtp_weights"] = memory_snapshot(
            mx, "before_mtp_weights"
        )
        draft, draft_meta = oracle.load_mtp_draft(
            mx,
            Path(args.source),
            model,
            Path(args.mtp_weights),
            profile=profile,
        )
        result["mtp"] = draft_meta
        result["memory_after_mtp_weights"] = memory_snapshot(
            mx, "after_mtp_weights"
        )
        result["mtp_weight_peak_since_reset"] = result["memory_after_mtp_weights"][
            "peak_memory_bytes"
        ]
        reset_peak_memory(mx)
        result["memory_before_mtp_generation"] = memory_snapshot(
            mx, "before_mtp_generation"
        )
        profiled = oracle.mtp_greedy(
            mx,
            model,
            tokenizer,
            draft,
            prompt_ids,
            args.max_tokens,
            batch_verify=False,
            profile=profile,
        )
        result["mtp_result"] = profiled
        result["exact_match"] = baseline["token_ids"] == profiled["token_ids"]
        result["memory_after_mtp_generation"] = memory_snapshot(
            mx, "after_mtp_generation"
        )
        result["mtp_peak_since_generation_reset"] = result[
            "memory_after_mtp_generation"
        ]["peak_memory_bytes"]
        result["profile"] = profile.as_dict()

        timing = profiled["timing"]
        component_seconds = {
            "target_prefill": float(timing.get("target_prefill_seconds", 0.0)),
            "first_target_logits": float(timing.get("first_logits_seconds", 0.0)),
            "draft_prefill": float(profile.durations.get("draft_prefill", 0.0)),
            "draft_proposal": float(profile.durations.get("draft_proposal", 0.0)),
            "target_cache_snapshot": float(profile.durations.get("target_cache_snapshot", 0.0)),
            "target_verification": float(profile.durations.get("target_verification", 0.0)),
            "token_compare_acceptance": float(profile.durations.get("token_compare_acceptance", 0.0)),
            "rollback_replay": float(profile.durations.get("rollback_replay", 0.0)),
            "draft_state_synchronization": float(profile.durations.get("draft_state_synchronization", 0.0)),
        }
        total = float(timing["total_seconds"])
        component_seconds["other"] = max(0.0, total - sum(component_seconds.values()))
        result["component_time_breakdown"] = {
            name: {
                "seconds": seconds,
                "percent_of_total": 100.0 * seconds / total if total else None,
            }
            for name, seconds in component_seconds.items()
        }
        result["profile_summary"] = {
            "total_seconds": total,
            "target_calls": profile.counts.get("target_forward_calls", 0),
            "target_tokens_processed": profile.counts.get("target_tokens_processed", 0),
            "replayed_target_calls": profile.counts.get("replayed_target_calls", 0),
            "replayed_target_tokens": profile.counts.get("replayed_target_tokens", 0),
            "draft_calls": profile.counts.get("draft_calls", 0),
            "proposed_tokens": profiled.get("proposed_tokens", 0),
            "accepted_tokens": profiled.get("accepted_tokens", 0),
            "rejected_tokens": profiled.get("rejected_tokens", 0),
            "rollback_checks": profiled.get("rollback_checks", 0),
            "rollback_exact": profiled.get("rollback_exact", 0),
            "diagnostic_sync_seconds": profile.diagnostic_sync_seconds,
            "diagnostic_sync_calls": profile.diagnostic_sync_calls,
            "cache_snapshot_count": len(
                [item for item in profile.checkpoints if item["label"] == "after_snapshot"]
            ),
            "max_snapshot_total_cache_array_bytes": max(
                [
                    item["total_cache_array_bytes"]
                    for item in profile.checkpoints
                    if item["label"] == "after_snapshot"
                ],
                default=0,
            ),
            "max_after_verification_active_memory_bytes": max(
                [
                    item["active_memory_bytes"] or 0
                    for item in profile.checkpoints
                    if item["label"] == "after_verification"
                ],
                default=0,
            ),
        }
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
    parser.add_argument("--context", choices=("short", "4k", "8k"), default="4k")
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
        "exact_match": result.get("exact_match"),
        "profile_summary": result.get("profile_summary"),
    }, sort_keys=True))
    return 0 if result["exception"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
