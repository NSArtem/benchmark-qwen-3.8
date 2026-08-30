#!/usr/bin/env python3
"""Run one model on the frozen Stage 5 held-out suite.

This is deliberately separate from candidate-selection harnesses.  The suite
path is supplied explicitly and the output is raw final-selection evidence;
this script never writes calibration, ranking, or allocation artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
SCALE_DIR = ROOT / "research/stage3/persistent-scaling/harness"
STAGE4_HARNESS = ROOT / "research/stage4/harness"
STAGE7_HARNESS = ROOT / "research/stage7/harness"
for path in (SCALE_DIR, STAGE4_HARNESS, STAGE7_HARNESS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_scale as scale_utils  # noqa: E402
from run_quality_v2 import evaluate_case, sha256_text, write_json  # noqa: E402
from dwq_patch import apply_dwq_patch  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_model(args: argparse.Namespace) -> int:
    suite_path = Path(args.suite).resolve()
    specs = read_jsonl(suite_path)
    if args.case_ids:
        selected = set(args.case_ids)
        specs = [spec for spec in specs if spec["case_id"] in selected]
        missing = selected - {spec["case_id"] for spec in specs}
        if missing:
            raise ValueError(f"unknown case IDs: {sorted(missing)}")

    output_root = Path(args.output_root)
    for name in ("outputs", "tokens", "state_signatures", "logs"):
        (output_root / name).mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    result: dict[str, Any] = {
        "experiment_id": args.experiment_id or f"P2-HELDOUT-{args.label.upper()}",
        "phase": 2,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": scale_utils.oracle_utils.current_commit(),
        "model_label": args.label,
        "model_path": str(Path(args.model).resolve()),
        "suite_path": str(suite_path),
        "suite_sha256": hashlib.sha256(suite_path.read_bytes()).hexdigest(),
        "case_count": len(specs),
        "selected_case_ids": [spec["case_id"] for spec in specs],
        "model": None,
        "dwq_patch": str(Path(args.dwq_patch).resolve()) if args.dwq_patch else None,
        "dwq_patched_modules": [],
        "cases": [],
        "exception": None,
        "overall_classification": None,
        "selection_policy": "final-selection evidence only; not used for calibration, sensitivity, allocation, or mini selection",
    }
    try:
        mx, load, generate_step, make_prompt_cache, make_sampler, _ = scale_utils.oracle_utils.load_runtime()
        result["model"] = scale_utils.oracle_utils.model_metadata(args.model)
        result["os_memory_before"] = scale_utils.oracle_utils.os_snapshot()
        model, tokenizer = load(args.model)
        result["dwq_patched_modules"] = apply_dwq_patch(model, args.dwq_patch)
        result["memory_after_model_load"] = scale_utils.oracle_utils.memory_snapshot(mx, "stage5_heldout_after_model_load", started)
        for spec in specs:
            case_started = time.perf_counter()
            record: dict[str, Any] = {
                "case_id": spec["case_id"],
                "category": spec["category"],
                "score_kind": spec.get("score_kind"),
                "prompt_sha256": sha256_text(spec["prompt"]),
                "max_tokens": spec["max_tokens"],
                "exception": None,
            }
            try:
                canonical = scale_utils.canonical_prompt(tokenizer, [{"role": "user", "content": spec["prompt"]}])
                cache = make_prompt_cache(model)
                stage = scale_utils.oracle_utils.generate_stage(
                    mx,
                    generate_step,
                    make_sampler,
                    tokenizer,
                    model,
                    canonical["token_ids"],
                    cache,
                    spec["max_tokens"],
                    f"stage5:heldout:{args.label}:{spec['case_id']}",
                    started,
                )
                generated_ids = [int(value) for value in stage["generated_token_ids"]]
                generated_text = stage["decoded_output"]
                checks = evaluate_case(spec, generated_text, generated_ids)
                output_path = output_root / "outputs" / f"{spec['case_id']}.txt"
                token_path = output_root / "tokens" / f"{spec['case_id']}.json"
                state_path = output_root / "state_signatures" / f"{spec['case_id']}.json"
                output_path.write_text(generated_text, encoding="utf-8")
                write_json(token_path, {"token_ids": generated_ids, "token_count": len(generated_ids), "sha256": scale_utils.exact_hash(generated_ids)})
                write_json(state_path, {"case_id": spec["case_id"], "cache_diagnostics": stage["cache_diagnostics"]})
                record.update(
                    {
                        "prompt_tokens": len(canonical["token_ids"]),
                        "prompt_token_sha256": scale_utils.exact_hash(canonical["token_ids"]),
                        "generated_tokens": len(generated_ids),
                        "generated_token_hash": scale_utils.exact_hash(generated_ids),
                        "decoded_output": generated_text,
                        "decoded_output_sha256": sha256_text(generated_text),
                        "checks": checks,
                        "first_logprob_summary": stage["first_logprob_summary"],
                        "finish_reason": stage["finish_reason"],
                        "early_eos": stage["finish_reason"] == "stop" and len(generated_ids) <= 8,
                        "ttft_seconds": stage["ttft_seconds"],
                        "prompt_tps": stage["prompt_tps"],
                        "decode_tps": stage["decode_tps"],
                        "total_seconds": stage["total_seconds"],
                        "memory_after_generation": stage["memory_after_generation"],
                        "cache_state_signature": stage["cache_diagnostics"][-1].get("state_signature_sha256"),
                        "state_signature_path": str(state_path),
                        "raw_output_path": str(output_path),
                        "raw_token_path": str(token_path),
                    }
                )
                del cache
                scale_utils.oracle_utils.safe_clear(mx)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                record["exception"] = f"{type(exc).__name__}: {exc}"
                record["traceback"] = traceback.format_exc()
            record["elapsed_seconds"] = time.perf_counter() - case_started
            result["cases"].append(record)
            print(json.dumps({"case_id": record["case_id"], "mechanical_pass": (record.get("checks") or {}).get("mechanical_pass"), "finish_reason": record.get("finish_reason"), "generated_tokens": record.get("generated_tokens"), "exception": record.get("exception")}, sort_keys=True), flush=True)
        result["os_memory_after"] = scale_utils.oracle_utils.os_snapshot()
        failures = [item for item in result["cases"] if item.get("exception") or item.get("checks", {}).get("mechanical_pass") is False]
        result["overall_classification"] = "PASS" if not failures else "OBJECTIVE_OR_HARNESS_FAILURES_PRESENT"
    except BaseException as exc:
        result["exception"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        result["overall_classification"] = "ERROR"
    finally:
        result["total_elapsed_seconds"] = time.perf_counter() - started
        write_json(output_root / "model.json", result)
        summary = {
            "experiment_id": result["experiment_id"],
            "model_label": args.label,
            "case_count": result["case_count"],
            "overall_classification": result["overall_classification"],
            "objective_pass": sum(bool(item.get("checks", {}).get("objective_pass")) for item in result["cases"]),
            "objective_total": sum(item.get("score_kind") == "objective" for item in specs),
            "semi_objective_pass": sum(bool(item.get("checks", {}).get("semi_objective_pass")) for item in result["cases"]),
            "semi_objective_total": sum(item.get("score_kind") == "semi-objective" for item in specs),
            "early_eos": sum(bool(item.get("early_eos")) for item in result["cases"]),
            "repetition": sum(bool(item.get("checks", {}).get("repetition_detected")) for item in result["cases"]),
            "exceptions": sum(bool(item.get("exception")) for item in result["cases"]),
            "total_elapsed_seconds": result["total_elapsed_seconds"],
        }
        write_json(output_root / "logs" / "summary.json", summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
    return 1 if result.get("exception") else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--suite", default="research/stage5/heldout/heldout_v1.jsonl")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--case-ids", nargs="*", default=None)
    parser.add_argument("--dwq-patch", default=None)
    return run_model(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
