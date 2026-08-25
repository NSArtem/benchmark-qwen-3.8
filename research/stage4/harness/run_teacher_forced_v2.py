#!/usr/bin/env python3
"""Bounded teacher-forced metrics on the frozen V2 representative subset.

For each case the model scores only the supplied reference answer conditioned
on the rendered user prompt.  Logits are reduced chunk-by-chunk to NLL and
top-1 agreement; raw logits are never saved.
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

ROOT = Path(__file__).resolve().parents[3]
SCALE_DIR = ROOT / "research/stage3/persistent-scaling/harness"
if str(SCALE_DIR) not in sys.path:
    sys.path.insert(0, str(SCALE_DIR))
import run_scale as scale_utils  # noqa: E402


SUITE_ROOT = ROOT / "research/stage4"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def text_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def score_case(mx: Any, model: Any, tokenizer: Any, make_prompt_cache: Any, spec: dict[str, Any], chunk_size: int) -> dict[str, Any]:
    canonical = scale_utils.canonical_prompt(tokenizer, [{"role": "user", "content": spec["prompt"]}])
    target_ids = scale_utils.normalize_ids(tokenizer.encode(spec["teacher_target"], add_special_tokens=False))
    if not target_ids:
        raise ValueError(f"empty teacher target: {spec['case_id']}")
    prompt_ids = [int(value) for value in canonical["token_ids"]]
    sequence = prompt_ids + [int(value) for value in target_ids]
    cache = make_prompt_cache(model)
    target_nll = 0.0
    target_correct = 0.0
    target_count = 0
    chunk_count = 0
    try:
        # A logit at input position i predicts sequence[i + 1].  Target token
        # j therefore corresponds to input position len(prompt)-1+j.
        first_target_input = len(prompt_ids) - 1
        last_target_input = first_target_input + len(target_ids)
        for start in range(0, len(sequence) - 1, chunk_size):
            stop = min(start + chunk_size, len(sequence) - 1)
            inputs = mx.array([sequence[start:stop]])
            logits = model(inputs, cache=cache)
            focus_start = max(start, first_target_input)
            focus_stop = min(stop, last_target_input)
            if focus_start < focus_stop:
                local_start = focus_start - start
                local_stop = focus_stop - start
                focus_logits = logits[:, local_start:local_stop, :]
                focus_targets = mx.array([sequence[focus_start + 1 : focus_stop + 1]])
                logprobs = mx.take_along_axis(focus_logits, focus_targets[..., None], axis=-1).squeeze(-1)
                logprobs = logprobs - mx.logsumexp(focus_logits, axis=-1)
                nll = -mx.sum(logprobs)
                correct = mx.sum(mx.argmax(focus_logits, axis=-1) == focus_targets)
                mx.eval(nll, correct)
                target_nll += float(nll.item())
                target_correct += float(correct.item())
                target_count += focus_stop - focus_start
                del focus_logits, focus_targets, logprobs, nll, correct
            del logits, inputs
            chunk_count += 1
    finally:
        del cache
        scale_utils.oracle_utils.safe_clear(mx)
    mean_nll = target_nll / target_count
    return {
        "case_id": spec["case_id"],
        "category": spec["category"],
        "prompt_sha256": digest(spec["prompt"]),
        "target_sha256": digest(target_ids),
        "prompt_tokens": len(prompt_ids),
        "target_tokens": len(target_ids),
        "evaluated_targets": target_count,
        "chunk_size": chunk_size,
        "chunk_count": chunk_count,
        "average_nll_nats": mean_nll,
        "perplexity": math.exp(mean_nll),
        "top1_accuracy": target_correct / target_count,
        "average_target_logprob_nats": -mean_nll,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", choices=("calibration", "evaluation"), default="evaluation")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    suite_path = SUITE_ROOT / ("calibration/calibration_v2.jsonl" if args.split == "calibration" else "evaluation/evaluation_v2.jsonl")
    specs = [item for item in read_jsonl(suite_path) if item.get("teacher_target")]
    result: dict[str, Any] = {
        "experiment_id": f"P1-{args.split.upper()}-{args.label.upper()}-TEACHER-FORCED-V2",
        "phase": 1,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": scale_utils.oracle_utils.current_commit(),
        "model_label": args.label,
        "model_path": str(Path(args.model).resolve()),
        "split": args.split,
        "suite_path": str(suite_path),
        "suite_sha256": text_digest(suite_path.read_text(encoding="utf-8")),
        "case_ids": [item["case_id"] for item in specs],
        "chunk_size": args.chunk_size,
        "model": None,
        "cases": [],
        "exception": None,
    }
    try:
        mx, load, _generate_step, _make_prompt_cache, _make_sampler, _metadata = scale_utils.oracle_utils.load_runtime()
        model, tokenizer = load(args.model)
        result["model"] = scale_utils.oracle_utils.model_metadata(args.model)
        result["memory_after_model_load"] = scale_utils.oracle_utils.memory_snapshot(mx, "stage4_teacher_forced_after_model_load", started)
        for spec in specs:
            case_started = time.perf_counter()
            try:
                record = score_case(mx, model, tokenizer, _make_prompt_cache, spec, args.chunk_size)
                record["elapsed_seconds"] = time.perf_counter() - case_started
                result["cases"].append(record)
                print(json.dumps(record, sort_keys=True), flush=True)
            except BaseException as exc:
                record = {"case_id": spec["case_id"], "category": spec["category"], "exception": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
                result["cases"].append(record)
                print(json.dumps(record, sort_keys=True), flush=True)
        result["memory_after_eval"] = scale_utils.oracle_utils.memory_snapshot(mx, "stage4_teacher_forced_after_eval", started)
    except BaseException as exc:
        result["exception"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    result["total_elapsed_seconds"] = time.perf_counter() - started
    valid = [item for item in result["cases"] if not item.get("exception")]
    result["mean_nll_nats"] = sum(item["average_nll_nats"] for item in valid) / len(valid) if valid else None
    result["mean_top1_accuracy"] = sum(item["top1_accuracy"] for item in valid) / len(valid) if valid else None
    result["total_target_tokens"] = sum(item["evaluated_targets"] for item in valid)
    result["total_nll_nats"] = sum(item["average_nll_nats"] * item["evaluated_targets"] for item in valid)
    result["weighted_mean_nll_nats"] = result["total_nll_nats"] / result["total_target_tokens"] if result["total_target_tokens"] else None
    result["weighted_top1_accuracy"] = sum(item["top1_accuracy"] * item["evaluated_targets"] for item in valid) / result["total_target_tokens"] if result["total_target_tokens"] else None
    write_json(Path(args.output), result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 1 if result["exception"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
