#!/usr/bin/env python3
"""Run one pinned Qwen3.8 model on the frozen Quality Suite V2.

The runner records raw outputs, generated token IDs, first-token summaries,
finish reasons, objective/semi-objective checks, and MLX memory metrics.  It
does not judge model identity or use a live service.  Run each model in a
fresh process and compare the saved JSON afterward.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
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


G64_MODEL = "/Users/artem/.cache/huggingface/hub/models--lukaskremla--Qwen3.8-27B-3bit-MLX-TextOnly/snapshots/c98bba5926f51fec1c8d8737e577221673f524d7"
G128_MODEL = "/Users/artem/models/Qwen3.8-27B-3bit-g128-MLX"
SUITE_ROOT = ROOT / "research/stage4"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_json_output(text: str) -> tuple[Any | None, bool]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.I)
        candidate = re.sub(r"\s*```$", "", candidate)
    decoder = json.JSONDecoder()
    for index, char in enumerate(candidate):
        if char not in "{[":
            continue
        try:
            value, end = decoder.raw_decode(candidate[index:])
        except json.JSONDecodeError:
            continue
        trailing = candidate[index + end :].strip()
        return value, not trailing
    return None, False


def repetition_check(token_ids: list[int]) -> dict[str, Any]:
    # A repeated short phrase is normal in a long explanation.  Treat it as
    # pathological only when an n-gram recurs enough times to occupy a
    # substantial fraction of the generation; this avoids the Stage 3 false
    # positives on numbered coding answers.
    for ngram_size in (8, 6, 4):
        counts: dict[tuple[int, ...], int] = {}
        for index in range(0, max(len(token_ids) - ngram_size + 1, 0)):
            ngram = tuple(token_ids[index : index + ngram_size])
            counts[ngram] = counts.get(ngram, 0) + 1
        repeated = max(counts.values(), default=0)
        # Repeated short Markdown/list scaffolding is common in otherwise
        # coherent test plans.  Only flag stronger repetition signatures.
        threshold = {8: 5, 6: 12, 4: 16}[ngram_size]
        if repeated >= threshold:
            return {"repetition_detected": True, "repeated_ngram_size": ngram_size, "repeated_ngram_count": repeated}
    return {"repetition_detected": False, "repeated_ngram_size": None, "repeated_ngram_count": 0}


def evaluate_case(spec: dict[str, Any], text: str, token_ids: list[int]) -> dict[str, Any]:
    stripped = text.strip()
    lower = stripped.lower()
    groups = {
        str(index): any(term.lower() in lower for term in terms)
        for index, terms in enumerate(spec.get("required_groups", []), start=1)
    }
    def contains_phrase(phrase: str) -> bool:
        escaped = re.escape(phrase.lower())
        return re.search(rf"(?<!\w){escaped}(?!\w)", lower) is not None

    forbidden = {term: not contains_phrase(term) for term in spec.get("forbidden_terms", [])}
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    prefixes = spec.get("line_prefixes")
    line_shape_ok = True
    if prefixes is not None:
        line_shape_ok = len(lines) == len(prefixes) and all(line.startswith(prefix) for line, prefix in zip(lines, prefixes))
    exact_lines = spec.get("exact_lines")
    exact_lines_ok = exact_lines is None or lines == exact_lines
    max_chars = spec.get("max_chars")
    max_chars_ok = max_chars is None or len(stripped) <= max_chars
    structured = spec.get("structured")
    parsed, strict_json = parse_json_output(stripped) if structured else (None, False)
    json_shape_ok = None
    json_types_ok = None
    json_values_ok = None
    if structured:
        json_shape_ok = isinstance(parsed, dict) and set(parsed) == set(structured["keys"]) and strict_json
        json_types_ok = bool(json_shape_ok)
        if json_types_ok:
            expected_types = {"str": str, "int": int, "bool": bool, "list": list}
            for key, type_name in structured["types"].items():
                if type(parsed[key]) is not expected_types[type_name]:
                    json_types_ok = False
        json_values_ok = bool(json_shape_ok and json_types_ok and all(parsed.get(key) == value for key, value in structured["values"].items()))
    answer_regex = spec.get("answer_regex")
    answer_ok = None if answer_regex is None else bool(re.search(answer_regex, stripped, flags=re.I))
    repetition = repetition_check(token_ids)
    mechanical_pass = bool(
        stripped
        and all(groups.values())
        and all(forbidden.values())
        and line_shape_ok
        and exact_lines_ok
        and max_chars_ok
        and (json_values_ok is not False)
        and (answer_ok is not False)
        and not repetition["repetition_detected"]
    )
    score_kind = spec.get("score_kind", "objective")
    return {
        "score_kind": score_kind,
        "mechanical_pass": mechanical_pass,
        "objective_pass": mechanical_pass if score_kind == "objective" else None,
        "semi_objective_pass": mechanical_pass if score_kind == "semi-objective" else None,
        "required_groups": groups,
        "forbidden_terms_absent": forbidden,
        "line_shape_ok": line_shape_ok,
        "exact_lines_ok": exact_lines_ok,
        "line_count": len(lines),
        "max_chars_ok": max_chars_ok,
        "answer_ok": answer_ok,
        "json_parse_ok": strict_json if structured else None,
        "json_shape_ok": json_shape_ok,
        "json_types_ok": json_types_ok,
        "json_values_ok": json_values_ok,
        "parsed_json": parsed if structured and strict_json else None,
        **repetition,
    }


def run_model(args: argparse.Namespace) -> int:
    split_path = SUITE_ROOT / ("calibration/calibration_v2.jsonl" if args.split == "calibration" else "evaluation/evaluation_v2.jsonl")
    specs = read_jsonl(split_path)
    if args.case_ids:
        selected = set(args.case_ids)
        specs = [spec for spec in specs if spec["case_id"] in selected]
        missing = selected - {spec["case_id"] for spec in specs}
        if missing:
            raise ValueError(f"unknown case IDs: {sorted(missing)}")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "outputs").mkdir(exist_ok=True)
    (output_root / "tokens").mkdir(exist_ok=True)
    (output_root / "state_signatures").mkdir(exist_ok=True)
    (output_root / "logs").mkdir(exist_ok=True)
    started = time.perf_counter()
    result: dict[str, Any] = {
        "experiment_id": f"P1-{args.split.upper()}-{args.label.upper()}-QUALITY-V2",
        "phase": 1,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": scale_utils.oracle_utils.current_commit(),
        "model_label": args.label,
        "model_path": str(Path(args.model).resolve()),
        "split": args.split,
        "suite_path": str(split_path),
        "suite_sha256": sha256_text(split_path.read_text(encoding="utf-8")),
        "case_count": len(specs),
        "selected_case_ids": [spec["case_id"] for spec in specs],
        "model": None,
        "cases": [],
        "exception": None,
        "overall_classification": None,
    }
    try:
        mx, load, generate_step, make_prompt_cache, make_sampler, _ = scale_utils.oracle_utils.load_runtime()
        result["model"] = scale_utils.oracle_utils.model_metadata(args.model)
        result["os_memory_before"] = scale_utils.oracle_utils.os_snapshot()
        model, tokenizer = load(args.model)
        result["memory_after_model_load"] = scale_utils.oracle_utils.memory_snapshot(mx, f"stage4_{args.split}_{args.label}_after_model_load", started)
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
                    f"stage4:{args.split}:{args.label}:{spec['case_id']}",
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
            print(json.dumps({"case_id": record["case_id"], "classification": (record.get("checks") or {}).get("mechanical_pass"), "finish_reason": record.get("finish_reason"), "generated_tokens": record.get("generated_tokens"), "exception": record.get("exception")}, sort_keys=True), flush=True)
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
            "split": args.split,
            "case_count": result["case_count"],
            "overall_classification": result["overall_classification"],
            "objective_pass": sum(bool(item.get("checks", {}).get("objective_pass")) for item in result["cases"]),
            "semi_objective_pass": sum(bool(item.get("checks", {}).get("semi_objective_pass")) for item in result["cases"]),
            "exceptions": sum(bool(item.get("exception")) for item in result["cases"]),
            "total_elapsed_seconds": result["total_elapsed_seconds"],
        }
        write_json(output_root / "logs" / "summary.json", summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
    return 1 if result.get("exception") else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("model",), required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", choices=("calibration", "evaluation"), default="evaluation")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--case-ids", nargs="*", default=None)
    args = parser.parse_args()
    return run_model(args)


if __name__ == "__main__":
    raise SystemExit(main())
