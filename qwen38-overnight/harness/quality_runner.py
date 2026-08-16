"""Run the bounded deterministic quality smoke for one candidate target."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx

from mlx_vlm.generate import stream_generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load, prepare_inputs, should_add_special_tokens


TEMPERATURE = 0.0
PREFILL_STEP_SIZE = 2048
MAX_TOKENS = 256


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def progress(path: Path, **fields: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    path.write_text(json.dumps({"timestamp": time.time(), **fields}, sort_keys=True) + "\n", encoding="utf-8")


def format_prompt(processor: Any, model: Any, raw: str) -> tuple[str, int]:
    formatted = apply_chat_template(
        processor,
        model.config,
        raw,
        enable_thinking=False,
        thinking_mode="disabled",
    )
    tokenized = prepare_inputs(
        processor,
        prompts=formatted,
        add_special_tokens=should_add_special_tokens(model.config.model_type, processor),
    )
    return formatted, int(tokenized["input_ids"].size)


def generate(model: Any, processor: Any, prompt: str, progress_path: Path, task: str) -> dict[str, Any]:
    started = time.perf_counter()
    first: float | None = None
    pieces: list[str] = []
    count = 0
    last = None
    for response in stream_generate(
        model,
        processor,
        prompt,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        enable_thinking=False,
        prefill_step_size=PREFILL_STEP_SIZE,
        verbose=False,
    ):
        now = time.perf_counter()
        if first is None:
            first = now
        pieces.append(response.text or "")
        count = int(response.generation_tokens)
        last = response
        progress(progress_path, status="decoding", task=task, output_tokens=count)
    finished = time.perf_counter()
    if last is None:
        raise RuntimeError(f"no generation response for {task}")
    text = "".join(pieces)
    prompt_tokens = int(last.prompt_tokens)
    prompt_tps = float(last.prompt_tps or 0.0)
    return {
        "text": text,
        "output_tokens": count,
        "input_tokens": prompt_tokens,
        "prefill_tok_s": prompt_tps,
        "decode_tok_s": float(last.generation_tps or 0.0),
        "ttft_seconds": first - started if first is not None else None,
        "wall_seconds": finished - started,
        "output_sha256": digest(text),
    }


def code_task() -> tuple[str, Callable[[str], tuple[bool, str]]]:
    prompt = """Coding/debugging smoke. Return only executable Python code, with no markdown or explanation. Define exactly this function:

def merge_intervals(intervals):

It must sort arbitrary [start, end] pairs, merge overlapping or touching intervals, and return a new list. Do not mutate the input. Empty input returns []."""

    def validate(output: str) -> tuple[bool, str]:
        candidate = output.strip()
        fence = re.search(r"```(?:python)?\s*(.*?)```", candidate, re.IGNORECASE | re.DOTALL)
        if fence:
            candidate = fence.group(1).strip()
        start = candidate.find("def merge_intervals")
        if start >= 0:
            candidate = candidate[start:]
        try:
            tree = ast.parse(candidate)
            namespace: dict[str, Any] = {}
            safe_builtins = {
                "sorted": sorted,
                "list": list,
                "len": len,
                "min": min,
                "max": max,
                "enumerate": enumerate,
                "range": range,
            }
            exec(compile(tree, "<quality-code>", "exec"), {"__builtins__": safe_builtins}, namespace)
            function = namespace.get("merge_intervals")
            if not callable(function):
                return False, "missing merge_intervals function"
            cases = [
                ([], []),
                ([[1, 3], [2, 6], [8, 10], [9, 12]], [[1, 6], [8, 12]]),
                ([[5, 7], [1, 2], [2, 4]], [[1, 7]]),
                ([[1, 4], [4, 5]], [[1, 5]]),
            ]
            for value, expected in cases:
                original = [item[:] for item in value]
                actual = function(value)
                if actual != expected:
                    return False, f"case {value!r} returned {actual!r}, expected {expected!r}"
                if value != original:
                    return False, "input was mutated"
            return True, "all merge-interval unit tests passed"
        except Exception as exc:
            return False, f"validator exception: {type(exc).__name__}: {exc}"

    return prompt, validate


def json_task() -> tuple[str, Callable[[str], tuple[bool, str]]]:
    prompt = """Structured-output smoke. Return exactly one JSON object and no markdown. It must have exactly these keys:
{
  "language": "typescript",
  "risk": "medium",
  "files": ["apps/api/src/application/job-service.ts", "packages/shared/src/idempotency.ts"],
  "needs_tests": true
}
The files must remain in this order; use the exact values and JSON booleans."""

    def validate(output: str) -> tuple[bool, str]:
        try:
            value = json.loads(output.strip())
        except json.JSONDecodeError as exc:
            return False, f"invalid JSON: {exc}"
        expected = {
            "language": "typescript",
            "risk": "medium",
            "files": ["apps/api/src/application/job-service.ts", "packages/shared/src/idempotency.ts"],
            "needs_tests": True,
        }
        return (value == expected, "exact schema/value match" if value == expected else f"got {value!r}")

    return prompt, validate


def reasoning_task() -> tuple[str, Callable[[str], tuple[bool, str]]]:
    prompt = "Compute 17 * 19. Return exactly `ANSWER: 323` and nothing else."

    def validate(output: str) -> tuple[bool, str]:
        ok = output.strip() == "ANSWER: 323"
        return ok, "exact objective answer" if ok else f"got {output.strip()!r}"

    return prompt, validate


def retrieval_task() -> tuple[str, Callable[[str], tuple[bool, str]]]:
    records = []
    for index in range(1, 81):
        records.append(
            f"record={index:03d} tenant=tenant-{(index % 7) + 1} state={'queued' if index % 3 == 0 else 'running'} "
            f"lease={'current' if index % 5 == 0 else 'expired'} checksum=lantern-{index:03d}-"
            f"{'blue' if index % 2 else 'amber'} note=similar-looking control text {index * 17}."
        )
    prompt = """Long-context retrieval smoke. Read the records below. Return exactly `ANSWER: lantern-057-blue` and nothing else. Do not infer from nearby records; match record=057 exactly.

""" + "\n".join(records)

    def validate(output: str) -> tuple[bool, str]:
        ok = output.strip() == "ANSWER: lantern-057-blue"
        return ok, "exact disambiguated retrieval" if ok else f"got {output.strip()!r}"

    return prompt, validate


def instruction_task() -> tuple[str, Callable[[str], tuple[bool, str]]]:
    prompt = """Instruction-following smoke. Return exactly three lowercase ASCII lines, in this order, with no code fence or extra whitespace:
mode=coding
format=json
items=3
All constraints are mandatory."""

    def validate(output: str) -> tuple[bool, str]:
        expected = "mode=coding\nformat=json\nitems=3"
        ok = output.strip() == expected
        return ok, "exact constrained format" if ok else f"got {output.strip()!r}"

    return prompt, validate


TASKS = [
    ("A-coding", code_task),
    ("B-structured-json", json_task),
    ("C-reasoning", reasoning_task),
    ("D-long-context-retrieval", retrieval_task),
    ("E-instruction-following", instruction_task),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--progress-file", type=Path, required=True)
    args = parser.parse_args()
    model = None
    try:
        progress(args.progress_file, status="loading", model=args.model)
        load_started = time.perf_counter()
        model, processor = load(args.model)
        load_seconds = time.perf_counter() - load_started
        print(json.dumps({"status": "loaded", "model": args.model, "load_seconds": load_seconds}), flush=True)
        mx.random.seed(0)
        tasks: list[dict[str, Any]] = []
        raw_outputs: dict[str, str] = {}
        for name, factory in TASKS:
            progress(args.progress_file, status="task_start", task=name)
            raw_prompt, validator = factory()
            formatted, input_tokens = format_prompt(processor, model, raw_prompt)
            measurement = generate(model, processor, formatted, args.progress_file, name)
            passed, reason = validator(measurement["text"])
            task = {
                "task": name,
                "status": "pass" if passed else "fail",
                "reason": reason,
                "input_tokens": input_tokens,
                "output_tokens": measurement["output_tokens"],
                "prefill_tok_s": measurement["prefill_tok_s"],
                "decode_tok_s": measurement["decode_tok_s"],
                "ttft_seconds": measurement["ttft_seconds"],
                "wall_seconds": measurement["wall_seconds"],
                "output_sha256": measurement["output_sha256"],
            }
            tasks.append(task)
            raw_outputs[name] = measurement["text"]
            print(json.dumps({"status": task["status"], "task": name, "reason": reason}), flush=True)
        result = {
            "status": "complete",
            "candidate": args.candidate,
            "model_id": args.model,
            "mode": "target-only-quality",
            "load_seconds": load_seconds,
            "total": len(tasks),
            "passed": sum(item["status"] == "pass" for item in tasks),
            "tasks": tasks,
            "raw_output_file": str(args.raw_output),
        }
        write_json(args.raw_output, raw_outputs)
        write_json(args.result_json, result)
        progress(args.progress_file, status="complete", passed=result["passed"], total=result["total"])
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        result = {
            "status": "failed_or_aborted",
            "candidate": args.candidate,
            "model_id": args.model,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        write_json(args.result_json, result)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
