#!/usr/bin/env python3
"""Capture local calibration logits from one reference model.

This is a serialized, calibration-only teacher capture.  It intentionally
rejects the Stage 5 held-out path and writes one compact float16 safetensors
file per case so a later student process never needs the teacher model
resident at the same time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx


ROOT = Path(__file__).resolve().parents[3]
SCALE_DIR = ROOT / "research/stage3/persistent-scaling/harness"
if str(SCALE_DIR) not in sys.path:
    sys.path.insert(0, str(SCALE_DIR))
import run_scale as scale_utils  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def token_hash(token_ids: list[int]) -> str:
    return sha256_bytes(json.dumps(token_ids, separators=(",", ":")).encode())


def normalize_ids(value: Any) -> list[int]:
    if hasattr(value, "encodings") and value.encodings:
        return [int(item) for item in value.encodings[0].ids]
    if isinstance(value, dict):
        value = value["input_ids"]
    elif hasattr(value, "input_ids"):
        value = value.input_ids
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], (list, tuple)):
        value = value[0]
    return [int(item) for item in value]


def calibration_prompt(tokenizer: Any, prompt: str) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    from_text = normalize_ids(tokenizer.encode(rendered, add_special_tokens=False))
    direct = normalize_ids(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )
    if from_text != direct:
        raise RuntimeError("chat-template string/tokenized forms disagree")
    return from_text


def snapshot(mx_module: Any) -> dict[str, Any]:
    return {
        "active_bytes": int(mx_module.get_active_memory()),
        "cache_bytes": int(mx_module.get_cache_memory()),
        "active_plus_cache_bytes": int(
            mx_module.get_active_memory() + mx_module.get_cache_memory()
        ),
        "peak_bytes": int(mx_module.get_peak_memory()),
        "max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }


def os_snapshot() -> dict[str, str]:
    output: dict[str, str] = {}
    for command in (("memory_pressure", "-Q"), ("vm_stat",)):
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=5, check=False
            )
            output[command[0]] = (completed.stdout or completed.stderr).strip()
        except (OSError, subprocess.TimeoutExpired) as exc:
            output[command[0]] = f"unavailable: {exc}"
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--experiment-id", default="P3-TEACHER-LOGITS-G64")
    args = parser.parse_args()

    calibration_path = Path(args.calibration).resolve()
    output_root = Path(args.output_root).resolve()
    if "heldout" in str(calibration_path).lower():
        raise ValueError("held-out data is forbidden for teacher calibration capture")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    logits_root = output_root / "logits"
    logits_root.mkdir()

    specs = read_jsonl(calibration_path)
    if not specs:
        raise ValueError("calibration split is empty")
    calibration_sha256 = sha256_bytes(calibration_path.read_bytes())
    started = time.perf_counter()
    result: dict[str, Any] = {
        "experiment_id": args.experiment_id,
        "phase": 3,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": scale_utils.oracle_utils.current_commit(),
        "model_path": str(Path(args.model).resolve()),
        "calibration_path": str(calibration_path),
        "calibration_sha256": calibration_sha256,
        "case_count": len(specs),
        "cases": [],
        "os_memory_before": os_snapshot(),
    }

    from mlx_lm.utils import load

    model, tokenizer = load(args.model)
    result["model"] = scale_utils.oracle_utils.model_metadata(args.model)
    result["memory_after_model_load"] = snapshot(mx)
    for spec in specs:
        case_id = spec["case_id"]
        token_ids = calibration_prompt(tokenizer, spec["prompt"])
        if len(token_ids) < 2:
            raise ValueError(f"calibration case too short: {case_id}")
        inputs = mx.array([token_ids])
        logits = model(inputs)
        mx.eval(logits)
        if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] != len(token_ids):
            raise ValueError(f"unexpected logits shape for {case_id}: {logits.shape}")
        logits16 = logits[0].astype(mx.float16)
        mx.eval(logits16)
        output_path = logits_root / f"{case_id}.safetensors"
        mx.save_safetensors(str(output_path), {"logits": logits16})
        argmax = mx.argmax(logits[0, :-1], axis=-1)
        mx.eval(argmax)
        argmax_ids = [int(value) for value in argmax.tolist()]
        result["cases"].append(
            {
                "case_id": case_id,
                "category": spec.get("category"),
                "prompt_sha256": sha256_bytes(spec["prompt"].encode()),
                "token_ids": token_ids,
                "token_count": len(token_ids),
                "token_ids_sha256": token_hash(token_ids),
                "target_count": len(token_ids) - 1,
                "teacher_argmax_sha256": token_hash(argmax_ids),
                "logits_path": str(output_path),
                "logits_dtype": "float16",
                "logits_shape": [len(token_ids), int(logits.shape[-1])],
                "memory_after_case": snapshot(mx),
            }
        )
        del argmax, logits16, logits, inputs
        scale_utils.oracle_utils.safe_clear(mx)

    result["os_memory_after"] = os_snapshot()
    result["total_elapsed_seconds"] = round(time.perf_counter() - started, 3)
    (output_root / "manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: result[key] for key in ("experiment_id", "case_count", "total_elapsed_seconds")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
