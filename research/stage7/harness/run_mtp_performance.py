#!/usr/bin/env python3
"""Measure the pinned-runtime Qwen3.8 MTP prototype in fresh processes.

The baseline and MTP modes are intentionally separate invocations.  The MTP
invocation receives the baseline token IDs from its matching fresh baseline
run, so exactness is checked without keeping a second target cache resident.
The MTP path is the sequential snapshot/replay oracle; a batched target
verification shortcut is not used because its separate correctness probe
failed token identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
import traceback
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
SCALE_DIR = ROOT / "research/stage3/persistent-scaling/harness"
MTP_DIR = ROOT / "research/stage7/harness"
for path in (SCALE_DIR, MTP_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_scale as scale_utils  # noqa: E402
from run_mtp_correctness import load_mtp_draft, mtp_greedy  # noqa: E402


DEFAULT_MODEL = "/Users/artem/models/Qwen3.8-27B-3bit-g128-MLX"
DEFAULT_SOURCE = "/Users/artem/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
DEFAULT_MTP_WEIGHTS = ROOT / "research/stage7/artifacts/mtp/qwen3_8-mtp-4bit-g128/model.safetensors"
DEFAULT_MAX_TOKENS = 64

CONTEXTS = {
    "short": ROOT / "research/context-ladder/prompts/prompt-1024.tokens.json",
    "4k": ROOT / "research/context-ladder/prompts/prompt-4096.tokens.json",
    "8k": ROOT / "research/context-ladder/prompts/prompt-8192.tokens.json",
    "16k": ROOT / "qwen38-mtp-smoke/coding-prompt.txt",
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def model_filesystem_bytes(model_path: Path) -> int:
    return sum(
        item.stat().st_size
        for item in model_path.glob("*.safetensors")
        if item.is_file()
    )


def model_parameter_bytes(model: Any) -> int:
    from mlx.utils import tree_flatten

    return sum(int(value.nbytes) for _key, value in tree_flatten(model.parameters()))


def prompt_data(
    context: str,
    tokenizer: Any,
) -> tuple[list[int], dict[str, Any]]:
    prompt_path = CONTEXTS[context]
    if prompt_path.suffix == ".json":
        payload = json.loads(prompt_path.read_text(encoding="utf-8"))
        prompt_ids = [int(item) for item in payload["tokens"]]
        metadata = {
            "prompt_path": str(prompt_path),
            "prompt_file_sha256": hashlib.sha256(
                Path(payload["prompt_file"]).read_bytes()
                if Path(payload["prompt_file"]).is_file()
                else (ROOT / payload["prompt_file"]).read_bytes()
            ).hexdigest(),
            "prompt_token_sha256": hashlib.sha256(
                json.dumps(prompt_ids, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "prompt_tokens": len(prompt_ids),
            "target_prompt_tokens": payload.get("target_prompt_tokens"),
            "token_file_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
        }
        return prompt_ids, metadata

    prompt_text = prompt_path.read_text(encoding="utf-8")
    canonical = scale_utils.canonical_prompt(
        tokenizer, [{"role": "user", "content": prompt_text}]
    )
    metadata = {
        "prompt_path": str(prompt_path),
        "prompt_file_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
        "prompt_token_sha256": hashlib.sha256(
            json.dumps(canonical["token_ids"], separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "prompt_tokens": len(canonical["token_ids"]),
        "target_prompt_tokens": len(canonical["token_ids"]),
    }
    return [int(item) for item in canonical["token_ids"]], metadata


def baseline_run(
    mx: Any,
    generate_step: Any,
    make_prompt_cache: Any,
    make_sampler: Any,
    model: Any,
    tokenizer: Any,
    prompt_ids: list[int],
    max_tokens: int,
    started: float,
) -> dict[str, Any]:
    cache = make_prompt_cache(model)
    stage = scale_utils.oracle_utils.generate_stage(
        mx,
        generate_step,
        make_sampler,
        tokenizer,
        model,
        prompt_ids,
        cache,
        max_tokens,
        "phase10-baseline",
        started,
    )
    return scale_utils.stage_json(stage)


def run(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    output_root = Path(args.output_root).resolve()
    result: dict[str, Any] = {
        "experiment_id": args.experiment_id
        or f"P10-MTP-PERF-{args.context.upper()}-{args.mode.upper()}",
        "phase": 10,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": None,
        "mode": args.mode,
        "context_label": args.context,
        "model_name": "g128",
        "model_path": str(Path(args.model).resolve()),
        "model_payload_bytes": None,
        "model_filesystem_bytes": model_filesystem_bytes(Path(args.model).resolve()),
        "mtp_enabled": args.mode == "mtp",
        "mtp_precision": "4-bit" if args.mode == "mtp" else None,
        "mtp_weights": str(Path(args.mtp_weights).resolve())
        if args.mtp_weights
        else None,
        "max_tokens": args.max_tokens,
        "verification_mode": "sequential_snapshot_replay"
        if args.mode == "mtp"
        else "generate_step",
        "exception": None,
        "safety_stop_reason": None,
    }
    try:
        result["git_commit"] = scale_utils.oracle_utils.current_commit()
        mx, load, generate_step, make_prompt_cache, make_sampler, _ = (
            scale_utils.oracle_utils.load_runtime()
        )
        result["os_memory_before"] = scale_utils.oracle_utils.os_snapshot()
        model, tokenizer = load(args.model)
        result["model_payload_bytes"] = model_parameter_bytes(model)
        result["memory_after_model_load"] = scale_utils.oracle_utils.memory_snapshot(
            mx, "phase10_after_model_load", started
        )
        prompt_ids, prompt_meta = prompt_data(args.context, tokenizer)
        result.update(prompt_meta)

        if args.mode == "baseline":
            baseline = baseline_run(
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
            result["baseline"] = baseline
            result["output"] = {
                "token_ids": baseline["generated_token_ids"],
                "token_hash": baseline["generated_token_hash"],
                "text": baseline["decoded_output"],
            }
            result["exact_match"] = True
            result["acceptance_rate"] = None
        else:
            draft, draft_meta = load_mtp_draft(
                mx,
                Path(args.source),
                model,
                Path(args.mtp_weights).resolve() if args.mtp_weights else None,
            )
            result["mtp"] = draft_meta
            result["memory_after_mtp_load"] = scale_utils.oracle_utils.memory_snapshot(
                mx, "phase10_after_mtp_load", started
            )
            mtp = mtp_greedy(
                mx,
                model,
                tokenizer,
                draft,
                prompt_ids,
                args.max_tokens,
                batch_verify=False,
            )
            result["mtp_result"] = mtp
            result["output"] = {
                "token_ids": mtp["token_ids"],
                "token_hash": mtp["token_hash"],
                "text": mtp["text"],
            }
            baseline_payload = None
            if args.baseline_json:
                baseline_payload = json.loads(
                    Path(args.baseline_json).read_text(encoding="utf-8")
                )
                result["baseline_reference_path"] = str(
                    Path(args.baseline_json).resolve()
                )
                expected = baseline_payload["output"]["token_ids"]
                result["exact_match"] = expected == mtp["token_ids"]
                result["baseline_reference_token_hash"] = baseline_payload["output"][
                    "token_hash"
                ]
            else:
                result["exact_match"] = None
            result["acceptance_rate"] = mtp.get("acceptance_rate")

        result["memory_after_generation"] = scale_utils.oracle_utils.memory_snapshot(
            mx, "phase10_after_generation", started
        )
        result["os_memory_after"] = scale_utils.oracle_utils.os_snapshot()
        result["rss_max_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if args.mode == "baseline":
            result["classification"] = "MTP_PERF_BASELINE_REFERENCE"
        elif result["exact_match"] is True:
            result["classification"] = "MTP_PERF_SEQUENTIAL_EXACT"
        else:
            result["classification"] = "MTP_PERF_EXACTNESS_MISMATCH"
    except BaseException as exc:
        result["exception"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        result["classification"] = "MTP_PERF_ERROR"
    finally:
        result["total_elapsed_seconds"] = time.perf_counter() - started
        write_json(output_root / f"{args.context}-{args.mode}.json", result)
        summary = {
            "experiment_id": result["experiment_id"],
            "mode": result["mode"],
            "context": result["context_label"],
            "prompt_tokens": result.get("prompt_tokens"),
            "classification": result.get("classification"),
            "exact_match": result.get("exact_match"),
            "acceptance_rate": result.get("acceptance_rate"),
            "generated_tokens": (
                result.get("output", {}).get("token_ids")
                and len(result["output"]["token_ids"])
            ),
            "decode_tps": (
                result.get("baseline", {}).get("decode_tps")
                if args.mode == "baseline"
                else (
                    max(len(result.get("output", {}).get("token_ids", [])) - 1, 0)
                    / result["mtp_result"]["timing"]["decode_seconds"]
                    if result.get("mtp_result", {}).get("timing", {}).get("decode_seconds")
                    else None
                )
            ),
            "total_elapsed_seconds": result["total_elapsed_seconds"],
            "exception": result.get("exception"),
        }
        write_json(output_root / f"{args.context}-{args.mode}-summary.json", summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if result.get("classification") in {
        "MTP_PERF_BASELINE_REFERENCE",
        "MTP_PERF_SEQUENTIAL_EXACT",
    } else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline", "mtp"), required=True)
    parser.add_argument("--context", choices=sorted(CONTEXTS), required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--mtp-weights", default=str(DEFAULT_MTP_WEIGHTS))
    parser.add_argument("--baseline-json", default=None)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--experiment-id", default=None)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
