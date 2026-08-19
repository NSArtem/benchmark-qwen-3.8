#!/usr/bin/env python3
"""Bounded teacher-forced NLL comparison for one Qwen3.8 quantization.

This intentionally loads one model per process and evaluates the same saved
token sequence without generation or persistent cache reuse.  The logits are
reduced to scalar target log-probability and top-1 accuracy before the result
is written; raw logits are never saved.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.night2.harness.protocol_cache_harness import (
    PINNED_MLX_LM_REVISION,
    current_commit,
    load_runtime,
    memory_snapshot,
    model_metadata,
    os_snapshot,
    write_json,
)


def token_hash(tokens: list[int]) -> str:
    payload = json.dumps(tokens, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--token-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=0)
    args = parser.parse_args()

    started = time.perf_counter()
    result: dict[str, Any] = {
        "experiment_id": f"P9-NLL-{Path(args.model_path).name}",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": current_commit(),
        "protocol_mode": "teacher_forced_full_sequence_no_cache",
        "model_path": str(Path(args.model_path).resolve()),
        "token_file": str(Path(args.token_file).resolve()),
        "max_tokens": args.max_tokens,
        "mlx_lm_revision": PINNED_MLX_LM_REVISION,
        "exception": None,
    }

    try:
        token_record = json.loads(Path(args.token_file).read_text(encoding="utf-8"))
        tokens = token_record["tokens"] if isinstance(token_record, dict) else token_record
        tokens = [int(value) for value in tokens]
        if len(tokens) < 3:
            raise ValueError("token file must contain at least three tokens")
        if args.offset < 0 or args.offset >= len(tokens) - 2:
            raise ValueError(f"offset {args.offset} is outside token sequence of length {len(tokens)}")
        tokens = tokens[args.offset : args.offset + args.max_tokens]
        if len(tokens) < 3:
            raise ValueError("selected token slice must contain at least three tokens")
        result["token_offset"] = args.offset
        result["chunk_size"] = args.chunk_size
        if args.chunk_size > 0:
            result["protocol_mode"] = "teacher_forced_chunked_cache"
        result["source_token_record"] = {
            key: value
            for key, value in token_record.items()
            if key != "tokens"
        } if isinstance(token_record, dict) else None
        result["tokens_used"] = len(tokens)
        result["token_sha256"] = token_hash(tokens)

        mx, load, _generate_step, _make_prompt_cache, _make_sampler, _stream = load_runtime()
        model, _tokenizer = load(args.model_path)
        result["model"] = model_metadata(args.model_path)
        result["memory_after_model_load"] = memory_snapshot(mx, "after_model_load", started)

        if args.chunk_size > 0:
            cache = _make_prompt_cache(model)
            total_nll = 0.0
            total_target_logprob = 0.0
            total_correct = 0.0
            evaluated_targets = 0
            chunk_count = 0
            for start in range(0, len(tokens) - 1, args.chunk_size):
                stop = min(start + args.chunk_size, len(tokens) - 1)
                input_ids = tokens[start:stop]
                target_ids = tokens[start + 1 : stop + 1]
                inputs = mx.array([input_ids])
                targets = mx.array([target_ids])
                logits = model(inputs, cache=cache)
                target_logprobs = mx.take_along_axis(
                    logits, targets[..., None], axis=-1
                ).squeeze(-1) - mx.logsumexp(logits, axis=-1)
                chunk_nll = -mx.sum(target_logprobs)
                chunk_correct = mx.sum(mx.argmax(logits, axis=-1) == targets)
                mx.eval(chunk_nll, chunk_correct)
                chunk_value = float(chunk_nll.item())
                total_nll += chunk_value
                total_target_logprob -= chunk_value
                total_correct += float(chunk_correct.item())
                evaluated_targets += len(target_ids)
                chunk_count += 1
                del chunk_nll, chunk_correct, target_logprobs, logits, targets, inputs
            nll_value = total_nll / evaluated_targets
            target_logprob_value = total_target_logprob / evaluated_targets
            top1_accuracy_value = total_correct / evaluated_targets
            result["chunk_count"] = chunk_count
            result["evaluated_targets"] = evaluated_targets
            del cache
        else:
            inputs = mx.array([tokens[:-1]])
            targets = mx.array([tokens[1:]])
            logits = model(inputs)
            # The input sequence is tokens[:-1]; each returned position predicts
            # the following target token tokens[1:].  Keep all returned positions.
            target_logprobs = mx.take_along_axis(
                logits, targets[..., None], axis=-1
            ).squeeze(-1) - mx.logsumexp(logits, axis=-1)
            nll = -mx.mean(target_logprobs)
            top1 = mx.argmax(logits, axis=-1)
            top1_accuracy = mx.mean(top1 == targets)
            mx.eval(nll, target_logprobs, top1_accuracy)
            nll_value = float(nll.item())
            target_logprob_value = float(mx.mean(target_logprobs).item())
            top1_accuracy_value = float(top1_accuracy.item())
            evaluated_targets = len(tokens) - 1

        result.update(
            {
                "evaluated_targets": evaluated_targets,
                "average_nll_nats": nll_value,
                "perplexity": math.exp(nll_value),
                "average_target_logprob_nats": target_logprob_value,
                "top1_accuracy": top1_accuracy_value,
                "memory_after_eval": memory_snapshot(mx, "after_eval", started),
                "os_memory_after_eval": os_snapshot(),
            }
        )
        for name in ("top1", "target_logprobs", "nll", "top1_accuracy", "logits", "targets", "inputs"):
            if name in locals():
                del locals()[name]
        gc.collect()
        try:
            mx.synchronize()
        except BaseException:
            pass
        result["memory_after_release"] = memory_snapshot(mx, "after_release", started)
    except BaseException as exc:
        import traceback

        result["exception"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        result["total_elapsed_seconds"] = time.perf_counter() - started
        write_json(Path(args.output), result)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)

    return 1 if result["exception"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
