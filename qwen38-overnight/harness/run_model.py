"""Run one Qwen3.8 target-only or native-MTP MLX measurement.

The process is intentionally one-shot: it loads one target (and optionally one
native MTP drafter), performs the requested sanity or practical run, persists
its result, and exits so the next candidate can reclaim all MLX allocations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import signal
import time
from pathlib import Path
from typing import Any

import mlx.core as mx

from mlx_vlm.generate import stream_generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.speculative.drafters import load_drafter, validate_drafter_compatibility
from mlx_vlm.speculative.utils import format_speculative_stats
from mlx_vlm.utils import load, prepare_inputs, should_add_special_tokens


CHECK_PROMPT = "Reply with exactly: OK"
TEMPERATURE = 0.0
SEED = 0
PREFILL_STEP_SIZE = 2048
DEFAULT_MAX_TOKENS = 128


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def token_int(value: Any) -> int:
    if hasattr(value, "item"):
        value = value.item()
    return int(value)


def progress(path: Path, status: str, **fields: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    payload = {"timestamp": time.time(), "status": status, **fields}
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def reset_peak_memory() -> None:
    reset = getattr(mx, "reset_peak_memory", None)
    if callable(reset):
        reset()


def mlx_memory() -> dict[str, float | None]:
    try:
        mx.synchronize()
        return {
            "active_gb": float(mx.get_active_memory() / 1e9),
            "cache_gb": float(mx.get_cache_memory() / 1e9),
            "peak_gb": float(mx.get_peak_memory() / 1e9),
        }
    except Exception:
        return {"active_gb": None, "cache_gb": None, "peak_gb": None}


def generation_kwargs(max_tokens: int) -> dict[str, Any]:
    return {
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
        "enable_thinking": False,
        "prefill_step_size": PREFILL_STEP_SIZE,
        "verbose": False,
    }


def apply_prompt(processor: Any, model: Any, raw_prompt: str) -> tuple[str, int]:
    formatted = apply_chat_template(
        processor,
        model.config,
        raw_prompt,
        enable_thinking=False,
        thinking_mode="disabled",
    )
    tokenized = prepare_inputs(
        processor,
        prompts=formatted,
        add_special_tokens=should_add_special_tokens(model.config.model_type, processor),
    )
    return formatted, int(tokenized["input_ids"].size)


def consume_generation(
    model: Any,
    processor: Any,
    prompt: str,
    kwargs: dict[str, Any],
    progress_path: Path,
    label: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    first_yield_at: float | None = None
    text_parts: list[str] = []
    tokens_by_count: dict[int, int] = {}
    last_response = None
    for response in stream_generate(model, processor, prompt, **kwargs):
        now = time.perf_counter()
        if first_yield_at is None:
            first_yield_at = now
        if response.text:
            text_parts.append(response.text)
        count = int(response.generation_tokens)
        if response.token is not None and count > 0:
            tokens_by_count[count] = token_int(response.token)
        last_response = response
        progress(progress_path, "decoding", label=label, generated_tokens=count)
    finished = time.perf_counter()
    if last_response is None:
        raise RuntimeError("mlx-vlm produced no GenerationResult")
    generated_count = int(last_response.generation_tokens)
    generated_tokens = [tokens_by_count[index] for index in sorted(tokens_by_count)]
    if len(generated_tokens) != generated_count:
        raise RuntimeError(
            f"could not reconstruct generated token IDs: saw {len(generated_tokens)}, expected {generated_count}"
        )
    prompt_tokens = int(last_response.prompt_tokens)
    prompt_tps = float(last_response.prompt_tps or 0.0)
    generation_tps = float(last_response.generation_tps or 0.0)
    return {
        "text": "".join(text_parts),
        "tokens": generated_tokens,
        "generation_tokens": generated_count,
        "prompt_tokens": prompt_tokens,
        "prompt_tps": prompt_tps,
        "prefill_seconds": prompt_tokens / prompt_tps if prompt_tps > 0 else None,
        "generation_tps": generation_tps,
        "decode_seconds_after_first_token": (
            finished - first_yield_at if first_yield_at is not None else None
        ),
        "time_to_first_generated_token_seconds": (
            first_yield_at - started if first_yield_at is not None else None
        ),
        "wall_seconds": finished - started,
        "peak_mlx_memory_gb": float(last_response.peak_memory),
        "peak_mlx_memory_after_sync_gb": float(mx.get_peak_memory() / 1e9),
        "finish_reason": last_response.finish_reason,
    }


def clear_speculative_stats(draft_model: Any) -> None:
    for name in ("accept_lens", "draft_lens"):
        value = getattr(draft_model, name, None)
        if isinstance(value, list):
            value.clear()
    for name in (
        "speculative_total_rounds",
        "speculative_total_accepted",
        "speculative_total_drafted",
    ):
        if hasattr(draft_model, name):
            setattr(draft_model, name, 0)


def speculative_stats(draft_model: Any | None) -> dict[str, Any] | None:
    if draft_model is None:
        return None
    accepted_lens = getattr(draft_model, "accept_lens", None) or []
    draft_lens = getattr(draft_model, "draft_lens", None) or []
    accepted = [float(value) for value in accepted_lens]
    drafted = [int(value) for value in draft_lens]
    rounds = len(accepted)
    accepted_total = sum(accepted)
    drafted_total = sum(drafted) if len(drafted) == rounds else None
    config = getattr(draft_model, "config", None)
    return {
        "formatted": format_speculative_stats(draft_model),
        "rounds": rounds,
        "accepted_draft_tokens": int(accepted_total),
        "drafted_tokens": drafted_total,
        "acceptance_rate": (
            accepted_total / drafted_total if drafted_total not in (None, 0) else None
        ),
        "mean_accepted_draft_tokens_per_round": accepted_total / rounds if rounds else None,
        "mean_accepted_tokens_per_round_including_bonus": (
            (accepted_total + rounds) / rounds if rounds else None
        ),
        "accept_lens": [int(value) for value in accepted],
        "draft_lens": drafted,
        "block_size": getattr(config, "block_size", None),
        "runtime_block_size": getattr(config, "runtime_block_size", None),
        "target_forward_passes": None,
    }


def draft_configuration(draft_model: Any | None) -> dict[str, Any] | None:
    if draft_model is None:
        return None
    config = getattr(draft_model, "config", None)
    return {
        "model_type": getattr(config, "model_type", None),
        "block_size": getattr(config, "block_size", None),
        "runtime_block_size": getattr(config, "runtime_block_size", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "vocab_size": getattr(config, "vocab_size", None),
        "max_position_embeddings": getattr(config, "max_position_embeddings", None),
    }


def load_bundle(model_id: str, mode: str, draft_model_id: str | None) -> tuple[Any, Any, Any | None, str | None, float, dict[str, Any] | None]:
    started = time.perf_counter()
    model, processor = load(model_id)
    draft_model = None
    resolved_kind = None
    if mode == "mtp":
        if not draft_model_id:
            raise ValueError("MTP mode requires --draft-model")
        draft_model, resolved_kind = load_drafter(draft_model_id, kind="mtp")
        if resolved_kind != "mtp":
            raise RuntimeError(f"native MTP was not selected; resolved={resolved_kind!r}")
        validate_drafter_compatibility(model, draft_model, resolved_kind)
    return (
        model,
        processor,
        draft_model,
        resolved_kind,
        time.perf_counter() - started,
        draft_configuration(draft_model),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("target", "mtp"), required=True)
    parser.add_argument("--draft-model")
    parser.add_argument("--phase", choices=("tiny", "practical"), required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--text-output", type=Path, required=True)
    parser.add_argument("--tokens-output", type=Path, required=True)
    parser.add_argument("--formatted-output", type=Path, required=True)
    parser.add_argument("--progress-file", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_prompt = args.prompt.read_text(encoding="utf-8")
    progress(args.progress_file, "starting", phase=args.phase, mode=args.mode, model=args.model)
    model: Any | None = None
    processor: Any | None = None
    draft_model: Any | None = None
    resolved_kind: str | None = None
    load_seconds: float | None = None
    draft_config: dict[str, Any] | None = None
    try:
        print(json.dumps({"status": "loading", "model": args.model, "mode": args.mode}), flush=True)
        model, processor, draft_model, resolved_kind, load_seconds, draft_config = load_bundle(
            args.model, args.mode, args.draft_model
        )
        progress(args.progress_file, "loaded", load_seconds=load_seconds)
        print(json.dumps({"status": "loaded", "load_seconds": load_seconds}), flush=True)
        mx.random.seed(SEED)

        check_formatted, check_input_tokens = apply_prompt(processor, model, CHECK_PROMPT)
        check_kwargs = generation_kwargs(12)
        if draft_model is not None:
            check_kwargs.update({"draft_model": draft_model, "draft_kind": resolved_kind})
        print(json.dumps({"status": "sanity_start", "prompt_tokens": check_input_tokens}), flush=True)
        progress(args.progress_file, "sanity_start", prompt_tokens=check_input_tokens)
        check = consume_generation(model, processor, check_formatted, check_kwargs, args.progress_file, "sanity")
        check_stats = speculative_stats(draft_model)
        sanity = {
            "output": check["text"],
            "generated_tokens": check["generation_tokens"],
            "prompt_tokens": check["prompt_tokens"],
            "generation_tps": check["generation_tps"],
            "wall_seconds": check["wall_seconds"],
            "speculative": check_stats,
        }
        print(json.dumps({"status": "sanity_done", **sanity}, sort_keys=True), flush=True)
        if draft_model is not None:
            clear_speculative_stats(draft_model)

        if args.phase == "tiny":
            result = {
                "status": "complete",
                "stage": "tiny",
                "model_id": args.model,
                "mode": args.mode,
                "draft_model": args.draft_model if args.mode == "mtp" else None,
                "requested_draft_kind": "mtp" if args.mode == "mtp" else None,
                "resolved_draft_kind": resolved_kind,
                "draft_configuration": draft_config,
                "load_seconds": load_seconds,
                "sanity_check": sanity,
                "input_tokens": check["prompt_tokens"],
                "output_tokens": check["generation_tokens"],
                "output_text": check["text"],
                "output_text_sha256": digest(check["text"]),
                "temperature": TEMPERATURE,
                "thinking_enabled": False,
                "batch_size": 1,
                "prefill_step_size": PREFILL_STEP_SIZE,
                "max_tokens": 12,
                "mlx_memory": mlx_memory(),
            }
            args.result_json.parent.mkdir(parents=True, exist_ok=True)
            args.text_output.parent.mkdir(parents=True, exist_ok=True)
            args.text_output.write_text(check["text"], encoding="utf-8")
            args.tokens_output.write_text(json.dumps(check["tokens"], indent=2) + "\n", encoding="utf-8")
            write_json(args.result_json, result)
            progress(args.progress_file, "complete", stage="tiny")
            return 0

        progress(args.progress_file, "formatting_practical")
        formatted_prompt, input_tokens = apply_prompt(processor, model, raw_prompt)
        args.formatted_output.parent.mkdir(parents=True, exist_ok=True)
        args.formatted_output.write_text(formatted_prompt, encoding="utf-8")
        print(
            json.dumps(
                {
                    "status": "prepared",
                    "raw_prompt_chars": len(raw_prompt),
                    "formatted_prompt_chars": len(formatted_prompt),
                    "input_tokens": input_tokens,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        reset_peak_memory()
        mx.random.seed(SEED)
        kwargs = generation_kwargs(args.max_tokens)
        if draft_model is not None:
            kwargs.update({"draft_model": draft_model, "draft_kind": resolved_kind})
        progress(args.progress_file, "practical_start", input_tokens=input_tokens)
        print(json.dumps({"status": "practical_start", "input_tokens": input_tokens}), flush=True)
        measured = consume_generation(model, processor, formatted_prompt, kwargs, args.progress_file, "practical")
        mx.synchronize()
        measured_stats = speculative_stats(draft_model)
        text_output = measured["text"]
        token_output = measured["tokens"]
        args.text_output.parent.mkdir(parents=True, exist_ok=True)
        args.tokens_output.parent.mkdir(parents=True, exist_ok=True)
        args.text_output.write_text(text_output, encoding="utf-8")
        args.tokens_output.write_text(json.dumps(token_output, indent=2) + "\n", encoding="utf-8")
        result = {
            "status": "complete",
            "stage": "practical",
            "model_id": args.model,
            "mode": args.mode,
            "draft_model": args.draft_model if args.mode == "mtp" else None,
            "requested_draft_kind": "mtp" if args.mode == "mtp" else None,
            "resolved_draft_kind": resolved_kind,
            "compatibility_validation": "passed" if args.mode == "mtp" else None,
            "draft_configuration": draft_config,
            "load_seconds": load_seconds,
            "raw_prompt_sha256": digest(raw_prompt),
            "formatted_prompt_sha256": digest(formatted_prompt),
            "raw_prompt_chars": len(raw_prompt),
            "formatted_prompt_chars": len(formatted_prompt),
            "input_tokens": input_tokens,
            "output_tokens": measured["generation_tokens"],
            "max_tokens": args.max_tokens,
            "temperature": TEMPERATURE,
            "seed": SEED,
            "thinking_enabled": False,
            "batch_size": 1,
            "prefill_step_size": PREFILL_STEP_SIZE,
            "measured": {key: value for key, value in measured.items() if key not in {"text", "tokens"}},
            "speculative": measured_stats,
            "output_text_sha256": digest(text_output),
            "output_token_ids_sha256": digest(json.dumps(token_output)),
            "output_text_file": str(args.text_output),
            "output_tokens_file": str(args.tokens_output),
            "formatted_prompt_file": str(args.formatted_output),
            "sanity_check": sanity,
            "mlx_memory": mlx_memory(),
            "process_max_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9,
        }
        write_json(args.result_json, result)
        progress(args.progress_file, "complete", stage="practical", output_tokens=measured["generation_tokens"])
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False), flush=True)
        return 0
    except KeyboardInterrupt as exc:
        result = {
            "status": "aborted_keyboard_interrupt",
            "model_id": args.model,
            "mode": args.mode,
            "draft_model": args.draft_model if args.mode == "mtp" else None,
            "load_seconds": load_seconds,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        write_json(args.result_json, result)
        return 130
    except Exception as exc:
        result = {
            "status": "failed_or_aborted",
            "model_id": args.model,
            "mode": args.mode,
            "draft_model": args.draft_model if args.mode == "mtp" else None,
            "requested_draft_kind": "mtp" if args.mode == "mtp" else None,
            "load_seconds": load_seconds,
            "draft_configuration": draft_config,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        write_json(args.result_json, result)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return 2
    finally:
        progress(args.progress_file, "exiting")


if __name__ == "__main__":
    raise SystemExit(main())
