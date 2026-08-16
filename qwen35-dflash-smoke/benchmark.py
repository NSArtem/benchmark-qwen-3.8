"""Minimal compatibility check and one measured Qwen3.5 DFlash run."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import mlx.core as mx

from mlx_vlm.generate import stream_generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.speculative.drafters import load_drafter, validate_drafter_compatibility
from mlx_vlm.speculative.utils import format_speculative_stats
from mlx_vlm.utils import load, prepare_inputs, should_add_special_tokens


TARGET_MODEL = "mlx-community/Qwen3.5-9B-4bit"
DRAFT_MODEL = "z-lab/Qwen3.5-9B-DFlash"
REQUESTED_DRAFT_KIND = "dflash"
MAX_TOKENS = 256
CHECK_TOKENS = 4
TEMPERATURE = 0.0
SEED = 0
PREFILL_STEP_SIZE = 2048
CHECK_PROMPT = "Return exactly the word READY."


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def token_as_int(token: Any) -> int:
    if hasattr(token, "item"):
        token = token.item()
    return int(token)


def capture(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            command, text=True, stderr=subprocess.STDOUT, timeout=20
        ).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def number_after(label: str, text: str | None) -> int | None:
    if text is None:
        return None
    match = re.search(rf"{re.escape(label)}:\s+(\d+)", text)
    return int(match.group(1)) if match else None


def memory_snapshot() -> dict[str, Any]:
    swap_raw = capture(["sysctl", "-n", "vm.swapusage"])
    pressure_raw = capture(["memory_pressure", "-Q"])
    vm_raw = capture(["vm_stat"])
    swap_used = None
    if swap_raw:
        match = re.search(r"used\s*=\s*([0-9.]+)M", swap_raw)
        swap_used = float(match.group(1)) if match else None
    free_percent = None
    if pressure_raw:
        match = re.search(r"free percentage:\s*([0-9.]+)%", pressure_raw, re.I)
        free_percent = float(match.group(1)) if match else None
    return {
        "swap_used_mb": swap_used,
        "free_percent": free_percent,
        "pageouts": number_after("Pageouts", vm_raw),
        "swapouts": number_after("Swapouts", vm_raw),
    }


def reset_peak_memory() -> None:
    reset = getattr(mx, "reset_peak_memory", None)
    if callable(reset):
        reset()


def generation_kwargs(max_tokens: int) -> dict[str, Any]:
    return {
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
        "enable_thinking": False,
        "prefill_step_size": PREFILL_STEP_SIZE,
        "verbose": False,
    }


def consume_generation(
    model: Any,
    processor: Any,
    prompt: str,
    kwargs: dict[str, Any],
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
            tokens_by_count[count] = token_as_int(response.token)
        last_response = response

    finished = time.perf_counter()
    if last_response is None:
        raise RuntimeError("mlx-vlm produced no GenerationResult")

    generation_count = int(last_response.generation_tokens)
    generated_tokens = [tokens_by_count[i] for i in sorted(tokens_by_count)]
    if len(generated_tokens) != generation_count:
        raise RuntimeError(
            "could not reconstruct generated token IDs: "
            f"saw {len(generated_tokens)}, expected {generation_count}"
        )

    prompt_tokens = int(last_response.prompt_tokens)
    prompt_tps = float(last_response.prompt_tps or 0.0)
    generation_tps = float(last_response.generation_tps or 0.0)
    first_token_seconds = (
        first_yield_at - started if first_yield_at is not None else None
    )
    decode_tail_seconds = (
        finished - first_yield_at if first_yield_at is not None else None
    )
    return {
        "text": "".join(text_parts),
        "tokens": generated_tokens,
        "generation_tokens": generation_count,
        "prompt_tokens": prompt_tokens,
        "prompt_tps": prompt_tps,
        "prompt_seconds_inferred": prompt_tokens / prompt_tps if prompt_tps > 0 else None,
        "generation_tps": generation_tps,
        "decode_time_after_first_token_seconds": decode_tail_seconds,
        "decode_tps_observed_after_first_token": (
            generation_count / decode_tail_seconds
            if decode_tail_seconds and decode_tail_seconds > 0
            else None
        ),
        "time_to_first_generated_token_seconds": first_token_seconds,
        "wall_seconds": finished - started,
        "peak_mlx_memory_gb": float(last_response.peak_memory),
        "peak_mlx_memory_after_sync_gb": float(mx.get_peak_memory() / 1e9),
        "finish_reason": last_response.finish_reason,
    }


def speculative_stats(draft_model: Any) -> dict[str, Any]:
    accepted_lens = getattr(draft_model, "accept_lens", None) or []
    draft_lens = getattr(draft_model, "draft_lens", None) or []
    accepted = [float(value) for value in accepted_lens]
    drafted = [int(value) for value in draft_lens]
    rounds = len(accepted)
    accepted_draft_tokens = sum(accepted)
    drafted_tokens = sum(drafted) if len(drafted) == rounds else None
    return {
        "formatted": format_speculative_stats(draft_model),
        "rounds": rounds,
        "accepted_draft_tokens": int(accepted_draft_tokens),
        "drafted_tokens": drafted_tokens,
        "rejected_draft_tokens": (
            drafted_tokens - int(accepted_draft_tokens)
            if drafted_tokens is not None
            else None
        ),
        "acceptance_rate": (
            accepted_draft_tokens / drafted_tokens
            if drafted_tokens is not None and drafted_tokens > 0
            else None
        ),
        "mean_accepted_draft_tokens_per_round": (
            accepted_draft_tokens / rounds if rounds else None
        ),
        "mean_accepted_tokens_per_round_including_bonus": (
            (accepted_draft_tokens + rounds) / rounds if rounds else None
        ),
        "accept_lens": [int(value) for value in accepted],
        "draft_lens": drafted,
        "lifetime_rounds": getattr(draft_model, "speculative_total_rounds", None),
        "lifetime_accepted_draft_tokens": getattr(
            draft_model, "speculative_total_accepted", None
        ),
        "lifetime_drafted_tokens": getattr(
            draft_model, "speculative_total_drafted", None
        ),
        "target_forward_passes": None,
    }


def recursive_array_stats(value: Any) -> tuple[int, int]:
    if isinstance(value, dict):
        totals = [recursive_array_stats(item) for item in value.values()]
        return sum(item[0] for item in totals), sum(item[1] for item in totals)
    if isinstance(value, (list, tuple)):
        totals = [recursive_array_stats(item) for item in value]
        return sum(item[0] for item in totals), sum(item[1] for item in totals)
    nbytes = getattr(value, "nbytes", None)
    size = getattr(value, "size", None)
    if nbytes is not None:
        return int(nbytes), int(size or 0)
    return 0, 0


def draft_configuration(draft_model: Any) -> dict[str, Any]:
    config = getattr(draft_model, "config", None)
    fields = (
        "model_type",
        "block_size",
        "runtime_block_size",
        "draft_window_size",
        "target_layer_ids",
        "num_target_layers",
        "num_hidden_layers",
        "hidden_size",
        "vocab_size",
        "max_position_embeddings",
    )
    values = {field: getattr(config, field, None) for field in fields}
    weight_bytes, weight_elements = recursive_array_stats(draft_model.parameters())
    values.update(
        {
            "loaded_parameter_array_bytes": weight_bytes,
            "loaded_parameter_array_mb": weight_bytes / 1e6,
            "loaded_parameter_array_elements": weight_elements,
        }
    )
    return values


def load_bundle() -> tuple[Any, Any, Any, str, float, dict[str, Any]]:
    started = time.perf_counter()
    model, processor = load(TARGET_MODEL)
    draft_model, resolved_kind = load_drafter(
        DRAFT_MODEL, kind=REQUESTED_DRAFT_KIND
    )
    if resolved_kind != REQUESTED_DRAFT_KIND:
        raise RuntimeError(
            f"DFlash was not selected: requested={REQUESTED_DRAFT_KIND!r}, "
            f"resolved={resolved_kind!r}"
        )
    validate_drafter_compatibility(model, draft_model, resolved_kind)
    config = draft_configuration(draft_model)
    return model, processor, draft_model, resolved_kind, time.perf_counter() - started, config


def run_check(result_path: Path) -> None:
    model, processor, draft_model, resolved_kind, load_seconds, config = load_bundle()
    prompt = apply_chat_template(
        processor,
        model.config,
        CHECK_PROMPT,
        enable_thinking=False,
        thinking_mode="disabled",
    )
    mx.random.seed(SEED)
    kwargs = generation_kwargs(CHECK_TOKENS)
    kwargs.update({"draft_model": draft_model, "draft_kind": resolved_kind})
    check = consume_generation(model, processor, prompt, kwargs)
    stats = speculative_stats(draft_model)
    if check["generation_tokens"] <= 0 or stats["rounds"] <= 0:
        raise RuntimeError(
            "short generation completed without observable DFlash speculative rounds"
        )
    result = {
        "status": "ok",
        "target_model": TARGET_MODEL,
        "draft_model": DRAFT_MODEL,
        "requested_draft_kind": REQUESTED_DRAFT_KIND,
        "resolved_draft_kind": resolved_kind,
        "compatibility_validation": "passed",
        "dflash_dispatch_confirmed": True,
        "load_seconds": load_seconds,
        "draft_configuration": config,
        "check_prompt": CHECK_PROMPT,
        "check_output": check["text"],
        "check_generation_tokens": check["generation_tokens"],
        "check_speculative": stats,
    }
    write_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


def run_measure(args: argparse.Namespace) -> None:
    raw_prompt = args.prompt.read_text(encoding="utf-8")
    model, processor, draft_model, resolved_kind, load_seconds, config = load_bundle()
    formatted_prompt = apply_chat_template(
        processor,
        model.config,
        raw_prompt,
        enable_thinking=False,
        thinking_mode="disabled",
    )
    tokenized = prepare_inputs(
        processor,
        prompts=formatted_prompt,
        add_special_tokens=should_add_special_tokens(model.config.model_type, processor),
    )
    prepared_prompt_tokens = int(tokenized["input_ids"].size)

    mx.random.seed(SEED)
    reset_peak_memory()
    before = memory_snapshot()
    kwargs = generation_kwargs(MAX_TOKENS)
    kwargs.update({"draft_model": draft_model, "draft_kind": resolved_kind})
    print("measured_run_start")
    measured = consume_generation(model, processor, formatted_prompt, kwargs)
    mx.synchronize()
    after = memory_snapshot()
    print("measured_run_done")

    output_text = measured["text"]
    output_tokens = measured["tokens"]
    args.text_output.parent.mkdir(parents=True, exist_ok=True)
    args.tokens_output.parent.mkdir(parents=True, exist_ok=True)
    args.text_output.write_text(output_text, encoding="utf-8")
    args.tokens_output.write_text(
        json.dumps(output_tokens, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "status": "ok",
        "target_model": TARGET_MODEL,
        "draft_model": DRAFT_MODEL,
        "requested_draft_kind": REQUESTED_DRAFT_KIND,
        "resolved_draft_kind": resolved_kind,
        "compatibility_validation": "passed",
        "dflash_dispatch_confirmed": True,
        "prompt_file": str(args.prompt),
        "raw_prompt_sha256": digest_text(raw_prompt),
        "formatted_prompt_sha256": digest_text(formatted_prompt),
        "raw_prompt_chars": len(raw_prompt),
        "formatted_prompt_chars": len(formatted_prompt),
        "prepared_prompt_tokens": prepared_prompt_tokens,
        "load_seconds": load_seconds,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "seed": SEED,
        "thinking_enabled": False,
        "batch_size": 1,
        "prefill_step_size": PREFILL_STEP_SIZE,
        "draft_block_size_override": None,
        "draft_configuration": config,
        "measured": {
            key: value for key, value in measured.items() if key not in {"text", "tokens"}
        },
        "output_text_sha256": digest_text(output_text),
        "output_token_ids_sha256": digest_text(json.dumps(output_tokens)),
        "output_text_file": str(args.text_output),
        "output_tokens_file": str(args.tokens_output),
        "memory_before": before,
        "memory_after": after,
        "memory_delta": {
            key: (
                after[key] - before[key]
                if isinstance(before.get(key), (int, float))
                and isinstance(after.get(key), (int, float))
                else None
            )
            for key in ("swap_used_mb", "pageouts", "swapouts")
        },
        "speculative": speculative_stats(draft_model),
    }
    write_json(args.result_json, report)
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("check", "measure"), required=True)
    parser.add_argument("--prompt", type=Path)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--text-output", type=Path)
    parser.add_argument("--tokens-output", type=Path)
    args = parser.parse_args()
    if args.mode == "measure" and not all(
        (args.prompt, args.text_output, args.tokens_output)
    ):
        parser.error("measure mode requires --prompt, --text-output, and --tokens-output")
    return args


def main() -> None:
    args = parse_args()
    try:
        if args.mode == "check":
            run_check(args.result_json)
        else:
            run_measure(args)
    except Exception as exc:
        failure = {
            "status": "failed",
            "mode": args.mode,
            "target_model": TARGET_MODEL,
            "draft_model": DRAFT_MODEL,
            "requested_draft_kind": REQUESTED_DRAFT_KIND,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        write_json(args.result_json, failure)
        raise


if __name__ == "__main__":
    main()
