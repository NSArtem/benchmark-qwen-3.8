"""Single-run Qwen3.5-9B mlx-vlm baseline/native-MTP smoke test.

The two modes intentionally share this exact generation path.  The script loads the
target through mlx-vlm, applies the target tokenizer's chat template with thinking
disabled, consumes ``stream_generate`` (the same path used by the mlx-vlm CLI), and
records the final mlx-vlm timing fields plus the emitted token IDs.
"""

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
DRAFT_MODEL = "mlx-community/Qwen3.5-9B-MTP-4bit"
MAX_TOKENS = 256
WARMUP_TOKENS = 8
TEMPERATURE = 0.0
PREFILL_STEP_SIZE = 2048


def _capture(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            command, text=True, stderr=subprocess.STDOUT, timeout=20
        ).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def _number_after(label: str, text: str | None) -> int | None:
    if text is None:
        return None
    match = re.search(rf"{re.escape(label)}:\s+(\d+)", text)
    return int(match.group(1)) if match else None


def system_memory_snapshot() -> dict[str, Any]:
    swap_raw = _capture(["sysctl", "-n", "vm.swapusage"])
    pressure_raw = _capture(["memory_pressure", "-Q"])
    vm_raw = _capture(["vm_stat"])

    swap_used = None
    if swap_raw is not None:
        match = re.search(r"used\s*=\s*([0-9.]+)M", swap_raw)
        swap_used = float(match.group(1)) if match else None
    free_percent = None
    if pressure_raw is not None:
        match = re.search(r"free percentage:\s*(\d+)%", pressure_raw, re.I)
        free_percent = int(match.group(1)) if match else None

    return {
        "swap_used_mb": swap_used,
        "free_percent": free_percent,
        "pageouts": _number_after("Pageouts", vm_raw),
        "swapouts": _number_after("Swapouts", vm_raw),
    }


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest_text(value: str) -> str:
    return _digest_bytes(value.encode("utf-8"))


def _token_as_int(token: Any) -> int:
    if hasattr(token, "item"):
        token = token.item()
    return int(token)


def _reset_peak_memory() -> None:
    reset = getattr(mx, "reset_peak_memory", None)
    if callable(reset):
        reset()


def _generation_kwargs(max_tokens: int) -> dict[str, Any]:
    return {
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
        "enable_thinking": False,
        "prefill_step_size": PREFILL_STEP_SIZE,
        "verbose": False,
    }


def _consume_generation(model: Any, processor: Any, prompt: str, kwargs: dict[str, Any]):
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
            # stream_generate yields each token and then a final flush result.  The
            # generation-token count makes the final result idempotent here.
            tokens_by_count[count] = _token_as_int(response.token)
        last_response = response

    finished = time.perf_counter()
    if last_response is None:
        raise RuntimeError("mlx-vlm produced no GenerationResult")

    generated_tokens = [tokens_by_count[i] for i in sorted(tokens_by_count)]
    generation_count = int(last_response.generation_tokens)
    if len(generated_tokens) != generation_count:
        raise RuntimeError(
            "could not reconstruct all generated token IDs: "
            f"saw {len(generated_tokens)}, expected {generation_count}"
        )

    prompt_tokens = int(last_response.prompt_tokens)
    prompt_tps = float(last_response.prompt_tps)
    generation_tps = float(last_response.generation_tps)
    prompt_seconds = prompt_tokens / prompt_tps if prompt_tps > 0 else None
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
        "prompt_seconds_inferred": prompt_seconds,
        "generation_tps": generation_tps,
        "decode_tail_seconds_observed": decode_tail_seconds,
        "decode_tps_observed_including_first": (
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


def _speculative_stats(draft_model: Any) -> dict[str, Any] | None:
    if draft_model is None:
        return None
    accept_lens = getattr(draft_model, "accept_lens", None)
    draft_lens = getattr(draft_model, "draft_lens", None)
    if not accept_lens:
        return {
            "formatted": format_speculative_stats(draft_model),
            "rounds": 0,
            "accepted_draft_tokens": 0,
            "drafted_tokens": 0,
            "acceptance_rate": None,
            "mean_accepted_draft_tokens_per_round": None,
            "mean_accepted_tokens_per_round_including_bonus": None,
            "accept_lens": [],
            "draft_lens": [],
        }

    accepted = [float(value) for value in accept_lens]
    drafted = [int(value) for value in (draft_lens or [])]
    rounds = len(accepted)
    accepted_draft_tokens = sum(accepted)
    drafted_tokens = sum(drafted) if len(drafted) == rounds else None
    return {
        "formatted": format_speculative_stats(draft_model),
        "rounds": rounds,
        "accepted_draft_tokens": int(accepted_draft_tokens),
        "drafted_tokens": drafted_tokens,
        "acceptance_rate": (
            accepted_draft_tokens / drafted_tokens
            if drafted_tokens is not None and drafted_tokens > 0
            else None
        ),
        "mean_accepted_draft_tokens_per_round": accepted_draft_tokens / rounds,
        "mean_accepted_tokens_per_round_including_bonus": (
            (accepted_draft_tokens + rounds) / rounds
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
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline", "mtp"), required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--text-output", type=Path, required=True)
    parser.add_argument("--tokens-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_prompt = args.prompt.read_text(encoding="utf-8")
    load_started = time.perf_counter()
    model, processor = load(TARGET_MODEL)
    draft_model = None
    resolved_kind = None
    native_block_size = None
    if args.mode == "mtp":
        draft_model, resolved_kind = load_drafter(DRAFT_MODEL, kind="mtp")
        validate_drafter_compatibility(model, draft_model, resolved_kind)
        native_block_size = getattr(getattr(draft_model, "config", None), "block_size", None)
    load_seconds = time.perf_counter() - load_started

    formatted_prompt = apply_chat_template(
        processor,
        model.config,
        raw_prompt,
        enable_thinking=False,
        thinking_mode="disabled",
    )
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    tokenized = prepare_inputs(
        processor,
        prompts=formatted_prompt,
        add_special_tokens=should_add_special_tokens(model.config.model_type, processor),
    )
    prepared_prompt_tokens = int(tokenized["input_ids"].size)

    generation_kwargs = _generation_kwargs(WARMUP_TOKENS)
    if draft_model is not None:
        generation_kwargs.update({"draft_model": draft_model, "draft_kind": "mtp"})
    print(f"mode={args.mode}")
    print(f"target_model={TARGET_MODEL}")
    print(f"draft_model={DRAFT_MODEL if draft_model is not None else None}")
    print(f"resolved_draft_kind={resolved_kind}")
    print(f"native_configured_block_size={native_block_size}")
    print(f"raw_prompt_chars={len(raw_prompt)}")
    print(f"formatted_prompt_chars={len(formatted_prompt)}")
    print(f"prepared_prompt_tokens={prepared_prompt_tokens}")
    print(f"load_seconds={load_seconds:.3f}")
    print(f"warmup_tokens={WARMUP_TOKENS}")
    print("warmup_start")
    for _ in stream_generate(model, processor, formatted_prompt, **generation_kwargs):
        pass
    mx.synchronize()
    print("warmup_done")

    _reset_peak_memory()
    memory_before = system_memory_snapshot()
    measured_kwargs = _generation_kwargs(MAX_TOKENS)
    if draft_model is not None:
        measured_kwargs.update({"draft_model": draft_model, "draft_kind": "mtp"})
    print("measured_run_start")
    measured = _consume_generation(model, processor, formatted_prompt, measured_kwargs)
    mx.synchronize()
    memory_after = system_memory_snapshot()
    print("measured_run_done")

    text_output = measured["text"]
    token_output = measured["tokens"]
    args.text_output.parent.mkdir(parents=True, exist_ok=True)
    args.tokens_output.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.text_output.write_text(text_output, encoding="utf-8")
    args.tokens_output.write_text(json.dumps(token_output, indent=2) + "\n", encoding="utf-8")

    report = {
        "mode": args.mode,
        "target_model": TARGET_MODEL,
        "draft_model": DRAFT_MODEL if draft_model is not None else None,
        "resolved_draft_kind": resolved_kind,
        "native_configured_block_size": native_block_size,
        "draft_block_size_override": None,
        "prompt_file": str(args.prompt),
        "raw_prompt_sha256": _digest_text(raw_prompt),
        "formatted_prompt_sha256": _digest_text(formatted_prompt),
        "raw_prompt_chars": len(raw_prompt),
        "formatted_prompt_chars": len(formatted_prompt),
        "prepared_prompt_tokens": prepared_prompt_tokens,
        "load_seconds": load_seconds,
        "warmup_tokens": WARMUP_TOKENS,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "thinking_enabled": False,
        "batch_size": 1,
        "prefill_step_size": PREFILL_STEP_SIZE,
        "measured": {
            key: value
            for key, value in measured.items()
            if key not in {"text", "tokens"}
        },
        "output_text_sha256": _digest_text(text_output),
        "output_token_ids_sha256": _digest_text(json.dumps(token_output)),
        "output_text_file": str(args.text_output),
        "output_tokens_file": str(args.tokens_output),
        "memory_before": memory_before,
        "memory_after": memory_after,
        "memory_delta": {
            key: (
                memory_after[key] - memory_before[key]
                if isinstance(memory_before.get(key), (int, float))
                and isinstance(memory_after.get(key), (int, float))
                else None
            )
            for key in ("swap_used_mb", "pageouts", "swapouts")
        },
        "speculative": _speculative_stats(draft_model),
    }
    args.result_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
