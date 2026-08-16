"""Qwen3.8-27B target-only/native-MTP feasibility smoke test."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import resource
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import mlx.core as mx

from mlx_vlm.generate import stream_generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.speculative.drafters import load_drafter, validate_drafter_compatibility
from mlx_vlm.speculative.utils import format_speculative_stats
from mlx_vlm.utils import load, prepare_inputs, should_add_special_tokens


TARGET_MODEL = "mlx-community/Qwen3.8-27B-4bit"
DRAFT_MODEL = "mlx-community/Qwen3.8-27B-MTP-4bit"
MAX_TOKENS = 128
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


def capture(command: list[str], timeout: float = 10.0) -> str | None:
    try:
        return subprocess.check_output(
            command, text=True, stderr=subprocess.STDOUT, timeout=timeout
        ).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def vm_stat_snapshot() -> dict[str, int | None]:
    raw = capture(["vm_stat"])
    if raw is None:
        return {}
    page_size_match = re.search(r"page size of (\d+) bytes", raw)
    names = {
        "Pageins": "pageins",
        "Pageouts": "pageouts",
        "Swapins": "swapins",
        "Swapouts": "swapouts",
        "Pages stored in compressor": "compressor_pages",
    }
    result: dict[str, int | None] = {
        "page_size": int(page_size_match.group(1)) if page_size_match else 4096
    }
    for line in raw.splitlines():
        for source, destination in names.items():
            match = re.match(rf"\s*{re.escape(source)}:\s+(\d+)\.?\s*$", line)
            if match:
                result[destination] = int(match.group(1))
                break
    return result


def system_memory_snapshot() -> dict[str, Any]:
    swap_raw = capture(["sysctl", "-n", "vm.swapusage"])
    pressure_raw = capture(["memory_pressure", "-Q"])
    swap: dict[str, float | None] = {
        "total_mb": None,
        "used_mb": None,
        "free_mb": None,
    }
    if swap_raw:
        for label in swap:
            match = re.search(rf"{label.removesuffix('_mb')}\s*=\s*([0-9.]+)M", swap_raw)
            if match:
                swap[label] = float(match.group(1))
    free_percent = None
    if pressure_raw:
        match = re.search(r"free percentage:\s*([0-9.]+)%", pressure_raw, re.I)
        if match:
            free_percent = float(match.group(1))
    return {
        "swap": swap,
        "free_percent": free_percent,
        **vm_stat_snapshot(),
    }


def mlx_memory_snapshot() -> dict[str, float | None]:
    try:
        mx.synchronize()
        return {
            "active_gb": float(mx.get_active_memory() / 1e9),
            "cache_gb": float(mx.get_cache_memory() / 1e9),
            "peak_gb": float(mx.get_peak_memory() / 1e9),
        }
    except Exception:
        return {"active_gb": None, "cache_gb": None, "peak_gb": None}


class MemoryMonitor:
    """Small system sampler; it records evidence and does not stop on swap growth alone."""

    def __init__(self, path: Path, mode: str, interval_seconds: float = 5.0):
        self.path = path
        self.mode = mode
        self.interval_seconds = interval_seconds
        self.phase = "startup"
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.samples: list[dict[str, Any]] = []
        self.measured_samples: list[dict[str, Any]] = []
        self.measured_before: dict[str, Any] | None = None
        self.thread = threading.Thread(target=self._run, name="qwen38-memory-monitor", daemon=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def _write(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def start(self) -> None:
        self._write(
            {
                "event": "monitor_start",
                "mode": self.mode,
                "timestamp": time.time(),
                "memory": system_memory_snapshot(),
            }
        )
        self.thread.start()

    def mark_phase(self, phase: str) -> None:
        with self.lock:
            self.phase = phase
        self._write({"event": "phase", "mode": self.mode, "phase": phase, "timestamp": time.time()})

    def reset_measured_window(self) -> None:
        before = system_memory_snapshot()
        with self.lock:
            self.phase = "measured"
            self.measured_before = before
            self.measured_samples.clear()
        self._write(
            {
                "event": "measured_window_start",
                "mode": self.mode,
                "timestamp": time.time(),
                "memory": before,
            }
        )

    def _run(self) -> None:
        last = system_memory_snapshot()
        last_time = time.monotonic()
        while not self.stop_event.wait(self.interval_seconds):
            current = system_memory_snapshot()
            now = time.monotonic()
            elapsed = max(now - last_time, 1e-6)
            sample: dict[str, Any] = {
                "event": "sample",
                "mode": self.mode,
                "phase": self.phase,
                "timestamp": time.time(),
                "elapsed_seconds": round(elapsed, 3),
                "memory": current,
            }
            for counter in ("pageins", "pageouts", "swapins", "swapouts"):
                old = last.get(counter)
                new = current.get(counter)
                sample[f"{counter}_per_second"] = (
                    round(max(int(new) - int(old), 0) / elapsed, 1)
                    if isinstance(old, int) and isinstance(new, int)
                    else None
                )
            with self.lock:
                self.samples.append(sample)
                if self.phase == "measured":
                    self.measured_samples.append(sample)
            self._write(sample)
            last = current
            last_time = now

    def summary(self) -> dict[str, Any]:
        final = system_memory_snapshot()
        with self.lock:
            measured_before = self.measured_before
            measured_samples = list(self.measured_samples)
            all_samples = list(self.samples)

        free_values = [
            sample["memory"].get("free_percent")
            for sample in measured_samples
            if isinstance(sample.get("memory", {}).get("free_percent"), (int, float))
        ]
        pageout_rates = [
            sample.get("pageouts_per_second")
            for sample in measured_samples
            if isinstance(sample.get("pageouts_per_second"), (int, float))
        ]
        swapout_rates = [
            sample.get("swapouts_per_second")
            for sample in measured_samples
            if isinstance(sample.get("swapouts_per_second"), (int, float))
        ]
        return {
            "before_measured": measured_before,
            "after_process": final,
            "sample_count_total": len(all_samples),
            "sample_count_measured": len(measured_samples),
            "free_percent_min_measured": min(free_values) if free_values else None,
            "max_pageouts_per_second": max(pageout_rates) if pageout_rates else None,
            "max_swapouts_per_second": max(swapout_rates) if swapout_rates else None,
            "paging_observed": bool(pageout_rates or swapout_rates),
            "sustained_thrashing_assessment": None,
            "note": "Sustained thrashing is assessed from the saved samples, not from swap growth alone.",
            "measured_samples": measured_samples,
        }

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=12)
        self._write({"event": "monitor_stop", "mode": self.mode, "timestamp": time.time()})


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
    generated_count = int(last_response.generation_tokens)
    generated_tokens = [tokens_by_count[i] for i in sorted(tokens_by_count)]
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
        "prefill_seconds_inferred": prompt_tokens / prompt_tps if prompt_tps > 0 else None,
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
    accepted_draft_tokens = sum(accepted)
    drafted_tokens = sum(drafted) if len(drafted) == rounds else None
    return {
        "formatted": format_speculative_stats(draft_model),
        "rounds": rounds,
        "accepted_draft_tokens": int(accepted_draft_tokens),
        "drafted_tokens": drafted_tokens,
        "rejected_draft_tokens": (
            drafted_tokens - int(accepted_draft_tokens) if drafted_tokens is not None else None
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


def load_bundle(mode: str) -> tuple[Any, Any, Any | None, str | None, float, dict[str, Any] | None]:
    started = time.perf_counter()
    model, processor = load(TARGET_MODEL)
    draft_model = None
    resolved_kind = None
    config = None
    if mode == "mtp":
        draft_model, resolved_kind = load_drafter(DRAFT_MODEL, kind="mtp")
        if resolved_kind != "mtp":
            raise RuntimeError(
                f"native MTP was not selected: requested='mtp', resolved={resolved_kind!r}"
            )
        validate_drafter_compatibility(model, draft_model, resolved_kind)
        config = draft_configuration(draft_model)
    return model, processor, draft_model, resolved_kind, time.perf_counter() - started, config


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline", "mtp"), required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--text-output", type=Path, required=True)
    parser.add_argument("--tokens-output", type=Path, required=True)
    parser.add_argument("--memory-log", type=Path, required=True)
    args = parser.parse_args()

    monitor = MemoryMonitor(args.memory_log, args.mode)
    monitor.start()
    model = None
    processor = None
    draft_model = None
    resolved_kind = None
    load_seconds = None
    draft_config = None
    check_result = None
    try:
        print(
            json.dumps(
                {
                    "status": "loading",
                    "mode": args.mode,
                    "target_model": TARGET_MODEL,
                    "draft_model": DRAFT_MODEL if args.mode == "mtp" else None,
                    "max_tokens": MAX_TOKENS,
                    "prompt_file": str(args.prompt),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        model, processor, draft_model, resolved_kind, load_seconds, draft_config = load_bundle(args.mode)
        monitor.mark_phase("loaded")
        raw_prompt = args.prompt.read_text(encoding="utf-8")
        check_prompt, _ = apply_prompt(processor, model, CHECK_PROMPT)
        mx.random.seed(SEED)
        check_kwargs = generation_kwargs(CHECK_TOKENS)
        if draft_model is not None:
            check_kwargs.update({"draft_model": draft_model, "draft_kind": resolved_kind})
        print(json.dumps({"status": "sanity_check_start", "mode": args.mode}), flush=True)
        check = consume_generation(model, processor, check_prompt, check_kwargs)
        check_result = {
            "output": check["text"],
            "generated_tokens": check["generation_tokens"],
            "speculative": speculative_stats(draft_model),
        }
        print(json.dumps({"status": "sanity_check_done", **check_result}, sort_keys=True), flush=True)
        if draft_model is not None:
            clear_speculative_stats(draft_model)

        monitor.mark_phase("long_prompt_prepare")
        formatted_prompt, prepared_prompt_tokens = apply_prompt(processor, model, raw_prompt)
        print(
            json.dumps(
                {
                    "status": "prepared",
                    "raw_prompt_chars": len(raw_prompt),
                    "formatted_prompt_chars": len(formatted_prompt),
                    "prepared_prompt_tokens": prepared_prompt_tokens,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        reset_peak_memory()
        monitor.reset_measured_window()
        mx.random.seed(SEED)
        kwargs = generation_kwargs(MAX_TOKENS)
        if draft_model is not None:
            kwargs.update({"draft_model": draft_model, "draft_kind": resolved_kind})
        print(json.dumps({"status": "measured_start", "mode": args.mode}), flush=True)
        measured = consume_generation(model, processor, formatted_prompt, kwargs)
        mx.synchronize()
        monitor.mark_phase("measured_done")
        memory = monitor.summary()

        output_text = measured["text"]
        output_tokens = measured["tokens"]
        args.text_output.parent.mkdir(parents=True, exist_ok=True)
        args.tokens_output.parent.mkdir(parents=True, exist_ok=True)
        args.text_output.write_text(output_text, encoding="utf-8")
        args.tokens_output.write_text(
            json.dumps(output_tokens, indent=2) + "\n", encoding="utf-8"
        )
        report = {
            "status": "complete",
            "mode": args.mode,
            "target_model": TARGET_MODEL,
            "draft_model": DRAFT_MODEL if args.mode == "mtp" else None,
            "requested_draft_kind": "mtp" if args.mode == "mtp" else None,
            "resolved_draft_kind": resolved_kind,
            "compatibility_validation": "passed" if args.mode == "mtp" else None,
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
            "draft_configuration": draft_config,
            "sanity_check": check_result,
            "measured": {
                key: value for key, value in measured.items() if key not in {"text", "tokens"}
            },
            "output_text_sha256": digest_text(output_text),
            "output_token_ids_sha256": digest_text(json.dumps(output_tokens)),
            "output_text_file": str(args.text_output),
            "output_tokens_file": str(args.tokens_output),
            "memory": memory,
            "mlx_memory_after_sync": mlx_memory_snapshot(),
            "process_max_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9,
        }
        write_json(args.result_json, report)
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False), flush=True)
        return 0
    except KeyboardInterrupt:
        report = {
            "status": "aborted_keyboard_interrupt",
            "mode": args.mode,
            "target_model": TARGET_MODEL,
            "draft_model": DRAFT_MODEL if args.mode == "mtp" else None,
            "requested_draft_kind": "mtp" if args.mode == "mtp" else None,
            "resolved_draft_kind": resolved_kind,
            "load_seconds": load_seconds,
            "draft_configuration": draft_config,
            "sanity_check": check_result,
            "memory": monitor.summary(),
            "error": "KeyboardInterrupt; partial logs preserved",
        }
        write_json(args.result_json, report)
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        return 130
    except Exception as exc:
        report = {
            "status": "failed_or_aborted",
            "mode": args.mode,
            "target_model": TARGET_MODEL,
            "draft_model": DRAFT_MODEL if args.mode == "mtp" else None,
            "requested_draft_kind": "mtp" if args.mode == "mtp" else None,
            "resolved_draft_kind": resolved_kind,
            "load_seconds": load_seconds,
            "draft_configuration": draft_config,
            "sanity_check": check_result,
            "memory": monitor.summary(),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        write_json(args.result_json, report)
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        return 2
    finally:
        monitor.close()


if __name__ == "__main__":
    raise SystemExit(main())
