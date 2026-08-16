"""One measured mlx-dspark smoke-test run.

This is intentionally a single-trial harness.  The model is loaded once, a short same-prompt
warm-up is run to remove first-use compilation from the quoted result, and then exactly one
baseline or DSpark generation is timed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import resource
import subprocess
import threading
import time
from pathlib import Path

import mlx.core as mx

from mlx_dspark.calibrate import apply_small_m, apply_wide_gemm, static_cap
from mlx_dspark.generate import encode_prompt, greedy_generate, speculative_generate
from mlx_dspark.load import (
    load_drafter,
    load_target,
    lookup_drafts_default,
    resolve_mode,
)


def mb(value: float) -> float:
    return round(value / (1024 * 1024), 1)


def mlx_snapshot() -> dict[str, float | None]:
    try:
        mx.synchronize()
        return {
            "mlx_active_mb": mb(float(mx.get_active_memory())),
            "mlx_cache_mb": mb(float(mx.get_cache_memory())),
            "mlx_peak_mb": mb(float(mx.get_peak_memory())),
        }
    except Exception:
        return {"mlx_active_mb": None, "mlx_cache_mb": None, "mlx_peak_mb": None}


def rss_snapshot() -> float:
    # macOS reports ru_maxrss in bytes; this process is only run on the requested Mac.
    return mb(float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


def swap_used_mb() -> float | None:
    try:
        text = subprocess.check_output(
            ["sysctl", "-n", "vm.swapusage"], text=True, stderr=subprocess.DEVNULL
        )
    except Exception:
        return None
    match = re.search(r"used\s*=\s*([0-9.]+)M", text)
    return float(match.group(1)) if match else None


def free_percent() -> int | None:
    try:
        text = subprocess.check_output(
            ["memory_pressure", "-Q"], text=True, stderr=subprocess.DEVNULL, timeout=10
        )
    except Exception:
        return None
    match = re.search(r"free percentage:\s*(\d+)%", text)
    return int(match.group(1)) if match else None


class MemoryMonitor:
    """Safety monitor for the 27B attempt.

    The process is aborted from generation callbacks after three consecutive samples show an
    extra 768 MiB of swap over the process-start baseline.  This is a stop condition, not a
    tuning parameter; normal measurements only record the samples.
    """

    def __init__(self, abort_growth_mb: float = 768.0) -> None:
        self.baseline = swap_used_mb()
        self.max_used = self.baseline
        self.abort_growth_mb = abort_growth_mb
        self.heavy_count = 0
        self.abort = threading.Event()
        self.stop = threading.Event()
        self.samples: list[float] = []
        self.thread = threading.Thread(target=self._run, name="swap-monitor", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=2)

    def _run(self) -> None:
        while not self.stop.wait(1.0):
            used = swap_used_mb()
            if used is None:
                continue
            self.samples.append(used)
            self.max_used = max(self.max_used or used, used)
            if self.baseline is not None and used - self.baseline >= self.abort_growth_mb:
                self.heavy_count += 1
            else:
                self.heavy_count = 0
            if self.heavy_count >= 3:
                self.abort.set()

    def check(self) -> None:
        if self.abort.is_set():
            raise RuntimeError(
                "aborted: sustained swap growth exceeded "
                f"{self.abort_growth_mb:.0f} MiB"
            )

    def snapshot(self) -> dict[str, float | None | int]:
        final = swap_used_mb()
        if final is not None:
            self.max_used = max(self.max_used or final, final)
        growth = None
        if self.baseline is not None and self.max_used is not None:
            growth = round(self.max_used - self.baseline, 1)
        return {
            "swap_before_mb": self.baseline,
            "swap_after_mb": final,
            "swap_peak_observed_mb": self.max_used,
            "swap_growth_mb": growth,
            "swap_samples": len(self.samples),
            "free_percent_after": free_percent(),
        }


def model_context_limit(target) -> int | None:
    for holder in (getattr(target, "model", None), getattr(getattr(target, "model", None), "args", None)):
        for name in ("max_position_embeddings", "max_seq_len", "max_context_length"):
            value = getattr(holder, name, None)
            if isinstance(value, int):
                return value
    return None


def run_generation(
    *,
    mode: str,
    target,
    tokenizer,
    drafter,
    prompt_ids: list[int],
    max_new_tokens: int,
    max_draft: int | None,
    lookup_drafts: bool,
    monitor: MemoryMonitor,
    label: str,
) -> dict:
    prefill_marks = list(range(2048, len(prompt_ids), 2048))
    started = time.perf_counter()
    prefill_done: float | None = None
    first_token: float | None = None

    def on_prefill(*args) -> None:
        nonlocal prefill_done
        position = args[-1] if args else None
        monitor.check()
        if position == len(prompt_ids):
            prefill_done = time.perf_counter()

    def on_round(**_kwargs) -> None:
        nonlocal first_token
        monitor.check()
        if first_token is None:
            first_token = time.perf_counter()

    if mode == "baseline":
        result = greedy_generate(
            target,
            tokenizer,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            apply_chat_template=False,
            on_round=on_round,
            on_prefill=on_prefill,
            prefill_marks=prefill_marks,
        )
    else:
        result = speculative_generate(
            target,
            tokenizer,
            drafter,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            max_draft_tokens=max_draft,
            lookup_drafts=lookup_drafts,
            temperature=0.0,
            apply_chat_template=False,
            on_round=on_round,
            on_prefill=on_prefill,
            prefill_marks=prefill_marks,
        )

    finished = time.perf_counter()
    if prefill_done is None:
        raise RuntimeError("generation did not report prompt prefill completion")
    prefill_seconds = prefill_done - started
    decode_seconds = finished - prefill_done
    total_seconds = finished - started
    text_digest = hashlib.sha256(result.text.encode("utf-8")).hexdigest()[:16]
    record = {
        "label": label,
        "mode": mode,
        "input_tokens": len(prompt_ids),
        "output_tokens": result.num_tokens,
        "prefill_seconds": round(prefill_seconds, 3),
        "prefill_tok_s": round(len(prompt_ids) / max(prefill_seconds, 1e-9), 1),
        "time_to_first_token_seconds": round(
            (first_token - started) if first_token is not None else float("nan"), 3
        ),
        "decode_seconds": round(decode_seconds, 3),
        "decode_tok_s": round(result.num_tokens / max(decode_seconds, 1e-9), 1),
        "total_wall_seconds": round(total_seconds, 3),
        "library_seconds": round(result.seconds, 3),
        "mean_accept_per_round": round(result.mean_accept_len, 3),
        "target_forwards": result.target_forwards,
        "rounds": result.num_rounds,
        "lookup_rounds": result.lookup_rounds,
        "finish_reason": result.finish_reason,
        "text_sha256_16": text_digest,
        "mlx_memory": mlx_snapshot(),
        "process_max_rss_mb": rss_snapshot(),
    }
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("baseline", "dspark"), required=True)
    parser.add_argument("--prompt-file", default="coding_prompt.md")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--abort-swap-growth-mb",
        type=float,
        default=768.0,
        help="safety stop after three samples above this swap growth",
    )
    args = parser.parse_args()

    resolved_mode, target_repo, drafter_repo = resolve_mode(args.model, mode=args.mode)
    monitor = MemoryMonitor(abort_growth_mb=args.abort_swap_growth_mb)
    monitor.start()
    print(
        json.dumps(
            {
                "status": "loading",
                "mode": resolved_mode,
                "target": target_repo,
                "drafter": drafter_repo,
                "requested_context_window": 32768,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        target, tokenizer = load_target(target_repo, require_tap=resolved_mode == "dspark")
        drafter = None
        selected_cap = None
        lookup = False
        if resolved_mode == "dspark":
            drafter, _ = load_drafter(drafter_repo, quantize=True, bits=4)
            # These are the CLI's default machine-selected paths. No cap sweep is performed.
            apply_small_m(
                target,
                drafter,
                target_repo=target_repo,
                drafter_repo=drafter_repo,
                verbose=True,
            )
            selected_cap = static_cap(
                target,
                drafter,
                mode="dspark",
                target_repo=target_repo,
                drafter_repo=drafter_repo,
                verbose=True,
            )
            lookup = lookup_drafts_default(target_repo)
        apply_wide_gemm(
            target,
            drafter,
            target_repo=target_repo,
            verbose=True,
        )
        # Calibration uses temporary synthetic KV/verification buffers. Release the allocator
        # cache before the real long-context workload; this is cleanup, not a decoding knob.
        mx.synchronize()
        mx.clear_cache()
        monitor.check()
        load_memory = {**mlx_snapshot(), "process_max_rss_mb": rss_snapshot(), **monitor.snapshot()}
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")
        prompt_ids = encode_prompt(tokenizer, prompt, use_chat=True)
        context_limit = model_context_limit(target)
        print(
            json.dumps(
                {
                    "status": "loaded",
                    "target": target_repo,
                    "drafter": drafter_repo,
                    "mode": resolved_mode,
                    "selected_max_draft": selected_cap,
                    "lookup_drafts": lookup,
                    "input_tokens": len(prompt_ids),
                    "context_limit_detected": context_limit,
                    "load_memory": load_memory,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if args.check_only:
            print(json.dumps({"status": "check_only", **monitor.snapshot()}, sort_keys=True), flush=True)
            return 0

        # One same-prompt warm-up removes the first long-prefill/Metal compilation from the
        # quoted measurement. It is explicitly not included in the reported run.
        print(
            json.dumps(
                {"status": "warmup", "tokens": args.warmup_tokens, "input_tokens": len(prompt_ids)},
                sort_keys=True,
            ),
            flush=True,
        )
        warmup = run_generation(
            mode=resolved_mode,
            target=target,
            tokenizer=tokenizer,
            drafter=drafter,
            prompt_ids=prompt_ids,
            max_new_tokens=args.warmup_tokens,
            max_draft=selected_cap,
            lookup_drafts=lookup,
            monitor=monitor,
            label="warmup",
        )
        print(json.dumps({"status": "warmup_done", **warmup}, sort_keys=True), flush=True)
        mx.reset_peak_memory()
        monitor.check()
        print(
            json.dumps(
                {"status": "measured", "mode": resolved_mode, "output_tokens_target": args.max_new_tokens},
                sort_keys=True,
            ),
            flush=True,
        )
        measured = run_generation(
            mode=resolved_mode,
            target=target,
            tokenizer=tokenizer,
            drafter=drafter,
            prompt_ids=prompt_ids,
            max_new_tokens=args.max_new_tokens,
            max_draft=selected_cap,
            lookup_drafts=lookup,
            monitor=monitor,
            label="measured",
        )
        print(
            json.dumps(
                {
                    "status": "complete",
                    "target": target_repo,
                    "drafter": drafter_repo,
                    "mode": resolved_mode,
                    "selected_max_draft": selected_cap,
                    "lookup_drafts": lookup,
                    "requested_context_window": 32768,
                    "measured": measured,
                    "memory": {**monitor.snapshot(), **mlx_snapshot(), "process_max_rss_mb": rss_snapshot()},
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "aborted_or_failed",
                    "target": target_repo,
                    "drafter": drafter_repo,
                    "error": f"{type(error).__name__}: {error}",
                    "memory": {**monitor.snapshot(), **mlx_snapshot(), "process_max_rss_mb": rss_snapshot()},
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 2
    finally:
        monitor.close()


if __name__ == "__main__":
    raise SystemExit(main())
