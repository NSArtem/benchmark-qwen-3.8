#!/usr/bin/env python3
"""Run one llama.cpp case and persist timing/resource metadata immediately."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent


def resolve_input_path(value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve()
    for base in (REPO_ROOT, ROOT):
        resolved = (base / candidate).resolve()
        if resolved.exists():
            return resolved
    return (REPO_ROOT / candidate).resolve()


def repo_relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def swap_used_mb() -> float | None:
    try:
        text = subprocess.run(
            ["sysctl", "vm.swapusage"],
            check=False,
            capture_output=True,
            text=True,
        ).stdout
    except OSError:
        return None
    match = re.search(r"used\s*=\s*([0-9.]+)M", text)
    return float(match.group(1)) if match else None


def rss_mb(pid: int) -> float | None:
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "rss="],
            check=False,
            capture_output=True,
            text=True,
        )
        value = result.stdout.strip()
        return float(value) / 1024.0 if value else None
    except (OSError, ValueError):
        return None


def parse_timing(text: str) -> dict[str, float | int | None]:
    result: dict[str, float | int | None] = {
        "prompt_tokens": None,
        "output_tokens": None,
        "prompt_seconds": None,
        "prompt_tps": None,
        "generation_seconds": None,
        "generation_tps": None,
        "total_seconds": None,
    }
    prompt = re.search(
        r"prompt eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*([0-9]+)\s*tokens[^\n]*?([0-9.]+)\s*tokens per second",
        text,
    )
    generation = re.search(
        r"(?<!prompt )eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*([0-9]+)\s*tokens[^\n]*?([0-9.]+)\s*tokens per second",
        text,
    )
    total = re.search(
        r"total time\s*=\s*([0-9.]+) ms /\s*([0-9]+) tokens",
        text,
    )
    if prompt:
        result.update(
            prompt_tokens=int(prompt.group(2)),
            prompt_seconds=float(prompt.group(1)) / 1000.0,
            prompt_tps=float(prompt.group(3)),
        )
    if generation:
        result.update(
            output_tokens=int(generation.group(2)),
            generation_seconds=float(generation.group(1)) / 1000.0,
            generation_tps=float(generation.group(3)),
        )
    if total:
        result["total_seconds"] = float(total.group(1)) / 1000.0
    return result


def parse_kv_buffer_mb(text: str) -> float | None:
    """Return the final non-zero CPU+Metal KV allocation reported by llama.cpp."""
    values = [
        float(value)
        for value in re.findall(
            r"(?:CPU|MTL0) KV buffer size\s*=\s*([0-9.]+) MiB", text
        )
        if float(value) > 0
    ]
    if not values:
        return None
    if len(values) >= 2:
        return round(sum(values[-2:]), 2)
    return round(values[-1], 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--repo", default="unsloth/Qwen3.8-27B-GGUF")
    parser.add_argument("--quant", default="UD-IQ3_XXS")
    parser.add_argument("--model-size", type=int, default=11913559104)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--ubatch", type=int, default=512)
    parser.add_argument("--kv-k", default="f16")
    parser.add_argument("--kv-v", default="f16")
    parser.add_argument("--flash-attention", default="auto")
    parser.add_argument("--fit-target", type=int, default=2048)
    parser.add_argument("--load-mode", default=None)
    parser.add_argument("--model-file", default="Qwen3.8-27B-UD-IQ3_XXS.gguf")
    args = parser.parse_args()

    prompt = resolve_input_path(args.prompt_file)
    output = ROOT / "outputs" / f"{args.name}.txt"
    log = ROOT / "logs" / f"{args.name}.log"
    meta = ROOT / "logs" / f"{args.name}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "llama-cli",
        "--offline",
        "-hf",
        f"{args.repo}:{args.quant}",
        "--no-mmproj",
        "--no-repack",
        "--fit",
        "on",
        "--fit-target",
        str(args.fit_target),
        "-c",
        str(args.context),
        "-n",
        str(args.output_tokens),
        "-b",
        str(args.batch),
        "-ub",
        str(args.ubatch),
        "-fa",
        args.flash_attention,
        "--reasoning",
        "off",
        "--reasoning-format",
        "none",
        "--temp",
        "0",
        "--seed",
        "42",
        "--top-k",
        "0",
        "--top-p",
        "1",
        "--min-p",
        "0",
        "--repeat-penalty",
        "1",
        "-ctk",
        args.kv_k,
        "-ctv",
        args.kv_v,
        "--single-turn",
        "--conversation",
        "--simple-io",
        "--perf",
        "--verbose",
        "--log-colors",
        "off",
        "-f",
        str(prompt),
        "-o",
        str(output),
    ]
    if args.load_mode is not None:
        insert_at = command.index("-c")
        command[insert_at:insert_at] = ["--load-mode", args.load_mode]

    started = time.monotonic()
    before_swap = swap_used_mb()
    peak = None
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env={**os.environ, "LLAMA_ARG_OFFLINE": "1"},
        )
        while process.poll() is None:
            sample = rss_mb(process.pid)
            if sample is not None:
                peak = sample if peak is None else max(peak, sample)
            time.sleep(0.5)
        return_code = process.returncode
    elapsed = time.monotonic() - started

    text = log.read_text(encoding="utf-8", errors="replace")
    timing = parse_timing(text)
    kv_buffer_mb = parse_kv_buffer_mb(text)
    final_window = text[-30000:]
    failure = None
    if return_code != 0:
        failure = f"llama-cli exited with code {return_code}"
    elif "Error: Compute error." in final_window:
        failure = "llama.cpp reported a compute error"
    elif not output.exists():
        failure = "llama-cli produced no output file"

    record = {
        "model_repo": args.repo,
        "model_file": args.model_file,
        "quant": args.quant,
        "model_size": args.model_size,
        "context_target": args.context,
        **timing,
        "coherent": None,
        "success": failure is None,
        "kv_type_k": args.kv_k,
        "kv_type_v": args.kv_v,
        "kv_buffer_mb": kv_buffer_mb,
        "ubatch": args.ubatch,
        "batch": args.batch,
        "flash_attention": args.flash_attention,
        "rss_peak_mb": peak,
        "swap_before_mb": before_swap,
        "swap_after_mb": swap_used_mb(),
        "failure_reason": failure,
        "output_file": repo_relative(output),
        "log_file": repo_relative(log),
        "wall_seconds": elapsed,
        "command": command,
    }
    meta.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True), flush=True)
    raise SystemExit(0 if failure is None else 1)


if __name__ == "__main__":
    main()
