"""A8: speed/memory ladder, one context length per guarded process."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .common import guarded_module
from .data import load_split
from .models import add_model_args, load_any, model_args, write_json
from .reference import RESULTS

BASE_CONTEXTS = (4096, 8192, 16384)
LONG_CONTEXTS = (24576, 32768)
LONG_MAX_GIB = 9.0
MAX_TOKENS = 128
PREFILL_STEP = 512


def ladder_prompt(n: int) -> np.ndarray:
    data = Path(__file__).resolve().parents[3] / "benchmarks" / "results" / "stage10" / "data"
    stream = np.concatenate(
        [np.load(data / "code_eval.npy").reshape(-1), np.load(data / "en_eval.npy").reshape(-1)]
    ).astype(np.int32)
    if n > stream.size:
        raise ValueError(f"ladder context {n} exceeds {stream.size} available tokens")
    return stream[:n]


def contexts_for(weights_gib: float) -> list[int]:
    return list(BASE_CONTEXTS) + (list(LONG_CONTEXTS) if weights_gib <= LONG_MAX_GIB else [])


def _child(args) -> None:
    import mlx.core as mx
    from mlx_lm.generate import generate_step

    model, _tokenizer = load_any(args.loader, args.model, args.rotate_dtype)
    prompt = mx.array(ladder_prompt(args.ctx))
    mx.reset_peak_memory()
    marks: dict[str, float] = {}

    def progress(done: int, total: int) -> None:
        if done == total - 1 or total <= 1:
            marks["prefill_done"] = time.perf_counter()

    started = time.perf_counter()
    times = []
    tokens = []
    for token, _logprobs in generate_step(
        prompt,
        model,
        max_tokens=MAX_TOKENS,
        sampler=lambda logprobs: mx.argmax(logprobs, axis=-1),
        prefill_step_size=PREFILL_STEP,
        prompt_progress_callback=progress,
    ):
        tokens.append(int(token.item() if hasattr(token, "item") else token))
        times.append(time.perf_counter())
    prefill_s = marks.get("prefill_done", times[0]) - started
    ttft_s = times[0] - started
    decode_s = times[-1] - times[0]
    result = {
        "ctx": args.ctx,
        "prefill_s": prefill_s,
        "prefill_tok_s": args.ctx / prefill_s,
        "ttft_s": ttft_s,
        "decode_tok_s": (len(times) - 1) / decode_s if decode_s > 0 else None,
        "generated_tokens": len(tokens),
        "peak_mlx_gib": mx.get_peak_memory() / 2**30,
    }
    write_json(RESULTS / args.exp / f"ladder_{args.ctx}_child.json", result)
    print(json.dumps(result))


def run_ladder(args) -> list[dict]:
    rows = []
    for ctx in contexts_for(args.weights_gib):
        child_path = RESULTS / args.exp / f"ladder_{ctx}_child.json"
        child_path.unlink(missing_ok=True)
        run = guarded_module(
            "stage10.ladder",
            ["--child", "--ctx", str(ctx), *model_args(args)],
            RESULTS / "logs" / f"{args.exp}_ladder_{ctx}.log",
        )
        row = {"ctx": ctx}
        if child_path.is_file():
            row.update(json.loads(child_path.read_text()))
        row.update(
            free_pct_before=run["free_pct_before"],
            swap_mib_before=run["swap_mib_before"],
            free_pct_after=run["free_pct_after"],
            swap_mib_after=run["swap_mib_after"],
            free_pct_min=run["free_pct_min"],
            safety_stop=run["safety_stop"],
            exit_code=run["exit_code"],
            wall_s=run["wall_s"],
        )
        rows.append(row)
        write_json(RESULTS / args.exp / "ladder.json", rows)
        print(json.dumps(row))
        if run["safety_stop"] or run["exit_code"] != 0:
            break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_args(parser)
    parser.add_argument("--ctx", type=int)
    parser.add_argument("--weights-gib", type=float, default=None)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if args.child:
        _child(args)
        return
    if args.weights_gib is None:
        parser.error("--weights-gib is required for the ladder driver")
    rows = run_ladder(args)
    if any(row["exit_code"] != 0 and not row["safety_stop"] for row in rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
