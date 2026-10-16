"""A4: BF16 reference log-probs over the 48 eval sequences (= E00)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .common import guarded_module
from .data import load_split


REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS = REPO_ROOT / "benchmarks" / "results" / "stage10"
REFERENCE_DIR = RESULTS / "reference"
TOP_K = 32


def _child(output: Path, sequence_batch_size: int) -> None:
    import mlx.core as mx

    from .layerwise import forward_source_hidden, score_hidden, source_lm_head
    from .source import SOURCE, build_model, read_source

    started = time.monotonic()
    mx.reset_peak_memory()
    tokens = load_split("eval")
    source_plan = read_source(SOURCE)
    sanitizer_model = build_model(source_plan["config"])
    hidden = forward_source_hidden(
        mx.array(tokens),
        source_plan=source_plan,
        sanitizer_model=sanitizer_model,
        sequence_batch_size=sequence_batch_size,
    )
    head = source_lm_head(source_plan, sanitizer_model)
    pieces: dict[str, list[np.ndarray]] = {}
    for row in range(tokens.shape[0]):
        scores = score_hidden(hidden[row : row + 1], head, tokens[row : row + 1], top_k=TOP_K)
        for key, value in scores.items():
            pieces.setdefault(key, []).append(value)
    scores = {key: np.concatenate(value, axis=0) for key, value in pieces.items()}
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "nll.npy", (-scores["token_logp"]).astype(np.float32))
    np.save(output / "top1.npy", scores["top1"].astype(np.int32))
    np.save(output / "top32_ids.npy", scores["topk_ids"].astype(np.int32))
    np.save(output / "top32_logp.npy", scores["topk_logp"].astype(np.float32))
    info = {
        "shape": list(scores["token_logp"].shape),
        "wall_s": time.monotonic() - started,
        "peak_mlx_gib": mx.get_peak_memory() / 2**30,
        "sequence_batch_size": sequence_batch_size,
    }
    (output / "run.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(info))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REFERENCE_DIR)
    parser.add_argument("--sequence-batch-size", type=int, default=4)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if args.child:
        _child(args.output, args.sequence_batch_size)
        return
    result = guarded_module(
        "stage10.reference",
        ["--child", "--output", str(args.output), "--sequence-batch-size", str(args.sequence_batch_size)],
        RESULTS / "logs" / "reference.log",
    )
    print(json.dumps(result, indent=2))
    if result["exit_code"] != 0 or result["safety_stop"]:
        sys.exit(result["exit_code"] or 1)


if __name__ == "__main__":
    main()
