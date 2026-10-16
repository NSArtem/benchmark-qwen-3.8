"""A7: per-domain NLL, top-1 agreement, and kl_top32 against the BF16 reference."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .common import guarded_module
from .data import DOMAINS, domain_rows, load_split
from .models import add_model_args, load_any, model_args, write_json
from .reference import REFERENCE_DIR, RESULTS

CHUNK = 256


def score_against_reference(hidden, lm_head, tokens: np.ndarray, ref_ids: np.ndarray):
    """Per-position NLL, argmax, and log-probs at the reference top-32 ids."""
    import mlx.core as mx

    positions = tokens.shape[-1] - 1
    nll, top1, at_ref = [], [], []
    for start in range(0, positions, CHUNK):
        stop = min(start + CHUNK, positions)
        logits = lm_head(hidden[:, start:stop, :]).astype(mx.float32)
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        targets = mx.array(tokens[None, start + 1 : stop + 1], dtype=mx.int32)
        picked = mx.take_along_axis(logp, targets[..., None], axis=-1).squeeze(-1)
        arg = mx.argmax(logp, axis=-1)
        ids = mx.array(ref_ids[None, start:stop], dtype=mx.int32)
        gathered = mx.take_along_axis(logp, ids, axis=-1)
        mx.eval(picked, arg, gathered)
        nll.append(-np.asarray(picked)[0])
        top1.append(np.asarray(arg)[0].astype(np.int32))
        at_ref.append(np.asarray(gathered)[0])
        del logits, logp
    return np.concatenate(nll), np.concatenate(top1), np.concatenate(at_ref)


def metrics(nll, top1, at_ref, ref_nll, ref_top1, ref_ids_logp) -> dict:
    p = np.exp(ref_ids_logp.astype(np.float64))
    kl = np.sum(p * (ref_ids_logp.astype(np.float64) - at_ref.astype(np.float64)), axis=-1)
    nll_mean = float(np.mean(nll, dtype=np.float64))
    return {
        "nll_mean": nll_mean,
        "ppl": float(np.exp(nll_mean)),
        "delta_nll": nll_mean - float(np.mean(ref_nll, dtype=np.float64)),
        "top1_agree": float(np.mean(top1 == ref_top1)),
        "kl_top32": float(np.mean(kl)),
        "positions": int(nll.size),
    }


def report(nll, top1, at_ref, reference) -> dict:
    out = {"all": metrics(nll, top1, at_ref, *reference)}
    for domain, rows in domain_rows("eval").items():
        out[domain] = metrics(nll[rows], top1[rows], at_ref[rows], *(r[rows] for r in reference))
    return out


def load_reference(directory: Path = REFERENCE_DIR):
    return (
        np.load(directory / "nll.npy"),
        np.load(directory / "top1.npy"),
        np.load(directory / "top32_logp.npy"),
    ), np.load(directory / "top32_ids.npy")


def _child(args) -> None:
    import mlx.core as mx

    started = time.monotonic()
    tokens = load_split("eval")
    reference, ref_ids = load_reference()
    model, _tokenizer = load_any(args.loader, args.model, args.rotate_dtype)
    mx.reset_peak_memory()
    text = model.language_model
    rows = [[], [], []]
    for i, sequence in enumerate(tokens):
        hidden = text.model(mx.array(sequence[None], dtype=mx.int32))
        mx.eval(hidden)
        for store, value in zip(rows, score_against_reference(hidden, text.lm_head, sequence, ref_ids[i])):
            store.append(value)
        del hidden
        mx.clear_cache()
    nll, top1, at_ref = (np.stack(r) for r in rows)
    out_dir = RESULTS / args.exp
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "nll.npy", nll.astype(np.float32))
    result = report(nll, top1, at_ref, reference)
    result["_run"] = {"wall_s": time.monotonic() - started, "peak_mlx_gib": mx.get_peak_memory() / 2**30}
    write_json(out_dir / "eval.json", result)
    print(json.dumps(result["all"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_args(parser)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if args.child:
        _child(args)
        return
    result = guarded_module(
        "stage10.evaluate", ["--child", *model_args(args)], RESULTS / "logs" / f"{args.exp}_eval.log"
    )
    write_json(RESULTS / args.exp / "eval_run.json", result)
    print(json.dumps(result, indent=2))
    if result["exit_code"] != 0 or result["safety_stop"]:
        sys.exit(result["exit_code"] or 1)


if __name__ == "__main__":
    main()
