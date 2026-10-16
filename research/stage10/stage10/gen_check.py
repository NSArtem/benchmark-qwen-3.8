"""A9: greedy generation check on five fixed prompts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter

from .common import guarded_module
from .models import add_model_args, load_any, model_args, write_json
from .reference import RESULTS

MAX_TOKENS = 1024
PROMPTS = {
    "P1": "Write a Python function merge_intervals(intervals) that merges overlapping intervals. Include a docstring and three doctest examples.",
    "P2": "A train leaves at 14:35 and arrives at 18:10 the same day. It stops for 7 minutes at each of 4 stations. How many minutes is it actually moving? Explain step by step.",
    "P3": "Describe the water cycle for a 12-year-old in about 200 words.",
    "P4": "Return a JSON object describing three fictional books with fields title, author, year, genres (array). Output only JSON.",
    "P5": "Объясни простыми словами, что такое квантование нейросетей и зачем оно нужно. Ответ — 5–7 предложений.",
}


def ngram_stats(tokens: list[int], n: int = 4) -> tuple[float, int]:
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not grams:
        return 0.0, 0
    counts = Counter(grams)
    return len(counts) / len(grams), max(counts.values())


def json_valid(text: str) -> bool:
    stripped = text.strip()
    stripped = re.sub(r"```[A-Za-z0-9_-]*", "", stripped).strip()
    try:
        json.loads(stripped)
    except ValueError:
        return False
    return True


def score(prompt_id: str, text: str, tokens: list[int], hit_eos: bool) -> dict:
    distinct4, max4 = ngram_stats(tokens)
    row = {
        "id": prompt_id,
        "n_tokens": len(tokens),
        "hit_eos": hit_eos,
        "distinct4": distinct4,
        "max4": max4,
    }
    if prompt_id == "P2":
        row["p2_has_187"] = "187" in text
    if prompt_id == "P4":
        row["p4_json_valid"] = json_valid(text)
    return row


def _child(args) -> None:
    import mlx.core as mx
    from mlx_lm.generate import generate_step

    model, tokenizer = load_any(args.loader, args.model, args.rotate_dtype)
    eos = set(tokenizer.eos_token_ids)
    rows = []
    for prompt_id, content in PROMPTS.items():
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=True,
        )
        if isinstance(prompt, dict):
            prompt = prompt["input_ids"]
        tokens: list[int] = []
        hit_eos = False
        for token, _ in generate_step(
            mx.array(prompt, dtype=mx.int32),
            model,
            max_tokens=MAX_TOKENS,
            sampler=lambda logprobs: mx.argmax(logprobs, axis=-1),
        ):
            token = int(token.item() if hasattr(token, "item") else token)
            if token in eos:
                hit_eos = True
                break
            tokens.append(token)
        text = tokenizer.decode(tokens)
        row = score(prompt_id, text, tokens, hit_eos)
        row["text"] = text
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "text"}, ensure_ascii=False))
        mx.clear_cache()
    write_json(RESULTS / args.exp / "gen_check.json", rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_args(parser)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if args.child:
        _child(args)
        return
    result = guarded_module(
        "stage10.gen_check", ["--child", *model_args(args)], RESULTS / "logs" / f"{args.exp}_gen.log"
    )
    write_json(RESULTS / args.exp / "gen_run.json", result)
    print(json.dumps(result, indent=2))
    if result["exit_code"] != 0 or result["safety_stop"]:
        sys.exit(result["exit_code"] or 1)


if __name__ == "__main__":
    main()
