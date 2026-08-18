#!/usr/bin/env python3
"""Prepare exact-token realistic Lantern prompts for the cold context ladder."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


TARGETS = (1024, 2048, 4096, 8192, 12288)


def compact_source(source: str) -> str:
    source = source.replace("\n## Task\n## Task\n", "\n## Task\n", 1)
    start_marker = "\n## Operational observations\n"
    end_marker = "\n## Additional relevant excerpts\n"
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    source = source[:start] + source[end:]
    repeated = """The current API has two PostgreSQL pools in the same process in one deployment: the request
pool and a background pool used by the outbox publisher. They share the database but not a
transaction. An outbox publish can be repeated after a process crash, and the Redis subscriber
can receive the same payload twice. A sequence number is scoped to one tenant and job, not
globally to the stream.

"""
    return source.replace(repeated, "", 1)


def normalize_ids(encoded: Any) -> list[int]:
    if isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    elif hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    if encoded and isinstance(encoded[0], (list, tuple)):
        encoded = encoded[0]
    return [int(value) for value in encoded]


def chat_ids(tokenizer: Any, content: str) -> list[int]:
    return normalize_ids(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )


def split_body_and_task(source: str) -> tuple[str, str]:
    marker = "\n## Task\n"
    task_start = source.rfind(marker)
    if task_start < 0:
        raise ValueError("source has no final Task section")
    return source[:task_start], "## Task\n" + source[task_start + len(marker) :]


def make_prompt(body: str, task: str, cut: int) -> str:
    prefix = body[:cut].rstrip()
    return (prefix + "\n\n" if prefix else "") + task


def exact_prompt(tokenizer: Any, body: str, task: str, target: int) -> tuple[str, list[int], int]:
    cache: dict[int, int] = {}

    def count(cut: int) -> int:
        if cut not in cache:
            cache[cut] = len(chat_ids(tokenizer, make_prompt(body, task, cut)))
        return cache[cut]

    if count(0) > target:
        raise ValueError(f"final task exceeds target {target}")
    lo, hi = 0, len(body)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count(mid) <= target:
            lo = mid
        else:
            hi = mid - 1
    center = lo
    candidates = []
    for cut in range(max(0, center - 64), min(len(body), center + 64) + 1):
        if count(cut) == target:
            candidates.append(cut)
    if candidates:
        cut = max(candidates)
    else:
        # Tokenization can skip a requested count at a character boundary.
        # Keep the closest natural prefix and record its actual count rather
        # than padding the prompt with artificial text.
        cut = center
    prompt = make_prompt(body, task, cut)
    ids = chat_ids(tokenizer, prompt)
    return prompt, ids, cut


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="coding_prompt.md")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", default="research/context-ladder/prompts")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    source_path = Path(args.source)
    source = compact_source(source_path.read_text(encoding="utf-8"))
    body, task = split_body_and_task(source)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for target in TARGETS:
        prompt, ids, cut = exact_prompt(tokenizer, body, task, target)
        prompt_path = output_dir / f"prompt-{target}.txt"
        token_path = output_dir / f"prompt-{target}.tokens.json"
        prompt_path.write_text(prompt, encoding="utf-8")
        record = {
            "target_prompt_tokens": target,
            "actual_prompt_tokens": len(ids),
            "main_prefill_tokens": len(ids) - 1,
            "prompt_file": str(prompt_path),
            "token_file": str(token_path),
            "prompt_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
            "raw_prompt_tokens": len(normalize_ids(tokenizer.encode(prompt, add_special_tokens=False))),
            "raw_text_length": len(prompt),
            "source_cut_character": cut,
            "chat_template": "apply_chat_template(add_generation_prompt=true, enable_thinking=false)",
        }
        token_path.write_text(
            json.dumps({**record, "tokens": ids}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        records.append(record)
    manifest = {
        "model_path": args.model_path,
        "source": str(source_path),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "source_transformation": "same compact Lantern reduction as qwen38-mlx0321-alignment",
        "targets": list(TARGETS),
        "records": records,
    }
    manifest_path = output_dir.parent / "prompt-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
