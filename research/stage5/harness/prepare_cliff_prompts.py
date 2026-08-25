#!/usr/bin/env python3
"""Prepare one frozen coding prompt family for the Stage 5 cliff study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


TARGETS = (8192, 10240, 12288, 14336, 16727)


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
    candidates = [
        cut
        for cut in range(max(0, center - 128), min(len(body), center + 128) + 1)
        if count(cut) == target
    ]
    cut = max(candidates) if candidates else center
    prompt = make_prompt(body, task, cut)
    ids = chat_ids(tokenizer, prompt)
    return prompt, ids, cut


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="qwen38-mtp-smoke/coding-prompt.txt")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", default="research/stage5/artifacts/cliff/prompts")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    source_path = Path(args.source)
    source_bytes = source_path.read_bytes()
    source = source_bytes.decode("utf-8")
    body, task = split_body_and_task(source)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
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
            "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "source_cut_character": cut,
            "chat_template": "apply_chat_template(add_generation_prompt=true, enable_thinking=false)",
        }
        token_path.write_text(
            json.dumps({**record, "tokens": ids}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        records.append(record)

    manifest = {
        "source": str(source_path),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "model_path": args.model_path,
        "targets": list(TARGETS),
        "records": records,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
