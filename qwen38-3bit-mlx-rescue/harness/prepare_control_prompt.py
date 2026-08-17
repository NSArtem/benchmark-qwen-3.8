#!/usr/bin/env python3
"""Prepare an approximately 8K realistic control prompt for the KV8 check."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


MODEL = "lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly"
TARGET_TOKENS = 8_192


def tokenizer_for_model(model: str):
    path = snapshot_download(
        model,
        local_files_only=True,
        allow_patterns=["tokenizer*", "chat_template.jinja", "config.json", "generation_config.json"],
    )
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def chat_tokens(tokenizer, text: str) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    if encoded and isinstance(encoded[0], (list, tuple)):
        encoded = encoded[0]
    return [int(token) for token in encoded]


def compact_source(source: str) -> str:
    source = source.replace("\n## Task\n## Task\n", "\n## Task\n", 1)
    start_marker = "\n## Operational observations\n"
    end_marker = "\n## Additional relevant excerpts\n"
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    source = source[:start] + source[end:]
    repeated_contract_note = """The current API has two PostgreSQL pools in the same process in one deployment: the request
pool and a background pool used by the outbox publisher. They share the database but not a
transaction. An outbox publish can be repeated after a process crash, and the Redis subscriber
can receive the same payload twice. A sequence number is scoped to one tenant and job, not
globally to the stream.

"""
    return source.replace(repeated_contract_note, "", 1)


def raw_count(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def make_prompt(source: str, tokenizer: Any, target: int) -> str:
    sections = source.split("\n## ")
    if len(sections) < 3:
        raise ValueError("expected sectioned coding prompt")
    sections = [sections[0]] + ["## " + section for section in sections[1:]]
    tail_index = next(
        index for index, section in enumerate(sections) if section.startswith("## API contract notes")
    )
    tail = "\n".join(sections[tail_index:]).rstrip() + "\n"
    candidates: list[str] = []
    for end in range(1, tail_index + 1):
        candidate = "\n".join(sections[:end]).rstrip() + "\n\n" + tail
        candidates.append(candidate)
    valid = [candidate for candidate in candidates if len(chat_tokens(tokenizer, candidate)) <= target]
    if not valid:
        raise ValueError("could not fit the control prompt")
    prompt = max(valid, key=lambda value: len(chat_tokens(tokenizer, value)))
    if abs(len(chat_tokens(tokenizer, prompt)) - target) > 512:
        raise ValueError(f"control prompt is not close enough to target: {len(chat_tokens(tokenizer, prompt))}")
    return prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="coding_prompt.md")
    parser.add_argument("--output", default="qwen38-3bit-mlx-rescue/prompt-8k-control.txt")
    parser.add_argument("--manifest", default="qwen38-3bit-mlx-rescue/prompt-8k-control-manifest.json")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--target", type=int, default=TARGET_TOKENS)
    args = parser.parse_args()

    source = compact_source(Path(args.source).read_text(encoding="utf-8"))
    tokenizer = tokenizer_for_model(args.model)
    prompt = make_prompt(source, tokenizer, args.target)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(prompt, encoding="utf-8")
    manifest = {
        "model": args.model,
        "source": args.source,
        "chat_template": "tokenizer.apply_chat_template; add_generation_prompt=true; enable_thinking=false",
        "target_tokens": args.target,
        "raw_input_tokens": raw_count(tokenizer, prompt),
        "final_chat_tokens": len(chat_tokens(tokenizer, prompt)),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "path": str(output),
        "purpose": "same realistic prompt for normal-KV and KV8 coherence control",
    }
    Path(args.manifest).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
