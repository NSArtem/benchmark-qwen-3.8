#!/usr/bin/env python3
"""Prepare a deterministic, chat-formatted ~16K Qwen3.8 prompt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


MODEL = "lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly"
TARGET_TOKENS = 16_384


def tokenizer_for_model(model: str):
    path = snapshot_download(
        model,
        local_files_only=True,
        allow_patterns=["tokenizer*", "chat_template.jinja", "config.json", "generation_config.json"],
    )
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def token_ids(tokenizer, raw_prompt: str) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": raw_prompt}],
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
    return [int(value) for value in encoded]


def raw_token_count(tokenizer, raw_prompt: str) -> int:
    return len(tokenizer.encode(raw_prompt, add_special_tokens=False))


def compact_realistic_source(source: str) -> str:
    """Remove only redundant summary material from the existing coding prompt.

    The source accidentally contains two adjacent Task headings.  The
    Operational observations section repeats facts covered by the included
    code and the later request traces; omitting that summary keeps the task
    and realistic repository excerpts intact while getting close to 16K.
    """

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
    source = source.replace(repeated_contract_note, "", 1)
    return source


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="coding_prompt.md")
    parser.add_argument("--output", default="qwen38-3bit-mlx-rescue/prompt-16k.txt")
    parser.add_argument("--manifest", default="qwen38-3bit-mlx-rescue/prompt-manifest.json")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--target", type=int, default=TARGET_TOKENS)
    args = parser.parse_args()

    source_path = Path(args.source)
    output_path = Path(args.output)
    manifest_path = Path(args.manifest)
    source = compact_realistic_source(source_path.read_text(encoding="utf-8"))
    tokenizer = tokenizer_for_model(args.model)
    final_ids = token_ids(tokenizer, source)
    if abs(len(final_ids) - args.target) > 64:
        raise SystemExit(
            f"compact prompt is {len(final_ids)} tokens, outside target tolerance around {args.target}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(source, encoding="utf-8")
    manifest = {
        "model": args.model,
        "source": str(source_path),
        "chat_template": "tokenizer.apply_chat_template; add_generation_prompt=true; enable_thinking=false",
        "target_tokens": args.target,
        "raw_input_tokens": raw_token_count(tokenizer, source),
        "final_chat_tokens": len(final_ids),
        "sha256": sha256(output_path),
        "path": str(output_path),
        "prompt_edit": "removed the duplicate Task heading and the redundant Operational observations summary; retained the repository excerpts, later supplementary excerpts, traces, and full Task",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
