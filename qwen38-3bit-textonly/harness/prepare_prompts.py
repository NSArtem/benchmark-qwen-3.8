#!/usr/bin/env python3
"""Create deterministic Qwen3.8 chat-template context variants."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


MODEL = "lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly"
TARGETS = {"16k": 16_384, "24k": 24_576, "32k": 32_768}


def tokenizer_for_model(model: str):
    # The model cache was populated by mlx-lm with an allow-list and therefore
    # may intentionally omit the Hub README/.gitattributes files.  Restrict
    # the local snapshot request to the tokenizer/config files we need.
    path = snapshot_download(
        model,
        local_files_only=True,
        allow_patterns=["tokenizer*", "chat_template.jinja", "config.json", "generation_config.json"],
    )
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def count_chat_tokens(tokenizer, text: str) -> int:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if isinstance(encoded, dict) or hasattr(encoded, "input_ids"):
        encoded = encoded["input_ids"]
    if encoded and isinstance(encoded[0], (list, tuple)):
        encoded = encoded[0]
    return len(encoded)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_variant(prefix: str, supplement: str, suffix: str, tokenizer, target: int):
    lines = supplement.splitlines(keepends=True)
    current = count_chat_tokens(tokenizer, prefix + suffix)
    if current > target:
        return prefix + suffix, current, 0
    # Tokenization is monotonic enough for this purpose.  A binary search keeps
    # prompt preparation bounded even when the supplement has many code lines.
    low, high = 0, len(lines)
    while low < high:
        middle = (low + high + 1) // 2
        candidate = prefix + "".join(lines[:middle]) + suffix
        if count_chat_tokens(tokenizer, candidate) <= target:
            low = middle
        else:
            high = middle - 1
    selected = "".join(lines[:low])
    current = count_chat_tokens(tokenizer, prefix + selected + suffix)
    return prefix + selected + suffix, current, len(selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="coding_prompt.md")
    parser.add_argument(
        "--supplement",
        default="qwen38-3bit-textonly/harness/supplemental_context.md",
    )
    parser.add_argument("--output-dir", default="qwen38-3bit-textonly")
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    source = Path(args.source).read_text(encoding="utf-8")
    supplement = Path(args.supplement).read_text(encoding="utf-8")
    marker = "\n## Task\n"
    if marker not in source:
        raise SystemExit(f"source does not contain {marker!r}")
    prefix, suffix = source.split(marker, 1)
    suffix = marker + suffix
    tokenizer = tokenizer_for_model(args.model)
    output_dir = Path(args.output_dir)
    variants_dir = output_dir / "prompts"
    variants_dir.mkdir(parents=True, exist_ok=True)

    canonical = prefix + supplement + suffix
    canonical_path = output_dir / "prompt.txt"
    canonical_path.write_text(canonical, encoding="utf-8")
    manifest = {
        "model": args.model,
        "tokenizer_source": args.model,
        "chat_template": "Qwen3.8 tokenizer apply_chat_template; add_generation_prompt=true; enable_thinking=false",
        "source": args.source,
        "supplement": args.supplement,
        "canonical_prompt": {
            "path": str(canonical_path),
            "sha256": sha256(canonical_path),
            "prompt_tokens": count_chat_tokens(tokenizer, canonical),
        },
        "variants": {},
    }
    for name, target in TARGETS.items():
        variant, tokens, supplement_chars = make_variant(
            prefix, supplement, suffix, tokenizer, target
        )
        path = variants_dir / f"{name}.txt"
        path.write_text(variant, encoding="utf-8")
        manifest["variants"][name] = {
            "target_tokens": target,
            "prompt_tokens": tokens,
            "supplement_chars": supplement_chars,
            "path": str(path),
            "sha256": sha256(path),
        }
        print(f"{name}: target={target} actual={tokens} supplement_chars={supplement_chars} sha256={sha256(path)}", flush=True)
    manifest_path = output_dir / "prompt-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"canonical: actual={manifest['canonical_prompt']['prompt_tokens']} sha256={manifest['canonical_prompt']['sha256']}", flush=True)


if __name__ == "__main__":
    main()
