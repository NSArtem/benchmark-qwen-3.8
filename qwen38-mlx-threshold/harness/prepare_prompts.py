#!/usr/bin/env python
"""Build realistic Qwen3.8 chat prompts at controlled final token lengths."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


TARGETS = (512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608)
MODEL = "lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly"


def compact_source(source: str) -> str:
    """Reuse the existing realistic source while removing only redundancy."""

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


def ids_from_chat(tokenizer: Any, content: str) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    elif hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    if encoded and isinstance(encoded[0], (list, tuple)):
        encoded = encoded[0]
    return [int(value) for value in encoded]


def raw_count(tokenizer: Any, content: str) -> int:
    return len(tokenizer.encode(content, add_special_tokens=False))


def choose_prompt(body: str, task: str, tokenizer: Any, target: int) -> tuple[str, int]:
    """Choose a line-preserving prefix, then fine-tune only the final source line."""

    lines = body.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))

    def make(cut: int) -> str:
        prefix = body[:cut].rstrip()
        return (prefix + "\n\n" if prefix else "") + task

    def count(cut: int) -> int:
        return len(ids_from_chat(tokenizer, make(cut)))

    candidates = [(count(cut), cut) for cut in offsets]
    under = [(tokens, cut) for tokens, cut in candidates if tokens <= target]
    if not under:
        raise ValueError(f"final task itself exceeds target {target}")
    _, best_cut = max(under, key=lambda item: item[0])

    next_cut = len(body)
    for cut in offsets:
        if cut > best_cut:
            next_cut = cut
            break

    # Binary search the first source line after the best complete-line prefix.
    lo, hi = best_cut, next_cut
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count(mid) <= target:
            lo = mid
        else:
            hi = mid - 1
    fine_cut = lo

    # Check nearby cuts because tokenization is not strictly one token per byte.
    nearby = range(max(best_cut, fine_cut - 32), min(next_cut, fine_cut + 32) + 1)
    best = min(
        ((abs(count(cut) - target), count(cut), cut) for cut in nearby),
        key=lambda item: (item[0], -item[1]),
    )
    _, actual, cut = best
    prompt = make(cut)
    if abs(actual - target) > 16:
        raise ValueError(f"target {target} produced {actual} tokens")
    return prompt, actual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="coding_prompt.md")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-root", default="qwen38-mlx-threshold")
    parser.add_argument("--extra-targets", nargs="*", type=int, default=[])
    args = parser.parse_args()

    from transformers import AutoTokenizer

    source = compact_source(Path(args.source).read_text(encoding="utf-8"))
    marker = "\n## Task\n"
    task_start = source.rfind(marker)
    if task_start < 0:
        raise ValueError("canonical source has no final Task section")
    body = source[:task_start]
    task = "## Task\n" + source[task_start + len(marker) :]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)

    root = Path(args.output_root)
    prompt_dir = root / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    records = []
    task_hash = hashlib.sha256(task.encode("utf-8")).hexdigest()
    targets = tuple(sorted(set(TARGETS).union(args.extra_targets)))
    for target in targets:
        prompt, actual = choose_prompt(body, task, tokenizer, target)
        path = prompt_dir / f"{target:04d}.txt"
        path.write_text(prompt, encoding="utf-8")
        record = {
            "target_prompt_tokens": target,
            "actual_raw_tokens": raw_count(tokenizer, prompt),
            "actual_chat_tokens": actual,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "path": str(path),
            "task_sha256": task_hash,
            "chat_template": "tokenizer.apply_chat_template; add_generation_prompt=true; enable_thinking=false",
        }
        records.append(record)

    manifest = {
        "model": MODEL,
        "model_path": args.model_path,
        "source": args.source,
        "source_transformation": "existing realistic Lantern material; redundant operational summary and duplicate Task heading removed",
        "task_sha256": task_hash,
        "records": records,
    }
    (root / "prompt-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
