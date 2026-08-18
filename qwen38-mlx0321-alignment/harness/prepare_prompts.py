#!/usr/bin/env python3
"""Build the exact adjacent prompt family for the MLX 0.32.1 test.

The source material and final task are inherited from the previous threshold
experiment.  Only the amount of source context before the byte-identical task
is changed.  Counts are measured after the same chat template used by the
inference runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


TARGETS = (512, 513, 640, 641, 768, 769)


def compact_source(source: str) -> str:
    """Apply the established Lantern-prompt reduction from the prior tests."""

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


def chat_ids(tokenizer: Any, content: str) -> list[int]:
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


def split_body_and_task(source: str) -> tuple[str, str]:
    marker = "\n## Task\n"
    task_start = source.rfind(marker)
    if task_start < 0:
        raise ValueError("canonical source has no final Task section")
    body = source[:task_start]
    task = "## Task\n" + source[task_start + len(marker) :]
    return body, task


def make_prompt(body: str, task: str, cut: int) -> str:
    prefix = body[:cut].rstrip()
    return (prefix + "\n\n" if prefix else "") + task


def exact_prompt(body: str, task: str, tokenizer: Any, target: int) -> tuple[str, list[int], int]:
    """Find a character cut whose final chat-formatted length is exactly target."""

    count_cache: dict[int, int] = {}

    def count(cut: int) -> int:
        if cut not in count_cache:
            count_cache[cut] = len(chat_ids(tokenizer, make_prompt(body, task, cut)))
        return count_cache[cut]

    # Find the largest complete source-line prefix below the target.  Searching
    # only the following natural-language line keeps adjacent prompts close.
    offsets = [0]
    for line in body.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    line_counts = [(count(cut), cut) for cut in offsets]
    under = [(tokens, cut) for tokens, cut in line_counts if tokens <= target]
    if not under:
        raise ValueError(f"the final task itself exceeds target {target}")
    _, line_start = max(under, key=lambda item: item[1])
    following = next((cut for cut in offsets if cut > line_start), len(body))

    # Token counts are monotonic enough for a binary search, but character cuts
    # can share tokenizer merges.  Search a bounded neighborhood around the
    # boundary instead of retokenizing every character in a long source line.
    lo, hi = line_start, following
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count(mid) <= target:
            lo = mid
        else:
            hi = mid - 1
    boundary = lo
    candidates: list[tuple[int, int]] = []
    search_lo = max(line_start, boundary - 96)
    search_hi = min(following, boundary + 96)
    for cut in range(search_lo, search_hi + 1):
        actual = count(cut)
        if actual == target:
            candidates.append((cut, actual))
    if not candidates:
        # A long token can skip a count at the line boundary.  Search a modest
        # neighborhood around the closest source-line prefix before failing.
        nearest = min(line_counts, key=lambda item: abs(item[0] - target))
        center = nearest[1]
        lo = max(0, center - 2048)
        hi = min(len(body), center + 2048)
        for cut in range(lo, hi + 1):
            actual = count(cut)
            if actual == target:
                candidates.append((cut, actual))
    if not candidates:
        closest = min(line_counts, key=lambda item: abs(item[0] - target))
        raise ValueError(
            f"could not construct exact target {target}; closest line prefix "
            f"is {closest[0]} tokens at cut {closest[1]}"
        )

    # The latest exact cut preserves the most realistic context while keeping
    # all shorter prompts as prefixes of the same source material.
    cut, actual = max(candidates, key=lambda item: item[0])
    prompt = make_prompt(body, task, cut)
    ids = chat_ids(tokenizer, prompt)
    if len(ids) != target:
        raise AssertionError((target, len(ids), cut))
    return prompt, ids, cut


def diff_summary(left: list[int], right: list[int]) -> dict[str, Any]:
    common = 0
    for a, b in zip(left, right):
        if a != b:
            break
        common += 1
    suffix = 0
    while suffix < len(left) - common and suffix < len(right) - common:
        if left[-1 - suffix] != right[-1 - suffix]:
            break
        suffix += 1
    differing = [i for i, (a, b) in enumerate(zip(left, right)) if a != b]
    insertion_index = None
    inserted_token_ids: list[int] = []
    if len(right) == len(left) + 1:
        for index in range(len(right)):
            if right[:index] == left[:index] and right[index + 1 :] == left[index:]:
                insertion_index = index
                inserted_token_ids = right[index : index + 1]
                break
    return {
        "left_tokens": len(left),
        "right_tokens": len(right),
        "common_prefix_tokens": common,
        "common_suffix_tokens": suffix,
        "first_differing_index": differing[0] if differing else None,
        "last_differing_index": differing[-1] if differing else None,
        "differing_positions_in_overlap": len(differing),
        "length_delta": len(right) - len(left),
        "one_token_insertion_index": insertion_index,
        "inserted_token_ids": inserted_token_ids,
        "suffix_matches_after_insertion": insertion_index is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="coding_prompt.md")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-root", default="qwen38-mlx0321-alignment")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    source = compact_source(Path(args.source).read_text(encoding="utf-8"))
    body, task = split_body_and_task(source)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)

    root = Path(args.output_root)
    prompt_dir = root / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    token_map: dict[int, list[int]] = {}
    task_hash = hashlib.sha256(task.encode("utf-8")).hexdigest()

    for target in TARGETS:
        prompt, ids, cut = exact_prompt(body, task, tokenizer, target)
        path = prompt_dir / f"prompt-{target}.txt"
        path.write_text(prompt, encoding="utf-8")
        token_path = prompt_dir / f"prompt-{target}.tokens.json"
        record = {
            "target_prompt_tokens": target,
            "actual_prompt_tokens": len(ids),
            "main_prefill_tokens": len(ids) - 1,
            "raw_text_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "raw_text_length": len(prompt),
            "actual_raw_tokens": len(tokenizer.encode(prompt, add_special_tokens=False)),
            "first_16_token_ids": ids[:16],
            "last_16_token_ids": ids[-16:],
            "prompt_file": str(path),
            "token_file": str(token_path),
            "source_cut_character": cut,
            "task_sha256": task_hash,
            "chat_template": "tokenizer.apply_chat_template; add_generation_prompt=true; enable_thinking=false",
        }
        token_path.write_text(
            json.dumps(
                {
                    "target_prompt_tokens": target,
                    "actual_prompt_tokens": len(ids),
                    "main_prefill_tokens": len(ids) - 1,
                    "tokens": ids,
                    "raw_text_sha256": record["raw_text_sha256"],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        records.append(record)
        token_map[target] = ids

    pairs = {}
    for left, right in ((512, 513), (640, 641), (768, 769)):
        pairs[f"{left}_vs_{right}"] = diff_summary(token_map[left], token_map[right])

    manifest = {
        "model": "lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly",
        "model_path": args.model_path,
        "source": args.source,
        "source_transformation": "same compact Lantern source and byte-identical final Task used by the prior threshold experiment",
        "task_sha256": task_hash,
        "targets": list(TARGETS),
        "records": records,
        "adjacent_pair_token_diffs": pairs,
    }
    (root / "prompt-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
