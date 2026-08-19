#!/usr/bin/env python3
"""Build a protocol-harness oracle adapter from a completed chat-safe run.

The adapter contains the safe run's generated tokens as the A-side reference,
plus the one-user fragment tokens needed for the exact-token B oracle.  It does
not run inference; the subsequent protocol harness starts fresh B/C processes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.night2.harness.protocol_cache_harness import (
    chat_tokens,
    load_runtime,
    read_json,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--safe-result", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    safe = read_json(Path(args.safe_result))
    _mx, load, _generate_step, _make_prompt_cache, _make_sampler, _stream = load_runtime()
    _model, tokenizer = load(args.model_path)

    turns = []
    for source in safe.get("turns", []):
        fragment = chat_tokens(
            tokenizer,
            [{"role": "user", "content": source["query_text"]}],
        )
        turns.append(
            {
                "turn": source["turn"],
                "prompt_fragment_token_ids": fragment,
                "cache_committed_token_ids": source["generated_token_ids"],
                "generated_token_ids": source["generated_token_ids"],
                "generated_token_hash": source["generated_token_hash"],
                "decoded_text": source["decoded_text"],
                "classification": source["classification"],
                "replay_detected": source["replay_detected"],
                "previous_turn_prefix_overlap": source[
                    "previous_turn_prefix_overlap"
                ],
            }
        )
    adapter = {
        "experiment_id": f"{safe['experiment_id']}-oracle-adapter",
        "protocol_mode": "chat_safe_reference_as_mode_a_for_oracles",
        "model": safe.get("model"),
        "turns": turns,
        "source_safe_result": str(Path(args.safe_result).resolve()),
    }
    write_json(Path(args.output), adapter)
    print(Path(args.output).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

