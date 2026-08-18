#!/usr/bin/env python
"""Apply conservative human classifications to persisted full-model outputs."""

from __future__ import annotations

import json
from pathlib import Path


CLASSIFICATIONS = {
    "sanity-metal-0512": "BROKEN",
    "sanity-ops-0512": "BROKEN",
    "metal-0512-r1": "BROKEN",
    "metal-0512-r2": "BROKEN",
    "metal-0512-r3": "BROKEN",
    "metal-0512-r4": "BROKEN",
    "metal-0512-r5": "BROKEN",
    # r1 begins as normal prose but does not reach the final task in the
    # 128-token cap; the other four are visibly repetitive/blank.
    "metal-0640-r1": "DEGRADED",
    "metal-0640-r2": "BROKEN",
    "metal-0640-r3": "BROKEN",
    "metal-0640-r4": "BROKEN",
    "metal-0640-r5": "BROKEN",
    "metal-0768-r1": "BROKEN",
    "metal-0768-r2": "BROKEN",
    "metal-0768-r3": "BROKEN",
    "metal-0768-r4": "BROKEN",
    "metal-0768-r5": "BROKEN",
    "ops-0512-r1": "BROKEN",
    # This run starts with a useful answer but is truncated before it can
    # answer the task; it is not counted as a coherent completion.
    "ops-0512-r2": "DEGRADED",
    "ops-0512-r3": None,
    "ops-0640-r1": "BROKEN",
    "ops-0640-r2": "BROKEN",
    "ops-0640-r3": None,
    "ops-0768-r1": "BROKEN",
    "ops-0768-r2": None,
    "ops-0768-r3": None,
}


def main() -> None:
    path = Path("qwen38-mlx-gated-delta-ab/results.json")
    aggregate = json.loads(path.read_text(encoding="utf-8"))
    seen = set()
    for test in aggregate.get("tests", []):
        test_id = test.get("test_id")
        if test_id in CLASSIFICATIONS:
            test["category"] = "full_model"
            test["output_classification"] = CLASSIFICATIONS[test_id]
            test["classification_basis"] = "manual inspection of exact persisted output; OOM remains unclassified"
            seen.add(test_id)
    missing = sorted(set(CLASSIFICATIONS) - seen)
    if missing:
        raise SystemExit(f"missing expected full-model results: {missing}")
    aggregate["full_model_classifications"] = {
        "COHERENT": "answers the task with normal prose/code and no pathological repetition",
        "DEGRADED": "some useful normal output, but incomplete or partially corrupted",
        "BROKEN": "repetition, blank/special-token soup, unrelated output, or immediate EOS",
        "null": "run failed before generation, usually Metal insufficient memory",
    }
    path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
