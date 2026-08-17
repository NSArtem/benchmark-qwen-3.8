#!/usr/bin/env python
"""Add manual raw-output classifications and hashes after visual review."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path("qwen38-mlx-threshold")

CLASSIFICATIONS = {
    "primary-0512": ("COHERENT", "Useful answer with normal prose and code-oriented diagnosis."),
    "primary-1024": ("BROKEN", "Replacement-character token followed by repeated </think>."),
    "primary-1536": ("DEGRADED", "Useful diagnosis, but begins with an invalid </think> token."),
    "primary-2048": ("BROKEN", "Single unrelated CJK token followed by repeated blank output."),
    "primary-2560": ("BROKEN", "Repeated markdown asterisks with no progression."),
    "primary-3072": ("BROKEN", "Repeated checklist fragment with no progression."),
    "primary-3584": ("BROKEN", "Unrelated Cyrillic/meta output and no task answer."),
    "primary-4096": ("BROKEN", "Unrelated CJK token and repeated numeric punctuation."),
    "primary-4608": ("BROKEN", "Plausible opening collapses into repeated sentence fragments."),
    "bisect-0768": ("BROKEN", "Repeated unrelated Italian token."),
    "bisect-0640": ("BROKEN", "Repeated HABitat token soup."),
    "boundary-0512-r1": ("COHERENT", "Useful coding-agent diagnosis."),
    "boundary-0512-r2": ("COHERENT", "Useful coding-agent diagnosis; differs from repeat 1."),
    "boundary-0640-r1": ("COHERENT", "Coherent prose, but it drifts to an unrelated order-domain scenario."),
    "boundary-0640-r2": ("DEGRADED", "Useful diagnosis after an invalid leading </think> token."),
    "chunk-0512-s1024": ("COHERENT", "Useful diagnosis with normal prose."),
    "chunk-0512-s512": ("BROKEN", "Repeated # token."),
    "chunk-0512-s256": ("COHERENT", "Useful coding-agent diagnosis."),
    "chunk-0640-s1024": ("DEGRADED", "Useful diagnosis after an invalid leading </think> token."),
    "chunk-0640-s512": ("COHERENT", "Useful incident diagnosis."),
    "chunk-0640-s256": ("COHERENT", "Useful incident diagnosis."),
    "firsttoken-0512": ("BROKEN", "First token # is followed by the same token repeated 15 times."),
    "firsttoken-0640": ("DEGRADED", "First token is </think>; the remaining short diagnostic tokens start a useful answer."),
    "boundary-0768-r1": ("BROKEN", "Repeated f token."),
    "boundary-0768-r2": ("BROKEN", "Unrelated CJK/token soup."),
}


def main() -> None:
    path = ROOT / "results.json"
    aggregate = json.loads(path.read_text(encoding="utf-8"))
    for result in aggregate["tests"]:
        output_path = Path(result["output_file"])
        if output_path.exists():
            result["output_sha256"] = hashlib.sha256(output_path.read_bytes()).hexdigest()
        if result["test_id"] in CLASSIFICATIONS:
            classification, reason = CLASSIFICATIONS[result["test_id"]]
            result["output_classification"] = classification
            result["classification_reason"] = reason
        elif not result.get("success"):
            result["output_classification"] = None
            result["classification_reason"] = "No generated output; runtime failed before completion."
    aggregate["classification_policy"] = "Manual review of every persisted output; no generous interpretation of repetition or token soup."
    aggregate["tests"] = sorted(aggregate["tests"], key=lambda item: item["test_id"])
    path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
