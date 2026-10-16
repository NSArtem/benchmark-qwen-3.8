"""A10: assemble results/<EXP_ID>/summary.json and gen_check.json."""

from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

from .models import write_json
from .reference import RESULTS

STAGE10 = Path(__file__).resolve().parents[1]
STATUSES = ("OK", "PARTIAL", "FAILED_MEMORY", "FAILED_ERROR", "SKIPPED_UNSAFE", "SKIPPED_UNSUPPORTED")
EVAL_KEYS = ("nll_mean", "ppl", "delta_nll", "top1_agree", "kl_top32")
LADDER_KEYS = (
    "ctx", "prefill_s", "prefill_tok_s", "ttft_s", "decode_tok_s", "peak_mlx_gib",
    "free_pct_before", "free_pct_after", "free_pct_min", "swap_mib_before",
    "swap_mib_after", "safety_stop",
)
GEN_KEYS = ("id", "n_tokens", "hit_eos", "distinct4", "max4", "p2_has_187", "p4_json_valid")


def stack() -> dict:
    return {
        "python": platform.python_version(),
        "mlx": importlib.metadata.version("mlx"),
        "mlx_lm": importlib.metadata.version("mlx-lm"),
    }


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=STAGE10, capture_output=True, text=True, check=True
    ).stdout.strip()


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def assemble(
    exp_id: str,
    recipe: str,
    status: str,
    *,
    checkpoint: dict | None = None,
    build: dict | None = None,
    notes: list[str] | None = None,
    stack_override: dict | None = None,
) -> dict:
    if status not in STATUSES:
        raise ValueError(f"invalid status {status!r}")
    out_dir = RESULTS / exp_id
    evaluation = _read(out_dir / "eval.json")
    ladder = _read(out_dir / "ladder.json")
    gen = _read(out_dir / "gen_check.json")
    if checkpoint is None:
        checkpoint = _read(out_dir / "checkpoint.json")
    if build is None:
        build = _read(out_dir / "build.json")
    summary = {
        "exp_id": exp_id,
        "recipe": recipe,
        "status": status,
        "stack": stack_override or stack(),
        "git_commit": git_commit(),
        "date": datetime.date.today().isoformat(),
        "checkpoint": checkpoint,
        "build": build,
        "eval": (
            {part: {k: evaluation[part][k] for k in EVAL_KEYS} for part in ("all", "en", "code", "ru")}
            if evaluation
            else None
        ),
        "ladder": [{k: row.get(k) for k in LADDER_KEYS} for row in ladder] if ladder else None,
        "gen_check": [{k: row[k] for k in GEN_KEYS if k in row} for row in gen] if gen else None,
        "notes": notes or [],
    }
    target = STAGE10 / "results" / exp_id
    write_json(target / "summary.json", summary)
    if gen:
        write_json(target / "gen_check.json", gen)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", required=True)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--status", required=True, choices=STATUSES)
    parser.add_argument("--note", action="append", default=[])
    args = parser.parse_args()
    print(json.dumps(assemble(args.exp, args.recipe, args.status, notes=args.note), indent=2))


if __name__ == "__main__":
    main()
