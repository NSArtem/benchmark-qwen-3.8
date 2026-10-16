"""Load a model under test: mlx-lm checkpoints or Stage 10 candidates."""

from __future__ import annotations

import json
from pathlib import Path

LOADERS = ("mlx_lm", "candidate")


def load_any(loader: str, path: str | Path, rotate_dtype: str | None = None):
    """Return ``(model, tokenizer)``. Never touches the wired limit."""
    if loader == "mlx_lm":
        from mlx_lm import load

        if rotate_dtype is not None:
            raise ValueError("rotate_dtype applies only to Stage 10 candidates")
        model, tokenizer = load(str(path))
        model.eval()
        return model, tokenizer
    if loader == "candidate":
        from mlx_lm.utils import load_tokenizer

        from .candidate import load_candidate

        model, config = load_candidate(path, rotate_dtype=rotate_dtype)
        tokenizer = load_tokenizer(Path(path), eos_token_ids=config.get("eos_token_id"))
        return model, tokenizer
    raise ValueError(f"unknown loader {loader!r}; expected one of {LOADERS}")


def add_model_args(parser) -> None:
    parser.add_argument("--exp", required=True)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--loader", choices=LOADERS, required=True)
    parser.add_argument("--rotate-dtype", default=None)


def model_args(args) -> list[str]:
    out = ["--exp", args.exp, "--model", str(args.model), "--loader", args.loader]
    if args.rotate_dtype:
        out += ["--rotate-dtype", args.rotate_dtype]
    return out


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
