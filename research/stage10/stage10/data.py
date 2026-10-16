"""Build the fixed English, code, and Russian token windows for Stage 10."""

from __future__ import annotations

import argparse
import hashlib
import json
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


SOURCE = Path(
    "/Users/artem/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/"
    "snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
)
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO_ROOT / "benchmarks" / "results" / "stage10" / "data"
DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "data_manifest.json"
SEQUENCE_LENGTH = 2048
EVAL_WINDOWS = 16
CALIBRATION_WINDOWS = 32


@dataclass(frozen=True)
class Document:
    source_id: str
    text: str


def _input_ids(tokenizer: Any, text: str) -> np.ndarray:
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    return np.asarray(ids, dtype=np.int32).reshape(-1)


def make_windows(
    documents: Sequence[Document],
    tokenizer: Any,
    count: int,
    sequence_length: int = SEQUENCE_LENGTH,
) -> np.ndarray:
    """Join documents, tokenize without special tokens, and take full windows."""
    if count <= 0 or sequence_length <= 0:
        raise ValueError("count and sequence_length must be positive")
    if not documents:
        raise ValueError("at least one source document is required")
    joined = "\n\n".join(document.text for document in documents)
    ids = _input_ids(tokenizer, joined)
    needed = count * sequence_length
    if ids.size < needed:
        raise ValueError(f"tokenized corpus has {ids.size} tokens; {needed} required")
    return ids[:needed].reshape(count, sequence_length)


def make_domain_arrays(
    eval_documents: Sequence[Document],
    calibration_documents: Sequence[Document],
    tokenizer: Any,
    *,
    sequence_length: int = SEQUENCE_LENGTH,
    eval_count: int = EVAL_WINDOWS,
    calibration_count: int = CALIBRATION_WINDOWS,
) -> tuple[np.ndarray, np.ndarray]:
    """Build eval/calibration arrays and reject any shared source identity."""
    eval_ids = {document.source_id for document in eval_documents}
    calibration_ids = {document.source_id for document in calibration_documents}
    overlap = eval_ids & calibration_ids
    if overlap:
        raise ValueError(f"eval/calibration share source documents: {sorted(overlap)[:5]}")
    eval_array = make_windows(eval_documents, tokenizer, eval_count, sequence_length)
    calibration_array = make_windows(
        calibration_documents, tokenizer, calibration_count, sequence_length
    )
    return eval_array, calibration_array


def _revision(repo_id: str) -> str:
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(repo_id, revision="main")
    if not info.sha:
        raise RuntimeError(f"Hugging Face returned no commit sha for {repo_id}")
    return info.sha


def _dataset(repo_id: str, config: str | None, split: str, revision: str, *, streaming: bool = False):
    from datasets import load_dataset

    return load_dataset(
        repo_id,
        config,
        split=split,
        revision=revision,
        streaming=streaming,
    )


def _text_documents(dataset: Iterable[dict[str, Any]], source_prefix: str) -> list[Document]:
    documents = []
    for index, row in enumerate(dataset):
        text = row.get("text")
        if isinstance(text, str) and text:
            documents.append(Document(f"{source_prefix}:{index}", text))
    return documents


def _stdlib_documents(directories: Sequence[str]) -> tuple[list[Document], list[str]]:
    stdlib = Path(sysconfig.get_paths()["stdlib"])
    paths: list[Path] = []
    for dirname in directories:
        root = stdlib / dirname
        if not root.is_dir():
            raise FileNotFoundError(f"Python stdlib directory missing: {root}")
        paths.extend(path for path in root.rglob("*.py") if path.is_file())
    paths.sort(key=lambda path: path.relative_to(stdlib).as_posix())
    docs = [
        Document(
            "code:" + path.relative_to(stdlib).as_posix(),
            path.read_text(encoding="utf-8", errors="replace"),
        )
        for path in paths
    ]
    return docs, [path.relative_to(stdlib).as_posix() for path in paths]


def _eligible_russian_articles(dataset: Iterable[dict[str, Any]]) -> tuple[list[Document], list[Document]]:
    """Return eligible article ordinals [0,2000) and [10000,12000)."""
    eval_docs: list[Document] = []
    calibration_docs: list[Document] = []
    eligible_index = 0
    for row in dataset:
        text = row.get("text")
        if not isinstance(text, str) or len(text) < 2000:
            continue
        doc = Document(f"ru:eligible:{eligible_index}", text)
        if eligible_index < 2000:
            eval_docs.append(doc)
        elif 10000 <= eligible_index < 12000:
            calibration_docs.append(doc)
        eligible_index += 1
        if eligible_index >= 12000:
            break
    if len(eval_docs) != 2000 or len(calibration_docs) != 2000:
        raise ValueError(
            "Russian stream ended before 12000 articles with at least 2000 characters "
            f"(eval={len(eval_docs)}, calibration={len(calibration_docs)})"
        )
    return eval_docs, calibration_docs


def _save_array(path: Path, array: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array.astype(np.int32, copy=False), allow_pickle=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": path.name, "shape": list(array.shape), "dtype": "int32", "sha256": digest}


def build_token_sets(
    *,
    source: Path = SOURCE,
    output_dir: Path = DEFAULT_OUTPUT,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    """Fetch all specified corpora and save the fixed token arrays and manifest."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    manifest: dict[str, Any] = {
        "sequence_length": SEQUENCE_LENGTH,
        "eval_windows_per_domain": EVAL_WINDOWS,
        "calibration_windows_per_domain": CALIBRATION_WINDOWS,
        "tokenizer_source": str(source),
        "datasets": {},
        "arrays": {},
    }

    en_repo = "Salesforce/wikitext"
    en_revision = _revision(en_repo)
    en_eval = _text_documents(
        _dataset(en_repo, "wikitext-2-raw-v1", "test", en_revision), "en:test"
    )
    en_calibration = _text_documents(
        _dataset(en_repo, "wikitext-2-raw-v1", "train", en_revision), "en:train"
    )
    manifest["datasets"]["en"] = {
        "repo": en_repo,
        "config": "wikitext-2-raw-v1",
        "revision": en_revision,
        "eval_split": "test",
        "calibration_split": "train",
    }

    code_eval, code_eval_files = _stdlib_documents(("json", "email", "asyncio"))
    code_calibration, code_calibration_files = _stdlib_documents(
        ("http", "xml", "concurrent", "logging", "unittest")
    )
    manifest["datasets"]["code"] = {
        "python": __import__("platform").python_version(),
        "stdlib_root": sysconfig.get_paths()["stdlib"],
        "eval_files": code_eval_files,
        "calibration_files": code_calibration_files,
    }

    ru_repo = "wikimedia/wikipedia"
    ru_revision = _revision(ru_repo)
    ru_stream = _dataset(
        ru_repo, "20231101.ru", "train", ru_revision, streaming=True
    )
    ru_eval, ru_calibration = _eligible_russian_articles(ru_stream)
    manifest["datasets"]["ru"] = {
        "repo": ru_repo,
        "config": "20231101.ru",
        "revision": ru_revision,
        "split": "train",
        "streaming": True,
        "min_characters": 2000,
        "eval_eligible_article_ordinals": [0, 1999],
        "calibration_eligible_article_ordinals": [10000, 11999],
    }

    domains = {
        "en": (en_eval, en_calibration),
        "code": (code_eval, code_calibration),
        "ru": (ru_eval, ru_calibration),
    }
    for domain, (eval_docs, calibration_docs) in domains.items():
        eval_array, calibration_array = make_domain_arrays(
            eval_docs, calibration_docs, tokenizer
        )
        manifest["arrays"][f"{domain}_eval"] = _save_array(
            output_dir / f"{domain}_eval.npy", eval_array
        )
        manifest["arrays"][f"{domain}_calibration"] = _save_array(
            output_dir / f"{domain}_calibration.npy", calibration_array
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    result = build_token_sets(
        source=args.source, output_dir=args.output_dir, manifest_path=args.manifest
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


DOMAINS = ("en", "code", "ru")


def load_split(split: str, data_dir: Path = DEFAULT_OUTPUT) -> np.ndarray:
    """Return the token windows of one split, domains concatenated in DOMAINS order."""
    arrays = [np.load(data_dir / f"{domain}_{split}.npy", allow_pickle=False) for domain in DOMAINS]
    return np.concatenate(arrays, axis=0).astype(np.int32)


def domain_rows(split: str = "eval") -> dict[str, slice]:
    """Row ranges of each domain inside ``load_split(split)``."""
    count = EVAL_WINDOWS if split == "eval" else CALIBRATION_WINDOWS
    return {domain: slice(i * count, (i + 1) * count) for i, domain in enumerate(DOMAINS)}
