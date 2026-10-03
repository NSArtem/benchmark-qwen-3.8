#!/usr/bin/env python3
"""Build the production Qwen3.8-27B affine 3-bit / group-128 MLX checkpoint.

The only input is the original BF16 ``Qwen/Qwen3.8-27B`` Hugging Face
snapshot at ``SOURCE_REVISION``.  Everything else comes from this file and the
pinned mlx-lm / transformers code in ``uv.lock``; no existing MLX checkpoint
is read.

Weights: conversion is bounded-memory and narrower than mlx-lm's general
converter.  Source tensors are loaded one transformer layer at a time,
sanitized with the pinned mlx-lm Qwen3.5-family implementation, quantized
(affine, 3 bits, group size 128), and flushed to ~512 MiB safetensor shards.
Peak MLX memory is about 3.3 GiB instead of the full 50+ GiB source.  Only
``lm_head.weight`` and ``model.language_model.*`` are kept; the vision tower
(``model.visual.*``) and MTP head (``mtp.*``) are dropped.  Expected output:
851 source tensors -> 498 quantized modules, 1,847 tensors, 19 shards,
10,930,715,648 bytes (10.180 GiB).

Metadata is derived from the source:

- ``config.json``: ``derive_text_config`` (source config + generation config).
- ``tokenizer.json``, ``tokenizer_config.json``, ``chat_template.jinja``:
  ``write_tokenizer_assets`` re-serializes the source tokenizer through the
  pinned transformers ``TokenizersBackend``.
- ``generation_config.json``: copied verbatim, as ``mlx_lm.utils.save`` does.
- ``model.safetensors.index.json``: ``total_size`` from the written tensors,
  ``total_parameters`` from pinned ``mlx_lm.utils.get_total_parameters``.

``--self-test`` checks all of this against the source without loading weight
arrays.  The converter never changes the MLX wired-memory policy.

Canonical identity: ``tools/g128-reference-manifest.json`` records the size and
SHA-256 of every file of the validated canonical checkpoint.  A default build
is verified against it before it is moved to ``--output``, and
``--verify-output DIR`` checks any existing checkpoint using only the standard
library: no source, MLX, network or reference checkpoint is needed.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import re
import resource
import shutil
import struct
import subprocess
import tempfile
import sys
import time
import tomllib
from pathlib import Path
from typing import Any


SOURCE_REPO = "Qwen/Qwen3.8-27B"
SOURCE_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"

# Hub hashes of the metadata files at SOURCE_REVISION: git blob ids for plain
# files, the LFS sha256 for tokenizer.json.  config derivation and tokenizer
# handling were validated against exactly these files.
SOURCE_METADATA = {
    "config.json": ("git-blob", "706cebd746c4b6f2b1d1f892630867acfdfd3df8"),
    "generation_config.json": ("git-blob", "023756cfadf88e5bf69eefeee3e172f38c448d64"),
    "model.safetensors.index.json": ("git-blob", "da35e3c564457dface7d138f0b6cac284ff8958c"),
    "chat_template.jinja": ("git-blob", "c0c686f9c38d70d179fb7b5f5aa7530bc913dda3"),
    "tokenizer_config.json": ("git-blob", "5de744b3fca2129d7186979ae47c06be33903243"),
    "tokenizer.json": (
        "sha256",
        "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
    ),
}
SOURCE_SHARD_COUNT = 18
SOURCE_DOWNLOAD_HINT = (
    f"uv run hf download {SOURCE_REPO} --revision {SOURCE_REVISION} "
    "--include '*.safetensors' " + " ".join(f"--include {name}" for name in SOURCE_METADATA)
)

GROUP_SIZE = 128
BITS = 3
QUANTIZATION = {"group_size": GROUP_SIZE, "bits": BITS, "mode": "affine"}
DEFAULT_SHARD_BYTES = 512 * (1 << 20)

# Verified Qwen3.8-27B text architecture under the pinned mlx-lm revision.
TEXT_LAYERS = 64
FULL_ATTENTION_LAYERS = 16
RETAINED_SOURCE_TENSORS = 851
QUANTIZED_MODULES = 498
EXPECTED_TEXT_CONFIG = {
    "model_type": "qwen3_5_text",
    "num_hidden_layers": TEXT_LAYERS,
    "hidden_size": 5120,
    "vocab_size": 248320,
    "full_attention_interval": 4,
    "tie_word_embeddings": False,
}

# MTP is not part of the production model: its tensors are dropped, so the
# config must not advertise an MTP head either.  Pinned mlx-lm ignores these.
MTP_TEXT_CONFIG_KEYS = ("mtp_num_hidden_layers", "mtp_use_dedicated_embeddings")

TEXT_PREFIX = "model.language_model."
LM_HEAD_KEY = "lm_head.weight"
DROPPED_PREFIXES = {"visual": "model.visual.", "mtp": "mtp."}
SENTINEL_FRAGMENT = "__qwen38_chunk_sentinel__"
LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.")

MODEL_NAME = "Qwen3.8-27B-3bit-g128-MLX"
REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = Path(__file__).resolve().with_name("g128-reference-manifest.json")
MANIFEST_SCHEMA_VERSION = 1

TOKENIZER_OUTPUT_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
TOKENIZER_PROBES = (
    "def f(x):\n    return x**2  # ok\n",
    "नमस्ते दुनिया",
    "สวัสดีครับ",
    "café Ünïcödé — “quotes” 你好 🙂",
    "<|im_start|>user\nhi<|im_end|>\n",
)


# --------------------------------------------------------------------------
# Source metadata


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def file_digest(path: Path, kind: str) -> str:
    data = path.read_bytes()
    if kind == "git-blob":
        return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
    if kind == "sha256":
        return hashlib.sha256(data).hexdigest()
    raise ValueError(f"unknown digest kind: {kind}")


def verify_source_metadata(source: Path) -> dict[str, str]:
    """Require the exact SOURCE_REVISION metadata files in ``source``."""

    missing = [name for name in SOURCE_METADATA if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"source snapshot {source} is missing {missing}; fetch the complete "
            f"source with:\n  {SOURCE_DOWNLOAD_HINT}"
        )
    digests = {}
    for name, (kind, expected) in SOURCE_METADATA.items():
        actual = file_digest(source / name, kind)
        if actual != expected:
            raise ValueError(
                f"{name} does not match {SOURCE_REPO}@{SOURCE_REVISION} "
                f"({kind} {actual}, expected {expected})"
            )
        digests[name] = actual
    return digests


def validate_source_config(config: dict[str, Any]) -> None:
    """Recognize the BF16 Qwen3.8-27B (Qwen3.5-family VL) source config."""

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(f"unsupported source config: {message}")

    require(config.get("model_type") == "qwen3_5", f"model_type={config.get('model_type')!r}")
    require(
        config.get("architectures") == ["Qwen3_5ForConditionalGeneration"],
        f"architectures={config.get('architectures')!r}",
    )
    require(isinstance(config.get("text_config"), dict), "no nested text_config")
    require(config.get("tie_word_embeddings") is False, "tied word embeddings")
    text = config["text_config"]
    for key, expected in EXPECTED_TEXT_CONFIG.items():
        require(text.get(key) == expected, f"text_config.{key}={text.get(key)!r}")
    require(not text.get("num_experts"), "MoE text model")
    layer_types = text.get("layer_types", [])
    require(len(layer_types) == TEXT_LAYERS, f"{len(layer_types)} layer_types")
    require(
        layer_types.count("full_attention") == FULL_ATTENTION_LAYERS
        and layer_types.count("linear_attention") == TEXT_LAYERS - FULL_ATTENTION_LAYERS,
        "unexpected layer_types mix",
    )
    require(isinstance(text.get("rope_parameters"), dict), "missing rope_parameters")


def derive_text_config(
    source_config: dict[str, Any], generation_config: dict[str, Any]
) -> dict[str, Any]:
    """Return the text-only MLX ``config.json`` for the BF16 source config.

    Pinned mlx-lm maps ``model_type`` ``qwen3_5`` to ``mlx_lm.models.qwen3_5``,
    whose ``ModelArgs`` keeps the nested ``text_config`` (no flattening) and
    ignores vision keys.  The edits mirror pinned mlx-lm's own conversion:

    - drop ``vision_config`` (``mlx_lm.utils.save_config`` does the same);
    - rename ``rope_parameters.rope_type`` to ``type``, which
      ``TextModelArgs.__post_init__`` does in place on the shared dict;
    - copy ``eos_token_id`` from ``generation_config.json``, as
      ``mlx_lm.utils.load_config`` does at load time;
    - record the quantization as ``quantization`` and ``quantization_config``.

    The MTP head keys are also removed because the checkpoint has no MTP
    weights.  The input is not mutated, and the output is key-sorted.
    """

    validate_source_config(source_config)
    config = copy.deepcopy(source_config)
    config.pop("vision_config", None)
    text = config["text_config"]
    for key in MTP_TEXT_CONFIG_KEYS:
        text.pop(key, None)
    rope = text["rope_parameters"]
    if "type" not in rope and "rope_type" in rope:
        rope["type"] = rope.pop("rope_type")
    if eos_token_id := generation_config.get("eos_token_id"):
        config["eos_token_id"] = eos_token_id
    config["quantization"] = dict(QUANTIZATION)
    config["quantization_config"] = dict(QUANTIZATION)
    return dict(sorted(config.items()))


def read_safetensors_header(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("rb") as stream:
        raw_size = stream.read(8)
        if len(raw_size) != 8:
            raise ValueError(f"short safetensors header: {path}")
        header_size = struct.unpack("<Q", raw_size)[0]
        header = json.loads(stream.read(header_size))
    header.pop("__metadata__", None)
    return {
        name: {
            "shape": meta["shape"],
            "dtype": meta["dtype"],
            "data_bytes": meta["data_offsets"][1] - meta["data_offsets"][0],
        }
        for name, meta in header.items()
    }


def select_text_tensors(weight_map: dict[str, str]) -> tuple[dict[str, str], dict[str, int]]:
    """Split the source index into retained text tensors and dropped counts.

    Any key outside the known text, vision and MTP families is an error, so a
    new tensor family is never dropped silently.
    """

    kept: dict[str, str] = {}
    dropped = {family: 0 for family in DROPPED_PREFIXES}
    unknown = []
    for key, filename in weight_map.items():
        if key == LM_HEAD_KEY or key.startswith(TEXT_PREFIX):
            kept[key] = filename
            continue
        family = next((f for f, p in DROPPED_PREFIXES.items() if key.startswith(p)), None)
        if family is None:
            unknown.append(key)
        else:
            dropped[family] += 1
    if unknown:
        raise ValueError(f"unrecognized source tensors: {sorted(unknown)[:5]}")
    if not kept:
        raise ValueError("source index has no Qwen3.8 text tensors")
    return kept, dropped


def plan_chunks(weight_map: dict[str, str]) -> list[tuple[int | None, list[str]]]:
    by_layer: dict[int, list[str]] = {}
    non_layer: list[str] = []
    for key in sorted(weight_map):
        match = LAYER_RE.match(key)
        if match:
            by_layer.setdefault(int(match.group(1)), []).append(key)
        else:
            non_layer.append(key)

    chunks: list[tuple[int | None, list[str]]] = []
    if non_layer:
        chunks.append((None, non_layer))
    chunks.extend((layer, by_layer[layer]) for layer in sorted(by_layer))
    return chunks


def output_key(source_key: str) -> str:
    """Pinned ``qwen3_5.Model.sanitize`` key mapping for retained tensors."""

    if source_key.startswith(TEXT_PREFIX):
        return source_key.replace("model.language_model", "language_model.model", 1)
    return f"language_model.{source_key}"


def source_headers(source: Path, weight_map: dict[str, str]) -> dict[str, dict[str, Any]]:
    headers: dict[str, dict[str, Any]] = {}
    for filename in sorted(set(weight_map.values())):
        for key, meta in read_safetensors_header(source / filename).items():
            if key in weight_map:
                headers[key] = meta
    return headers


def read_source(source: Path) -> dict[str, Any]:
    """Load and validate all source metadata; no weight arrays are read."""

    digests = verify_source_metadata(source)
    source_config = load_json(source / "config.json")
    generation_config = load_json(source / "generation_config.json")
    config = derive_text_config(source_config, generation_config)

    index = load_json(source / "model.safetensors.index.json")
    all_files = sorted(set(index["weight_map"].values()))
    if len(all_files) != SOURCE_SHARD_COUNT:
        raise ValueError(f"expected {SOURCE_SHARD_COUNT} source shards, found {len(all_files)}")
    weight_map, dropped = select_text_tensors(index["weight_map"])
    if len(weight_map) != RETAINED_SOURCE_TENSORS:
        raise ValueError(
            f"expected {RETAINED_SOURCE_TENSORS} retained source tensors, found {len(weight_map)}"
        )
    headers = source_headers(source, weight_map)
    if set(headers) != set(weight_map):
        raise ValueError("source index/header tensor mismatch")
    if {meta["dtype"] for meta in headers.values()} != {"BF16"}:
        raise ValueError("retained source tensors are not all BF16")
    return {
        "digests": digests,
        "config": config,
        "weight_map": weight_map,
        "dropped": dropped,
        "headers": headers,
    }


# --------------------------------------------------------------------------
# Pinned mlx-lm model


def build_model(config: dict[str, Any]):
    """Instantiate the pinned mlx-lm model; parameters stay lazy (no weights)."""

    from mlx_lm.utils import _get_classes

    # ModelArgs/TextModelArgs keep references into the dict and mutate it.
    config = copy.deepcopy(config)
    model_class, model_args_class = _get_classes(config)
    return model_class(model_args_class.from_dict(config))


def quantization_keys(model) -> set[str]:
    keys = set()
    for path, module in model.named_modules():
        if hasattr(module, "to_quantized"):
            keys.add(f"{path}.weight")
    return keys


def parameter_sizes(model) -> dict[str, int]:
    from mlx.utils import tree_flatten

    return {key: int(value.size) for key, value in tree_flatten(model.parameters())}


def total_parameters(model) -> int:
    """``total_parameters`` exactly as pinned ``mlx_lm.utils.save_model`` writes it.

    ``get_total_parameters`` sums leaf modules only, so the GatedDeltaNet
    ``A_log`` and ``dt_bias`` vectors held directly by each linear-attention
    block (48 layers x 96 values) are not counted.
    """

    from mlx_lm.utils import get_total_parameters

    return int(get_total_parameters(model))


def verify_model_plan(model, weight_map, headers) -> dict[str, Any]:
    """Check the source tensors map one-to-one onto the pinned model."""

    layers = model.language_model.layers
    linear_layers = sum(1 for layer in layers if layer.is_linear)
    if len(layers) != TEXT_LAYERS or linear_layers != TEXT_LAYERS - FULL_ATTENTION_LAYERS:
        raise ValueError(f"unexpected layer layout: {len(layers)} layers, {linear_layers} linear")

    qkeys = quantization_keys(model)
    if len(qkeys) != QUANTIZED_MODULES:
        raise ValueError(f"expected {QUANTIZED_MODULES} quantization keys, found {len(qkeys)}")

    sizes = parameter_sizes(model)
    source_sizes = {output_key(key): math.prod(meta["shape"]) for key, meta in headers.items()}
    if set(sizes) != set(source_sizes):
        missing = sorted(set(sizes) - set(source_sizes))
        extra = sorted(set(source_sizes) - set(sizes))
        raise ValueError(f"source/model key mismatch; missing={missing[:3]} extra={extra[:3]}")
    mismatched = sorted(key for key in sizes if sizes[key] != source_sizes[key])
    if mismatched:
        raise ValueError(f"source/model size mismatch: {mismatched[:3]}")
    if not qkeys <= set(sizes):
        raise ValueError("quantization key without a source tensor")
    leaked = [key for key in sizes if "visual" in key or "mtp." in key]
    if leaked:
        raise ValueError(f"vision/MTP tensor would be emitted: {leaked[:3]}")

    parameters = total_parameters(model)
    direct = sum(size for key, size in sizes.items() if key.endswith((".A_log", ".dt_bias")))
    if sum(sizes.values()) - parameters != direct:
        raise ValueError("pinned get_total_parameters semantics changed")
    return {
        "layers": len(layers),
        "linear_attention_layers": linear_layers,
        "quantization_keys": len(qkeys),
        "retained_source_tensors": len(weight_map),
        "retained_source_shards": len(set(weight_map.values())),
        "retained_source_bytes": sum(meta["data_bytes"] for meta in headers.values()),
        "source_parameter_count": sum(source_sizes.values()),
        "total_parameters": parameters,
        "uncounted_direct_parameters": direct,
    }


def sanitize_chunk(model, loaded: dict[str, Any]) -> dict[str, Any]:
    """Apply pinned Qwen3.5 sanitization while preserving its global norm rule.

    TextModel.sanitize shifts every norm when *any* unsanitized depthwise
    conv1d exists anywhere in the source. A one-layer chunk would otherwise
    miss that global condition for full-attention layers, so an inert
    conv1d-shaped sentinel is included for the duration of the pinned
    ``model.sanitize`` call and removed afterward.
    """

    import mlx.core as mx

    sentinel_key = f"model.language_model.{SENTINEL_FRAGMENT}.conv1d.weight"
    loaded[sentinel_key] = mx.zeros((1, 1, 2), dtype=mx.float32)
    sanitized = model.sanitize(loaded)
    sanitized = {
        key: value
        for key, value in sanitized.items()
        if SENTINEL_FRAGMENT not in key
    }
    return sanitized


# --------------------------------------------------------------------------
# Tokenizer assets


def write_tokenizer_assets(source: Path, output: Path) -> list[str]:
    """Write tokenizer/chat-template files the way pinned mlx-lm conversion does.

    ``mlx_lm.utils.save`` calls ``tokenizer.save_pretrained`` and copies
    ``generation_config.json``.  The source ``tokenizer_config.json`` names
    ``Qwen2Tokenizer``; pinned transformers then rebuilds the pre-tokenizer
    with a regex lacking ``\\p{M}`` (which changes Devanagari and Thai
    tokenization), so the tokenizer is loaded as ``TokenizersBackend``, which
    keeps the source ``tokenizer.json`` intact.  ``tool_parser_type`` is what
    pinned ``mlx_lm`` infers from the chat template.
    """

    from mlx_lm.tokenizer_utils import _infer_tool_parser
    from transformers import TokenizersBackend

    tokenizer = TokenizersBackend.from_pretrained(source, local_files_only=True)
    tokenizer.init_kwargs.pop("local_files_only", None)
    if tool_parser_type := _infer_tool_parser(tokenizer.chat_template):
        tokenizer.init_kwargs["tool_parser_type"] = tool_parser_type
    tokenizer.save_pretrained(output)
    shutil.copyfile(source / "generation_config.json", output / "generation_config.json")
    return sorted(TOKENIZER_OUTPUT_FILES + ("generation_config.json",))


def _normalized_tokenizer_json(path: Path) -> dict[str, Any]:
    data = load_json(path)
    merges = data["model"].get("merges", [])
    data["model"]["merges"] = [m if isinstance(m, str) else " ".join(m) for m in merges]
    return data


def verify_tokenizer_assets(source: Path, output: Path) -> dict[str, Any]:
    """Check the written tokenizer loads like the source ``tokenizer.json``."""

    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    unexpected = sorted(
        p.name for p in output.iterdir()
        if p.name.startswith(("tokenizer", "special_tokens", "chat_template", "vocab", "merges"))
        and p.name not in TOKENIZER_OUTPUT_FILES
    )
    if unexpected:
        raise ValueError(f"unexpected tokenizer files written: {unexpected}")
    # Only the merges list encoding ("a b" vs ["a", "b"]) may differ.
    if _normalized_tokenizer_json(output / "tokenizer.json") != _normalized_tokenizer_json(
        source / "tokenizer.json"
    ):
        raise ValueError("written tokenizer.json differs from the source tokenizer.json")

    runtime = AutoTokenizer.from_pretrained(output, local_files_only=True)
    if type(runtime).__name__ != "TokenizersBackend":
        raise ValueError(f"runtime tokenizer class is {type(runtime).__name__}")
    reference = Tokenizer.from_file(str(source / "tokenizer.json"))
    for text in TOKENIZER_PROBES:
        expected = reference.encode(text, add_special_tokens=False).ids
        if runtime.encode(text, add_special_tokens=False) != expected:
            raise ValueError(f"tokenization differs from source for {text!r}")
    template = (source / "chat_template.jinja").read_text(encoding="utf-8")
    if runtime.chat_template != template:
        raise ValueError("chat template differs from source chat_template.jinja")
    config = load_json(output / "tokenizer_config.json")
    eos_ids = load_json(source / "generation_config.json")["eos_token_id"]
    if runtime.eos_token_id not in eos_ids:
        raise ValueError(f"tokenizer eos {runtime.eos_token_id} not in {eos_ids}")
    return {
        "tokenizer_class": config.get("tokenizer_class"),
        "tool_parser_type": config.get("tool_parser_type"),
        "probes_match_source": len(TOKENIZER_PROBES),
        "eos_token_id": runtime.eos_token_id,
    }


# --------------------------------------------------------------------------
# Canonical output manifest

# Files every canonical checkpoint has besides its weight shards.
REQUIRED_OUTPUT_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    *TOKENIZER_OUTPUT_FILES,
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ManifestError(ValueError):
    """The manifest file itself is unreadable or not a supported schema."""


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def locked_toolchain(root: Path = REPO_ROOT) -> dict[str, str]:
    """The pinned conversion toolchain from ``uv.lock`` and ``.python-version``."""

    lock = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
    packages = {package["name"]: package for package in lock["package"]}
    return {
        "python": (root / ".python-version").read_text(encoding="utf-8").strip(),
        "mlx": packages["mlx"]["version"],
        "mlx_lm_revision": packages["mlx-lm"]["source"]["git"].rsplit("#", 1)[1],
        "transformers": packages["transformers"]["version"],
        "tokenizers": packages["tokenizers"]["version"],
    }


def model_entries(model_dir: Path) -> list[str]:
    """Entry names in ``model_dir`` that count as checkpoint content.

    Hidden entries (``.DS_Store`` and the like) are filesystem noise and are
    ignored.  Every other entry, file or directory, is content, so verification
    reports anything the manifest does not list.
    """

    return sorted(path.name for path in model_dir.iterdir() if not path.name.startswith("."))


def artifact_summary(model_dir: Path) -> dict[str, Any]:
    """Structural facts read from ``config.json`` and the index (no hashing)."""

    config = load_json(model_dir / "config.json")
    index = load_json(model_dir / "model.safetensors.index.json")
    weight_map = index["weight_map"]
    return {
        "quantization": config.get("quantization"),
        "quantization_config": config.get("quantization_config"),
        "shard_files": sorted(set(weight_map.values())),
        "tensor_count": len(weight_map),
        "quantized_modules": sum(key.endswith(".scales") for key in weight_map),
        "weight_payload_bytes": index["metadata"]["total_size"],
        "total_parameters": index["metadata"]["total_parameters"],
    }


def calculate_manifest(model_dir: Path) -> dict[str, Any]:
    """Describe the checkpoint in ``model_dir`` as a canonical manifest.

    The result has no paths, timestamps or host details, so the same
    checkpoint always gives the same manifest (see ``manifest_text``).
    """

    model_dir = Path(model_dir)
    names = model_entries(model_dir)
    summary = artifact_summary(model_dir)
    shards = summary["shard_files"]
    missing = sorted(set(REQUIRED_OUTPUT_FILES + tuple(shards)) - set(names))
    extra = sorted(set(names) - set(REQUIRED_OUTPUT_FILES) - set(shards))
    if missing or extra:
        raise ValueError(f"not a complete g128 checkpoint: missing={missing} extra={extra}")
    if summary["quantization"] != QUANTIZATION or summary["quantization_config"] != QUANTIZATION:
        raise ValueError(f"config quantization is {summary['quantization']}, expected {QUANTIZATION}")

    files = {}
    for name in names:
        path = model_dir / name
        if not path.is_file():
            raise ValueError(f"not a regular file: {name}")
        files[name] = {"size": path.stat().st_size, "sha256": sha256_file(path)}
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "model": {
            "name": MODEL_NAME,
            "source_repo": SOURCE_REPO,
            "source_revision": SOURCE_REVISION,
            "quantization": dict(QUANTIZATION),
        },
        "build": {"tool": "tools/build_g128.py", "shard_bytes": DEFAULT_SHARD_BYTES},
        "toolchain": locked_toolchain(),
        "expected": {
            "total_files": len(files),
            "total_bytes": sum(entry["size"] for entry in files.values()),
            "weight_shards": len(shards),
            "weight_payload_bytes": summary["weight_payload_bytes"],
            "tensor_count": summary["tensor_count"],
            "quantized_modules": summary["quantized_modules"],
            "total_parameters": summary["total_parameters"],
            "retained_source_tensors": RETAINED_SOURCE_TENSORS,
        },
        "files": files,
    }


def manifest_text(manifest: dict[str, Any]) -> str:
    """The one serialization of a manifest: sorted keys, two-space indent."""

    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def _require_file_entry(where: str, entry: Any) -> None:
    if (
        not isinstance(entry, dict)
        or not isinstance(entry.get("size"), int)
        or isinstance(entry.get("size"), bool)
        or entry["size"] < 0
        or not isinstance(entry.get("sha256"), str)
        or not SHA256_RE.match(entry["sha256"])
    ):
        raise ManifestError(f"{where} needs an integer 'size' and a lowercase hex 'sha256'")


def validate_manifest(manifest: Any) -> dict[str, Any]:
    """Check the manifest's schema and internal consistency; return it."""

    if not isinstance(manifest, dict):
        raise ManifestError("manifest must be a JSON object")
    version = manifest.get("schema_version")
    if version != MANIFEST_SCHEMA_VERSION:
        raise ManifestError(
            f"unsupported manifest schema_version {version!r} (expected {MANIFEST_SCHEMA_VERSION})"
        )
    for key in ("model", "toolchain", "expected", "files"):
        if not isinstance(manifest.get(key), dict):
            raise ManifestError(f"manifest is missing the '{key}' object")

    model = manifest["model"]
    identity = {
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "quantization": QUANTIZATION,
    }
    for key, expected in identity.items():
        if model.get(key) != expected:
            raise ManifestError(f"manifest model.{key} is {model.get(key)!r}, expected {expected!r}")

    files = manifest["files"]
    if not files:
        raise ManifestError("manifest lists no files")
    for name, entry in files.items():
        if "/" in name or "\\" in name or name.startswith("."):
            raise ManifestError(f"manifest file name must be a plain visible file name: {name!r}")
        _require_file_entry(f"files[{name!r}]", entry)
    missing = sorted(set(REQUIRED_OUTPUT_FILES) - set(files))
    if missing:
        raise ManifestError(f"manifest does not cover required files {missing}")

    expected = manifest["expected"]
    actual_totals = {
        "total_files": len(files),
        "total_bytes": sum(entry["size"] for entry in files.values()),
        "weight_shards": sum(name.endswith(".safetensors") for name in files),
    }
    for key, actual in actual_totals.items():
        if expected.get(key) != actual:
            raise ManifestError(f"manifest expected.{key} is {expected.get(key)!r}, files give {actual}")

    legacy = manifest.get("legacy_variants", {})
    if not isinstance(legacy, dict):
        raise ManifestError("manifest legacy_variants must be an object")
    for name, variants in legacy.items():
        if name not in files or not isinstance(variants, list):
            raise ManifestError(f"legacy_variants[{name!r}] must be a list for a listed file")
        for number, variant in enumerate(variants):
            _require_file_entry(f"legacy_variants[{name!r}][{number}]", variant)
            if not isinstance(variant.get("note"), str) or not variant["note"]:
                raise ManifestError(f"legacy_variants[{name!r}][{number}] needs a 'note'")
    return manifest


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise ManifestError(f"cannot read manifest {path}: {exc}") from exc
    try:
        manifest = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"manifest {path} is not valid JSON: {exc}") from exc
    return validate_manifest(manifest)


def verify_manifest(
    model_dir: Path, manifest: dict[str, Any], allow_legacy: bool = False
) -> dict[str, Any]:
    """Compare ``model_dir`` with ``manifest`` file by file.

    Every listed file must exist with the exact size and SHA-256.  Any other
    visible entry is a failure (strict: a canonical checkpoint contains
    nothing else); hidden entries such as ``.DS_Store`` are ignored.  A file
    matching one of the manifest's ``legacy_variants`` fails too, with the
    variant's note, unless ``allow_legacy`` is set.

    Returns ``{"ok", "failures", "notes", "files", "bytes"}``; each failure
    names the offending file.
    """

    validate_manifest(manifest)
    model_dir = Path(model_dir)
    files = manifest["files"]
    legacy = manifest.get("legacy_variants", {})
    failures: list[str] = []
    notes: list[str] = []
    if not model_dir.is_dir():
        failures.append(f"FAIL {model_dir}: not a directory")
        return {"ok": False, "failures": failures, "notes": notes, "files": 0, "bytes": 0}

    for name in sorted(set(model_entries(model_dir)) - set(files)):
        failures.append(f"FAIL {name}: unexpected entry, not in the canonical manifest")

    verified_files = 0
    verified_bytes = 0
    legacy_delta = 0
    for name, entry in sorted(files.items()):
        path = model_dir / name
        if not path.exists():
            failures.append(f"FAIL {name}: missing")
            continue
        if not path.is_file():
            failures.append(f"FAIL {name}: not a regular file")
            continue
        size = path.stat().st_size
        digest = sha256_file(path) if size == entry["size"] or name in legacy else None
        if size == entry["size"] and digest == entry["sha256"]:
            verified_files += 1
            verified_bytes += size
            continue
        variant = next(
            (v for v in legacy.get(name, []) if v["size"] == size and v["sha256"] == digest),
            None,
        )
        if variant is not None:
            legacy_delta += size - entry["size"]
            if allow_legacy:
                notes.append(f"NOTE {name}: accepted legacy variant: {variant['note']}")
                verified_files += 1
                verified_bytes += size
            else:
                failures.append(
                    f"FAIL {name}: matches a known legacy variant, not the canonical file\n"
                    f"  {variant['note']}\n"
                    "  rerun with --allow-legacy to accept it"
                )
            continue
        if size != entry["size"]:
            failures.append(
                f"FAIL {name}: size mismatch\n"
                f"  expected size: {entry['size']}\n"
                f"  actual size:   {size}"
            )
        else:
            failures.append(
                f"FAIL {name}: sha256 mismatch\n"
                f"  expected sha256: {entry['sha256']}\n"
                f"  actual sha256:   {digest}"
            )

    # Every visible regular file counts, so extra or truncated content shows
    # up here too.  A known legacy variant is already reported per file.
    content_bytes = sum(
        (model_dir / name).stat().st_size
        for name in model_entries(model_dir)
        if (model_dir / name).is_file()
    )
    expected_bytes = manifest["expected"]["total_bytes"] + legacy_delta
    if content_bytes != expected_bytes:
        failures.append(
            f"FAIL aggregate size: {content_bytes} bytes of content, expected {expected_bytes}"
        )

    failures.extend(_structure_failures(model_dir, manifest))
    return {
        "ok": not failures,
        "failures": failures,
        "notes": notes,
        "files": verified_files,
        "bytes": verified_bytes,
    }


def _structure_failures(model_dir: Path, manifest: dict[str, Any]) -> list[str]:
    """Readable config/index mismatches, reported alongside any hash failure."""

    try:
        summary = artifact_summary(model_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return [f"FAIL config.json/model.safetensors.index.json: unreadable structure: {exc!r}"]
    failures = []
    quantization = manifest["model"]["quantization"]
    for key in ("quantization", "quantization_config"):
        if summary[key] != quantization:
            failures.append(f"FAIL config.json: {key} is {summary[key]}, expected {quantization}")
    expected = manifest["expected"]
    actual = dict(summary, weight_shards=len(summary["shard_files"]))
    for key in (
        "weight_shards",
        "tensor_count",
        "quantized_modules",
        "weight_payload_bytes",
        "total_parameters",
    ):
        if key in expected and actual[key] != expected[key]:
            failures.append(
                f"FAIL model.safetensors.index.json: {key} is {actual[key]}, expected {expected[key]}"
            )
    unlisted = sorted(set(summary["shard_files"]) - set(manifest["files"]))
    if unlisted:
        failures.append(f"FAIL model.safetensors.index.json: references unlisted shards {unlisted}")
    return failures


def report_verification(result: dict[str, Any], manifest_path: Path, stream=sys.stdout) -> None:
    for line in result["failures"] + result["notes"]:
        print(line, file=stream)
    if result["ok"]:
        print("PASS", file=stream)
        print("canonical g128 manifest verified", file=stream)
        print(f"{result['files']} files", file=stream)
        print(f"{result['bytes']:,} bytes", file=stream)
        accepted = f" ({len(result['notes'])} accepted legacy variant(s))" if result["notes"] else ""
        print(f"SHA-256 all match{accepted}", file=stream)
    else:
        print(f"FAIL: {len(result['failures'])} problem(s) against {manifest_path}", file=stream)


# --------------------------------------------------------------------------
# Self-test


def no_wired_limit_calls() -> bool:
    text = Path(__file__).read_text(encoding="utf-8")
    return re.search(r"set_wired" + r"_limit\s*\(", text) is None


def self_test(source: Path) -> dict[str, Any]:
    import mlx.core as mx

    source = source.resolve()
    plan = read_source(source)
    config = plan["config"]
    model = build_model(config)
    model_plan = verify_model_plan(model, plan["weight_map"], plan["headers"])

    chunks = plan_chunks(plan["weight_map"])
    if [layer for layer, _ in chunks] != [None, *range(TEXT_LAYERS)]:
        raise AssertionError("unexpected chunk plan")
    emitted = {output_key(key) for key in plan["weight_map"]}
    if any("visual" in key or "mtp." in key for key in emitted):
        raise AssertionError("vision/MTP tensor would be emitted")

    with mx.stream(mx.cpu):
        shifted = sanitize_chunk(
            model,
            {"model.language_model.layers.3.input_layernorm.weight": mx.ones((4,), dtype=mx.bfloat16)},
        )
        shifted_value = float(shifted["language_model.model.layers.3.input_layernorm.weight"][0])
    if abs(shifted_value - 2.0) > 0.01:
        raise AssertionError(f"global norm shift was not preserved: {shifted_value}")

    with tempfile.TemporaryDirectory(prefix="qwen38-g128-tokenizer-") as tmp:
        written = write_tokenizer_assets(source, Path(tmp))
        tokenizer = verify_tokenizer_assets(source, Path(tmp))

    if not no_wired_limit_calls():
        raise AssertionError("converter calls set_wired_limit")
    return {
        "status": "SOURCE_SELF_TEST_PASS",
        "source": str(source),
        "source_revision": SOURCE_REVISION,
        "source_metadata_verified": sorted(plan["digests"]),
        "model_type": config["model_type"],
        "text_model_type": config["text_config"]["model_type"],
        "dropped_source_tensors": plan["dropped"],
        **model_plan,
        "chunks": len(chunks),
        "global_norm_shift_test": shifted_value,
        "tokenizer_files": written,
        **tokenizer,
        "quantization": QUANTIZATION,
        "wired_limit_calls": 0,
    }


# --------------------------------------------------------------------------
# Conversion


def memory_snapshot(stage: str, elapsed_seconds: float) -> dict[str, Any]:
    import mlx.core as mx

    active = int(mx.get_active_memory())
    cache = int(mx.get_cache_memory())
    peak = int(mx.get_peak_memory())
    return {
        "stage": stage,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "active_bytes": active,
        "cache_bytes": cache,
        "active_plus_cache_bytes": active + cache,
        "peak_bytes": peak,
        "active_gib": round(active / (1 << 30), 6),
        "cache_gib": round(cache / (1 << 30), 6),
        "active_plus_cache_gib": round((active + cache) / (1 << 30), 6),
        "peak_gib": round(peak / (1 << 30), 6),
        "max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }


def os_snapshot() -> dict[str, str]:
    result: dict[str, str] = {}
    for command in (("vm_stat",), ("memory_pressure", "-Q"), ("sysctl", "vm.swapusage")):
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            result[command[0]] = f"unavailable: {exc}"
        else:
            result[command[0]] = (completed.stdout or completed.stderr).strip()
    return result


def append_metric(path: Path | None, record: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def flush_shard(
    output: Path,
    shard_index: int,
    shard_weights: dict[str, Any],
    output_map: dict[str, str],
) -> tuple[str, int]:
    import mlx.core as mx

    if not shard_weights:
        return "", 0
    temp_name = f"model-{shard_index:05d}-of-XXXXX.safetensors"
    mx.save_safetensors(
        str(output / temp_name),
        dict(sorted(shard_weights.items())),
        metadata={"format": "mlx"},
    )
    for key in shard_weights:
        output_map[key] = temp_name
    size = sum(int(value.nbytes) for value in shard_weights.values())
    return temp_name, size


def convert(args: argparse.Namespace) -> dict[str, Any]:
    import mlx.core as mx

    source = Path(args.source).resolve()
    final_output = Path(args.output).resolve()
    metrics_path = Path(args.metrics).resolve() if args.metrics else None
    if not source.is_dir():
        raise FileNotFoundError(f"source must be a directory: {source}")
    if final_output.exists():
        raise FileExistsError(f"refusing to overwrite output: {final_output}")
    # Build into a staging directory so an interrupted run never leaves a
    # partial checkpoint at the final path.
    output = final_output.with_name(final_output.name + ".partial")
    if output.exists():
        raise FileExistsError(f"remove the stale partial build first: {output}")

    # Only the default shard layout can reproduce the canonical files.
    canonical = args.shard_bytes == DEFAULT_SHARD_BYTES
    manifest = load_manifest(MANIFEST_PATH) if canonical else None

    started = time.monotonic()
    plan = read_source(source)
    config = plan["config"]
    weight_map = plan["weight_map"]
    model = build_model(config)
    model_plan = verify_model_plan(model, weight_map, plan["headers"])
    qkeys = quantization_keys(model)

    output.mkdir(parents=True)
    if metrics_path is not None:
        metrics_path.unlink(missing_ok=True)
    append_metric(metrics_path, {"event": "start", "source": str(source), "os": os_snapshot()})
    append_metric(
        metrics_path,
        {
            "event": "source_plan",
            "source_revision": SOURCE_REVISION,
            "dropped_source_tensors": plan["dropped"],
            **model_plan,
        },
    )

    chunks = plan_chunks(weight_map)
    target_dtype = mx.bfloat16
    cast_predicate = getattr(model, "cast_predicate", lambda _: True)
    quantized_modules = 0
    output_map: dict[str, str] = {}
    shard_weights: dict[str, Any] = {}
    shard_bytes = 0
    shard_index = 1
    total_size = 0
    saved_shards: list[str] = []

    for chunk_index, (layer_index, chunk_keys) in enumerate(chunks, start=1):
        loaded: dict[str, Any] = {}
        for filename in sorted({weight_map[key] for key in chunk_keys}):
            file_data = mx.load(str(source / filename))
            for key in chunk_keys:
                if weight_map[key] == filename:
                    loaded[key] = file_data[key]
            del file_data

        loaded = sanitize_chunk(model, loaded)
        expected_chunk_keys = {output_key(key) for key in chunk_keys}
        if expected_chunk_keys != set(loaded):
            missing = sorted(expected_chunk_keys - set(loaded))
            extra = sorted(set(loaded) - expected_chunk_keys)
            raise ValueError(f"sanitization key mismatch; missing={missing[:3]} extra={extra[:3]}")

        for key in sorted(list(loaded)):
            weight = loaded.pop(key)
            if cast_predicate(key) and mx.issubdtype(weight.dtype, mx.floating):
                weight = weight.astype(target_dtype)

            if key in qkeys:
                if weight.ndim < 2 or weight.shape[-1] % GROUP_SIZE:
                    raise ValueError(
                        f"invalid g128 quantization shape for {key}: {weight.shape}"
                    )
                quantized, scales, biases = mx.quantize(weight, **QUANTIZATION)
                mx.eval(quantized, scales, biases)
                quantized_modules += 1
                outputs = {
                    key: quantized,
                    key.replace(".weight", ".scales"): scales,
                    key.replace(".weight", ".biases"): biases,
                }
            else:
                mx.eval(weight)
                outputs = {key: weight}

            for output_key_name, value in outputs.items():
                shard_weights[output_key_name] = value
                shard_bytes += int(value.nbytes)
                total_size += int(value.nbytes)

            if shard_bytes >= args.shard_bytes:
                shard_name, _ = flush_shard(output, shard_index, shard_weights, output_map)
                saved_shards.append(shard_name)
                del shard_weights
                shard_weights = {}
                shard_bytes = 0
                shard_index += 1
                gc.collect()
                mx.clear_cache()

        del loaded
        gc.collect()
        mx.clear_cache()
        append_metric(
            metrics_path,
            {
                "event": "chunk",
                "chunk_index": chunk_index,
                "chunk_count": len(chunks),
                "layer_index": layer_index,
                "source_tensor_count": len(chunk_keys),
                **memory_snapshot(f"after_chunk_{chunk_index}", time.monotonic() - started),
            },
        )

    if shard_weights:
        shard_name, _ = flush_shard(output, shard_index, shard_weights, output_map)
        saved_shards.append(shard_name)
        del shard_weights
        gc.collect()
        mx.clear_cache()

    if quantized_modules != QUANTIZED_MODULES:
        raise RuntimeError(f"quantized {quantized_modules} modules, expected {QUANTIZED_MODULES}")
    total_shards = len(saved_shards)
    if total_shards == 0:
        raise RuntimeError("conversion produced no output shards")
    rename_map: dict[str, str] = {}
    if total_shards == 1:
        old_name = saved_shards[0]
        new_name = "model.safetensors"
        (output / old_name).rename(output / new_name)
        rename_map[old_name] = new_name
    else:
        for index, old_name in enumerate(saved_shards, start=1):
            new_name = f"model-{index:05d}-of-{total_shards:05d}.safetensors"
            (output / old_name).rename(output / new_name)
            rename_map[old_name] = new_name
    output_map = {key: rename_map[name] for key, name in output_map.items()}

    with (output / "config.json").open("w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=4)

    index = {
        "metadata": {
            "total_size": total_size,
            "total_parameters": model_plan["total_parameters"],
        },
        "weight_map": dict(sorted(output_map.items())),
    }
    with (output / "model.safetensors.index.json").open("w", encoding="utf-8") as stream:
        json.dump(index, stream, indent=4)
    write_tokenizer_assets(source, output)
    tokenizer = verify_tokenizer_assets(source, output)
    print("conversion complete", file=sys.stderr)
    if manifest is None:
        verification = "skipped: non-default --shard-bytes is not the canonical layout"
        print(f"canonical manifest verification {verification}", file=sys.stderr)
    else:
        print("verifying canonical output...", file=sys.stderr)
        result = verify_manifest(output, manifest)
        if not result["ok"]:
            report_verification(result, MANIFEST_PATH, stream=sys.stderr)
            raise RuntimeError(
                f"build does not match the canonical manifest; output left at {output}"
            )
        print("PASS: canonical g128 manifest matched", file=sys.stderr)
        verification = f"PASS: {result['files']} files, {result['bytes']} bytes"
    output.rename(final_output)

    result = {
        "event": "complete",
        "source": str(source),
        "source_revision": SOURCE_REVISION,
        "output": str(final_output),
        "output_shards": total_shards,
        "output_bytes": total_size,
        "output_gib": round(total_size / (1 << 30), 6),
        "output_tensor_count": len(output_map),
        "output_parameter_count": index["metadata"]["total_parameters"],
        "quantization": QUANTIZATION,
        "quantized_modules": quantized_modules,
        "tokenizer": tokenizer,
        "manifest_verification": verification,
        **memory_snapshot("complete", time.monotonic() - started),
        "os": os_snapshot(),
    }
    append_metric(metrics_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Qwen3.8-27B affine 3-bit/g128 text-only MLX checkpoint "
            f"from the BF16 {SOURCE_REPO} snapshot (revision {SOURCE_REVISION})."
        ),
        epilog=f"Fetch the source with:\n  {SOURCE_DOWNLOAD_HINT}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", help=f"local BF16 {SOURCE_REPO} snapshot directory")
    parser.add_argument("--output", help="new output directory (must not exist)")
    parser.add_argument("--metrics", help="optional progress JSONL path")
    parser.add_argument(
        "--shard-bytes",
        type=int,
        default=DEFAULT_SHARD_BYTES,
        help="output shard target; default 512 MiB",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="validate the source and conversion plan without loading weight arrays",
    )
    parser.add_argument(
        "--verify-output",
        metavar="DIR",
        help="verify an existing checkpoint against the canonical manifest and exit",
    )
    parser.add_argument(
        "--manifest",
        default=str(MANIFEST_PATH),
        help="manifest for --verify-output; default: the committed reference manifest",
    )
    parser.add_argument(
        "--allow-legacy",
        action="store_true",
        help="with --verify-output, accept files matching a documented legacy variant",
    )
    args = parser.parse_args()

    if args.verify_output:
        manifest_path = Path(args.manifest)
        try:
            manifest = load_manifest(manifest_path)
        except ManifestError as exc:
            print(f"FAIL manifest: {exc}")
            sys.exit(2)
        result = verify_manifest(Path(args.verify_output), manifest, args.allow_legacy)
        report_verification(result, manifest_path)
        sys.exit(0 if result["ok"] else 1)
    if not args.source:
        parser.error("--source is required unless --verify-output is used")
    if args.self_test:
        print(json.dumps(self_test(Path(args.source)), indent=2, sort_keys=True))
        return
    if not args.output:
        parser.error("--output is required unless --self-test is used")
    print(json.dumps(convert(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
