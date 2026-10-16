"""Validated access to the fixed BF16 source snapshot.

The source metadata and model sanitization helpers below are copied from
``tools/build_g128.py``. The source revision and norm handling remain tied to
that implementation; production files are never written here.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import mmap
import re
import struct
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SOURCE_REPO = "Qwen/Qwen3.8-27B"
SOURCE_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
SOURCE = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots" / SOURCE_REVISION
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
MTP_TEXT_CONFIG_KEYS = ("mtp_num_hidden_layers", "mtp_use_dedicated_embeddings")
TEXT_PREFIX = "model.language_model."
LM_HEAD_KEY = "lm_head.weight"
DROPPED_PREFIXES = {"visual": "model.visual.", "mtp": "mtp."}
SENTINEL_FRAGMENT = "__qwen38_chunk_sentinel__"
LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.")
QUANTIZATION = {"group_size": 128, "bits": 3, "mode": "affine"}


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
    missing = [name for name in SOURCE_METADATA if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"source snapshot {source} is missing {missing}")
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
    """Recognize the BF16 Qwen3.8-27B text configuration."""

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
    require(not text.get("num_experts"), "MoE text config")
    layer_types = text.get("layer_types", [])
    require(len(layer_types) == TEXT_LAYERS, f"{len(layer_types)} layer_types")
    require(
        layer_types.count("full_attention") == FULL_ATTENTION_LAYERS
        and layer_types.count("linear_attention") == TEXT_LAYERS - FULL_ATTENTION_LAYERS,
        "unexpected layer_types mix",
    )
    require(isinstance(text.get("rope_parameters"), dict), "missing rope_parameters")


# Copied from tools/build_g128.py: derive_text_config.
def derive_text_config(
    source_config: dict[str, Any], generation_config: dict[str, Any]
) -> dict[str, Any]:
    """Derive the text-only MLX config with the production converter edits."""
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


def _read_safetensors_header(path: Path) -> tuple[dict[str, dict[str, Any]], int]:
    with path.open("rb") as stream:
        raw_size = stream.read(8)
        if len(raw_size) != 8:
            raise ValueError(f"short safetensors header: {path}")
        header_size = struct.unpack("<Q", raw_size)[0]
        header = json.loads(stream.read(header_size))
    data_start = 8 + header_size
    header.pop("__metadata__", None)
    return header, data_start


def read_safetensors_header(path: Path) -> dict[str, dict[str, Any]]:
    header, _data_start = _read_safetensors_header(path)
    return {
        name: {
            "shape": meta["shape"],
            "dtype": meta["dtype"],
            "data_bytes": meta["data_offsets"][1] - meta["data_offsets"][0],
        }
        for name, meta in header.items()
    }


def select_text_tensors(weight_map: dict[str, str]) -> tuple[dict[str, str], dict[str, int]]:
    kept: dict[str, str] = {}
    dropped = {family: 0 for family in DROPPED_PREFIXES}
    unknown = []
    for key, filename in weight_map.items():
        if key == LM_HEAD_KEY or key.startswith(TEXT_PREFIX):
            kept[key] = filename
            continue
        family = next((f for f, prefix in DROPPED_PREFIXES.items() if key.startswith(prefix)), None)
        if family is None:
            unknown.append(key)
        else:
            dropped[family] += 1
    if unknown:
        raise ValueError(f"unrecognized source tensors: {sorted(unknown)[:5]}")
    if not kept:
        raise ValueError("source index has no Qwen3.8 text tensors")
    return kept, dropped


# Copied from tools/build_g128.py: plan_chunks.
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


# Copied from tools/build_g128.py: output_key.
def output_key(source_key: str) -> str:
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


# Copied from tools/build_g128.py: read_source, with its validation helpers.
def read_source(source: Path = SOURCE) -> dict[str, Any]:
    """Load and validate source metadata and tensor headers, without arrays."""
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
        "source": source,
        "digests": digests,
        "config": config,
        "weight_map": weight_map,
        "dropped": dropped,
        "headers": headers,
    }


# Copied from tools/build_g128.py: build_model.
def build_model(config: dict[str, Any]):
    """Instantiate the mlx-lm model shell; parameter arrays remain lazy."""
    from mlx_lm.utils import _get_classes

    model_config = copy.deepcopy(config)
    model_class, model_args_class = _get_classes(model_config)
    return model_class(model_args_class.from_dict(model_config))


# Copied from tools/build_g128.py: sanitize_chunk. The sentinel is required:
# qwen3_5.Model.sanitize shifts all norm weights if any source depthwise conv1d
# is still in source layout. Per-layer sanitation would miss that global
# condition on full-attention layers without this inert conv1d-shaped marker.
def sanitize_chunk(model, loaded: dict[str, Any]) -> dict[str, Any]:
    import mlx.core as mx

    sentinel_key = f"model.language_model.{SENTINEL_FRAGMENT}.conv1d.weight"
    loaded[sentinel_key] = mx.zeros((1, 1, 2), dtype=mx.float32)
    sanitized = model.sanitize(loaded)
    return {key: value for key, value in sanitized.items() if SENTINEL_FRAGMENT not in key}


def load_source_tensors(
    source: Path, weight_map: dict[str, str], source_keys: Iterable[str]
) -> dict[str, Any]:
    """Read only the named tensors from their source safetensors shards."""
    import mlx.core as mx

    dtype_map = {
        "BF16": (np.dtype("<u2"), mx.bfloat16),
        "F16": (np.dtype("<f2"), None),
        "F32": (np.dtype("<f4"), None),
        "F64": (np.dtype("<f8"), None),
        "U8": (np.dtype("u1"), None),
        "U16": (np.dtype("<u2"), None),
        "U32": (np.dtype("<u4"), None),
        "U64": (np.dtype("<u8"), None),
        "I8": (np.dtype("i1"), None),
        "I16": (np.dtype("<i2"), None),
        "I32": (np.dtype("<i4"), None),
        "I64": (np.dtype("<i8"), None),
        "BOOL": (np.dtype("bool"), None),
    }

    keys_by_file: dict[str, list[str]] = {}
    for key in source_keys:
        try:
            filename = weight_map[key]
        except KeyError as exc:
            raise KeyError(f"source tensor is not indexed: {key}") from exc
        keys_by_file.setdefault(filename, []).append(key)
    loaded = {}
    for filename, keys in sorted(keys_by_file.items()):
        path = source / filename
        header, data_start = _read_safetensors_header(path)
        with path.open("rb") as stream, mmap.mmap(
            stream.fileno(), 0, access=mmap.ACCESS_READ
        ) as mapped:
            for key in keys:
                meta = header[key]
                try:
                    np_dtype, mlx_dtype = dtype_map[meta["dtype"]]
                except KeyError as exc:
                    raise TypeError(f"unsupported safetensors dtype {meta['dtype']!r} for {key}") from exc
                start, stop = meta["data_offsets"]
                count = math.prod(meta["shape"])
                values = np.frombuffer(
                    mapped,
                    dtype=np_dtype,
                    count=count,
                    offset=data_start + start,
                ).reshape(meta["shape"])
                value = mx.array(values)
                if mlx_dtype is not None:
                    value = value.view(mlx_dtype)
                mx.eval(value)
                loaded[key] = value
                del value, values
    return loaded


def layer_source_keys(weight_map: dict[str, str], layer_idx: int) -> list[str]:
    prefix = f"model.language_model.layers.{layer_idx}."
    return sorted(key for key in weight_map if key.startswith(prefix))


def load_sanitized_layer(model, source: Path, weight_map: dict[str, str], layer_idx: int):
    keys = layer_source_keys(weight_map, layer_idx)
    if not keys:
        raise KeyError(f"no source tensors for decoder layer {layer_idx}")
    return sanitize_chunk(model, load_source_tensors(source, weight_map, keys))


def load_sanitized_tensors(model, source: Path, weight_map: dict[str, str], keys: Iterable[str]):
    return sanitize_chunk(model, load_source_tensors(source, weight_map, keys))

