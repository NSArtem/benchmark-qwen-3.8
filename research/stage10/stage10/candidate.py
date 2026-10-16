"""A6: Stage 10 candidate checkpoint format, writer, and loader.

``QLinear``/``QEmbedding`` are based on ``HadamardQuantizedLinear`` and
``HadamardQuantizedEmbedding`` in mlx-lm ``models/prism_hadamard_qwen35.py``,
without the fixed 2-bit check: each module carries its own bits, group size,
Hadamard block (0 = not rotated), and rotation compute dtype.
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

import mlx.nn as nn

from .qformat import hadamard_rotate, make_signs, rotate_weight


MAX_SHARD_BYTES = 2_000_000_000
ROW_CHUNK = 16384
PRODUCTION = Path("/Users/artem/models/Qwen3.8-27B-3bit-g128-MLX")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json")
ROTATE_DTYPES = ("float32", "bfloat16", "float16")


def _mx_dtype(name: str):
    import mlx.core as mx

    if name not in ROTATE_DTYPES:
        raise ValueError(f"unsupported rotate dtype {name!r}")
    return getattr(mx, name)


class QLinear(nn.Module):
    """Affine-quantized linear layer with an optional folded input rotation."""

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bits: int,
        group_size: int,
        block: int,
        rotate_dtype: str = "float32",
        scale_dtype=None,
    ):
        import mlx.core as mx

        super().__init__()
        if input_dims % group_size or (input_dims * bits) % 32:
            raise ValueError(f"invalid quantized width {input_dims} for bits={bits}, group={group_size}")
        if block and input_dims % block:
            raise ValueError(f"Hadamard block {block} does not divide width {input_dims}")
        scale_dtype = scale_dtype or mx.bfloat16
        self.bits = bits
        self.group_size = group_size
        self.block = block
        self.rotate_dtype = rotate_dtype
        self._rotate_mx_dtype = _mx_dtype(rotate_dtype)
        self.weight = mx.zeros((output_dims, input_dims * bits // 32), dtype=mx.uint32)
        self.scales = mx.zeros((output_dims, input_dims // group_size), dtype=scale_dtype)
        self.biases = mx.zeros_like(self.scales)
        if block:
            self.signs = mx.ones((input_dims,), dtype=mx.float32)
        self.freeze()

    def _rotate(self, x, inverse=False):
        return hadamard_rotate(x, self.block, self.signs, inverse=inverse, dtype=self._rotate_mx_dtype)

    def __call__(self, x):
        import mlx.core as mx

        if self.block:
            x = self._rotate(x)
        return mx.quantized_matmul(
            x,
            self.weight,
            self.scales,
            self.biases,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
        )


class QEmbedding(QLinear):
    """Dequantizes the selected rows and rotates them back to the embedding basis."""

    def __call__(self, indices):
        import mlx.core as mx

        shape = indices.shape
        indices = indices.reshape(-1)
        out = mx.dequantize(
            self.weight[indices],
            self.scales[indices],
            self.biases[indices],
            group_size=self.group_size,
            bits=self.bits,
        )
        out = out.reshape(*shape, -1).astype(self.scales.dtype)
        if self.block:
            out = self._rotate(out, inverse=True)
        return out

    def as_linear(self, x):
        return QLinear.__call__(self, x)


def module_record(path: str, bits: int, block: int, group_size: int = 128, rotate_dtype: str = "float32"):
    return {"path": path, "bits": bits, "group_size": group_size, "block": block, "rotate_dtype": rotate_dtype}


def quantize_module(weight, record: dict[str, Any], seed: int, *, quantizer=None) -> dict[str, Any]:
    """RTN (or ``quantizer``) for one module, rotated first when ``block > 0``.

    Unrotated weights are cast to bfloat16 and quantized, exactly as
    ``tools/build_g128.py`` does. Rotated weights are rotated in float32, cast
    to bfloat16, then quantized. Returns ``weight``/``scales``/``biases``
    (and ``signs``) arrays without the module prefix.
    """
    import mlx.core as mx

    from .qformat import rtn

    block = record["block"]
    out: dict[str, Any] = {}
    signs = None
    if block:
        signs = mx.array(make_signs(seed, record["path"], weight.shape[-1]))
        out["signs"] = signs
    if quantizer is not None:
        if block:
            weight = rotate_weight(weight, block, signs)
        q, scales, biases = quantizer(weight.astype(mx.bfloat16), record)
        out.update(weight=q, scales=scales, biases=biases)
        return out
    # Rows are independent for rotation and RTN, so large tables go in chunks.
    pieces = []
    for start in range(0, weight.shape[0], ROW_CHUNK):
        rows = weight[start : start + ROW_CHUNK]
        if block:
            rows = rotate_weight(rows, block, signs)
        piece = rtn(rows.astype(mx.bfloat16), record["bits"], record["group_size"])
        mx.eval(*piece)
        pieces.append(piece)
        del rows
    if len(pieces) == 1:
        q, scales, biases = pieces[0]
    else:
        q, scales, biases = (mx.concatenate(parts, axis=0) for parts in zip(*pieces))
    out.update(weight=q, scales=scales, biases=biases)
    return out


def candidate_config(base_config: dict[str, Any], exp_id: str, recipe: str, seed: int, modules: list[dict]) -> dict:
    config = copy.deepcopy(base_config)
    config.pop("quantization", None)
    config.pop("quantization_config", None)
    config["stage10"] = {"exp_id": exp_id, "recipe": recipe, "seed": seed, "modules": modules}
    return config


class CandidateWriter:
    """Writes safetensors shards of at most ``MAX_SHARD_BYTES`` and an index."""

    def __init__(self, output: Path, max_shard_bytes: int = MAX_SHARD_BYTES):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        if any(self.output.glob("*.safetensors")):
            raise FileExistsError(f"{self.output} already contains safetensors shards")
        self.max_shard_bytes = max_shard_bytes
        self.pending: dict[str, Any] = {}
        self.pending_bytes = 0
        self.shards: list[tuple[str, list[str]]] = []
        self.total_bytes = 0

    def add(self, key: str, value) -> None:
        import mlx.core as mx

        mx.eval(value)
        size = int(value.nbytes)
        if self.pending and self.pending_bytes + size > self.max_shard_bytes:
            self.flush()
        self.pending[key] = value
        self.pending_bytes += size
        self.total_bytes += size

    def flush(self) -> None:
        import mlx.core as mx

        if not self.pending:
            return
        name = f"model-{len(self.shards) + 1:05d}.safetensors"
        mx.save_safetensors(str(self.output / name), self.pending, metadata={"format": "mlx"})
        self.shards.append((name, sorted(self.pending)))
        self.pending = {}
        self.pending_bytes = 0
        mx.clear_cache()

    def finish(self, config: dict[str, Any], tokenizer_dir: Path | None = PRODUCTION) -> None:
        self.flush()
        weight_map = {}
        for name, keys in self.shards:
            final = name.replace(".safetensors", f"-of-{len(self.shards):05d}.safetensors")
            (self.output / name).rename(self.output / final)
            weight_map.update({key: final for key in keys})
        index = {"metadata": {"total_size": self.total_bytes}, "weight_map": dict(sorted(weight_map.items()))}
        (self.output / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
        (self.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        if tokenizer_dir is not None:
            for filename in TOKENIZER_FILES:
                if (tokenizer_dir / filename).is_file():
                    shutil.copy2(tokenizer_dir / filename, self.output / filename)


def build_model_shell(config: dict[str, Any], rotate_dtype: str | None = None):
    """Model from ``config`` with every Stage 10 module replaced by QLinear/QEmbedding."""
    from mlx.utils import tree_unflatten
    from mlx_lm.utils import _get_classes

    shell_config = {key: value for key, value in config.items() if key != "stage10"}
    model_class, args_class = _get_classes(shell_config)
    model = model_class(args_class.from_dict(shell_config))
    modules = dict(model.named_modules())
    replacements = []
    seen = set()
    for record in config["stage10"]["modules"]:
        path = record["path"]
        if path in seen:
            raise ValueError(f"duplicate candidate module: {path}")
        seen.add(path)
        original = modules.get(path)
        if not isinstance(original, (nn.Linear, nn.Embedding)):
            raise ValueError(f"unknown or unsupported candidate module: {path}")
        if "bias" in original:
            raise ValueError(f"linear with bias is unsupported: {path}")
        cls = QEmbedding if isinstance(original, nn.Embedding) else QLinear
        rows, width = original.weight.shape
        replacements.append(
            (
                path,
                cls(
                    width,
                    rows,
                    record["bits"],
                    record["group_size"],
                    record["block"],
                    rotate_dtype or record.get("rotate_dtype", "float32"),
                ),
            )
        )
    model.update_modules(tree_unflatten(replacements))
    return model


def candidate_weight_files(path: Path) -> list[Path]:
    return sorted(Path(path).glob("*.safetensors"))


def load_candidate(path: str | Path, rotate_dtype: str | None = None, *, lazy: bool = False):
    """Load a candidate directory; missing or extra weight keys are errors."""
    import mlx.core as mx

    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    model = build_model_shell(config, rotate_dtype)
    weights: dict[str, Any] = {}
    for shard in candidate_weight_files(path):
        weights.update(mx.load(str(shard)))
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    if not lazy:
        mx.eval(model.parameters())
    return model, config


def checkpoint_size(path: Path, modules: Iterable[dict] | None = None) -> dict[str, Any]:
    """``bytes``/``gib`` from shard file sizes; ``bpw`` from tensor payload bytes."""
    from .source import read_safetensors_header

    files = candidate_weight_files(path)
    file_bytes = sum(f.stat().st_size for f in files)
    payload = sum(meta["data_bytes"] for f in files for meta in read_safetensors_header(f).values())
    return {
        "path": str(path),
        "bytes": file_bytes,
        "gib": file_bytes / 2**30,
        "bpw": 8 * payload / 26_895_993_856,
        "payload_bytes": payload,
    }
