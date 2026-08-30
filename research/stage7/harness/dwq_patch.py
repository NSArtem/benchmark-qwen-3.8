"""Apply a small Stage 7 DWQ scale/bias patch to an MLX model in memory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn


def apply_dwq_patch(model: Any, path: str | Path | None) -> list[str]:
    """Apply a cumulative bounded-DWQ patch without copying model weights."""

    if path is None:
        return []
    patch_path = Path(path).resolve()
    if not patch_path.is_file():
        raise FileNotFoundError(patch_path)
    modules = dict(model.named_modules())
    payload = mx.load(str(patch_path))
    patched: dict[str, set[str]] = {}
    try:
        for key, value in payload.items():
            if not (key.endswith(".scales") or key.endswith(".biases")):
                raise ValueError(f"unexpected DWQ patch key: {key}")
            module_path, parameter = key.rsplit(".", 1)
            module = modules.get(module_path)
            if module is None:
                raise KeyError(f"patch module not present in model: {module_path}")
            if not isinstance(module, nn.QuantizedLinear):
                raise TypeError(f"patch target is not QuantizedLinear: {module_path}")
            current = getattr(module, parameter, None)
            if current is None or list(current.shape) != list(value.shape):
                raise ValueError(f"patch shape mismatch for {key}: {value.shape}")
            setattr(module, parameter, value.astype(current.dtype))
            patched.setdefault(module_path, set()).add(parameter)
        incomplete = [path for path, parameters in patched.items() if parameters != {"scales", "biases"}]
        if incomplete:
            raise ValueError(f"DWQ patch is missing scales or biases: {incomplete}")
        for module_path in patched:
            module = modules[module_path]
            mx.eval(module.scales, module.biases)
    finally:
        del payload
    return sorted(patched)
