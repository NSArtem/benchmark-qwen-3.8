"""A5: MLX affine packing, RTN, and signed blockwise Hadamard rotation."""

from __future__ import annotations

import math
import zlib

import numpy as np


GROUP_SIZE = 128
SUPPORTED_BITS = (2, 3, 4)


def pack(q, bits: int):
    """Pack integer codes ``q`` [out, in] into the uint32 layout of ``mx.quantize``.

    MLX stores each row as a little-endian bit stream: code ``k`` occupies
    bits ``k*bits .. (k+1)*bits-1``. For 3 bits this is 8 codes per 3 bytes.
    """
    import mlx.core as mx

    if bits not in SUPPORTED_BITS:
        raise ValueError(f"unsupported bits {bits}")
    codes = np.asarray(q, dtype=np.uint8)
    if codes.ndim != 2 or (codes.shape[1] * bits) % 32:
        raise ValueError(f"cannot pack shape {codes.shape} at {bits} bits")
    if codes.size and int(codes.max()) >= 2**bits:
        raise ValueError(f"code out of range for {bits} bits")
    rows, cols = codes.shape
    out = np.empty((rows, cols * bits // 32), dtype=np.uint32)
    shifts = np.arange(bits, dtype=np.uint8)
    step = max(1, (1 << 26) // max(1, cols * bits))
    for start in range(0, rows, step):
        block = codes[start : start + step]
        stream = ((block[..., None] >> shifts) & 1).reshape(block.shape[0], -1)
        packed = np.packbits(stream, axis=-1, bitorder="little")
        out[start : start + step] = packed.view("<u4")
    return mx.array(out)


def rtn(w, bits: int, group_size: int = GROUP_SIZE):
    import mlx.core as mx

    return mx.quantize(w, group_size=group_size, bits=bits, mode="affine")


def hadamard_rotate(x, block, signs, inverse=False, dtype=None):
    """Signed, normalized blockwise Walsh-Hadamard rotation of the last axis.

    Copied from mlx-lm ``models/prism_hadamard_qwen35.py``; the compute dtype
    is a parameter here (default float32).
    """
    import mlx.core as mx

    dtype_compute = mx.float32 if dtype is None else dtype
    shape, out_dtype = x.shape, x.dtype
    x = x.astype(dtype_compute)
    if not inverse:
        x = x * signs.astype(dtype_compute)
    x = mx.hadamard_transform(x.reshape(-1, block), scale=1 / math.sqrt(block))
    x = x.reshape(shape)
    if inverse:
        x = x * signs.astype(dtype_compute)
    return x.astype(out_dtype)


def make_signs(seed: int, module_path: str, n: int) -> np.ndarray:
    rng = np.random.default_rng([seed, zlib.crc32(module_path.encode())])
    return (rng.integers(0, 2, n) * 2 - 1).astype(np.float32)


def rotate_weight(w, block: int, signs):
    """Fold the input rotation into ``w`` [out, in] in float32; returns float32."""
    import mlx.core as mx

    return hadamard_rotate(w.astype(mx.float32), block, signs)
