import unittest

import mlx.core as mx
import numpy as np

from stage10.qformat import hadamard_rotate, make_signs, pack, rtn


class PackTest(unittest.TestCase):
    def test_pack_matches_dequantize(self):
        rng = np.random.default_rng(0)
        out, cols, group = 64, 1024, 128
        for bits in (2, 3, 4):
            with self.subTest(bits=bits):
                q = rng.integers(0, 2**bits, (out, cols)).astype(np.uint8)
                scales = rng.standard_normal((out, cols // group)).astype(np.float32)
                biases = rng.standard_normal((out, cols // group)).astype(np.float32)
                got = mx.dequantize(
                    pack(q, bits), mx.array(scales), mx.array(biases), group_size=group, bits=bits
                )
                # q * scale + bias with one final rounding to float32 (MLX uses a
                # fused multiply-add); float64 holds q * scale exactly.
                qg = q.astype(np.float64).reshape(out, cols // group, group)
                exact = qg * scales[..., None].astype(np.float64) + biases[..., None].astype(np.float64)
                expected = exact.astype(np.float32).reshape(out, cols)
                np.testing.assert_array_equal(np.asarray(got), expected)

    def test_pack_reproduces_mx_quantize_codes(self):
        w = mx.random.normal((32, 512), key=mx.random.key(1)).astype(mx.bfloat16)
        for bits in (2, 3, 4):
            with self.subTest(bits=bits):
                packed, scales, biases = rtn(w, bits)
                deq = mx.dequantize(packed, scales, biases, group_size=128, bits=bits)
                s = np.repeat(np.asarray(scales.astype(mx.float32)), 128, axis=-1)
                b = np.repeat(np.asarray(biases.astype(mx.float32)), 128, axis=-1)
                q = np.rint((np.asarray(deq.astype(mx.float32)) - b) / s).astype(np.uint8)
                self.assertEqual(np.asarray(pack(q, bits)).tobytes(), np.asarray(packed).tobytes())

    def test_pack_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            pack(np.full((1, 128), 8, dtype=np.uint8), 3)


class RotationTest(unittest.TestCase):
    def test_folded_linear_matches(self):
        rng = np.random.default_rng(2)
        w = mx.array(rng.standard_normal((256, 2048)).astype(np.float32))
        x = mx.array(rng.standard_normal((8, 2048)).astype(np.float32))
        signs = mx.array(make_signs(0, "layers.0.mlp.up_proj", 2048))
        expected = x @ w.T
        got = hadamard_rotate(x, 1024, signs) @ hadamard_rotate(w, 1024, signs).T
        rel = float(mx.max(mx.abs(got - expected)) / mx.max(mx.abs(expected)))
        self.assertLess(rel, 1e-5)

    def test_embedding_round_trip(self):
        rng = np.random.default_rng(3)
        table = mx.array(rng.standard_normal((16, 2048)).astype(np.float32))
        signs = mx.array(make_signs(0, "embed_tokens", 2048))
        back = hadamard_rotate(hadamard_rotate(table, 1024, signs), 1024, signs, inverse=True)
        self.assertLess(float(mx.max(mx.abs(back - table))), 1e-5)

    def test_signs_are_deterministic(self):
        a = make_signs(0, "a", 64)
        np.testing.assert_array_equal(a, make_signs(0, "a", 64))
        self.assertFalse(np.array_equal(a, make_signs(1, "a", 64)))
        self.assertEqual(set(np.unique(a)), {-1.0, 1.0})
        self.assertEqual(a.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
