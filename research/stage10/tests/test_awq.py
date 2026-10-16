import unittest

import mlx.core as mx
from mlx.utils import tree_flatten

from stage10.awq import ALPHAS, InputCapture, awq_scales, fold_scales, group_mse, search_alpha
from stage10.gptq import input_group, linear_children, run_layer_batches
from stage10.qformat import rtn


class AWQTest(unittest.TestCase):
    def test_alpha_grid_and_normalization(self):
        self.assertEqual(ALPHAS, (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
        X = mx.random.normal((64, 256), key=mx.random.key(0)) * mx.exp(mx.random.normal((256,), key=mx.random.key(1)))
        s = awq_scales(X, 0.5)
        self.assertAlmostEqual(float(mx.max(s) * mx.min(s)), 1.0, places=4)
        self.assertTrue(bool(mx.all(awq_scales(X, 0.0) == 1.0)))

    def test_alpha_zero_is_plain_rtn(self):
        X = mx.random.normal((128, 256), key=mx.random.key(2))
        W = mx.random.normal((32, 256), key=mx.random.key(3)).astype(mx.bfloat16)
        q, s, b = rtn(W, 3)
        Wq = mx.dequantize(q, s, b, group_size=128, bits=3).astype(mx.float32)
        expected = float(mx.mean((X @ W.astype(mx.float32).T - X @ Wq.T) ** 2))
        self.assertAlmostEqual(group_mse(X, [W], awq_scales(X, 0.0)), expected, places=5)
        alpha, mses, _ = search_alpha(X, [W])
        self.assertEqual(len(mses), 11)
        self.assertEqual(mses[ALPHAS.index(alpha)], min(mses))

    def test_folding_preserves_layer_function(self):
        from tests.test_candidate import TINY
        from stage10.layerwise import fresh_decoder_layer, text_args

        args = text_args(TINY)
        for layer_idx in (0, 1):
            with self.subTest(layer=layer_idx):
                mx.random.seed(10 + layer_idx)
                layer = fresh_decoder_layer(args, layer_idx)
                prefix = f"language_model.model.layers.{layer_idx}."
                weights = {prefix + k: v for k, v in tree_flatten(layer.parameters())}
                # Non-trivial norm weights, as after sanitize.
                for k in list(weights):
                    if k.endswith("layernorm.weight"):
                        weights[k] = 1.0 + 0.1 * mx.random.normal(weights[k].shape)
                layer = fresh_decoder_layer(args, layer_idx, weights)
                hidden = [mx.random.normal((2, 16, 256))]
                reference = run_layer_batches(layer, hidden, args, layer_idx)[0]
                capture = InputCapture()
                run_layer_batches(layer, hidden, args, layer_idx, hook=capture, keep=False)
                members = {}
                for local in sorted(linear_children(layer)):
                    members.setdefault(input_group(local), []).append(local)
                for group in members:
                    if group in ("linear_attn.in", "self_attn.in", "mlp.in", "mlp.down_proj"):
                        s = awq_scales(capture.inputs(group), 0.7)
                        fold_scales(weights, layer_idx, group, members[group], s)
                folded = fresh_decoder_layer(args, layer_idx, weights)
                out = run_layer_batches(folded, hidden, args, layer_idx)[0]
                rel = float(mx.max(mx.abs(out - reference)) / mx.max(mx.abs(reference)))
                self.assertLess(rel, 1e-4)


if __name__ == "__main__":
    unittest.main()
