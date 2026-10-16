import unittest

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from stage10.gptq import (
    HessianHook,
    gptq_module,
    gptq_quantize_weight,
    install_quantized,
    inverse_hessian_factor,
    linear_children,
    output_mse,
    rotate_hessian,
    run_layer_batches,
)
from stage10.qformat import hadamard_rotate, make_signs, pack, rtn


def correlated_inputs(rows, width, seed):
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((rows, width // 8)).astype(np.float32)
    mix = rng.standard_normal((width // 8, width)).astype(np.float32)
    noise = 0.1 * rng.standard_normal((rows, width)).astype(np.float32)
    scale = np.exp(rng.standard_normal(width)).astype(np.float32)
    return mx.array((base @ mix + noise) * scale)


class _OneLinear(nn.Module):
    def __init__(self, width, out):
        super().__init__()
        self.lin = nn.Linear(width, out, bias=False)

    def __call__(self, x):
        return self.lin(x)


class GPTQTest(unittest.TestCase):
    def test_matches_mlx_lm_gptq_at_4_bits(self):
        from mlx_lm.quant.gptq import gptq_quantize

        width, out = 256, 48
        mx.random.seed(0)
        model = _OneLinear(width, out)
        W = model.lin.weight
        mx.eval(W)
        data = correlated_inputs(4 * 64, width, 1).reshape(4, 64, width)
        H = mx.zeros((width, width))
        for start in range(0, 4, 2):
            xf = data[start : start + 2].reshape(-1, width)
            H = H + xf.T @ xf
        mx.eval(H)
        model, _ = gptq_quantize(model, data, 4, 128, 4, 128, batch_size=2)
        stock = mx.dequantize(model.lin.weight, model.lin.scales, model.lin.biases, group_size=128, bits=4)

        codes, scales, biases = gptq_quantize_weight(W, inverse_hessian_factor(H), 4)
        mine = mx.dequantize(pack(codes, 4), scales, biases, group_size=128, bits=4)
        diff = float(mx.max(mx.abs(mine - stock)))
        agree = float(mx.mean(mx.abs(mine - stock) < 1e-6))
        self.assertGreaterEqual(agree, 0.999, f"max diff {diff}")

    def test_gptq_beats_rtn_on_calibration_inputs(self):
        width, out = 512, 64
        X = correlated_inputs(2048, width, 2)
        W = mx.random.normal((out, width), key=mx.random.key(3)).astype(mx.bfloat16)
        H = X.T @ X
        for bits in (2, 3, 4):
            with self.subTest(bits=bits):
                q, s, b = rtn(W, bits)
                arrays, info = gptq_module(W, H, bits)
                self.assertFalse(info["fallback_rtn"])
                self.assertEqual(arrays["scales"].dtype, mx.bfloat16)
                before = output_mse(X, W, {"weight": q, "scales": s, "biases": b}, bits)
                after = output_mse(X, W, arrays, bits)
                self.assertLess(after, before)

    def test_rotated_hessian(self):
        X = correlated_inputs(300, 2048, 4)
        signs = mx.array(make_signs(0, "m", 2048))
        Xr = hadamard_rotate(X, 1024, signs)
        expected = Xr.T @ Xr
        got = rotate_hessian(X.T @ X, 1024, signs)
        rel = float(mx.max(mx.abs(got - expected)) / mx.max(mx.abs(expected)))
        self.assertLess(rel, 1e-4)

    def test_singular_hessian_falls_back_or_retries(self):
        W = mx.random.normal((16, 128), key=mx.random.key(6)).astype(mx.bfloat16)
        H = mx.zeros((128, 128))
        messages = []
        arrays, info = gptq_module(W, H, 3, log=messages.append, name="zero")
        self.assertTrue(info["fallback_rtn"])
        self.assertEqual(info["retries"], 3)
        self.assertEqual(len(messages), 4)
        self.assertEqual(arrays["weight"].shape, (16, 128 * 3 // 32))


class LayerPlumbingTest(unittest.TestCase):
    def test_hook_groups_and_quantized_layer(self):
        from tests.test_candidate import TINY
        from stage10.candidate import module_record, quantize_module
        from stage10.layerwise import fresh_decoder_layer, text_args

        args = text_args(TINY)
        for layer_idx in (0, 1):
            with self.subTest(layer=layer_idx):
                mx.random.seed(layer_idx)
                layer = fresh_decoder_layer(args, layer_idx)
                layer.set_dtype(mx.bfloat16)
                hidden = [mx.random.normal((2, 32, 256)).astype(mx.bfloat16) for _ in range(2)]
                hook = HessianHook(keep_sequences=3)
                hook.start()
                run_layer_batches(layer, hidden, args, layer_idx, hook=hook, keep=False)
                expected_groups = {"mlp.in", "mlp.down_proj"} | (
                    {"linear_attn.in", "linear_attn.out_proj"} if layer_idx == 0 else {"self_attn.in", "self_attn.o_proj"}
                )
                self.assertEqual(set(hook.H), expected_groups)
                self.assertEqual(hook.inputs("mlp.down_proj").shape[0], 3 * 32)
                # The shared-input Hessian equals Σ XᵀX of the post-attention norm input.
                attn_in = [layer.input_layernorm(h).reshape(-1, 256).astype(mx.float32) for h in hidden]
                group = "linear_attn.in" if layer_idx == 0 else "self_attn.in"
                ref = sum(x.T @ x for x in attn_in)
                self.assertLess(float(mx.max(mx.abs(hook.H[group] - ref)) / mx.max(mx.abs(ref))), 1e-3)

                records, quantized = {}, {}
                for local, module in linear_children(layer).items():
                    path = f"language_model.model.layers.{layer_idx}.{local}"
                    records[path] = module_record(path, 3, 128)
                    quantized[local] = quantize_module(module.weight, records[path], seed=0)
                reference = run_layer_batches(layer, hidden, args, layer_idx)
                install_quantized(layer, layer_idx, quantized, records)
                out = run_layer_batches(layer, hidden, args, layer_idx)
                self.assertTrue(all(type(m).__name__ == "QLinear" for _, m in layer.named_modules() if "proj" in _.split(".")[-1]))
                err = float(mx.mean(mx.abs(out[0].astype(mx.float32) - reference[0].astype(mx.float32))))
                self.assertTrue(np.isfinite(err))


if __name__ == "__main__":
    unittest.main()
