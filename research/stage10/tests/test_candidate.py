import json
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.utils import _get_classes

from stage10.candidate import (
    CandidateWriter,
    build_model_shell,
    candidate_config,
    load_candidate,
    module_record,
    quantize_module,
)


TINY = {
    "model_type": "qwen3_5",
    "tie_word_embeddings": False,
    "text_config": {
        "model_type": "qwen3_5_text",
        "hidden_size": 256,
        "intermediate_size": 512,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 64,
        "rms_norm_eps": 1e-6,
        "vocab_size": 384,
        "linear_num_value_heads": 2,
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 64,
        "linear_value_head_dim": 64,
        "linear_conv_kernel_dim": 4,
        "full_attention_interval": 2,
        "tie_word_embeddings": False,
        "rope_parameters": {
            "type": "default",
            "mrope_section": [3, 3, 2],
            "rope_theta": 10000,
            "partial_rotary_factor": 0.25,
        },
    },
}


def float_model():
    model_class, args_class = _get_classes(TINY)
    mx.random.seed(0)
    model = model_class(args_class.from_dict(TINY))
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    return model


def quantizable_paths(model):
    return [
        path
        for path, module in model.named_modules()
        if isinstance(module, (nn.Linear, nn.Embedding)) and path
    ]


class CandidateTest(unittest.TestCase):
    def roundtrip(self, rotated: bool):
        model = float_model()
        records = []
        for i, path in enumerate(sorted(quantizable_paths(model))):
            bits = (2, 3, 4)[i % 3]
            block = (128 if i % 2 else 0) if rotated else 0
            records.append(module_record(path, bits, block))
        config = candidate_config(TINY, "T0", "tiny", 0, records)

        modules = dict(model.named_modules())
        quantized = {}
        for record in records:
            arrays = quantize_module(modules[record["path"]].weight, record, seed=0)
            for name, value in arrays.items():
                quantized[f"{record['path']}.{name}"] = value
        others = {
            key: value
            for key, value in tree_flatten(model.parameters())
            if not any(key.startswith(r["path"] + ".") for r in records)
        }

        # Direct in-memory model with the same quantized modules.
        direct = build_model_shell(config)
        direct.load_weights(list({**others, **quantized}.items()), strict=True)
        direct.eval()

        with tempfile.TemporaryDirectory() as tmp:
            writer = CandidateWriter(Path(tmp), max_shard_bytes=200_000)
            for key, value in {**others, **quantized}.items():
                writer.add(key, value)
            writer.finish(config, tokenizer_dir=None)
            self.assertGreater(len(list(Path(tmp).glob("*.safetensors"))), 1)
            saved = json.loads((Path(tmp) / "config.json").read_text())
            self.assertNotIn("quantization", saved)
            self.assertEqual(saved["stage10"]["modules"], records)
            loaded, _ = load_candidate(tmp)

        tokens = mx.array(np.random.default_rng(1).integers(0, 384, (2, 40)), dtype=mx.int32)
        expected = direct(tokens)
        got = loaded(tokens)
        np.testing.assert_array_equal(
            np.asarray(got.astype(mx.float32)), np.asarray(expected.astype(mx.float32))
        )

        # The quantized model stays close to the float model (sanity, not a bound).
        ref = model(tokens).astype(mx.float32)
        self.assertTrue(bool(mx.all(mx.isfinite(got))))
        return float(mx.mean(mx.abs(got.astype(mx.float32) - ref)))

    def test_unrotated_mixed_bits(self):
        self.roundtrip(rotated=False)

    def test_rotated_mixed_bits(self):
        self.roundtrip(rotated=True)

    def test_strict_missing_key(self):
        model = float_model()
        path = sorted(quantizable_paths(model))[0]
        config = candidate_config(TINY, "T0", "tiny", 0, [module_record(path, 3, 0)])
        params = dict(tree_flatten(model.parameters()))
        params.pop(f"{path}.weight")
        with tempfile.TemporaryDirectory() as tmp:
            writer = CandidateWriter(Path(tmp))
            for key, value in params.items():
                writer.add(key, value)
            # The writer refuses this candidate; write it past that guard so the
            # loader's own strict check is what is tested here.
            with self.assertRaises(ValueError):
                writer.finish(config, tokenizer_dir=None)
            writer.finish(candidate_config(TINY, "T0", "tiny", 0, []), tokenizer_dir=None)
            (Path(tmp) / "config.json").write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                load_candidate(tmp)

    def test_finish_rejects_missing_module_tensors(self):
        path = "language_model.model.layers.0.mlp.down_proj"
        config = candidate_config(TINY, "T0", "tiny", 0, [module_record(path, 3, 0)])
        with tempfile.TemporaryDirectory() as tmp:
            writer = CandidateWriter(Path(tmp))
            writer.add("language_model.model.norm.weight", mx.ones((256,)))
            with self.assertRaises(ValueError):
                writer.finish(config, tokenizer_dir=None)

    def test_rotate_dtype_override(self):
        model = float_model()
        path = "language_model.model.layers.0.mlp.down_proj"
        config = candidate_config(TINY, "T0", "tiny", 0, [module_record(path, 3, 128)])
        shell = build_model_shell(config, rotate_dtype="bfloat16")
        module = dict(shell.named_modules())[path]
        self.assertEqual(module.rotate_dtype, "bfloat16")

    def test_row_chunking_is_exact(self):
        import stage10.candidate as candidate

        w = mx.random.normal((300, 256), key=mx.random.key(5)).astype(mx.bfloat16)
        for block in (0, 128):
            record = module_record("language_model.lm_head", 3, block)
            whole = quantize_module(w, record, seed=0)
            old = candidate.ROW_CHUNK
            candidate.ROW_CHUNK = 64
            try:
                chunked = quantize_module(w, record, seed=0)
            finally:
                candidate.ROW_CHUNK = old
            for key in whole:
                self.assertEqual(
                    np.asarray(whole[key].view(mx.uint8)).tobytes(),
                    np.asarray(chunked[key].view(mx.uint8)).tobytes(),
                )


if __name__ == "__main__":
    unittest.main()
