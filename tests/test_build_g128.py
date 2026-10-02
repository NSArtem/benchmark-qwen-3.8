"""Source-only g128 builder planning, without weights or Metal.

``tests/fixtures/qwen38-source`` holds the real ``config.json`` and
``generation_config.json`` of ``Qwen/Qwen3.8-27B`` at the pinned source
revision; their Hub hashes are checked below.  The pinned mlx-lm model is
instantiated with lazy parameters only, so no weight memory is allocated.

Tests that need the complete BF16 snapshot (tokenizer files and safetensor
headers) read it from ``QWEN38_SOURCE`` (default: the Hugging Face cache) and
are skipped when it is absent.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import re
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "tools" / "build_g128.py"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "qwen38-source"


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_g128", BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load_builder()
SOURCE = Path(
    os.environ.get(
        "QWEN38_SOURCE",
        Path.home()
        / ".cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots"
        / build.SOURCE_REVISION,
    )
)
HAVE_SOURCE = all((SOURCE / name).is_file() for name in build.SOURCE_METADATA)


def _source_config() -> dict:
    return json.loads((FIXTURES / "config.json").read_text(encoding="utf-8"))


def _generation_config() -> dict:
    return json.loads((FIXTURES / "generation_config.json").read_text(encoding="utf-8"))


def _derived() -> dict:
    return build.derive_text_config(_source_config(), _generation_config())


def _source_key(output_key: str) -> str:
    if output_key.startswith("language_model.model."):
        return output_key.replace("language_model.model", "model.language_model", 1)
    return output_key.removeprefix("language_model.")


class FixtureTests(unittest.TestCase):
    def test_fixtures_are_the_pinned_source_files(self) -> None:
        for name in ("config.json", "generation_config.json"):
            kind, expected = build.SOURCE_METADATA[name]
            self.assertEqual(build.file_digest(FIXTURES / name, kind), expected, name)


class DeriveTextConfigTests(unittest.TestCase):
    def test_exact_changes_from_source(self) -> None:
        source = _source_config()
        derived = _derived()
        self.assertEqual(
            set(source) ^ set(derived),
            {"vision_config", "eos_token_id", "quantization", "quantization_config"},
        )
        self.assertNotIn("vision_config", derived)
        self.assertEqual(derived["eos_token_id"], [248046, 248044])
        self.assertEqual(derived["quantization"], {"group_size": 128, "bits": 3, "mode": "affine"})
        self.assertEqual(derived["quantization_config"], derived["quantization"])
        for key in set(source) & set(derived) - {"text_config"}:
            self.assertEqual(derived[key], source[key], key)

        text, source_text = derived["text_config"], source["text_config"]
        self.assertEqual(set(source_text) - set(text), set(build.MTP_TEXT_CONFIG_KEYS))
        self.assertFalse(set(text) - set(source_text))
        for key in set(text) - {"rope_parameters"}:
            self.assertEqual(text[key], source_text[key], key)
        rope = dict(source_text["rope_parameters"])
        rope["type"] = rope.pop("rope_type")
        self.assertEqual(text["rope_parameters"], rope)

    def test_nested_text_config_kept_for_pinned_mlx_lm(self) -> None:
        derived = _derived()
        self.assertEqual(derived["model_type"], "qwen3_5")
        self.assertEqual(derived["architectures"], ["Qwen3_5ForConditionalGeneration"])
        self.assertEqual(derived["text_config"]["model_type"], "qwen3_5_text")
        self.assertEqual(derived["text_config"]["num_hidden_layers"], 64)

    def test_deterministic_sorted_and_pure(self) -> None:
        source = _source_config()
        before = copy.deepcopy(source)
        first = build.derive_text_config(source, _generation_config())
        second = build.derive_text_config(source, _generation_config())
        self.assertEqual(source, before)
        self.assertEqual(json.dumps(first), json.dumps(second))
        self.assertEqual(list(first), sorted(first))

    def test_rejects_unrecognized_sources(self) -> None:
        def mutated(edit):
            config = _source_config()
            edit(config)
            return config

        cases = {
            "model_type": lambda c: c.update(model_type="qwen3"),
            "flat": lambda c: c.pop("text_config"),
            "tied": lambda c: c.update(tie_word_embeddings=True),
            "layers": lambda c: c["text_config"].update(num_hidden_layers=48),
            "moe": lambda c: c["text_config"].update(num_experts=64),
            "mix": lambda c: c["text_config"]["layer_types"].__setitem__(0, "full_attention"),
        }
        for name, edit in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                build.derive_text_config(mutated(edit), _generation_config())


class SourceKeyTests(unittest.TestCase):
    def test_select_text_tensors(self) -> None:
        weight_map = {
            "lm_head.weight": "a",
            "model.language_model.embed_tokens.weight": "a",
            "model.language_model.layers.0.mlp.up_proj.weight": "b",
            "model.visual.blocks.0.attn.qkv.weight": "b",
            "model.visual.merger.linear_fc1.weight": "c",
            "mtp.fc.weight": "c",
        }
        kept, dropped = build.select_text_tensors(weight_map)
        self.assertEqual(
            set(kept),
            {
                "lm_head.weight",
                "model.language_model.embed_tokens.weight",
                "model.language_model.layers.0.mlp.up_proj.weight",
            },
        )
        self.assertEqual(dropped, {"visual": 2, "mtp": 1})

    def test_unknown_tensor_family_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            build.select_text_tensors({"lm_head.weight": "a", "model.audio.x.weight": "a"})

    def test_output_key_mapping(self) -> None:
        self.assertEqual(build.output_key("lm_head.weight"), "language_model.lm_head.weight")
        self.assertEqual(
            build.output_key("model.language_model.layers.7.linear_attn.A_log"),
            "language_model.model.layers.7.linear_attn.A_log",
        )

    def test_chunk_plan(self) -> None:
        chunks = build.plan_chunks(
            {
                "model.language_model.layers.10.mlp.up_proj.weight": "b",
                "lm_head.weight": "a",
                "model.language_model.norm.weight": "a",
                "model.language_model.layers.2.mlp.up_proj.weight": "a",
            }
        )
        self.assertEqual([layer for layer, _ in chunks], [None, 2, 10])
        self.assertEqual(chunks[0][1], ["lm_head.weight", "model.language_model.norm.weight"])


class SourceMetadataTests(unittest.TestCase):
    def test_missing_files_name_the_download_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError) as caught:
                build.verify_source_metadata(Path(tmp))
        message = str(caught.exception)
        self.assertIn(f"--revision {build.SOURCE_REVISION}", message)
        # hf ignores --include when bare filenames follow, so each needs a flag.
        for name in build.SOURCE_METADATA:
            self.assertIn(f"--include {name}", message)

    def test_modified_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for name in build.SOURCE_METADATA:
                (Path(tmp) / name).write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                build.verify_source_metadata(Path(tmp))


class PinnedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = build.build_model(_derived())
        cls.sizes = build.parameter_sizes(cls.model)

    def test_layer_layout(self) -> None:
        layers = self.model.language_model.layers
        self.assertEqual(len(layers), 64)
        self.assertEqual(sum(1 for layer in layers if layer.is_linear), 48)

    def test_quantization_key_count(self) -> None:
        keys = build.quantization_keys(self.model)
        self.assertEqual(len(keys), build.QUANTIZED_MODULES)
        self.assertIn("language_model.lm_head.weight", keys)
        self.assertIn("language_model.model.embed_tokens.weight", keys)
        self.assertLessEqual(keys, set(self.sizes))

    def test_total_parameters(self) -> None:
        self.assertEqual(build.total_parameters(self.model), 26_895_993_856)
        self.assertEqual(sum(self.sizes.values()), 26_895_998_464)
        direct = sum(v for k, v in self.sizes.items() if k.endswith((".A_log", ".dt_bias")))
        self.assertEqual(direct, 48 * (48 + 48))

    def test_build_does_not_mutate_config(self) -> None:
        config = _derived()
        before = copy.deepcopy(config)
        build.build_model(config)
        self.assertEqual(config, before)

    def test_source_layout_maps_onto_model(self) -> None:
        weight_map = {_source_key(key): "shard" for key in self.sizes}
        self.assertEqual(len(weight_map), build.RETAINED_SOURCE_TENSORS)
        weight_map.update({"model.visual.patch_embed.proj.weight": "v", "mtp.fc.weight": "m"})
        kept, dropped = build.select_text_tensors(weight_map)
        self.assertEqual(dropped, {"visual": 1, "mtp": 1})
        emitted = {build.output_key(key) for key in kept}
        self.assertEqual(emitted, set(self.sizes))
        self.assertFalse([key for key in emitted if "visual" in key or "mtp." in key])

        headers = {
            _source_key(key): {"shape": [size], "dtype": "BF16", "data_bytes": 2 * size}
            for key, size in self.sizes.items()
        }
        plan = build.verify_model_plan(self.model, kept, headers)
        self.assertEqual(plan["quantization_keys"], 498)
        self.assertEqual(plan["uncounted_direct_parameters"], 4608)

    def test_global_norm_shift_survives_chunking(self) -> None:
        import mlx.core as mx

        with mx.stream(mx.cpu):
            shifted = build.sanitize_chunk(
                self.model,
                {
                    "model.language_model.layers.3.input_layernorm.weight": mx.ones(
                        (4,), dtype=mx.bfloat16
                    )
                },
            )
            value = float(shifted["language_model.model.layers.3.input_layernorm.weight"][0])
        self.assertEqual(set(shifted), {"language_model.model.layers.3.input_layernorm.weight"})
        self.assertEqual(value, 2.0)


class IndependenceTests(unittest.TestCase):
    def test_no_existing_checkpoint_dependency(self) -> None:
        text = BUILDER.read_text(encoding="utf-8")
        for needle in ("baseline", "lukaskremla", "/Users/artem/models", "MLX-TextOnly"):
            self.assertNotIn(needle, text)

    def test_no_wired_limit_calls(self) -> None:
        text = BUILDER.read_text(encoding="utf-8")
        self.assertIsNone(re.search(r"set_wired_limit\s*\(", text))
        self.assertTrue(build.no_wired_limit_calls())


@unittest.skipUnless(HAVE_SOURCE, f"BF16 source snapshot not found at {SOURCE}")
class RealSourceTests(unittest.TestCase):
    def test_tokenizer_assets_match_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            written = build.write_tokenizer_assets(SOURCE, Path(tmp))
            result = build.verify_tokenizer_assets(SOURCE, Path(tmp))
            files = sorted(p.name for p in Path(tmp).iterdir())
            tokenizer_config = json.loads((Path(tmp) / "tokenizer_config.json").read_text())
        self.assertEqual(files, written)
        self.assertEqual(result["tokenizer_class"], "TokenizersBackend")
        self.assertEqual(result["tool_parser_type"], "qwen3_coder")
        self.assertNotIn("local_files_only", tokenizer_config)
        self.assertNotIn("chat_template", tokenizer_config)

    def test_self_test(self) -> None:
        result = build.self_test(SOURCE)
        self.assertEqual(result["status"], "SOURCE_SELF_TEST_PASS")
        self.assertEqual(result["retained_source_tensors"], 851)
        self.assertEqual(result["dropped_source_tensors"], {"visual": 333, "mtp": 15})
        self.assertEqual(result["quantization_keys"], 498)
        self.assertEqual(result["total_parameters"], 26_895_993_856)


if __name__ == "__main__":
    unittest.main()
