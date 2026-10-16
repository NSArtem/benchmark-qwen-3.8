"""Canonical g128 manifest: generation, verification and corruption detection.

All corruption tests use tiny temporary checkpoints; no weights, MLX or
Metal.  ``QWEN38_G128_VERIFY=/path/to/checkpoint`` additionally verifies a
real checkpoint against the committed manifest (hashes ~10 GiB).
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "tools" / "build_g128.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_g128", BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load_builder()
SHARDS = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")


def write_checkpoint(directory: Path) -> None:
    """A tiny stand-in with the canonical file layout."""

    quantization = dict(build.QUANTIZATION)
    files = {
        "config.json": {"quantization": quantization, "quantization_config": quantization},
        "generation_config.json": {"eos_token_id": [1, 2]},
        "model.safetensors.index.json": {
            "metadata": {"total_size": 12, "total_parameters": 34},
            "weight_map": {
                "a.weight": SHARDS[0],
                "a.scales": SHARDS[0],
                "a.biases": SHARDS[0],
                "b.weight": SHARDS[1],
            },
        },
        "tokenizer.json": {"model": {"merges": []}},
        "tokenizer_config.json": {"tokenizer_class": "TokenizersBackend"},
    }
    for name, data in files.items():
        (directory / name).write_text(json.dumps(data, indent=4), encoding="utf-8")
    (directory / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
    (directory / SHARDS[0]).write_bytes(b"shard-one-bytes")
    (directory / SHARDS[1]).write_bytes(b"shard-two-bytes!")


class TinyCheckpointTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.model = Path(self._tmp.name) / "model"
        self.model.mkdir()
        write_checkpoint(self.model)
        self.manifest = build.calculate_manifest(self.model)

    def verify(self, **kwargs) -> dict:
        return build.verify_manifest(self.model, self.manifest, **kwargs)

    def assertFailsOn(self, result: dict, name: str, reason: str) -> str:
        self.assertFalse(result["ok"])
        matching = [f for f in result["failures"] if f.startswith(f"FAIL {name}: {reason}")]
        self.assertTrue(matching, result["failures"])
        return matching[0]


class CalculateManifestTests(TinyCheckpointTestCase):
    def test_shape(self) -> None:
        manifest = self.manifest
        self.assertEqual(manifest["schema_version"], build.MANIFEST_SCHEMA_VERSION)
        self.assertEqual(sorted(manifest["files"]), sorted(build.REQUIRED_OUTPUT_FILES + SHARDS))
        self.assertEqual(manifest["expected"]["total_files"], 8)
        self.assertEqual(manifest["expected"]["weight_shards"], 2)
        self.assertEqual(manifest["expected"]["tensor_count"], 4)
        self.assertEqual(manifest["expected"]["quantized_modules"], 1)
        self.assertEqual(
            manifest["expected"]["total_bytes"],
            sum(path.stat().st_size for path in self.model.iterdir()),
        )
        self.assertEqual(manifest["toolchain"], build.locked_toolchain())

    def test_deterministic_and_location_independent(self) -> None:
        text = build.manifest_text(self.manifest)
        moved = Path(self._tmp.name) / "elsewhere"
        self.model.rename(moved)
        self.assertEqual(build.manifest_text(build.calculate_manifest(moved)), text)
        self.assertNotIn(self._tmp.name, text)
        self.assertNotIn(str(Path.home()), text)

    def test_ignores_hidden_files(self) -> None:
        (self.model / ".DS_Store").write_bytes(b"junk")
        self.assertEqual(build.calculate_manifest(self.model), self.manifest)

    def test_refuses_incomplete_or_extra_content(self) -> None:
        (self.model / "notes.txt").write_text("x")
        with self.assertRaisesRegex(ValueError, "extra=\\['notes.txt'\\]"):
            build.calculate_manifest(self.model)
        (self.model / "notes.txt").unlink()
        (self.model / "tokenizer.json").unlink()
        with self.assertRaisesRegex(ValueError, "missing=\\['tokenizer.json'\\]"):
            build.calculate_manifest(self.model)

    def test_refuses_wrong_quantization(self) -> None:
        config = {"quantization": {"group_size": 64, "bits": 3, "mode": "affine"}}
        config["quantization_config"] = config["quantization"]
        (self.model / "config.json").write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "quantization"):
            build.calculate_manifest(self.model)


class VerifyManifestTests(TinyCheckpointTestCase):
    def test_valid_checkpoint_passes(self) -> None:
        result = self.verify()
        self.assertTrue(result["ok"], result["failures"])
        self.assertEqual(result["files"], 8)
        self.assertEqual(result["bytes"], self.manifest["expected"]["total_bytes"])

    def test_changed_content_fails(self) -> None:
        (self.model / SHARDS[1]).write_bytes(b"different content, different length")
        message = self.assertFailsOn(self.verify(), SHARDS[1], "size mismatch")
        self.assertIn("expected size: 16", message)
        self.assertIn("actual size:   35", message)

    def test_same_length_change_fails_by_sha256(self) -> None:
        path = self.model / SHARDS[0]
        data = bytearray(path.read_bytes())
        data[3] ^= 0x01
        path.write_bytes(bytes(data))
        message = self.assertFailsOn(self.verify(), SHARDS[0], "sha256 mismatch")
        expected = self.manifest["files"][SHARDS[0]]["sha256"]
        self.assertIn(f"expected sha256: {expected}", message)
        self.assertIn(f"actual sha256:   {build.sha256_file(path)}", message)

    def test_truncated_file_fails(self) -> None:
        path = self.model / SHARDS[0]
        path.write_bytes(path.read_bytes()[:-1])
        result = self.verify()
        self.assertFailsOn(result, SHARDS[0], "size mismatch")
        self.assertFailsOn(result, "aggregate size", "")

    def test_missing_file_fails(self) -> None:
        (self.model / "tokenizer_config.json").unlink()
        self.assertFailsOn(self.verify(), "tokenizer_config.json", "missing")

    def test_extra_visible_entries_fail_hidden_entries_are_ignored(self) -> None:
        (self.model / ".DS_Store").write_bytes(b"junk")
        self.assertTrue(self.verify()["ok"])
        (self.model / "model-00003-of-00003.safetensors").write_bytes(b"x")
        (self.model / "logs").mkdir()
        result = self.verify()
        self.assertFailsOn(result, "model-00003-of-00003.safetensors", "unexpected entry")
        self.assertFailsOn(result, "logs", "unexpected entry")

    def test_quantization_mismatch_is_named(self) -> None:
        config = json.loads((self.model / "config.json").read_text())
        config["quantization"]["bits"] = 4
        (self.model / "config.json").write_text(json.dumps(config, indent=4))
        result = self.verify()
        self.assertFailsOn(result, "config.json", "sha256 mismatch")
        self.assertFailsOn(result, "config.json", "quantization is")

    def test_index_structure_mismatch_is_named(self) -> None:
        index = json.loads((self.model / "model.safetensors.index.json").read_text())
        index["weight_map"]["c.weight"] = SHARDS[1]
        (self.model / "model.safetensors.index.json").write_text(json.dumps(index, indent=4))
        result = self.verify()
        self.assertFailsOn(result, "model.safetensors.index.json", "tensor_count is 5, expected 4")

    def test_missing_directory_fails(self) -> None:
        result = build.verify_manifest(self.model / "absent", self.manifest)
        self.assertFalse(result["ok"])
        self.assertIn("not a directory", result["failures"][0])

    def test_legacy_variant_is_strict_by_default(self) -> None:
        legacy = b'{"legacy": true}'
        self.manifest["legacy_variants"] = {
            "generation_config.json": [
                {"size": len(legacy), "sha256": build.hashlib.sha256(legacy).hexdigest(),
                 "note": "older form"}
            ]
        }
        (self.model / "generation_config.json").write_bytes(legacy)
        strict = self.verify()
        message = self.assertFailsOn(strict, "generation_config.json", "matches a known legacy")
        self.assertIn("older form", message)
        self.assertEqual(len(strict["failures"]), 1, strict["failures"])
        allowed = self.verify(allow_legacy=True)
        self.assertTrue(allowed["ok"], allowed["failures"])
        self.assertEqual(len(allowed["notes"]), 1)
        (self.model / "generation_config.json").write_bytes(b'{"legacy": false}')
        self.assertFailsOn(self.verify(allow_legacy=True), "generation_config.json", "size mismatch")


class ManifestSchemaTests(TinyCheckpointTestCase):
    def write(self, text: str) -> Path:
        path = Path(self._tmp.name) / "manifest.json"
        path.write_text(text, encoding="utf-8")
        return path

    def test_round_trip(self) -> None:
        path = self.write(build.manifest_text(self.manifest))
        self.assertEqual(build.load_manifest(path), self.manifest)

    def test_schema_version_mismatch(self) -> None:
        for version in (2, "1", None):
            manifest = copy.deepcopy(self.manifest)
            manifest["schema_version"] = version
            with self.assertRaisesRegex(build.ManifestError, "schema_version"):
                build.load_manifest(self.write(json.dumps(manifest)))

    def test_malformed_manifest(self) -> None:
        with self.assertRaisesRegex(build.ManifestError, "not valid JSON"):
            build.load_manifest(self.write('{"schema_version": 1,'))
        with self.assertRaisesRegex(build.ManifestError, "JSON object"):
            build.load_manifest(self.write("[]"))
        with self.assertRaisesRegex(build.ManifestError, "cannot read"):
            build.load_manifest(Path(self._tmp.name) / "absent.json")

    def test_invalid_contents(self) -> None:
        cases = {
            "sha256": lambda m: m["files"][SHARDS[0]].update(sha256="ABC"),
            "size": lambda m: m["files"][SHARDS[0]].update(size="15"),
            "plain visible file name": lambda m: m["files"].update(
                {"../escape": m["files"].pop(SHARDS[0])}
            ),
            "required files": lambda m: m["files"].pop("config.json"),
            "expected.total_bytes": lambda m: m["expected"].update(total_bytes=1),
            "source_revision": lambda m: m["model"].update(source_revision="main"),
            "model.quantization": lambda m: m["model"].update(quantization={"bits": 4}),
            "'files' object": lambda m: m.pop("files"),
            "note": lambda m: m.update(
                legacy_variants={"config.json": [dict(m["files"]["config.json"])]}
            ),
        }
        for message, mutate in cases.items():
            with self.subTest(message):
                manifest = copy.deepcopy(self.manifest)
                mutate(manifest)
                with self.assertRaisesRegex(build.ManifestError, re.escape(message)):
                    build.validate_manifest(manifest)
                with self.assertRaises(build.ManifestError):
                    build.verify_manifest(self.model, manifest)


class VerifyCliTests(TinyCheckpointTestCase):
    def run_cli(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(BUILDER), *args],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_exit_codes_and_output(self) -> None:
        manifest = Path(self._tmp.name) / "manifest.json"
        manifest.write_text(build.manifest_text(self.manifest))
        args = ("--verify-output", str(self.model), "--manifest", str(manifest))

        passed = self.run_cli(*args)
        self.assertEqual(passed.returncode, 0, passed.stdout + passed.stderr)
        self.assertIn("PASS\ncanonical g128 manifest verified\n8 files\n", passed.stdout)

        (self.model / SHARDS[1]).write_bytes(b"shard-two-bytes?")
        failed = self.run_cli(*args)
        self.assertEqual(failed.returncode, 1)
        self.assertIn(f"FAIL {SHARDS[1]}: sha256 mismatch", failed.stdout)

        manifest.write_text("{")
        broken = self.run_cli(*args)
        self.assertEqual(broken.returncode, 2)
        self.assertIn("FAIL manifest:", broken.stdout)

    def test_verify_needs_no_source(self) -> None:
        result = self.run_cli("--verify-output", str(self.model))
        self.assertEqual(result.returncode, 1)
        self.assertNotIn("--source is required", result.stderr)


class CommittedManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = build.load_manifest()

    def test_canonical_serialization(self) -> None:
        text = build.MANIFEST_PATH.read_text(encoding="utf-8")
        self.assertEqual(text, build.manifest_text(self.manifest))
        self.assertNotIn("/Users/", text)
        self.assertLess(len(text), 16_384)

    def test_identity(self) -> None:
        self.assertEqual(self.manifest["model"]["name"], build.MODEL_NAME)
        self.assertEqual(self.manifest["model"]["source_revision"], build.SOURCE_REVISION)
        self.assertEqual(self.manifest["model"]["quantization"], build.QUANTIZATION)
        self.assertEqual(self.manifest["build"]["shard_bytes"], build.DEFAULT_SHARD_BYTES)

    def test_toolchain_matches_lock(self) -> None:
        # A runtime upgrade must regenerate (and revalidate) the manifest.
        self.assertEqual(self.manifest["toolchain"], build.locked_toolchain())

    def test_coverage(self) -> None:
        files = self.manifest["files"]
        expected = self.manifest["expected"]
        shards = [f"model-{i:05d}-of-00019.safetensors" for i in range(1, 20)]
        self.assertEqual(sorted(files), sorted(build.REQUIRED_OUTPUT_FILES + tuple(shards)))
        self.assertEqual(expected["total_files"], 25)
        self.assertEqual(expected["weight_shards"], 19)
        self.assertEqual(expected["weight_payload_bytes"], 10_930_715_648)
        self.assertEqual(expected["tensor_count"], 1_847)
        self.assertEqual(expected["quantized_modules"], build.QUANTIZED_MODULES)
        self.assertEqual(expected["retained_source_tensors"], build.RETAINED_SOURCE_TENSORS)
        self.assertEqual(expected["total_parameters"], 26_895_993_856)

    def test_only_documented_legacy_variants(self) -> None:
        legacy = self.manifest["legacy_variants"]
        self.assertEqual(sorted(legacy), ["config.json", "tokenizer_config.json"])
        self.assertEqual(len(legacy["config.json"]), 1)
        for key in build.MTP_TEXT_CONFIG_KEYS:
            self.assertIn(key, legacy["config.json"][0]["note"])
        self.assertEqual(len(legacy["tokenizer_config.json"]), 1)
        self.assertIn("extra_special_tokens", legacy["tokenizer_config.json"][0]["note"])


@unittest.skipUnless(os.environ.get("QWEN38_G128_VERIFY"), "set QWEN38_G128_VERIFY to a checkpoint")
class RealCheckpointTests(unittest.TestCase):
    def test_real_checkpoint(self) -> None:
        model = Path(os.environ["QWEN38_G128_VERIFY"])
        result = build.verify_manifest(model, build.load_manifest(), allow_legacy=True)
        self.assertTrue(result["ok"], result["failures"])
        self.assertEqual(result["files"], 25)


if __name__ == "__main__":
    unittest.main()
