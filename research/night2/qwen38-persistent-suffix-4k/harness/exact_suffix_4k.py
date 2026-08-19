#!/usr/bin/env python3
"""One focused Qwen3.8 exact-incremental-suffix diagnostic.

The persistent child runs a 4K canonical user turn and then feeds the same
cache only the suffix of the complete canonical turn-2 chat template.  A
second, serialized child generates turn 2 from that complete template with an
empty cache.  Both children use direct pinned ``generate_step`` so the MLX
wired state is left untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "/Users/artem/models/Qwen3.8-27B-3bit-g128-MLX"
INITIAL_PROMPT_PATH = REPO_ROOT / "research/context-ladder/prompts/prompt-4096.txt"
FOLLOW_UP = (
    "Which single invariant from your previous diagnosis should the API enforce "
    "first, and why? Answer in 3-5 sentences."
)
PINNED_MLX_LM_REVISION = "254d153fdeb6f150edd4fc5a54f9828638481fa8"
PREFILL_STEP_SIZE = 2048
INITIAL_MAX_TOKENS = 96
FOLLOW_UP_MAX_TOKENS = 64


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_token_ids(value: Any) -> list[int]:
    """Normalize Transformers 5.x BatchEncoding and older list returns."""

    if hasattr(value, "encodings") and value.encodings:
        return [int(item) for item in value.encodings[0].ids]
    if isinstance(value, dict):
        value = value["input_ids"]
    elif hasattr(value, "input_ids"):
        value = value.input_ids
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], (list, tuple)):
        value = value[0]
    return [int(item) for item in value]


def token_labels(tokenizer: Any, ids: list[int]) -> list[str | None]:
    labels: list[str | None] = []
    for token_id in ids:
        try:
            labels.append(str(tokenizer.convert_ids_to_tokens(int(token_id))))
        except BaseException:
            labels.append(None)
    return labels


def canonical_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> dict[str, Any]:
    """Render and independently verify one canonical Qwen chat-template turn."""

    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    encoded_from_string = normalize_token_ids(
        tokenizer.encode(rendered, add_special_tokens=False)
    )
    direct_encoded = normalize_token_ids(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )
    if encoded_from_string != direct_encoded:
        raise RuntimeError("chat-template string/tokenized forms disagree")
    return {
        "rendered_text": rendered,
        "token_ids": encoded_from_string,
        "token_count": len(encoded_from_string),
        "token_sha256": sha256_json(encoded_from_string),
    }


def lcp(left: list[int], right: list[int]) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def flatten_arrays(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "shape") and hasattr(value, "dtype") and hasattr(value, "nbytes"):
        return [value]
    if isinstance(value, (list, tuple)):
        result: list[Any] = []
        for item in value:
            result.extend(flatten_arrays(item))
        return result
    return []


def cache_arrays(entry: Any) -> list[Any]:
    try:
        return flatten_arrays(entry.state)
    except BaseException:
        return []


def array_metadata(mx: Any, array: Any, with_signature: bool) -> dict[str, Any]:
    item: dict[str, Any] = {
        "shape": [int(value) for value in array.shape],
        "dtype": str(array.dtype),
        "nbytes": int(array.nbytes),
    }
    if not with_signature:
        return item
    try:
        flat = array.reshape((-1,))
        size = int(flat.size)
        if size <= 4:
            sample = flat
        else:
            sample = mx.concatenate([flat[:2], flat[-2:]])
        mx.eval(sample)
        item["sample"] = sample.tolist()
    except BaseException as exc:
        item["sample_error"] = f"{type(exc).__name__}: {exc}"
    return item


def cache_diagnostics(mx: Any, cache: list[Any], label: str, with_signature: bool = True) -> dict[str, Any]:
    """Capture compact per-layer cache metadata without copying full states."""

    if with_signature:
        try:
            mx.synchronize()
        except BaseException:
            pass
    layers: list[dict[str, Any]] = []
    offsets: list[int] = []
    for index, entry in enumerate(cache):
        class_name = type(entry).__name__
        item: dict[str, Any] = {
            "layer_index": index,
            "cache_class": class_name,
            "offset": None,
            "size": None,
            "trimmable": None,
            "nbytes": None,
            "states": [],
            "state_signature_sha256": None,
        }
        try:
            value = getattr(entry, "offset")
            value = value() if callable(value) else value
            item["offset"] = int(value)
            offsets.append(int(value))
        except BaseException:
            pass
        try:
            value = entry.size()
            item["size"] = int(value)
        except BaseException:
            pass
        try:
            item["trimmable"] = bool(entry.is_trimmable())
        except BaseException:
            pass
        try:
            item["nbytes"] = int(entry.nbytes)
        except BaseException:
            pass
        states = [array_metadata(mx, array, with_signature) for array in cache_arrays(entry)]
        item["states"] = [
            {key: value for key, value in state.items() if key != "sample"}
            for state in states
        ]
        if with_signature:
            item["state_signature_sha256"] = sha256_json(
                {
                    "cache_class": class_name,
                    "offset": item["offset"],
                    "size": item["size"],
                    "states": states,
                }
            )
        layers.append(item)
    unique_offsets = sorted(set(offsets))
    return {
        "label": label,
        "layer_count": len(layers),
        "class_counts": {
            name: sum(1 for item in layers if item["cache_class"] == name)
            for name in sorted({item["cache_class"] for item in layers})
        },
        "offsets": unique_offsets,
        "offsets_consistent": len(unique_offsets) <= 1,
        "logical_prefix_token_count": unique_offsets[0] if len(unique_offsets) == 1 else None,
        "state_signature_sha256": sha256_json(
            [item["state_signature_sha256"] for item in layers]
        )
        if with_signature
        else None,
        "layers": layers,
    }


def offset_snapshot(cache: list[Any], label: str) -> dict[str, Any]:
    """Cheap prompt-boundary snapshot used inside generate_step's callback."""

    layers: list[dict[str, Any]] = []
    offsets: list[int] = []
    for index, entry in enumerate(cache):
        value: int | None = None
        try:
            value = int(getattr(entry, "offset"))
            offsets.append(value)
        except BaseException:
            pass
        layers.append(
            {
                "layer_index": index,
                "cache_class": type(entry).__name__,
                "offset": value,
            }
        )
    unique_offsets = sorted(set(offsets))
    return {
        "label": label,
        "offsets": unique_offsets,
        "offsets_consistent": len(unique_offsets) <= 1,
        "logical_prefix_token_count": unique_offsets[0] if len(unique_offsets) == 1 else None,
        "layers": layers,
    }


def gib(value: int | None) -> float | None:
    return value / (1024**3) if value is not None else None


def memory_snapshot(mx: Any, label: str, started: float) -> dict[str, Any]:
    try:
        mx.synchronize()
    except BaseException:
        pass
    values: dict[str, int | None] = {}
    for name, getter in (
        ("active", mx.get_active_memory),
        ("cache", mx.get_cache_memory),
        ("peak", mx.get_peak_memory),
    ):
        try:
            values[name] = int(getter())
        except BaseException:
            values[name] = None
    total = (
        values["active"] + values["cache"]
        if values["active"] is not None and values["cache"] is not None
        else None
    )
    return {
        "label": label,
        "elapsed_seconds": time.perf_counter() - started,
        "active_gib": gib(values["active"]),
        "cache_gib": gib(values["cache"]),
        "active_plus_cache_gib": gib(total),
        "peak_gib": gib(values["peak"]),
        "active_bytes": values["active"],
        "cache_bytes": values["cache"],
        "peak_bytes": values["peak"],
        "rss_max_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }


def command_snapshot(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=15)
        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout[-12000:],
            "stderr": completed.stderr[-4000:],
        }
    except BaseException as exc:
        return {"exception": f"{type(exc).__name__}: {exc}"}


def os_snapshot() -> dict[str, Any]:
    return {
        "memory_pressure": command_snapshot(["memory_pressure", "-Q"]),
        "vm_stat": command_snapshot(["vm_stat"]),
        "vm_swapusage": command_snapshot(["sysctl", "vm.swapusage"]),
    }


def model_metadata(model_path: str) -> dict[str, Any]:
    config = json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8"))
    return {
        "model_path": str(Path(model_path).resolve()),
        "quantization": config.get("quantization", config.get("quantization_config")),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "mlx_version": metadata.version("mlx"),
        "mlx_lm_version": metadata.version("mlx-lm"),
        "mlx_lm_revision": PINNED_MLX_LM_REVISION,
        "wired_policy": "direct generate_step; MLX wired state untouched",
    }


def scalar_token(value: Any) -> int:
    try:
        return int(value.item())
    except BaseException:
        return int(value)


def generate_stage(
    mx: Any,
    generate_step: Any,
    make_sampler: Any,
    tokenizer: Any,
    model: Any,
    prompt_ids: list[int],
    cache: list[Any],
    max_tokens: int,
    label: str,
    started: float,
) -> dict[str, Any]:
    prompt_boundary: list[dict[str, Any]] = []

    def callback(processed: int, total: int) -> None:
        if processed == total:
            # In pinned generate_step this callback is invoked after the first
            // prompt-derived token has been sampled and its next model step
            // has already been scheduled. The offset therefore includes one
            // yielded token; it is not a pure prompt-only snapshot.
            prompt_boundary.append(
                offset_snapshot(cache, f"{label}:after_prompt_and_first_generation_step")
            )

    before = cache_diagnostics(mx, cache, f"{label}:before_generation")
    generation_started = time.perf_counter()
    generator = generate_step(
        mx.array(prompt_ids),
        model,
        max_tokens=max_tokens,
        sampler=make_sampler(0.0, 1.0),
        prompt_cache=cache,
        prefill_step_size=PREFILL_STEP_SIZE,
        prompt_progress_callback=callback,
    )
    generated_ids: list[int] = []
    first_token_time: float | None = None
    after_first: dict[str, Any] | None = None
    finish_reason = "length"
    eos_ids = {int(value) for value in tokenizer.eos_token_ids}
    try:
        while len(generated_ids) < max_tokens:
            try:
                token, _logprobs = next(generator)
            except StopIteration:
                finish_reason = "generator_stop"
                break
            if first_token_time is None:
                first_token_time = time.perf_counter()
                after_first = offset_snapshot(cache, f"{label}:after_first_yield")
            token_id = scalar_token(token)
            generated_ids.append(token_id)
            if token_id in eos_ids:
                finish_reason = "eos"
                break
    finally:
        generator.close()
    completed = time.perf_counter()
    try:
        mx.synchronize()
    except BaseException:
        pass
    if after_first is None:
        after_first = offset_snapshot(cache, f"{label}:after_first_yield_unavailable")
    after = cache_diagnostics(mx, cache, f"{label}:after_generation")
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    ttft = first_token_time - generation_started if first_token_time is not None else None
    generation_seconds = (
        completed - first_token_time if first_token_time is not None else None
    )
    decode_tokens = max(len(generated_ids) - 1, 0)
    return {
        "label": label,
        "prompt_token_ids": prompt_ids,
        "prompt_tokens": len(prompt_ids),
        "generated_token_ids": generated_ids,
        "generated_tokens": len(generated_ids),
        "generated_token_hash": sha256_json(generated_ids),
        "decoded_output": output_text,
        "finish_reason": finish_reason,
        "eos_token_ids": sorted(eos_ids),
        "ttft_seconds": ttft,
        "prompt_tps": len(prompt_ids) / ttft if ttft else None,
        "generation_seconds": generation_seconds,
        "decode_tps": decode_tokens / generation_seconds if generation_seconds else None,
        "total_seconds": completed - generation_started,
        "cache_diagnostics": [before, *prompt_boundary, after_first, after],
        "memory_after_generation": memory_snapshot(mx, f"{label}:after_generation", started),
    }


def save_token_artifact(
    path: Path, ids: list[int], tokenizer: Any | None = None, name: str | None = None
) -> None:
    value: dict[str, Any] = {
        "name": name or path.stem,
        "token_count": len(ids),
        "sha256": sha256_json(ids),
        "token_ids": ids,
        "first_32_token_ids": ids[:32],
        "last_32_token_ids": ids[-32:],
    }
    if tokenizer is not None:
        value["first_32_tokens"] = token_labels(tokenizer, ids[:32])
        value["last_32_tokens"] = token_labels(tokenizer, ids[-32:])
    write_json(path, value)


def current_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
        return completed.stdout.strip()
    except BaseException:
        return None


def load_runtime() -> tuple[Any, Any, Any, Any, Any, Any]:
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler

    return mx, load, generate_step, make_prompt_cache, make_sampler, metadata


def run_persistent(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    result: dict[str, Any] = {
        "experiment_id": "PERSISTENT-SUFFIX-4K",
        "mode": "persistent_exact_canonical_suffix",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": current_commit(),
        "exception": None,
        "safety_stop_reason": None,
    }
    try:
        mx, load, generate_step, make_prompt_cache, make_sampler, _ = load_runtime()
        result["model"] = model_metadata(args.model_path)
        result["os_memory_before"] = os_snapshot()
        model, tokenizer = load(args.model_path)
        result["memory_after_model_load"] = memory_snapshot(mx, "after_model_load", started)
        initial_text = INITIAL_PROMPT_PATH.read_text(encoding="utf-8")
        initial = canonical_prompt(tokenizer, [{"role": "user", "content": initial_text}])
        write_json(
            EXPERIMENT_ROOT / "tokens/initial-prompt.json",
            {
                "source_path": str(INITIAL_PROMPT_PATH),
                "source_sha256": sha256_bytes(INITIAL_PROMPT_PATH.read_bytes()),
                **initial,
            },
        )
        cache = make_prompt_cache(model)
        initial_generation = generate_stage(
            mx,
            generate_step,
            make_sampler,
            tokenizer,
            model,
            initial["token_ids"],
            cache,
            INITIAL_MAX_TOKENS,
            "turn1_initial",
            started,
        )
        initial_ids = initial_generation["generated_token_ids"]
        initial_output = initial_generation["decoded_output"]
        write_json(EXPERIMENT_ROOT / "outputs/persistent-turn1.json", initial_generation)
        (EXPERIMENT_ROOT / "outputs/persistent-turn1.txt").write_text(initial_output, encoding="utf-8")
        save_token_artifact(
            EXPERIMENT_ROOT / "tokens/persistent-turn1-generated.json", initial_ids, tokenizer, "persistent_turn1_generated"
        )

        expected_cache_prefix = initial["token_ids"] + initial_ids
        after_initial = initial_generation["cache_diagnostics"][-1]
        measured_prefix = after_initial.get("logical_prefix_token_count")
        if measured_prefix is None:
            measured_prefix = len(expected_cache_prefix)
            measured_prefix_source = "expected_prompt_plus_yielded_tokens_no_uniform_KV_offset"
        else:
            measured_prefix_source = "uniform_attention_KV_offsets_after_generation"

        assistant_candidates: list[dict[str, Any]] = []
        for skip_special_tokens in (True, False):
            assistant_text = tokenizer.decode(initial_ids, skip_special_tokens=skip_special_tokens)
            turn2 = canonical_prompt(
                tokenizer,
                [
                    {"role": "user", "content": initial_text},
                    {"role": "assistant", "content": assistant_text},
                    {"role": "user", "content": FOLLOW_UP},
                ],
            )
            prefix_match = lcp(turn2["token_ids"], expected_cache_prefix)
            assistant_candidates.append(
                {
                    "skip_special_tokens": skip_special_tokens,
                    "assistant_text": assistant_text,
                    "canonical": turn2,
                    "prefix_lcp_with_expected_cache_stream": prefix_match,
                    "expected_cache_prefix_exact": turn2["token_ids"][: len(expected_cache_prefix)] == expected_cache_prefix,
                }
            )
        selected = next(
            (item for item in assistant_candidates if item["expected_cache_prefix_exact"]),
            max(assistant_candidates, key=lambda item: item["prefix_lcp_with_expected_cache_stream"]),
        )
        turn2 = selected["canonical"]
        full_turn2_ids = turn2["token_ids"]
        if int(measured_prefix) > len(full_turn2_ids):
            raise RuntimeError(
                f"measured cache prefix {measured_prefix} exceeds canonical turn2 length {len(full_turn2_ids)}"
            )
        suffix = full_turn2_ids[int(measured_prefix) :]
        canonical_prefix_at_measured = full_turn2_ids[: int(measured_prefix)]
        cache_prefix_matches_expected = canonical_prefix_at_measured == expected_cache_prefix[: int(measured_prefix)]
        save_token_artifact(
            EXPERIMENT_ROOT / "tokens/expected-cache-prefix.json",
            expected_cache_prefix,
            tokenizer,
            "expected_cache_prefix_prompt_plus_yielded_turn1_tokens",
        )
        save_token_artifact(
            EXPERIMENT_ROOT / "tokens/canonical-turn2.json", full_turn2_ids, tokenizer, "canonical_full_turn2"
        )
        save_token_artifact(
            EXPERIMENT_ROOT / "tokens/turn2-exact-suffix.json", suffix, tokenizer, "exact_incremental_turn2_suffix"
        )
        result["initial_prompt"] = initial
        result["follow_up_text"] = FOLLOW_UP
        result["turn1_generation"] = initial_generation
        result["cache_prefix_accounting"] = {
            "generate_step_semantics": {
                "prefill_evaluates_prompt_except_last_then_evaluates_last_prompt_token": True,
                "each_yielded_token_is_fed_through_the_precomputed_next_step": True,
                "one_internal_lookahead_token_is_sampled_but_not_yielded_at_max_length": True,
                "eos_yielded_token_is_counted_by_measured_cache_offset": True,
            },
            "expected_cache_prefix_token_ids": expected_cache_prefix,
            "expected_cache_prefix_token_count": len(expected_cache_prefix),
            "measured_cached_prefix_token_count": int(measured_prefix),
            "measured_prefix_source": measured_prefix_source,
            "attention_offsets_after_turn1": after_initial.get("offsets"),
            "attention_offsets_consistent": after_initial.get("offsets_consistent"),
            "generated_finish_reason": initial_generation["finish_reason"],
            "generated_eos_seen": initial_generation["finish_reason"] == "eos",
            "canonical_candidate_diagnostics": [
                {
                    key: value
                    for key, value in item.items()
                    if key != "canonical" and key != "assistant_text"
                }
                for item in assistant_candidates
            ],
            "selected_assistant_decode_skip_special_tokens": selected["skip_special_tokens"],
            "selected_assistant_text": selected["assistant_text"],
            "selected_assistant_text_sha256": sha256_bytes(selected["assistant_text"].encode("utf-8")),
            "canonical_prefix_matches_expected_cache_stream": cache_prefix_matches_expected,
            "canonical_prefix_lcp_with_expected_cache_stream": lcp(full_turn2_ids, expected_cache_prefix),
            "canonical_full_turn2_token_count": len(full_turn2_ids),
            "incremental_suffix_token_ids": suffix,
            "incremental_suffix_token_count": len(suffix),
            "incremental_suffix_sha256": sha256_json(suffix),
            "incremental_suffix_first_32_token_ids": suffix[:32],
            "incremental_suffix_last_32_token_ids": suffix[-32:],
            "incremental_suffix_first_32_tokens": token_labels(tokenizer, suffix[:32]),
            "incremental_suffix_last_32_tokens": token_labels(tokenizer, suffix[-32:]),
        }
        persistent_turn2 = generate_stage(
            mx,
            generate_step,
            make_sampler,
            tokenizer,
            model,
            suffix,
            cache,
            FOLLOW_UP_MAX_TOKENS,
            "turn2_persistent_exact_suffix",
            started,
        )
        result["persistent_turn2"] = persistent_turn2
        (EXPERIMENT_ROOT / "outputs/persistent-turn2.txt").write_text(
            persistent_turn2["decoded_output"], encoding="utf-8"
        )
        write_json(EXPERIMENT_ROOT / "outputs/persistent-turn2.json", persistent_turn2)
        save_token_artifact(
            EXPERIMENT_ROOT / "tokens/persistent-turn2-generated.json",
            persistent_turn2["generated_token_ids"],
            tokenizer,
            "persistent_turn2_generated",
        )
        result["memory_after_persistent_turn2"] = memory_snapshot(
            mx, "after_persistent_turn2", started
        )
        result["os_memory_after"] = os_snapshot()
    except BaseException as exc:
        result["exception"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    result["total_elapsed_seconds"] = time.perf_counter() - started
    write_json(EXPERIMENT_ROOT / "persistent.json", result)
    print(json.dumps({"mode": "persistent", "exception": result["exception"]}), flush=True)
    return 1 if result["exception"] else 0


def run_fresh(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    persistent = read_json(EXPERIMENT_ROOT / "persistent.json")
    result: dict[str, Any] = {
        "experiment_id": "FRESH-FULL-TRANSCRIPT-TURN2-4K",
        "mode": "fresh_full_canonical_transcript",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": current_commit(),
        "exception": None,
        "safety_stop_reason": None,
    }
    try:
        mx, load, generate_step, make_prompt_cache, make_sampler, _ = load_runtime()
        result["model"] = model_metadata(args.model_path)
        result["os_memory_before"] = os_snapshot()
        model, tokenizer = load(args.model_path)
        result["memory_after_model_load"] = memory_snapshot(mx, "after_model_load", started)
        canonical_ids = persistent["cache_prefix_accounting"]["incremental_suffix_token_ids"]
        full_turn2_ids = persistent["cache_prefix_accounting"]["canonical_full_turn2_token_count"]
        full_turn2_ids = read_json(EXPERIMENT_ROOT / "tokens/canonical-turn2.json")["token_ids"]
        reconstructed = canonical_prompt(
            tokenizer,
            [
                {"role": "user", "content": INITIAL_PROMPT_PATH.read_text(encoding="utf-8")},
                {
                    "role": "assistant",
                    "content": persistent["cache_prefix_accounting"].get("selected_assistant_text", ""),
                },
                {"role": "user", "content": FOLLOW_UP},
            ],
        ) if persistent["cache_prefix_accounting"].get("selected_assistant_text") else None
        result["canonical_turn2_token_count"] = len(full_turn2_ids)
        result["canonical_turn2_token_sha256"] = sha256_json(full_turn2_ids)
        result["canonical_turn2_reconstructed_in_fresh_child"] = (
            reconstructed is not None and reconstructed["token_ids"] == full_turn2_ids
        )
        cache = make_prompt_cache(model)
        fresh_turn2 = generate_stage(
            mx,
            generate_step,
            make_sampler,
            tokenizer,
            model,
            full_turn2_ids,
            cache,
            FOLLOW_UP_MAX_TOKENS,
            "turn2_fresh_full_transcript",
            started,
        )
        result["fresh_turn2"] = fresh_turn2
        (EXPERIMENT_ROOT / "outputs/fresh-turn2.txt").write_text(
            fresh_turn2["decoded_output"], encoding="utf-8"
        )
        write_json(EXPERIMENT_ROOT / "outputs/fresh-turn2.json", fresh_turn2)
        save_token_artifact(
            EXPERIMENT_ROOT / "tokens/fresh-turn2-generated.json",
            fresh_turn2["generated_token_ids"],
            tokenizer,
            "fresh_turn2_generated",
        )
        result["memory_after_fresh_turn2"] = memory_snapshot(mx, "after_fresh_turn2", started)
        result["os_memory_after"] = os_snapshot()
    except BaseException as exc:
        result["exception"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    result["total_elapsed_seconds"] = time.perf_counter() - started
    write_json(EXPERIMENT_ROOT / "fresh.json", result)
    print(json.dumps({"mode": "fresh", "exception": result["exception"]}), flush=True)
    return 1 if result["exception"] else 0


def supporting_semantics(text: str) -> dict[str, Any]:
    lower = text.lower()
    terms = ["invariant", "enforce", "first"]
    return {
        "terms": terms,
        "term_presence": {term: term in lower for term in terms},
        "all_terms_present": all(term in lower for term in terms),
        "note": "supporting evidence only; raw output and replay comparison require human review",
    }


def combine_results() -> int:
    persistent = read_json(EXPERIMENT_ROOT / "persistent.json")
    result: dict[str, Any] = {
        "experiment_id": "QWEN38-PERSISTENT-SUFFIX-4K",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": current_commit(),
        "model_path": DEFAULT_MODEL,
        "quantization": {"mode": "affine", "bits": 3, "group_size": 128},
        "protocol": "exact canonical full-chat-template suffix; no later BOS or standalone follow-up template",
        "persistent": persistent,
    }
    fresh_path = EXPERIMENT_ROOT / "fresh.json"
    if fresh_path.exists():
        fresh = read_json(fresh_path)
        result["fresh"] = fresh
    else:
        fresh = None
        result["fresh"] = None
    if persistent.get("exception") or not fresh or fresh.get("exception"):
        result["comparison"] = {"status": "ERROR"}
    else:
        p = persistent["persistent_turn2"]["generated_token_ids"]
        f = fresh["fresh_turn2"]["generated_token_ids"]
        initial = persistent["turn1_generation"]["generated_token_ids"]
        prefix = lcp(p, initial)
        result["comparison"] = {
            "status": "OK",
            "persistent_turn2_prefix_lcp_with_turn1": prefix,
            "persistent_turn2_prefix_overlap_fraction": prefix / len(p) if p else 0.0,
            "stale_replay_detected": bool(p and prefix / len(p) >= 0.90),
            "persistent_turn2_supporting_semantics": supporting_semantics(
                persistent["persistent_turn2"]["decoded_output"]
            ),
            "fresh_turn2_supporting_semantics": supporting_semantics(
                fresh["fresh_turn2"]["decoded_output"]
            ),
            "persistent_vs_fresh_token_equal": p == f,
            "persistent_vs_fresh_token_lcp": lcp(p, f),
            "persistent_vs_fresh_text_equal": persistent["persistent_turn2"]["decoded_output"]
            == fresh["fresh_turn2"]["decoded_output"],
            "persistent_turn2_ttft_seconds": persistent["persistent_turn2"]["ttft_seconds"],
            "fresh_turn2_ttft_seconds": fresh["fresh_turn2"]["ttft_seconds"],
            "persistent_turn2_decode_tps": persistent["persistent_turn2"]["decode_tps"],
            "fresh_turn2_decode_tps": fresh["fresh_turn2"]["decode_tps"],
            "persistent_turn2_incremental_suffix_tokens": persistent[
                "cache_prefix_accounting"
            ]["incremental_suffix_token_count"],
            "fresh_turn2_full_prompt_tokens": len(
                fresh["fresh_turn2"]["prompt_token_ids"]
            ),
        }
    result["raw_artifacts"] = {
        "persistent_json": str(EXPERIMENT_ROOT / "persistent.json"),
        "fresh_json": str(EXPERIMENT_ROOT / "fresh.json"),
        "persistent_output": str(EXPERIMENT_ROOT / "outputs/persistent-turn2.txt"),
        "fresh_output": str(EXPERIMENT_ROOT / "outputs/fresh-turn2.txt"),
    }
    write_json(EXPERIMENT_ROOT / "results.json", result)
    print(json.dumps({"mode": "combined", "status": result["comparison"]["status"]}), flush=True)
    return 1 if result["comparison"]["status"] != "OK" else 0


def run_all(args: argparse.Namespace) -> int:
    logs = EXPERIMENT_ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    child_base = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model-path",
        args.model_path,
    ]
    for mode in ("persistent", "fresh"):
        command = [*child_base, "--mode", mode]
        completed = subprocess.run(command, capture_output=True, text=True)
        (logs / f"{mode}.stdout.log").write_text(completed.stdout, encoding="utf-8")
        (logs / f"{mode}.stderr.log").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            return completed.returncode
    return combine_results()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("run", "persistent", "fresh", "combine"), required=True)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    args = parser.parse_args()
    if args.mode == "run":
        raise SystemExit(run_all(args))
    if args.mode == "persistent":
        raise SystemExit(run_persistent(args))
    if args.mode == "fresh":
        raise SystemExit(run_fresh(args))
    raise SystemExit(combine_results())


if __name__ == "__main__":
    main()
