#!/usr/bin/env python3
"""Same-boundary Qwen3.8 cache oracle.

The oracle uses the preserved Night-2 4K canonical token artifacts and runs
three paths over the same intended conversation:

* A: an empty cache receives the complete canonical turn-2 transcript once;
* B: a new cache receives the initial prompt, turn-1 generation, and exact
  canonical suffix as separate model calls;
* C: the real exact-suffix persistent conversation reuses one cache across
  those same two calls.

B and C are executed sequentially in one process so compact cache signatures
and first-token logprobs can be compared directly without serializing tensors.
The model weights are loaded once, but every path owns a distinct cache. This
is a correctness diagnostic, not a cold performance comparison.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata as metadata
import json
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
NIGHT2_ROOT = REPO_ROOT / "research/night2/qwen38-persistent-suffix-4k"
DEFAULT_MODEL = "/Users/artem/models/Qwen3.8-27B-3bit-g128-MLX"
PINNED_MLX_LM_REVISION = "254d153fdeb6f150edd4fc5a54f9828638481fa8"
PREFILL_STEP_SIZE = 2048
INITIAL_MAX_TOKENS = 96
FOLLOW_UP_MAX_TOKENS = 64


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
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
        flattened: list[Any] = []
        for item in value:
            flattened.extend(flatten_arrays(item))
        return flattened
    return []


def cache_arrays(entry: Any) -> list[Any]:
    for attribute in ("state", "cache"):
        try:
            value = getattr(entry, attribute)
            arrays = flatten_arrays(value)
            if arrays:
                return arrays
        except BaseException:
            pass
    try:
        return flatten_arrays([entry.keys, entry.values])
    except BaseException:
        return []


def array_metadata(mx: Any, array: Any) -> dict[str, Any]:
    item: dict[str, Any] = {
        "shape": [int(value) for value in array.shape],
        "dtype": str(array.dtype),
        "nbytes": int(array.nbytes),
    }
    try:
        flat = array.reshape((-1,))
        size = int(flat.size)
        sample = flat if size <= 4 else mx.concatenate([flat[:2], flat[-2:]])
        mx.eval(sample)
        item["sample"] = sample.tolist()
    except BaseException as exc:
        item["sample_error"] = f"{type(exc).__name__}: {exc}"
    return item


def cache_diagnostics(mx: Any, cache: list[Any], label: str) -> dict[str, Any]:
    """Return compact per-layer metadata and sampled deterministic signatures."""

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
            item["size"] = int(entry.size())
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
        states = [array_metadata(mx, array) for array in cache_arrays(entry)]
        item["states"] = states
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
        ),
        "layers": layers,
    }


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
        "active_bytes": values["active"],
        "cache_bytes": values["cache"],
        "active_plus_cache_bytes": total,
        "peak_bytes": values["peak"],
        "active_gib": values["active"] / 2**30 if values["active"] is not None else None,
        "cache_gib": values["cache"] / 2**30 if values["cache"] is not None else None,
        "active_plus_cache_gib": total / 2**30 if total is not None else None,
        "peak_gib": values["peak"] / 2**30 if values["peak"] is not None else None,
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


def scalar_token(value: Any) -> int:
    try:
        return int(value.item())
    except BaseException:
        return int(value)


def logprob_summary(logprobs: Any, mx: Any) -> tuple[dict[str, Any], Any]:
    """Keep only top-k values in JSON while retaining one host vector in memory."""

    import numpy as np

    # Qwen3.8 returns bfloat16 logprobs on this pinned path. NumPy does not
    # accept MLX's bfloat16 PEP 3118 buffer, so explicitly materialize a
    # float32 MLX array before crossing the device/host boundary.
    try:
        logprobs = logprobs.astype(mx.float32)
        mx.eval(logprobs)
    except BaseException:
        pass
    values = np.array(logprobs, copy=True).reshape(-1)
    k = min(8, values.size)
    indices = np.argsort(values)[-k:][::-1]
    summary = {
        "vocab_size": int(values.size),
        "top_k_ids": [int(index) for index in indices],
        "top_k_logprobs": [float(values[index]) for index in indices],
        "top1_id": int(indices[0]) if k else None,
        "logprob_sha256": sha256_bytes(values.astype(np.float32, copy=False).tobytes()),
    }
    return summary, values


def safe_clear(mx: Any) -> None:
    try:
        mx.synchronize()
    except BaseException:
        pass
    try:
        mx.clear_cache()
    except BaseException:
        pass
    gc.collect()


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
    *,
    kv_bits: int | None = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
) -> dict[str, Any]:
    prompt_callbacks: list[dict[str, Any]] = []

    def callback(processed: int, total: int) -> None:
        if processed == total:
            # In pinned generate_step this callback runs after the prompt's
            # final model step and the first generation step have been
            # scheduled. It is therefore explicitly not called a pure
            # prompt-only snapshot.
            prompt_callbacks.append(
                cache_diagnostics(
                    mx,
                    cache,
                    f"{label}:after_prompt_and_first_generation_step",
                )
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
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
        quantized_kv_start=quantized_kv_start,
    )
    generated_ids: list[int] = []
    first_token_time: float | None = None
    first_logprob_summary: dict[str, Any] | None = None
    first_logprob_values: Any | None = None
    finish_reason = "length"
    eos_ids = {int(value) for value in (tokenizer.eos_token_ids or [])}
    try:
        while len(generated_ids) < max_tokens:
            try:
                token, logprobs = next(generator)
            except StopIteration:
                finish_reason = "generator_stop"
                break
            if first_token_time is None:
                first_token_time = time.perf_counter()
                first_logprob_summary, first_logprob_values = logprob_summary(logprobs, mx)
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
    after_first = cache_diagnostics(mx, cache, f"{label}:after_first_yield")
    after = cache_diagnostics(mx, cache, f"{label}:after_generation")
    ttft = first_token_time - generation_started if first_token_time is not None else None
    generation_seconds = (
        completed - first_token_time if first_token_time is not None else None
    )
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return {
        "label": label,
        "prompt_token_ids": prompt_ids,
        "prompt_tokens": len(prompt_ids),
        "generated_token_ids": generated_ids,
        "generated_tokens": len(generated_ids),
        "generated_token_hash": sha256_json(generated_ids),
        "decoded_output": decoded,
        "finish_reason": finish_reason,
        "eos_token_ids": sorted(eos_ids),
        "ttft_seconds": ttft,
        "prompt_tps": len(prompt_ids) / ttft if ttft else None,
        "generation_seconds": generation_seconds,
        "decode_tps": max(len(generated_ids) - 1, 0) / generation_seconds
        if generation_seconds
        else None,
        "total_seconds": completed - generation_started,
        "first_logprob_summary": first_logprob_summary,
        "first_logprob_values": first_logprob_values,
        "cache_diagnostics": [before, *prompt_callbacks, after_first, after],
        "memory_after_generation": memory_snapshot(mx, f"{label}:after_generation", started),
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


def current_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return result.stdout.strip()
    except BaseException:
        return None


def load_runtime() -> tuple[Any, Any, Any, Any, Any, Any]:
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler

    return mx, load, generate_step, make_prompt_cache, make_sampler, metadata


def load_reference_conversation(tokenizer: Any) -> dict[str, Any]:
    initial_artifact = read_json(NIGHT2_ROOT / "tokens/initial-prompt.json")
    turn1_artifact = read_json(NIGHT2_ROOT / "tokens/persistent-turn1-generated.json")
    prefix_artifact = read_json(NIGHT2_ROOT / "tokens/expected-cache-prefix.json")
    canonical_artifact = read_json(NIGHT2_ROOT / "tokens/canonical-turn2.json")
    suffix_artifact = read_json(NIGHT2_ROOT / "tokens/turn2-exact-suffix.json")
    persistent = read_json(NIGHT2_ROOT / "persistent.json")
    initial_text = (REPO_ROOT / "research/context-ladder/prompts/prompt-4096.txt").read_text(
        encoding="utf-8"
    )
    follow_up = persistent["follow_up_text"]
    initial_ids = [int(x) for x in initial_artifact["token_ids"]]
    turn1_ids = [int(x) for x in turn1_artifact["token_ids"]]
    expected_prefix_ids = [int(x) for x in prefix_artifact["token_ids"]]
    canonical_ids = [int(x) for x in canonical_artifact["token_ids"]]
    suffix_ids = [int(x) for x in suffix_artifact["token_ids"]]
    assistant_text = persistent["cache_prefix_accounting"]["selected_assistant_text"]
    canonical = canonical_prompt(
        tokenizer,
        [
            {"role": "user", "content": initial_text},
            {"role": "assistant", "content": assistant_text},
            {"role": "user", "content": follow_up},
        ],
    )
    if canonical["token_ids"] != canonical_ids:
        raise RuntimeError("preserved canonical turn-2 artifact failed tokenizer reconstruction")
    if expected_prefix_ids != initial_ids + turn1_ids:
        raise RuntimeError("preserved expected cache prefix is not prompt plus turn-1 tokens")
    if canonical_ids[: len(expected_prefix_ids)] != expected_prefix_ids:
        raise RuntimeError("canonical turn-2 artifact does not begin with expected cache prefix")
    measured_prefix = int(
        persistent["turn1_generation"]["cache_diagnostics"][-1]["logical_prefix_token_count"]
    )
    if canonical_ids[measured_prefix:] != suffix_ids:
        raise RuntimeError("preserved exact suffix does not match measured canonical prefix")
    return {
        "source": str(NIGHT2_ROOT),
        "initial_prompt_path": str(REPO_ROOT / "research/context-ladder/prompts/prompt-4096.txt"),
        "initial_text_sha256": sha256_bytes(initial_text.encode("utf-8")),
        "follow_up": follow_up,
        "initial_prompt_ids": initial_ids,
        "reference_turn1_ids": turn1_ids,
        "reference_turn1_sha256": sha256_json(turn1_ids),
        "expected_cache_prefix_ids": expected_prefix_ids,
        "canonical_turn2_ids": canonical_ids,
        "canonical_turn2_sha256": sha256_json(canonical_ids),
        "exact_suffix_ids": suffix_ids,
        "exact_suffix_sha256": sha256_json(suffix_ids),
        "measured_prefix_from_prior_run": measured_prefix,
        "selected_assistant_text": assistant_text,
        "tokenizer_reconstruction": canonical,
        "artifact_hashes": {
            "initial_prompt": initial_artifact.get("sha256"),
            "reference_turn1": turn1_artifact.get("sha256"),
            "expected_prefix": prefix_artifact.get("sha256"),
            "canonical_turn2": canonical_artifact.get("sha256"),
            "exact_suffix": suffix_artifact.get("sha256"),
        },
    }


def output_checks(ids: list[int], text: str, previous: list[int], terms: list[str]) -> dict[str, Any]:
    common = lcp(ids, previous)
    lower = text.lower()
    return {
        "longest_common_prefix_with_turn1": common,
        "prefix_overlap_fraction": common / len(ids) if ids else 0.0,
        "exact_prefix_replay": bool(ids and ids == previous[: len(ids)]),
        "replay_detected": bool(ids and common / len(ids) >= 0.90),
        "term_presence": {term: term in lower for term in terms},
        "all_terms_present": all(term in lower for term in terms),
        "text_nonempty": bool(text.strip()),
        "classification": (
            "BROKEN"
            if not text.strip() or len(ids) < 8 or (ids and common / len(ids) >= 0.90)
            else "COHERENT"
        ),
    }


def path_a(
    mx: Any,
    generate_step: Any,
    make_prompt_cache: Any,
    make_sampler: Any,
    tokenizer: Any,
    model: Any,
    conversation: dict[str, Any],
    started: float,
) -> dict[str, Any]:
    cache = make_prompt_cache(model)
    stage = generate_stage(
        mx,
        generate_step,
        make_sampler,
        tokenizer,
        model,
        conversation["canonical_turn2_ids"],
        cache,
        FOLLOW_UP_MAX_TOKENS,
        "path_a_one_shot_turn2",
        started,
    )
    stage["output_checks"] = output_checks(
        stage["generated_token_ids"],
        stage["decoded_output"],
        conversation["reference_turn1_ids"],
        ["invariant", "enforce", "first"],
    )
    result = {
        "path": "A",
        "execution_mode": "fresh_one_shot_full_transcript",
        "logical_prompt_tokens": len(conversation["canonical_turn2_ids"]),
        "processed_suffix_tokens": None,
        "generation": stage,
    }
    del cache
    safe_clear(mx)
    return result


def path_segmented_or_persistent(
    mx: Any,
    generate_step: Any,
    make_prompt_cache: Any,
    make_sampler: Any,
    tokenizer: Any,
    model: Any,
    conversation: dict[str, Any],
    started: float,
    path_name: str,
    execution_mode: str,
) -> dict[str, Any]:
    cache = make_prompt_cache(model)
    initial_stage = generate_stage(
        mx,
        generate_step,
        make_sampler,
        tokenizer,
        model,
        conversation["initial_prompt_ids"],
        cache,
        INITIAL_MAX_TOKENS,
        f"path_{path_name.lower()}_turn1_initial",
        started,
    )
    turn1_ids = initial_stage["generated_token_ids"]
    measured_prefix = initial_stage["cache_diagnostics"][-1].get(
        "logical_prefix_token_count"
    )
    measured_prefix_source = "attention_cache_offset"
    if measured_prefix is None:
        measured_prefix = len(conversation["initial_prompt_ids"]) + len(turn1_ids)
        measured_prefix_source = "expected_prompt_plus_yielded_tokens_fallback"
    measured_prefix = int(measured_prefix)
    assistant_text = tokenizer.decode(turn1_ids, skip_special_tokens=True)
    reconstructed = canonical_prompt(
        tokenizer,
        [
            {"role": "user", "content": conversation["initial_text"]},
            {"role": "assistant", "content": assistant_text},
            {"role": "user", "content": conversation["follow_up"]},
        ],
    )
    reference_turn1_equal = turn1_ids == conversation["reference_turn1_ids"]
    if reference_turn1_equal:
        canonical_ids = conversation["canonical_turn2_ids"]
        suffix_ids = conversation["exact_suffix_ids"]
        canonical_source = "preserved_reference_turn1"
    else:
        canonical_ids = reconstructed["token_ids"]
        if measured_prefix > len(canonical_ids):
            raise RuntimeError(
                f"{path_name}: measured prefix {measured_prefix} exceeds reconstructed transcript"
            )
        suffix_ids = canonical_ids[measured_prefix:]
        canonical_source = "path_local_turn1_reconstruction"
    expected_stream = conversation["initial_prompt_ids"] + turn1_ids
    prefix_lcp = lcp(canonical_ids, expected_stream)
    prefix_exact_at_measurement = canonical_ids[:measured_prefix] == expected_stream[:measured_prefix]
    if measured_prefix > len(canonical_ids):
        raise RuntimeError(f"{path_name}: cache offset exceeds canonical transcript")
    if canonical_ids[measured_prefix:] != suffix_ids:
        raise RuntimeError(f"{path_name}: suffix accounting mismatch")
    turn2_stage = generate_stage(
        mx,
        generate_step,
        make_sampler,
        tokenizer,
        model,
        suffix_ids,
        cache,
        FOLLOW_UP_MAX_TOKENS,
        f"path_{path_name.lower()}_turn2_exact_suffix",
        started,
    )
    turn2_stage["output_checks"] = output_checks(
        turn2_stage["generated_token_ids"],
        turn2_stage["decoded_output"],
        turn1_ids,
        ["invariant", "enforce", "first"],
    )
    result = {
        "path": path_name,
        "execution_mode": execution_mode,
        "logical_prompt_tokens": len(canonical_ids),
        "processed_suffix_tokens": len(suffix_ids),
        "protocol": {
            "initial_prompt_tokens": len(conversation["initial_prompt_ids"]),
            "turn1_generated_tokens": len(turn1_ids),
            "expected_cached_prefix_tokens": len(expected_stream),
            "measured_cached_prefix_tokens": measured_prefix,
            "measured_prefix_source": measured_prefix_source,
            "canonical_turn2_tokens": len(canonical_ids),
            "exact_suffix_tokens": len(suffix_ids),
            "cached_prefix_plus_supplied_suffix_tokens": measured_prefix + len(suffix_ids),
            "canonical_prefix_lcp_with_local_stream": prefix_lcp,
            "canonical_prefix_exact_at_measured_boundary": prefix_exact_at_measurement,
            "reference_turn1_equal": reference_turn1_equal,
            "canonical_source": canonical_source,
            "suffix_sha256": sha256_json(suffix_ids),
            "suffix_token_ids": suffix_ids,
            "logical_equation_holds": (
                prefix_exact_at_measurement
                and measured_prefix + len(suffix_ids) == len(canonical_ids)
            ),
        },
        "turn1": initial_stage,
        "turn2": turn2_stage,
    }
    del cache
    safe_clear(mx)
    return result


def stage_without_internal(stage: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in stage.items()
        if key != "first_logprob_values"
    }


def path_without_internal(path: dict[str, Any]) -> dict[str, Any]:
    result = dict(path)
    if "generation" in result:
        result["generation"] = stage_without_internal(result["generation"])
    for key in ("turn1", "turn2"):
        if key in result:
            result[key] = stage_without_internal(result[key])
    return result


def find_diag(stage: dict[str, Any], suffix: str) -> dict[str, Any] | None:
    for item in stage.get("cache_diagnostics", []):
        if str(item.get("label", "")).endswith(suffix):
            return item
    return None


def boundary_map(path: dict[str, Any]) -> dict[str, dict[str, Any] | None]:
    if path["path"] == "A":
        return {"after_turn2_final": find_diag(path["generation"], "after_generation")}
    return {
        "after_turn1_final": find_diag(path["turn1"], "after_generation"),
        "before_turn2_suffix": find_diag(path["turn2"], "before_generation"),
        "after_turn2_suffix_and_first_generation_step": find_diag(
            path["turn2"], "after_prompt_and_first_generation_step"
        ),
        "after_turn2_first_yield": find_diag(path["turn2"], "after_first_yield"),
        "after_turn2_final": find_diag(path["turn2"], "after_generation"),
    }


def compare_diag(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    if left is None or right is None:
        return {"available": False, "signature_equal": None, "offsets_equal": None}
    left_layers = left.get("layers", [])
    right_layers = right.get("layers", [])
    mismatch_indices = [
        index
        for index, (a, b) in enumerate(zip(left_layers, right_layers))
        if a.get("state_signature_sha256") != b.get("state_signature_sha256")
        or a.get("cache_class") != b.get("cache_class")
        or a.get("offset") != b.get("offset")
    ]
    if len(left_layers) != len(right_layers):
        mismatch_indices.extend(range(min(len(left_layers), len(right_layers)), max(len(left_layers), len(right_layers))))
    return {
        "available": True,
        "signature_equal": left.get("state_signature_sha256") == right.get("state_signature_sha256"),
        "offsets_equal": left.get("offsets") == right.get("offsets"),
        "layer_count_left": len(left_layers),
        "layer_count_right": len(right_layers),
        "mismatching_layer_indices": mismatch_indices,
        "first_mismatching_layer": mismatch_indices[0] if mismatch_indices else None,
        "left_class_counts": left.get("class_counts"),
        "right_class_counts": right.get("class_counts"),
        "left_signature": left.get("state_signature_sha256"),
        "right_signature": right.get("state_signature_sha256"),
    }


def compare_logprobs(left: dict[str, Any], right: dict[str, Any], stage: str) -> dict[str, Any]:
    import numpy as np

    a = left.get("first_logprob_values")
    b = right.get("first_logprob_values")
    if a is None or b is None:
        return {"available": False, "stage": stage}
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    diff = np.abs(a - b)
    left_top = set(left.get("first_logprob_summary", {}).get("top_k_ids", []))
    right_top = set(right.get("first_logprob_summary", {}).get("top_k_ids", []))
    return {
        "available": True,
        "stage": stage,
        "max_abs_logprob_diff": float(np.max(diff)),
        "mean_abs_logprob_diff": float(np.mean(diff)),
        "top1_equal": left.get("first_logprob_summary", {}).get("top1_id")
        == right.get("first_logprob_summary", {}).get("top1_id"),
        "top_k_overlap": len(left_top & right_top),
        "top_k_size": max(len(left_top), len(right_top)),
        "left_logprob_sha256": left.get("first_logprob_summary", {}).get("logprob_sha256"),
        "right_logprob_sha256": right.get("first_logprob_summary", {}).get("logprob_sha256"),
    }


def compare_paths(path_b: dict[str, Any], path_c: dict[str, Any]) -> dict[str, Any]:
    b_boundaries = boundary_map(path_b)
    c_boundaries = boundary_map(path_c)
    boundary_comparisons = {
        name: compare_diag(b_boundaries.get(name), c_boundaries.get(name))
        for name in (
            "after_turn1_final",
            "before_turn2_suffix",
            "after_turn2_suffix_and_first_generation_step",
            "after_turn2_first_yield",
            "after_turn2_final",
        )
    }
    logits = [
        compare_logprobs(path_b["turn1"], path_c["turn1"], "turn1_first_generated_token"),
        compare_logprobs(path_b["turn2"], path_c["turn2"], "turn2_first_generated_token"),
    ]
    state_equal = all(
        item.get("signature_equal") is True for item in boundary_comparisons.values()
    )
    offsets_equal = all(
        item.get("offsets_equal") is True for item in boundary_comparisons.values()
    )
    turn1_equal = path_b["turn1"]["generated_token_ids"] == path_c["turn1"]["generated_token_ids"]
    turn2_equal = path_b["turn2"]["generated_token_ids"] == path_c["turn2"]["generated_token_ids"]
    no_replay = not path_b["turn2"]["output_checks"]["replay_detected"] and not path_c["turn2"]["output_checks"]["replay_detected"]
    return {
        "boundary_comparisons": boundary_comparisons,
        "logit_comparisons": logits,
        "all_boundary_state_signatures_equal": state_equal,
        "all_boundary_offsets_equal": offsets_equal,
        "turn1_token_stream_equal": turn1_equal,
        "turn2_token_stream_equal": turn2_equal,
        "protocol_equations_both_hold": path_b["protocol"]["logical_equation_holds"]
        and path_c["protocol"]["logical_equation_holds"],
        "no_stale_replay_in_b_or_c": no_replay,
        "gate": (
            "PASS"
            if state_equal
            and offsets_equal
            and turn1_equal
            and turn2_equal
            and no_replay
            and path_b["protocol"]["logical_equation_holds"]
            and path_c["protocol"]["logical_equation_holds"]
            else "MATERIAL_B_VS_C_DIVERGENCE"
        ),
    }


def append_stage3_result(record: dict[str, Any]) -> None:
    path = REPO_ROOT / "research/stage3/results.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def run_oracle(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    result: dict[str, Any] = {
        "experiment_id": "P1-SAME-BOUNDARY-G128",
        "phase": 1,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": current_commit(),
        "model_path": args.model,
        "exception": None,
        "safety_stop_reason": None,
    }
    try:
        mx, load, generate_step, make_prompt_cache, make_sampler, _ = load_runtime()
        result["model"] = model_metadata(args.model)
        result["os_memory_before"] = os_snapshot()
        model, tokenizer = load(args.model)
        result["memory_after_model_load"] = memory_snapshot(mx, "after_model_load", started)
        conversation = load_reference_conversation(tokenizer)
        conversation["initial_text"] = (
            REPO_ROOT / "research/context-ladder/prompts/prompt-4096.txt"
        ).read_text(encoding="utf-8")
        write_json(EXPERIMENT_ROOT / "tokens/conversation.json", conversation)

        # Path A is intentionally a single fresh-cache full-transcript call.
        path_a_result = path_a(
            mx, generate_step, make_prompt_cache, make_sampler, tokenizer, model, conversation, started
        )
        # B and C each create a new cache, but within each path the same cache
        # object survives turn 1 and the exact suffix. C is the real persistent
        # path; B is the fresh-cache segmented control.
        path_b_result = path_segmented_or_persistent(
            mx,
            generate_step,
            make_prompt_cache,
            make_sampler,
            tokenizer,
            model,
            conversation,
            started,
            "B",
            "fresh_segmented_same_boundaries",
        )
        path_c_result = path_segmented_or_persistent(
            mx,
            generate_step,
            make_prompt_cache,
            make_sampler,
            tokenizer,
            model,
            conversation,
            started,
            "C",
            "true_persistent_exact_suffix",
        )
        b_vs_c = compare_paths(path_b_result, path_c_result)
        a_vs_c = {
            "turn2_token_lcp": lcp(
                path_a_result["generation"]["generated_token_ids"],
                path_c_result["turn2"]["generated_token_ids"],
            ),
            "turn2_token_equal": path_a_result["generation"]["generated_token_ids"]
            == path_c_result["turn2"]["generated_token_ids"],
            "a_classification": path_a_result["generation"]["output_checks"]["classification"],
            "c_classification": path_c_result["turn2"]["output_checks"]["classification"],
            "a_replay_detected": path_a_result["generation"]["output_checks"]["replay_detected"],
            "c_replay_detected": path_c_result["turn2"]["output_checks"]["replay_detected"],
        }
        result["conversation"] = conversation
        result["paths"] = {
            "A": path_without_internal(path_a_result),
            "B": path_without_internal(path_b_result),
            "C": path_without_internal(path_c_result),
        }
        result["comparisons"] = {"B_vs_C": b_vs_c, "A_vs_C": a_vs_c}
        result["decision_gate"] = b_vs_c["gate"]
        result["memory_after_paths"] = memory_snapshot(mx, "after_all_paths", started)
        result["os_memory_after"] = os_snapshot()

        for label, path_result in (("path-a", path_a_result), ("path-b", path_b_result), ("path-c", path_c_result)):
            if label == "path-a":
                stage_records = {"turn2": path_result["generation"]}
                output_records = {"turn2": path_result["generation"]["decoded_output"]}
            else:
                stage_records = {"turn1": path_result["turn1"], "turn2": path_result["turn2"]}
                output_records = {
                    "turn1": path_result["turn1"]["decoded_output"],
                    "turn2": path_result["turn2"]["decoded_output"],
                }
            write_json(
                EXPERIMENT_ROOT / "state_signatures" / f"{label}.json",
                {
                    "path": path_result["path"],
                    "execution_mode": path_result["execution_mode"],
                    "stages": {
                        name: stage["cache_diagnostics"] for name, stage in stage_records.items()
                    },
                },
            )
            for turn_name, text in output_records.items():
                (EXPERIMENT_ROOT / "outputs" / f"{label}-{turn_name}.txt").write_text(
                    text, encoding="utf-8"
                )
            for turn_name, stage in stage_records.items():
                write_json(
                    EXPERIMENT_ROOT / "tokens" / f"{label}-{turn_name}.json",
                    {
                        "token_ids": stage["generated_token_ids"],
                        "token_count": stage["generated_tokens"],
                        "sha256": stage["generated_token_hash"],
                        "decoded_output_sha256": sha256_bytes(stage["decoded_output"].encode("utf-8")),
                    },
                )
        write_json(EXPERIMENT_ROOT / "results.json", result)
        lines = [
            "# Phase 1 same-boundary cache oracle",
            "",
            f"Decision gate: **{result['decision_gate']}**",
            "",
            "| path | execution | turn-1 tokens | turn-2 prompt/suffix | turn-2 output |",
            "|---|---|---:|---:|---|",
            f"| A | fresh one-shot full transcript | n/a | {len(conversation['canonical_turn2_ids'])} full | {path_a_result['generation']['generated_token_hash']} |",
            f"| B | fresh segmented exact suffix | {path_b_result['turn1']['generated_tokens']} | {path_b_result['turn2']['prompt_tokens']} suffix | {path_b_result['turn2']['generated_token_hash']} |",
            f"| C | true persistent exact suffix | {path_c_result['turn1']['generated_tokens']} | {path_c_result['turn2']['prompt_tokens']} suffix | {path_c_result['turn2']['generated_token_hash']} |",
            "",
            "B-vs-C boundary comparison:",
            "",
            "```json",
            json.dumps(b_vs_c, indent=2, sort_keys=True),
            "```",
            "",
            "Interpretation: A-vs-C output divergence is expected to be a separate",
            "one-shot-versus-segmented numerical-sensitivity question. The scaling",
            "gate is determined by B-vs-C state, logits, token stream, protocol",
            "accounting, and replay checks.",
            "",
        ]
        (EXPERIMENT_ROOT / "results.md").write_text("\n".join(lines), encoding="utf-8")
        append_stage3_result(
            {
                "timestamp": result["timestamp"],
                "experiment_id": result["experiment_id"],
                "phase": 1,
                "git_commit": result["git_commit"],
                "python_version": result["model"]["python_version"],
                "mlx_version": result["model"]["mlx_version"],
                "mlx_lm_revision": PINNED_MLX_LM_REVISION,
                "model_name": "Qwen3.8-27B-3bit-g128",
                "model_path": args.model,
                "quant_bits": result["model"]["quantization"]["bits"],
                "quant_group_size": result["model"]["quantization"]["group_size"],
                "execution_mode": "A/B/C_same_boundary_oracle",
                "persistent_cache": True,
                "kv_bits": None,
                "prompt_path": conversation["initial_prompt_path"],
                "prompt_sha256": conversation["initial_text_sha256"],
                "total_logical_tokens": len(conversation["canonical_turn2_ids"]),
                "processed_suffix_tokens": len(conversation["exact_suffix_ids"]),
                "generated_tokens": path_c_result["turn2"]["generated_tokens"],
                "generated_token_hash": path_c_result["turn2"]["generated_token_hash"],
                "classification": path_c_result["turn2"]["output_checks"]["classification"],
                "replay_detected": path_c_result["turn2"]["output_checks"]["replay_detected"],
                "structured_output_valid": None,
                "ttft_seconds": path_c_result["turn2"]["ttft_seconds"],
                "prompt_tps": path_c_result["turn2"]["prompt_tps"],
                "decode_tps": path_c_result["turn2"]["decode_tps"],
                "total_seconds": result["memory_after_paths"]["elapsed_seconds"],
                "active_memory_bytes": path_c_result["turn2"]["memory_after_generation"]["active_bytes"],
                "cache_memory_bytes": path_c_result["turn2"]["memory_after_generation"]["cache_bytes"],
                "active_plus_cache_bytes": path_c_result["turn2"]["memory_after_generation"]["active_plus_cache_bytes"],
                "peak_memory_bytes": path_c_result["turn2"]["memory_after_generation"]["peak_bytes"],
                "rss_bytes": path_c_result["turn2"]["memory_after_generation"]["rss_max_bytes"],
                "cache_state_signature": b_vs_c["boundary_comparisons"]["after_turn2_final"].get("right_signature"),
                "exception": None,
                "safety_stop_reason": None,
                "decision_gate": result["decision_gate"],
                "b_vs_c": b_vs_c,
                "a_vs_c": a_vs_c,
                "raw_results": str(EXPERIMENT_ROOT / "results.json"),
            }
        )
    except BaseException as exc:
        result["exception"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        message = str(exc).lower()
        if "out of memory" in message or "metal" in message or "gpu" in message:
            result["safety_stop_reason"] = "runtime_exception_requires_manual_safety_review"
        write_json(EXPERIMENT_ROOT / "results.json", result)
        (EXPERIMENT_ROOT / "logs/failure.log").write_text(result["traceback"], encoding="utf-8")
    finally:
        result["total_elapsed_seconds"] = time.perf_counter() - started
    print(
        json.dumps(
            {
                "experiment_id": result["experiment_id"],
                "decision_gate": result.get("decision_gate"),
                "exception": result.get("exception"),
                "elapsed_seconds": result["total_elapsed_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 1 if result.get("exception") else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    raise SystemExit(run_oracle(args))


if __name__ == "__main__":
    main()
