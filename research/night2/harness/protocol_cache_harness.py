#!/usr/bin/env python3
"""Protocol-aware Qwen3.8 cache correctness harness.

The script deliberately keeps MLX imports lazy so the orchestrator itself does
not hold a model.  It runs the suspect persistent protocol in one child
process, then starts a fresh child for each exact-token and full-transcript
oracle turn.  The outer command must be run with the canonical MLX 0.32.1
environment and with the local no-wired runtime dependencies available.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
PINNED_MLX_LM_REVISION = "254d153fdeb6f150edd4fc5a54f9828638481fa8"
DEFAULT_MODEL = "/Users/artem/.cache/huggingface/hub/models--lukaskremla--Qwen3.8-27B-3bit-MLX-TextOnly/snapshots/c98bba5926f51fec1c8d8737e577221673f524d7"
DEFAULT_FOLLOWUPS = [
    "Answer ONLY which single database invariant should be enforced first. Name the invariant and the enforcement boundary.",
    "Answer ONLY with one focused regression test and its expected failure assertion. Do not discuss other fixes.",
]


def json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def current_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return completed.stdout.strip()
    except BaseException:
        return None


def command_snapshot(command: list[str], tail: int = 12000) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=20,
        )
        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout[-tail:],
            "stderr": completed.stderr[-4000:],
        }
    except BaseException as exc:
        return {"exception": f"{type(exc).__name__}: {exc}"}


def os_snapshot() -> dict[str, Any]:
    # These are collected after a turn rather than inside the timing window.
    return {
        "memory_pressure": command_snapshot(["memory_pressure", "-Q"]),
        "vm_stat": command_snapshot(["vm_stat"]),
        "vm_swapusage": command_snapshot(["sysctl", "vm.swapusage"]),
    }


def _gib(value: int | None) -> float | None:
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
        "timestamp": time.time(),
        "elapsed_seconds": time.perf_counter() - started,
        "active_bytes": values["active"],
        "active_gib": _gib(values["active"]),
        "cache_bytes": values["cache"],
        "cache_gib": _gib(values["cache"]),
        "active_plus_cache_bytes": total,
        "active_plus_cache_gib": _gib(total),
        "peak_bytes": values["peak"],
        "peak_gib": _gib(values["peak"]),
        "rss_max_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }


def _is_array(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype") and hasattr(value, "nbytes")


def _flatten_arrays(value: Any) -> list[Any]:
    if value is None:
        return []
    if _is_array(value):
        return [value]
    if isinstance(value, (list, tuple)):
        result: list[Any] = []
        for item in value:
            result.extend(_flatten_arrays(item))
        return result
    return []


def _finite_json_value(value: Any) -> Any:
    if isinstance(value, bool | int | str) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, list):
        return [_finite_json_value(v) for v in value]
    if isinstance(value, tuple):
        return [_finite_json_value(v) for v in value]
    try:
        return _finite_json_value(value.item())
    except BaseException:
        return str(value)


def tensor_metadata(mx: Any, array: Any) -> dict[str, Any]:
    if array is None:
        return {"shape": None, "dtype": None, "nbytes": 0, "sample": None}
    result: dict[str, Any] = {
        "shape": [int(x) for x in array.shape],
        "dtype": str(array.dtype),
        "nbytes": int(array.nbytes),
    }
    try:
        flat = array.reshape((-1,))
        size = int(flat.size)
        if size <= 4:
            sample = flat
        else:
            sample = mx.concatenate([flat[:2], flat[-2:]])
        mx.eval(sample)
        result["sample"] = _finite_json_value(sample.tolist())
    except BaseException as exc:
        result["sample_error"] = f"{type(exc).__name__}: {exc}"
    return result


def cache_state_arrays(cache_entry: Any) -> list[Any]:
    if hasattr(cache_entry, "cache"):
        return _flatten_arrays(cache_entry.cache)
    if hasattr(cache_entry, "keys"):
        return _flatten_arrays([cache_entry.keys, cache_entry.values])
    try:
        return _flatten_arrays(cache_entry.state)
    except BaseException:
        return []


def cache_diagnostics(mx: Any, cache: Iterable[Any], label: str) -> dict[str, Any]:
    try:
        mx.synchronize()
    except BaseException:
        pass
    layers: list[dict[str, Any]] = []
    class_counts: dict[str, int] = {}
    offsets: list[int] = []
    signature_layers: list[Any] = []
    for index, entry in enumerate(cache):
        class_name = type(entry).__name__
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        item: dict[str, Any] = {
            "index": index,
            "class": class_name,
            "trimmable": None,
            "offset": None,
            "size": None,
            "nbytes": None,
            "states": [],
        }
        try:
            item["trimmable"] = bool(entry.is_trimmable())
        except BaseException:
            pass
        for attribute in ("offset", "size"):
            try:
                value = getattr(entry, attribute)
                value = value() if callable(value) else value
                item[attribute] = int(value)
                if attribute == "offset":
                    offsets.append(int(value))
            except BaseException:
                pass
        try:
            item["nbytes"] = int(entry.nbytes)
        except BaseException:
            pass
        states = [tensor_metadata(mx, array) for array in cache_state_arrays(entry)]
        item["states"] = [
            {key: value for key, value in state.items() if key != "sample"}
            for state in states
        ]
        signature_payload = {
            "class": class_name,
            "offset": item["offset"],
            "size": item["size"],
            "states": states,
        }
        item["state_signature_sha256"] = json_hash(signature_payload)
        signature_layers.append(signature_payload)
        layers.append(item)
    summary = {
        "layer_count": len(layers),
        "class_counts": class_counts,
        "offset_min": min(offsets) if offsets else None,
        "offset_max": max(offsets) if offsets else None,
        "state_signature_sha256": json_hash(signature_layers),
    }
    return {"label": label, "summary": summary, "layers": layers}


def token_ids(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict) and "input_ids" in value:
        value = value["input_ids"]
    return [int(item) for item in value]


def chat_tokens(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    value = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if isinstance(value, str):
        return [int(x) for x in tokenizer.encode(value, add_special_tokens=False)]
    return token_ids(value)


def raw_query_tokens(tokenizer: Any, query: str) -> list[int]:
    return [int(x) for x in tokenizer.encode(query, add_special_tokens=False)]


def token_labels(tokenizer: Any, ids: list[int]) -> list[str | None]:
    labels: list[str | None] = []
    for token_id in ids:
        try:
            value = tokenizer.convert_ids_to_tokens(int(token_id))
            labels.append(str(value))
        except BaseException:
            labels.append(None)
    return labels


def boundary_metadata(tokenizer: Any, previous_ids: list[int], current_ids: list[int]) -> dict[str, Any]:
    previous_tail = previous_ids[-12:]
    current_head = current_ids[:12]
    return {
        "previous_tail_ids": previous_tail,
        "previous_tail_tokens": token_labels(tokenizer, previous_tail),
        "current_head_ids": current_head,
        "current_head_tokens": token_labels(tokenizer, current_head),
        "joined_ids": previous_tail + current_head,
        "joined_tokens": token_labels(tokenizer, previous_tail + current_head),
    }


def longest_common_prefix(current: list[int], previous: list[int]) -> int:
    count = 0
    for left, right in zip(current, previous):
        if left != right:
            break
        count += 1
    return count


def max_contiguous_match(current: list[int], previous: list[int]) -> int:
    best = 0
    for i, left in enumerate(current):
        for j, right in enumerate(previous):
            if left != right:
                continue
            size = 0
            while i + size < len(current) and j + size < len(previous):
                if current[i + size] != previous[j + size]:
                    break
                size += 1
            best = max(best, size)
    return best


def ngram_set(ids: list[int], n: int) -> set[tuple[int, ...]]:
    if len(ids) < n:
        return set()
    return {tuple(ids[index : index + n]) for index in range(len(ids) - n + 1)}


def overlap_metrics(current: list[int], previous_answers: list[list[int]]) -> dict[str, Any]:
    all_metrics: list[dict[str, Any]] = []
    current_four = ngram_set(current, 4)
    current_eight = ngram_set(current, 8)
    for previous_index, previous in enumerate(previous_answers, start=1):
        lcp = longest_common_prefix(current, previous)
        four = ngram_set(previous, 4)
        eight = ngram_set(previous, 8)
        all_metrics.append(
            {
                "previous_turn": previous_index,
                "previous_tokens": len(previous),
                "longest_common_prefix_tokens": lcp,
                "prefix_overlap_fraction": lcp / len(current) if current else 0.0,
                "exact_prefix_replay": bool(current and current == previous[: len(current)]),
                "max_contiguous_match_tokens": max_contiguous_match(current, previous),
                "shared_4gram_count": len(current_four & four),
                "shared_8gram_count": len(current_eight & eight),
            }
        )
    max_prefix = max(
        (item["prefix_overlap_fraction"] for item in all_metrics),
        default=0.0,
    )
    return {
        "previous_answers": all_metrics,
        "max_prefix_overlap_fraction": max_prefix,
        "replay_detected": bool(max_prefix >= 0.90 and current),
    }


def semantic_checks(turn: int, text: str) -> dict[str, Any]:
    lower = text.lower()
    expected = {
        2: ["invariant", "enforce"],
        3: ["regression", "assert"],
    }.get(turn, [])
    return {
        "expected_distinctive_terms": expected,
        "term_presence": {term: term in lower for term in expected},
        "all_expected_terms_present": all(term in lower for term in expected),
        "note": "keyword presence is supporting evidence only; replay/oracle checks decide correctness",
    }


def classify_output(text: str, ids: list[int], turn: int, previous_answers: list[list[int]]) -> tuple[str, dict[str, Any]]:
    overlap = overlap_metrics(ids, previous_answers)
    stripped = text.strip()
    broken = not stripped or len(ids) < 8 or overlap["replay_detected"]
    if len(ids) >= 24:
        tail = ids[-min(32, len(ids)) :]
        if len(set(tail)) <= 2:
            broken = True
    if broken:
        classification = "BROKEN"
    else:
        punctuation = sum(char in "!.,;:?" for char in stripped)
        classification = "DEGRADED" if punctuation > max(48, len(stripped) * 0.75) else "COHERENT"
    return classification, {
        "overlap": overlap,
        "semantic": semantic_checks(turn, text),
    }


def load_runtime() -> tuple[Any, Any, Any, Any, Any, Any]:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler
    from research.runtime.qwen38_runtime import stream_generate_no_wired

    return mx, load, generate_step, make_prompt_cache, make_sampler, stream_generate_no_wired


def collect_generation(
    mx: Any,
    stream_generate_no_wired: Any,
    make_sampler: Any,
    tokenizer: Any,
    model: Any,
    prompt_ids: list[int],
    prompt_cache: Any,
    max_tokens: int,
    prefill_step_size: int,
    cache_label: str,
    started: float,
) -> dict[str, Any]:
    diagnostics: list[dict[str, Any]] = []

    def progress(processed: int, total: int) -> None:
        if processed == total:
            diagnostics.append(
                cache_diagnostics(mx, prompt_cache, cache_label)
            )

    generation_started = time.perf_counter()
    first_token_time: float | None = None
    responses: list[Any] = []
    for response in stream_generate_no_wired(
        model,
        tokenizer,
        mx.array(prompt_ids),
        max_tokens=max_tokens,
        sampler=make_sampler(0.0, 1.0),
        prompt_cache=prompt_cache,
        prefill_step_size=prefill_step_size,
        prompt_progress_callback=progress,
    ):
        if first_token_time is None:
            first_token_time = time.perf_counter()
            diagnostics.append(
                cache_diagnostics(mx, prompt_cache, "after_first_generated_token")
            )
        responses.append(response)
    completed = time.perf_counter()
    generated_ids = [int(response.token) for response in responses]
    try:
        raw_text = "".join(response.text for response in responses)
    except BaseException:
        raw_text = ""
    try:
        decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    except BaseException:
        decoded_text = raw_text
    last = responses[-1] if responses else None
    return {
        "generated_token_ids": generated_ids,
        "generated_token_hash": json_hash(generated_ids),
        "generated_tokens": len(generated_ids),
        "cache_committed_token_ids": generated_ids,
        "raw_text": raw_text,
        "decoded_text": decoded_text,
        "ttft_seconds": first_token_time - generation_started if first_token_time is not None else None,
        "prompt_tps": getattr(last, "prompt_tps", None),
        "decode_tps": getattr(last, "generation_tps", None),
        "generation_seconds": completed - first_token_time if first_token_time is not None else None,
        "total_seconds": completed - generation_started,
        "prompt_tokens": len(prompt_ids),
        "responses_seen": len(responses),
        "cache_diagnostics": diagnostics,
        "memory_after_generation": memory_snapshot(mx, "after_generation", started),
        "os_memory_after_generation": os_snapshot(),
    }


def model_metadata(model_path: str) -> dict[str, Any]:
    config = json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8"))
    quant = config.get("quantization", config.get("quantization_config"))
    return {
        "model_path": str(Path(model_path).resolve()),
        "quantization": quant,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "mlx_version": metadata.version("mlx"),
        "mlx_lm_version": metadata.version("mlx-lm"),
        "mlx_lm_revision": PINNED_MLX_LM_REVISION,
        "wired_policy": "direct generate_step; MLX wired state untouched",
    }


def build_turn_record(
    tokenizer: Any,
    turn: int,
    query: str,
    prompt_fragment: list[int],
    generation: dict[str, Any],
    previous_answers: list[list[int]],
    output_dir: Path,
    protocol_mode: str,
    oracle_prompt_ids: list[int] | None = None,
    prompt_kind: str = "incremental_chat_fragment",
) -> dict[str, Any]:
    generated_ids = generation["generated_token_ids"]
    classification, checks = classify_output(
        generation["decoded_text"], generated_ids, turn, previous_answers
    )
    output_path = output_dir / f"turn-{turn}.txt"
    token_path = output_dir / f"turn-{turn}.tokens.json"
    query_token_ids = raw_query_tokens(tokenizer, query)
    output_path.write_text(generation["raw_text"], encoding="utf-8")
    write_json(token_path, generated_ids)
    previous_boundary = previous_answers[-1] if previous_answers else []
    record = {
        "turn": turn,
        "query_text": query,
        "query_sha256": text_hash(query),
        "query_token_ids": query_token_ids,
        "query_token_count": len(query_token_ids),
        "prompt_kind": prompt_kind,
        "prompt_fragment_token_ids": prompt_fragment,
        "prompt_fragment_sha256": json_hash(prompt_fragment),
        "prompt_fragment_tokens": len(prompt_fragment),
        "oracle_prompt_token_ids": oracle_prompt_ids,
        "oracle_prompt_sha256": json_hash(oracle_prompt_ids) if oracle_prompt_ids is not None else None,
        "oracle_prompt_tokens": len(oracle_prompt_ids) if oracle_prompt_ids is not None else None,
        "generated_token_ids": generated_ids,
        "cache_committed_token_ids": generation["cache_committed_token_ids"],
        "generated_token_hash": generation["generated_token_hash"],
        "generated_tokens": generation["generated_tokens"],
        "raw_text": generation["raw_text"],
        "decoded_text": generation["decoded_text"],
        "raw_output_path": str(output_path),
        "raw_token_ids_path": str(token_path),
        "classification": classification,
        "replay_detected": checks["overlap"]["replay_detected"],
        "previous_turn_prefix_overlap": checks["overlap"]["max_prefix_overlap_fraction"],
        "overlap_metrics": checks["overlap"],
        "semantic_checks": checks["semantic"],
        "boundary": boundary_metadata(tokenizer, previous_boundary, prompt_fragment),
        "ttft_seconds": generation["ttft_seconds"],
        "prompt_tps": generation["prompt_tps"],
        "decode_tps": generation["decode_tps"],
        "generation_seconds": generation["generation_seconds"],
        "total_seconds": generation["total_seconds"],
        "memory_after_generation": generation["memory_after_generation"],
        "os_memory_after_generation": generation["os_memory_after_generation"],
        "cache_diagnostics": generation["cache_diagnostics"],
    }
    return record


def base_result(model_path: str, experiment_id: str, protocol_mode: str, plan: dict[str, Any]) -> dict[str, Any]:
    result = {
        "experiment_id": experiment_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": current_commit(),
        "protocol_mode": protocol_mode,
        "prompt_path": plan["initial_prompt_path"],
        "prompt_sha256": plan["initial_prompt_sha256"],
        "follow_up_prompts": plan["follow_up_prompts"],
        "prefill_step_size": plan["prefill_step_size"],
        "initial_max_tokens": plan["initial_max_tokens"],
        "follow_up_max_tokens": plan["follow_up_max_tokens"],
        "model": model_metadata(model_path),
        "turns": [],
        "exception": None,
        "safety_stop_reason": None,
    }
    return result


def run_mode_a(args: argparse.Namespace) -> int:
    plan = read_json(Path(args.conversation_file))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = base_result(args.model_path, args.experiment_id, "mode_a_persistent_incremental", plan)
    started = time.perf_counter()
    try:
        mx, load, _generate_step, make_prompt_cache, make_sampler, stream = load_runtime()
        model, tokenizer = load(args.model_path)
        result["memory_after_model_load"] = memory_snapshot(mx, "after_model_load", started)
        cache = make_prompt_cache(model)
        cache_history = [cache_diagnostics(mx, cache, "after_cache_creation")]
        previous_answers: list[list[int]] = []
        queries = plan["queries"]
        for index, query in enumerate(queries, start=1):
            fragment = chat_tokens(tokenizer, [{"role": "user", "content": query}])
            if index > 1:
                cache_history.append(
                    cache_diagnostics(mx, cache, "before_followup_prefill")
                )
            generation = collect_generation(
                mx,
                stream,
                make_sampler,
                tokenizer,
                model,
                fragment,
                cache,
                plan["initial_max_tokens"] if index == 1 else plan["follow_up_max_tokens"],
                plan["prefill_step_size"],
                "after_initial_prompt_prefill" if index == 1 else "after_followup_prompt_processing",
                started,
            )
            generation["cache_diagnostics"] = cache_history + generation["cache_diagnostics"]
            cache_history = []
            if index == 1:
                generation["cache_diagnostics"].append(
                    cache_diagnostics(mx, cache, "after_initial_generation")
                )
            else:
                generation["cache_diagnostics"].append(
                    cache_diagnostics(mx, cache, "after_followup_generation")
                )
            turn = build_turn_record(
                tokenizer,
                index,
                query,
                fragment,
                generation,
                previous_answers,
                output_dir,
                "mode_a_persistent_incremental",
            )
            result["turns"].append(turn)
            previous_answers.append(turn["cache_committed_token_ids"])
            peak = turn["memory_after_generation"].get("peak_gib")
            if peak is not None and peak >= args.stop_peak_gib:
                result["safety_stop_reason"] = (
                    f"peak {peak:.3f} GiB reached configured unattended stop {args.stop_peak_gib:.3f} GiB"
                )
                break
        result["os_memory_final"] = os_snapshot()
    except BaseException as exc:
        import traceback

        result["exception"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        result["total_elapsed_seconds"] = time.perf_counter() - started
        write_json(output_dir / "mode-a.json", result)
        print(json.dumps({"mode": "A", "output": str(output_dir / "mode-a.json"), "exception": result["exception"]}, sort_keys=True), flush=True)
    return 1 if result["exception"] else 0


def build_exact_prompt(a_result: dict[str, Any], turn: int) -> list[int]:
    sequence: list[int] = []
    for index in range(turn):
        prior = a_result["turns"][index]
        sequence.extend(prior["prompt_fragment_token_ids"])
        if index < turn - 1:
            sequence.extend(prior["cache_committed_token_ids"])
    return sequence


def build_full_transcript_prompt(tokenizer: Any, plan: dict[str, Any], a_result: dict[str, Any], turn: int) -> list[int]:
    messages: list[dict[str, str]] = []
    for index in range(turn):
        messages.append({"role": "user", "content": plan["queries"][index]})
        if index < turn - 1:
            messages.append({"role": "assistant", "content": a_result["turns"][index]["decoded_text"]})
    return chat_tokens(tokenizer, messages)


def run_oracle_child(args: argparse.Namespace) -> int:
    plan = read_json(Path(args.conversation_file))
    a_result = read_json(Path(args.mode_a_result))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = args.mode
    protocol_mode = "mode_b_exact_token_fresh" if mode == "b-child" else "mode_c_full_transcript_fresh"
    result = base_result(args.model_path, args.experiment_id, protocol_mode, plan)
    started = time.perf_counter()
    try:
        mx, load, _generate_step, make_prompt_cache, make_sampler, stream = load_runtime()
        model, tokenizer = load(args.model_path)
        result["memory_after_model_load"] = memory_snapshot(mx, "after_model_load", started)
        turn_index = args.turn_index
        query = plan["queries"][turn_index - 1]
        if mode == "b-child":
            oracle_prompt = build_exact_prompt(a_result, turn_index)
            prompt_kind = "exact_token_concatenation"
        else:
            oracle_prompt = build_full_transcript_prompt(tokenizer, plan, a_result, turn_index)
            prompt_kind = "full_transcript_chat_template"
        cache = make_prompt_cache(model)
        cache_created = cache_diagnostics(mx, cache, "after_cache_creation")
        generation = collect_generation(
            mx,
            stream,
            make_sampler,
            tokenizer,
            model,
            oracle_prompt,
            cache,
            plan["initial_max_tokens"] if turn_index == 1 else plan["follow_up_max_tokens"],
            plan["prefill_step_size"],
            "after_oracle_prompt_processing",
            started,
        )
        generation["cache_diagnostics"] = [cache_created] + generation["cache_diagnostics"]
        generation["cache_diagnostics"].append(cache_diagnostics(mx, cache, "after_oracle_generation"))
        previous_answers = [item["cache_committed_token_ids"] for item in a_result["turns"][: turn_index - 1]]
        fragment = oracle_prompt
        turn = build_turn_record(
            tokenizer,
            turn_index,
            query,
            fragment,
            generation,
            previous_answers,
            output_dir,
            protocol_mode,
            oracle_prompt_ids=oracle_prompt,
            prompt_kind=prompt_kind,
        )
        result["turns"].append(turn)
        result["oracle_kind"] = prompt_kind
        result["oracle_prompt_token_ids"] = oracle_prompt
        result["oracle_prompt_sha256"] = json_hash(oracle_prompt)
        result["oracle_prompt_tokens"] = len(oracle_prompt)
        result["os_memory_final"] = os_snapshot()
    except BaseException as exc:
        import traceback

        result["exception"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        result["total_elapsed_seconds"] = time.perf_counter() - started
        write_json(output_dir / f"turn-{args.turn_index}.json", result)
        print(json.dumps({"mode": mode, "turn": args.turn_index, "output": str(output_dir / f"turn-{args.turn_index}.json"), "exception": result["exception"]}, sort_keys=True), flush=True)
    return 1 if result["exception"] else 0


def child_command(args: argparse.Namespace, mode: str, output_dir: Path, output_file: Path | None = None, turn: int | None = None) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--mode",
        mode,
        "--model-path",
        args.model_path,
        "--conversation-file",
        args.conversation_file,
        "--output-dir",
        str(output_dir),
        "--experiment-id",
        args.experiment_id,
        "--stop-peak-gib",
        str(args.stop_peak_gib),
        "--initial-max-tokens",
        str(args.initial_max_tokens),
        "--follow-up-max-tokens",
        str(args.follow_up_max_tokens),
        "--prefill-step-size",
        str(args.prefill_step_size),
    ]
    if output_file is not None:
        command.extend(["--output-file", str(output_file)])
    if turn is not None:
        command.extend(["--turn-index", str(turn)])
    return command


def run_child(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            env=os.environ.copy(),
        )
    return completed.returncode


def oracle_relation(a: dict[str, Any], b: dict[str, Any] | None, c: dict[str, Any] | None) -> dict[str, Any]:
    a_turn = a
    b_turn = b
    c_turn = c
    a_ids = a_turn.get("generated_token_ids", [])
    b_ids = b_turn.get("generated_token_ids", []) if b_turn else None
    c_ids = c_turn.get("generated_token_ids", []) if c_turn else None
    a_b_equal = b_ids is not None and a_ids == b_ids
    b_c_equal = b_ids is not None and c_ids is not None and b_ids == c_ids
    a_c_equal = c_ids is not None and a_ids == c_ids
    if b_ids is None or c_ids is None:
        classification = "UNRESOLVED"
    elif not a_b_equal and not b_c_equal:
        classification = "BOTH"
    elif not a_b_equal:
        classification = "CACHE_STATE_BUG"
    elif not b_c_equal:
        classification = "INCREMENTAL_PROTOCOL_BUG"
    else:
        classification = "PASS"
    return {
        "turn": a_turn["turn"],
        "a_generated_token_hash": a_turn.get("generated_token_hash"),
        "exact_token_oracle_hash": b_turn.get("generated_token_hash") if b_turn else None,
        "full_transcript_oracle_hash": c_turn.get("generated_token_hash") if c_turn else None,
        "a_vs_b_exact_token_equal": a_b_equal,
        "b_vs_c_exact_token_equal": b_c_equal,
        "a_vs_c_exact_token_equal": a_c_equal,
        "classification": classification,
        "a_classification": a_turn.get("classification"),
        "b_classification": b_turn.get("classification") if b_turn else None,
        "c_classification": c_turn.get("classification") if c_turn else None,
        "a_replay_detected": a_turn.get("replay_detected"),
        "oracle_relation": (
            "A=B=C"
            if classification == "PASS"
            else "A differs from B; cache state is suspect"
            if classification == "CACHE_STATE_BUG"
            else "A approximately matches B, but exact incremental stream differs from full transcript"
            if classification == "INCREMENTAL_PROTOCOL_BUG"
            else "A differs from B and B differs from C"
            if classification == "BOTH"
            else "oracle child unavailable"
        ),
    }


def append_jsonl(path: Path, entries: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        for entry in entries:
            stream.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")


def orchestrate(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty raw artifact directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = Path(args.prompt_file).resolve()
    initial_prompt = prompt_path.read_text(encoding="utf-8")
    followups = args.follow_up if args.follow_up else DEFAULT_FOLLOWUPS
    if len(followups) < 2:
        raise SystemExit("at least two strongly differentiated follow-ups are required")
    plan = {
        "initial_prompt_path": str(prompt_path),
        "initial_prompt_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
        "queries": [initial_prompt, *followups],
        "follow_up_prompts": followups,
        "prefill_step_size": args.prefill_step_size,
        "initial_max_tokens": args.initial_max_tokens,
        "follow_up_max_tokens": args.follow_up_max_tokens,
    }
    plan_path = output_dir / "conversation-plan.json"
    write_json(plan_path, plan)
    args.conversation_file = str(plan_path)

    mode_a_dir = output_dir / "mode-a"
    mode_a_result = mode_a_dir / "mode-a.json"
    mode_a_command = child_command(args, "a-child", mode_a_dir)
    mode_a_rc = run_child(mode_a_command, output_dir / "mode-a.log")
    if mode_a_rc != 0 or not mode_a_result.exists():
        aggregate = {
            "experiment_id": args.experiment_id,
            "protocol_mode": "orchestrator",
            "conversation_plan": str(plan_path),
            "mode_a_returncode": mode_a_rc,
            "exception": "Mode A child failed; oracle branches not started",
        }
        write_json(output_dir / "aggregate.json", aggregate)
        return 1

    b_results: dict[int, dict[str, Any]] = {}
    c_results: dict[int, dict[str, Any]] = {}
    oracle_failures: list[str] = []
    if not args.skip_oracles:
        for mode, collection, label in (
            ("b-child", b_results, "mode-b-exact"),
            ("c-child", c_results, "mode-c-transcript"),
        ):
            for turn in range(1, len(plan["queries"]) + 1):
                child_dir = output_dir / label
                command = child_command(args, mode, child_dir, turn=turn)
                command.extend(["--mode-a-result", str(mode_a_result)])
                rc = run_child(command, output_dir / f"{label}-turn-{turn}.log")
                child_path = child_dir / f"turn-{turn}.json"
                if rc != 0 or not child_path.exists():
                    oracle_failures.append(f"{label} turn {turn} returncode={rc}")
                    continue
                loaded = read_json(child_path)
                if loaded.get("turns"):
                    collection[turn] = loaded["turns"][0]

    a_result = read_json(mode_a_result)
    relations = []
    for a_turn in a_result.get("turns", []):
        turn = int(a_turn["turn"])
        relations.append(oracle_relation(a_turn, b_results.get(turn), c_results.get(turn)))
    if oracle_failures:
        overall = "UNRESOLVED"
    elif args.skip_oracles:
        overall = "MODE_A_ONLY"
    else:
        classifications = {item["classification"] for item in relations}
        overall = "PASS" if classifications == {"PASS"} else ("BOTH" if "BOTH" in classifications else next(iter(classifications), "UNRESOLVED"))
    aggregate = {
        "experiment_id": args.experiment_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": current_commit(),
        "protocol_mode": "orchestrator",
        "model_path": args.model_path,
        "conversation_plan": str(plan_path),
        "mode_a_result": str(mode_a_result),
        "mode_b_results": {str(k): str(output_dir / "mode-b-exact" / f"turn-{k}.json") for k in b_results},
        "mode_c_results": {str(k): str(output_dir / "mode-c-transcript" / f"turn-{k}.json") for k in c_results},
        "relations": relations,
        "overall_classification": overall,
        "oracle_failures": oracle_failures,
        "exception": a_result.get("exception"),
        "safety_stop_reason": a_result.get("safety_stop_reason"),
    }
    aggregate_path = output_dir / "aggregate.json"
    write_json(aggregate_path, aggregate)

    result_entries: list[dict[str, Any]] = []
    for a_turn in a_result.get("turns", []):
        turn = int(a_turn["turn"])
        relation = next(item for item in relations if item["turn"] == turn)
        result_entries.append(
            {
                "timestamp": aggregate["timestamp"],
                "experiment_id": f"{args.experiment_id}-A-T{turn}",
                "git_commit": aggregate["git_commit"],
                "python_version": a_result["model"].get("python_version"),
                "mlx_version": a_result["model"].get("mlx_version"),
                "mlx_lm_revision": PINNED_MLX_LM_REVISION,
                "model_path": args.model_path,
                "quantization": a_result["model"].get("quantization"),
                "protocol_mode": "mode_a_persistent_incremental",
                "prompt_path": plan["initial_prompt_path"],
                "prompt_sha256": plan["initial_prompt_sha256"],
                "prompt_tokens": a_turn.get("prompt_fragment_tokens"),
                "incremental_prompt_tokens": a_turn.get("prompt_fragment_tokens"),
                "generated_tokens": a_turn.get("generated_tokens"),
                "generated_token_hash": a_turn.get("generated_token_hash"),
                "previous_turn_prefix_overlap": a_turn.get("previous_turn_prefix_overlap"),
                "replay_detected": a_turn.get("replay_detected"),
                "classification": a_turn.get("classification"),
                "exact_token_oracle_hash": relation.get("exact_token_oracle_hash"),
                "full_transcript_oracle_hash": relation.get("full_transcript_oracle_hash"),
                "oracle_relation": relation.get("classification"),
                "ttft_seconds": a_turn.get("ttft_seconds"),
                "prompt_tps": a_turn.get("prompt_tps"),
                "decode_tps": a_turn.get("decode_tps"),
                "total_seconds": a_turn.get("total_seconds"),
                "active_gib": a_turn.get("memory_after_generation", {}).get("active_gib"),
                "cache_gib": a_turn.get("memory_after_generation", {}).get("cache_gib"),
                "active_plus_cache_gib": a_turn.get("memory_after_generation", {}).get("active_plus_cache_gib"),
                "peak_gib": a_turn.get("memory_after_generation", {}).get("peak_gib"),
                "os_memory_metrics": a_turn.get("os_memory_after_generation"),
                "cache_layer_summary": [item.get("summary") for item in a_turn.get("cache_diagnostics", [])],
                "exception": a_result.get("exception"),
                "safety_stop_reason": a_result.get("safety_stop_reason"),
                "raw_artifact": str(aggregate_path),
            }
        )
    for relation in relations:
        for mode_label, collection in (("mode_b_exact_token_fresh", b_results), ("mode_c_full_transcript_fresh", c_results)):
            turn = relation["turn"]
            child = collection.get(turn)
            if child is None:
                continue
            result_entries.append(
                {
                    "timestamp": aggregate["timestamp"],
                    "experiment_id": f"{args.experiment_id}-{mode_label}-T{turn}",
                    "git_commit": aggregate["git_commit"],
                    "python_version": a_result["model"].get("python_version"),
                    "mlx_version": a_result["model"].get("mlx_version"),
                    "mlx_lm_revision": PINNED_MLX_LM_REVISION,
                    "model_path": args.model_path,
                    "quantization": a_result["model"].get("quantization"),
                    "protocol_mode": mode_label,
                    "prompt_path": plan["initial_prompt_path"],
                    "prompt_sha256": plan["initial_prompt_sha256"],
                    "prompt_tokens": child.get("prompt_fragment_tokens"),
                    "incremental_prompt_tokens": None,
                    "generated_tokens": child.get("generated_tokens"),
                    "generated_token_hash": child.get("generated_token_hash"),
                    "previous_turn_prefix_overlap": child.get("previous_turn_prefix_overlap"),
                    "replay_detected": child.get("replay_detected"),
                    "classification": child.get("classification"),
                    "exact_token_oracle_hash": relation.get("exact_token_oracle_hash"),
                    "full_transcript_oracle_hash": relation.get("full_transcript_oracle_hash"),
                    "oracle_relation": relation.get("classification"),
                    "ttft_seconds": child.get("ttft_seconds"),
                    "prompt_tps": child.get("prompt_tps"),
                    "decode_tps": child.get("decode_tps"),
                    "total_seconds": child.get("total_seconds"),
                    "active_gib": child.get("memory_after_generation", {}).get("active_gib"),
                    "cache_gib": child.get("memory_after_generation", {}).get("cache_gib"),
                    "active_plus_cache_gib": child.get("memory_after_generation", {}).get("active_plus_cache_gib"),
                    "peak_gib": child.get("memory_after_generation", {}).get("peak_gib"),
                    "os_memory_metrics": child.get("os_memory_after_generation"),
                    "cache_layer_summary": [item.get("summary") for item in child.get("cache_diagnostics", [])],
                    "exception": None,
                    "safety_stop_reason": None,
                    "raw_artifact": str(aggregate_path),
                }
            )
    append_jsonl(REPO_ROOT / "research/night2/results.jsonl", result_entries)
    print(json.dumps({"aggregate": str(aggregate_path), "overall_classification": overall, "oracle_failures": oracle_failures}, sort_keys=True), flush=True)
    return 0 if overall in {"PASS", "MODE_A_ONLY", "CACHE_STATE_BUG", "INCREMENTAL_PROTOCOL_BUG", "BOTH"} else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("orchestrate", "a-child", "b-child", "c-child"), default="orchestrate")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--prompt-file", default="research/context-ladder/prompts/prompt-4096.txt")
    parser.add_argument("--conversation-file", default=None)
    parser.add_argument("--output-dir", default="research/night2/cache-probes/default")
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--mode-a-result", default=None)
    parser.add_argument("--experiment-id", default="P3-PROTOCOL")
    parser.add_argument("--turn-index", type=int, default=None)
    parser.add_argument("--initial-max-tokens", type=int, default=128)
    parser.add_argument("--follow-up-max-tokens", type=int, default=64)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--stop-peak-gib", type=float, default=16.0)
    parser.add_argument("--skip-oracles", action="store_true")
    parser.add_argument("--follow-up", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "orchestrate":
        return orchestrate(args)
    if args.conversation_file is None:
        raise SystemExit("--conversation-file is required for child modes")
    if args.mode == "a-child":
        return run_mode_a(args)
    if args.mode_a_result is None or args.turn_index is None:
        raise SystemExit("oracle child modes require --mode-a-result and --turn-index")
    return run_oracle_child(args)


if __name__ == "__main__":
    raise SystemExit(main())
