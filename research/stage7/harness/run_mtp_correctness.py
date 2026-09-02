#!/usr/bin/env python3
"""Guarded Qwen3.8 native-MTP correctness prototype.

This is intentionally a research harness, not a production integration.  It
uses the local executable mlx_vlm Qwen3.5 MTP reference for the draft block,
but keeps the target model and target cache in the pinned MLX-LM runtime.
Rejected target verification suffixes are handled by a deliberately slow
snapshot-and-replay path so both KV and GatedDelta/ArraysCache state are
preserved before any performance work is considered.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
SCALE_DIR = ROOT / "research/stage3/persistent-scaling/harness"
ORACLE_DIR = ROOT / "research/stage3/cache-boundary-oracle/harness"
for path in (SCALE_DIR, ORACLE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_scale as scale_utils  # noqa: E402


DEFAULT_MODEL = "/Users/artem/models/Qwen3.8-27B-3bit-g128-MLX"
DEFAULT_SOURCE = "/Users/artem/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
DEFAULT_MAX_TOKENS = 32
MTP_BLOCK_SIZE = 3
PINNED_MLX_LM_REVISION = "254d153fdeb6f150edd4fc5a54f9828638481fa8"

PROMPTS = [
    (
        "coding",
        "A Python function returns a mutable default list and state leaks between calls. State the minimal fix and why it works.",
    ),
    (
        "reasoning",
        "A sequence is [2, 4, 2, 6, 4]. If each distinct value is counted once, what is the count? Reply with the number and one short explanation.",
    ),
    (
        "prose",
        "Continue this sentence in two concise sentences: The small observatory opened its roof after the storm.",
    ),
    (
        "structured",
        'Return JSON only with keys "answer" and "reason" for the question: what is 3 + 5?',
    ),
]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def exact_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def scalar(value: Any) -> int:
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def ids(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    return [int(item) for item in value]


def tokenizer_eos_ids(tokenizer: Any) -> set[int]:
    value = getattr(tokenizer, "eos_token_ids", None)
    if value is None:
        value = getattr(tokenizer, "eos_token_id", None)
    if value is None:
        return set()
    if isinstance(value, int):
        return {int(value)}
    return {int(item) for item in value}


def _copy_array(mx: Any, value: Any) -> Any:
    if value is None:
        return None
    return mx.contiguous(mx.array(value, dtype=value.dtype))


def _copy_cache_entry(mx: Any, entry: Any, arrays_cache_type: Any, kv_cache_type: Any) -> Any:
    """Copy the two cache classes used by pinned Qwen3.5 text inference."""
    if entry is None:
        return None
    if isinstance(entry, arrays_cache_type):
        out = arrays_cache_type(len(entry.cache))
        out.cache = [_copy_array(mx, item) for item in entry.cache]
        if getattr(entry, "left_padding", None) is not None:
            out.left_padding = _copy_array(mx, entry.left_padding)
        if getattr(entry, "lengths", None) is not None:
            out.lengths = _copy_array(mx, entry.lengths)
        return out
    if isinstance(entry, kv_cache_type):
        out = kv_cache_type()
        out.keys = _copy_array(mx, entry.keys)
        out.values = _copy_array(mx, entry.values)
        out.offset = int(entry.offset)
        return out
    raise TypeError(f"unsupported target cache entry: {type(entry).__name__}")


def copy_cache(mx: Any, cache: list[Any]) -> list[Any]:
    from mlx_lm.models.cache import ArraysCache, KVCache

    out = [_copy_cache_entry(mx, entry, ArraysCache, KVCache) for entry in cache]
    arrays: list[Any] = []
    for entry in out:
        if entry is None:
            continue
        if isinstance(entry, ArraysCache):
            arrays.extend(item for item in entry.cache if item is not None)
        else:
            if entry.keys is not None:
                arrays.extend([entry.keys, entry.values])
    if arrays:
        mx.eval(arrays)
    return out


def cache_arrays(cache: list[Any]) -> list[Any]:
    arrays: list[Any] = []
    for entry in cache:
        if entry is None:
            continue
        state = getattr(entry, "cache", None)
        if state is not None:
            arrays.extend(item for item in state if item is not None)
        elif getattr(entry, "keys", None) is not None:
            arrays.extend([entry.keys, entry.values])
    return arrays


def _profile_count(profile: Any | None, name: str, amount: int = 1) -> None:
    if profile is not None:
        profile.count(name, amount)


def _profile_event(profile: Any | None, name: str, **payload: Any) -> None:
    if profile is not None:
        profile.event(name, **payload)


def _profile_start(profile: Any | None, name: str) -> None:
    if profile is not None:
        profile.start(name)


def _profile_stop(profile: Any | None, name: str, **payload: Any) -> None:
    if profile is not None:
        profile.stop(name, **payload)


def eval_cache(
    mx: Any,
    cache: list[Any],
    hidden: Any | None = None,
    profile: Any | None = None,
    phase: str = "other",
) -> None:
    arrays = cache_arrays(cache)
    if hidden is not None:
        arrays.append(hidden)
    if arrays:
        _profile_start(profile, "mx_eval")
        mx.eval(arrays)
        _profile_stop(profile, "mx_eval", phase=phase, array_count=len(arrays))


def target_forward_sequential(
    mx: Any,
    target_inner: Any,
    input_ids: Any,
    cache: list[Any],
    profile: Any | None = None,
    phase: str = "verify",
) -> Any:
    """Run target tokens one at a time, matching baseline decode semantics."""
    hidden_parts = []
    for index in range(int(input_ids.shape[1])):
        _profile_count(profile, "target_forward_calls")
        _profile_count(profile, "target_tokens_processed")
        if phase == "replay":
            _profile_count(profile, "replayed_target_calls")
            _profile_count(profile, "replayed_target_tokens")
        step_hidden = target_inner(input_ids[:, index : index + 1], cache=cache)
        # Force each recurrent/cache transition before the next token. This is
        # deliberately slow; Phase 8 is an exactness oracle, not a speed path.
        eval_cache(mx, cache, step_hidden, profile, phase)
        hidden_parts.append(step_hidden)
    if not hidden_parts:
        return mx.zeros((1, 0, 5120), dtype=mx.bfloat16)
    return mx.concatenate(hidden_parts, axis=1)


def target_prefill_baseline_schedule(
    mx: Any,
    target_inner: Any,
    input_ids: Any,
    cache: list[Any],
    prefill_step_size: int = 2048,
    profile: Any | None = None,
) -> Any:
    """Match pinned ``generate_step``'s chunked prefill and final decode step."""
    if int(input_ids.shape[1]) == 0:
        return mx.zeros((1, 0, 5120), dtype=mx.bfloat16)
    hidden_parts = []
    total = int(input_ids.shape[1])
    processed = 0
    while total - processed > 1:
        remaining = total - processed - 1
        n_to_process = min(int(prefill_step_size), remaining)
        _profile_count(profile, "target_forward_calls")
        _profile_count(profile, "target_tokens_processed", n_to_process)
        prefix_hidden = target_inner(
            input_ids[:, processed : processed + n_to_process], cache=cache
        )
        eval_cache(mx, cache, prefix_hidden, profile, "prefill")
        hidden_parts.append(prefix_hidden)
        processed += n_to_process
    _profile_count(profile, "target_forward_calls")
    _profile_count(profile, "target_tokens_processed")
    last_hidden = target_inner(input_ids[:, -1:], cache=cache)
    eval_cache(mx, cache, last_hidden, profile, "prefill")
    hidden_parts.append(last_hidden)
    return mx.concatenate(hidden_parts, axis=1)


def load_mtp_draft(
    mx: Any,
    source_path: Path,
    model: Any,
    weights_path: Path | None = None,
    profile: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load only the source MTP tensors through the local executable reference."""
    from mlx_vlm.speculative.drafters.qwen3_5_mtp import Qwen3_5MTPDraftModel
    from mlx_vlm.speculative.drafters.qwen3_5_mtp.config import Qwen3_5MTPConfig

    config = json.loads((source_path / "config.json").read_text(encoding="utf-8"))
    text_config = config["text_config"]
    draft_config = Qwen3_5MTPConfig.from_dict(
        {"model_type": "qwen3_5_mtp", "text_config": text_config}
    )
    draft = Qwen3_5MTPDraftModel(draft_config)

    if weights_path is None:
        shard_path = source_path / "model-00018-of-00018.safetensors"
        shard = mx.load(str(shard_path))
        mtp_weights = {key: value for key, value in shard.items() if key.startswith("mtp.")}
        if len(mtp_weights) != 15:
            raise RuntimeError(f"expected 15 source MTP tensors, found {len(mtp_weights)}")
        source_payload = sum(int(value.nbytes) for value in mtp_weights.values())
        # The source HF/Qwen RMSNorm parameter is centered at zero and the MLX
        # RMSNorm parameter is the direct multiplier.  The local reference's
        # sanitizer performs this exact conversion.
        mtp_weights = Qwen3_5MTPDraftModel.sanitize(None, mtp_weights)
        draft.load_weights(list(mtp_weights.items()))
        del shard, mtp_weights
        precision = "bf16"
        reference = "mlx_vlm 0.6.13 qwen3_5_mtp"
    else:
        import mlx.nn as nn

        quantization = json.loads((weights_path.parent / "config.json").read_text(encoding="utf-8")).get("quantization", {})
        nn.quantize(
            draft,
            group_size=int(quantization.get("group_size", 128)),
            bits=int(quantization.get("bits", 4)),
            mode=quantization.get("mode", "affine"),
            class_predicate=lambda _path, module: isinstance(module, nn.Linear),
        )
        candidate_weights = mx.load(str(weights_path))
        draft.load_weights(list(candidate_weights.items()))
        source_payload = int(json.loads((weights_path.parent / "manifest.json").read_text(encoding="utf-8")).get("source_mtp_payload_bytes", 0))
        del candidate_weights
        precision = f"{int(quantization.get('bits', 4))}-bit"
        reference = "stage7 source-backed affine MTP artifact"
    mx.eval(draft.parameters())
    gc.collect()
    _profile_start(profile, "cache_clear")
    mx.clear_cache()
    _profile_stop(profile, "cache_clear")
    draft.reset(model)
    return draft, {
        "source_tensor_count": 15,
        "source_payload_bytes": source_payload,
        "source_precision": "bf16",
        "loaded_precision": precision,
        "block_size": int(draft_config.block_size),
        "reference": reference,
        "norm_conversion": "raw HF centered RMSNorm vectors shifted by +1.0",
    }


def baseline_greedy(
    mx: Any,
    generate_step: Any,
    make_sampler: Any,
    make_prompt_cache: Any,
    model: Any,
    tokenizer: Any,
    prompt_ids: list[int],
    max_tokens: int,
) -> dict[str, Any]:
    cache = make_prompt_cache(model)
    generator = generate_step(
        mx.array(prompt_ids),
        model,
        max_tokens=max_tokens,
        sampler=make_sampler(0.0, 1.0),
        prompt_cache=cache,
        prefill_step_size=2048,
    )
    eos_ids = tokenizer_eos_ids(tokenizer)
    generated: list[int] = []
    finish_reason = "length"
    try:
        while len(generated) < max_tokens:
            try:
                token, _ = next(generator)
            except StopIteration:
                finish_reason = "generator_stop"
                break
            token_id = scalar(token)
            generated.append(token_id)
            if token_id in eos_ids:
                finish_reason = "eos"
                break
    finally:
        generator.close()
    try:
        mx.synchronize()
    except BaseException:
        pass
    del cache
    return {
        "token_ids": generated,
        "token_hash": exact_hash(generated),
        "finish_reason": finish_reason,
        "text": tokenizer.decode(generated, skip_special_tokens=True),
    }


def mtp_greedy(
    mx: Any,
    model: Any,
    tokenizer: Any,
    draft: Any,
    prompt_ids: list[int],
    max_tokens: int,
    batch_verify: bool = False,
    profile: Any | None = None,
) -> dict[str, Any]:
    """Run one exact-greedy MTP stream with explicit hybrid rollback."""
    mtp_started = time.perf_counter()
    language_model = model.language_model
    target_inner = language_model.model
    target_head = language_model.lm_head
    target_cache = model.make_cache()
    prompt = mx.array(prompt_ids, dtype=mx.int32)[None, :]
    target_prefill_started = time.perf_counter()
    _profile_event(profile, "target_prefill_start")
    hidden = target_prefill_baseline_schedule(
        mx, target_inner, prompt, target_cache, profile=profile
    )
    target_prefill_seconds = time.perf_counter() - target_prefill_started
    _profile_event(profile, "target_prefill_end", seconds=target_prefill_seconds)
    first_logits_started = time.perf_counter()
    first_logits = target_head(hidden[:, -1:, :])
    first_bonus = scalar(mx.argmax(first_logits, axis=-1).reshape(-1)[0])
    first_logits_seconds = time.perf_counter() - first_logits_started
    _profile_event(profile, "first_target_logits", seconds=first_logits_seconds)

    eos_ids = tokenizer_eos_ids(tokenizer)
    generated = [first_bonus]
    draft.reset(model)
    if first_bonus in eos_ids or max_tokens <= 1:
        decode_seconds = time.perf_counter() - mtp_started
        return {
            "token_ids": generated[:max_tokens],
            "token_hash": exact_hash(generated[:max_tokens]),
            "finish_reason": "eos" if first_bonus in eos_ids else "length",
            "text": tokenizer.decode(generated[:max_tokens], skip_special_tokens=True),
            "proposed_tokens": 0,
            "accepted_tokens": 0,
            "rejected_tokens": 0,
            "fallback_tokens": 0,
            "rounds": 0,
            "rollback_checks": 0,
            "rollback_exact": 0,
            "rollback_failures": [],
            "timing": {
                "total_seconds": decode_seconds,
                "ttft_seconds": decode_seconds,
                "target_prefill_seconds": target_prefill_seconds,
                "first_logits_seconds": first_logits_seconds,
                "draft_prefill_seconds": 0.0,
                "proposal_seconds": 0.0,
                "verify_seconds": 0.0,
                "replay_seconds": 0.0,
                "draft_update_seconds": 0.0,
                "decode_seconds": 0.0,
                "verification_mode": "batch" if batch_verify else "sequential",
            },
        }

    hidden_for_bonus = hidden[:, -1:, :]
    draft_prefill_started = time.perf_counter()
    _profile_start(profile, "draft_prefill")
    draft.prefill_from_target_hidden(
        prompt,
        hidden,
        first_bonus,
        None,
        token_dtype=mx.int32,
        greedy=True,
    )
    _profile_start(profile, "synchronize")
    mx.synchronize()
    _profile_stop(profile, "synchronize", phase="draft_prefill")
    _profile_stop(profile, "draft_prefill")
    draft_prefill_seconds = time.perf_counter() - draft_prefill_started
    draft.set_shared_kv({}, len(prompt_ids), position=max(len(prompt_ids) - 1, 0), kv_valid_len=len(prompt_ids))

    proposed_total = 0
    accepted_total = 0
    rejected_total = 0
    fallback_total = 0
    rounds = 0
    rollback_checks = 0
    rollback_exact = 0
    rollback_failures: list[dict[str, Any]] = []
    round_trace: list[dict[str, Any]] = []
    finish_reason = "length"
    bonus = first_bonus
    proposal_seconds = 0.0
    verify_seconds = 0.0
    replay_seconds = 0.0
    draft_update_seconds = 0.0
    decode_started = time.perf_counter()

    while len(generated) < max_tokens:
        _profile_event(profile, "round_start", round=rounds + 1)
        proposal_started = time.perf_counter()
        _profile_start(profile, "draft_proposal")
        draft_tokens = draft.draft_block(
            bonus,
            hidden_for_bonus,
            None,
            MTP_BLOCK_SIZE,
            None,
            token_dtype=mx.int32,
            greedy=True,
        )
        draft_list = ids(draft_tokens)
        _profile_count(profile, "draft_calls")
        _profile_count(profile, "draft_tokens_proposed", len(draft_list))
        _profile_stop(profile, "draft_proposal", proposed_tokens=len(draft_list))
        proposal_seconds += time.perf_counter() - proposal_started
        if not draft_list:
            raise RuntimeError("MTP draft produced zero tokens")
        proposed_total += len(draft_list)

        before_verify = target_cache
        if profile is not None:
            profile.cache_checkpoint("before_snapshot", before_verify)
        _profile_start(profile, "target_cache_snapshot")
        verify_cache = copy_cache(mx, before_verify)
        _profile_stop(profile, "target_cache_snapshot")
        if profile is not None:
            profile.cache_checkpoint("after_snapshot", verify_cache)
        verify_input = mx.concatenate(
            [mx.array([[bonus]], dtype=mx.int32), draft_tokens.astype(mx.int32)], axis=1
        )
        verify_started = time.perf_counter()
        _profile_start(profile, "target_verification")
        if batch_verify:
            _profile_count(profile, "target_forward_calls")
            _profile_count(profile, "target_tokens_processed", int(verify_input.shape[1]))
            verify_hidden = target_inner(verify_input, cache=verify_cache)
            eval_cache(mx, verify_cache, verify_hidden, profile, "verify")
        else:
            verify_hidden = target_forward_sequential(
                mx, target_inner, verify_input, verify_cache, profile, "verify"
            )
        verify_logits = target_head(verify_hidden)
        target_tokens = ids(mx.argmax(verify_logits, axis=-1))
        eval_cache(mx, verify_cache, verify_hidden, profile, "verify")
        _profile_stop(profile, "target_verification")
        verify_seconds += time.perf_counter() - verify_started
        if profile is not None:
            profile.cache_checkpoint("after_verification", verify_cache)

        accepted = 0
        _profile_start(profile, "token_compare_acceptance")
        while accepted < len(draft_list) and target_tokens[accepted] == draft_list[accepted]:
            accepted += 1
        next_bonus = target_tokens[accepted]
        _profile_stop(
            profile,
            "token_compare_acceptance",
            accepted=accepted,
            proposed=len(draft_list),
        )
        accepted_total += accepted
        rejected = len(draft_list) - accepted
        rejected_total += rejected
        fallback_total += 1
        rounds += 1
        round_trace.append(
            {
                "round": rounds,
                "bonus": bonus,
                "draft_tokens": draft_list,
                "target_tokens": target_tokens,
                "accepted": accepted,
                "rejected": rejected,
                "next_bonus": next_bonus,
            }
        )

        committed_tokens = draft_list[:accepted] + [next_bonus]
        for token_id in committed_tokens:
            if len(generated) >= max_tokens:
                break
            generated.append(int(token_id))
            if token_id in eos_ids:
                finish_reason = "eos"
                break

        # MTP draft state follows the accepted target hidden, including the
        # final target token used as the next bonus.  It is safe to skip only
        # after the output budget/eos terminates the stream.
        if finish_reason == "eos" or len(generated) >= max_tokens:
            if profile is not None:
                profile.cache_checkpoint("after_commit_or_output_stop", verify_cache)
            break
        draft_update_started = time.perf_counter()
        _profile_start(profile, "draft_state_synchronization")
        draft.accept_verified_tokens(
            verify_hidden,
            draft_tokens,
            accepted,
            [next_bonus],
            None,
            token_dtype=mx.int32,
            greedy=True,
        )
        _profile_start(profile, "synchronize")
        mx.synchronize()
        _profile_stop(profile, "synchronize", phase="draft_state_synchronization")
        _profile_stop(profile, "draft_state_synchronization")
        draft_update_seconds += time.perf_counter() - draft_update_started

        hidden_for_bonus = verify_hidden[:, accepted : accepted + 1, :]
        if rejected:
            # The target verification cache contains bonus plus all drafts.
            # Recompute only bonus plus the accepted draft prefix from the
            # untouched pre-round cache.  ``verify_cache`` is a deep copy, so
            # ``before_verify`` itself is already the rollback snapshot. This
            # avoids a second full cache copy while restoring both KV and
            # recurrent state; trimming only KV would be incorrect for Qwen3.8.
            replay_started = time.perf_counter()
            _profile_start(profile, "rollback_replay")
            replay_cache = before_verify
            del verify_cache
            replay_input = mx.array([[bonus] + draft_list[:accepted]], dtype=mx.int32)
            replay_hidden = target_forward_sequential(
                mx, target_inner, replay_input, replay_cache, profile, "replay"
            )
            eval_cache(mx, replay_cache, replay_hidden, profile, "replay")
            target_cache = replay_cache
            rollback_checks += 1
            replay_next = scalar(mx.argmax(target_head(replay_hidden[:, -1:, :]), axis=-1).reshape(-1)[0])
            replay_seconds += time.perf_counter() - replay_started
            _profile_stop(
                profile,
                "rollback_replay",
                accepted=accepted,
                replay_tokens=int(replay_input.shape[1]),
            )
            if replay_next == next_bonus:
                rollback_exact += 1
            else:
                rollback_failures.append(
                    {
                        "round": rounds,
                        "accepted": accepted,
                        "draft_tokens": draft_list,
                        "target_tokens": target_tokens,
                        "replay_next": replay_next,
                        "expected_next": next_bonus,
                    }
                )
        else:
            target_cache = verify_cache
        if profile is not None:
            profile.cache_checkpoint("after_rollback_or_commit", target_cache)

        bonus = next_bonus

    decode_seconds = time.perf_counter() - decode_started
    return {
        "token_ids": generated[:max_tokens],
        "token_hash": exact_hash(generated[:max_tokens]),
        "finish_reason": finish_reason,
        "text": tokenizer.decode(generated[:max_tokens], skip_special_tokens=True),
        "proposed_tokens": proposed_total,
        "accepted_tokens": accepted_total,
        "rejected_tokens": rejected_total,
        "fallback_tokens": fallback_total,
        "acceptance_rate": accepted_total / proposed_total if proposed_total else None,
        "rounds": rounds,
        "rollback_checks": rollback_checks,
        "rollback_exact": rollback_exact,
        "rollback_failures": rollback_failures,
        "round_trace": round_trace,
        "timing": {
            "total_seconds": time.perf_counter() - mtp_started,
            "ttft_seconds": decode_started - mtp_started,
            "target_prefill_seconds": target_prefill_seconds,
            "first_logits_seconds": first_logits_seconds,
            "draft_prefill_seconds": draft_prefill_seconds,
            "proposal_seconds": proposal_seconds,
            "verify_seconds": verify_seconds,
            "replay_seconds": replay_seconds,
            "draft_update_seconds": draft_update_seconds,
            "decode_seconds": decode_seconds,
            "verification_mode": "batch" if batch_verify else "sequential",
        },
    }


def run(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root)
    started = time.perf_counter()
    result: dict[str, Any] = {
        "experiment_id": args.experiment_id or "P8-MTP-CORRECTNESS-G128-BF16",
        "phase": 8,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": None,
        "model_name": "g128",
        "model_path": str(Path(args.model).resolve()),
        "canonical_environment": "envs/mlx-0.32.1",
        "mlx_lm_revision": PINNED_MLX_LM_REVISION,
        "mtp_enabled": True,
        "mtp_precision": "bf16",
        "mtp_source_path": str(Path(args.source).resolve()),
        "verification_mode": "batch" if args.batch_verify else "sequential",
        "cases": [],
        "exact_match": False,
        "rollback_exact": False,
        "classification": None,
        "exception": None,
        "safety_stop_reason": None,
    }
    try:
        result["git_commit"] = scale_utils.oracle_utils.current_commit()
        mx, load, generate_step, make_prompt_cache, make_sampler, _ = scale_utils.oracle_utils.load_runtime()
        result["os_memory_before"] = scale_utils.oracle_utils.os_snapshot()
        model, tokenizer = load(args.model)
        result["memory_after_model_load"] = scale_utils.oracle_utils.memory_snapshot(mx, "stage8_mtp_after_model_load", started)
        draft, draft_meta = load_mtp_draft(mx, Path(args.source), model, Path(args.mtp_weights).resolve() if args.mtp_weights else None)
        result["mtp"] = draft_meta
        result["memory_after_mtp_load"] = scale_utils.oracle_utils.memory_snapshot(mx, "stage8_mtp_after_mtp_load", started)

        all_exact = True
        all_rollback = True
        for category, prompt_text in PROMPTS:
            canonical = scale_utils.canonical_prompt(tokenizer, [{"role": "user", "content": prompt_text}])
            baseline = baseline_greedy(
                mx, generate_step, make_sampler, make_prompt_cache, model, tokenizer,
                canonical["token_ids"], args.max_tokens,
            )
            mtp = mtp_greedy(
                mx,
                model,
                tokenizer,
                draft,
                canonical["token_ids"],
                args.max_tokens,
                batch_verify=args.batch_verify,
            )
            exact = baseline["token_ids"] == mtp["token_ids"]
            rollback_ok = not mtp["rollback_failures"] and mtp["rollback_checks"] == mtp["rollback_exact"]
            all_exact = all_exact and exact
            all_rollback = all_rollback and rollback_ok
            case = {
                "category": category,
                "prompt_sha256": hashlib.sha256(prompt_text.encode()).hexdigest(),
                "prompt_tokens": len(canonical["token_ids"]),
                "baseline": baseline,
                "mtp": mtp,
                "exact_match": exact,
                "longest_common_prefix": next((i for i, (a, b) in enumerate(zip(baseline["token_ids"], mtp["token_ids"])) if a != b), min(len(baseline["token_ids"]), len(mtp["token_ids"]))),
                "rollback_ok": rollback_ok,
            }
            result["cases"].append(case)
            print(json.dumps({"category": category, "exact_match": exact, "rollback_ok": rollback_ok, "baseline_tokens": len(baseline["token_ids"]), "mtp_tokens": len(mtp["token_ids"]), "proposed": mtp.get("proposed_tokens"), "accepted": mtp.get("accepted_tokens"), "acceptance_rate": mtp.get("acceptance_rate")}, sort_keys=True), flush=True)
            del baseline, mtp
            scale_utils.oracle_utils.safe_clear(mx)

        result["exact_match"] = all_exact
        result["rollback_exact"] = all_rollback
        result["classification"] = "EXACT_GREEDY_MTP_CORRECTNESS_PASS" if all_exact and all_rollback else "EXACT_GREEDY_MTP_CORRECTNESS_FAIL"
        result["os_memory_after"] = scale_utils.oracle_utils.os_snapshot()
    except BaseException as exc:
        result["exception"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        result["classification"] = "MTP_CORRECTNESS_ERROR"
    finally:
        result["total_elapsed_seconds"] = time.perf_counter() - started
        write_json(output_root / "correctness.json", result)
        summary = {
            "experiment_id": result["experiment_id"],
            "classification": result["classification"],
            "exact_match": result["exact_match"],
            "rollback_exact": result["rollback_exact"],
            "case_count": len(result["cases"]),
            "exceptions": int(result["exception"] is not None),
            "total_elapsed_seconds": result["total_elapsed_seconds"],
        }
        write_json(output_root / "summary.json", summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if result["classification"] == "EXACT_GREEDY_MTP_CORRECTNESS_PASS" else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--mtp-weights", default=None, help="Optional standalone quantized MTP model.safetensors")
    parser.add_argument("--batch-verify", action="store_true", help="Use one target forward for each verified proposal block")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
