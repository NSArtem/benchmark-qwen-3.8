#!/usr/bin/env python3
"""Sequential-equivalent transactional Qwen3.8 MTP verifier.

This reuses the Stage 7 draft architecture and round semantics.  Its target
transaction differs only in ownership: recurrent ArraysCache state is copied
locally, while historical KVCache objects are shared and rolled back with
offset trimming.  The target candidate is still evaluated one token at a time
because the Phase 5 oracle proved the fused recurrent sequence path diverges.
"""

from __future__ import annotations

import time
from typing import Any

from run_mtp_correctness import (
    cache_arrays,
    eval_cache,
    exact_hash,
    ids,
    target_forward_sequential,
    target_prefill_baseline_schedule,
    tokenizer_eos_ids,
)


def _copy_array(mx: Any, value: Any) -> Any:
    if value is None:
        return None
    return mx.contiguous(mx.array(value, dtype=value.dtype))


def copy_recurrent_cache(mx: Any, cache: list[Any]) -> tuple[list[Any], int]:
    """Copy only ArraysCache arrays; retain KV entries by reference."""

    from mlx_lm.models.cache import ArraysCache, KVCache

    copied: list[Any] = []
    copied_bytes = 0
    arrays: list[Any] = []
    for entry in cache:
        if isinstance(entry, ArraysCache):
            out = ArraysCache(len(entry.cache))
            out.cache = []
            for value in entry.cache:
                new_value = _copy_array(mx, value)
                out.cache.append(new_value)
                if new_value is not None:
                    copied_bytes += int(new_value.nbytes)
                    arrays.append(new_value)
            out.left_padding = _copy_array(mx, getattr(entry, "left_padding", None))
            out.lengths = _copy_array(mx, getattr(entry, "lengths", None))
            copied.append(out)
        elif isinstance(entry, KVCache):
            # The target is serialized. KVCache appends only at the current
            # offset and trim() can discard the speculative suffix.
            copied.append(entry)
        else:
            copied.append(entry)
    if arrays:
        mx.eval(arrays)
    return copied, copied_bytes


def recurrent_checkpoint(cache: list[Any]) -> list[Any]:
    """Capture recurrent array references after an evaluated token step."""

    from mlx_lm.models.cache import ArraysCache

    checkpoint: list[Any] = []
    for entry in cache:
        if isinstance(entry, ArraysCache):
            out = ArraysCache(len(entry.cache))
            # Qwen3.8's recurrent implementation assigns new arrays to both
            # slots.  The arrays are evaluated before this function is called,
            # so retaining references avoids another 154 MB copy per token.
            out.cache = list(entry.cache)
            out.left_padding = getattr(entry, "left_padding", None)
            out.lengths = getattr(entry, "lengths", None)
            checkpoint.append(out)
        else:
            checkpoint.append(entry)
    return checkpoint


def _entry_checkpoint(entry: Any) -> Any:
    """Make a cheap checkpoint wrapper for one recurrent cache entry."""

    from mlx_lm.models.cache import ArraysCache

    if not isinstance(entry, ArraysCache):
        return entry
    out = ArraysCache(len(entry.cache))
    out.cache = list(entry.cache)
    out.left_padding = getattr(entry, "left_padding", None)
    out.lengths = getattr(entry, "lengths", None)
    return out


def _single_token_mask(mask: Any, position: int, sequence_length: int) -> Any:
    """Slice a batch-shaped SSM mask for a one-token recurrent update."""

    if mask is None or isinstance(mask, str):
        return mask
    shape = tuple(getattr(mask, "shape", ()))
    if shape and shape[-1] == sequence_length:
        return mask[..., position : position + 1]
    return mask


def layerwise_target_forward(
    mx: Any,
    qwen_module: Any,
    target_inner: Any,
    input_ids: Any,
    cache: list[Any],
) -> tuple[Any, list[list[Any]]]:
    """Run a short target chunk with sequential recurrent layers.

    GatedDelta/ArraysCache layers are invoked with S=1 in candidate order and
    evaluated before the next token reaches that same layer. Full-attention
    layers receive the whole candidate sequence and retain their normal KV
    cache append. The returned checkpoints contain the recurrent state after
    each complete candidate token; KV entries are shared because the caller
    trims their offsets transactionally.
    """

    from mlx_lm.models.cache import ArraysCache

    sequence_length = int(input_ids.shape[1])
    hidden = target_inner.embed_tokens(input_ids)
    fa_mask = qwen_module.create_attention_mask(
        hidden, cache[target_inner.fa_idx]
    )
    ssm_mask = qwen_module.create_ssm_mask(hidden, cache[target_inner.ssm_idx])

    # Keep the recurrent entry produced by each layer at every token boundary.
    # Later calls update a cache entry by assignment, so the MLX array
    # references remain valid after the following token is evaluated.
    recurrent_by_position: list[dict[int, Any]] = [
        {} for _ in range(sequence_length)
    ]
    for layer_index, (layer, layer_cache) in enumerate(
        zip(target_inner.layers, cache)
    ):
        if layer.is_linear:
            parts: list[Any] = []
            for position in range(sequence_length):
                step = layer(
                    hidden[:, position : position + 1, :],
                    mask=_single_token_mask(ssm_mask, position, sequence_length),
                    cache=layer_cache,
                )
                # The next token at this layer must observe the completed
                # recurrent transition. This is intentionally diagnostic and
                # is the same synchronization boundary as the exact oracle.
                eval_cache(mx, cache, step)
                recurrent_by_position[position][layer_index] = _entry_checkpoint(
                    layer_cache
                )
                parts.append(step)
            hidden = mx.concatenate(parts, axis=1)
        else:
            hidden = layer(hidden, mask=fa_mask, cache=layer_cache)
            eval_cache(mx, cache, hidden)

    hidden = target_inner.norm(hidden)
    eval_cache(mx, cache, hidden)
    checkpoints: list[list[Any]] = []
    for position in range(sequence_length):
        checkpoint: list[Any] = []
        for layer_index, layer_cache in enumerate(cache):
            if isinstance(layer_cache, ArraysCache):
                checkpoint.append(recurrent_by_position[position][layer_index])
            else:
                checkpoint.append(layer_cache)
        checkpoints.append(checkpoint)
    return hidden, checkpoints


def cache_nbytes(cache: list[Any]) -> dict[str, int]:
    from mlx_lm.models.cache import ArraysCache, KVCache

    recurrent = 0
    attention = 0
    for entry in cache:
        if isinstance(entry, ArraysCache):
            recurrent += sum(int(value.nbytes) for value in entry.cache if value is not None)
        elif isinstance(entry, KVCache):
            if entry.keys is not None:
                attention += int(entry.keys.nbytes) + int(entry.values.nbytes)
    return {
        "recurrent_bytes": recurrent,
        "attention_bytes": attention,
        "total_bytes": recurrent + attention,
    }


def _kv_offsets(cache: list[Any]) -> list[int | None]:
    return [int(entry.offset) if hasattr(entry, "offset") else None for entry in cache]


def _commit_transaction(
    verify_cache: list[Any],
    recurrent_checkpoints: list[list[Any]],
    before_offsets: list[int | None],
    commit_count: int,
    verify_count: int,
    rejected: int,
) -> list[Any]:
    """Select the accepted recurrent checkpoint and trim shared KV state."""

    commit_count = max(1, min(int(commit_count), int(verify_count)))
    for index, entry in enumerate(verify_cache):
        before = before_offsets[index]
        if before is None or not hasattr(entry, "offset"):
            continue
        desired = before + commit_count
        if int(entry.offset) > desired:
            entry.trim(int(entry.offset) - desired)
        if int(entry.offset) != desired:
            raise RuntimeError(
                f"transaction KV offset layer {index}: {entry.offset} != {desired}"
            )

    if commit_count == verify_count and not rejected:
        return verify_cache

    selected = recurrent_checkpoints[commit_count - 1]
    # The checkpoint contains the recurrent entries and the shared KV entries
    # at their already-trimmed accepted offsets.
    return selected


def transactional_mtp_greedy(
    mx: Any,
    model: Any,
    tokenizer: Any,
    draft: Any,
    prompt_ids: list[int],
    max_tokens: int,
    forward_mode: str = "sequential",
) -> dict[str, Any]:
    """Run exact-candidate MTP with recurrent-only transactions."""

    if forward_mode not in {"sequential", "layerwise"}:
        raise ValueError(f"unsupported transactional forward mode: {forward_mode}")

    started = time.perf_counter()
    language_model = model.language_model
    target_inner = language_model.model
    target_head = language_model.lm_head
    qwen_module = None
    if forward_mode == "layerwise":
        import mlx_lm.models.qwen3_5 as qwen_module
    target_cache = model.make_cache()
    prompt = mx.array(prompt_ids, dtype=mx.int32)[None, :]
    target_prefill_started = time.perf_counter()
    hidden = target_prefill_baseline_schedule(mx, target_inner, prompt, target_cache)
    target_prefill_seconds = time.perf_counter() - target_prefill_started
    first_logits_started = time.perf_counter()
    first_bonus = int(mx.argmax(target_head(hidden[:, -1:, :]), axis=-1).reshape(-1)[0].item())
    first_logits_seconds = time.perf_counter() - first_logits_started
    eos = tokenizer_eos_ids(tokenizer)
    generated = [first_bonus]
    committed_inputs: list[int] = []
    draft.reset(model)
    if first_bonus in eos or max_tokens <= 1:
        target_prefill_calls = (len(prompt_ids) - 1 + 2048 - 1) // 2048 + 1 if prompt_ids else 0
        total_seconds = time.perf_counter() - started
        return {
            "token_ids": generated[:max_tokens],
            "token_hash": exact_hash(generated[:max_tokens]),
            "text": tokenizer.decode(generated[:max_tokens], skip_special_tokens=True),
            "finish_reason": "eos" if first_bonus in eos else "length",
            "proposed_tokens": 0,
            "accepted_tokens": 0,
            "rejected_tokens": 0,
            "rounds": 0,
            "committed_target_inputs": committed_inputs,
            "target_cache": target_cache,
            "target_calls": target_prefill_calls,
            "target_tokens_processed": len(prompt_ids),
            "recurrent_copy_bytes": 0,
            "max_transaction_cache_bytes": cache_nbytes(target_cache),
            "timing": {
                "total_seconds": total_seconds,
                "target_prefill_seconds": target_prefill_seconds,
                "first_logits_seconds": first_logits_seconds,
                "draft_prefill_seconds": 0.0,
                "decode_seconds": 0.0,
            },
        }

    hidden_for_bonus = hidden[:, -1:, :]
    draft_prefill_started = time.perf_counter()
    draft.prefill_from_target_hidden(
        prompt,
        hidden,
        first_bonus,
        None,
        token_dtype=mx.int32,
        greedy=True,
    )
    mx.synchronize()
    draft_prefill_seconds = time.perf_counter() - draft_prefill_started
    draft.set_shared_kv(
        {}, len(prompt_ids), position=max(len(prompt_ids) - 1, 0), kv_valid_len=len(prompt_ids)
    )

    proposed_total = 0
    accepted_total = 0
    rejected_total = 0
    rounds = 0
    target_prefill_calls = (len(prompt_ids) - 1 + 2048 - 1) // 2048 + 1 if prompt_ids else 0
    target_calls = target_prefill_calls
    target_tokens_processed = len(prompt_ids)
    recurrent_copy_bytes = 0
    max_transaction = cache_nbytes(target_cache)
    round_trace: list[dict[str, Any]] = []
    bonus = first_bonus
    finish_reason = "length"
    decode_started = time.perf_counter()

    while len(generated) < max_tokens:
        draft_tokens = draft.draft_block(
            bonus,
            hidden_for_bonus,
            None,
            3,
            None,
            token_dtype=mx.int32,
            greedy=True,
        )
        draft_list = ids(draft_tokens)
        if not draft_list:
            raise RuntimeError("MTP draft produced zero tokens")
        proposed_total += len(draft_list)

        before_offsets = _kv_offsets(target_cache)
        verify_cache, copied_bytes = copy_recurrent_cache(mx, target_cache)
        recurrent_copy_bytes += copied_bytes
        verify_input = mx.concatenate(
            [mx.array([[bonus]], dtype=mx.int32), draft_tokens.astype(mx.int32)],
            axis=1,
        )
        verify_count = int(verify_input.shape[1])
        if forward_mode == "layerwise":
            verify_hidden, recurrent_checkpoints = layerwise_target_forward(
                mx, qwen_module, target_inner, verify_input, verify_cache
            )
            target_calls += 1
            target_tokens_processed += verify_count
        else:
            recurrent_checkpoints = []
            verify_hidden_parts: list[Any] = []
            for index in range(verify_count):
                step_hidden = target_inner(
                    verify_input[:, index : index + 1], cache=verify_cache
                )
                target_calls += 1
                target_tokens_processed += 1
                eval_cache(mx, verify_cache, step_hidden)
                verify_hidden_parts.append(step_hidden)
                recurrent_checkpoints.append(recurrent_checkpoint(verify_cache))
            verify_hidden = mx.concatenate(verify_hidden_parts, axis=1)
        verify_logits = target_head(verify_hidden)
        target_tokens = ids(mx.argmax(verify_logits, axis=-1))
        eval_cache(mx, verify_cache, verify_hidden)

        accepted = 0
        while accepted < len(draft_list) and target_tokens[accepted] == draft_list[accepted]:
            accepted += 1
        next_bonus = target_tokens[accepted]
        accepted_total += accepted
        rejected = len(draft_list) - accepted
        rejected_total += rejected
        rounds += 1

        committed_tokens = draft_list[:accepted] + [next_bonus]
        output_slots = max_tokens - len(generated)
        output_count = min(len(committed_tokens), output_slots)
        generated.extend(int(item) for item in committed_tokens[:output_count])
        if next_bonus in eos and output_count > 0 and generated[-1] == next_bonus:
            finish_reason = "eos"
        commit_count = output_count
        if commit_count <= 0:
            raise RuntimeError("MTP round produced no committable target input")
        target_cache = _commit_transaction(
            verify_cache,
            recurrent_checkpoints,
            before_offsets,
            commit_count,
            verify_count,
            rejected,
        )
        committed_inputs.extend([bonus] + draft_list[: commit_count - 1])
        max_transaction = {
            key: max(max_transaction.get(key, 0), value)
            for key, value in cache_nbytes(target_cache).items()
        }
        round_trace.append(
            {
                "round": rounds,
                "bonus": bonus,
                "draft_tokens": draft_list,
                "target_tokens": target_tokens,
                "accepted": accepted,
                "rejected": rejected,
                "next_bonus": next_bonus,
                "verify_count": verify_count,
                "commit_count": commit_count,
                "cache_offsets_after_commit": _kv_offsets(target_cache),
                "transaction_recurrent_copy_bytes": copied_bytes,
                "transaction_cache_bytes": cache_nbytes(target_cache),
            }
        )

        if finish_reason == "eos" or len(generated) >= max_tokens:
            break

        draft.accept_verified_tokens(
            verify_hidden,
            draft_tokens,
            accepted,
            [next_bonus],
            None,
            token_dtype=mx.int32,
            greedy=True,
        )
        mx.synchronize()
        hidden_for_bonus = verify_hidden[:, accepted : accepted + 1, :]
        bonus = next_bonus

    decode_seconds = time.perf_counter() - decode_started
    total_seconds = time.perf_counter() - started
    return {
        "token_ids": generated[:max_tokens],
        "token_hash": exact_hash(generated[:max_tokens]),
        "text": tokenizer.decode(generated[:max_tokens], skip_special_tokens=True),
        "finish_reason": finish_reason,
        "proposed_tokens": proposed_total,
        "accepted_tokens": accepted_total,
        "rejected_tokens": rejected_total,
        "acceptance_rate": accepted_total / proposed_total if proposed_total else None,
        "rounds": rounds,
        "round_trace": round_trace,
        "committed_target_inputs": committed_inputs,
        "target_cache": target_cache,
        "target_calls": target_calls,
        "target_tokens_processed": target_tokens_processed,
        "forward_mode": forward_mode,
        "recurrent_copy_bytes": recurrent_copy_bytes,
        "max_transaction_cache_bytes": max_transaction,
        "timing": {
            "total_seconds": total_seconds,
            "target_prefill_seconds": target_prefill_seconds,
            "first_logits_seconds": first_logits_seconds,
            "draft_prefill_seconds": draft_prefill_seconds,
            "decode_seconds": decode_seconds,
        },
    }
