# Night-2 upstream investigation

Investigation date: 2026-08-19 (Asia/Yerevan)

## Source revisions inspected

- Canonical local source: pinned mlx-lm revision
  `254d153fdeb6f150edd4fc5a54f9828638481fa8`, loaded from the existing uv
  archive cache. The local `qwen3_5.py`, `models/cache.py`, `models/base.py`,
  `models/gated_delta.py`, `generate.py`, and `chat.py` were inspected.
- Current upstream source was inspected from the official `main` branch raw
  files on 2026-08-19:
  [qwen3_5.py](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5.py),
  [models/cache.py](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py),
  [generate.py](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py),
  and [chat.py](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/chat.py).
  The current Qwen3.5 recurrent model structure still uses `ArraysCache` for
  linear/recurrent layers and `KVCache` for attention layers; it does not
  provide a direct correctness fix for this project's one-process
  incremental `stream_generate` protocol.

## Relevant official history

### mlx-lm issue #980 and PR #1006

Official issue [#980](https://github.com/ml-explore/mlx-lm/issues/980) describes
prefix-cache reuse as broken for hybrid models, including Qwen3.5, because
recurrent `ArraysCache` state cannot be trimmed like an attention KV cache.
The issue distinguishes the non-trimmable recurrent state from ordinary
`KVCache` state.

Official PR [#1006](https://github.com/ml-explore/mlx-lm/pull/1006) proposed
checkpointing a non-trimmable cache at the last-user-message boundary using a
sentinel-token comparison, so a later request can reuse a complete prefix
without trimming recurrent state. The branch included commits `f040218`,
`82574b4`, and `c2e01bc`, including a later fix for cache corruption in the
batched checkpoint callback. The PR was closed because the broader batch
refactor in PR #1072 superseded it.

This is relevant architecture evidence, but it is not a drop-in fix for the
canonical runtime: this investigation calls `generate_step` directly with one
mutable cache and does not use the upstream server's prompt-trie checkpoint
path. It also does not establish that the pinned recurrent incremental kernel
is numerically equivalent to a fresh full prefill at every chunk boundary.

### mlx-lm PR #1072

Official PR [#1072](https://github.com/ml-explore/mlx-lm/pull/1072) was merged
as commit `3f9d179` on 2026-04-01. Its description records two Qwen3.5 batch
fixes: extracting convolution state correctly for right-padded batches, and
zeroing masked positions in the gated-delta kernel to avoid uninitialized
outputs. Those fixes target continuous/batched prompt processing. The
canonical Night-2 direct generation path uses unpadded single-sequence calls
with no SSM mask, so this is not evidence that it fixes the observed A/B
divergence or the stale replay.

### Other relevant reports

- [mlx-lm #903](https://github.com/ml-explore/mlx-lm/issues/903) reports that
  Qwen3.5 prompt caching does not produce useful reuse in the server path.
- [mlx-lm #259](https://github.com/ml-explore/mlx-lm/issues/259) reports
  different logits when reusing a prompt cache for a repeated prompt. It is
  older and concerns other model families, but supports treating cached
  logits/state as something requiring an explicit correctness oracle.
- [QwenLM/Qwen3.8 #37](https://github.com/QwenLM/Qwen3.8/issues/37) reports a
  separate Qwen3.5 KV-cache prefill `TypeError` caused by passing an ordinary
  KV cache where the SSM cache mask interface was expected. The canonical
  `make_prompt_cache(model)` path creates the model's hybrid cache and does
  not reproduce that exception.

## Comparison with Night-2 evidence

Night-2's fresh Mode-B exact-token oracle and Mode-C full-transcript oracle
show two layers of failure:

1. At 8K with the historical follow-ups, persistent Mode A equals exact-token
   Mode B and reproduces the stale first-64-token answer, while Mode C is
   different and coherent. This is an invalid incremental chat serialization
   protocol, not a successful warm cache.
2. At 4K, Mode A differs from Mode B even when both represent the same exact
   token stream; all 48 recurrent cache signatures differ while the 16
   attention cache signatures match. This is a separate recurrent-state/chunk
   equivalence problem in the direct persistent path.

The upstream work confirms that hybrid caches require per-cache-type and
message-boundary handling, but no inspected official change proves the
canonical direct persistent protocol safe. Night-2 therefore implements and
validates a full-transcript re-prefill `chat-safe` path and keeps persistent
incremental reuse explicitly untrusted.

