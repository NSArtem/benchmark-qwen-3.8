# MLX 0.32.1 Qwen3.8 prefill-OOM localization

## Scope and fixed inputs

Only `envs/mlx-0.32.1` was used. The checkpoint was loaded from the existing
local snapshot, with mlx-lm Git commit
`254d153fdeb6f150edd4fc5a54f9828638481fa8`. No dependency, lockfile, sysctl,
model, correctness, or alignment changes were made.

The exact reused prompt is `snapshots/prompt-0768.txt`:

- SHA256: `e43371c867d970f5001796f970c080914a17c4bd34fcb21a15115619c3333767`
- UTF-8 text length: 3344 bytes/characters
- raw tokenizer tokens: 757
- final chat-formatted tokens: 768
- first 16 token IDs: `[248045, 846, 198, 2, 65214, 12182, 2193, 271, 2523, 513, 14912, 430, 264, 10505, 8057, 4613]`
- last 16 token IDs: `[524, 16525, 264, 491, 28560, 5171, 13, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271]`

## MLX memory API and device

The installed MLX 0.32.1 exposes all required counters/setters:

`get_active_memory`, `get_cache_memory`, `get_peak_memory`,
`reset_peak_memory`, `set_cache_limit`, `set_memory_limit`,
`set_wired_limit`, `clear_cache`, `device_info`, and `synchronize`.

`mx.device_info()`:

| Field | Bytes | GiB |
| --- | ---: | ---: |
| `memory_size` | 19,327,352,832 | 18.000000 |
| `max_recommended_working_set_size` | 12,884,918,272 | 12.000015 |
| `max_buffer_length` | 9,663,676,416 | 9.000000 |

The pinned `mlx_lm.generate.wired_limit()` implementation is in
`envs/mlx-0.32.1/.venv/lib/python3.14/site-packages/mlx_lm/generate.py` at
`wired_limit()` (around lines 221–258). It computes model array bytes, sets
the wired limit to `max_recommended_working_set_size`, then restores the old
limit after synchronizing the generation stream. `stream_generate()` wraps
its `generate_step()` iterator in this context (around line 705).

## Tests

| Test | Path | Wired limit | Result | Failure point |
| --- | --- | --- | --- | --- |
| TEST A `direct-default` | direct `generate_step`, `max_tokens=1`, prefill 2048 | default (`0`) | `PASS_FIRST_TOKEN` | first token ID 2, text `#` |
| TEST B `direct-wired-recommended` | same direct path | 12,884,918,272 bytes | `OOM` | `mx.eval([c.state for c in prompt_cache])` |
| manual `manual-prefill-default` | explicit `make_prompt_cache` and one 767-token prefill | default (`0`) | `PASS_MANUAL_PREFILL` | cache state evaluation succeeds |
| manual `manual-prefill-wired-recommended` | same explicit prefill | 12,884,918,272 bytes | `OOM` | `mx.eval([entry.state for entry in prompt_cache])` |

The wired-limit setter returned `0` as the previous value in both wired tests.
No cache-limit or prefill-step-size variants were run because the wired-limit
gate was decisive.

## Memory around the decisive operation

Values are MLX counters; `active+cache` is the sum requested for this test.

| Test/checkpoint | Active bytes (GiB) | Cache bytes (GiB) | Active+cache bytes (GiB) | Peak bytes (GiB) |
| --- | ---: | ---: | ---: | ---: |
| direct default, before `generate_step` | 11,771,725,832 (10.963274) | 10,564 (0.000010) | 11,771,736,396 (10.963284) | 11,771,725,832 (10.963274) |
| direct default, after first token | 12,033,387,276 (11.206965) | 57,651,644 (0.053692) | 12,091,038,920 (11.260657) | 13,379,395,664 (12.460533) |
| wired direct, before `generate_step` | 11,771,725,832 (10.963274) | 10,564 (0.000010) | 11,771,736,396 (10.963284) | 11,771,725,832 (10.963274) |
| wired direct, OOM snapshot | 12,249,270,920 (11.408023) | 590,878,782 (0.550299) | 12,840,149,702 (11.958321) | 12,840,149,702 (11.958321) |
| manual default, immediately before cache-state eval | 11,771,726,950 (10.963275) | 10,052 (0.000009) | 11,771,737,002 (10.963284) | 11,771,726,950 (10.963275) |
| manual default, after cache-state eval | 11,991,926,798 (11.168352) | 1,298,874,430 (1.209671) | 13,290,801,228 (12.378023) | 13,379,395,668 (12.460533) |
| manual wired, immediately before cache-state eval | 11,771,726,950 (10.963275) | 10,052 (0.000009) | 11,771,737,002 (10.963284) | 11,771,726,950 (10.963275) |
| manual wired, OOM snapshot | 12,236,802,666 (11.396411) | 617,551,934 (0.575140) | 12,854,354,600 (11.971551) | 12,854,354,600 (11.971551) |

The explicit prompt cache contains 64 entries before evaluation: 48
`ArraysCache` entries and 16 `KVCache` entries. Their exposed `nbytes` values
are zero before the prefill evaluation; the recurrent/cache state is lazy.

## Answers

1. Direct `generate_step` with default wired state passes at 768 tokens.
2. The same call OOMs when the wired limit is set to the recommended working
   set size.
3. Immediately before the failing evaluation, active memory was
   11,771,725,832 bytes, cache was 10,564 bytes, and active+cache was
   11,771,736,396 bytes. The post-failure snapshot reached
   12,840,149,702 active+cache bytes in the direct test. The manual test’s
   post-failure snapshot was 12,854,354,600 bytes.
4. The cache-limit control was not run because the wired gate was decisive.
   The default-cache baseline passes, so cached buffers alone are not shown to
   be sufficient to cause the OOM.
5. The OOM occurs during `mx.eval` of recurrent/KV cache state, after graph
   construction and cache creation. It occurs before the first decode token.
6. A separate `stream_generate` run was not repeated. The installed source
   shows that it differs by wrapping the `generate_step` iterator in
   `wired_limit()`.
7. A separate explicit-cache `stream_generate` control was not needed. Manual
   pre-creation of the cache reproduces the same default-pass/wired-OOM split.
8. Smaller prefill steps were not tested; the decisive result was obtained at
   the requested default-equivalent prefill step of 2048.
9. The strongest diagnosis is **A: wired-residency handling**, with **D:
   transient allocation during cache-state evaluation** as the mechanism that
   crosses the wired-residency boundary. The model does fit this prompt under
   default wired state, so “model fundamentally does not fit” is not supported.
10. The highest-value next experiment is a minimal stock-vs-bypassed
    `mlx_lm.generate.wired_limit()` control around the same one-token
    `stream_generate` call, with the same memory snapshots. This should be
    done before any correctness/alignment or context-boundary work.

## Conclusion

The MLX 0.32.1 Qwen3.8 prefill OOM is not an unconditional model-fit failure.
The exact 768-token prefill succeeds with default wired state and fails only
when MLX’s recommended 12 GiB wired limit is applied. The failure is localized
to cache-state evaluation (`mx.eval([c.state for c in prompt_cache])`), not
model/cache creation or graph construction. The primary current hypothesis is
therefore a wired-residency/accounting interaction with a transient prefill
allocation.
