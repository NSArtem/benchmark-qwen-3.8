# Qwen3.5 native MTP + APC server smoke test

Run date: 2026-08-17

## Environment

- Hardware: MacBook Pro Mac15,6, Apple M3 Pro (11 cores), 18 GB unified memory
- macOS: 15.7.8 (24G824)
- uv: 0.12.5 (Homebrew, aarch64)
- Python: 3.14.7
- MLX: 0.32.0
- mlx-vlm: 0.6.13
- mlx-lm: 0.31.3
- transformers: 5.15.0
- huggingface-hub: 1.27.0
- httpx: 0.28.1

No package upgrade was needed. The installed server exposes `--draft-model`,
`--draft-kind mtp`, and APC through `APC_ENABLED=1`; there is no APC CLI flag in
this version. The server was bound to `127.0.0.1:8099`.

## Models and configuration

- Target: `mlx-community/Qwen3.5-9B-4bit`
- Native MTP drafter: `mlx-community/Qwen3.5-9B-MTP-4bit`
- Native dispatch: `--draft-kind mtp`
- MTP draft block size: not overridden; the checkpoint default was used
- APC: `APC_ENABLED=1`, APC trace enabled, optional disk tier capped at 1 GB
- APC cache path: `qwen35-mtp-apc-smoke/apc-cache/` (ignored by git)
- APC runtime block size: 16, 2048 blocks (server defaults; distinct from MTP draft block size)
- Requests: temperature 0, top-p 1, seed 0, thinking disabled, non-streaming, 192 max tokens
- Both requests used the same `X-APC-Tenant: qwen35-apc-smoke` value
- No no-MTP diagnostic was needed because the MTP+APC run produced a verified cache hit

The exact prior practical coding context was reused as
[`coding-prefix.txt`](coding-prefix.txt). It is 61,240 bytes / 61,236 characters.
After server chat formatting/tokenization, Turn 1 contained 16,781 prompt tokens
and Turn 2 contained 17,005 prompt tokens.

## Results

| Turn        | Prompt tokens | Cached tokens | Cache hit | TTFT | Prefill time | Decode tok/s | Wall time | MTP acceptance |
| ----------- | ------------: | ------------: | --------: | ---: | -----------: | -----------: | --------: | -------------: |
| Turn 1 cold |        16,781 |             0 |    0.00% | 64.829 s | 64.228 s | 22.78 | 73.318 s | 75 / 233 = 32.19% (117 rounds) |
| Turn 2 warm |        17,005 |        16,765 |   98.59% |  1.491 s |  1.389 s | 29.80 |  7.947 s | 102 / 179 = 56.98% (90 rounds) |

Turn 2's 16,765 cached tokens are the large shared coding prefix; only 240
prompt tokens were uncached. The server independently reported one APC lookup
hit, 16,765 matched tokens, and one exact hit in `/v1/cache/stats`. The API
response itself reported `usage.prompt_tokens_details.cached_tokens: 16765`.
This is an actual cache hit, not merely the startup "APC enabled" message.

The API's uncached-prompt timing was 261.27 tok/s cold and 172.83 tok/s warm
(240 uncached tokens). The server's aggregate prefill-rate field was 261.27
tok/s and 12,245.44 tok/s respectively; the timing field is more useful here
because APC makes the warm request's total prompt length mostly cached.

## Derived results and memory

- Prefill time: 64.228 s -> 1.389 s, a 46.25x reduction (97.84% less time).
- TTFT: 64.829 s -> 1.491 s, a 43.48x improvement (63.338 s saved).
- Total request wall time: 73.318 s -> 7.947 s, a 9.23x improvement (65.371 s saved).
- Warm native-MTP decode: 29.80 tok/s, essentially the known prior ~29.85 tok/s
  result (0.15% lower). The large latency improvement is APC prefill reuse, not
  a claimed MTP decode multiplier.
- Server-reported peak MLX memory: 9.365 GB cold and 10.148 GB warm, an observed
  difference of 0.783 GB while the APC-backed continuation was active.
- The lightweight snapshots saw swap used move from 123 MB before Turn 1 to
  729 MB after Turn 2, and free-memory percentage move from 51% to 23%.
  There was no OOM, sustained collapse, or request failure; attribution from
  only two snapshots is limited, so this is recorded as visible but manageable
  memory pressure rather than a controlled APC-only overhead measurement.
- The bounded APC disk tier occupied about 581 MB at the end and was not staged.

## Verdict

1. Yes. APC worked together with native MTP: the API reported 16,765 cached
   tokens on Turn 2 and the server cache stats showed an exact lookup hit.
2. Turn 2 served 16,765 of 17,005 prompt tokens from cache.
3. The reused fraction was 98.59% of the Turn 2 prompt, representing nearly all
   of the shared ~16K coding prefix.
4. TTFT improved from 64.829 s to 1.491 s, or 43.48x.
5. Total request latency improved from 73.318 s to 7.947 s, or 9.23x.
6. Yes. Native-MTP decode remained near the previous 30 tok/s: 29.80 tok/s warm.
7. Memory pressure was visible but not practically catastrophic in this short
   run; the warm peak was 0.783 GB above the cold request and swap rose, without
   OOM or severe paging behavior.
8. Yes. This is practically usable as a local coding-agent backend when turns
   share a long prefix: the second turn avoided almost all prefill while keeping
   native-MTP generation speed.
9. There is no APC/MTP compatibility blocker to resolve before a later DFlash
   comparison. A future production evaluation may separately examine cache
   lifetime and memory policy, but this smoke test is sufficient to move on.

The result cleanly separates the benefits: APC removes repeated long-context
prefill; native MTP preserves roughly 30 tok/s decode. They should not be
combined into a misleading single throughput number.

## Correctness and limitations

Both requests returned HTTP 200 with 192 completion tokens and no server/API
errors. Turn 2 included the exact Turn 1 assistant response: the saved response
content hash and the Turn 2 assistant-message hash are both
`607d7b3dd144d1e2380891ac0a903b007d6ac01073c30bf14a2e4b92154db378`.
The continuation completed coherently, so no obvious speculative-decoding
corruption was observed. This was not a baseline-vs-MTP output identity test;
the two requests intentionally ask different conversational turns.

This is one cold and one warm request, as requested. It does not characterize
variance across trials, cache eviction, concurrency, or long-run memory behavior.

## Reproduction artifacts and commits

- Harness/config commit: `03f5f72` (`benchmark: add Qwen3.5 MTP APC server smoke test`)
- Results commit: pending
- Final report commit: pending

Saved artifacts include [`results.json`](results.json), [`commands.txt`](commands.txt),
[`server.log`](server.log), [`turn1-response.json`](turn1-response.json),
[`turn2-response.json`](turn2-response.json), [`client.py`](client.py), and
[`launch-server.sh`](launch-server.sh). The model/Hugging Face cache, `.venv`,
`.uv-cache`, and APC disk cache are ignored and are not committed.
