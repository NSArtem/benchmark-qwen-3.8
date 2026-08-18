# Night-2 plan

Mission: localize the invalid old 8K persistent-cache result, establish a
protocol-aware correctness-safe chat path, and validate the source-backed
affine 3-bit/group-128 model without violating the 18 GB safety envelope.

Operating gates:

- Canonical runtime is `envs/mlx-0.32.1` through `uv run --project ... --no-sync`.
- Default MLX wired state is preserved; no `mx.set_wired_limit` and no system
  wired-memory changes.
- Heavy MLX/Qwen workloads are serialized and use fresh processes for cold
  comparisons.
- A grammatical stale replay is `BROKEN`; token IDs and oracle relations are
  authoritative.
- Do not run 24K automatically and never run 32K tonight.

## Phase state

1. **Recover state / ledger** — in progress. Re-read project policy, verify the
   historical replay, preserve and verify g128, create this namespace.
2. **Correct the record** — create `ERRATA.md` and separately commit the
   superseded 8K conclusion.
3. **Protocol-aware harness** — implement Mode A (current persistent cache),
   Mode B (exact-token fresh replay), Mode C (full-transcript fresh chat),
   token-level overlap/LCP/n-gram metrics, cache diagnostics, and raw outputs.
4. **Failure boundary** — use g64 at approximately 4K/6K/7K/8K, then run B/C
   at a passing and failing point; classify cache vs protocol vs harness.
5. **Cache internals / upstream** — instrument hybrid cache metadata and review
   pinned vs official upstream sources/issues without changing the environment.
6. **Safe chat implementation** — prefer a full-transcript re-prefill fallback
   unless persistent reuse passes the two oracles; expose `chat-safe` and only
   proven `chat-fast`.
7. **g128 interactive validation** — only after a safe path exists; test
   approximately 4K/8K/12K and conditionally 16K with per-turn telemetry.
8. **Quality / performance** — compare g64 and g128 in separate fresh
   processes using a bounded deterministic objective, then controlled 4K/8K
   repeats if time and safety permit.
9. **Long-context headroom** — only after correctness and g128 gates; test
   g128+KV8 at ~16K and, if materially justified, prefill 1024 vs 2048. A
   ~20K probe is conditional; 24K is gated and 32K is forbidden.
10. **Commands / report** — document exact safe commands, create
    `FINAL_REPORT.md`, run final git status/log, and commit small artifacts.

## Decision gates

- Never call old P3-WARM8K `EXCELLENT`; its correctness conclusion is
  superseded pending revalidation.
- Gate A/B/C classification is based on raw token IDs: A differs from B means
  `CACHE_STATE_BUG`; A≈B but both differ semantically from C means
  `INCREMENTAL_PROTOCOL_BUG`; both means `BOTH`; corrected reproduction that
  invalidates the prior setup is `HARNESS_BUG`.
- A safe user-facing mode must respond to every strongly differentiated query,
  pass exact-token and full-transcript comparisons where applicable, and pass
  g64 ~4K and ~8K sessions with independent 8K repeats.
- Any Metal OOM, GPU timeout, severe/repeated pressure, UI instability, or
  repeated allocation failure stops that branch.
