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

1. **Recover state / ledger** — complete. Historical replay, persistent g128,
   and the Night-2 namespace were verified.
2. **Correct the record** — complete and committed as `8ba5437`.
3. **Protocol-aware harness** — complete and committed as `93f8048`.
4. **Failure boundary** — complete; the corrected evidence classifies the issue
   as `BOTH`.
5. **Cache internals / upstream** — complete; official sources are recorded in
   `upstream.md`.
6. **Safe chat implementation** — complete and committed as `9ed2ca9`;
   `chat-safe` is trusted and persistent fast reuse remains unresolved.
7. **g128 interactive validation** — complete and committed as `bf17b47`;
   oracle-validated at ~4K/~8K/~12K/~16K.
8. **Quality / performance** — complete; quality is material on the measured
   corpus and g128 has a modest repeated 4K/8K speed/memory advantage.
9. **Long-context headroom** — complete for the gated g128+KV8 ~16K probe;
   prefill tuning and larger contexts were correctly skipped.
10. **Commands / report** — complete after the final documentation commit.

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
