# Night-2 status

Last updated: 2026-08-19T09:18:00+04:00, Phase 6 g64 4K/8K safe acceptance complete.

## Current phase

Phase 6 — the upstream review, safe implementation, and g64 ~4K/~8K
acceptance gates are complete. The combined cache diagnosis remains `BOTH`:
invalid incremental framing plus recurrent-cache state drift.

## Last completed experiment

`P0-HISTORICAL-REPLAY`: the preserved old 8K artifact was checked directly.
Turn 1 has 128 tokens; turn 2 has 64 tokens; turn 2 equals turn 1's first 64
tokens exactly. Turn-2 hash and first-64 turn-1 hash are both
`cb7ec89a619202c1a75d78f8b67140100a66b9ec3bfcd36af3ea3be66b0a32ce`.
Longest common prefix and prefix overlap are 64 and 1.0. The old
`P3-WARM8K` `EXCELLENT` correctness conclusion is explicitly superseded and
invalid pending revalidation; its timing/memory observations remain historical
performance data only.

`P0-G128-PRESERVE`: the temporary source-backed g128 checkpoint was moved to
`/Users/artem/models/Qwen3.8-27B-3bit-g128-MLX`. Config is affine 3-bit,
group-size 128; 19 shards and 1,847 tensor keys are intact; header payload is
10,930,715,648 bytes (10.180022 GiB), exactly equal to the index metadata.

`P2-HARNESS`: `research/night2/harness/protocol_cache_harness.py` is committed
as `93f8048`. It records raw token IDs/text, exact hashes, prefix and n-gram
replay metrics, semantic keyword support checks, prompt join tokens, MLX
active/cache/peak memory, OS snapshots, and compact hybrid-cache layer
signatures at creation, prompt processing, first generated token, and turn
completion. Its orchestrator runs Mode A persistently and Mode B/C in fresh
child processes per turn.

`P3-G64-4K-rerun`: the corrected persistent session completed 128/64/64
tokens. All three outputs were distinct and locally relevant; no replay was
detected. Mode-A timings were 56.982 s initial TTFT, then 0.822 and 0.819 s
follow-up TTFT, with 10.903/10.999/10.984 tok/s decode. Peak was 14.142 GiB.
The compact cache signatures showed 48 `ArraysCache` plus 16 `KVCache` at all
checkpoints, with uniform KV offsets 4097 → 4224 → 4256 → 4319 → 4351 → 4414
across the initial and two follow-up turns. This is a Mode-A pass, not yet an
oracle-backed correctness result.

The first corrected launch's only failure was the recorded parent/child plan
handoff bug; it stopped before model load and is not inference evidence.

`P3-G64-6K`: a fresh 6,144-token Mode-A session also completed 128/64/64
tokens with no replay. Peak was 14.455 GiB; follow-up TTFTs were 0.845 and
0.863 s; decode was 10.475 and 10.592 tok/s. All cache layers remained the
expected 48 `ArraysCache` + 16 `KVCache`, and uniform KV offsets advanced
6145 → 6272 → 6367 → 6462. This remains local Mode-A evidence pending B/C.

`P3-G64-7K`: a fresh 7,168-token Mode-A session completed 128/64/64 tokens
with no replay. Peak was 14.456 GiB; follow-up TTFTs were 0.852 and 0.884 s;
decode was 10.515 and 10.514 tok/s. Uniform cache offsets advanced 7169 →
7296 → 7391 → 7486. The failure has not appeared through ~7K under the new
strong follow-up protocol.

`P3-G64-8K`: the strong two-follow-up session completed coherently, but its
follow-ups naturally stopped at 43 and 53 tokens; this is not evidence that
the cache is safe because it does not exercise the historical 64-token replay.

`P3-G64-8K-old-followups-repeat`: using the historical initial prompt,
historical three follow-ups, 128/64/64/64 budgets, and the same pinned direct
runtime reproduced every old generated-token hash exactly. Turn 2 is the first
64 tokens of turn 1 (`cb7ec89a...`), with 1.0 prefix overlap and authoritative
`BROKEN` classification. KV offsets advanced 8192 → 8319 → 8353 → 8416, so no
reset or obvious wrap was observed. The strong-prompt non-reproduction is
therefore a prompt/output-condition difference, not a correctness repair.

`P3-G64-4K-oracles`: turn 1 is exactly `A=B=C`. At turns 2 and 3, Mode A
differs from the exact-token fresh oracle, while Mode B also differs from the
full-transcript fresh oracle. At the same turn-2 offset, all 48 recurrent
`ArraysCache` layer signatures differ between A and B, while all 16 attention
`KVCache` layer signatures match. This localizes the cache-state component to
the recurrent cache path; the B/C difference separately demonstrates that the
incremental token framing is not equivalent to the full chat transcript.
The aggregate classification is `BOTH` for turns 2–3.

`P3-G64-8K-oracles-rerun`: the historical failing session's turn 1 is exactly
`A=B=C`. At turn 2, A and exact-token B both emit the stale first-64 replay,
while full-transcript C emits a distinct coherent response; this is
`INCREMENTAL_PROTOCOL_BUG` by the task rule. A/B cache signatures already
differ in all 48 recurrent layers at the same offset, although the output is
still identical. At turn 3 all 48 recurrent and all 16 KV signatures differ
while the output hash still matches; at turn 4 the output diverges too and the
relation is `BOTH`.

`P5-UPSTREAM`: official mlx-lm issue/PR history was inspected and recorded in
`research/night2/upstream.md`. Upstream issue #980 and PR #1006 describe the
non-trimmable hybrid-cache problem and a server-side message-boundary
checkpoint approach. PR #1006 was closed as redundant with merged PR #1072
(commit `3f9d179`), which also contains later Qwen3.5 batched conv-state and
masked gated-delta fixes. None is a direct, tested fix for this project's
single-sequence mutable-cache protocol; the pinned recurrent model/cache
structure remains materially relevant.

`P6-SAFE-IMPLEMENT`: `research/runtime/qwen38_runtime.py` now provides
`NoWiredTranscriptChat`. It renders the complete logical system/user/assistant
transcript on every turn and creates a fresh hybrid cache for that turn. The
CLI exposes `chat-safe`, and the existing `chat` command is a safe alias. The
old persistent `NoWiredChat` class remains available only for controlled
experiments and is explicitly documented as untrusted.

`P6-G64-SAFE-4K`: the safe session completed the initial 4,096-token prompt
plus three historical follow-ups (128/64/64/64 tokens). All four outputs were
coherent and distinct. Fresh Mode-B exact-token and Mode-C full-transcript
children were run against the exact saved safe outputs. Safe A equals fresh C
at all four turns; safe A intentionally differs from the invalid exact
fragment B at turns 2–4. The per-turn safe TTFTs were 54.974, 54.271, 55.106,
and 57.442 s; decode was 10.969, 11.075, 10.977, and 10.911 tok/s; peak was
14.142–14.143 GiB and active+cache was 11.433–11.463 GiB. The aggregate
oracle classifier is `BOTH` only because it reports the expected divergence
between the safe full transcript and the deliberately fragment-based B; the
safe-vs-C correctness gate is `PASS` for every turn.

The first safe 4K artifact (`P6-G64-safe-4K-rerun4/safe.json`) retained an
invalid comparison to an earlier two-follow-up oracle and therefore reports
`UNRESOLVED` internally. It is preserved; the matching four-follow-up
oracle artifact is authoritative for this gate.

`P6-G64-SAFE-8K-REPEAT1`: the safe session completed the historical failing
8,191-token prompt plus three follow-ups (128/64/64/64). All four outputs were
coherent and non-replaying. Fresh Mode-B and Mode-C children were run against
the exact saved safe outputs. Safe A equals full-transcript C at all turns;
exact-token B differs from safe A at turns 2–4 and has the historical stale
turn-2 hash `cb7ec89a...`. TTFT was 106.428/109.292/111.047/116.938 s;
decode was 10.626/10.577/10.626/10.650 tok/s; peak was 14.766–14.768 GiB;
active+cache was 11.683–11.714 GiB. No safety stop occurred.

`P6-G64-SAFE-8K-REPEAT2`: an independent fresh safe process completed the
same four turns coherently with no safety stop. Its prompt hashes and all four
generated-token hashes exactly matched Repeat 1. TTFT was
108.010/109.942/111.844/112.280 s; decode was 10.588/10.638/10.682/10.710
tok/s; peak and active+cache matched the same 14.766–14.768 and
11.683–11.714 GiB envelopes.

## Current diagnosis of the cache bug

The old 8K replay is now classified as an incremental-protocol failure, not a
low-TTFT success. The existing `NoWiredChat` passes a freshly templated
one-user fragment into one persistent hybrid `prompt_cache`; exact-token replay
shows that this stream is internally reproducible at the stale turn, but it is
not equivalent to the full chat transcript. Separate layer signatures show
the recurrent cache is also not generally equivalent to full-stream replay.

The canonical project `.venv` is currently only a bare virtualenv in this
sandbox. The exact pinned packages remain in the existing uv archive cache; the
heavy commands will use the project Python via the canonical `uv run ...
--no-sync` prefix with explicit read-only `PYTHONPATH`/MLX 0.32.1 Metal library
paths. This does not alter project dependency files or install a new runtime.

## Current trusted chat mode

`chat-safe` is trusted for the validated g64 ~4K and ~8K protocols. It performs true
full-transcript re-prefill and does not retain recurrent or KV state between
turns. Persistent incremental reuse remains untrusted; no `chat-fast` mode is
published. Two independent ~8K safe repeats are complete.

## Persistent g128 path

`/Users/artem/models/Qwen3.8-27B-3bit-g128-MLX` (untracked model weights; do not
commit). It is independent of `/private/tmp` and has passed header/index
integrity verification.

## Best validated warm context

g64 ~8K multi-turn is correctness-safe under the full-transcript path. This is
not a warm-cache TTFT result: follow-ups re-prefill the complete transcript in
about 109–117 s at ~10.6 tok/s decode. The ~4K safe follow-ups take about
54–57 s. Historical persistent-cache warm timings remain invalid for
correctness.

## Best cold context

Source-backed g128: 16,727 actual prompt tokens, coherent, 15.236 GiB peak in
the prior saved fresh-process probe. This is historical cold evidence, not a
permission to probe larger contexts.

## Safety state

No Night-2 inference workload is running. No wired-memory setting has been
changed. `ps` is unavailable in this sandbox, so process absence cannot be
independently established; all heavy launches will remain serialized. The
g128 checkpoint is preserved outside temporary storage.

## Next action

Re-read AGENTS.md, TASK.md, and this STATUS.md, then enter Phase 7: validate
the persistent g128 checkpoint with the same safe chat protocol at ~4K, ~8K,
and ~12K. Only consider ~16K if those gates remain coherent and memory-safe.
