# Night-2 status

Last updated: 2026-08-19T07:28:41+04:00, Phase 9 4K/8K performance gate complete.

## Current phase

Phase 10 — gated g128+KV8 long-context headroom is next. Phase 9 controlled
g64/g128 performance repeats and Phase 8 quality comparison are complete. The
combined cache diagnosis remains `BOTH`: invalid incremental framing plus
recurrent-cache state drift.

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

`P7-G128-SAFE-4K`: the persistent source-backed g128 checkpoint completed the
same four-turn safe protocol coherently with no replay or safety stop. Prompt
tokens were 4,096/4,259/4,360/4,456; TTFT was 53.152/53.704/54.893/56.091 s;
decode was 11.831/11.722/11.735/11.731 tok/s; peak was 13.360–13.361 GiB;
active+cache was 10.665–10.742 GiB. This is safe full-transcript re-prefill,
not persistent cache reuse.

`P7-G128-SAFE-4K-ORACLES-RERUN`: the matching fresh-process run used the exact
same three follow-ups as the safe session. Safe A equals full-transcript C at
all four turns, with generated hashes
`cc2075e0...`, `c2d9ed89...`, `765a12425...`, and `41882be4...`. Safe A
differs from exact-token B at turns 2–4 because B intentionally represents the
invalid one-user-fragment protocol. The authoritative safe-vs-C correctness
gate is `PASS` for every turn. A prior oracle directory that accidentally used
the default two follow-ups is preserved as a harness-bookkeeping correction and
is not evidence.

`P7-G128-SAFE-8K-REPEAT1`: the historical 8,191-token prompt plus the same
three follow-ups completed all four turns coherently with no replay or safety
stop. Prompt tokens were 8,191/8,354/8,455/8,551; TTFT was
104.164/106.370/107.998/109.843 s; decode was 11.445/11.422/11.269/11.540
tok/s; peak was 13.983–13.986 GiB and active+cache was 10.914–11.398 GiB.
The final OS snapshot reported 28% free memory; swap was used but no hard
safety condition occurred.

`P7-G128-SAFE-8K-ORACLES-RERUN`: the matching fresh-process run reproduced all
four safe generated-token hashes in full-transcript C. Exact-token B diverged
at turns 2–4, as expected for the invalid fragment protocol. The authoritative
safe-vs-C correctness gate is `PASS` for every turn.

`P7-G128-SAFE-12K`: the 12,288-token prompt plus the same three follow-ups
completed all four turns coherently with no replay or safety stop. Prompt
tokens were 12,288/12,451/12,552/12,648; TTFT was
161.970/163.269/165.115/166.413 s; decode was 11.092/11.201/11.261/11.225
tok/s; peak was 14.609–14.611 GiB and active+cache was 11.161–11.285 GiB.
The final OS snapshot reported 26% free memory; no hard safety condition
occurred.

`P7-G128-SAFE-12K-ORACLES-RERUN`: the matching fresh-process run reproduced
all four safe generated-token hashes in full-transcript C. Exact-token B
diverged at turns 2–4, as expected for the invalid fragment protocol. The
authoritative safe-vs-C correctness gate is `PASS` for every turn.

`P7-G128-SAFE-16K`: the existing validated 16,727-token coding prompt plus
the same three follow-ups completed all four turns coherently with no replay or
safety stop. Prompt tokens were 16,727/16,890/16,991/17,087; TTFT was
228.212/227.252/228.502/231.586 s; decode was 10.683/10.842/10.865/10.832
tok/s; peak was 15.236 GiB and active+cache was 11.433–11.447 GiB. The final
OS snapshot reported 25% free memory and no hard safety condition. This is a
safe Mode-A/full-transcript run but remains pending its matching fresh oracle.

`P7-G128-SAFE-16K-ORACLES-RERUN`: the matching fresh-process run reproduced
all four safe generated-token hashes in full-transcript C. Exact-token B
diverged at turns 2–4, as expected for the invalid fragment protocol. The
authoritative safe-vs-C correctness gate is `PASS` for every turn. Child peak
remained below the conservative 15.5 GiB stop; after completion OS free memory
returned to 91%.

`P9-QUALITY-NLL-FULL4096-CHUNK256`: the same exact 4,096-token coding prompt
was evaluated in separate fresh g64 and g128 processes using 256-token
teacher-forced cache chunks. g64 measured average NLL 4.138706 nats/token,
PPL 62.7216, and target top-1 accuracy 0.557509; peak was 12.161 GiB and
active+cache was 11.968 GiB. g128 measured average NLL 5.833455,
PPL 341.5368, and top-1 accuracy 0.441514; peak was 11.454 GiB and
active+cache was 11.398 GiB. The measured delta is +1.694750 nats/token,
5.4453x PPL, and -0.115995 top-1 accuracy. This is
`MATERIAL_ON_MEASURED_CORPUS` evidence for this coding prompt, not a
universal quality claim; a later code-heavy slice had only a small delta. The
initial uncorrected g64 NLL attempt is preserved as `HARNESS_BUG` because
logits/target lengths were misaligned; it was corrected before the comparison
gate.

`P9-G64/G128-PERF-4K`: two fresh-process repeats per model used the same
4,096-token prompt, greedy decoding, 128 output tokens, prefill step 2048,
and no KV8/speculation. Both models were coherent and deterministic across
repeats. g64 runs measured TTFT 54.514/54.145 s, prompt throughput
75.142/75.649 tok/s, decode 10.882/10.873 tok/s, peak 14.142 GiB, and
active+cache 11.403 GiB. g128 measured TTFT 51.296/51.294 s, prompt
throughput 79.850/79.853 tok/s, decode 11.664/11.623 tok/s, peak 13.360 GiB,
and active+cache 10.685 GiB. This is a positive but bounded 4K performance
result; the matching 8K gate is recorded below.

`P9-G64/G128-PERF-8K`: two fresh-process repeats per model used the same
8,191-token prompt and the same settings. Both models were coherent and
deterministic across repeats. g64 medians were TTFT 109.262 s, prompt 74.967
tok/s, decode 10.412 tok/s, peak 14.766 GiB, active+cache 11.654 GiB. g128
medians were TTFT 105.975 s, prompt 77.293 tok/s, decode 11.275 tok/s, peak
13.983 GiB, active+cache 11.392 GiB. The g128 differences were -3.288 s TTFT,
+0.863 tok/s decode, and -0.783 GiB peak. No safety stop occurred and memory
returned to 91% free.

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

`chat-safe` is trusted for the validated g64 ~4K/~8K and g128 ~4K/~8K/~12K/~16K
protocols. It performs true full-transcript re-prefill and does not retain
recurrent or KV state between turns. Persistent incremental reuse remains
untrusted; no `chat-fast` mode is published. Two independent g64 ~8K safe
repeats are complete.

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

The g128 safe path is oracle-validated through ~16K with matching
full-transcript oracles at ~4K, ~8K, ~12K, and ~16K. The ~16K safe peak was
15.236 GiB; no long-context safety incident occurred.

The largest correctness-safe tested warm context is g128 ~16K. The practical
interactive recommendation remains to be selected after Phase 9 performance
and Phase 10 headroom gates; ~8K is the current conservative default while
~16K is a correctness-safe but high-latency option.

## Best cold context

Source-backed g128: 16,727 actual prompt tokens, coherent, 15.236 GiB peak in
the prior saved fresh-process probe. This is historical cold evidence, not a
permission to probe larger contexts.

## Safety state

No Night-2 inference workload is running. No wired-memory setting has been
changed. `ps` is unavailable in this sandbox, so process absence cannot be
independently established; all heavy launches will remain serialized. The
g128 checkpoint is preserved outside temporary storage. The completed ~16K
oracle and quality comparison returned the system to healthy telemetry; no
inference workload is active.

## Next action

Re-read AGENTS.md, TASK.md, and this STATUS.md, then enter Phase 10A: inspect
the pinned KV8 API/source and run one conservative g128+KV8 ~16K probe only if
the documented safety gate remains satisfied. Do not probe 20K, 24K, or 32K
without the explicit later gates; never change wired-memory settings.
