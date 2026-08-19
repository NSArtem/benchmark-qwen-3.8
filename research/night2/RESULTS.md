# Night-2 results ledger

This file summarizes only Night-2 evidence. Raw prior artifacts remain under
their original paths and are not rewritten.

## Phase 0

### P0-HISTORICAL-REPLAY

Measured fact: `research/warm-cache-8k/turn-2.tokens.json` contains 64 token
IDs and is byte-for-byte equal as a token sequence to the first 64 IDs in
`turn-1.tokens.json`. Both compact-token SHA256 values are
`cb7ec89a619202c1a75d78f8b67140100a66b9ec3bfcd36af3ea3be66b0a32ce`.
This invalidates the old 8K `EXCELLENT` correctness label. The old timing and
memory values remain usable only as historical performance observations.

### P0-G128-PRESERVE

Measured fact: the source-backed affine 3-bit/group-128 checkpoint now lives at
`/Users/artem/models/Qwen3.8-27B-3bit-g128-MLX`. It has 1,847 indexed/header
keys, 19 shards, zero missing/extra keys, and 10,930,715,648 bytes of tensor
payload (10.180022 GiB), matching `metadata.total_size`.

## Phase 3

### P3-G64-4K-rerun

Measured fact: the corrected current persistent incremental protocol completed
an initial 4,096-token prompt and two deliberately different 31-token
follow-ups with 128/64/64 generated tokens. All outputs were locally coherent,
distinct, and had no >=90% previous-answer prefix replay. Initial TTFT was
56.982 s; follow-up TTFTs were 0.822 and 0.819 s; decode was 10.903, 10.999,
and 10.984 tok/s. Peak was 14.142 GiB and after-turn active+cache was
11.463/11.433/11.448 GiB.

The cache diagnostic facts were uniform 48 `ArraysCache` plus 16 `KVCache`
layers. KV offsets advanced coherently across turns (4097 after initial prompt,
4224 after the initial 128 generated tokens, then 4256/4319 and 4351/4414 for
the two follow-ups). This is a local Mode-A pass only; exact-token and
full-transcript fresh oracle comparisons remain required.

### P3-G64-6K

Measured fact: a fresh 6,144-token g64 persistent session completed 128/64/64
tokens with no >=90% prefix replay. Initial TTFT was 92.750 s at 66.316
tok/s prompt throughput; follow-up TTFTs were 0.845 and 0.863 s; follow-up
decode was 10.475 and 10.592 tok/s. Peak was 14.455 GiB and after-turn
active+cache was 11.585/11.554/11.570 GiB. Uniform cache offsets advanced
6145 → 6272 → 6367 → 6462 across the session.

### P3-G64-7K

Measured fact: a fresh 7,168-token g64 persistent session completed 128/64/64
tokens with no prefix replay. Initial TTFT was 96.049 s at 74.710 tok/s;
follow-up TTFTs were 0.852 and 0.884 s; follow-up decode was 10.515 and
10.514 tok/s. Peak was 14.456 GiB and after-turn active+cache was
11.647/11.617/11.632 GiB. Uniform cache offsets advanced 7169 → 7296 → 7391 →
7486.

### P3-G64-8K

Measured fact: the new strong two-follow-up 8,191-token Mode-A session was
locally coherent with no replay; its follow-ups ended at 43 and 53 tokens.
Initial TTFT was 113.848 s, follow-up TTFTs were 0.861 and 0.857 s, and peak
was 14.766 GiB. This does not exercise the old 64-token replay condition.

### P3-G64-8K-old-followups-repeat

Measured fact: the exact old four-turn query protocol reproduced the old
artifact's four generated-token hashes exactly. Turn 2 had 64 tokens and hash
`cb7ec89a619202c1a75d78f8b67140100a66b9ec3bfcd36af3ea3be66b0a32ce`, equal to
turn 1's first 64 IDs. It is `BROKEN` despite grammatical text. Mode-A KV
offsets advanced 8192 → 8319 → 8353 → 8416 and did not reset or clamp.
The cause is not localized until the fresh B/C oracles are run.

### P3-G64-4K-oracles

Measured fact: turn 1 generated exactly the same tokens in persistent Mode A,
exact-token fresh Mode B, and full-transcript fresh Mode C. At turns 2–3, A
differed from B and B differed from C. At turn 2 the A/B prompt streams both
represent 4,255 tokens and end at the same KV offset (4,256), but all 48
`ArraysCache` recurrent-layer state signatures differ; the 16 `KVCache`
attention-layer signatures match. This is `BOTH`: a recurrent cache-state
divergence plus a separate incremental/full-transcript framing difference.

### P3-G64-8K-oracles-rerun

Measured fact: the old 8K turn-1 output is exactly `A=B=C`. At turn 2,
persistent A and exact-token B both reproduce the stale first 64 turn-1 tokens,
while full-transcript C is coherent and different, so the old replay is an
`INCREMENTAL_PROTOCOL_BUG`. A/B recurrent cache signatures already differ in
all 48 `ArraysCache` layers; by turn 3 all 48 recurrent and 16 KV signatures
differ, and by turn 4 the generated output also diverges. The combined
session evidence is `BOTH`, not a trusted warm-chat result.

## Classification convention

Every important result must retain raw output and token IDs. A grammatical
answer that exactly replays a prior answer to a different query is `BROKEN`.
Oracle relationships are recorded separately from superficial text labels.

## Phase 5–6 decision

Official upstream history confirms that Qwen3.5/Qwen3.8-style hybrid caches
need message-boundary-aware handling because recurrent `ArraysCache` state is
not trimmable. The relevant server-side checkpoint work was superseded by
merged batch refactoring, but it is not a direct fix for this project's
single-sequence persistent `generate_step` protocol. A safe local fallback was
therefore implemented: `NoWiredTranscriptChat` stores logical messages,
renders the entire transcript each turn, and creates a fresh cache per turn.
The existing `chat` CLI command now aliases this safe mode; the old persistent
class remains only for diagnostics and is not called correct.

### P6-G64-SAFE-4K-RERUN4 + matching fresh oracles

Measured fact: the safe runtime generated four coherent turns for the 4,096
token prompt and historical three-follow-up sequence. Fresh Mode-C
full-transcript children matched all four safe generated token sequences
exactly. Safe-vs-Mode-B differed at turns 2–4, as expected because Mode B is
the exact-token replay of the invalid one-user-fragment protocol. Safe TTFTs
were 54.974/54.271/55.106/57.442 s; decode was
10.969/11.075/10.977/10.911 tok/s; peak was 14.142–14.143 GiB.

The first safe artifact compared against a mismatched earlier two-follow-up
oracle and is retained as a bookkeeping correction, not used as the gate.

### P6-G64-SAFE-8K-REPEAT1 + REPEAT2

Measured fact: both independent safe sessions completed the historical 8,191
token prompt and three follow-ups with four coherent, non-replaying outputs.
Repeat 1 matched fresh full-transcript C exactly at all turns; exact-token B
diverged at turns 2–4, with the known stale replay hash at B turn 2. Repeat 2
matched Repeat 1's prompt and generated-token hashes exactly at all four turns.
Repeat-1 TTFTs were 106.428/109.292/111.047/116.938 s and Repeat-2 TTFTs
108.010/109.942/111.844/112.280 s. Decode was about 10.6 tok/s; peak was
14.766–14.768 GiB and no safety stop occurred.

### P7-G128-SAFE-4K + matching fresh oracles

Measured fact: source-backed affine 3-bit/group-128 g128 completed the same
four-turn safe full-transcript protocol coherently. Prompt tokens were
4,096/4,259/4,360/4,456; TTFT was 53.152/53.704/54.893/56.091 s; decode was
11.831/11.722/11.735/11.731 tok/s; peak was 13.360–13.361 GiB and
active+cache was 10.665–10.742 GiB.

Measured fact: the matching fresh-process full-transcript oracle reproduced
all four safe generated-token hashes exactly. The exact-token fragment oracle
diverged at turns 2–4, as expected for the deliberately invalid incremental
protocol. The authoritative safe-vs-full-transcript gate is `PASS` at every
turn. A first oracle attempt with only the harness default two follow-ups is
retained separately as a bookkeeping correction and is excluded from this
conclusion.

### P7-G128-SAFE-8K-REPEAT1 + matching fresh oracles

Measured fact: the source-backed g128 safe full-transcript path completed the
historical 8,191-token prompt plus three follow-ups with four coherent,
non-replaying outputs. Prompt tokens were 8,191/8,354/8,455/8,551; TTFT was
104.164/106.370/107.998/109.843 s; decode was 11.445/11.422/11.269/11.540
tok/s; peak was 13.983–13.986 GiB and active+cache was 10.914–11.398 GiB.
No safety stop occurred, and the final OS memory snapshot reported 28% free.

Measured fact: matching fresh full-transcript C reproduced all four safe
generated-token hashes exactly. Exact-token fragment B diverged at turns 2–4,
which is the expected comparison against the invalid incremental protocol.
The authoritative safe-vs-full-transcript correctness gate is `PASS` at all
four turns.

### P7-G128-SAFE-16K

Measured fact: the source-backed g128 safe full-transcript path completed the
existing 16,727-token coding prompt plus three follow-ups with four coherent,
non-replaying outputs. Prompt tokens were 16,727/16,890/16,991/17,087; TTFT
was 228.212/227.252/228.502/231.586 s; decode was
10.683/10.842/10.865/10.832 tok/s; peak was 15.236 GiB and active+cache was
11.433–11.447 GiB. No safety stop occurred, and final OS free memory was 25%.
This is a local safe-path result pending the matching fresh-process oracle.

### P7-G128-SAFE-16K + matching fresh oracles

Measured fact: the matching fresh full-transcript C reproduced all four safe
generated-token hashes exactly at the 16K coding prompt. Exact-token fragment B
diverged at turns 2–4, which is the expected comparison against the invalid
incremental protocol. The authoritative safe-vs-full-transcript correctness
gate is `PASS` at every turn. Fresh child peaks stayed below the conservative
15.5 GiB stop, and final memory returned to 91% free.

### P7-G128-SAFE-12K + matching fresh oracles

Measured fact: the source-backed g128 safe full-transcript path completed the
12,288-token prompt plus three follow-ups with four coherent, non-replaying
outputs. Prompt tokens were 12,288/12,451/12,552/12,648; TTFT was
161.970/163.269/165.115/166.413 s; decode was 11.092/11.201/11.261/11.225
tok/s; peak was 14.609–14.611 GiB and active+cache was 11.161–11.285 GiB.
No safety stop occurred, and the final OS snapshot reported 26% free.

Measured fact: matching fresh full-transcript C reproduced all four safe
generated-token hashes exactly. Exact-token fragment B diverged at turns 2–4,
which is the expected comparison against the invalid incremental protocol.
The authoritative safe-vs-full-transcript correctness gate is `PASS` at all
four turns.
