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

## Classification convention

Every important result must retain raw output and token IDs. A grammatical
answer that exactly replays a prior answer to a different query is `BROKEN`.
Oracle relationships are recorded separately from superficial text labels.
