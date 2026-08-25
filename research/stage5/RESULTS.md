# Stage 5 Results

## Phase 0 — recovery

### Measured facts

The starting worktree was clean on `main`. Stage 4 artifacts were recovered
from `research/stage4/`. The local high-precision Qwen3.8-27B source is the
snapshot `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, recorded at 55,562,855,904
bytes across 18 shards. The bounded converter is
`research/harness/convert_qwen38_textonly_g128.py`; the existing validated
runtime is `research/runtime/qwen38_chat.py`.

Entering models are g64, g128, and the mixed34 control. Stage 4 measured V2
objective scores of 23/24, 23/24, and 24/24 respectively. Stage 4 mixed34
persistent runtime measured 10.743 tok/s at 8K-class and 5.719 tok/s at
16K-class, while the earlier g128 normal-KV baseline measured 11.105 and
10.409 tok/s at those approximate contexts.

### Inference

The mixed34 long-context cost is a high-priority reproducibility question, not
yet a localized memory-causality claim. Stage 5 must first interleave fresh
processes and correlate decode with context, active+cache, peak, RSS, and
system pressure.

## Phase 1 — mixed34/g128 decode-cliff localization (in progress)

### Measured facts

The frozen Stage 5 cliff prompt family uses the Stage 4 coding-agent source
(`67dcccb037993b679c5593073bbc1875ad61541827a6fc3bb6720410b8096903`) and
the g128 tokenizer. It produces exact chat lengths 8,191, 10,240, 12,288,
14,336, and 16,727 tokens. The required first fresh 16K control,
`P1A-16K-G128-R1`, completed coherently with 192 generated tokens:

| model | prompt | TTFT | prefill tok/s | decode tok/s | peak | end active+cache | RSS max | finish |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| g128 | 16,727 | 242.118 s | 69.088 | 10.583 | 15.236 GiB | 11.403 GiB | 2.44 GiB | length |

Its raw OS-after snapshot reported 22% free memory and approximately 11.2 GiB
encrypted swap. The next interleaved mixed34 process loaded its 11,451,399,168
byte payload (10.666 GiB active) but was interrupted during prefill before a
first token after the system had accumulated approximately 14.4 GiB swap. It
is classified as a safety stop, not as a Metal OOM or quality result. No second
16K process was started concurrently.

### Safety inference

The requested six-run 16K interleave is not safe to repeat under the observed
system pressure. The gated isolated mixed34 retry completed after recovery and
reproduced the Stage 4 long-context result. A lower-context sweep will locate
the transition without repeatedly reproducing the same high-memory prefill
load. Stage 4's prior five-turn mixed34 data remains controlled supporting
evidence; the measurements establish correlation with context/residency, not
causality by themselves.

The isolated mixed34 retry was:

| model | prompt | TTFT | prefill tok/s | decode tok/s | peak | end active+cache | RSS max | finish |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| mixed34 | 16,727 | 240.786 s | 69.470 | 4.890 | 15.721 GiB | 11.873 GiB | 5.55 GiB | length |

The fresh 8K g128 point was 11.137 tok/s at 13.983 GiB peak. Its following
mixed34 process was intentionally interrupted after model load/setup when the
system fell to 32% free memory and ~13.7 GiB swap; it produced no tokens. The
remaining 10K/12K/14K sweep points were not run after the safety gate engaged.

### Phase 1 cliff table

| evidence | model | actual prompt | decode tok/s | peak GiB | result |
|---|---|---:|---:|---:|---|
| fresh Stage 5 | g128 | 8,191 | 11.137 | 13.983 | coherent |
| Stage 4 controlled | mixed34 | 8,191 | 10.743 mean | 14.468 | 5/5 persistent pass |
| Stage 4 controlled | g128 | 12,288 | 10.803 | 14.609 | prior baseline |
| Stage 4 controlled | g128 | 16,727 | 10.409 | 15.236 | prior baseline |
| fresh Stage 5 | g128 | 16,727 | 10.583 | 15.236 | coherent |
| Stage 4 controlled | mixed34 | 16,727 | 5.719 mean | 15.721 | 5/5 persistent pass |
| fresh Stage 5 | mixed34 | 16,727 | 4.890 | 15.721 | coherent |

### Phase 1 gate conclusion

The mixed34 long-context decode cliff is reproduced by an independent fresh
process: 4.890 tok/s versus 10.583 tok/s for fresh g128 at the same 16,727
tokens, and 5.719 tok/s in the earlier five-turn mixed34 control. The first
observed transition is bounded between the 8,191-token mixed34 result
(10.743 tok/s) and 16,727 tokens (4.890 tok/s); the exact onset between those
points is not measured because repeating the full prefill accumulated 12–14 GiB
swap and repeatedly crossed the system safety concern. The 16,727-token mixed34
peak is 15.721 GiB and its end active+cache is 11.873 GiB. These observations
establish a practical long-context penalty and a residency correlation, not a
causal proof that payload size alone causes the slowdown.
