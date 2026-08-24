# Stage 3 results ledger

This is the human-readable synthesis for the current run. Raw prior evidence
is preserved under `research/` and `research/night2/`; new machine-readable
records are appended to `results.jsonl`.

## Phase 0 — recovered baseline

Measured facts and source paths are recorded in `STATUS.md`. The current
candidate is source-backed affine 3-bit/group-128 g128. The highest-value
unresolved question is whether exact-suffix persistent cache state is
consistent with a fresh execution using the same token stream and call
boundaries.

Decision gate: do not scale persistent exact-suffix chat until the Phase 1 B-vs-C
comparison passes, or until a documented limitation prevents a stronger claim.

## Phase 1 — same-boundary cache correctness oracle

### Measured fact

The source-backed g128 model was run with direct pinned `generate_step`, greedy
decoding, normal KV, and default MLX wired state. The preserved 4K canonical
conversation had 4,096 initial prompt tokens, 96 turn-1 tokens, a measured
cached prefix of 4,192 tokens, a 38-token canonical exact suffix, and a 4,230
token full turn-2 transcript. The logical accounting equation held in both
segmented paths: `4,192 + 38 = 4,230`.

Path B (fresh segmented) and Path C (true persistent) matched at every saved
boundary: after turn 1, before the suffix, after the suffix plus the first
scheduled generation step, after the first turn-2 yield, and after turn 2.
All 64 layers had equal compact state signatures and offsets, with 48
`ArraysCache` and 16 `KVCache` layers. The B/C first-token logprob vectors had
0.0 maximum and mean absolute difference at both generation stages and 8/8
top-k overlap. Turn-1 and turn-2 token sequences were identical, both outputs
were coherent, and no stale replay was detected. The gate is `PASS` and exact
suffix persistent cache is validated for scaling under the tested boundaries.

### Inference

The old persistent-versus-fresh full-transcript divergence is not evidence of a
persistent-state bug in this 4K protocol. Path A was coherent and non-replaying
but shared only 42 initial generated tokens with B/C, which is consistent with
one-shot versus segmented numerical sensitivity in this hybrid recurrent model.
Phase 2 localization is not justified by the B-vs-C evidence.

### Artifact locations

The full result is `research/stage3/cache-boundary-oracle/results.json`;
complete generated IDs and decoded outputs are under `tokens/` and `outputs/`,
and compact per-layer signatures are under `state_signatures/`. The first
attempt's bfloat16 host-conversion failure is preserved in the same namespace
and in `research/stage3/results.jsonl` as a harness error, not inference data.

## Phase 3 — exact-suffix persistent scaling

The source-backed g128 model passed a five-turn persistent exact-canonical-
suffix conversation at every tested prompt size. All turns had exact cache
offset equations, complete generated token IDs, coherent output, and no
replay. The final logical offsets and peak binary GiB were:

| context | final offset | peak | warm TTFT range | warm decode average |
|---|---:|---:|---:|---:|
| 8K | 8,676 | 13.983 GiB | 0.735–1.098 s | 11.105 tok/s |
| 12K | 12,760 | 14.609 GiB | 0.736–1.121 s | 10.803 tok/s |
| 16K-class | 17,219 | 15.236 GiB | 0.807–1.223 s | 10.409 tok/s |

The corrected 8K turn-2 fresh full-transcript oracle used 8,352 prompt
tokens, was coherent, had exact persistent/oracle output token equality for
29/29 tokens, and measured 116.796 s TTFT and 11.036 tok/s decode. The
corrected 16K turn-5 oracle used 17,155 prompt tokens, was coherent, and
passed semantically; its output shared a 50-token prefix with the persistent
turn. This is retained as segmentation/numerical sensitivity, not replay.

The first parent-run 8K oracle is deliberately retained but excluded from
evidence because a pre-fix transcript builder duplicated the initial user
prompt and measured 16,536 tokens. The harness-only failures and subsequent
EOS-boundary normalization are all recorded in `STATUS.md` and
`results.jsonl`.

Raw artifacts: `research/stage3/persistent-scaling-final/`, with prior failed
attempts preserved in sibling `persistent-scaling/`, `persistent-scaling-retry/`,
and `persistent-scaling-retry2/` directories.

## Phase 4 — long-session endurance

The g128 normal-KV persistent session began from the validated 8K prompt and
completed 16 coherent linked follow-ups plus the initial response. Every
completed turn had exact cache-prefix/suffix equations, no stale replay, and a
valid compact 64-layer state signature. Warm TTFT remained 1.549–1.624 s and
decode remained 10.58–11.17 tok/s through turn 17; logical context grew from
8,191 to 10,439 prompt tokens. Peak memory stayed at 13.983 GiB, RSS stayed
approximately 6.68 GB, and periodic `memory_pressure` returned successfully
without severe pressure; swap use decreased during the run.

The gate stopped at turn 17 because the raw output was only four tokens:
`The smallest worker` followed by EOS. This was not replay or a protocol
accounting failure, but it was incomplete and therefore `BROKEN` under the
quality oracle. The exact first-failing output, token IDs, prompt, suffix,
cache diagnostics, and state signature are preserved under
`research/stage3/endurance-final/`. The 20–30-turn endurance requirement is
therefore **not passed**; no continuation was run.

## Phase 5 — g64 vs g128 diverse quality

Both models were tested in separate fresh processes with the same 38 greedy
prompts and 64-token output cap (shorter caps for selected retrieval cases).
Complete generated token IDs, decoded text, hashes, first-token logprob
summaries, repetition checks, objective checks, and cache signatures are in
`research/stage3/quality-final/`.

| category | cases | g64 pass | g128 pass | delta (g128-g64) |
|---|---:|---:|---:|---:|
| coding | 10 | 2 | 0 | -20.0 pp |
| instruction | 6 | 6 | 6 | 0.0 pp |
| structured | 6 | 6 | 6 | 0.0 pp |
| reasoning | 6 | 4 | 3 | -16.7 pp |
| retrieval | 5 | 3 | 3 | 0.0 pp |
| prose | 5 | 4 | 4 | 0.0 pp |
| **total** | **38** | **25** | **22** | **-7.9 pp** |

The strict objective suite shows a measurable g128 disadvantage concentrated
in coding and reasoning, while instruction, JSON structure, retrieval, and
prose continuity are tied. Raw outputs show that many coding misses are
relevant answers cut off by the 64-token cap, so the result should not be read
as a standalone estimate of unconstrained coding quality. The prior 4K
teacher-forced reference is directionally consistent: mean NLL was 4.1387
nats for g64 and 5.8335 for g128, a +1.6947-nat g128 delta over 4,095
targets; it is supplementary rather than the sole decision criterion.

The g128 classification is **PERFORMANCE/LONG-CONTEXT MODE**, not DEFAULT:
the approximately 0.783-GiB tensor saving and observed decode advantage are
useful, but the quality evidence is not a clean tie. The fresh-process suite
had no inference exceptions; measured peak was 14.456 GiB for g64 and 13.673
GiB for g128.

## Phase 6 — KV8 long-context lever

The same 16K-class prompt family and five exact-suffix follow-ups were run in
separate fresh g128 child processes. The full comparison, per-turn token
artifacts, compact state signatures, and OS snapshots are under
`research/stage3/kv8-final/`.

| mode | turns | cold TTFT | warm TTFT median | warm decode mean | active+cache peak | peak | correctness |
|---|---:|---:|---:|---:|---:|---:|---|
| normal KV | 5 | 224.005 s | 1.169 s | 10.513 tok/s | 11.418 GiB | 15.236 GiB | PASS |
| KV8, group 64, start 0 | 5 | 223.990 s | 1.169 s | 9.522 tok/s | 11.390 GiB | 14.771 GiB | PASS |

KV8 saved 0.465 GiB of peak (3.1% of the normal peak) and imposed a 9.4%
warm-decode penalty. All offsets and suffix equations passed in both modes;
KV8 used 48 `ArraysCache` plus 16 `QuantizedKVCache` layers. The mode is
therefore **SELECTABLE LONG-CONTEXT MODE** when the extra headroom matters,
but it is rejected as the normal small-context default. Raw outputs remained
current-turn relevant and non-replaying; one KV8 recommendation differed
substantially from the normal-KV recommendation, which is a quality-sensitivity
caveat rather than a cache-state failure.

## Phase 7/8 — conservative stop

20K was **not tested**. There is no existing natural 20K prompt family in the
repository, the validated normal-KV 16K-class peak is already 15.236 GiB, and
synthetic padding would not answer the practical coding-agent question cleanly.
The measured KV8 lever is documented without using it to justify a risky
upward probe. Optional decode acceleration was also **not tested** because the
20–30-turn endurance gate stopped at turn 17 and the task makes acceleration a
bonus only after that gate passes.
