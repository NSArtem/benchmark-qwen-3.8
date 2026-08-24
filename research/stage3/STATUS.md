# Stage 3 status

Last updated: 2026-08-24 — Stage 3 complete; final report written and awaiting
the final documentation commit.

## Current phase

The required Stage 3 work is complete. Phases 0, 1, 3, 4, 5, and 6 are
complete. Phase 2 was correctly skipped
because the same-boundary B-vs-C cache gate passed. Phase 7 (20K) was
conservatively skipped: no natural 20K prompt family exists, normal 16K is
already near the unattended envelope, and synthetic padding would add little
practical evidence. Phase 8 acceleration is skipped because the endurance
gate did not pass.

## Recovered baseline

| item | measured fact |
|---|---|
| g64 reference | affine 3-bit/group-64; tensor payload 11,771,132,928 bytes = 10.9627218 GiB |
| g128 candidate | source-backed affine 3-bit/group-128; 1,847 tensors / 19 shards; payload 10,930,715,648 bytes = 10.1800222 GiB |
| payload saving | 840,417,280 bytes = 0.7826996 GiB, approximately 7.14% |
| canonical runtime | Python 3.14.7; MLX 0.32.1; mlx-lm 0.31.3 at pinned revision `254d153fdeb6f150edd4fc5a54f9828638481fa8` |
| inference policy | direct `generate_step`; normal KV unless explicitly testing KV8; MLX wired state untouched |

All new memory values use binary GiB. Model weights and the archive-backed
environment are outside the repository and must not be committed.

## Safety state

- Canonical commands use `uv run --project envs/mlx-0.32.1 --no-sync ...`.
- The read-only uv archive launcher is required because the project venv is
  bare; it does not upgrade or mutate the canonical environment.
- No `mx.set_wired_limit()` call has been made, and no system wired-memory
  setting has been changed.
- Heavy MLX/Qwen workloads launched for this run have been serialized.
- Previous failed harness attempts and invalid cache artifacts remain
  preserved in their original namespaces.

## Phase 1 — same-boundary cache oracle

The 4K g128 B-vs-C gate passed. Fresh segmented and true persistent execution
used the same token stream and call boundaries: all 48 `ArraysCache` and 16
`KVCache` layers had equal signatures and offsets at every required boundary.
Turn-1 and turn-2 token IDs matched exactly; first-token logprob max/mean
absolute differences were 0.0 and top-8 overlap was 8/8. The logical equation
`4,192 + 38 = 4,230` held. Both outputs were coherent and non-replaying.

Fresh full-transcript path A was also coherent but shared only 42 generated
tokens with B/C. This is retained as one-shot/segmented numerical sensitivity,
not persistent-cache corruption, so Phase 2 localization was skipped.

## Phase 3 — persistent exact-suffix scaling

The corrected protocol completed five coherent, non-replaying turns at every
tested context. Exact suffix equations and complete token artifacts are under
`research/stage3/persistent-scaling-final/`.

| prompt | final offset | cold TTFT | warm TTFT range | warm decode | peak |
|---|---:|---:|---:|---:|---:|
| 8,191 tokens | 8,676 | 110.729 s | 0.735–1.098 s | 11.105 tok/s | 13.983 GiB |
| 12,288 tokens | 12,760 | 193.462 s | 0.736–1.121 s | 10.803 tok/s | 14.609 GiB |
| 16,727 tokens | 17,219 | 246.012 s | 0.807–1.223 s | 10.409 tok/s | 15.236 GiB |

The corrected 8K fresh semantic oracle used 8,352 prompt tokens, passed, and
was exactly token-equal to the persistent turn (29/29 generated tokens). The
corrected 16K-class oracle used 17,155 tokens, passed semantically, and shared
50 initial output tokens with the persistent turn. The earlier 16,536-token
8K oracle is explicitly invalid because it duplicated the initial user prompt.

## Phase 4 — endurance

The g128 normal-KV persistent session completed 17 sequential turns from an
8K start; turns 1–16 were coherent, non-replaying, exactly suffix-accounted,
and had valid 64-layer signatures. Warm TTFT was 1.549–1.624 s, decode
10.58–11.17 tok/s, context grew to 10,439 tokens, peak was 13.983 GiB, RSS
was approximately 6.68 GB, and periodic memory-pressure checks showed no
severe pressure.

Turn 17 generated only four tokens (`The smallest worker` plus EOS). It was
not replay or cache corruption, but was incomplete and therefore `BROKEN`.
The 20–30-turn endurance gate is `STOPPED_OR_UNRESOLVED`; no continuation was
run. The first failing raw artifact is under
`research/stage3/endurance-final/`.

## Phase 5 — g64 versus g128 quality

The deterministic fresh-process suite completed 38 cases per model across six
categories. Raw outputs, token IDs, hashes, checks, and first-token summaries
are under `research/stage3/quality-final/`; the machine-readable ledger is
`research/stage3/results.jsonl`.

| category | cases | g64 pass | g128 pass | delta |
|---|---:|---:|---:|---:|
| coding | 10 | 2 | 0 | -20.0 pp |
| instruction | 6 | 6 | 6 | 0.0 pp |
| structured | 6 | 6 | 6 | 0.0 pp |
| reasoning | 6 | 4 | 3 | -16.7 pp |
| retrieval | 5 | 3 | 3 | 0.0 pp |
| prose | 5 | 4 | 4 | 0.0 pp |
| **total** | **38** | **25 (65.8%)** | **22 (57.9%)** | **-7.9 pp** |

The strict objective signal is concentrated in coding and reasoning. Several
g128 coding answers were semantically on track but truncated at the 64-token
cap, so this is not a claim that intrinsic coding quality is exactly 20 points
lower. Prior 4K teacher-forced NLL nevertheless points the same direction:
4.1387 nats for g64 versus 5.8335 for g128 over 4,095 targets (+1.6947 nats).
The overall g128 verdict is `PERFORMANCE/LONG-CONTEXT MODE`: useful memory
saving and decode speed, but not a clean quality tie and therefore not
`DEFAULT`.

The suite had no inference exceptions. Its measured peak was 14.456 GiB for
g64 and 13.673 GiB for g128; fresh-case median decode was approximately
11.223 and 12.017 tok/s respectively. These are suite measurements, not the
warm persistent-chat numbers above.

## Phase 6 — persistent KV8 comparison

Normal KV and KV8 were each run in a separate fresh g128 child process using
the same 16K-class prompt family and five exact-suffix turns. Both modes
completed with `PASS`: all turns were coherent, non-replaying, and exactly
suffix-accounted. Normal KV measured 224.005 s cold TTFT, 1.169 s warm median
TTFT, 10.513 tok/s mean warm decode, 11.418 GiB active+cache peak, and
15.236 GiB peak. KV8 measured 223.990 s, 1.169 s, 9.522 tok/s, 11.390 GiB,
and 14.771 GiB respectively.

The measured KV8 saving was 0.465 GiB and the warm decode penalty was 9.4%.
The KV8 state had 48 `ArraysCache` and 16 `QuantizedKVCache` layers at every
saved boundary; offsets were internally consistent. This is
`SELECTABLE_LONG_CONTEXT_MODE`: use it when 16K headroom matters, but keep
normal KV as the default. Raw turn-5 answers differed in their implementation
recommendations while remaining current-turn relevant; this is a quality
sensitivity caveat, not evidence of cache corruption.

## Open decisions

- KV8: persistent correctness is now validated at 16K-class. It saves 0.465
  GiB but costs 9.4% warm decode, so it is a selectable long-context safety
  mode rather than the default.
- 20K: not tested. The natural prompt family ends at 16,727 tokens, normal
  peak is 15.236 GiB, and synthetic padding was not justified.
- Decode acceleration: not tested because Phase 4 stopped at a broken
  early-EOS turn and acceleration is optional only after endurance passes.

## Next action

`research/stage3/FINAL_REPORT.md` now contains the exact runtime
recommendation, measured normal/KV8 tradeoff, quality caveat, endurance stop,
and explicit 20K/acceleration skips. No further heavy workload is authorized
in this stage. Final action is to inspect `git status`/`git log`, commit this
report and status update, and return the required concise summary.
