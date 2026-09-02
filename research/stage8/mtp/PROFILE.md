# Stage 8 exact-MTP diagnostic profile

This report is diagnostic-only. Each run forced cache evaluation at the
instrumented boundaries, so its wall times are not headline benchmark times.
The raw traces are `profile-short.json`, `profile-4k-reset.json`, and
`profile-8k-reset.json`; the compact machine-readable summary is
`profile_summary.json`.

## Measured facts

The recovered Stage 7 path uses the source-backed 4-bit MTP sidecar with a
225,659,904-byte quantized parameter payload, block size 3, and a maximum of
two proposed draft tokens per round. The target verifier remains sequential
and deep-copies the complete hybrid cache before each round.

| context | total diagnostic wall | target prefill | target verify | draft proposal | snapshot | rollback/replay | proposed / accepted | target calls / tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| short / 1,023 | 22.456 s | 57.42% | 32.06% | 2.46% | 0.64% | 7.11% | 52 / 39 | 97 / 1,118 |
| 4K / 4,096 | 67.624 s | 78.07% | 14.43% | 2.12% | 0.54% | 4.70% | 68 / 29 | 138 / 4,231 |
| 8K / 8,191 | 126.873 s | 85.68% | 9.73% | 2.40% | 0.67% | 1.48% | 60 / 34 | 114 / 8,300 |

Token comparison itself was below 0.001% of wall time in all three runs.
Diagnostic synchronization was also negligible in elapsed time (0.005–0.010
seconds), but the explicit synchronization boundaries are algorithmically
important because recurrent state must be materialized before the next token.

At 8K, the profile counted 30 draft calls and 114 target calls processing
8,300 target tokens for 63 generated decode tokens. It replayed 19 rejected
target tokens. Thus the current exact algorithm does more target work than
baseline decode and cannot gain compute speed from cache copying alone.

## Memory attribution

The 8K clean post-weight peak reset measured 15,013,952,566 bytes for the
baseline control and 16,595,651,862 bytes for exact MTP: a 1,581,699,296-byte
same-run delta. The MTP weights added 225,661,186 bytes to active memory. A
single full-cache snapshot was 707,592,192 bytes:

- 153,944,064 bytes for the 48 recurrent `ArraysCache` state arrays;
- 553,648,128 bytes for the 16 full-attention KV arrays.

Weights plus that snapshot explain 933,253,378 bytes, or 59.0% of the
observed peak delta. The remaining 648,445,918 bytes is not assigned to a
single named component by MLX's aggregate counters and is retained as a
runtime-intermediate/lazy-graph residual, not claimed as a measured tensor
identity.

The corresponding maximum snapshot cache arrays were 237,830,144 bytes at
short context and 439,156,736 bytes at 4K. Same-run peak deltas at those
lengths were 225,661,194 and 280,195,306 bytes respectively; the snapshot
does not necessarily coincide with the process peak because MLX reuses and
releases allocations. The Stage 7 historical 8K comparison (13.98 GiB vs
15.46 GiB) independently reported approximately +1.47 GiB and is consistent
with this new clean profile's +1.58 GB measurement.

## Inference and hypothesis

MEASURED FACT: target prefill and target verification dominate wall time; the
Python acceptance loop, snapshot timing, and draft-state synchronization are
small.

INFERENCE: removing the full historical snapshot and rejected-prefix replay
can reduce memory and some wasted target work, but cannot produce a major
throughput gain while each proposed token still receives a full target
forward.

HYPOTHESIS: the residual 8K peak is mostly lazy MLX graph/intermediate
residency created by the target verification and replay calls. Confirming its
exact tensor ownership would require lower-level Metal allocation tracing,
which is outside the safe Stage 8 runtime scope.

The profiler is opt-in in `run_mtp_correctness.py`; ordinary benchmark calls
do not inherit its explicit diagnostic checkpoints.
