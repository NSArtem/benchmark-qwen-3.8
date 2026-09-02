# Qwen3.8 MLX Stage 8 Final Report

Date: 2026-09-02
Model: `/Users/artem/models/Qwen3.8-27B-3bit-g128-MLX`
Environment: `envs/mlx-0.32.1`, MLX 0.32.1, pinned mlx-lm revision
`254d153fdeb6f150edd4fc5a54f9828638481fa8`

## Executive verdict

### MEASURED FACT

- The HTTP 409 was reproduced at token level and fixed. It was caused by
  reconstructing a cached assistant turn from decoded text through Qwen's
  trimming chat template, not by an invalid conversation.
- The production server now preserves generated token IDs and finish reason,
  reuses exact suffix state when proven, and fresh-prefills valid histories
  when reuse cannot be proven. Lightweight coverage is 14/14; real g128
  recovery and the final long-agent run produced zero HTTP 409s.
- The exact transactional MTP verifier is token- and committed-state-exact at
  short, 4K, and 8K, but is slower than baseline and unsafe to take to 16K on
  this 18 GB machine. It is not a production acceleration.
- The final serialized coding-agent run passed at 10,589 logical tokens over
  29 assistant turns, with three forced truncations and three successful
  recoveries.

### INFERENCE

The correct production default remains greedy g128 with a practical 12K–16K
context. The MTP branch is a useful correctness/memory experiment, not a
general speed mode.

### HYPOTHESIS

A future fused recurrent kernel might make multi-token Qwen3.8 verification
useful, but that would be a new exact-kernel investigation and was not inferred
or implemented in Stage 8.

## 1. Truncated-completion HTTP 409

### Root cause and first mismatches

### MEASURED FACT

The old server treated decoded assistant text as the cache identity. On the
real newline-heavy g128 case, a max-256 completion ended with generated token
`198` (`\\n`). On the next request, the raw canonical template placed
`248046` (`<|im_end|>`) at the same absolute position because the assistant
template applies `|trim`. The first mismatch was:

- absolute token index: `314`;
- generated-stream index: `255`;
- cached token: `198`;
- reconstructed-template token: `248046`.

An independent partial-byte case proved a second failure mode: generated IDs
`[9008, 239, 102, 373]` decoded as `"👩�"`, then re-encoded as
`[9008, 239, 102, 5513]`; the first mismatch was generated index `3`,
absolute index `28`.

The old logic therefore raised `logical transcript is not an exact prefix of
the cache`, and the HTTP handler mislabeled that optimization miss as 409.

The oracle also showed:

- the message history itself was an exact extension;
- cache offset accounting was correct (`prompt + generated IDs`);
- the EOS case continued correctly after special-token text decoding;
- the real 409-class case was the chat-template boundary, with
  decode/re-encode identity still intact.

Raw evidence: `truncation/ROOT_CAUSE.md`, `truncation/repro.json`,
`truncation/real-targeted.json`.

### Fix verdict

The server now stores, per assistant generation, exact generated IDs, decoded
text, finish reason, and prompt-token boundary. During exact session
extension, authoritative IDs replace the rendered assistant-content span and
one canonical assistant boundary is retained. API response text is unchanged.

If the supplied transcript is valid but the exact cache prefix cannot be
proven, the runtime clears the cache and fresh-prefills the full canonical
transcript. It does not return 409 merely because reuse missed.

### Truncation table

| termination/input class | test count | exact repair/reuse | fresh fallback | failures | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| EOS | 1 | 1 | 0 | 0 | pass |
| max-token budgets 1, 2, 3, 31, 32, 63, 64, 127, 128, 255, 256 | 11 | 11 | 0 | 0 | pass |
| partial-byte Unicode | 1 | 1 | 0 | 0 | pass |
| real g128 targeted budgets 31, 128, 256 | 3 | 3 | 0 | 0 | pass |
| real repeated length continuations | 3 | 3 | 0 | 0 | pass |
| valid edited-history recovery | 1 | 0 | 1 | 0 | pass |
| explicit stop sequence | n/a | n/a | n/a | n/a | API does not expose `stop` |
| user cancellation/interruption | n/a | n/a | n/a | n/a | not supported by this API |

The lightweight tokenizer/session suite passed 14/14. The real recovery run
recorded three exact length follow-ups, one fresh fallback, zero conflicts,
and zero HTTP errors. Exact-reuse follow-up TTFT was approximately 0.595–0.969
s in that short real run; the edited-history fresh fallback took 2.614 s.

## 2. Exact MTP profile

### MEASURED FACT

The recovered Stage 7 path uses the source-backed 4-bit MTP sidecar, with 15
tensors, a 225,659,904-byte quantized parameter payload, and an actual maximum
of two proposed draft tokens per round. Diagnostic mode forces evaluation at
component boundaries; its timings are not headline benchmark timings.

### MTP profile table

| context | total | target prefill | target verify | draft proposal | snapshot | rollback/replay | compare/sync/other |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,023 | 22.456 s | 57.42% | 32.06% | 2.46% | 0.64% | 7.11% | 0.31% |
| 4,096 | 67.624 s | 78.07% | 14.43% | 2.12% | 0.54% | 4.70% | 0.14% |
| 8,191 | 126.873 s | 85.68% | 9.73% | 2.40% | 0.67% | 1.48% | 0.04% |

At 8K the run made 30 draft calls and 114 target calls, processed 8,300
target tokens, proposed 60, accepted 34, rejected 26, and replayed 19 target
tokens. Token comparison itself was below 0.001% of wall time. At 8K, removing
the measured snapshot and rollback/replay time arithmetically leaves about
124.137 s; target work and draft work remain, so this is not a speed claim.

### Memory attribution

At 8K, clean same-run peaks were:

- baseline: `15,013,952,566` bytes;
- exact MTP: `16,595,651,862` bytes;
- delta: `1,581,699,296` bytes (`1.473 GiB`);
- MTP weight active-memory increment: `225,661,186` bytes (`0.210 GiB`);
- one full cache snapshot: `707,592,192` bytes (`0.659 GiB`);
- recurrent snapshot: `153,944,064` bytes;
- full-attention KV snapshot: `553,648,128` bytes.

The weights plus one snapshot explain `933,253,378` bytes, 59.0% of the
observed peak delta. The full-history attention copy is therefore a measured
553.65 MB snapshot component, not the whole 1.47 GiB increase. Recurrent state
copy traffic accumulated to `4,618,321,920` bytes over the 8K rounds, but that
is repeated traffic rather than a simultaneous allocation.

### INFERENCE

The large overhead is dominated by target prefill/verification work and
runtime residency, not by token comparison or Python acceptance logic. The
transactional implementation removes full historical KV duplication from its
own transaction, but it still copies recurrent state per round.

### HYPOTHESIS

The unassigned `648,445,918`-byte 8K peak residual is mostly lazy MLX graph or
target/draft intermediate residency. Aggregate MLX counters do not identify
every Metal allocation by tensor owner.

Raw evidence: `mtp/profile_summary.json`, `mtp/profile-8k-reset.json`,
`mtp/PROFILE.md`.

## 3. Sequential-equivalent verifier

### First divergence

### MEASURED FACT

The predetermined target-token oracle used identical committed caches and the
same candidate stream. Width 1 was exact. Width 2 first diverged at:

| field | first difference |
| --- | --- |
| layer | 0 |
| component | `linear_attention` |
| cache type | `ArraysCache` |
| position | 0 |
| hidden max absolute difference | 0.0048828125 |
| recurrent state slot 0 difference | 0.25 |
| recurrent state slot 1 difference | 0.0911827087 |
| width 3 | same first divergence |

The first divergent component was recurrent, not full-attention KV. Short-run
top-1 IDs happened to remain equal; logits and state were not equal. The raw
oracle is `mtp/divergence.json` and the explanation is
`mtp/VERIFIER_DIVERGENCE.md`.

### Implemented transaction

The exact transactional prototype:

1. copies only the 48 recurrent `ArraysCache` arrays;
2. shares historical full-attention KV cache objects;
3. evaluates candidate tokens one at a time in normal target order;
4. retains recurrent references/checkpoints at candidate boundaries;
5. trims shared KV offsets back to the accepted boundary after rejection;
6. commits the recurrent checkpoint corresponding to the accepted prefix;
7. discards rejected suffix checkpoints and synchronizes draft state using the
   existing Stage 7 semantics.

It does not replay the full historical target prefix to commit an accepted
boundary and does not deep-copy the full historical attention KV cache.

### Verifier table

| verifier mode | token exact | committed state exact | target work | verdict |
| --- | --- | --- | --- | --- |
| original batched/chunked | no for width >1 | no | lower call count but divergent | rejected |
| layerwise recurrent-sequential + batched attention | no | no | one short call | rejected at layer 3 `KVCache` |
| transactional sequential recurrent + shared KV | yes at short/4K/8K | yes at short/4K/8K | one target call per candidate token | exact state experiment |

The layerwise variant's first failure was full-attention layer 3 `KVCache`,
with key/value differences `0.28125`/`0.1044921875`; it is not used in any
benchmark. The sequential transaction's independent full-cache state oracle
reported no divergence at short, 4K, or 8K.

### INFERENCE

The transaction is sequential-equivalent and avoids duplicating the full KV
history, but it does not create the compute reduction required for useful
speculative decoding. One full target forward still runs per candidate token,
and draft/transaction work is added.

### HYPOTHESIS

The fused `gated_delta_update` sequence kernel is the likely internal source of
the width>1 divergence, but the oracle proves only the layer/cache boundary,
not a specific Metal instruction or numerical cause.

## 4. MTP headline gate

### MEASURED FACT

Fresh-process comparisons used a 64-token output budget and the exact
transactional verifier:

| prompt | baseline decode | exact MTP decode | ratio / change | MTP acceptance | peak overhead |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 11.652 tok/s | 5.511 tok/s | 0.473x / -52.7% | 29/68 = 42.65% | +280,195,304 B (`0.261 GiB`) |
| 8,191 | 11.115 tok/s | 3.563 tok/s | 0.321x / -67.9% | 34/60 = 56.67% | +1,581,699,390 B (`1.473 GiB`) |
| 16,727 | 10.727 tok/s | not measured | n/a | n/a | not measured |

The 4K and 8K MTP outputs were token-identical to fresh baseline outputs and
their committed states passed the independent sequential oracle. The 16K
baseline completed at 10.727 tok/s with a 16,359,177,270-byte peak. The MTP
attempt was stopped during prefill at 4,100 free `vm_stat` pages (about 64 MiB)
with approximately 4.46 GiB compressor occupancy and 3.67 GiB wired pages.
No 16K MTP exactness, speed, or peak overhead is claimed.

The 8K standalone state-oracle artifact was written with an exact
classification before the Python/MLX interpreter later exited with code 139
during cleanup. The same-run headline measurements were already persisted; no
Metal OOM or GPU timeout occurred. The separate 16K attempt was stopped
deliberately at the documented memory boundary.

### Final MTP classification

`REJECTED` as a production acceleration; `EXPERIMENTAL` as a sequential state/
memory prototype. It is not enabled by the production server or CLI.

## 5. Long serialized coding-agent validation

The principal artifact is `agent/long-session-final2.json`. It ran the actual
serialized OpenAI-compatible server with baseline g128 and deterministic local
fixtures.

### Agent table

| metric | result |
| --- | ---: |
| final/max logical context | 10,589 tokens |
| assistant turns | 29 |
| recognized valid tool requests | 12 |
| explicitly injected tool results | 7 |
| invalid tool calls | 0 |
| missing expected tool requests | 0 |
| forced truncations | 3 |
| successful truncation recoveries | 3 |
| HTTP 409s / HTTP errors | 0 / 0 |
| exact reuse / fresh fallback | 28 / 0 requests |
| median warm TTFT | 1.062 s |
| p95 warm TTFT | 1.966 s |
| maximum warm TTFT | 92.613 s |
| median decode | 11.811 tok/s |
| final task success | true |

The 92.613 s maximum was the long deterministic tool-result prefill; warm
turns remained near one second. The final task included file inspection, an
initial wrong hypothesis, failing-test results and revisions, a long tool
payload, early-constraint references, and a six-line final solution. No tool
result was injected after an invalid request. Earlier negative controls
(`long-session-r2.json` and `long-session-r3.json`) remain as evidence that
unknown paths and malformed tool prefixes were classified rather than given
fabricated results.

The server's cumulative `truncation_continuations` counter was 8 because five
additional assistant turns also ended at their requested length; the deliberate
benchmark checkpoints were three.

The planning/revision turns also deliberately used 128- and 160-token caps;
six of those completed with `finish_reason=length`. The three checkpoint
turns use an 8-token cap (with a 1-token retry only if needed) to guarantee the
recovery path rather than relying on model verbosity.

### INFERENCE

The production transport, exact-suffix reuse, and length-continuation path
survived a real serialized session beyond 8K. The seven explicitly injected
tool results are the runner's delivery count; the raw transcript recognized 12
valid request matches and zero invalid matches.

### HYPOTHESIS

The text-only tool protocol and deterministic fixture aliases are adequate for
this reliability gate, but a provider-native structured-tool adapter would
need a separately specified contract.

## 6. Production commands for today

Use the fixed g128 baseline, greedy mode, and no MTP flags. The commands below
use the repository's read-only canonical environment setup; they do not sync,
upgrade, or alter wired-memory settings.

### Server command

```sh
cd /Users/artem/projects/benchmark-qwen-3.8
source research/stage4/canonical_env.sh
uv run --project envs/mlx-0.32.1 --no-sync python \
  research/runtime/qwen38_agent_server.py \
  --model /Users/artem/models/Qwen3.8-27B-3bit-g128-MLX \
  --host 127.0.0.1 --port 8000 --mode greedy \
  --max-tokens 512 --prefill-step-size 2048
```

### CLI command

```sh
cd /Users/artem/projects/benchmark-qwen-3.8
source research/stage4/canonical_env.sh
uv run --project envs/mlx-0.32.1 --no-sync python \
  research/runtime/qwen38_chat.py \
  --model /Users/artem/models/Qwen3.8-27B-3bit-g128-MLX \
  --mode greedy --max-tokens 512 --prefill-step-size 2048
```

Use normal KV at 12K–16K for routine work. Monitor `/metrics` on the server;
it exposes exact reuse, fresh fallback, conflict, and truncation counters.

## 7. Remaining ideas, not started

These are future ideas only; Stage 9 was not started automatically:

1. Implement and validate a numerically sequential-equivalent fused
   GatedDelta scan for width>1 verification.
2. Add lower-level MLX/Metal allocation attribution and short KV-delta
   storage to attack the unassigned peak residual safely.
3. Specify provider-native tool calls, cancellation, and explicit stop
   sequences, then extend long-session reliability coverage.

## Evidence and commits

Primary raw artifacts:

- truncation: `truncation/ROOT_CAUSE.md`, `truncation/repro.json`,
  `truncation/real-targeted.json`, `truncation/real-recovery-final.json`;
- MTP: `mtp/profile_summary.json`, `mtp/divergence.json`,
  `mtp/transactional-short.json`, `mtp/transactional-4k.json`,
  `mtp/transactional-8k.json`, and the headline JSON files;
- agent: `agent/long-session-final2.json` and its server log;
- machine ledger: `results.jsonl`.

Focused implementation commits already present are:

- `432d792` — `fix: continue sessions after length truncation`;
- `bde91c6` — `benchmark: profile and gate Qwen3.8 MTP verifier`.

The closeout harness, documentation, results, and this report are committed
separately at final handoff.
