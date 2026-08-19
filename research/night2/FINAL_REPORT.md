# Qwen3.8 Night-2 Final Report

Date: 2026-08-19
Runtime: MLX 0.32.1, pinned mlx-lm revision
`254d153fdeb6f150edd4fc5a54f9828638481fa8`
Machine: Apple M3 Pro, 18 GB unified memory, macOS 15.7.8

## Executive conclusion

The old 8K persistent-cache replay was reproduced exactly. It was not a
successful low-TTFT chat result. The failure involved both an invalid
incremental turn serialization and recurrent hybrid-cache state that did not
match the intended token stream. The safe fix is full-transcript re-prefill
with a fresh hybrid cache for every logical turn. That path is exposed as
`chat-safe`; persistent incremental reuse remains untrusted and no
`chat-fast` mode is published.

The source-backed affine 3-bit/group-128 checkpoint is correctness-safe under
the new full-transcript oracle through an actual 16,727-token initial prompt
(17,087 tokens on the fourth full-transcript turn). The practical default is
g128 normal `chat-safe` at about 8K context. Its safe follow-ups take about
106–110 seconds because they re-prefill the transcript; this is a correctness
tradeoff, not a warm-cache speed result.

g128 was modestly faster and used about 0.783 GiB less peak memory than g64 in
controlled 4K/8K fresh-process repeats. However, teacher-forced quality on the
measured 4,096-token coding prompt was materially worse for g128, so the
quality tradeoff must remain explicit.

At 16K, KV8 reduced g128 peak from 15.236 to 14.771 GiB and active+cache from
11.397 to 10.916 GiB, but reduced decode from 10.641 to 9.495 tok/s. KV8 is a
useful long-context headroom variant, not the normal 4K/8K default.

## Measured facts

### Models and persistence

- g64 baseline: `/Users/artem/.cache/huggingface/hub/models--lukaskremla--Qwen3.8-27B-3bit-MLX-TextOnly/snapshots/c98bba5926f51fec1c8d8737e577221673f524d7`, affine 3-bit/group-64.
- Persistent source-backed g128: `/Users/artem/models/Qwen3.8-27B-3bit-g128-MLX`, affine 3-bit/group-128, 19 shards, 1,847 tensor keys, payload 10,930,715,648 bytes (10.180022 GiB).
- No system wired-memory setting was changed. Canonical inference left MLX wired state untouched and did not call `mx.set_wired_limit`.
- All heavy MLX workloads were launched serially in fresh processes.

### Cache replay and oracle outcomes

The preserved historical 8K artifact has 128 turn-1 tokens and 64 turn-2
tokens. Turn 2 equals the first 64 tokens of turn 1 exactly:

`cb7ec89a619202c1a75d78f8b67140100a66b9ec3bfcd36af3ea3be66b0a32ce`

The longest common prefix is 64 and prefix overlap is 1.0. The grammatical
replay is authoritative `BROKEN`.

The corrected Mode A/B/C results were:

| Case | Mode A vs exact-token B | B vs full-transcript C | Diagnosis |
|---|---|---|---|
| g64 ~4K | materially different at turns 2–3; all 48 recurrent signatures differed while 16 KV signatures matched at turn 2 | materially different | `BOTH` |
| historical g64 ~8K | stale replay matched B at turn 2; later cache signatures and output diverged | B stale replay differed from C | protocol failure at first failure; overall `BOTH` with state drift later |
| safe g64 ~4K/~8K | safe A intentionally differs from invalid fragment B | safe A equals C at every turn | `PASS` safe-vs-C |
| safe g128 ~4K/~8K/~12K/~16K | safe A intentionally differs from invalid fragment B at turns 2–4 | safe A equals C at every turn | `PASS` safe-vs-C |

The old replay was therefore not caused by a simple prompt-length boundary or
KV offset reset. The historical 8K cache offsets advanced normally, but the
recurrent state was not a reliable representation of the logical transcript.

### Cache diagnosis

Measured fact: the old protocol held one persistent `prompt_cache` and passed
only a freshly templated one-user fragment for each follow-up. That fragment
does not serialize the logical `user/assistant/user/...` conversation.

Measured fact: the hybrid model has 48 non-trimmable `ArraysCache` recurrent
layers and 16 attention `KVCache` layers. At the 4K comparison point, the
attention cache signatures matched while all recurrent signatures differed.

Inference: the failure is `BOTH`—an incremental protocol/framing bug plus
recurrent cache-state drift. The old 8K result must not be optimized around.

## Corrections to prior conclusions

The prior `P3-WARM8K` `EXCELLENT` classification is superseded. Its TTFT and
decode measurements remain historical performance observations only; its
correctness conclusion is invalid because turn 2 was exact stale replay.

The old 4K result did not show the exact same replay, but it was not accepted
without the new oracle. g128 warm behavior had not previously been
established; it is now oracle-validated through ~16K under the safe fallback.

Harness construction errors were preserved rather than erased:

- an early parent/child plan handoff stopped before model load;
- one safe oracle attempt used the wrong number of follow-ups;
- the first g64 NLL attempt paired 1,021 logits with 1,022 targets and is
  recorded as `HARNESS_BUG`.

None of these errors explains away the reproduced historical 8K token replay.

## Cache diagnosis and fix

`research/runtime/qwen38_runtime.py` now provides `NoWiredTranscriptChat`.
It stores logical messages, renders the full transcript with the model's
actual chat template every turn, creates a new hybrid cache for that turn, and
uses deterministic greedy generation by default. The `chat` command is a
backward-compatible alias for `chat-safe`.

This is a correctness-preserving fallback, not a persistent-cache speed fix.
The old `NoWiredChat` implementation remains available only for controlled
diagnostics. A persistent `chat-fast` mode was not shipped because it still
fails the required protocol/state equivalence tests.

Official upstream history was inspected in `research/night2/upstream.md`:

- [mlx-lm issue #980](https://github.com/ml-explore/mlx-lm/issues/980)
  discusses the non-trimmable hybrid-cache/message-boundary problem.
- [mlx-lm PR #1006](https://github.com/ml-explore/mlx-lm/pull/1006) explored
  message-boundary checkpoints and was closed as redundant.
- [mlx-lm PR #1072](https://github.com/ml-explore/mlx-lm/pull/1072), merged as
  commit `3f9d179`, contains later Qwen3.5 recurrent/batched fixes.

Inference: upstream contains relevant hybrid-cache/message-boundary work, but
no directly tested single-sequence persistent-cache fix for this pinned
Qwen3.8 protocol was found. The local full-transcript fallback is the trusted
fix for this run.

## g128 interactive results

All safe sessions used the same three strongly differentiated follow-ups and
were compared against fresh Mode B and Mode C children. The following are
actual total prompt tokens per turn, not just newly appended tokens. Every
safe-vs-full-transcript oracle gate passed.

| Initial context | Prompt tokens by turn | TTFT seconds by turn | Decode tok/s by turn | Peak GiB |
|---:|---|---|---|---:|
| ~4K | 4,096 / 4,259 / 4,360 / 4,456 | 53.152 / 53.704 / 54.893 / 56.091 | 11.831 / 11.722 / 11.735 / 11.731 | 13.360–13.361 |
| ~8K | 8,191 / 8,354 / 8,455 / 8,551 | 104.164 / 106.370 / 107.998 / 109.843 | 11.445 / 11.422 / 11.269 / 11.540 | 13.983–13.986 |
| ~12K | 12,288 / 12,451 / 12,552 / 12,648 | 161.970 / 163.269 / 165.115 / 166.413 | 11.092 / 11.201 / 11.261 / 11.225 | 14.609–14.611 |
| ~16K | 16,727 / 16,890 / 16,991 / 17,087 | 228.212 / 227.252 / 228.502 / 231.586 | 10.683 / 10.842 / 10.865 / 10.832 | 15.236 |

Measured fact: g128 passes safe multi-turn chat at 4K, 8K, 12K, and 16K.
The follow-up TTFT is full-transcript prefill time; the historical sub-second
persistent-cache TTFT must not be used here.

g64 also passed the required safe 4K and two independent safe 8K sessions.
Its safe 8K follow-up TTFT was about 109–117 s at about 10.6 tok/s decode,
with 14.766–14.768 GiB peak.

## Quality and performance comparison

### Teacher-forced quality

The same exact token sequence was evaluated in separate fresh processes with
256-token cache chunks. Both models evaluated 4,095 targets from the same
4,096-token prompt; token sequence hash:
`901f877398a2e8855de13fa0757d8951436192bdbe59f2b8a1a88ff48bd8defc`.

| Model | Avg NLL (nats/token) | Perplexity | Target top-1 | Peak GiB | Active+cache GiB |
|---|---:|---:|---:|---:|---:|
| g64 | 4.138706 | 62.7216 | 0.557509 | 12.161 | 11.968 |
| g128 | 5.833455 | 341.5368 | 0.441514 | 11.454 | 11.398 |

Measured fact: g128 is +1.694750 nats/token, 5.4453x PPL, and -0.115995
top-1 on this prompt. This is `MATERIAL_ON_MEASURED_CORPUS` evidence.

Measured fact: a later 1,023-token code-heavy slice at offset 2,048 had only
a +0.033203 NLL delta and -0.004892 top-1 delta. Inference: the quality loss
is content-dependent, and this one coding prompt is not a universal model
quality estimate. A broader held-out task suite is still needed.

### Controlled fresh-process performance

Each point is the median of two deterministic fresh-process runs with greedy
128-token generation, prefill step 2048, no KV8, and no speculation/MTP.

| Prompt | Model | TTFT s | Prompt tok/s | Decode tok/s | Peak GiB | Active+cache GiB |
|---:|---|---:|---:|---:|---:|---:|
| 4,096 | g64 | 54.329 | 75.395 | 10.878 | 14.142 | 11.403 |
| 4,096 | g128 | 51.295 | 79.852 | 11.643 | 13.360 | 10.685 |
| 8,191 | g64 | 109.262 | 74.967 | 10.412 | 14.766 | 11.654 |
| 8,191 | g128 | 105.975 | 77.293 | 11.275 | 13.983 | 11.392 |

Measured fact: g128 is ~3.0–3.3 s lower TTFT, ~0.77–0.86 tok/s faster in
decode, and ~0.783 GiB lower peak in these repeats. Inference: this is a
useful but modest practical advantage, not a reason to ignore the measured
quality delta.

## Long-context safety

### KV8 at 16K

Pinned API/source inspection confirmed that `kv_bits=8`, group size 64,
`quantized_kv_start=0` quantizes the attention KV caches while leaving the
recurrent caches alone. One fresh g128 run on the exact 16,727-token coding
prompt generated 128 coherent tokens:

| Configuration | TTFT s | Prompt tok/s | Decode tok/s | Peak GiB | Active+cache GiB |
|---|---:|---:|---:|---:|---:|
| normal g128 | 219.611 | 76.167 | 10.641 | 15.236 | 11.397 |
| g128 + KV8 | 223.354 | 74.890 | 9.495 | 14.771 | 10.916 |

KV8 saved 0.465 GiB peak and 0.481 GiB active+cache, but added 3.744 s TTFT
and reduced decode by 1.146 tok/s (~10.8%). No Metal OOM, GPU timeout,
severe pressure condition, UI incident, or allocation failure occurred.

KV8 was not run at 4K/8K tonight because it was already rejected for normal
short-context use by the prior saved results, and its 16K safety value was the
only remaining decision question. A 1024-vs-2048 prefill sweep was also
skipped: the 2048 path had no transient allocation failure and both normal and
KV8 16K peaks were below the conservative stop.

### 20K, 24K, and 32K

20K was not tested. The validated natural Lantern prompt family ends at
16,727 actual tokens. Extending it to 20K would require synthetic padding and
would not answer the practical coding-agent question; the measured 16K normal
and KV8 envelopes already establish the useful safety tradeoff without
approaching the unattended danger boundary.

24K was skipped because it is an optional gated probe, not a task requirement,
and the user explicitly prohibited blind 24K testing. The 20K prerequisite
was not established, and no larger-context evidence was needed to choose the
safe default.

32K was not tested because it is explicitly forbidden tonight.

The maximum correctness-safe tested context is g128 normal full-transcript
chat at 16,727 initial prompt tokens and 17,087 total prompt tokens on its
fourth turn. The maximum measured memory-safe one-shot variant is the same
16,727-token workload with g128+KV8 at 14.771 GiB peak.

## Recommended runtime

Use the commands in [`RUNTIME_COMMANDS.md`](RUNTIME_COMMANDS.md).

Recommended configuration:

- model: persistent source-backed g128 path;
- command: `chat-safe`;
- decoding: greedy (`temperature=0`, `top_p=1`);
- normal KV cache, no MTP/speculation, prefill step 2048;
- working context: about 8K by default; use about 4K when 106–110 s
  full-transcript follow-ups are too slow;
- use KV8 only as an explicitly chosen ~16K one-shot headroom variant.

The safe runtime prints the no-wired policy and model metadata. It requires no
`sysctl`, no wired-limit adjustment, and no modification of installed
mlx-lm files.

## Things deliberately not done

- No MLX 0.32.0 runs.
- No system-wide wired-memory changes and no `mx.set_wired_limit` calls.
- No new weight quantizations; mixed_2_6 remains rejected.
- No concurrent heavy MLX workloads.
- No blind 20K/24K probing and no 32K probing.
- No attempt to declare persistent incremental cache reuse correct from low
  TTFT alone.

## Unresolved questions

1. A genuinely correct persistent fast-cache protocol for the 48 recurrent
   non-trimmable layers remains unresolved. The highest-value next experiment
   is a minimal message-boundary/checkpoint implementation modeled on the
   upstream hybrid-cache work, validated against exact-token and full-transcript
   oracles at 4K and 8K.
2. The g128 quality delta needs a broader held-out corpus and objectively
   checkable coding tasks before making a general quality claim.
3. Multi-turn KV8 chat has not been oracle-validated; only one-shot 16K KV8
   output was tested.
4. Process-list access and direct UI observation were unavailable in this
   sandbox, so safety conclusions rely on saved MLX memory and macOS telemetry.

## Commits and artifacts

Relevant logical commits completed during Night 2:

- `8ba5437` — correct the superseded warm-cache conclusion;
- `93f8048` — add the protocol-aware cache oracle;
- `9ed2ca9` — implement correctness-safe multi-turn semantics;
- `4e10a8c` — validate the safe chat oracle;
- `bf17b47` — validate g128 warm context through ~16K;
- `9444b25` — compare g64/g128 teacher-forced quality;
- `dac88e8` — recheck controlled g64/g128 performance;
- final documentation commit contains this report, runtime commands, and the
  Phase-10 ledger/raw artifact.

Raw token IDs, outputs, memory snapshots, oracle comparisons, and machine
readable entries remain under `research/night2/` and are not replaced by this
summary.
