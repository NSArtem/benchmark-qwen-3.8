# Qwen3.8-27B MLX Investigation — Final Report

Date: 2026-08-18  
Machine: Apple M3 Pro, 18 GiB unified memory, macOS 15.7.8  
Canonical runtime: Python 3.14.7, MLX 0.32.1, mlx-lm revision
`254d153fdeb6f150edd4fc5a54f9828638481fa8`  
Baseline model: `lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly`, affine 3-bit,
group size 64

## Executive decision

The best tested configuration for the stated interactive coding-agent goal is
the original affine 3-bit/group-64 model with the local no-wired runtime helper:

```text
MLX 0.32.1
direct generate_step
default MLX wired state; no wired-limit setter
greedy decoding
no speculation, MTP, APC, or activation quantization
persistent prompt cache for chat
```

Use an approximately 8K prompt context for warm interactive conversations. The
4K and 8K persistent-cache profiles were both `EXCELLENT`: follow-up TTFT was
about 0.7–1.2 seconds and decode was about 10.65–11.02 tok/s, with no material
cache growth and coherent responses.

The affine 3-bit/group-128 candidate is the best measured cold-context option:
it was source-converted correctly and remained coherent through an actual
16,727-token fresh-process coding probe at 15.236 GiB MLX peak. It was not
measured as a warm chat session, and its 219.6-second cold TTFT makes it a
maximum cold probe rather than the normal interactive recommendation.

The optional mixed_2_6/group-128 candidate saved more memory but produced an
identical pathological sentence loop in two fresh short runs. It is rejected.
KV8 also preserved correctness but saved too little memory for its decode
penalty and is rejected for normal use.

## 1. Baseline results

The 0.32.1 alignment artifact tested final prompt lengths 512, 513, 640, 641,
768, and 769 in three fresh processes each: 18/18 were coherent, all outputs
were deterministic within each length, and there were no OOMs. This rejects
simple 512/513/640/641/768/769 prefill alignment as the historical corruption
cause.

The fresh cold g64 context ladder is:

| Target | Actual prompt tokens | Result | TTFT (s) | Prompt tok/s | Decode tok/s | Active+cache GiB | Peak GiB | OS free after |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ~1K | 1,023 | COHERENT | 13.352 | 76.615 | 11.112 | 11.389 | 12.860 | 28% |
| ~2K | 2,048 | COHERENT | 25.671 | 79.780 | 11.002 | 11.399 | 13.824 | 26% |
| ~4K | 4,096 | COHERENT | 51.383 | 79.715 | 10.858 | 11.449 | 14.142 | 25% |
| ~8K | 8,191 | COHERENT | 104.330 | 78.511 | 10.415 | 11.634 | 14.766 | 24% |
| ~12K | 12,288 | COHERENT | 160.908 | 76.366 | 7.734 | 12.047 | 15.392 | 24% |

An existing manual real coding prompt also reached 16,767 tokens coherently at
63.348 prompt tok/s, 4.118 decode tok/s, and 17.200 GB reported MLX peak. Its
raw telemetry is not in the checkout, and it is not repeated because that
memory level is not comfortable on an 18 GB machine.

Raw ladder artifacts are under `research/context-ladder/`; the consolidated
ledger is `research/RESULTS.md` and `research/results.jsonl`.

## 2. Cause of the known MLX 0.32.1 OOM

The pinned mlx-lm generation wrapper forces MLX's wired limit to approximately
the device's recommended working-set size, about 12 GiB here. In the saved
controlled test, that policy caused a Metal OOM during
`mx.eval([c.state for c in prompt_cache])` around a 768-token prompt.

The same basic direct prefill with default MLX wired state passed. The failure
is therefore a wired-residency/accounting interaction in the stock wrapper,
not evidence that the checkpoint, model architecture, or simple prefill
alignment is corrupt. No system wired-memory setting was changed during this
investigation.

## 3. Canonical no-wired runtime

`research/runtime/qwen38_runtime.py` is the reusable local helper. It calls the
pinned `generate_step` path directly, leaves MLX's wired state untouched, and
prints:

```text
[qwen38-runtime] mlx-lm wired_limit wrapper disabled; MLX wired state untouched
```

It supports one-shot generation, persistent-cache chat, greedy/temperature
sampling, prefill step size, and the pinned KV8 arguments. The short and
moderate validation runs were coherent after fixing an interface mistake that
initially bypassed the chat template. The helper does not modify installed
mlx-lm files and does not call `mx.set_wired_limit()`.

## 4. Warm / interactive performance

The same persistent chat-cache process was used for one substantial coding
turn followed by three short follow-ups.

| Initial context | Initial TTFT / prompt / decode | Follow-up TTFTs | Follow-up decode | After-turn active+cache | Classification |
| ---: | --- | --- | --- | --- | --- |
| 4,096 | 51.484 s / 79.710 / 10.943 tok/s | 0.726, 1.103, 0.728 s | 11.024, 11.017, 11.005 tok/s | 11.463, 11.433, 11.447, 11.447 GiB | EXCELLENT |
| 8,191 | 104.222 s / 78.667 / 10.547 tok/s | 0.816, 1.202, 0.828 s | 10.664, 10.651, 10.645 tok/s | 11.714, 11.684, 11.697, 11.697 GiB | EXCELLENT |

The initial turn still pays the cold prefill cost; the improvement is in
incremental turns. This is the strongest evidence for the 8K practical
interactive recommendation.

## 5. Quantization size and g128 result

Header-only analysis of the baseline found 1,847 tensors, 498 quantized
modules, and exact current tensor storage of 11,771,132,928 bytes (10.963 GiB).
All observed input widths are divisible by 128.

| Candidate | Effective bpw | Tensor payload | Saving vs g64 |
| --- | ---: | ---: | ---: |
| affine 3-bit g64 | 3.501 | 10.963 GiB | baseline |
| affine 3-bit g128 | 3.251 | 10.180 GiB | 0.783 GiB / 7.14% |
| mixed_2_6 g64 | 3.121 | 9.772 GiB | 1.191 GiB / 10.87% |
| mixed_2_6 g128 | 2.871 | 8.989 GiB | 1.974 GiB / 18.00% |
| affine 2-bit g128 | 2.251 | 7.049 GiB | 3.913 GiB / 35.70% |

The g128 candidate was converted from the appropriate high-precision source,
not from the already-quantized baseline. Source conversion used a bounded,
layer-oriented local converter because the stock full-model converter was not
safe to launch on this machine. The exact source was `Qwen/Qwen3.8-27B`,
revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.

The affine g128 output contains 1,847 tensors in 19 shards with an exact
10,930,715,648-byte payload (10.180022 GiB). Conversion took 24.920 seconds;
maximum converter MLX peak was 3.330 GiB, maximum active+cache was 0.508 GiB,
maximum RSS was 3.436 GiB, and the completion snapshot showed 92% free system
memory. Inference materialized model memory was 10.180602 GiB.

Measured affine g128 correctness and context:

| Prompt | Classification | TTFT (s) | Prompt tok/s | Decode tok/s | Peak GiB |
| ---: | --- | ---: | ---: | ---: | ---: |
| 33 | COHERENT | 1.066 | 30.972 | 11.947 | 10.505 |
| 768 | COHERENT | 10.122 | 75.876 | 11.882 | 11.678 |
| 4,096 | COHERENT | 51.272 | 79.888 | 11.614 | 13.360 |
| 8,191 | COHERENT | 103.864 | 78.863 | 11.204 | 13.983 |
| 12,288 | COHERENT | 158.238 | 77.655 | 10.830 | 14.609 |
| 16,727 | COHERENT | 219.611 | 76.167 | 10.641 | 15.236 |

Thus g128 was successfully converted, loaded, and coherent through the
largest tested context. It was not validated with a persistent warm cache.

## 6. KV8 study

Pinned Qwen3.8 has 48 recurrent `ArraysCache` layers and 16 full-attention
`KVCache` layers. With `kv_bits=8`, group size 64, and start 0, only the 16
full-attention KV caches are quantized.

| Configuration | Context | Peak | Decode | Difference from no-KV8 |
| --- | ---: | ---: | ---: | --- |
| g64 + KV8 | 4,096 | 14.029 GiB | 10.575 tok/s | −0.113 GiB, −2.6% |
| g64 + KV8 | 8,191 | 14.535 GiB | 9.862 tok/s | −0.231 GiB, −5.3% |
| g128 + KV8 | 8,191 | 13.752 GiB | 10.417 tok/s | −0.231 GiB, −7.0% |

All tested KV8 outputs were coherent and TTFT was effectively unchanged, but
the memory savings were too small and decode slowed. KV8 is rejected for the
normal recommendation; no 12K/16K KV8 matrix was run.

## 7. Practical comparison and recommendation

| Configuration | Context evidence | Warm result | Recommendation |
| --- | --- | --- | --- |
| g64 no-wired | fresh 12,288; existing policy-summary 16,767 | 4K/8K EXCELLENT | **Normal interactive winner; use 8K** |
| g64 + KV8 | coherent 4,096/8,191 | not measured | Reject: small saving, slower decode |
| g128 no-wired | coherent fresh 16,727 | not measured | Best cold-context candidate; use for controlled one-shot experiments |
| g128 + KV8 | coherent 8,191 | not measured | Reject: no useful extra saving |
| mixed_2_6 g128 | BROKEN short output in two repeats | not measured | Reject; skip larger tests |

Maximum coherent context has two provenance levels: the largest saved,
fresh-process artifact is g128 at 16,727 actual prompt tokens; the project
also retains the earlier manual g64 policy-summary result at 16,767 tokens.
The recommended practical context is 8K for persistent interactive g64 chat.
For a cold g64 run, 12,288 is the largest fresh point with substantial but
still measured headroom; 17+ GiB states are not routine recommendations.

## 8. Exact runtime commands

Use the baseline model for normal interactive work:

```bash
MODEL=/Users/artem/.cache/huggingface/hub/models--lukaskremla--Qwen3.8-27B-3bit-MLX-TextOnly/snapshots/c98bba5926f51fec1c8d8737e577221673f524d7

uv run --project envs/mlx-0.32.1 --no-sync \
  python research/runtime/qwen38_runtime.py info --model "$MODEL"

uv run --project envs/mlx-0.32.1 --no-sync \
  python research/runtime/qwen38_runtime.py chat \
  --model "$MODEL" --max-tokens 256 --temperature 0 \
  --prefill-step-size 2048
```

Enter one user message per line. `/reset` clears the persistent prompt cache;
`/quit` exits. The helper prints the no-wired status at startup and reports
per-turn timing on stderr.

For one-shot generation with the normal chat template:

```bash
uv run --project envs/mlx-0.32.1 --no-sync \
  python research/runtime/qwen38_runtime.py generate \
  --model "$MODEL" --prompt-file research/context-ladder/prompts/prompt-8192.txt \
  --max-tokens 256 --temperature 0 --prefill-step-size 2048
```

The tested affine g128 artifact is an untracked temporary output at
`/private/tmp/qwen38-g128-mlx`; it is suitable for controlled cold
experiments, not the normal command above. No KV8 command is recommended.

## 9. Unresolved issues

- The pinned stock mlx-lm wired-limit policy remains unsafe for this model; the
  local no-wired helper is required.
- Affine g128 warm-cache behavior was not measured. It is the main missing
  practical comparison.
- The mixed converter initially emitted `.weight`-suffixed per-module config
  keys; this loader-compatibility bug was fixed and committed. The corrected
  mixed artifact still deterministically loops on the short control, so the
  underlying quality failure is unresolved.
- The cold harness's simple classifier missed the mixed long-substring loop;
  manual output review remains necessary for important gates.
- The 16,767-token g64 policy-summary run has no raw telemetry in this checkout;
  the fresh 16,727-token g128 run is the stronger saved artifact.
- `ps` and `pgrep` were denied by this execution sandbox, so process absence
  could not be independently confirmed; heavy workloads were serialized by
  command session and never intentionally overlapped.
- Natural prompt tokenization produced 1,023 and 8,191 tokens for the nominal
  1K/8K ladder labels, and 16,727 rather than 16,767 for the saved g128 coding
  prompt. These are recorded actual counts, not estimates.

## 10. Rejected hypotheses

- Simple prefill alignment caused the historical corruption: rejected by 18/18
  coherent, deterministic 0.32.1 alignment runs.
- The 3-bit g64 checkpoint or architecture is intrinsically broken at long
  context: rejected by coherent g64 and source-backed g128 long probes.
- The stock recommended wired limit is a safe memory remedy: rejected by the
  controlled Metal OOM; default wired state passes the same basic prefill.
- KV8 is a practically valuable lever on this model: rejected by the small
  savings and measured decode penalties.
- Mixed_2_6 g128 is an immediately useful higher-context configuration:
  rejected by identical pathological short-output loops.

## 11. Things deliberately not done

- No `iogpu.wired_limit_mb` or other system wired-memory setting was changed.
- No MLX 0.32.0 run was started.
- No GGUF rerun was started.
- The stock unbounded 27B converter was not launched.
- No g64 17+ GiB run was repeated solely for duplication.
- No g128 20K, 24K, or 32K probes were run after 16,727 was coherent and the
  practical objective was answered with lower exposure.
- No KV8 12K/16K matrix was run after the 8K tradeoff was negative.
- No DWQ, affine 2-bit, or additional large quantized candidates were made.
- The rejected mixed candidate's 768/4K/long gates were skipped after the
  prescribed short repeat reproduced the loop.
- No model weights were uploaded or committed; temporary converted outputs
  remain outside Git.

## 12. Next experiment

Measure affine g128 with the same persistent four-turn chat script at 8K and,
only if memory remains comfortable, 12K. This directly tests whether the
0.783 GiB model saving converts into a better warm interactive context while
preserving the already demonstrated coherence. Separately, investigate the
mixed loop with a focused quality/debugging study before producing any new
mixed long-context artifact.

## Final conclusion: fact, inference, future

Measured fact: default-wired MLX 0.32.1 with the original g64 model is coherent
through a fresh 12,288-token cold probe and has excellent 4K/8K warm-cache
behavior. Source-backed affine g128 saves 0.783 GiB of tensor payload and is
coherent through an actual 16,727-token cold probe. KV8 saves only 0.231 GiB
at 8K and slows decode. Mixed_2_6 g128 loops on the short control.

Inference: g64 no-wired with an 8K persistent cache is the safest useful
interactive configuration on this 18 GB Mac. Affine g128 is the better
controlled cold-context option, but it should not be promoted to the normal
interactive winner without warm-cache measurements.

Future hypothesis: warm affine g128 may provide a useful 8K–12K interactive
profile; the next experiment should test that directly. The mixed loop may be
caused by a quality-sensitive policy or conversion detail, but no conclusion
about its cause should be drawn from the current measurements alone.

## Research commits

Key completed commits include:

```text
1247b88  research scaffold
0f9cd47  baseline ladder
01c7190  no-wired runtime
d2fb480  4K warm cache
f4dacb4  8K warm cache
221757a  quantization size analysis
c12bace  bounded converter
39859ff  source-backed affine g128 conversion
cdab6b6  affine g128 correctness
afbf502  affine g128 context envelope
7c6b640  KV8 study
f10e503  mixed converter support
1adc61b  mixed metadata-path fix
fae0c05  corrected mixed rebuild
558d4aa  mixed candidate rejection
```
