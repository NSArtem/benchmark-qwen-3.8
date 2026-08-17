# Qwen3.8-27B MLX correctness threshold

Date: 2026-08-17. Runtime: mlx-lm commit
`254d153fdeb6f150edd4fc5a54f9828638481fa8`. Checkpoint:
`lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly`.

## Executive result

There is not a clean deterministic architectural threshold under this pinned
MLX runtime. The default 2048-step sweep shows a sharp transition region, but
repeated greedy runs at the same prompt length produce different outputs:

```text
512 tokens: 3/3 max-128 default runs COHERENT, but a dedicated max-16
            first-token run was BROKEN
640 tokens: BROKEN, COHERENT, DEGRADED across three max-128 default runs
768 tokens: 3/3 max-128 default runs BROKEN
```

The most defensible length statement is therefore:

```text
512 is the highest tested point consistently coherent in the primary
max-128/default-prefill repeats, but it is not reproducible across the
first-token max-16 diagnostic.

768 is the smallest tested point reliably broken (three default-prefill
runs, all broken). The 512–768 interval contains nondeterministic behavior;
640 is not a reliable boundary.
```

This is more specific than “MLX breaks around 4K,” but it does **not** support
an absolute-length-only threshold. Prefill chunk size changes the result, and
the runtime is nondeterministic even with the default argmax sampler.

## Primary sweep

All primary cases used the same realistic Lantern coding-agent material, the
same byte-identical final task, default prefill step 2048, no KV/activation
quantization, `enable_thinking=false`, greedy `argmax`, and 128 requested
output tokens.

| Target | Actual chat tokens | Classification | Prompt tok/s | Decode tok/s | Peak GB | First token |
|---:|---:|:---:|---:|---:|---:|:---|
| 512 | 512 | COHERENT | 81.65 | 11.10 | 13.09 | `**` |
| 1024 | 1023 | BROKEN | 82.56 | 11.16 | 13.81 | replacement character |
| 1536 | 1537 | DEGRADED | 75.38 | 11.13 | 14.28 | `</think>` |
| 2048 | 2048 | BROKEN | 152.13 | 11.08 | 14.84 | unrelated CJK |
| 2560 | 2560 | BROKEN | 134.89 | 11.04 | 14.84 | `**` loop |
| 3072 | 3072 | BROKEN | 117.35 | 11.10 | 14.84 | `-` loop |
| 3584 | 3578 | BROKEN | 114.47 | 11.05 | 14.84 | unrelated Cyrillic |
| 4096 | 4096 | BROKEN | 135.23 | 10.96 | 15.19 | replacement character |
| 4608 | 4608 | BROKEN | 125.68 | 10.86 | 15.19 | `The` then repetition |

The 1537-token output contains a substantial useful diagnosis after an invalid
leading `</think>` token, so it is DEGRADED rather than being called coherent
generously. The 1023/2048/2560/3072/3578/4096 outputs are obvious repetition,
token soup, or unrelated-language failures.

## Boundary repeats and nondeterminism

| Prompt | Run | First token | Classification |
|---:|---|:---|:---:|
| 512 | primary | `**` | COHERENT |
| 512 | repeat 1 | `**` | COHERENT |
| 512 | repeat 2 | `#` | COHERENT |
| 512 | max-16 diagnostic | `#` then `#` repeated | BROKEN |
| 640 | initial bisection | `<` / `HABitat` loop | BROKEN |
| 640 | repeat 1 | `**` | COHERENT, with domain drift |
| 640 | repeat 2 | `</think>` | DEGRADED |
| 768 | initial bisection | unrelated Italian repetition | BROKEN |
| 768 | repeat 1 | `f` repeated | BROKEN |
| 768 | repeat 2 | unrelated CJK/token soup | BROKEN |

Greedy output was not byte-identical across repeats. For example, the 512
default runs have different output hashes and first-token IDs despite identical
prompt, model, runtime, and prefill setting. The 640 result changes class across
repeats. This nondeterminism is itself an important correctness failure.

## Prefill-step experiment

The same exact prompt was used in every row; only `prefill_step_size` changed.

| Prompt | Step 2048 | Step 1024 | Step 512 | Step 256 |
|---:|:---:|:---:|:---:|:---:|
| 512 | mixed across repeats; primary COHERENT | COHERENT | BROKEN | COHERENT |
| 640 | BROKEN / COHERENT / DEGRADED | DEGRADED | COHERENT | COHERENT |

Changing chunk size can move or remove the failure for a fixed prompt, but it
does not produce a simple scaling law such as “the threshold equals two
chunks.” Both 512 and 640 are below the default 2048 chunk, yet changing the
step still changes the output. This favors a chunk/state-sensitive runtime
bug, with nondeterministic Metal execution or state handling also implicated.

## First-token diagnostic

- At 640 with the default step, the first generated token was already
  `</think>` (ID 248069); this is pathological at the first-token boundary.
- At 512 in the dedicated max-16 diagnostic, the first token was `#` (ID 2),
  a plausible markdown token, but the next 15 tokens were the same token
  repeated. That is immediate decode/state collapse.
- At 1023 in the primary sweep, the first token decoded as a replacement
  character and was followed by repeated `</think>`.

Therefore corruption is sometimes already present in the first token and in
other runs begins as a plausible token followed by immediate collapse. This
is not evidence for ordinary sampling drift; the runtime uses argmax by
default.

## Memory

Memory does not explain the threshold. Primary peak MLX memory rises gradually:

```text
512:   13.09 GB
640:   13.21–13.23 GB
768:   13.38 GB
1023:  13.81 GB
1537:  14.28 GB
2048–3578: about 14.84 GB
4096–4608: about 15.19 GB
```

The primary runs complete without OOM. The only alternate-quant failure was a
separate mixed 3.5-BPW checkpoint that reached 13.67 GB and failed during
Metal synchronization; it was not used for threshold classification.

## Direct answers

1. **Largest reliably coherent prompt:** No fully reliable length was found
   under the complete protocol. 512 tokens is the highest point with three
   coherent default max-128 runs, but its separate max-16 diagnostic was
   broken.
2. **Smallest reliably broken prompt:** 768 tokens, the lowest tested point
   broken in three default-prefill runs. 640 is mixed rather than reliably
   broken.
3. **Sharpness:** The transition is concentrated between 512 and 768, but it
   is stochastic rather than a single deterministic cliff.
4. **Power-of-two boundary:** No obvious absolute 1024/2048 boundary. The
   first failures occur below 1024, and all boundary prompts below 2048 use a
   single default prefill chunk.
5. **Prefill dependence:** Yes. Changing 2048/1024/512/256 changes the result
   for fixed 512- and 640-token prompts, but not according to a simple scaling
   rule.
6. **First-token corruption:** Sometimes yes: 640 and 1023 show pathological
   first tokens.
7. **Later decode collapse:** Also yes: 512 can start with `#` and then
   repeat it immediately.
8. **Memory change:** Gradual only; there is no memory discontinuity around
   the correctness transition.
9. **Strongest MLX suspect:** The evaluated Metal `GatedDeltaNet` path in
   `mlx_lm/models/gated_delta.py`, together with Qwen's persistent
   `ArraysCache` convolution/recurrent state and the mixed `KVCache` path in
   `qwen3_5.py`.
10. **Supporting evidence:** Fixed prompts change class when chunk size
    changes; the generator evaluates and mutates the same hybrid cache across
    chunks; first-token corruption is observed; the custom Metal kernel is
    used in eval mode; memory is not the cause.
11. **Evidence against overclaiming:** There is no isolated source-level
    proof, no clean monotonic threshold, and the alternative quant control
    could not run because it did not fit. Stable `A_log`/`dt_bias` loading also
    argues against one simple random-initialization explanation.
12. **Current diagnosis:** Favors an MLX hybrid-runtime/state-update issue,
    but the formal conclusion remains **unresolved** between runtime behavior
    and a checkpoint/conversion interaction.
13. **Fresh self-conversion:** Yes, now justified as the next isolating
    experiment. The current checkpoint-specific conversion cannot be ruled out
    because the alternate quant did not fit.
14. **Exact next conversion experiment:** Starting from the official
    Qwen3.8-27B HF checkpoint in an external directory, use the pinned CLI:

    ```sh
    UV_CACHE_DIR=/private/tmp/qwen38-mlx-threshold-uv-cache \
      uvx --from 'git+https://github.com/ml-explore/mlx-lm@254d153fdeb6f150edd4fc5a54f9828638481fa8' \
      mlx_lm.convert \
      --hf-path /path/to/official/Qwen3.8-27B \
      --mlx-path /private/tmp/qwen38-self-converted-q3 \
      --quantize --q-bits 3 --q-group-size 64 --q-mode affine \
      --dtype bfloat16
    ```

    Then run only the 512/640/768 prompts with default step 2048 and step
    512, using this same harness. A different threshold or stable output would
    implicate the current conversion; the same mixed/nondeterministic failure
    would strengthen the runtime diagnosis.
15. **Upstream issue quality:** Yes. The pinned commit, exact prompt hashes,
    raw outputs, first-token logprobs, repeat nondeterminism, chunk matrix,
    memory data, and source line references are sufficient for a high-quality
    upstream issue. The report should lead with the minimal 512/640/768 cases.

## Prompt provenance

The prompt body reuses the existing realistic Lantern source/code/project
material. Only the prefix length changes. The final task is byte-identical in
all generated prompt files; task SHA256:
`fbcf518ac8bfc9554567c0c0309b9950b3ca17cea7bd2818f2cf64f3f6ebc87d`.

| File | Target | Actual chat tokens | SHA256 |
|---|---:|---:|---|
| `0512.txt` | 512 | 512 | `13498200a902b0e3800abbec81703d4a082ebd68d2591dea21628398678659b0` |
| `0640.txt` | 640 | 640 | `c5fa4c4514d254f69f79a997436c54bedfd10db4f6fa5480d5017f686223e922` |
| `0768.txt` | 768 | 768 | `e43371c867d970f5001796f970c080914a17c4bd34fcb21a15115619c3333767` |
| `1024.txt` | 1024 | 1023 | `56f745eb7e18b0e0a9bd9a5bf760a5a2a0efbb1c6f67a37114b58a235690e8cf` |
| `1536.txt` | 1536 | 1537 | `944df48ce4005b378c13f98d78fdc3c1ea0aa0e254a153bd6d90bc025b692884` |
| `2048.txt` | 2048 | 2048 | `72ba00c19bc154abaa3abf150c3e1ee97e8d4058c28a2e20f2964a61630ded15` |
| `2560.txt` | 2560 | 2560 | `d197b93594c1dd23c8195cf7686971711272672fc3d9065802f2d52692edcada` |
| `3072.txt` | 3072 | 3072 | `4903dc46ad8557db932efa3ef69c90506c47c372424f6b39ac9402a435742f02` |
| `3584.txt` | 3584 | 3578 | `3d6c321baeb07d1f444d0c2ed57a833170646cae248155a70dc167eaff2e366c` |
| `4096.txt` | 4096 | 4096 | `8a20fe6339d8f0b3b08eace995df55c27609423d982fc4351600a42b91362284` |
| `4608.txt` | 4608 | 4608 | `b0b21dd2a042b13d867588cf4a87790faeaa70c4e63d49e5278af44337f95176` |

## Artifacts

- [`results.json`](results.json) contains all 26 persisted tests and required
  per-run fields.
- [`prompt-manifest.json`](prompt-manifest.json) records raw/chat counts and
  hashes.
- [`source-inspection.md`](source-inspection.md) records the pinned code-path
  review.
- [`logs/`](logs/) contains stdout/stderr and per-case JSON.
- [`outputs/`](outputs/) contains every exact generated output, including
  failures and repeats.
- [`harness/run_case.py`](harness/run_case.py) is the public-API runner.
