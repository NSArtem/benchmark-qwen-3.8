# Qwen3.8 MLX 0.32.1 Prefill-Alignment Test

## Result

All 18 counted runs completed the full 128-token greedy completion. Every
completion was coherent; no OOM occurred.

| Final tokens | Main prefill | Run 1 | Run 2 | Run 3 |
|---:|---:|---|---|---|
| 512 | 511 | COHERENT | COHERENT | COHERENT |
| 513 | 512 | COHERENT | COHERENT | COHERENT |
| 640 | 639 | COHERENT | COHERENT | COHERENT |
| 641 | 640 | COHERENT | COHERENT | COHERENT |
| 768 | 767 | COHERENT | COHERENT | COHERENT |
| 769 | 768 | COHERENT | COHERENT | COHERENT |

Counts: `COHERENT 18/18`, `DEGRADED 0/18`, `BROKEN 0/18`, `OOM 0/18`.

The completions begin with token ID `2`, decoded as `#`. The first 16 generated
token IDs are identical across the three fresh processes at every individual
length. The complete generated token-ID sequence is also identical across the
three fresh processes at every length.

Unique completion hashes by length are:

| Final tokens | Generated-token SHA256 |
|---:|---|
| 512 | `b693f5d56889bfa48f4528f46b27d7d6984228d959c53a9863942ed2b573fe0d` |
| 513 | `d08d2eb8a75c57701b2dc463481ef42e1640f39d64dff910e15dfa9e5687f74a` |
| 640 | `7eb4d751e624ba3da2157afad34f9265ec5bef2aff552a0826d8c7275c7d1f4f` |
| 641 | `51d53d5d9a7382b116dcb49d6bc1c11ed21da01907eb344480056b33352b5d72` |
| 768 | `f0d9f6b4fa45e2dcd4571a6c2bd43d0b750354039ff05bdb29fe773d7ed9b5a8` |
| 769 | `f0d9f6b4fa45e2dcd4571a6c2bd43d0b750354039ff05bdb29fe773d7ed9b5a8` |

The raw completions are saved under `outputs/`; they contain normal incident
diagnosis prose/Markdown rather than repetition, token soup, or collapse.

## Prompt construction

The prompts reuse the compact Lantern coding-agent context and the same final
task as the prior threshold experiment. The final task is byte-identical in
all six raw prompts and has SHA256
`fbcf518ac8bfc9554567c0c0309b9950b3ca17cea7bd2818f2cf64f3f6ebc87d`.
Counts below are after `apply_chat_template(..., add_generation_prompt=True,
enable_thinking=False)` using the Qwen tokenizer.

| Target/final | Actual | Main prefill | Raw tokens | Raw SHA256 |
|---:|---:|---:|---:|---|
| 512 | 512 | 511 | 501 | `0f3e14f9ceee867f549a108012f285119b7fb171fb2538e268e3be549ec45705` |
| 513 | 513 | 512 | 502 | `50248327d566b666ea8a87ee46db131c1846278f77a2a15eb84b2ceb3d0575c4` |
| 640 | 640 | 639 | 629 | `19e7e67c2013948276db3ede3e8e782452ad099e529b806a142a7fc40d25f909` |
| 641 | 641 | 640 | 630 | `52e9c7e4018cdc2597b08c89c212a104ae587b535d10972f70ef5b667184853e` |
| 768 | 768 | 767 | 757 | `89cfb4367bd3f8df6164985ca523819a682fda2c4c802a8d29a30ac936669cfc` |
| 769 | 769 | 768 | 758 | `6519f67669cfb10c4d85bf19be7a9bc6681856811431ff05baf290c2039dffab` |

Each adjacent pair has an exact one-token insertion before the unchanged task
suffix:

- 512 → 513: insert token `1788` at index 242;
- 640 → 641: insert token `5331` at index 370;
- 768 → 769: insert token `3082` at index 498.

The saved `prompt-*.tokens.json` files contain the complete token sequences and
the manifest records these comparisons.

## Pair analysis

| Pair | Unaligned side | Aligned side | Unaligned result | Aligned result | Same first token/16? | Same full completion? |
|---|---:|---:|---|---|---|---|
| 512 vs 513 | 511 prefill | 512 prefill | 3/3 coherent | 3/3 coherent | Yes / Yes | No |
| 640 vs 641 | 639 prefill | 640 prefill | 3/3 coherent | 3/3 coherent | Yes / Yes | No |
| 768 vs 769 | 767 prefill | 768 prefill | 3/3 coherent | 3/3 coherent | Yes / Yes | Yes |

The different full completions for the first two pairs are expected from their
different prompt content, not evidence of nondeterminism. Each side is stable
across its fresh-process repeats. The 768/769 completions happen to be
identical for this 128-token generation.

## Memory observations

MLX memory values below are from run 1 of each length; the three repeats at a
given length had the same reported peak. The runner did not set a wired limit.

| Final tokens | Active after first token (GiB) | Cache after first token (GiB) | Active+cache (GiB) | Peak (GiB) |
|---:|---:|---:|---:|---:|
| 512 | 11.192 | 0.126 | 11.318 | 12.188 |
| 513 | 11.183 | 0.213 | 11.396 | 12.187 |
| 640 | 11.183 | 0.147 | 11.330 | 12.325 |
| 641 | 11.183 | 0.024 | 11.207 | 12.327 |
| 768 | 11.210 | 0.183 | 11.393 | 12.461 |
| 769 | 11.199 | 0.164 | 11.363 | 12.459 |

The adjacent peak differences were approximately -0.0007 GiB, +0.0012 GiB,
and -0.0018 GiB respectively. There is no material N-versus-N+1 memory jump.
Model active memory after load was approximately 10.963 GiB. The small gradual
peak increase from 512 to 768 is normal context growth, not an alignment cliff.

## Explicit answers

1. **Is corruption reproducible with default wired state?** No. It was not
   reproduced at any of the six lengths in 18 direct `generate_step()` runs.
2. **Is there a clear aligned-vs-unaligned effect?** No. Both sides of all
   three pairs were 3/3 coherent and deterministic.
3. **Which lengths are coherent/degraded/broken?** 512, 513, 640, 641, 768,
   and 769 were all coherent; none were degraded or broken.
4. **Are fresh-process runs token-level deterministic?** Yes: same first token,
   same first 16 tokens, and same complete 128-token sequence within each
   length.
5. **Does first-token behavior differ across repeats?** No. Every run produced
   token ID 2 (`#`).
6. **Does memory materially differ between N and N+1?** No; the measured peak
   changes were under 0.002 GiB and varied in sign.
7. **Decision case:** **D — all coherent**. The historical corruption is not
   reproduced under the current controlled MLX 0.32.1, direct-generation,
   default-wired execution path. This is not a claim that the historical bug is
   fixed.
8. **Highest-value next experiment:** replay the exact historical failing
   execution path and artifacts—especially the old prompt bytes/template and
   whether the run used `stream_generate()`/automatic wired handling—under
   MLX 0.32.1. First establish which path reproduces corruption versus the
   direct default-wired path; do not infer a fix from this alignment result.

## Scope and reproducibility

Only `envs/mlx-0.32.1` was used. The model was loaded from the existing local
snapshot. No MLX 0.32.0, GGUF, alternate checkpoint, dependency change,
prefill-size sweep, wired-limit change, sysctl change, stream generation, or
additional alignment length was used in the counted experiment.
