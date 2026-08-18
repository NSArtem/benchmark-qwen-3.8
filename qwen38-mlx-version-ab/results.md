# Qwen3.8 MLX 0.32.0 vs 0.32.1

## Scope

This was a fixed two-version A/B test. The checkpoint, local snapshot, mlx-lm
Git commit, prompts, chat template, default prefill, sampler, and output limit
were held constant. No dependency, lockfile, memory, or performance tuning was
performed.

The pinned mlx-lm API's default sampler is greedy `argmax`; passing a `temp`
keyword is not supported by this revision, so the runner intentionally uses the
default sampler without a custom sampling argument.

## Environment verification

Both environments loaded successfully for the short control:

| Environment | Python | MLX | mlx-lm metadata | mlx-lm Git commit |
| --- | --- | --- | --- | --- |
| `mlx0320` | 3.14.7 | 0.32.0 | 0.31.3 | `254d153fdeb6f150edd4fc5a54f9828638481fa8` |
| `mlx0321` | 3.14.7 | 0.32.1 | 0.31.3 | `254d153fdeb6f150edd4fc5a54f9828638481fa8` |

The commit was verified from installed PEP 610 `direct_url.json` metadata.

## Prompts

The prompt files were reused from the prior experiments and copied into this
directory without changing bytes:

| Prompt | Source/copy | Raw tokens | Chat tokens | SHA256 |
| --- | --- | ---: | ---: | --- |
| short control | `prompts/short-control.txt` | 22 | 33 | `a06c9a7705e30b631d33e8c336f7d466d13b9cec267ff5bc0977d8fd3faffa18` |
| problematic control | `prompts/0768.txt` | 757 | 768 | `e43371c867d970f5001796f970c080914a17c4bd34fcb21a15115619c3333767` |

The 768-token prompt SHA256 was identical in all six fresh-process attempts.

## Results matrix

| Run | MLX | Prompt | Classification | First token | First 16 token IDs / failure |
| --- | ---: | ---: | --- | --- | --- |
| short | 0.32.0 | 33 | COHERENT | `As` / 2053 | `[2053, 264, 3914, 10507, 11, 353, 3849, 9218, 421, 10306, 5833, 52459, 318, 3026, 49, 8]` |
| short | 0.32.1 | 33 | COHERENT | `As` / 2053 | identical to 0.32.0 |
| 768 r1 | 0.32.0 | 768 | BROKEN | `irl` / 2722 | repeated `irl`/newline pattern |
| 768 r2 | 0.32.0 | 768 | BROKEN | ` assol` / 161227 | repeated `assoluto`/`assoluzione` fragments |
| 768 r3 | 0.32.0 | 768 | BROKEN | `中` / 95789 | repeated `中` with spaces |
| 768 r1 | 0.32.1 | 768 | BROKEN* | none | Metal Insufficient Memory before first token |
| 768 r2 | 0.32.1 | 768 | BROKEN* | none | Metal Insufficient Memory before first token |
| 768 r3 | 0.32.1 | 768 | BROKEN* | none | Metal Insufficient Memory before first token |

`BROKEN*` denotes an execution failure rather than malformed generated text;
the exact error was:

`RuntimeError: [METAL] Command buffer execution failed: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory).`

All raw completions, logs, and per-run JSON records are retained under
`outputs/` and `logs/`. The MLX 0.32.0 output SHA256 values were distinct for
all three 768-token runs. The three MLX 0.32.1 output files are identical empty
files because generation never began.

## Explicit answers

1. Both environments loaded successfully for the short control.
2. The short known-good prompt worked in MLX 0.32.0 and produced coherent prose.
3. It also worked in MLX 0.32.1; the full output and first 16 token IDs matched.
4. MLX 0.32.0 produced three broken completions at 768 tokens.
5. MLX 0.32.1 produced three reproducible Metal OOM failures at 768 tokens,
   before the first token.
6. MLX 0.32.0 was not deterministic across fresh processes: first token IDs
   were 2722, 161227, and 95789, with three different completions.
7. MLX 0.32.1 was deterministic in its failure mode across the three repeats,
   but did not generate tokens, so generation determinism cannot be assessed.
8. Upgrading MLX core materially changed the 768-token failure mode from
   nondeterministic token soup to a repeatable Metal memory failure. It did not
   restore coherent 768-token generation. Short-context behavior was unchanged.
9. Decision case: **D**, with the qualification that MLX 0.32.1 cannot execute
   the fixed 768-token case under these default settings, so this is not a
   clean correctness A/B separation.
10. The next experiment should first isolate/confirm the MLX 0.32.1 memory
    regression at this exact fixed prompt and settings. Do not start the
    512/513, 640/641, or 768/769 alignment experiment until both versions can
    execute the same case.

## Conclusion

Changing only MLX core from 0.32.0 to 0.32.1 did **not** make Qwen3.8 coherent
at the known-problematic 768-token context. It changed the observed behavior:
0.32.0 generated different pathological outputs in every fresh process,
whereas 0.32.1 failed reproducibly with Metal OOM before generation.
