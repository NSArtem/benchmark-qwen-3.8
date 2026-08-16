# Qwen3.5-9B DFlash smoke test

Run date: 2026-08-17

## Environment

- Hardware: MacBook Pro Mac15,6, Apple M3 Pro (11 cores), 18 GB unified memory
- macOS: 15.7.8 (24G824)
- uv: 0.12.5 (Homebrew, aarch64)
- Python: 3.14.7
- MLX: 0.32.0
- mlx-vlm: 0.6.13
- mlx-lm: 0.31.3
- transformers: 5.15.0
- huggingface-hub: 1.27.0

No dependency change was needed. The installed `mlx-vlm` exposes
`--draft-model`, `--draft-kind dflash`, and `--draft-block-size`; no block-size
override was used.

## Models and workload

- Target: `mlx-community/Qwen3.5-9B-4bit`
- DFlash drafter: `z-lab/Qwen3.5-9B-DFlash`
- Prompt: [`coding-prompt.txt`](coding-prompt.txt), copied byte-for-byte from
  the previous Qwen3.5 experiment
- Prompt SHA256: `67dcccb037993b679c5593073bbc1875ad61541827a6fc3bb6720410b8096903`
- Actual formatted input: 16,727 tokens
- Generation: 256 tokens, batch size 1, temperature 0, seed 0, thinking disabled
- Prefill step size: 2,048, matching the previous direct-generation harness
- APC: not used
- Server: not used

## Compatibility check

The 4-token sanity check loaded both models, passed
`validate_drafter_compatibility`, resolved `draft_kind` to `dflash`, and emitted
one DFlash speculative round with 3 drafted and 2 accepted tokens. It returned
`READY`. This establishes genuine DFlash dispatch; the drafter was not run as a
standalone language model and there was no autoregressive fallback.

Checkpoint/runtime configuration:

- Configured DFlash block size: 16
- Runtime block-size override: none
- Runtime block-size field: null
- Draft-window setting: null
- Target layer IDs: `[1, 5, 9, 13, 17, 21, 25, 29]`
- Loaded DFlash parameter arrays: approximately 2.584 GB
- Target forward-pass count: not exposed by this runtime

## Results

The first two rows are previous measurements, not rerun in this experiment.

| Mode                          | Input tokens | Decode tok/s | vs baseline | vs MTP | Mean accept | Peak MLX | Verdict |
| ----------------------------- | -----------: | -----------: | ----------: | -----: | ----------: | -------: | ------- |
| Existing baseline reference   |       16,727 |        24.16 |       1.00x |  0.81x |           — |  9.915 GB | previous reference |
| Existing native MTP reference |       16,727 |        29.85 |       1.24x |  1.00x | 2.09 tokens/round | 10.052 GB | previous reference |
| DFlash                        |       16,727 |        22.10 |       0.91x |  0.74x | 2.42 tokens/round | 11.812 GB | slower than both |

DFlash measured 259.42 prompt tok/s, 64.479 s prefill, 65.115 s TTFT,
22.096 decode tok/s, 11.600 s post-first-token decode time, and 76.715 s total
wall time. The comparable prior baseline and MTP prefill times were about
63.997 s and 63.991 s respectively, so DFlash provided no long-context prefill
benefit here.

## DFlash acceptance behavior

- Verification rounds: 106
- Drafted tokens: 331
- Accepted draft tokens: 150
- Rejected draft tokens: 181
- Draft acceptance rate: 45.32%
- Mean accepted draft tokens per round: 1.42
- Mean accepted tokens including the target bonus per round: 2.42
- Mean draft length: 3.12 tokens
- The runtime adaptively reduced most rounds to 3-token drafts under the
  configured 16-token ceiling; this was its normal behavior, not a sweep or a
  manually tuned block size.

## Memory and correctness

Server-reported peak MLX memory was 11.812 GB, approximately 1.897 GB above the
previous baseline reference and 1.759 GB above the previous MTP reference. The
lightweight system snapshot went from 46% to 22% free memory; swap used changed
from 1,253.5 MB to 535.4 MB, while pageouts increased by 509. There was no OOM,
severe sustained paging, or system collapse. The single before/after snapshot
cannot characterize long-run pressure, but DFlash clearly required more memory
than native MTP.

The exact 256 generated token IDs and text are saved in
[`dflash-tokens.json`](dflash-tokens.json) and
[`dflash-output.txt`](dflash-output.txt). Compared with the saved previous
outputs on the same prompt/configuration, DFlash shared the first 63 tokens and
first differed at token position 63 (zero-based) against both baseline and MTP.
The text remained coherent and there was no obvious corruption; the prior
baseline and MTP outputs themselves first differed at position 174, so exact
sequence identity is sensitive to dispatch/tie-breaking behavior in this setup.

## Direct answers

1. Yes. Qwen3.5-9B DFlash worked through mlx-vlm on this Mac, with a passed
   target/drafter validation and observed DFlash rounds.
2. The checkpoint-configured block size was 16. No override, runtime block-size
   setting, or draft-window setting was present.
3. DFlash achieved 22.10 tok/s at 16,727 input tokens.
4. It drafted 331 tokens over 106 rounds and accepted 150 (45.32%); mean
   accepted length was 2.42 tokens per round including the target bonus.
5. It reached 0.91x the plain baseline, about 8.5% slower.
6. It reached 0.74x native MTP, about 26.0% slower.
7. Its peak was about 1.90 GB above the previous baseline and 1.76 GB above
   native MTP; the loaded DFlash arrays alone were about 2.584 GB.
8. No sustained paging/thrashing was evident in this short run; swap usage fell,
   although free memory reached 22% and pageouts increased.
9. No. DFlash does not beat native MTP here, so an APC+DFlash follow-up is not
   justified by this smoke test.
10. Yes. Native MTP should remain the preferred configuration for this Mac and
    workload.

## Verdict and limitations

DFlash is functional but not materially faster on this practical long-context
workload. At 22.10 tok/s it is slower than the plain target reference and much
slower than the already-working 29.85 tok/s native-MTP configuration. The
additional memory cost is also substantial relative to native MTP. Stop here as
requested; no APC, server, block-size sweep, context sweep, or alternate model
was tested.

## Artifacts and commits

- Compatibility evidence: [`dflash-check.json`](dflash-check.json),
  [`dflash-check.log`](dflash-check.log)
- Measured log: [`dflash.log`](dflash.log)
- Harness: [`benchmark.py`](benchmark.py)
- Reproduction commands: [`commands.txt`](commands.txt)
- Harness/config commit: `2eb576f` (`benchmark: add Qwen3.5 DFlash smoke test`)
- Results commit: pending

The model/Hugging Face cache, `.venv`, and Python caches were not staged. The
pre-existing unrelated archives in the repository root were preserved.
