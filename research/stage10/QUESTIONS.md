# Questions and blockers

- Resolved 2026-10-16: default-shell `uv lock` could not reach PyPI. Managed
  network escalation was approved; `uv.lock` was generated and the exact
  Stage 10 environment installed with no version changes.

- Resolved 2026-10-16 (recheck: Metal available, gpu:0, Apple M3 Pro): `import mlx_lm` failed in the default sandbox with
  `[metal::load_device] No Metal device available`. No model weights were
  loaded. Run future MLX checks in an execution context with Metal access,
  through the required preflight and watchdog.

- The Russian source says articles `0–1999` and `10000–11999` "with ≥ 2000
  characters" without stating whether ordinals refer to the raw stream or the
  stream after filtering. The implementation numbers only articles that pass
  the 2000-character filter, then selects those ordinal ranges.

## STOP — 2026-10-16 (resolved: user authorized resume; A3-sanitize PASS 55/55)

- A3-loop passed: 4094 scored positions, max absolute log-probability
  difference 0.0, argmax agreement 1.0.
- A3-sanitize attempt 3 failed after two code fixes with
  `ValueError: 'bfloat16' is not a valid PEP 3118 buffer format string` in
  `_same_bytes`. No tensor comparisons were produced. TASK.md §7 says to stop
  the whole task after two honest fix attempts; no further A3-sanitize run was
  made.


## Interpretation choices — 2026-10-16 (resume session)

- A5 pack test: `mx.dequantize` computes `q * scale + bias` with one final
  rounding (fused multiply-add); a two-step float32 product and sum differ
  by 1 ULP at up to 17.7% of elements. The test keeps exact equality and
  computes the expected value in float64 (where `q * scale` is exact) with
  one rounding to float32. Random float32 scales and biases are unchanged.
- Rotated RTN: the weight is rotated in float32, cast to bfloat16, then
  quantized with `mx.quantize` (unrotated weights are cast to bfloat16 and
  quantized as in `tools/build_g128.py`). Scales/biases are bfloat16 in all
  recipes.
- Rotated GPTQ: `make_signs` depends on the module path, so modules that
  share an input have different rotations. One unrotated `H = Σ XᵀX` is
  accumulated per shared input, and each module uses `H_m = R_mᵀ H R_m`,
  which equals `Σ (X R_m)ᵀ (X R_m)`.
- GPTQ scales: as in mlx-lm `gptq.py`, per-group scales/biases come from
  `mx.quantize` on the float32 working weights; codes are computed with those
  float32 scales; scales/biases are then stored as bfloat16 (`set_dtype` of
  the original weight dtype in mlx-lm).
- GPTQ/AWQ forward embedding for rotated recipes: the BF16 embedding rows of
  the calibration tokens are rotated in float32 and inverse-rotated (as a
  rotated QEmbedding does), then cast to bfloat16.
- AWQ propagates only the 16 search sequences through the quantized layers,
  since only those are used.
- A8 `prefill_s` is the time from the `generate_step` call to the last
  prompt-progress callback (all prompt tokens but the last are in the
  cache); `ttft_s` is the time to the first yielded token.
- A9 `n_tokens` counts generated tokens excluding the EOS token.
- AWQ: `mean(|X|)` is clamped below at 1e-8 before `** α`, so an all-zero
  input channel cannot make `s / sqrt(max(s) * min(s))` divide by zero.
  Folded tensors are scaled in float32 and stored in their original dtype.
