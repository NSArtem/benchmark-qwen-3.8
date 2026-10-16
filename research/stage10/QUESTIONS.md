# Questions and blockers

- Resolved 2026-10-16: default-shell `uv lock` could not reach PyPI. Managed
  network escalation was approved; `uv.lock` was generated and the exact
  Stage 10 environment installed with no version changes.

- `import mlx_lm` failed in the default sandbox with
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

