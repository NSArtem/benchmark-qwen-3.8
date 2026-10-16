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

