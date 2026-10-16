# Questions and blockers

- `uv lock --python 3.14.7` could not reach `https://pypi.org/simple/` because
  DNS/network access is unavailable to shell commands. Keep the specified
  latest-version pins in `pyproject.toml`; retry lock resolution when network
  access is available. No dependency versions were changed.

- The Russian source says articles `0–1999` and `10000–11999` "with ≥ 2000
  characters" without stating whether ordinals refer to the raw stream or the
  stream after filtering. The implementation numbers only articles that pass
  the 2000-character filter, then selects those ordinal ranges.

