# Questions and blockers

- `uv lock --python 3.14.7` could not reach `https://pypi.org/simple/` because
  DNS/network access is unavailable to shell commands. Keep the specified
  latest-version pins in `pyproject.toml`; retry lock resolution when network
  access is available. No dependency versions were changed.

