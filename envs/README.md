# Controlled MLX environments

These are independent uv projects for reproducible Qwen3.8 experiments. They
are intentionally not a uv workspace: each directory has its own
`pyproject.toml`, `uv.lock`, `.python-version`, and local `.venv`.

| Environment | Python | MLX | mlx-lm |
| --- | --- | --- | --- |
| `mlx0320` | 3.14.7 | 0.32.0 | Git commit `254d153fdeb6f150edd4fc5a54f9828638481fa8` |
| `mlx0321` | 3.14.7 | 0.32.1 | Git commit `254d153fdeb6f150edd4fc5a54f9828638481fa8` |

The pinned mlx-lm checkout reports package metadata version `0.31.3`; the Git
commit, not that metadata version, identifies the source used here.

The lockfile comparison found no unrelated dependency-version differences.
`mlx` changes from 0.32.0 to 0.32.1 and the coupled macOS `mlx-metal` package
changes by the same version. The standalone project name and direct MLX
specifier also necessarily differ in the two lockfiles.

The exact source was verified in both environments through the installed PEP
610 metadata at
`mlx_lm-0.31.3.dist-info/direct_url.json`; it records
`https://github.com/ml-explore/mlx-lm.git` and commit
`254d153fdeb6f150edd4fc5a54f9828638481fa8`.

Use the project explicitly:

```sh
uv sync --project envs/mlx-0.32.0 --locked
uv run --project envs/mlx-0.32.0 python script.py

uv sync --project envs/mlx-0.32.1 --locked
uv run --project envs/mlx-0.32.1 python script.py
```

The optional wrappers provide the shorter equivalents:

```sh
envs/run-0320 python script.py
envs/run-0321 python script.py
```

Do not use bare `uvx --from git+...` for controlled Qwen3.8 experiments. The
pinned mlx-lm revision specifies only a lower bound for MLX, so an independent
uvx resolution can silently select a different MLX core version.

The root `pyproject.toml` and `uv.lock` remain a separate legacy/general
environment. They are not the canonical environments for controlled MLX A/B
experiments.

The nested `.venv` directories are local and ignored by Git. No model weights
or model caches belong in this repository.
