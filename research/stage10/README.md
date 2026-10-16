# Stage 10 tools

Run commands from this directory with `uv run ...`. The research stack was
started on 2026-10-16 with Python 3.14.7, `mlx==0.32.4`, and
`mlx-lm==0.32.1`, the latest PyPI releases observed at startup. Resolved
versions for the complete environment will be recorded here after locking.

Heavy commands must call `preflight()` and run through
`stage10.common.run_guarded`. Large arrays and process logs belong in the
git-ignored `benchmarks/results/stage10/` directory in the repository.
