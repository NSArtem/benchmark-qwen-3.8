# Stage 10 tools

Run commands from this directory with `uv run ...`. The research stack was
started on 2026-10-16 with Python 3.14.7, `mlx==0.32.4`, and
`mlx-lm==0.32.1`, the latest PyPI releases observed at startup. Resolved
direct dependency versions in `uv.lock`: Python 3.14.7; mlx 0.32.4;
mlx-lm 0.32.1; numpy 2.5.4; safetensors 0.8.0; datasets 5.1.0;
transformers 5.20.0; tokenizers 0.23.3; huggingface-hub 2.2.0.

Heavy commands must call `preflight()` and run through
`stage10.common.run_guarded`. Large arrays and process logs belong in the
git-ignored `benchmarks/results/stage10/` directory in the repository.
