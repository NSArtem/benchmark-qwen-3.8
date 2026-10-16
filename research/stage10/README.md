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

## Commands

- Unit tests: `uv run python -m unittest discover -s tests`
- A3 checks: `uv run python -m stage10.check_a3 --mode loop|sanitize`
- A4 BF16 reference (E00): `uv run python -m stage10.reference`
- Build one candidate: `uv run python -m stage10.build --exp E02`
  (`--compare-production` counts tensors byte-identical to production g128)
- A7 eval / A9 gen-check / A8 ladder for one model:
  `uv run python -m stage10.evaluate|gen_check --exp E01 --model <dir> --loader mlx_lm|candidate [--rotate-dtype bfloat16]`;
  `uv run python -m stage10.ladder ... --weights-gib <GiB> [--ctx N]`
  (4096, 8192, 16384, 24576, 32768; --ctx appends or replaces one length)
- Whole experiment (build, eval, gen-check, ladder; LOG.md entries appended):
  `uv run python -m stage10.run_exp --exp E03 [--steps build,eval,gen,ladder]`
- Summary and table: `uv run python -m stage10.summary --exp E03 --recipe had3 --status OK`;
  `uv run python -m stage10.make_table`

Every heavy step runs in a child process under `tools/memory_watchdog.sh`
after `preflight()`. Per-run outputs go to `benchmarks/results/stage10/<EXP_ID>/`;
candidates go to `/Users/artem/models-stage10/<EXP_ID>/`.

Amendment 1: gen-check limit 1024 tokens. E07–E12 runtime rotation dtype
is bfloat16 in build metadata and module config; weight folding remains float32.
