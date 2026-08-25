# Qwen3.8 Stage 4 Plan

This plan follows `TASK.md` and is intentionally gated. Heavy model work is
serialized; the canonical `envs/mlx-0.32.1` environment is used with
`uv run --project envs/mlx-0.32.1 --no-sync`, with a task-local `UV_CACHE_DIR`
only where the launcher cannot read the default cache.

| Phase | Gate / deliverable | Status |
|---|---|---|
| 0 | Recover Stage 3 evidence; create Stage 4 memory | completed |
| 1 | Corrected 30–40 case Quality Suite V2, validated and frozen | completed |
| 2 | One focused turn-17 early-EOS control study | completed |
| 3 | Bounded per-module sensitivity map and budget curve | completed |
| 4 | At most three serious new candidates: mixed 3/4 control, guided 3/4, optional 3/6 | completed |
| 5 | Mini tournament; advance at most one new candidate | completed |
| 6 | Full frozen-suite finalist comparison and role classification | in progress |
| 7 | Winner validation at approximately 8K and 16K persistent exact-suffix chat | pending |
| 8 | Production `qwen38-chat` runtime, tests, and documentation | pending |
| 9 | Learned-quant feasibility assessment only after primary work | pending |

Safety gates:

- never run two heavy Qwen jobs concurrently;
- never change system wired memory or call `mx.set_wired_limit()`;
- stop on Metal OOM, GPU timeout, UI instability, or unexplained runaway memory;
- never commit model weights or the high-precision source;
- keep calibration and evaluation splits separate and hash frozen artifacts;
- do not reopen the already validated B-vs-C cache oracle without contradictory evidence.
