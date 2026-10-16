## 2026-10-16 session start
cmd: `git status --short --branch`; `git log --oneline -20`; read `AGENTS.md` and `TASK.md`
commit: 09a3c06
result: OK
facts: main clean at start; `research/` did not exist. Active Python is 3.15.0; uv-managed Python 3.14.7 is installed. PyPI latest observed: mlx 0.32.4 (2026-10-14), mlx-lm 0.32.1 (2026-10-15).

## 2026-10-16 A1 common helpers
cmd: added `stage10/common.py`, project skeleton, and the section 11 checklist
commit: staged for `stage10: add common helpers`
result: IN_PROGRESS
facts: `host_check`, `preflight`, and watchdog-managed subprocess API implemented; no heavy workload run.

## 2026-10-16 Stage 10 dependency lock
cmd: `UV_CACHE_DIR=/private/tmp/stage10-uv-cache uv lock --python 3.14.7`
commit: n/a
result: FAILED_ERROR
facts: uv selected CPython 3.14.7; package index request failed after 3 retries because DNS lookup for pypi.org failed. No lockfile was generated. Details recorded in QUESTIONS.md.

## 2026-10-16 A1 commit
cmd: `git commit -m 'stage10: add common helpers'`
commit: c17cb5a
result: OK
facts: A1 helpers and Stage 10 project skeleton committed; `git status --ignored research/` showed no ignored Stage 10 files.

## 2026-10-16 A2 token set tool and unit tests
cmd: `UV_CACHE_DIR=/private/tmp/stage10-uv-cache UV_PROJECT_ENVIRONMENT=/Users/artem/projects/benchmark-qwen-3.8/.venv uv run --offline --no-sync --python /Users/artem/projects/benchmark-qwen-3.8/.venv/bin/python -m unittest discover -s tests`
commit: pending
result: PARTIAL
facts: 2 tests passed. Tests used the existing Python 3.14.7 root environment because the isolated Stage 10 dependencies could not be resolved. BF16 tokenizer file exists; datasets cache is empty and shell network resolution failed, so token arrays and `data_manifest.json` were not generated. Russian eligible-article ordinal interpretation is recorded in QUESTIONS.md.

## 2026-10-16 Stage 10 environment resolution
cmd: `UV_CACHE_DIR=/private/tmp/stage10-uv-cache uv lock --python 3.14.7`; `uv sync --python 3.14.7`
commit: pending
result: OK
facts: managed network escalation allowed PyPI access. Resolved Python 3.14.7, mlx 0.32.4, mlx-lm 0.32.1, numpy 2.5.4, safetensors 0.8.0, datasets 5.1.0, transformers 5.20.0, tokenizers 0.23.3, huggingface-hub 2.2.0. Details in README.md.

## 2026-10-16 A2 test in locked environment
cmd: `uv run python -m unittest discover -s tests`
commit: pending
result: OK
facts: Python 3.14.7; mlx 0.32.4; mlx-lm 0.32.1 from package metadata. 2 tests passed.

## 2026-10-16 MLX device probe
cmd: `uv run python -c 'import mlx_lm'` in default sandbox
commit: n/a
result: FAILED_ERROR
facts: MLX raised `[metal::load_device] No Metal device available. This typically occurs in headless, sandboxed, or virtualized macOS sessions where the GPU is not accessible.` No model weights were loaded.

## 2026-10-16 Stage 10 stack commit
cmd: `git commit -m 'stage10: pin research stack'`
commit: 6022df5
result: OK
facts: `uv.lock`, README resolved versions, and environment notes committed. Post-commit `git status --ignored research/` showed only `.venv/` and Python `__pycache__/` entries ignored.

## 2026-10-16 A2 token arrays
cmd: `uv run python -m stage10.data`
commit: pending
result: OK
wall: approximately 1m23s; peak_mlx: n/a; swap before/after: n/a
facts: Wikitext revision b08601e04326c79dfdd32d625aee71d232d685c3; Wikipedia revision b04c8d1ceb2f5cd4588862100d08de323dccfbaa. Six int32 arrays were written under `benchmarks/results/stage10/data/`, each 16x2048 (eval) or 32x2048 (calibration). Per-file SHA-256 and Python stdlib file lists are in `data_manifest.json`. No model weights were loaded.

## 2026-10-16 A3-loop attempt 1
cmd: `uv run python -m stage10.check_a3 --mode loop`
commit: pending
result: FAILED_ERROR
wall: 35.1s; peak_mlx: not recorded; swap before/after: 0/553.9 MiB
facts: preflight free=88%, swap=0 MiB. Watchdog minimum free sample was 33%; watchdog did not fire. Production checkpoint loaded. Comparison stopped at `AttributeError: module 'mlx.core' has no attribute 'log_softmax'` while scoring normal logits. No threshold result was produced.

## 2026-10-16 A3-loop attempt 2
cmd: `uv run python -m stage10.check_a3 --mode loop`
commit: pending
result: FAILED_ERROR
wall: 35.1s; peak_mlx: not recorded; swap before/after: 545.9/634.4 MiB
facts: preflight free=90%, swap=545.9 MiB. Watchdog minimum free sample was 31%; watchdog did not fire. Normal-call scoring completed. Manual pass stopped at `AttributeError: 'Qwen3_5TextModel' object has no attribute 'args'`; the `TextModel` wrapper owns `args`.

## 2026-10-16 A3-loop pass
cmd: `uv run python -m stage10.check_a3 --mode loop`
commit: pending
result: OK
wall: 115.4s; peak_mlx: not recorded; swap before/after: 610.3/482.3 MiB
facts: preflight free=89%, swap=610.3 MiB. Watchdog minimum free sample was 36%; watchdog did not fire. 2 sequences, 4094 scored positions, max absolute log-probability difference 0.0, argmax agreement 1.0. Result: `benchmarks/results/stage10/checks/a3_loop.json`.

## 2026-10-16 A3-sanitize attempt 1
cmd: `uv run python -m stage10.check_a3 --mode sanitize`
commit: pending
result: FAILED_ERROR
wall: 5.0s; peak_mlx: not recorded; swap before/after: 482.3/482.3 MiB
facts: preflight free=87%, swap=482.3 MiB; watchdog did not fire. Selective read failed on the BF16 source layer with `TypeError: data type 'bfloat16' not understood` in the installed safetensors MLX reader. No tensor comparisons were produced.

## 2026-10-16 A3-sanitize attempt 2
cmd: `uv run python -m stage10.check_a3 --mode sanitize`
commit: pending
result: FAILED_ERROR
wall: 5.0s; peak_mlx: not recorded; swap before/after: 458.3/458.3 MiB
facts: preflight free=87%, swap=458.3 MiB; watchdog did not fire. The source tensor reader loaded BF16 arrays; byte comparison stopped at NumPy conversion with `ValueError: 'bfloat16' is not a valid PEP 3118 buffer format string`. No tensor comparisons were produced.

## 2026-10-16 A3-sanitize attempt 3 and task stop
cmd: `uv run python -m stage10.check_a3 --mode sanitize`
commit: pending
result: FAILED_ERROR; STOP per TASK.md §7
wall: 5.0s; peak_mlx: not recorded; swap before/after: 458.3/458.3 MiB
facts: preflight free=87%, swap=458.3 MiB; watchdog did not fire. After two code fixes, byte comparison still raised `ValueError: 'bfloat16' is not a valid PEP 3118 buffer format string`. No tensor comparisons or dequantized differences were produced. A3-loop result remains 2 sequences, 4094 positions, max absolute log-probability difference 0.0, argmax agreement 1.0.

## 2026-10-16 pre-push rebase
cmd: `git pull --rebase origin main`
commit: no change
result: OK
facts: fetched origin/main; local main was up to date.

## 2026-10-16 Stage 10 checkpoint commits
cmd: `git log --oneline`
commit: 707264c
result: OK
facts: token set builder/tests 33ac42d; pinned research stack 6022df5; token data manifest 7c16904; A3 tools, check log, progress, and STOP question 707264c.

## 2026-10-16 A3-sanitize resume authorized by user
cmd: none
commit: 841fd33
result: OK
facts: The user authorized lifting the A3-sanitize STOP and resuming TASK.md. Fix: `_same_bytes` compared `str(dtype) == "bfloat16"`, but `str(mx.bfloat16)` is `mlx.core.bfloat16`; it now compares dtype objects and views every array as uint8 before NumPy conversion.

## 2026-10-16 A3-sanitize attempt 4
cmd: `uv run python -m stage10.check_a3 --mode sanitize`
commit: 841fd33 + uncommitted check_a3.py fix
result: FAILED_ERROR
wall: 5.0s; peak_mlx: not recorded; swap before/after: 322.25/322.25 MiB
facts: preflight free=86%, swap=322.25 MiB; watchdog did not fire. `ValueError: [quantize] The matrix to be quantized must have at least 2 dimension but it has only 1.` Cause: for keys without `.weight` (e.g. `A_log`), `key.replace(".weight", ".scales")` returns the key itself, which is in the production map. Fix: a tensor is treated as quantized only if its key ends in `.weight` and the `.scales` key exists.

## 2026-10-16 A3-sanitize attempt 5
cmd: `uv run python -m stage10.check_a3 --mode sanitize`
commit: 841fd33 + uncommitted check_a3.py fixes
result: OK
wall: 5.0s; peak_mlx: not recorded; swap before/after: 322.25/322.25 MiB
facts: preflight free=86%, swap=322.25 MiB; watchdog minimum free sample 86%; watchdog did not fire. Layers 0 and 3: 55 of 55 tensors byte-identical to production g128 (mlx 0.32.4 `mx.quantize`, affine, bits 3, group 128). Mismatches: none. Result: `benchmarks/results/stage10/checks/a3_sanitize.json`.

## 2026-10-16 recheck of earlier failures (user report: network outage and failed permission elevation)
cmd: PyPI JSON for mlx and mlx-lm; `uv run python -m stage10.data --output-dir <scratch>/data --manifest <scratch>/manifest.json`; `uv run python -m unittest discover -s tests`; MLX Metal probe `import mlx_lm; mx.metal.is_available()`
commit: c402dbf
result: OK
wall: data rebuild 1m00s; peak_mlx: n/a; swap before/after: n/a
facts: PyPI latest is still mlx 0.32.4 (uploaded 2026-10-14T21:15:37) and mlx-lm 0.32.1 (uploaded 2026-10-15T21:22:28), matching the pins. A2 rebuild into scratch: all 6 arrays have SHA-256 equal to data_manifest.json, dataset revisions equal, and the whole manifest is equal apart from file paths. Unit tests: 2 passed. Metal probe: `import mlx_lm` OK, default device gpu:0, `mx.metal.is_available()` True, device Apple M3 Pro, 1024x1024 matmul OK. Earlier FAILED_ERROR entries for the dependency lock (DNS for pypi.org), the A2 PARTIAL run, and the MLX device probe (sandbox) do not reproduce.

## 2026-10-16 E13/E14 reference downloads
cmd: `uv run hf download prism-ml/Ternary-Bonsai-2-27B-mlx-2bit --revision fcba37d2117a7077eac6b613b2668d14d9779edd --local-dir /Users/artem/models-stage10/E13`; `uv run hf download EschaLabs/Qwen3.8-27B-Escha-W2 --revision 561dae0aeea1efbdafe122f2e385fdeaa99127e7 --local-dir /Users/artem/models-stage10/E14`
commit: 375ac0e
result: OK
wall: about 9m total; peak_mlx: n/a; swap before/after: n/a
facts: both exit 0. E13 repo sha fcba37d2117a7077eac6b613b2668d14d9779edd, model.safetensors 8,595,477,990 bytes, config model_type `prism_hadamard_qwen35`, schema_version 2. E14 repo sha 561dae0aeea1efbdafe122f2e385fdeaa99127e7, two safetensors shards, 10 GiB directory. escha-mlx github.com/EschaLabs/escha-mlx HEAD 69f72187d93d74e281eeb1044f3ef95dfce671de (cloned to a scratch directory for inspection only; pyproject pins `mlx>=0.32.0,<0.40`, `mlx-lm>=0.31.3,<0.32`). No model weights loaded.

## 2026-10-16 A4 BF16 reference (E00)
cmd: `uv run python -m stage10.reference`
commit: 6f2639f (reference.py committed later in 1c7a916 without changes)
result: OK
wall: 21m06s (child 1265.6 s), peak_mlx: 5.08 GiB, swap before/after: 0.31/0.23 GiB
facts: preflight free 85%, swap 314.25 MiB; watchdog minimum free 43%; no SAFETY STOP. 48 eval sequences (en, code, ru; 16 each), sequence batch 4. Outputs in `benchmarks/results/stage10/reference/`: nll [48,2047] float32, top1 [48,2047] int32, top32_ids [48,2047,32] int32, top32_logp [48,2047,32] float32; all finite; top1 is inside top32 at every position. nll_mean all 1.29370 (ppl 3.64624), en 1.87462, code 0.68439, ru 1.32208.

## 2026-10-16 06:10 E01 eval
cmd: `uv run python -m stage10.evaluate --exp E01 --model /Users/artem/models/Qwen3.8-27B-3bit-g128-MLX --loader mlx_lm`
commit: c5b19b1
result: OK
wall: 21.4m, peak_mlx: 10.88 GiB, swap before/after: 0.22/0.90 GiB
facts: all: nll 1.50181, delta 0.20812, top1 0.8123, kl_top32 0.25379; en: nll 2.04591, delta 0.17128, top1 0.7902, kl_top32 0.21099; code: nll 0.93751, delta 0.25313, top1 0.8353, kl_top32 0.33756; ru: nll 1.52202, delta 0.19994, top1 0.8115, kl_top32 0.21282

## 2026-10-16 06:12 E01 gen-check
cmd: `uv run python -m stage10.gen_check --exp E01 --model /Users/artem/models/Qwen3.8-27B-3bit-g128-MLX --loader mlx_lm`
commit: c5b19b1
result: OK
wall: 2.3m, peak_mlx: n/a, swap before/after: 0.89/0.88 GiB
facts: P1: n 384, eos False, distinct4 0.824, max4 4; P2: n 384, eos False, distinct4 0.866, max4 5, has_187 False; P3: n 289, eos True, distinct4 0.997, max4 2; P4: n 188, eos True, distinct4 0.676, max4 6, json_valid True; P5: n 273, eos True, distinct4 0.996, max4 2

## 2026-10-16 06:19 E01 ladder
cmd: `uv run python -m stage10.ladder --exp E01 --model /Users/artem/models/Qwen3.8-27B-3bit-g128-MLX --loader mlx_lm --weights-gib 10.1802`
commit: c5b19b1
result: OK
wall: 7.0m, peak_mlx: 12.99 GiB, swap before/after: 0.88/0.90 GiB
facts: ctx 4096: decode 11.693055049211454, prefill 81.58771953039313 tok/s, ttft 50.52468066700021 s, peak 11.66203604079783 GiB, min free 27.0%, safety_stop False, exit 0; ctx 8192: decode 11.368501664269736, prefill 80.19223347029012 tok/s, ttft 102.49436033300026 s, peak 12.079651927575469 GiB, min free 26.0%, safety_stop False, exit 0; ctx 16384: decode 10.869989332111558, prefill 77.18084924720054 tok/s, ttft 213.28563587500048 s, peak 12.989091014489532 GiB, min free 23.0%, safety_stop False, exit 0
