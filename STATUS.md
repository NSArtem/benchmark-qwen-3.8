# STATUS: runtime upgrade to mlx-lm v0.32.0 (work in progress)

Branch `upgrade/mlx-lm-0.32.0`, WIP commit `fa7448b` (pushed). **Not validated;
do not merge.** The docs (README, AGENTS.md, docs/decisions.md,
docs/performance.md, the pyproject comment on tokenizers/transformers) still
describe the old pins on purpose. Update them only after validation passes.

Last updated 2026-10-16.

## Target versions

| Component | Old (validated) | New (branch) |
| --- | --- | --- |
| Python | 3.14.7 | 3.14.8 (uv-managed) |
| mlx / mlx-metal | 0.32.1 | 0.32.3 |
| mlx-lm | `254d153` (v0.31.3+24) | `a9bd8af` (v0.32.0) |
| transformers | 5.15.0 | 5.19.0 |
| tokenizers | 0.22.2 | 0.23.3 |
| huggingface-hub (transitive) | 1.27.0 | 2.2.0 (httpx replaced by httpx2) |

Python 3.15 is blocked because mlx and sentencepiece ship no cp315 wheels.
The stack has to move together: mlx-lm 0.32.0 needs `mlx>=0.32.2`, and
transformers 5.19 needs `tokenizers>=0.23.1`. `uv_build` is still `<0.13`;
0.13.0 exists but is untouched.

## Done on the branch

- **Pins and lock.** `uv lock` was run without `--upgrade`; only the packages
  above changed.
- **`tools/build_g128.py`.**
  - mlx-lm's `_infer_tool_parser` now takes the tokenizer object, not the
    template string. The call site is updated.
  - transformers 5.19 loads this checkpoint as `Qwen3_5Tokenizer` instead of
    `TokenizersBackend`. The old class-name check is replaced by a direct
    comparison against the source `tokenizer.json`: normalizer, split regex
    (with `\p{M}`), vocab, merges, added tokens and post-processor. The
    existing probe encodings are kept. Only ByteLevel offset and decoder
    flags differ.
  - Checked against the checkpoint's own `tokenizer.json`: 0 of 3,059 texts
    gave different token IDs, and HF `decode` matched on all 3,005 sequences
    (with and without skipping special tokens). Scripts:
    `benchmarks/results/upgrade-0.32/scripts/{tokcmp,deccmp}.py`.
- **Rebuild and manifest.** A source-only rebuild under the new toolchain
  (33.6 s, 3.33 GiB peak) is byte-identical to the manifest in 24 of 25
  files. The exception is `tokenizer_config.json`: transformers 5.19
  `save_pretrained` adds an `extra_special_tokens` list. That changes
  `all_special_ids` only, which nothing reads. Encode, decode, chat template,
  EOS IDs and tool parser are identical.
  - The manifest was regenerated with `calculate_manifest`, with the new file
    as canonical. The old production file is now a second `legacy_variants`
    entry, following the `config.json` precedent.
  - The production checkpoint passes `--verify-output --allow-legacy`, which
    it already needed for `config.json`.
- **`benchmarks/long_agent.py`.** After an expected tool request and its
  protocol retry both miss (cut-off or malformed, with no valid request for
  another file), the harness supplies the expected fixture itself, at most
  once per run (`MAX_HARNESS_TOOL_INJECTIONS = 1`).
  - Raw invalid and missing counts are unchanged. New fields:
    `harness_tool_injections` (with the model's attempts),
    `recovered_invalid_tool_calls`, `recovered_missing_tool_requests`, and
    `injected_tool_after_invalid`, which was previously hard-coded `False`.
  - A second miss still fails the run. Covered by 2 new unit tests.
- **Small fixes.** The legacy-variant test is updated, and
  `PINNED_MLX_LM_REVISION` in `chat.py` is bumped (the constant is unused).

## Test results (new stack)

| Check | Result |
| --- | --- |
| Unit tests | 135 OK, 1 skipped |
| `qwen38-server --self-test` | pass |
| `QWEN38_G128_VERIFY` real-checkpoint test | pass |
| `tool_smoke.py` | PASS, decode 11.5–11.9 tok/s (old 11.4–11.6) |
| `long_agent.py`, run 1 (before harness fix) | FAIL: tool request cut off at turn 18, so the long result was never injected |
| `long_agent.py`, run 2 (after harness fix) | **Stopped manually**: stalled in the long prefill (see below) |

The old-stack baseline on the same day (`old/long-agent.json`):
`PASS_LONG_SERIALIZED_AGENT`, 10,751 tokens, median decode 11.28 tok/s,
peak 14.01 GiB, 101 s for the long-result prefill, 284 s in total.

## Findings

1. **Greedy outputs change (expected, accepted).** The first difference is at
   turn 3 of `long_agent`, on an identical prompt, so it comes from the
   numerics (the new packed GatedDelta kernel, mlx-lm #1559, and mlx
   changes). Per-token logprob margins differ by about 0.1–1.5 between the
   stacks.
2. **The turn-18 failure is not specific to the stack.** After `_dependency`,
   EOS beats `_audit` by 0.125 on the new stack. The **old stack fails the same
   way on the same transcript** (by 0.625), and both stacks emit the full path
   on the old transcript. The benchmark was fragile to any change in
   wording, and the harness fix addresses that. Evidence:
   `turn18-probe-0.32.{1,3}.json`, script `scripts/probe18.py`.
3. **Open blocker: possible long-prefill regression.** In run 2 the harness
   injection worked. The next request then prefilled from about 2.2K to about
   10.7K tokens and made no progress for about 8 minutes (old stack: 101 s).
   - Swap reached 3.4 GB used with 690 MB free. The server was in
     uninterruptible wait at 9% CPU, which looks like paging.
   - It was stopped under the AGENTS.md safety rule. The 11 HTTP errors in
     `new/long-agent-v2.json` come from that interrupt.
   - This was the first prompt over about 2.3K tokens the new stack has run,
     so **the 12–16K target is unvalidated**.
   - The prefill loop in `generate_step` is essentially unchanged. Suspects:
     the packed GatedDelta prefill kernel (#1559) and the per-thread
     `generation_stream` (#1775/#1840), since the server generates on
     per-request threads.
   - Confound: the old baseline started at 0 swap, while this run started at
     about 2.4 GB of swap left by earlier runs.

## Next steps

1. Start from a clean host: reboot, or at least close other apps, so swap is
   near 0. Check `memory_pressure -Q` and `sysctl vm.swapusage`.
2. Run a controlled long-prefill A/B. Use one fresh process per run, guarded
   by `tools/memory_watchdog.sh`, and never two at once.
   - Call `generate_step` directly on the exact ~10.7K-token transcript from
     `new/long-agent-v2.json` (messages up to the `tool_result_harness` turn).
   - Time each 2048-token prefill chunk and record MLX peak and swap.
   - New stack first. Then the old stack in a temporary worktree:
     `git worktree add --detach <scratch>/oldstack main && uv sync --locked`.
3. If prefill is the same on a clean host, re-run `long_agent.py`, then the
   12K/16K checks from docs/performance.md. Then update the docs and pins and
   split `fa7448b` into focused commits.
4. If the new stack is clearly worse, isolate it: try mlx 0.32.3 with the old
   mlx-lm `254d153`, or mlx-lm with the packed kernel disabled. Otherwise
   keep the old pins.

## Artifacts (git-ignored, `benchmarks/results/upgrade-0.32/`)

- `old/`: tool-smoke and long-agent baseline on the old stack.
- `new/`: tool smoke, long-agent run 1 (`long-agent.json`) and run 2
  (`long-agent-v2.json`, interrupted), with server logs.
- `build.log`, `build-metrics.jsonl`: the passing source-only rebuild.
- `turn18-probe-0.32.{1,3}.json`, and in `scripts/`: `probe18.py` (turn-18
  logprob replay), `tokcmp.py` and `deccmp.py` (tokenizer equivalence).
