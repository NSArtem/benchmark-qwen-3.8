# Qwen3.8 overnight run plan

## Safety invariants

- `iogpu.wired_limit_mb` is checked at startup and must be exactly `0`.
- No `sudo`, GPU/wired-memory sysctl modification, swap hack, unrelated-process termination, Qwen3.8 4-bit retry, or custom kernel work.
- Each model run is a separate child process in a private process group.
- Watchdog sampling is approximately once per second. Swap growth by itself is not an abort condition.
- Abort thresholds: free memory ≤8% for 3 consecutive samples; free memory ≤12% with ≥25 page-outs/s or ≥500 swap-outs/s for 3 samples; critical ≥5000 swap-outs/s with pressure for 2 samples; Metal OOM; or 25 seconds of no child progress while pressure persists.
- Practical stage timeout is bounded at 20 minutes; tiny sanity stages are bounded at 8 minutes. A failed stage is recorded and the plan progresses.

## Workload

- Exact prompt copied from `coding_prompt.md` to `coding-prompt.txt`.
- Practical measurement: batch size 1, temperature 0, greedy, thinking disabled, 128 output tokens, prefill step 2048.
- Long-context conclusions require the actual model-formatted prompt to be approximately 15K–17K tokens; tiny sanity generations are not performance results.

## Ordered gates

1. Environment, metadata, watchdog harmless-process test.
2. Uniform 3-bit text-only target: tiny sanity, practical target-only, conditional native MTP, conditional APC.
3. Mixed 3.5 bpw target: metadata/risk gate, tiny sanity, practical target-only, conditional native MTP/APC.
4. 2-bit text-only target: target-only fallback/control; MTP/APC only after coherent quality smoke and only if needed.
5. Deterministic five-task quality smoke for viable candidates.
6. Ranking, final report, archive, Git hygiene verification.

Each completed or failed stage updates `results.json`, regenerates `results.md`, and commits only `qwen38-overnight/` plus the intended `.gitignore` change.
