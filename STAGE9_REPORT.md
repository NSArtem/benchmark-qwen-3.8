# Stage 9 report: OpenAI tool calling, streaming and Pi integration

Date: 2026-10-03. Host: M3 Pro, 18 GB, macOS 15.8.1. Model: Qwen3.8-27B g128
(`/Users/artem/models/Qwen3.8-27B-3bit-g128-MLX`), MLX 0.32.1, pinned mlx-lm.
Raw results are in the git-ignored `benchmarks/results/`.

**Verdict: INTEGRATION WORKS BUT MODEL/HARNESS LIMITATION REMAINS.**
Pi drives the local model through its standard OpenAI provider and completes
real repository work with zero protocol errors, zero invalid tool calls and
exact cache reuse on every continuation. 6 of 7 benchmark tasks pass in each of
two full runs. The remaining failure is the model's own reasoning under greedy
decoding (a repetition loop), compounded by a 16K window that leaves Pi
little room to compact.

## Answers

1. **Pi version:** pi-coding-agent 1.0.0 (Homebrew; `@earendil-works/pi-ai`
   `openai-completions`, openai-node 7.19.0).
2. **Standard `models.json` sufficient?** Yes. No provider extension, no
   `compat` overrides, no Pi patch. Pi's compaction settings must be set for a
   16K window (`integrations/pi/settings.example.json`); the defaults never
   compact a 16K window.
3. **Fields Pi requires:** `model`, `messages` (system string; user text-part
   arrays; assistant `content` null/string + `tool_calls`; `tool` with
   `tool_call_id`), `tools`, `stream: true`, `stream_options.include_usage`,
   `store: false`, `max_completion_tokens`. Never sent by default:
   `tool_choice`, `temperature`, `stop`, `parallel_tool_calls`,
   `response_format`. See `docs/pi-wire-contract.md`.
4. **Streaming required?** Yes. Pi always streams and fails without a
   `finish_reason` chunk. Headers go out before prefill, with heartbeats every
   10 s (Pi's idle timeout is 300 s, about one cold 16K prefill).
5. **Qwen's native tool format:** from the shipped template, `qwen3_coder` XML:
   `<tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n</function>\n</tool_call>`,
   strings verbatim and other values as JSON, several calls per turn. Tools
   render as `tojson` lines in the system block; results as `<tool_response>`
   blocks in one user turn. IDs are not in the prompt. `<tool_call>` is one
   non-special added token (248058). The only glue needed: OpenAI's JSON-string
   `arguments` are parsed into the mapping the template iterates.
6. **Parsing without compromising token truth:** the decoded text is parsed
   into an API view (content + typed `tool_calls`, `call_<hex>` IDs). The
   session keeps the generated IDs and the view it returned. A replayed
   assistant message (Pi re-serializes arguments, sends null content) is
   matched semantically only to identify its stored generation; its rendered
   span is then replaced by the stored IDs. Nothing is re-serialized to
   recover tokens. Malformed or truncated syntax returns as text, never as a
   call.
7. **Tool schema / cache identity:** the exact tool JSON (content and order)
   is part of the session identity. Changed, reordered, added or removed tools
   or `tool_choice: "none"` force a fresh prefill (`tool_set_changed`); the
   token-level prefix check is the final gate.
8. **Cancellation:** a write failure stops generation at the next token or
   heartbeat, and any interruption after generation starts discards the
   session (the hybrid cache cannot be trimmed). Pi drops aborted turns, so
   its next request is rebuilt by a fresh prefill.
9. **Tests:** 133 unit/protocol/tokenizer/HTTP tests pass (1 skipped: BF16
   source not present); `qwen38-server --self-test` passes. New: protocol
   matrix, template-format pins, a parser test that inverts the real template
   on Pi's own `edit` schema, session cache identity with tools, SSE text/tool
   streaming, Unicode, disconnect, heartbeat, overflow and mid-stream errors.
10. **Real-model tool smoke** (`benchmarks/tool_smoke.py`, final code):
    `PASS_TOOL_SMOKE`, 19/19 checks: native calls, results used, exact reuse
    on every continuation (22–35-token suffixes, ~0.6 s TTFT), schema change
    falls back, the model emits two parallel calls, no parse failures.
11. **Pi benchmark** (7 tasks, hidden evaluators, `benchmarks/pi_agent.py`):

    | Task | Run 1 | Run 2 |
    | --- | --- | --- |
    | single-file bug | pass | pass |
    | multi-file bug | pass | pass |
    | two stacked bugs | fail | fail |
    | search/read across files | pass | pass |
    | long command output | pass | pass |
    | early constraint | pass | pass |
    | three-prompt session | pass | pass |

    Run 1 used the code before the reconciliation fix; both runs are scored
    with the final evaluators. With `--mode sample` (temperature 0.7, top-p
    0.9) the failing task passed 2 of 2 (anecdotal, n=2).
12. **Tool calls (run 2):** 108 emitted, 0 invalid (schema or unknown tool),
    17 error results from valid calls (failing test commands, `edit` text not
    found), 2 parse failures (calls cut by Pi's output budget). Run 1: 97
    calls, 0 invalid, 0 parse failures.
13. **Protocol errors:** 0 HTTP errors, 0 409s, 0 stream errors, 0
    cancellations in both runs; stale replay 0 (two repeated, budget-cut tool
    calls are classified as repeats, not replay).
14. **Max context:** 12,306 tokens (run 2), 12,203 (run 1); the Stage 8
    benchmark reached 10,751.
15. **Representative latency (run 2):** warm TTFT median 1.4 s, p95 7.4 s,
    max 54 s (a 3.5K-token log result); cold first Pi turn ~22 s (~1.7K
    tokens); fresh prefill after compaction 156 s (11.7K tokens); decode
    median 11.6 tok/s; task wall time 66–320 s for passing tasks; MLX peak
    14.3 GiB.
16. **Failure taxonomy:**
    - Two stacked bugs, run 1: **MODEL_REASONING**. It made the round-trip test
      pass by making the day constant consistently wrong in both directions
      (`parse_duration('2d')` = 10,368,000). Caught by the semantic evaluator.
    - Same task, run 2: **MODEL_REASONING**. A greedy repetition loop (the
      same `python3 -c` probe with identical results); then **CONTEXT_LIMIT**:
      Pi shrank `max_completion_tokens` near its threshold, compacted once
      (the kept tail was still 11.7K tokens) and stopped at the second
      `length`.
    - **FIXTURE/EVALUATOR** issues found and fixed before they could count:
      the constraint evaluator compared `inspect.signature` text (Python 3.14
      prints `Optional[str]` as `str | None`); the long-session evaluator
      wrongly rejected lazy TTL expiry; the durations evaluator only ran the
      round-trip tests.
    - Server bugs found by the benchmark and fixed:
      `reconcile_canonical_history` shifted later assistant spans by earlier
      length changes and reused a stale EOS index (latent since Stage 8;
      tool-call turns triggered it). The effect was only spurious fresh
      fallbacks: 1 in run 1, 0 in run 2.
17. **Hermes:** Hermes Agent v0.21.5+6175.ga5e7df2 (upstream `a5e7df27`),
    installed with the official installer into a disposable HOME. Chat, a tool
    call with its result, and a small fix-the-test task passed via
    `provider: custom` with no Hermes-specific server code, with exact reuse
    between turns. One standards-level fix: the 400 for an unsupported
    `response_format` now says "is not supported", so Hermes's title
    generator retries without it. Not usable in production here: Hermes
    requires at least 64K context (the smoke declared 65,536 while the server
    enforced 16K), and its side requests replace the single session's cache.
18. **Production startup:**
    `uv run qwen38-server --model /Users/artem/models/Qwen3.8-27B-3bit-g128-MLX --host 127.0.0.1 --port 8000 --model-name qwen38-local`
    (defaults: greedy, `--max-context-tokens 16384`).
19. **Pi configuration and launch:** copy `integrations/pi/models.example.json`
    to `~/.pi/agent/models.json` and merge `integrations/pi/settings.example.json`
    into `~/.pi/agent/settings.json` (or set `PI_CODING_AGENT_DIR` to a
    directory holding both). Then run `pi --provider qwen38-local --model qwen38-local`,
    or `pi -p --provider qwen38-local --model qwen38-local "<task>"`.
20. **Remaining blockers for normal local coding:** model reliability on
    harder debugging (greedy loops, test-gaming fixes): sampling looked better
    in a 2-run probe but was not benchmarked. The 16K window: Pi's compaction
    at 16K keeps too much and recovers once. Speed: compactions and large tool
    outputs cost minutes. Not protocol or cache correctness.

## Other measurements

- **Canonical model gate (Phase 1):** production and the source-only rebuild
  produced identical token IDs on four greedy prompts and a two-turn chat.
  After the generation-loop extraction, production tokens were identical
  again. Both checkpoints are kept; production stays the path above.
- **Interleaving:** an orphaned Pi process from an aborted launch shared the
  server with a measured run. The server served both sessions by fresh
  prefill on every request (19/19), with no errors; both completed.
- **Host memory:** the first model load evicted resident apps to swap (+4 GB);
  after that, real runs grew swap by at most ~2 GB with free memory never
  below 20%. No watchdog stop fired after the first.

## Notes for the user

- `TRIMMING_REPORT.md` (untracked, from the previous stage) was present at
  session start and gone before the macOS-update restart. No command in this
  session touched it, and it is not in the Trash. It was never tracked.
- Hermes was installed only under a scratch HOME; `~/.hermes`, shell rc files
  and `~/.local/bin` were not modified. Pi ran with disposable agent
  directories; `~/.pi` was never created.
