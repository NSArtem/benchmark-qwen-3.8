# Pi wire contract (OpenAI Chat Completions)

Observed for **pi-coding-agent 1.0.0** (Homebrew `pi-coding-agent` 1.0.0,
`@earendil-works/pi-ai` `openai-completions` provider, openai-node 7.19.0).
Evidence: requests captured on 2026-10-03 from Pi running against a scripted
local endpoint (no model), plus the installed provider source
(`pi-ai/dist/api/openai-completions.js`). A sanitized session is in
`tests/fixtures/pi-1.0.0-requests.json`.

## Configuration path

A plain `models.json` provider with `"api": "openai-completions"` works. No
provider extension and no `compat` overrides are needed. Pi does **not** call
`GET /v1/models` for a configured provider; only a manual probe did.

## Requests

`POST {baseUrl}/chat/completions`, `Authorization: Bearer <apiKey>`,
`User-Agent: pi (<os>; <arch>)`, `x-stainless-timeout: 300`.

Every request body observed had exactly these top-level fields:

| Field | Value |
| --- | --- |
| `model` | the configured model id |
| `messages` | full transcript, every request |
| `stream` | always `true` |
| `stream_options` | `{"include_usage": true}` |
| `store` | `false` |
| `max_completion_tokens` | the model's `maxTokens` (never `max_tokens` for a generic endpoint) |
| `tools` | the active tool set, every request (4 default tools: `read`, `bash`, `edit`, `write`) |

Never sent with the default configuration: `tool_choice`, `temperature`,
`top_p`, `stop`, `parallel_tool_calls`, `response_format`, `n`,
`reasoning_effort`, session headers. `tool_choice` and `temperature` are only
sent if a caller sets them explicitly.

Messages:

- `system` (or `developer` only when the model is marked `reasoning: true`
  and the endpoint is detected as supporting it): one string, first only.
  It is byte-identical across the turns of a session (cwd is included, no
  timestamp).
- `user`: `content` is an **array of text parts**
  (`[{"type":"text","text":"..."}]`); images would add `image_url` parts.
- `assistant` with text: `content` is a string.
- `assistant` with tools: `content` is `null` when there is no text, or the
  text string; `tool_calls[]` = `{id, type:"function", function:{name,
  arguments}}`. `arguments` is **re-serialized** by Pi
  (`JSON.stringify(parsedArgs)`), e.g. `{"path":"hello.txt"}`, not the bytes
  the server streamed. IDs are replayed unchanged.
- `tool`: `{role:"tool", content:<string>, tool_call_id}`; no `name`. Results
  of a multi-call turn are consecutive tool messages in call order.
- A turn the user aborted is dropped: the next request may contain two
  consecutive `user` messages.

Transcripts are append-only between ordinary turns: every request of a
session was an exact prefix extension of the previous one plus the previous
assistant reply. Tool definitions were byte-identical across turns.

## Responses Pi requires

- `Content-Type: text/event-stream`, `data: <json>\n\n` chunks of
  `object: "chat.completion.chunk"`, terminated by `data: [DONE]`.
- A chunk with a non-null `choices[0].finish_reason` is mandatory; a stream
  without one fails with `Stream ended without finish_reason`.
- Mapping: `stop` → stop, `length` → length, `tool_calls` → toolUse; any
  other value is treated as an error.
- Text arrives in `delta.content`. Tool calls arrive in
  `delta.tool_calls[]`, keyed by `index` (falls back to `id`); `id` and
  `function.name` may arrive once, `function.arguments` fragments are
  concatenated and parsed as JSON at the end.
- `reasoning_content` / `reasoning` deltas would become thinking blocks and be
  replayed; the server does not emit them (thinking is off).
- Usage: read from any chunk with `usage` (the final `choices: []` chunk with
  `include_usage`). `prompt_tokens_details.cached_tokens` is read as cache-read
  tokens; context accounting uses `prompt_tokens + completion_tokens`.
- Errors: openai-node raises on an HTTP 4xx/5xx body or on an SSE chunk with
  an `error` object. An error message matching
  `maximum context length is <N> tokens` is classified as context overflow
  and triggers Pi's compact-and-retry.

## Timeouts, retries, cancellation

- `httpIdleTimeoutMs` (default 300 s) bounds the wait for response headers
  and body inactivity. A cold 16K prefill takes ~4 min on this Mac, so the
  server sends SSE headers before prefill and SSE comment heartbeats while
  the model is busy.
- Provider retries default to 0; agent-level retries (3) apply to transient
  failures.
- Ctrl-C / abort closes the TCP connection mid-stream (the endpoint sees a
  broken pipe). Pi does not persist the partial assistant turn.

## Compaction implications

Pi compacts when `contextTokens > contextWindow - compaction.reserveTokens`
(default reserve 16384) and keeps the newest `keepRecentTokens` (default
20000). With a 16K window the defaults never summarize anything, so the
transcript would grow past what the host can serve. The example Pi settings
therefore set `reserveTokens` and `keepRecentTokens` for a 16K window. A
compaction rewrites history, so the next request is a safe fresh prefill.

Observed with the real model (benchmark run 2): as the context approaches
the compaction threshold, Pi lowers `max_completion_tokens` (prompt plus
limit stayed at about 12.3K for a 16K window and reserve 4096: 628, 508 ...
28 tokens). A reply cut by that limit ends with `finish_reason: "length"`;
Pi then compacts once and retries. After that single recovery a second
`length` ends the run. Pi's `keepRecentTokens` is measured with its own
estimate; the compacted transcript in that run still held 11.7K real tokens.
