# Decision log

These decisions came from measurements on an Apple M3 Pro with 18 GB of
unified memory, using MLX 0.32.1 and the pinned mlx-lm revision. Full
evidence is in git under tag `research-stage8-complete`.

## Model: affine 3-bit, group 128 (g128)

**Decision:** g128 is the only supported checkpoint (10.180 GiB, 3.25 bits
per weight).

- On the corrected quality suite it scored 23/24 objective and 4/4
  semi-objective, the same as g64, which failed the same case. On the
  held-out v2 suite it scored 20/21 and 3/3.
- It is the smallest and fastest candidate tested, and it has the most
  long-context headroom: about 10.4–10.6 tok/s and a 15.24 GiB peak at
  16.7K tokens.

## g128 is built from the BF16 source alone

**Decision:** `tools/build_g128.py` takes only `--source`, the BF16
`Qwen/Qwen3.8-27B` snapshot at revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. The old `--baseline` input, an
existing text-only MLX checkpoint, is gone. The tool used to copy that
checkpoint's `config.json`, tokenizer files and `total_parameters`, so a
rebuild depended on a checkpoint that no longer exists locally. Each item is
now derived as follows:

- **`config.json`** is the source config with:
  - `vision_config` removed, as `mlx_lm.utils.save_config` does;
  - `text_config.rope_parameters.rope_type` renamed to `type`, which pinned
    `TextModelArgs.__post_init__` does in place;
  - `eos_token_id` (`[248046, 248044]`) copied from
    `generation_config.json`, as `mlx_lm.utils.load_config` does;
  - `quantization` and `quantization_config` added.

  It also drops `text_config.mtp_num_hidden_layers` and
  `mtp_use_dedicated_embeddings`, because the checkpoint has no MTP
  weights. Pinned mlx-lm ignores both keys. This is the only difference
  from the earlier production config.
- **Tokenizer files** are re-serialized from the source tokenizer through
  transformers `TokenizersBackend`. The output records the
  `tool_parser_type` that mlx-lm infers (`qwen3_coder`). Copying the source
  `tokenizer_config.json` unchanged would be wrong: it names
  `Qwen2Tokenizer`, and pinned transformers 5.15 then replaces the
  pre-tokenizer regex with one that lacks `\p{M}`, which changes Devanagari
  and Thai tokenization. The resulting `tokenizer.json`,
  `tokenizer_config.json` and `chat_template.jinja` are byte-identical to
  the production files. `generation_config.json` is copied unchanged.
- **`total_parameters`** (26,895,993,856) is pinned mlx-lm
  `get_total_parameters`, the value `save_model` writes. That function
  counts leaf modules only, so it leaves out the 4,608 `A_log`/`dt_bias`
  values held directly by the 48 GatedDelta blocks. The full source count is
  26,895,998,464.

**Validation (2026-10-02):** a source-only rebuild took 33 s with a 3.33 GiB
MLX peak. All 19 shards, the index and the tokenizer files were
byte-identical to production, covering all 1,847 tensors. Both configs
produce identical pinned `TextModelArgs` and parameter trees.

## The manifest is the oracle, not a second checkpoint

**Decision:** the BF16 source is the authoritative upstream asset that must
be recoverable. The g128 checkpoint is a reproducible artifact derived from
it. Its identity is `tools/g128-reference-manifest.json`, which records the
size and SHA-256 of all 25 files, the source revision, the quantization and
the pinned toolchain. Historical or alternate MLX checkpoints no longer need
to be kept as comparison oracles. A default build verifies itself against
the manifest before it is published, and `--verify-output` verifies any
existing directory.

- **Canonical `config.json`:** the source-only builder's output, without
  the MTP keys. The manifest was generated from the validated 2026-10-02
  rebuild, which matches production byte for byte in every other file. If
  the older production config were canonical, every correct future build
  would fail. The production file was left unmodified. The manifest lists
  its exact hash as a `legacy_variants` entry, and the generator asserted
  that it differs from the canonical config only by
  `mtp_num_hidden_layers: 1` and `mtp_use_dedicated_embeddings: false`.
  Plain verification rejects that variant by name, and `--allow-legacy`
  accepts it.
- **Strictness:** any visible entry not in the manifest is a failure, and
  hidden entries (`.DS_Store`) are ignored.
- **Toolchain:** a unit test requires the manifest's toolchain (Python,
  mlx, mlx-lm revision, transformers, tokenizers) to match `uv.lock`. An
  upgrade therefore forces the manifest to be revalidated and regenerated
  deliberately.

## Agent integration: standard OpenAI Chat Completions, no client adapters

**Decision:** the server implements the OpenAI Chat Completions subset that
real clients send, rendered only through the model's own chat template. Pi is
configured with a plain `models.json` provider (`api: "openai-completions"`);
there is no Pi extension, no `compat` override, and no client-specific
dialect. Evidence: `docs/pi-wire-contract.md` (Pi 1.0.0 capture and source).

- **Tool syntax comes from the template.** Qwen3.8's template defines
  `qwen3_coder` XML calls and `<tool_response>` results. The only glue is
  parsing OpenAI's JSON-string `arguments` into the mapping the template
  iterates.
- **Generated IDs stay authoritative.** Parsed `tool_calls` are an API view.
  A replayed assistant message is matched semantically (content trimmed,
  tool calls by id, name and parsed arguments) only to find its stored
  generation; the cache then uses the stored token IDs. Re-serializing the
  parsed call to recover tokens was rejected: clients re-serialize arguments
  and the model's own whitespace and value spelling would not survive.
- **Tools are part of the prompt identity.** Any change of the exact tool
  JSON (content, order, presence) or `tool_choice: "none"` forces a fresh
  prefill, before the token-level prefix check that would also catch it.
- **Malformed tool syntax is text, not a call.** A truncated or malformed
  `<tool_call>` returns the raw text with `finish_reason` `stop`/`length` and
  is counted (`tool_parse_failures`). A well-formed call to an undeclared tool
  or with a value that does not fit the schema type is returned as a call and
  counted, so the client's validator tells the model what was wrong.
- **Unenforceable options are rejected.** `tool_choice: "required"` or a named
  function, `stop`, `n > 1`, and non-text `response_format` get 400 rather
  than being silently ignored. `parallel_tool_calls` is accepted unenforced.
- **Streaming is required.** Pi always sends `stream: true` and fails without a
  `finish_reason` chunk. Headers and the role chunk are sent before prefill
  and `: keep-alive` comments every 10 s, because Pi's default 300 s idle
  timeout is close to a 16K cold prefill.
- **Cancellation discards the cache.** The hybrid cache cannot be trimmed, so
  a disconnect or error after generation starts invalidates the session.
  Pi drops an aborted turn from its transcript, so nothing reusable is lost.
- **A server-side context cap.** `--max-context-tokens 16384` (prompt plus
  completion) protects the 18 GB host and returns OpenAI's
  `context_length_exceeded` wording, which Pi treats as overflow and answers
  by compacting. Pi's default compaction settings assume large windows
  (reserve 16384, keep 20000), so the example settings use reserve 4096 and
  keep 6144 for a 16K window.
- **Repeated tool calls are not stale replay.** A token-identical tool call
  (re-running the tests after an edit) is legitimate and reported as
  `repeated_tool_call`; `replay_detected` flags token-identical prose only.
  The real gate is the per-request cache offset/prefix invariant.

## g64 is not the default

The original `lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly` uses group size
64. It is 0.78 GiB larger (10.96 GiB) and showed no quality advantage on the
corrected suites. Its larger footprint costs long-context speed on 18 GB: the
fresh ladder dropped to 7.7 tok/s at 12K, and a 16.7K prompt decoded at about
4.1 tok/s with a 17.2 GB peak.

## mixed_3_4 rejected: long-context cliff

The mlx-lm `mixed_3_4` recipe at g128 (10.66 GiB) scored 24/24 on the Stage
4 suite. In fresh processes, however, it decoded at 4.9–5.7 tok/s at 16.7K,
against 10.4–10.6 for g128, while matching g128 at 8K. Its peak was 0.485
GiB higher, matching its larger payload. The cliff is real and reproducible.
A small quality edge does not justify halving throughput at the target
context.

## mixed_2_6 and uniform 2-bit rejected

mixed_2_6 at g128 saved memory but fell into an identical repetition loop in
two fresh short runs. Lower-bit uniform quants were not pursued further.

## KL/loss-guided frontier (Q1/Q2/Q3) abandoned

A sensitivity-guided 3/4-bit frontier produced three checkpoints of
10.33–10.62 GiB. Q1, the smallest, scored below g128 on both suites (22/24
and 17/21). It was also slightly slower, at 10.26 versus about 10.4 tok/s at
16K, and heavier. The pre-registered gate therefore skipped Q2 and Q3.

## DWQ abandoned

Bounded, diversity-aware DWQ improved its own calibration objective. The
four-module candidate still lost one case on each practical suite, and its
held-out teacher NLL got worse by 0.003 nats/token. It never produced a
better complete model.

## MTP and speculative decoding: not in production

Qwen3.8 ships 15 `mtp.*` tensors. An exact greedy MTP prototype was built,
including a 4-bit draft head of 0.21 GiB with about 43–68% acceptance. It
was rejected for production for these reasons:

- **Batched verification is not exact.** Verifying 2 or more drafted tokens
  in one target call diverges at layer 0 in the recurrent (GatedDelta)
  state. The fused sequence update gives slightly different numbers than
  token-by-token decode.
- **Exact verification costs one target forward per token.** The exact
  transactional verifier copies only recurrent state, shares the KV history,
  and trims KV on rejection. It still runs one target forward per candidate
  token, so it cannot reduce target compute.
- **It is slower.** Measured speed was 0.47× baseline at 4K (5.5 vs 11.7
  tok/s) and 0.32× at 8K (3.6 vs 11.1 tok/s).
- **It needs more memory.** Peak memory was 1.47 GiB higher at 8K, and the
  16K run hit the memory safety stop.

Reopening MTP would require an exact fused multi-token GatedDelta kernel. That
would be new kernel work, not tuning.

## KV cache quantization: optional only

KV8 (8-bit KV cache, `--kv-bits 8`) was correct at 16K. It saved only 0.47
GiB of peak memory and cut decode from 10.5 to 9.5 tok/s. The flag is still
available, but the default uses normal KV.

## GGUF / llama.cpp not maintained

In the llama.cpp branch, the conventional `Q3_K_M` GGUF (12.9 GiB) failed to
load because of Metal OOM. `UD-IQ3_XXS` ran up to 32K, but decoded at only
about 4.4 tok/s at 16K, against about 10.5 for MLX g128. MLX direct
generation is the maintained runtime.

## DSpark and Qwen3.5 experiments

The mlx-dspark smoke test could not finish a 27B run without uncontrolled swap
growth. The Qwen3.5 MTP and DFlash smoke tests covered a different model
family. Neither produced anything this runtime uses.

## Context recommendation: 12K–16K

| Context | Decode speed | Peak memory | Verdict |
| ---: | ---: | ---: | --- |
| 8K | 11.1 tok/s | 14.0 GiB | comfortable |
| 12K | 10.8 tok/s | 14.6 GiB | usable |
| 16K | 10.4–10.6 tok/s | 15.2 GiB | practical upper end for daily use |
| 20K | 10.37 tok/s | 15.86 GiB | worked once on a clean host; only 23% memory free afterwards |

24K and above were not attempted on 18 GB.
