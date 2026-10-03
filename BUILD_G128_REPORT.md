# Source-only g128 build report (2026-10-02)

This report records the source-only rebuild validation. The maintained
summary is in `docs/decisions.md`, and the code is in commits 27b6812,
3d326c7 and 57ea694. Raw evidence and validation scripts are in the
git-ignored `benchmarks/results/g128-rebuild/`, which exists only on the
machine that ran the build.

**Status:** the source-only build is proven. It is byte-identical to
production in every weight shard, the index and the tokenizer files. One
item is still open: the real-model greedy and two-turn runtime test hit the
memory safety stop and has not been run (see #18–19).

## Answers

1. **Why did the old builder require `--baseline`?** It took the
   text-only config, the tokenizer files and the index metadata from an
   existing MLX checkpoint instead of deriving them. The model class and the
   quantization keys were also built from that config.
2. **What was taken from the baseline?** `config.json`, which was also used
   to construct the model and find the quantization keys; `tokenizer.json`,
   `tokenizer_config.json`, `chat_template.jinja`, `generation_config.json`
   and `special_tokens_map.json` (copied if present); and
   `metadata.total_parameters`.
3. **How is each item derived now?** The config comes from
   `derive_text_config(source config, source generation_config)`. The model
   and the quantization keys come from that config and pinned mlx-lm. The
   tokenizer files come from `TokenizersBackend.from_pretrained(source)` →
   `save_pretrained`. `generation_config.json` is copied from the source.
   `total_parameters` is pinned `get_total_parameters` on a model skeleton
   without weights.
4. **Source revision:** `Qwen/Qwen3.8-27B` @
   `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, pinned in the tool. The tool
   checks six metadata file hashes against the Hub tree.
5. **How is `config.json` derived?** The source has a nested `text_config`,
   and pinned mlx-lm keeps it nested, so nothing is flattened. The changes:
   - removed `vision_config`;
   - renamed `rope_parameters.rope_type` to `type`;
   - added `eos_token_id` `[248046, 248044]` from `generation_config.json`;
   - added `quantization` and `quantization_config`;
   - removed `mtp_num_hidden_layers` and `mtp_use_dedicated_embeddings`.

   Each edit except the last mirrors pinned mlx-lm's own conversion
   behavior. The MTP removal is deliberate; the checkpoint has no MTP
   weights. No production value is underivable.
6. **Where do the tokenizer and chat-template files come from?** From the
   BF16 source snapshot, re-serialized through pinned transformers
   `TokenizersBackend`, with mlx-lm's inferred `tool_parser_type`. Verbatim
   copying would be wrong: the source `tokenizer_config.json` says
   `Qwen2Tokenizer`, and transformers 5.15 then drops `\p{M}` from the
   pre-tokenizer regex. That changed the tokenization of Hindi and Thai
   probes.
7. **How is `total_parameters` computed?** With pinned
   `mlx_lm.utils.get_total_parameters`, the same function `save_model` uses:
   26,895,993,856, equal to production. It skips 4,608 non-leaf
   `A_log`/`dt_bias` values; the full source count is 26,895,998,464. The
   tool asserts this exact relation.
8. **Does `--self-test` work with source only?** Yes:
   `SOURCE_SELF_TEST_PASS` in about 3 s, without loading weights.
9. **Command (no `--baseline`):**
   `uv run python tools/build_g128.py --source "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0" --output /Users/artem/models/Qwen3.8-27B-3bit-g128-MLX-rebuild-test --metrics benchmarks/results/g128-rebuild/build-metrics.jsonl`
10. **Wall time:** 33.1 s.
11. **Peak memory:** MLX peak 3.33 GiB, max RSS 3.42 GiB, active+cache at
    most 0.51 GiB between chunks. The full BF16 model was never resident.
12. **Shards:** 19.
13. **Payload:** 10,930,715,648 bytes (10.180 GiB).
14. **Tensors and modules:** 1,847 tensors (BF16 1,349, U32 498) from 851
    source tensors; 498 quantized modules.
15. **Metadata vs production:** `tokenizer.json`, `tokenizer_config.json`,
    `chat_template.jinja`, `generation_config.json` and
    `model.safetensors.index.json` are byte-identical. `config.json` differs
    only by the two removed MTP keys, an intentional improvement. Pinned
    mlx-lm ignores them; both configs give identical `TextModelArgs` and
    parameter trees. There are no unexplained differences.
16. **Tensor inventories identical?** Yes. Names, shapes, dtypes, weight map
    and shard metadata all match.
17. **Tensor contents identical?** Yes. All 1,847 tensors (10,930,715,648
    bytes) match byte for byte, and so do all 19 shard files. The
    conversion is deterministic.
18. **Greedy outputs identical?** **Not run empirically.** Loading the
    production model for the first pass drove free memory to 10%, and macOS
    grew swap from 6 to 10 GB. The watchdog stopped the process before
    generation. Because the weights and tokenizer are byte-identical and
    the configs produce identical models, the runtime gets identical
    inputs. The empirical check is still owed.
19. **Does two-turn persistent chat pass?** **Not run**, for the same
    reason.
20. **Any remaining dependency on an existing MLX checkpoint?** None. The
    tool accepts only `--source`. A unit test asserts it contains no
    `baseline`, `lukaskremla`, `MLX-TextOnly` or `/Users/artem/models`
    reference.
21. **Could production g128 be deleted and rebuilt from the repo plus the
    BF16 source?** **Yes.** The rebuild matches production in all weights,
    the index and the tokenizer files. The only difference is the
    deliberately dropped MTP config keys.

## Source snapshot change (please note)

The local HF snapshot held only `config.json`, the index and the 18 weight
shards. `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` and
`generation_config.json` had never been downloaded, so the source was not
self-sufficient. I added these four files from the same revision with
`hf download`. All four were verified against the Hub tree hashes. No
existing file or blob changed.

## To finish the runtime check (healthy host only)

Close Firefox, or reboot, and check `sysctl vm.swapusage` first. Then run
these one at a time, never concurrently:

```sh
R=benchmarks/results/g128-rebuild
uv run python $R/runtime_equivalence.py /Users/artem/models/Qwen3.8-27B-3bit-g128-MLX $R/runtime-production.json
uv run python $R/runtime_equivalence.py /Users/artem/models/Qwen3.8-27B-3bit-g128-MLX-rebuild-test $R/runtime-rebuild.json
```

Each run does four greedy prompts (coding, reasoning, prose, structured; 96
tokens each) and then a two-turn exact-suffix chat. Compare `token_ids`
across the two JSON files, and check `two_turn.checks` in the rebuild file.

## Disk

Both copies remain: production `Qwen3.8-27B-3bit-g128-MLX` and
`Qwen3.8-27B-3bit-g128-MLX-rebuild-test` (10.2 GiB each). Deleting either
one is your call.
