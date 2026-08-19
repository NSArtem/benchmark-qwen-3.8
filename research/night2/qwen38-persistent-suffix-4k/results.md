# Qwen3.8 persistent exact-suffix diagnostic — 4K

Single g128 run using MLX 0.32.1, pinned mlx-lm revision
`254d153fdeb6f150edd4fc5a54f9828638481fa8`, direct `generate_step`, greedy
decoding, and untouched MLX wired state.  The initial canonical prompt was
4,096 tokens; turn 1 generated 96 tokens and turn 2 generated 64 tokens.

1. **What exact token stream was supposed to be represented by the cache?**

   Turn 1 was the canonical 4,096-token user chat-template stream followed by
   the exact 96 yielded turn-1 token IDs.  The expected cached prefix was thus
   4,192 tokens, SHA256
   `2090a312fe5608b6c2ce9f351d50354b952e28f753a6e683151b57d935a704cf` for
   the generated portion.  The complete canonical turn-2 transcript was
   4,230 tokens, SHA256
   `4d6f9c0ce1f5d2a01aa754bb48705049082dad6eda0651e9b13a52df5d671d64`.

2. **What prefix was actually cached after turn 1?**

   The 16 KVCache layers all reported the same offset: 4,192.  This exactly
   matched the 4,096 prompt tokens plus the 96 yielded tokens.  The 48
   ArraysCache recurrent layers expose no logical offset; their compact state
   signatures were saved before and after each stage.  The pinned
   `generate_step` progress callback is not a pure prompt-only boundary: its
   4,097 offset includes the first generated token because the first next-step
   has already been scheduled.  The corrected labels and raw traces preserve
   this fact.

3. **What exact suffix was supplied for turn 2?**

   The harness supplied `full_turn2_tokens[4192:]`: exactly 38 tokens, with no
   new BOS and no independently templated follow-up.  Its SHA256 is
   `206a157e360c2c73f638f8a9b9ab6af1879ed0ebf7d2754d9b871bceed2f729f`.
   The first 32 IDs were:

   ```text
   [248046, 198, 248045, 846, 198, 22365, 3074, 55687, 494, 678, 3555,
   22267, 1220, 279, 5165, 27247, 1118, 11, 321, 3069, 30, 21134, 303,
   220, 18, 12, 20, 22157, 13, 248046, 198, 248045]
   ```

   The last 32 IDs were:

   ```text
   [3074, 55687, 494, 678, 3555, 22267, 1220, 279, 5165, 27247, 1118, 11,
   321, 3069, 30, 21134, 303, 220, 18, 12, 20, 22157, 13, 248046, 198,
   248045, 74455, 198, 248068, 271, 248069, 271]
   ```

   The decoded boundary is `<|im_end|>\n<|im_start|>user ... <|im_end|>\n`
   followed by the assistant generation prompt.  Both tokenizer decode
   variants reproduced the expected 4,192-token prefix exactly.

4. **Did persistent turn 2 answer the new question correctly?**

   Yes, semantically in this 4K run.  It answered the new invariant question
   with atomic enforcement of the idempotency check and database mutation, and
   explained the duplicate-on-retry failure.  It was coherent but ended at
   the configured 64-token limit rather than at EOS.

5. **Did stale replay occur?**

   No.  Persistent turn 2 had zero-token longest common prefix with turn 1,
   zero prefix overlap, and `stale_replay_detected=false`.  Its token hash was
   `810fea0c23b8689c2664f25eccae03e04bed9bd6f4d39a4ba28b3d7015200830`,
   distinct from turn 1's hash.

6. **How did persistent output compare with fresh full-transcript output?**

   The fresh output was also a coherent answer to the same invariant
   question.  The two outputs were not token-identical: their longest common
   prefix was 42 of 64 tokens and their hashes differed.  The persistent and
   fresh final compact state signatures differed in all 64 layers (48
   ArraysCache and 16 KVCache).  Therefore exact suffix framing removes the
   malformed-protocol stale replay at 4K, but persistent cache state is not yet
   proven equivalent to an empty-cache full-transcript prefill.

7. **What were persistent vs fresh turn-2 TTFTs?**

   Persistent turn 2 processed 38 suffix tokens: TTFT **0.980 s**, prompt
   throughput **38.78 tok/s**, decode **11.63 tok/s**, peak **13.360 GiB**.
   Fresh turn 2 processed the complete 4,230-token transcript: TTFT **54.745
   s**, prompt throughput **77.27 tok/s**, decode **11.60 tok/s**, peak
   **13.361 GiB**.  The initial persistent turn was TTFT **54.442 s** and
   decode **11.60 tok/s**.

8. **Did cache offsets/state reveal anything suspicious?**

   No offset reset, wrap, or disagreement was observed.  Persistent offsets
   advanced uniformly 4,192 → 4,231 at the first generated-step boundary →
   4,294 after 64 turn-2 tokens.  Fresh offsets were 0 → 4,231 → 4,294.  The
   suspicious remaining result is state divergence despite the same canonical
   token stream: all compact final layer signatures differed.  This is an
   observation, not yet a causal diagnosis; the signatures are small sampled
   state checksums, not full tensor equality.

9. **What is the SINGLE recommended next experiment?**

   Run one 4K fresh-cache control that reproduces the persistent path's exact
   step boundaries—initial prompt prefill, one generated token at a time,
   exact suffix, then turn-2 generation—and compare its layer signatures and
   output with this run.  This isolates recurrent/cache batching dependence
   from token-protocol correctness.  Do not start that experiment yet.

Raw token IDs, cache diagnostics, outputs, logs, and the harness are in this
directory; `results.json` is the machine-readable summary.
