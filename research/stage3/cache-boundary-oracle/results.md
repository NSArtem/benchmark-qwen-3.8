# Phase 1 same-boundary cache oracle

Decision gate: **PASS**

| path | execution | turn-1 tokens | turn-2 prompt/suffix | turn-2 output |
|---|---|---:|---:|---|
| A | fresh one-shot full transcript | n/a | 4230 full | ecbadfd41fe02293fde5a5a15c9fc83a68a34ceda8fb53d61df56e3417fc9a5d |
| B | fresh segmented exact suffix | 96 | 38 suffix | 810fea0c23b8689c2664f25eccae03e04bed9bd6f4d39a4ba28b3d7015200830 |
| C | true persistent exact suffix | 96 | 38 suffix | 810fea0c23b8689c2664f25eccae03e04bed9bd6f4d39a4ba28b3d7015200830 |

B-vs-C boundary comparison:

```json
{
  "all_boundary_offsets_equal": true,
  "all_boundary_state_signatures_equal": true,
  "boundary_comparisons": {
    "after_turn1_final": {
      "available": true,
      "first_mismatching_layer": null,
      "layer_count_left": 64,
      "layer_count_right": 64,
      "left_class_counts": {
        "ArraysCache": 48,
        "KVCache": 16
      },
      "left_signature": "61e84826b55335b3d59cdcfc3670ba514d8118410bc5a49db604039e5232aa4e",
      "mismatching_layer_indices": [],
      "offsets_equal": true,
      "right_class_counts": {
        "ArraysCache": 48,
        "KVCache": 16
      },
      "right_signature": "61e84826b55335b3d59cdcfc3670ba514d8118410bc5a49db604039e5232aa4e",
      "signature_equal": true
    },
    "after_turn2_final": {
      "available": true,
      "first_mismatching_layer": null,
      "layer_count_left": 64,
      "layer_count_right": 64,
      "left_class_counts": {
        "ArraysCache": 48,
        "KVCache": 16
      },
      "left_signature": "1b123c1df009a09cb5026c07fe84f58ef02733031b43cf5011693d90eb14fa5c",
      "mismatching_layer_indices": [],
      "offsets_equal": true,
      "right_class_counts": {
        "ArraysCache": 48,
        "KVCache": 16
      },
      "right_signature": "1b123c1df009a09cb5026c07fe84f58ef02733031b43cf5011693d90eb14fa5c",
      "signature_equal": true
    },
    "after_turn2_first_yield": {
      "available": true,
      "first_mismatching_layer": null,
      "layer_count_left": 64,
      "layer_count_right": 64,
      "left_class_counts": {
        "ArraysCache": 48,
        "KVCache": 16
      },
      "left_signature": "1b123c1df009a09cb5026c07fe84f58ef02733031b43cf5011693d90eb14fa5c",
      "mismatching_layer_indices": [],
      "offsets_equal": true,
      "right_class_counts": {
        "ArraysCache": 48,
        "KVCache": 16
      },
      "right_signature": "1b123c1df009a09cb5026c07fe84f58ef02733031b43cf5011693d90eb14fa5c",
      "signature_equal": true
    },
    "after_turn2_suffix_and_first_generation_step": {
      "available": true,
      "first_mismatching_layer": null,
      "layer_count_left": 64,
      "layer_count_right": 64,
      "left_class_counts": {
        "ArraysCache": 48,
        "KVCache": 16
      },
      "left_signature": "726683aa750f4a4b3b93eda17d61214b73b796a10ea3f363027ae52b9e1e1133",
      "mismatching_layer_indices": [],
      "offsets_equal": true,
      "right_class_counts": {
        "ArraysCache": 48,
        "KVCache": 16
      },
      "right_signature": "726683aa750f4a4b3b93eda17d61214b73b796a10ea3f363027ae52b9e1e1133",
      "signature_equal": true
    },
    "before_turn2_suffix": {
      "available": true,
      "first_mismatching_layer": null,
      "layer_count_left": 64,
      "layer_count_right": 64,
      "left_class_counts": {
        "ArraysCache": 48,
        "KVCache": 16
      },
      "left_signature": "61e84826b55335b3d59cdcfc3670ba514d8118410bc5a49db604039e5232aa4e",
      "mismatching_layer_indices": [],
      "offsets_equal": true,
      "right_class_counts": {
        "ArraysCache": 48,
        "KVCache": 16
      },
      "right_signature": "61e84826b55335b3d59cdcfc3670ba514d8118410bc5a49db604039e5232aa4e",
      "signature_equal": true
    }
  },
  "gate": "PASS",
  "logit_comparisons": [
    {
      "available": true,
      "left_logprob_sha256": "fe04663ce2fbbe5ed953e570b42fc3b193b6a6fb0262246ec1f1ea0e4f55dc36",
      "max_abs_logprob_diff": 0.0,
      "mean_abs_logprob_diff": 0.0,
      "right_logprob_sha256": "fe04663ce2fbbe5ed953e570b42fc3b193b6a6fb0262246ec1f1ea0e4f55dc36",
      "stage": "turn1_first_generated_token",
      "top1_equal": true,
      "top_k_overlap": 8,
      "top_k_size": 8
    },
    {
      "available": true,
      "left_logprob_sha256": "689c204b12f5071ec64b4dea300ea86c3a74348c74d128bb3ff78edd86016c68",
      "max_abs_logprob_diff": 0.0,
      "mean_abs_logprob_diff": 0.0,
      "right_logprob_sha256": "689c204b12f5071ec64b4dea300ea86c3a74348c74d128bb3ff78edd86016c68",
      "stage": "turn2_first_generated_token",
      "top1_equal": true,
      "top_k_overlap": 8,
      "top_k_size": 8
    }
  ],
  "no_stale_replay_in_b_or_c": true,
  "protocol_equations_both_hold": true,
  "turn1_token_stream_equal": true,
  "turn2_token_stream_equal": true
}
```

Interpretation: A-vs-C output divergence is expected to be a separate
one-shot-versus-segmented numerical-sensitivity question. The scaling
gate is determined by B-vs-C state, logits, token stream, protocol
accounting, and replay checks.
