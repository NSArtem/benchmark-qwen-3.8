# Qwen3.8 quantization size analysis

Header-only analysis; no new model was loaded or converted.

Current checkpoint bytes: **10,930,715,648** (10.180 GiB); parameters: **26,895,993,856**.
Quantized modules: **498**; non-quantized storage: **0.005 GiB**.
Reconstructed trainable parameter count matches index after excluding **4,608** saved `A_log`/`dt_bias` buffer elements: **True**.

| Candidate | Effective bpw | Stored GiB | Saved GiB | Saved % | First-order extra tokens to 16 GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| affine_3_g64 | 3.501 | 10.963 | -0.783 | -7.69% | -4019 |
| affine_3_g128 | 3.251 | 10.180 | 0.000 | 0.00% | 0 |
| affine_2_g128 | 2.251 | 7.049 | 3.131 | 30.75% | 16075 |
| mixed_2_6_g64 | 3.121 | 9.772 | 0.408 | 4.01% | 2097 |
| mixed_2_6_g128 | 2.871 | 8.989 | 1.191 | 11.70% | 6116 |

Measured cold-curve fit: peak GiB ≈ 13.120 + 0.0001948 × prompt tokens.
Modules not divisible by g128: 0.

## Candidate ranking

- `affine_3_g128`: lowest quality risk and same 3-bit runtime family; expected saving is metadata-only and therefore modest.
- `mixed_2_6_g128`: built-in pinned predicate, larger saving than g128 affine, but mixed low-bit quality must be measured from original weights.
- `mixed_2_6_g64`: same mixed policy without the small g128 metadata saving.
- `affine_2_g128`: largest uniform saving among these candidates, but highest expected quality risk.

These are storage estimates, not correctness or quality claims.
