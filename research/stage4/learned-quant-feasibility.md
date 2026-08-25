# Learned-quant feasibility for Stage 5

Date: 2026-08-25  
Canonical environment: `envs/mlx-0.32.1`  
Pinned mlx-lm revision: `254d153fdeb6f150edd4fc5a54f9828638481fa8`

## Decision

No learned-quant conversion or training job was run in Stage 4. The primary
quality/runtime work is complete, and the pinned implementations show clear
bounded-memory or reproducibility hazards on the 18 GB machine. This is a
measured feasibility stop, not evidence that the methods cannot work after
adaptation.

The source hashes and machine-readable inspection are in
`artifacts/learned-quant/source-inspection.json`.

## Evidence and method ranking

| method | measured implementation fact | 18 GB Stage 4 verdict | Stage 5 value |
|---|---|---|---|
| Dynamic Quantization | The pinned sensitivity estimator keeps the full model, a deep-copied quantized model, and a deep-copied quantizable-layer map. Its target-BPW threshold search deep-copies the model repeatedly. | Unsafe as-is for Qwen3.8-27B. | Highest. Build a bounded, blockwise gradient/logit sensitivity variant and compare it with the existing output-distortion map. |
| GPTQ | The implementation wraps every quantizable linear layer with a Hessian accumulator before quantizing. A single float32 5120×5120 Hessian is about 100 MiB; many layers plus model residency are not a safe all-layer pass. Its default calibration helper downloads a remote file if absent. | Potentially useful, but only after serializing layer/block capture and using local fixed tokens. | Second. A small, memory-instrumented smoke on selected sensitive modules is worthwhile. |
| AWQ | The algorithm processes blocks and captures activation features, so a bounded adaptation is plausible. The pinned config supports `qwen3` but not the target `qwen3_5`; Qwen3.8 also has hybrid `linear_attn` and regular `self_attn` families. | Not runnable without an architecture-specific adapter; no Stage 4 smoke. | Third, conditional on a measured quality gap after Dynamic/GPTQ work. |
| DWQ | The pinned flow computes teacher logits from a full model, deep-copies or loads a full student, trains quantized scales/biases, defaults to a remote Tulu dataset, and calls `mx.set_wired_limit`. | Not safe or reproducible as-is; explicitly not run. | Fourth. Consider only with offline teacher targets and a layer/block-wise distillation design, potentially using a smaller teacher. |

These source observations are from the canonical pinned files:

- `dynamic_quant.py` SHA-256
  `7f50bfde16bc5ec6a9d30cc1caea3a58af80d4bd7343851985975c1650f66cf8`
- `dwq.py` SHA-256
  `38f018583c879a96972e5ac5b3b315021321c66b532a62e37dabf37850f7f1e1`
- `awq.py` SHA-256
  `a2d632786bd54d3aef9325940712961cd3c5c8556df248152f49113b83ed5f7c`
- `gptq.py` SHA-256
  `feab955de774b6975ff2287c0fffbede74c2cdc459bd33afaaa17c5c47f31c04`

For comparison with future upgrades, the corresponding upstream source
locations are the official mlx-lm files: [Dynamic Quantization](https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/quant/dynamic_quant.py), [DWQ](https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/quant/dwq.py), [AWQ](https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/quant/awq.py), and [GPTQ](https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/quant/gptq.py).

## Relationship to the Stage 4 sensitivity map

The Stage 4 map is a bounded activation-weighted output-distortion proxy: a
resident g128 module is compared with a temporary 4-bit/group-128 module over
the 12 frozen calibration cases, one source tensor chunk at a time. That makes
it suitable for ranking promotions, but it is not equivalent to Dynamic
Quantization's gradient-aligned teacher KL estimator. A future bounded Dynamic
Quant study should therefore measure whether the ranking changes before
building another large candidate.

## Safety and reproducibility constraints for Stage 5

Use only the checked-in local calibration/evaluation data, record hashes and
seeds, keep source/teacher tensors layer-bounded, capture active+cache/peak/RSS,
and reject any path that calls `mx.set_wired_limit` or depends on live web data.
Do not start with DWQ or a whole-model Dynamic Quant run on this machine.
