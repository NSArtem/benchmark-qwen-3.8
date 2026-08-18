#!/bin/zsh

# Every invocation below starts a fresh Python/uvx process. Keep the matrix
# sequential: the 27B checkpoint is intentionally close to the machine's
# recommended unified-memory working set.

set -u
PIN='git+https://github.com/ml-explore/mlx-lm@254d153fdeb6f150edd4fc5a54f9828638481fa8'
export UV_CACHE_DIR=/private/tmp/qwen38-mlx-threshold-uv-cache
MODEL_PATH='/Users/artem/.cache/huggingface/hub/models--lukaskremla--Qwen3.8-27B-3bit-MLX-TextOnly/snapshots/c98bba5926f51fec1c8d8737e577221673f524d7'
ROOT='qwen38-mlx-gated-delta-ab'

for repeat in 1 2 3 4 5; do
  for target in 0512 0640 0768; do
    uvx --from "$PIN" python qwen38-mlx-gated-delta-ab/harness/full_model_ab.py \
      --name "metal-${target}-r${repeat}" \
      --mode metal \
      --model-path "$MODEL_PATH" \
      --prompt-file "qwen38-mlx-threshold/prompts/${target}.txt" \
      --target-prompt-tokens "$((10#$target))" \
      --repeat-index "$repeat" \
      --root "$ROOT" \
      --notes 'required fresh-process Metal repeat'
  done
done

for repeat in 1 2 3; do
  for target in 0512 0640 0768; do
    uvx --from "$PIN" python qwen38-mlx-gated-delta-ab/harness/full_model_ab.py \
      --name "ops-${target}-r${repeat}" \
      --mode ops \
      --model-path "$MODEL_PATH" \
      --prompt-file "qwen38-mlx-threshold/prompts/${target}.txt" \
      --target-prompt-tokens "$((10#$target))" \
      --repeat-index "$repeat" \
      --root "$ROOT" \
      --notes 'required fresh-process OPS repeat; qwen3_5 symbol patched at runtime'
  done
done
