#!/bin/zsh
set -euo pipefail

cd "$(dirname "$0")/../../.."
source research/stage4/canonical_env.sh

g128="/Users/artem/models/Qwen3.8-27B-3bit-g128-MLX"
mixed34="/Users/artem/models/Qwen3.8-27B-mixed-3-4-control-g128"

run_one() {
  local label="$1"
  local target="$2"
  local model_path="$3"
  local quant_mode="$4"
  local out="research/stage5/artifacts/cliff/sweep/${target}/${label}"
  local prompt="research/stage5/artifacts/cliff/prompts/prompt-${target}.txt"
  local tokens="research/stage5/artifacts/cliff/prompts/prompt-${target}.tokens.json"
  mkdir -p "$out"
  memory_pressure -Q
  uv run --project envs/mlx-0.32.1 --no-sync python research/harness/run_cold_case.py \
    --model-path "$model_path" \
    --prompt-file "$prompt" \
    --token-file "$tokens" \
    --target-prompt-tokens "$target" \
    --result-file "$out/result.json" \
    --output-file "$out/output.txt" \
    --experiment-id "P1B-${target}-${label}" \
    --quant-mode "$quant_mode" \
    --quant-bits 3 \
    --quant-group-size 128 \
    --kv-group-size 64 \
    --quantized-kv-start 0 \
    --max-tokens 128 \
    --prefill-step-size 2048
}

run_one G128 8192 "$g128" affine
run_one MIXED34 8192 "$mixed34" mixed_3_4_control
run_one G128 10240 "$g128" affine
run_one MIXED34 10240 "$mixed34" mixed_3_4_control
run_one G128 12288 "$g128" affine
run_one MIXED34 12288 "$mixed34" mixed_3_4_control
run_one G128 14336 "$g128" affine
run_one MIXED34 14336 "$mixed34" mixed_3_4_control
