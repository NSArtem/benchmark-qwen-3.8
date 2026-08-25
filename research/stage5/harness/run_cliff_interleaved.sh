#!/bin/zsh
set -euo pipefail

cd "$(dirname "$0")/../../.."
source research/stage4/canonical_env.sh

prompt="research/stage5/artifacts/cliff/prompts/prompt-16727.txt"
tokens="research/stage5/artifacts/cliff/prompts/prompt-16727.tokens.json"
common=(
  --prompt-file "$prompt"
  --token-file "$tokens"
  --target-prompt-tokens 16727
  --max-tokens 192
  --prefill-step-size 2048
  --kv-group-size 64
  --quantized-kv-start 0
)

run_one() {
  local experiment_id="$1"
  local model_path="$2"
  local quant_mode="$3"
  local quant_bits="$4"
  local quant_group="$5"
  local output_dir="research/stage5/artifacts/cliff/interleaved/${experiment_id}"
  mkdir -p "$output_dir"
  uv run --project envs/mlx-0.32.1 --no-sync python research/harness/run_cold_case.py \
    --model-path "$model_path" \
    --result-file "$output_dir/result.json" \
    --output-file "$output_dir/output.txt" \
    --experiment-id "$experiment_id" \
    --quant-mode "$quant_mode" \
    --quant-bits "$quant_bits" \
    --quant-group-size "$quant_group" \
    "${common[@]}"
}

g128="/Users/artem/models/Qwen3.8-27B-3bit-g128-MLX"
mixed34="/Users/artem/models/Qwen3.8-27B-mixed-3-4-control-g128"

run_one P1A-16K-G128-R1 "$g128" affine 3 128
run_one P1A-16K-MIXED34-R1 "$mixed34" mixed_3_4_control 3 128
run_one P1A-16K-G128-R2 "$g128" affine 3 128
run_one P1A-16K-MIXED34-R2 "$mixed34" mixed_3_4_control 3 128
run_one P1A-16K-G128-R3 "$g128" affine 3 128
run_one P1A-16K-MIXED34-R3 "$mixed34" mixed_3_4_control 3 128
