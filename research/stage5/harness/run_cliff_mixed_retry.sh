#!/bin/zsh
set -euo pipefail

cd "$(dirname "$0")/../../.."
source research/stage4/canonical_env.sh

out="research/stage5/artifacts/cliff/retry-mixed34-16k"
mkdir -p "$out"
uv run --project envs/mlx-0.32.1 --no-sync python research/harness/run_cold_case.py \
  --model-path /Users/artem/models/Qwen3.8-27B-mixed-3-4-control-g128 \
  --prompt-file research/stage5/artifacts/cliff/prompts/prompt-16727.txt \
  --token-file research/stage5/artifacts/cliff/prompts/prompt-16727.tokens.json \
  --target-prompt-tokens 16727 \
  --result-file "$out/result.json" \
  --output-file "$out/output.txt" \
  --experiment-id P1A-16K-MIXED34-RETRY \
  --quant-mode mixed_3_4_control \
  --quant-bits 3 \
  --quant-group-size 128 \
  --kv-group-size 64 \
  --quantized-kv-start 0 \
  --max-tokens 128 \
  --prefill-step-size 2048
