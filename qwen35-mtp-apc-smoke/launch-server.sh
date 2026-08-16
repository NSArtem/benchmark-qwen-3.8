#!/bin/zsh
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_dir"

export APC_ENABLED=1
export APC_TRACE=1
export APC_DISK_PATH="$repo_dir/qwen35-mtp-apc-smoke/apc-cache"
export APC_DISK_MAX_GB=1
export UV_CACHE_DIR=.uv-cache

server_args=(
  --model mlx-community/Qwen3.5-9B-4bit
  --host 127.0.0.1
  --port "${PORT:-8099}"
  --log-level INFO
)

if [[ "${NO_MTP:-0}" == "1" ]]; then
  print "Starting optional APC control without a drafter" >&2
else
  server_args+=(
    --draft-model mlx-community/Qwen3.5-9B-MTP-4bit
    --draft-kind mtp
  )
fi

exec uv run mlx_vlm.server "${server_args[@]}"
