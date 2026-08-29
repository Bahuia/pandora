#!/usr/bin/env bash
set -euo pipefail

task="$1"
dataset="$2"
stage="$3"

data_root="${PANDORA_DATA_ROOT:-./data}"
model="${PANDORA_MODEL:-gpt-4o-mini}"
output_root="${PANDORA_OUTPUT_DIR:-./results}"
provider="${PANDORA_PROVIDER:-auto}"

endpoint_args=()
if [[ -n "${PANDORA_BASE_URL:-}" ]]; then
  endpoint_args+=(--base-url "$PANDORA_BASE_URL")
  if [[ "$provider" == "auto" ]]; then
    provider="openai-compatible"
  fi
fi

exec pandora \
  --task "$task" \
  --dataset "$dataset" \
  --stage "$stage" \
  --model "$model" \
  --provider "$provider" \
  --data-root "$data_root" \
  --output-dir "$output_root" \
  --shot-k 0 \
  --retrieval-mode disabled \
  "${endpoint_args[@]}" \
  "${@:4}"
