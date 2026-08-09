#!/usr/bin/env bash
set -Eeuo pipefail

base_url="${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"
model="${SERVED_MODEL_NAME:-qwen3.6-27b-fable}"
api_key="${API_KEY:-EMPTY}"

curl --fail --silent --show-error \
  -H "Authorization: Bearer ${api_key}" \
  "${base_url}/models"
echo

curl --fail --silent --show-error \
  -H "Authorization: Bearer ${api_key}" \
  -H 'Content-Type: application/json' \
  "${base_url}/chat/completions" \
  --data "$(printf '{\"model\":\"%s\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: vLLM SM120 ready\"}],\"max_tokens\":32,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}' "${model}")"
echo

