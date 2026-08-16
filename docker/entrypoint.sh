#!/usr/bin/env bash
set -Eeuo pipefail

bool_enabled() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

# A non-option command is an intentional entrypoint override, useful for
# diagnostics such as: docker run ... IMAGE python3 -c 'import torch; ...'
if [[ $# -gt 0 && "${1}" != --* ]]; then
  exec "$@"
fi

MODEL_ID="${MODEL_ID:-sakamakismile/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-NVFP4}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-27b-aeon-nvfp4}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"
ENABLE_ON_DEMAND="${ENABLE_ON_DEMAND:-true}"
VLLM_INTERNAL_PORT="${VLLM_INTERNAL_PORT:-8001}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-}"
DEFAULT_CHAT_TEMPLATE_KWARGS="${DEFAULT_CHAT_TEMPLATE_KWARGS:-}"
ENABLE_MTP="${ENABLE_MTP:-true}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"

if bool_enabled "${ENABLE_ON_DEMAND}"; then
  if [[ "${VLLM_INTERNAL_PORT}" == "${PORT}" ]]; then
    echo "VLLM_INTERNAL_PORT must differ from the public VLLM_PORT when on-demand mode is enabled." >&2
    exit 2
  fi
  serve_host="127.0.0.1"
  serve_port="${VLLM_INTERNAL_PORT}"
else
  serve_host="${HOST}"
  serve_port="${PORT}"
fi

# Do not put brace-containing JSON inside a ${VAR:-default} expression: when
# VAR is already set, Bash can leave the expression's final brace as a literal.
if [[ -z "${LIMIT_MM_PER_PROMPT}" ]]; then
  LIMIT_MM_PER_PROMPT='{"image":4,"video":0}'
fi
if [[ -z "${DEFAULT_CHAT_TEMPLATE_KWARGS}" ]]; then
  DEFAULT_CHAT_TEMPLATE_KWARGS='{"enable_thinking":true,"reasoning_effort":"medium","preserve_thinking":true}'
fi

# Accept either common Hugging Face token spelling without printing the secret.
if [[ -n "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
elif [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" && -z "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  compute_cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '[:space:]' || true)"
  gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || true)"
  if [[ -n "${compute_cap}" ]]; then
    echo "Detected GPU: ${gpu_name} (compute capability ${compute_cap})"
  fi
  if [[ -n "${compute_cap}" && "${compute_cap}" != 12.* ]]; then
    echo "Warning: this image is tuned for SM120, but the detected compute capability is ${compute_cap}." >&2
  fi
fi

args=(
  vllm serve "${MODEL_ID}"
  --host "${serve_host}"
  --port "${serve_port}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --dtype "${DTYPE:-auto}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --generation-config vllm
  --reasoning-parser qwen3
  --default-chat-template-kwargs "${DEFAULT_CHAT_TEMPLATE_KWARGS}"
  --enable-chunked-prefill
)

if bool_enabled "${ENABLE_PREFIX_CACHING:-true}"; then
  args+=(--enable-prefix-caching)
fi

if bool_enabled "${ENABLE_TOOLS:-true}"; then
  args+=(--enable-auto-tool-choice --tool-call-parser "${TOOL_CALL_PARSER:-qwen3_coder}")
fi

if bool_enabled "${LANGUAGE_MODEL_ONLY:-false}"; then
  args+=(--language-model-only)
else
  args+=(--limit-mm-per-prompt "${LIMIT_MM_PER_PROMPT}")
fi

# This checkpoint's restored native MTP head is enabled by default. Set
# ENABLE_MTP=false to disable speculative decoding for comparison or diagnosis.
if bool_enabled "${ENABLE_MTP}"; then
  speculative_config="${SPECULATIVE_CONFIG:-}"
  if [[ -z "${speculative_config}" ]]; then
    speculative_config='{"method":"mtp","num_speculative_tokens":3}'
  fi
  args+=(--speculative-config "${speculative_config}")
fi

if [[ -n "${KV_CACHE_DTYPE}" ]]; then
  args+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
fi

if [[ -n "${API_KEY:-}" ]]; then
  args+=(--api-key "${API_KEY}")
fi

if [[ -n "${MODEL_REVISION:-}" ]]; then
  args+=(--revision "${MODEL_REVISION}")
fi

echo "Configured vLLM model '${SERVED_MODEL_NAME}' from '${MODEL_ID}'"
echo "Profile: max_model_len=${MAX_MODEL_LEN}, gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}, max_num_seqs=${MAX_NUM_SEQS}, mtp=${ENABLE_MTP}, kv_cache_dtype=${KV_CACHE_DTYPE}"

# Additional vLLM flags may be supplied as Docker command/Post Arguments. Each
# must be a distinct argument so quoted JSON remains intact.
if bool_enabled "${ENABLE_ON_DEMAND}"; then
  echo "On-demand API listening on ${HOST}:${PORT}; model process starts on the first /v1 request"
  echo "Idle unload: ${IDLE_TIMEOUT_SECONDS:-600}s; model load timeout: ${MODEL_LOAD_TIMEOUT_SECONDS:-1200}s"
  exec python3 /usr/local/bin/vllm-on-demand-server \
    --listen-host "${HOST}" \
    --listen-port "${PORT}" \
    --upstream-port "${VLLM_INTERNAL_PORT}" \
    --idle-timeout "${IDLE_TIMEOUT_SECONDS:-600}" \
    --load-timeout "${MODEL_LOAD_TIMEOUT_SECONDS:-1200}" \
    --stop-timeout "${MODEL_STOP_TIMEOUT_SECONDS:-120}" \
    --min-free-vram-mib "${MIN_FREE_VRAM_MIB:-90000}" \
    --comfyui-base-url "${COMFYUI_BASE_URL:-}" \
    --comfyui-free-timeout "${COMFYUI_FREE_TIMEOUT_SECONDS:-120}" \
    -- "${args[@]}" "$@"
else
  echo "Starting persistent vLLM server on ${HOST}:${PORT}"
  exec "${args[@]}" "$@"
fi
