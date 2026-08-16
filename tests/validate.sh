#!/usr/bin/env bash
set -Eeuo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

bash -n "${repo_dir}/docker/entrypoint.sh"
bash -n "${repo_dir}/scripts/install-unraid.sh"
bash -n "${repo_dir}/scripts/smoke-test.sh"
python3 -m py_compile "${repo_dir}/docker/on_demand_server.py"
python3 -m unittest discover -s "${repo_dir}/tests" -p 'test_on_demand_server.py'

entrypoint_output="$(
  PATH="${repo_dir}/tests/bin:${PATH}" \
  MODEL_ID=test/model \
  SERVED_MODEL_NAME=test-name \
  ENABLE_ON_DEMAND=false \
  "${repo_dir}/docker/entrypoint.sh" --seed 7
)"
grep -Fxq '<vllm>' <<<"${entrypoint_output}"
grep -Fxq '<serve>' <<<"${entrypoint_output}"
grep -Fxq '<{"image":4,"video":0}>' <<<"${entrypoint_output}"
grep -Fxq '<{"enable_thinking":true,"reasoning_effort":"medium","preserve_thinking":true}>' <<<"${entrypoint_output}"
grep -Fxq '<--seed>' <<<"${entrypoint_output}"
grep -Fxq '<7>' <<<"${entrypoint_output}"
grep -Fxq '<qwen3_coder>' <<<"${entrypoint_output}"
grep -Fxq '<262144>' <<<"${entrypoint_output}"
grep -Fxq '<{"method":"mtp","num_speculative_tokens":3}>' <<<"${entrypoint_output}"
grep -Fxq '<fp8>' <<<"${entrypoint_output}"

entrypoint_output_with_hermes="$(
  PATH="${repo_dir}/tests/bin:${PATH}" \
  MODEL_ID=test/model \
  TOOL_CALL_PARSER=hermes \
  ENABLE_ON_DEMAND=false \
  "${repo_dir}/docker/entrypoint.sh"
)"
grep -Fxq '<hermes>' <<<"${entrypoint_output_with_hermes}"

entrypoint_output_with_json="$(
  PATH="${repo_dir}/tests/bin:${PATH}" \
  MODEL_ID=test/model \
  LIMIT_MM_PER_PROMPT='{"image":2,"video":0}' \
  DEFAULT_CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}' \
  SPECULATIVE_CONFIG='{"method":"mtp","num_speculative_tokens":1}' \
  ENABLE_MTP=true \
  ENABLE_ON_DEMAND=false \
  "${repo_dir}/docker/entrypoint.sh"
)"
grep -Fxq '<{"image":2,"video":0}>' <<<"${entrypoint_output_with_json}"
grep -Fxq '<{"enable_thinking":false}>' <<<"${entrypoint_output_with_json}"
grep -Fxq '<{"method":"mtp","num_speculative_tokens":1}>' <<<"${entrypoint_output_with_json}"

entrypoint_output_with_default_mtp="$(
  PATH="${repo_dir}/tests/bin:${PATH}" \
  ENABLE_MTP=true \
  ENABLE_ON_DEMAND=false \
  "${repo_dir}/docker/entrypoint.sh"
)"
grep -Fxq '<sakamakismile/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-NVFP4>' <<<"${entrypoint_output_with_default_mtp}"
grep -Fxq '<qwen3.8-27b-aeon-nvfp4>' <<<"${entrypoint_output_with_default_mtp}"
grep -Fxq '<{"method":"mtp","num_speculative_tokens":3}>' <<<"${entrypoint_output_with_default_mtp}"

entrypoint_output_without_mtp="$(
  PATH="${repo_dir}/tests/bin:${PATH}" \
  MODEL_ID=test/model \
  ENABLE_MTP=false \
  ENABLE_ON_DEMAND=false \
  "${repo_dir}/docker/entrypoint.sh"
)"
if grep -Fxq '<--speculative-config>' <<<"${entrypoint_output_without_mtp}"; then
  echo "MTP remained enabled despite ENABLE_MTP=false" >&2
  exit 1
fi

if command -v python3 >/dev/null 2>&1; then
python3 - "${repo_dir}/unraid/my-vLLM-SM120.xml" <<'PY'
import sys
import xml.etree.ElementTree as ET

path = sys.argv[1]
root = ET.parse(path).getroot()
assert root.tag == "Container"
assert root.findtext("Repository") == "ghcr.io/ethanfel/vllm-sm120-unraid:0.3.0"
assert root.findtext("WebUI") == "http://[IP]:[PORT:8000]/"
configs = {node.attrib.get("Target"): node for node in root.findall("Config")}
targets = set(configs)
required = {
    "/models", "8000", "MODEL_ID", "SERVED_MODEL_NAME", "MAX_MODEL_LEN",
    "ENABLE_ON_DEMAND", "IDLE_TIMEOUT_SECONDS", "VLLM_INTERNAL_PORT",
    "MIN_FREE_VRAM_MIB",
    "COMFYUI_BASE_URL",
    "MODEL_CACHE_DELETE_ENABLED",
    "TOOL_CALL_PARSER",
}
missing = required - targets
assert not missing, f"missing template targets: {sorted(missing)}"
assert configs["MODEL_ID"].attrib["Default"] == "sakamakismile/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-NVFP4"
assert configs["SERVED_MODEL_NAME"].attrib["Default"] == "qwen3.8-27b-aeon-nvfp4"
assert configs["MAX_MODEL_LEN"].attrib["Default"] == "262144"
assert configs["ENABLE_MTP"].attrib["Default"] == "true"
assert configs["KV_CACHE_DTYPE"].attrib["Default"] == "fp8"
print("Unraid XML and shell syntax are valid")
PY
else
  template="${repo_dir}/unraid/my-vLLM-SM120.xml"
  grep -Fq '<Container version="2">' "${template}"
  grep -Fq '<Repository>ghcr.io/ethanfel/vllm-sm120-unraid:0.3.0</Repository>' "${template}"
  grep -Fq '<WebUI>http://[IP]:[PORT:8000]/</WebUI>' "${template}"
  grep -Fq 'Target="MODEL_ID"' "${template}"
  grep -Fq 'Target="SERVED_MODEL_NAME"' "${template}"
  echo "Shell syntax and required Unraid XML fields are valid (python3 unavailable; basic XML check used)"
fi
