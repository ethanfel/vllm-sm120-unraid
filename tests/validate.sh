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
grep -Fxq '<{"enable_thinking":true,"preserve_thinking":true}>' <<<"${entrypoint_output}"
grep -Fxq '<--seed>' <<<"${entrypoint_output}"
grep -Fxq '<7>' <<<"${entrypoint_output}"
grep -Fxq '<qwen3_coder>' <<<"${entrypoint_output}"

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

if command -v python3 >/dev/null 2>&1; then
python3 - "${repo_dir}/unraid/my-vLLM-SM120.xml" <<'PY'
import sys
import xml.etree.ElementTree as ET

path = sys.argv[1]
root = ET.parse(path).getroot()
assert root.tag == "Container"
assert root.findtext("Repository") == "ghcr.io/ethanfel/vllm-sm120-unraid:0.1.1"
assert root.findtext("WebUI") == "http://[IP]:[PORT:8000]/"
targets = {node.attrib.get("Target") for node in root.findall("Config")}
required = {
    "/models", "8000", "MODEL_ID", "SERVED_MODEL_NAME", "MAX_MODEL_LEN",
    "ENABLE_ON_DEMAND", "IDLE_TIMEOUT_SECONDS", "VLLM_INTERNAL_PORT",
    "MIN_FREE_VRAM_MIB",
    "COMFYUI_BASE_URL",
    "TOOL_CALL_PARSER",
}
missing = required - targets
assert not missing, f"missing template targets: {sorted(missing)}"
print("Unraid XML and shell syntax are valid")
PY
else
  template="${repo_dir}/unraid/my-vLLM-SM120.xml"
  grep -Fq '<Container version="2">' "${template}"
  grep -Fq '<Repository>ghcr.io/ethanfel/vllm-sm120-unraid:0.1.1</Repository>' "${template}"
  grep -Fq '<WebUI>http://[IP]:[PORT:8000]/</WebUI>' "${template}"
  grep -Fq 'Target="MODEL_ID"' "${template}"
  grep -Fq 'Target="SERVED_MODEL_NAME"' "${template}"
  echo "Shell syntax and required Unraid XML fields are valid (python3 unavailable; basic XML check used)"
fi
