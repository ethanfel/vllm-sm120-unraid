#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this installer as root on the Unraid host." >&2
  exit 1
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="${VLLM_IMAGE:-ghcr.io/ethanfel/vllm-sm120-unraid:0.3.0}"
template_dir="/boot/config/plugins/dockerMan/templates-user"
template_target="${template_dir}/my-vLLM.xml"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable; install/configure Unraid's Nvidia Driver plugin first." >&2
  exit 1
fi

cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '[:space:]')"
if [[ "${cap}" != "12.0" ]]; then
  echo "Warning: detected compute capability ${cap}; this deployment is tuned for SM120." >&2
fi

mkdir -p /mnt/user/appdata/vllm "${template_dir}"

docker pull "${image}"

if [[ -f "${template_target}" ]]; then
  backup="${template_target}.bak.$(date +%Y%m%d-%H%M%S)"
  cp -a "${template_target}" "${backup}"
  echo "Backed up the previous template to ${backup}"
fi

cp "${repo_dir}/unraid/my-vLLM-SM120.xml" "${template_target}"
chmod 0600 "${template_target}"

echo "Installed ${image} and ${template_target}"
echo "Use Unraid Docker -> Add Container -> vLLM to review settings and start it."
