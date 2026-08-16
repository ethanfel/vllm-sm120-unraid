# vLLM on Unraid — RTX PRO 6000 Blackwell (SM120)

This repository publishes a small operational image on top of the official,
digest-pinned `vllm/vllm-openai:v0.27.1` release. It serves
`sakamakismile/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-NVFP4` through an OpenAI-compatible
API and persists model/JIT caches outside the container. A lightweight
wake-on-request proxy keeps port 8000 available while the actual vLLM process
is fully unloaded when idle.

Its AEON source uses an optimized Abliterix ablation plus SSM `conv1d` outlier
repair; it is not a behavioral SFT. The public NVFP4 checkpoint uses W4A4
group-size-16 dense weights while retaining vision, the language head,
recurrent Gated DeltaNet state, and the native MTP head at higher precision.
Its indexed weights total 20.56 GB. The default profile uses the native
262,144-token context on one 96 GB RTX PRO 6000, with FP8 KV cache and
three-token MTP speculative decoding.

Published image:

```text
ghcr.io/ethanfel/vllm-sm120-unraid:0.3.0
```

## Install on Unraid

Download or copy the repository to the server, then run as root:

```bash
chmod +x scripts/*.sh tests/*.sh docker/entrypoint.sh
./tests/validate.sh
./scripts/install-unraid.sh
```

The installer pulls the versioned GHCR image, backs up an existing `my-vLLM.xml`, and installs the new Unraid template. Review and start it from **Docker → Add Container → vLLM**. Unraid does not build the image locally.

The default 262K checkpoint profile needs the GPU mostly free. A loaded ComfyUI diffusion model can prevent vLLM from reserving enough VRAM even if ComfyUI is idle. Unload its models or stop that container before starting vLLM. ComfyUI can be restarted afterward as an API client, provided it does not load another large GPU model at the same time.

For Compose instead, copy `.env.example` to `.env`, review it, then run:

```bash
docker compose pull
docker compose up -d
```

The container itself starts with the model unloaded. The first `/v1` request downloads the model if it is not cached, loads vLLM, waits for readiness, and then forwards that same request. Follow progress with:

```bash
docker logs -f vLLM
```

Open the container's **WebUI** from Unraid for the interactive dashboard. It shows live lifecycle status and includes one-click tests for model discovery, chat completion, automatic tool calling, and local image input. Its **Cached models** section lists Hugging Face checkpoints and their disk usage without loading vLLM. Unused checkpoints can be removed after typing the full repository name; the currently configured model is protected from deletion. Deleted weights can be recovered only by downloading them again. Set `MODEL_CACHE_DELETE_ENABLED=false` to make the cache manager read-only.

Opening the dashboard and its `/docs` alias does not load the model; pressing **Load model**, opening **vLLM Swagger**, or running an API test does.

## ComfyUI / OpenAI-compatible clients

Use these values in an OpenAI-compatible LLM node:

- Base URL: `http://192.168.1.12:8000/v1`
- Model: `qwen3.8-27b-aeon-nvfp4`
- API key: `EMPTY` when the template's `API_KEY` is blank; otherwise use the configured value
- Chat completions route: `/chat/completions`

Test the server from the Unraid host:

```bash
VLLM_BASE_URL=http://127.0.0.1:8000/v1 ./scripts/smoke-test.sh
```

Qwen3.8 thinks by default. Clients that support extra OpenAI fields can request non-thinking mode with:

```json
{"chat_template_kwargs":{"enable_thinking":false}}
```

The server defaults to `enable_thinking=true`, `reasoning_effort=medium`, and
`preserve_thinking=true`. Medium reasoning avoids unnecessarily long thinking
traces while retaining Qwen's reasoning mode. Per-request
`chat_template_kwargs` can override these values.

Automatic tools default to `TOOL_CALL_PARSER=qwen3_coder`, matching this checkpoint's XML `<function>`/`<parameter>` tool format. vLLM also supports `hermes`, but only select it for a checkpoint/chat template that emits Hermes JSON tool calls.

Recommended starting sampling values are `temperature=1.0`, `top_p=0.95`, and `top_k=20` for general thinking, or `temperature=0.7`, `top_p=0.8`, `top_k=20`, and `presence_penalty=1.5` for non-thinking use.

## On-demand loading

On-demand mode is enabled by default. After the final API response finishes, an idle timer starts. At 600 seconds the vLLM child process is terminated, releasing its model weights, KV cache, CUDA allocations, and model-owned CPU memory. The controller itself stays healthy and consumes very little memory.

The OpenAI URL does not change:

```text
http://192.168.1.12:8000/v1
```

Inspect or control the model lifecycle from the trusted LAN:

```bash
curl http://192.168.1.12:8000/on-demand/status
curl -X POST http://192.168.1.12:8000/on-demand/load
curl -X POST http://192.168.1.12:8000/on-demand/unload
```

If `API_KEY` is configured, the load and unload endpoints require the same bearer token. The dashboard, `/docs`, status, and health do not start the model. Opening `/vllm/docs`, querying `/v1/models`, or making any other `/v1` request does start it. Set `IDLE_TIMEOUT_SECONDS` to change the default ten-minute delay, or set `ENABLE_ON_DEMAND=false` for the original always-loaded behavior.

The cached-model manager also stays outside `/v1`, so listing storage never wakes the model:

```bash
curl http://192.168.1.12:8000/on-demand/models
curl -X POST http://192.168.1.12:8000/on-demand/models/delete \
  -H 'Content-Type: application/json' \
  -d '{"repo_id":"owner/unused-model","confirmation":"owner/unused-model"}'
```

When `API_KEY` is configured, both model-manager endpoints require the same bearer token. Deletion accepts only repositories returned by the cache listing, requires an exact confirmation value, and always rejects the configured `MODEL_ID`. It removes only the matching Hugging Face `models--owner--repository` directory under `/models/hub`; vLLM and compiler caches are outside its scope. Set `MODEL_CACHE_DELETE_ENABLED=false` for listing without deletion.

Before launching vLLM, the controller requires 90,000 MiB of free VRAM by default. This prevents an LLM request from destabilizing an active ComfyUI render. If VRAM is low, it checks `COMFYUI_BASE_URL`: when Comfy's queue is completely idle it calls the supported `/free` endpoint and waits for cached models to be released; when a job is running or queued it returns `503` without interrupting anything.

An OpenAI node placed after a large diffusion/video stage in the same active Comfy workflow still needs an explicit **Unload Models** node immediately before the API node. Comfy cannot process its asynchronous `/free` flag while that same workflow is blocked waiting for the LLM response. The threshold is configurable with `MIN_FREE_VRAM_MIB`; setting it to `0` disables the guard. Leave `COMFYUI_BASE_URL` blank to disable automatic idle-cache release.

The Linux filesystem cache may retain recently read model files after unload. This cache is reclaimable and is automatically surrendered when applications need RAM; it is not pinned model memory.

## Tuning

- If startup runs out of VRAM, set `MAX_MODEL_LEN=131072`, then `65536`. Try `GPU_MEMORY_UTILIZATION=0.92` only if no other GPU workload is present.
- Set `LANGUAGE_MODEL_ONLY=true` if ComfyUI only sends text. This removes the vision encoder and leaves more cache headroom.
- MTP is enabled by default with `{"method":"mtp","num_speculative_tokens":3}`. Set `ENABLE_MTP=false` to compare baseline throughput or diagnose speculative-decoding issues.
- FP8 KV cache is enabled by default to preserve context capacity. Set `KV_CACHE_DTYPE=bfloat16` only for diagnosis; it approximately doubles KV-cache memory.
- Keep port 8000 on the trusted LAN. If the endpoint is reachable by untrusted clients, set a strong `API_KEY` and add firewall or reverse-proxy controls.

## Publishing

Tagged releases are validated and published by GitHub Actions to GHCR. The `v0.3.0` tag produces immutable `0.3.0`, moving `0.3`, and commit-SHA image tags. The workflow uses GitHub's package token; no registry credential is stored in this repository.

For a local development build:

```bash
docker build -t vllm-sm120-unraid:dev .
./tests/validate.sh
```
