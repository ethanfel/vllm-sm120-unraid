ARG VLLM_BASE_IMAGE=vllm/vllm-openai:v0.30.0@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90
FROM ${VLLM_BASE_IMAGE}

USER root

# vLLM 0.30.0 does not reload the separate MTP draft after level-2 sleep.
# Apply the pinned upstream fix until vLLM includes it in a release.
# https://github.com/vllm-project/vllm/pull/52487
RUN set -eux; \
    curl -fsSL --retry 3 https://github.com/vllm-project/vllm/pull/52487.diff -o /tmp/vllm-mtp-level2.diff; \
    echo '4c99d09a248f645626bd12b14760d9f969496d9e5d84dcbdc75dfd661b44de59  /tmp/vllm-mtp-level2.diff' | sha256sum -c -; \
    awk '/^diff --git / { if (copy) exit; if ($0 == "diff --git a/vllm/v1/worker/gpu_model_runner.py b/vllm/v1/worker/gpu_model_runner.py") copy=1 } copy' /tmp/vllm-mtp-level2.diff \
      | patch --batch --forward -p1 -d /usr/local/lib/python3.12/dist-packages; \
    python3 -m py_compile /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py; \
    rm /tmp/vllm-mtp-level2.diff

# These values matter for CUDA/Triton extensions compiled lazily on GeForce and
# RTX PRO Blackwell (compute capability 12.0). The official v0.30.0 image
# already contains the released SM12x support; this keeps any JIT work on the
# same architecture instead of compiling a broad set of kernels.
ENV TORCH_CUDA_ARCH_LIST=12.0 \
    CUDAARCHS=120 \
    CMAKE_CUDA_ARCHITECTURES=120 \
    CUDA_MODULE_LOADING=LAZY \
    HF_HOME=/models \
    XDG_CACHE_HOME=/models \
    VLLM_CACHE_ROOT=/models/vllm \
    TORCHINDUCTOR_CACHE_DIR=/models/torchinductor/sm120

COPY --chmod=0755 docker/entrypoint.sh /usr/local/bin/vllm-unraid-entrypoint
COPY --chmod=0755 docker/on_demand_server.py /usr/local/bin/vllm-on-demand-server
COPY docker/dashboard.html /usr/local/share/vllm-on-demand/dashboard.html

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=5 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1

ENTRYPOINT ["/usr/local/bin/vllm-unraid-entrypoint"]
