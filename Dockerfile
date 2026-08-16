ARG VLLM_BASE_IMAGE=vllm/vllm-openai:v0.27.1@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967
FROM ${VLLM_BASE_IMAGE}

USER root

# These values matter for CUDA/Triton extensions compiled lazily on GeForce and
# RTX PRO Blackwell (compute capability 12.0). The official v0.27.1 image
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
