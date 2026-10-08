# Official v0.31.0 default image is CUDA 13.0. The cu129 tag bundles torchcodec
# 0.17 built for CUDA 13, but that image only ships libnvrtc.so.12. An older
# torchcodec pin still linked libnvrtc.so.13, so it was not a compatible fix.
# Digest is the RepoDigest from `docker image inspect vllm/vllm-openai:v0.31.0`.
# Do not switch this base to latest or a nightly tag.
FROM vllm/vllm-openai:v0.31.0@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3

ENV PYTHONUNBUFFERED=1 \
    CONFIG_DIR=/app/configs \
    MODEL_PROFILE=qwen3.8-27b \
    HF_HOME=/cache/huggingface \
    VLLM_CACHE_ROOT=/cache/vllm \
    TORCH_HOME=/cache/torch \
    XDG_CACHE_HOME=/cache/xdg

WORKDIR /app

COPY controller/requirements.txt /app/controller/requirements.txt
# Keep the image torch/vLLM pins. Install controller packages only when missing.
RUN python3 -c "import fastapi,uvicorn,httpx" \
    || pip install --no-cache-dir --upgrade-strategy only-if-needed -r /app/controller/requirements.txt

COPY controller/ /app/controller/
COPY configs/ /app/configs/

EXPOSE 8000

# The base image entrypoint is `vllm serve`. Replace it so the controller is
# the container command. A CMD-only change would pass uvicorn arguments to vllm.
ENTRYPOINT ["python3", "-m", "uvicorn", "controller.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
CMD []
