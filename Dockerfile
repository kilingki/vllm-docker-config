# Digest is the inspected RepoDigest of vllm/vllm-openai:v0.31.0-cu129.
# Do not switch this base to latest or a nightly tag.
FROM vllm/vllm-openai:v0.31.0-cu129@sha256:ff29e51a9457f191deb332b96fa6440584fa941aa2a548449b604e2b1e5e1eeb

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

CMD ["python3", "-m", "uvicorn", "controller.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
