# vLLM Docker Runtime

Single-container Docker runtime for LLM and VLM inference with vLLM. The image contains the runtime only. Model weights stay on the host.

One published base URL is one logical model. The container starts unloaded. `POST /control/load` starts `vllm serve`, and OpenAI-compatible `/v1/*` requests are proxied to it. InferSwap uses the same control API as the other runtimes in this set. Load and unload bodies are empty or `{}`, and `GET /control/status` is the readiness source.

This repository is [MIT](LICENSE). vLLM is licensed separately. Model weights are not part of this repository.

## Contents

- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Model control](#model-control)
- [API examples](#api-examples)
- [Configuration](#configuration)
- [Multimodal](#multimodal)
- [InferSwap connection](#inferswap-connection)
- [Limitations](#limitations)
- [Tests](#tests)
- [Measurements](#measurements)
- [License](#license)

## Architecture

```text
External Client / InferSwap
    |
    |  OpenAI-compatible API + control API
    |  http://localhost:<PUBLIC_PORT>
    v
controller (FastAPI / uvicorn on :8000)
    |-- GET  /health
    |-- POST /control/load
    |-- POST /control/unload
    |-- GET  /control/status
    |-- /v1/*  --->  proxy
                      |
                      v
                 vllm serve 127.0.0.1:8080
                 (child process group, not published)
```

Compose sets `init: true`, so Docker init is PID 1 and the controller is the service process. Inside the container the controller binds `0.0.0.0:8000`. `PUBLIC_PORT` only changes the host publish port.

```text
container lifecycle  !=  model lifecycle

model load    = vllm serve process start
model unload  = vllm serve process termination
```

The container stays up while unloaded. Model state moves `unloaded` → `loading` → `ready` → `unloading` → `unloaded`. A load or unload failure sets `failed`, together with `residency` `resident`, `not_resident`, or `unknown`. An unexpected worker exit sets `failed`. `/health` only reports that the controller process is up. Model readiness is `GET /control/status`. A restart does not load the model again.

## Project structure

- `docker-compose.yml`: one `vllm-runtime` service
- `prepare-inferswap`: start the container when it is down, or leave a running container unchanged
- `.env.example`: host port, model directory, and profile name
- `configs/common.env`: internal bind, output limit, and load/unload timeouts
- `configs/models/<name>.env`: one model profile; the default example is `qwen3.8-27b`
- `controller/`: control API and `/v1` proxy
- `tests/test_cpu_*.py`: control contract tests, no GPU
- `tests/gpu_api.py`: independent GPU/API verification
- `tests/infer_qwen.py`: reasoning-effort and image checks

## Requirements

- NVIDIA GPU
- NVIDIA driver
- Docker and Docker Compose
- NVIDIA Container Toolkit

The verified image is `vllm/vllm-openai:v0.31.0`. This environment is Windows 11 + WSL2 Ubuntu 24.04 with an RTX 3090 24GB, single GPU.

## Quick start

1. Copy the example environment file and set the host directory that contains the Hugging Face model and the publish port.

```bash
cp .env.example .env
```

```env
PUBLIC_PORT=8010
HOST_MODELS_DIR=/path/to/models
MODEL_PROFILE=qwen3.8-27b
```

`HOST_MODELS_DIR` is a host path. Compose mounts it read-only at `/models`. The default profile reads the container path `/models/qwen3.8-27b`. That directory must contain the shards, config, tokenizer, and processor together.

2. Build the image and start the container. A new container does not load the model.

```bash
docker compose build
docker compose up -d
```

`./prepare-inferswap` starts the container with the equivalent of `up -d --no-build --no-recreate` when it is missing or stopped. It does not build an image and does not load the model. A newly started controller must report `unloaded` / `not_resident` / `active_requests=0`. If the container is already running and `GET /control/status` succeeds, the script exits without restarting or unloading it.

3. Confirm the unloaded state, load the profile, call chat, then unload.

```bash
curl -s http://localhost:8010/control/status
curl -s -X POST http://localhost:8010/control/load \
  -H "Content-Type: application/json" \
  -d '{}'
curl -s http://localhost:8010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 64
  }'
curl -s -X POST http://localhost:8010/control/unload
```

Add another model by creating `configs/models/<name>.env` and setting `MODEL_PROFILE` in `.env`, then recreate the container. `POST /control/load` does not take a model name.

## Model control

InferSwap talks to this runtime through:

```text
GET  /control/status
POST /control/load
POST /control/unload
/v1/*
```

`GET /health` is the container healthcheck. InferSwap does not use it for readiness. The healthcheck passes while the model is unloaded.

```json
{
  "controller": "ok"
}
```

Control requests take an empty body or `{}`. Any other JSON object is HTTP 400 `BAD_REQUEST`.

`GET /control/status` returns exactly four fields:

```json
{
  "state": "unloaded",
  "residency": "not_resident",
  "active_requests": 0,
  "last_error": null
}
```

`state` is `unloaded`, `loading`, `ready`, `unloading`, or `failed`. `residency` is `resident`, `not_resident`, or `unknown`. `active_requests` counts accepted inference requests. `GET /v1/models` is not counted. Other `/v1/*` requests count as one until upstream finishes them. A request that was already accepted runs to completion even if the client disconnects first. `last_error` is `null` after a successful load or unload, or `{"code","message"}` after a lifecycle failure.

`POST /control/load` returns 200 with `state=ready` and `residency=resident` when the server can accept inference. Calling it again while ready does not start a second `vllm serve`. `POST /control/unload` returns 200 with `state=unloaded`, `residency=not_resident`, and `active_requests=0` after the workers have exited. Calling it again in that state is a no-op. Unload returns 409 `BUSY` while `active_requests` is greater than 0. A load during unload, or an unload during load, returns 409 `LIFECYCLE_CONFLICT`.

Control errors use this body:

```json
{
  "error": {
    "code": "BUSY",
    "message": "runtime has active inference requests"
  }
}
```

`code` is `BAD_REQUEST`, `BUSY`, `LIFECYCLE_CONFLICT`, `LOAD_FAILED`, `UNLOAD_FAILED`, or `STATUS_FAILED`.

If `/v1/*` is called while `state` is not `ready`, the controller returns 503 `{"detail":"model is not ready"}` and does not call upstream.

A missing model directory is HTTP 500 `LOAD_FAILED`. `state` becomes `failed`, `residency` is `not_resident`, and `vllm serve` is not left running. Load can be retried.

The external admission limit is 4. A request past that limit is 429 `{"detail":"too many active requests"}`. Rejected requests are not counted as active.

## API examples

Chat. Send `max_tokens` or `max_completion_tokens` as an integer from 1 to 4096. If both fields are present, they must be equal. If both are omitted, the controller returns 400. A body that passes is forwarded unchanged.

```bash
curl -s http://localhost:8010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 64,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

Streaming:

```bash
curl -N http://localhost:8010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 64,
    "stream": true,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

Image. Reproduction tests use a base64 data URL. The profile cap is 4 images per request.

```bash
python3 - <<'PY'
import base64, json, os, urllib.request

path = os.environ.get("IMAGE_FILE", "photo.jpg")
with open(path, "rb") as f:
    b64 = base64.b64encode(f.read()).decode("ascii")
payload = {
    "model": "qwen3.8-27b",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ],
    }],
    "max_tokens": 128,
}
req = urllib.request.Request(
    "http://localhost:8010/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
print(urllib.request.urlopen(req).read().decode())
PY
```

## Configuration

Model name and inference options come from the profile. They are not hardcoded in Python or the Dockerfile.

```text
.env                         # HOST_MODELS_DIR, MODEL_PROFILE, PUBLIC_PORT
configs/common.env           # HOST, PORT, timeouts, body and output limits
configs/models/<name>.env    # default example: qwen3.8-27b.env
```

The default profile `qwen3.8-27b` is the Qwen3.8-27B VLM. `max_model_len` is 32768, `max_num_seqs` is 4, the KV dtype is `fp8_e4m3`, `max_pixels` is 262144, and the generation cap is 4096. Only `configs/common.env` sets `HOST` and `PORT`. A model env file that overrides either value is a configuration error.

`LOAD_TIMEOUT_SEC` defaults to 600. `UNLOAD_TIMEOUT_SEC` defaults to 30. Measured load and unload times are in [benchmarks/README.md](benchmarks/README.md).

## Multimodal

The default input is text and `image_url` on `/v1/chat/completions`. Video decoding and frame sampling are not a default input of this runtime. Several images are several content parts.

The verified caps are 4 images per request, processor `max_pixels` 262144, 4096 generated tokens, 4 concurrent requests, and an HTTP body of 32MiB (33554432 bytes). A larger body is 413 `{"detail":"request body too large"}`.

## InferSwap connection

This project provides the control API, `prepare-inferswap`, the inference path, and measured resource numbers. Registering the InferSwap config and running the VLM → ASR → FA → LLM sequence belong to InferSwap.

These are the connection values confirmed on this machine. `prepare.argv` is the absolute path of `prepare-inferswap`, and nothing else. `inferencePeakBytes` is the host peak of 23777 MiB during the four-request max window, minus the pre-load baseline of 1120 MiB.

```yaml
baseURL: http://127.0.0.1:8010
inferencePath: /v1/chat/completions
servedModel: qwen3.8-27b
alias: qwen3.8-27b
prepare:
  argv:
    - /home/kjh/workspace/vllm-docker-config/prepare-inferswap
resourceProfile:
  profileId: qwen3.8-27b-fp8-kv
  loadPeakBytes: 23098032128
  inferencePeakBytes: 23757586432
  unloadedResidualBytes: 0
  maxConcurrency: 4
  limits:
    maxBodyBytes: 33554432
    maxOutputTokens: 4096
    maxModelLen: 32768
    maxImagesPerPrompt: 4
    maxPixels: 262144
```

`maxConcurrency: 4` is the value after four concurrent requests passed with 4 images, 262144 pixels, and 4096 output tokens. It is not taken from `max_num_seqs=4` alone. The source JSON is `tests/outputs/inferswap_connection.json`.

## Limitations

- Audio input and video decoding are out of scope. Send selected frames as `image_url` parts.
- Remote image URLs are fetched by vLLM. A download failure comes back as an upstream error.
- The container is not privileged, and the Docker socket is not mounted.
- There is no authentication. Keep the published port on a private interface.
- Sleep Mode, multiple GPUs, LoRA, and MTP are not operating paths of this runtime.

## Tests

Contract tests, no GPU:

```bash
python3 -m pytest tests/test_cpu_config.py tests/test_cpu_control.py tests/test_cpu_prepare.py
```

These need a running container:

```bash
python3 tests/gpu_api.py --execute --base-url http://127.0.0.1:8010 --profile qwen3.8-27b --service vllm-runtime
python3 tests/infer_qwen.py
```

- `tests/gpu_api.py`: load, text, image, streaming, concurrency, long context, max multimodal, disconnect, lifecycle, failure, and restart. `--max-only` sends only the four max-condition requests again.
- `tests/infer_qwen.py`: four reasoning-effort text cases and two cat images. It requires `tests/test-cat-512.jpg` and `tests/test-cat.jpg`. Results go to `tests/outputs/<timestamp>-qwen-infer/INDEX.md`. A passing run leaves the model loaded.
- The default `BASE_URL` is `http://127.0.0.1:8010`.

## Measurements

Environment, argv, peaks, request observations, the BF16 KV comparison, and the connection example are in [benchmarks/README.md](benchmarks/README.md).

## License

[MIT](LICENSE)
