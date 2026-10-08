# 1단계 근거 기록

정적 조사, 실제 CLI 실행, controller 미로드 기동, GPU 모델 검증을 구분한다. 아래 CLI와 기동 확인은 GPU에서 모델을 load하거나 추론한 결과가 아니다. `tests/outputs/probe_support.json`의 `gpu_inference_success`는 `false`다. `tests/outputs/gpu_api.json`은 `not_run`이며 `tests/gpu_api.py --execute`는 실행하지 않았다.

기록 시각: 2026-10-07T19:54:55Z–2026-10-07T19:56:06Z probe. controller 기동 확인은 그 직전이다.

## 실제 CLI

`vllm/vllm-openai:v0.31.0-cu129`는 `torchcodec` 0.17이 `libnvrtc.so.13`과 `libcudart.so.13`에 연결되어 있는데 이미지에는 `libnvrtc.so.12`만 있다. `torchcodec==0.14.0`도 같은 `libnvrtc.so.13`을 요구해서, `.so.12`에 맞는 핀으로 CLI 의존성을 고칠 수 없었다. `.so.12`를 `.so.13`으로 바꾸거나 `torchcodec`을 가짜 모듈로 넣지 않았다.

공식 CUDA 13.0 이미지 `vllm/vllm-openai:v0.31.0`로 바꿨다. digest는 `sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3`다. `latest`와 nightly는 쓰지 않는다. 이 이미지의 `libnvrtc.so.13`은 `/usr/local/cuda/targets/x86_64-linux/lib/libnvrtc.so.13`이다.

최종 런타임 이미지 `vllm-docker-runtime:local`에서 우회 없이 확인했다.

- `vllm --version` 종료 코드 0, 출력 `0.31.0`
- `vllm serve --help` 종료 코드 0
- 이미지 안 버전: torch `2.13.0+cu130`, CUDA `13.0`, vLLM `0.31.0`, transformers `5.17.0`, torchcodec `0.17.0+cu130`
- `torch.cuda.get_device_capability()`는 `[8, 6]`이었다. 장치 확인이지 모델 load가 아니다.
- `ReasoningParserManager.get_reasoning_parser("qwen3")`는 `Qwen3ParserReasoningAdapter`를 import했다. allowlist `reasoning_parsers`는 `["qwen3"]`다. 이것은 parser 모듈 import이며 모델 template의 thinking 분리 성공이 아니다.

베이스 이미지 `Config.Entrypoint`는 `["vllm", "serve"]`이고 `Cmd`는 없었다. 최종 이미지 ENTRYPOINT는 controller다.

```text
python3 -m uvicorn controller.main:app --host 0.0.0.0 --port 8000 --workers 1
```

## controller 미로드 기동

`docker compose config`와 `docker compose up -d --no-build`로 `vllm-runtime`을 기동했다. 호스트 포트는 8000이 이미 사용 중이라 `.env`의 `PUBLIC_PORT=8010`이다. `HOST_MODELS_DIR`는 `/home/kjh/workspace/models/llm/tf`이고 컨테이너 안 경로는 `/models`다.

- `GET /health` 200 `{"controller":"ok"}`
- `GET /control/status` 200 `{"state":"unloaded","residency":"not_resident","active_requests":0,"last_error":null}`
- PID 1은 `/sbin/docker-init`이고 그 자식이 uvicorn이다. `vllm serve` 프로세스는 없었다.
- `POST /control/load`는 호출하지 않았다.

## 체크포인트

호스트 경로는 `/home/kjh/workspace/models/llm/tf/qwen3.8-27b`다. 가중치는 열지 않았고 `loaded`는 `false`다.

- `model_type`은 `qwen3_5`, architecture는 `Qwen3_5ForConditionalGeneration`이다.
- `quant_method`는 `compressed-tensors`, format은 `pack-quantized`, 4bit, group 128, symmetric이다. `kv_cache_scheme`은 `null`이다. `--quantization`은 넣지 않는다. Marlin kernel 선택은 아직 실행으로 확인하지 않았다.
- 레이어는 `linear_attention` 48개와 `full_attention` 16개다.
- processor는 `Qwen3VLProcessor`다. 프로필 `max_pixels` 262144는 preprocessor `longest_edge` 16777216보다 작다.

## 목표 argv

두 프로필 모두 저장소 allowlist로 argv 생성이 된다. 비교 프로필은 기본 `MODEL_PROFILE`이 아니다.

```text
vllm serve /models/qwen3.8-27b
  --host 127.0.0.1
  --port 8080
  --served-model-name qwen3.8-27b
  --dtype bfloat16
  --tensor-parallel-size 1
  --max-model-len 32768
  --max-num-seqs 4
  --max-num-batched-tokens 2048
  --gpu-memory-utilization 0.90
  --kv-cache-dtype fp8_e4m3
  --enable-prefix-caching
  --reasoning-parser qwen3
  --limit-mm-per-prompt {"image":4}
  --mm-processor-kwargs {"max_pixels":262144}
```

`configs/models/qwen3.8-27b-kv-bf16.env`는 같은 argv에서 `--kv-cache-dtype`만 빠진다. 이 생략은 BF16 KV 측정이 끝났다는 뜻이 아니다.

help에 있으나 빼 둔 것:

- `--quantization`. loader와 Marlin 선택은 미확정이다.
- 문자열 `auto`. dtype과 KV choice에는 `auto`가 있으나 넘기지 않는다.
- 생성 기본 상한을 4096 이하로 강제하는 플래그는 확인하지 못했다. `REQUIRE_EXPLICIT_OUTPUT_LIMIT=1`이라 두 출력 필드가 모두 없으면 400이다.

## FP8 KV와 attention

이 항목은 소스 읽기이지 GPU 실행이 아니다. 이미지 `fa_utils.py`의 `flash_attn_supports_kv_cache_dtype`는 FP8 KV를 SM90 계열의 FA3 또는 일부 FA4에 연결한다. 관측한 capability `[8, 6]`은 그 조건이 아니다. 체크포인트에는 KV scale scheme이 없다. 기본 프로필의 `KV_CACHE_DTYPE=fp8_e4m3`는 그대로 둔다. 32K, 외부 동시성 4, FP8 목표를 낮추지 않는다. 실제 backend, scale, Marlin, GDN dtype은 2단계 실행으로 남긴다.

## worker 수명

`v1/executor/multiproc_executor.py`는 `context.Process(...)`로 worker를 만든다. 이 프로세스 트리를 모델 load로 확인하지는 않았다. 컨트롤러는 `vllm serve`를 새 세션에서 띄우고, parent `wait()`만으로 `not_resident`를 선언하지 않는다.

## CPU 테스트와 실행기

`tests/outputs/cpu_control.json`, `tests/outputs/cpu_config.json`, `tests/outputs/cpu_prepare.json`은 fake worker와 fake Docker 결과이며 status는 `passed`다. `tests/gpu_api.py --execute`는 단계 번호로 거절하지 않고 §11.2 케이스를 수행하도록 구현되어 있다. 이 단계에서는 호출하지 않았다.

## InferSwap 연결 초안

측정값은 `README.md`의 연결 예시에 `UNMEASURED`로 두었다. `maxConcurrency`는 `max_num_seqs=4`만으로 4가 아니다.
