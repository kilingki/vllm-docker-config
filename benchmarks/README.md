# 1단계 근거 기록

이 파일은 접근 확인, 이미지 안의 코드, 체크포인트 metadata를 남긴다. CPU 테스트와 이 정적 근거는 GPU 추론 성공이 아니다. `tests/outputs/probe_support.json`의 `status`는 Docker, GPU 장치 목록, 체크포인트 파일, CLI 읽기가 끝났다는 뜻이고 `gpu_inference_success`는 `false`다. 모델 load, 텍스트/이미지 추론, GPU 메모리 측정은 하지 않았다.

기록 시각: 2026-10-07T08:38:51Z–2026-10-07T08:39:39Z. 기준 llama.cpp 런타임 `kilingki/llama-cpp-docker-config` HEAD는 설계와 같은 `e655a2f2ad3979bc5c4a7b4204b80bad08de42e9`였다.

## 접근

- Docker 29.2.1.
- GPU 0은 NVIDIA GeForce RTX 3090다. 컨테이너 안 `torch.cuda.get_device_capability()`는 `[8, 6]`이었다. 이것은 장치 확인이지 모델 load 결과가 아니다.
- 이미지 `vllm/vllm-openai:v0.31.0-cu129`, digest `sha256:ff29e51a9457f191deb332b96fa6440584fa941aa2a548449b604e2b1e5e1eeb`. Dockerfile은 이 digest를 고정한다. `latest`와 nightly는 쓰지 않는다.
- 이미지 안 버전: torch `2.13.0+cu129`, CUDA `12.9`, vLLM `0.31.0`, transformers `5.17.0`.
- `vllm` 콘솔 스크립트는 `libnvrtc.so.13`이 없어 `torchcodec` 로드에 실패했다. 호스트와 이미지에 있는 NVRTC는 `.so.12`다. allowlist 396개는 같은 이미지에서 `make_arg_parser().format_help()`로 읽었다. `configs/vllm_cli_allowlist.json`의 `gpu_success`는 `false`다.

## 체크포인트

호스트 경로는 `/home/kjh/workspace/models/llm/tf/qwen3.8-27b`다. Compose는 `HOST_MODELS_DIR`를 `/models`에 마운트하고 프로필은 `/models/qwen3.8-27b`를 읽으므로, 이 스냅샷의 부모 디렉터리가 `HOST_MODELS_DIR`다. 가중치는 열지 않았다.

- `model_type`은 `qwen3_5`, architecture는 `Qwen3_5ForConditionalGeneration`이다. 텍스트 config dtype은 `bfloat16`, `max_position_embeddings`는 262144다. 프로필의 `max_model_len` 32768은 이 값보다 작게 유지한다.
- 레이어는 `linear_attention` 48개와 `full_attention` 16개다. `mamba_ssm_dtype`은 `float32`다. 양자화 `ignore`에는 `linear_attn` 48개와 그 `norm`이 들어 있다.
- `quant_method`는 `compressed-tensors`, format은 `pack-quantized`, 가중치는 4bit, group 128, symmetric이다. `kv_cache_scheme`은 `null`이다. 이 metadata만으로 Marlin kernel이 선택된다고 보지 않는다. `--quantization`은 넣지 않는다.
- processor는 `Qwen3VLProcessor` / `Qwen2VLImageProcessorFast`다. `preprocessor_config.json`의 `size`는 `shortest_edge` 65536, `longest_edge` 16777216이다. 프로필의 `max_pixels` 262144는 그 상한보다 작다. vision config는 `qwen3_5`, depth 27이다.
- 가중치 파일은 `model.safetensors` 17646891544바이트, `model-visual-bf16.safetensors` 921497224바이트, `model-mtp-bf16.safetensors` 849400424바이트다. 이 크기는 GPU peak가 아니다.

## 목표 argv

아래 플래그 이름은 이미지 help에 있다. `--reasoning-parser` 이름 `qwen3`는 help의 선택 목록에 없고, 이미지 `vllm/reasoning/__init__.py`의 `_REASONING_PARSERS_TO_REGISTER`에만 있다. allowlist의 `reasoning_parsers`는 그래서 `null`이고, 컨트롤러는 그 플래그를 확인되지 않은 값으로 거절한다. 프로필 값은 바꾸지 않았다.

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
  --limit-mm-per-prompt {"image":4}
  --mm-processor-kwargs {"max_pixels":262144}
```

help에 있으나 빼 둔 것:

- `--quantization`. loader와 Marlin 선택은 미확정이다.
- 문자열 `auto`. dtype과 KV choice에는 `auto`가 있으나 넘기지 않는다.
- `--reasoning-parser qwen3`. 레지스트리에는 있고 help choice에는 없다.
- `configs/models/qwen3.8-27b-kv-bf16.env`는 비교 프로필이다. 기본 `MODEL_PROFILE`이 아니며 `--kv-cache-dtype`을 생략한다. 이 생략은 BF16 KV 측정이 끝났다는 뜻이 아니다.
- 생성 기본 상한을 4096 이하로 강제하는 플래그는 확인하지 못했다. `REQUIRE_EXPLICIT_OUTPUT_LIMIT=1`이라 두 출력 필드가 모두 없으면 400이다.

## FP8 KV와 attention

이미지 `vllm/v1/attention/backends/fa_utils.py`의 `flash_attn_supports_kv_cache_dtype`은 FP8 KV를 SM90의 FA3 또는 일부 FA4, 그리고 SM100의 FA4에만 연결한다. SM86은 그 조건에 없다. 체크포인트에는 KV scale scheme이 없다.

FlashAttention으로 FP8 KV를 강제하지 않는다. Triton이나 FlashInfer가 SM86에서 FP8 KV를 대신하는지는 미확정이다. 기본 프로필의 `KV_CACHE_DTYPE=fp8_e4m3`는 그대로 둔다. 32K, 외부 동시성 4, FP8 목표를 낮추지 않는다. scale calibration, decoder/vision attention backend, GDN state가 실행 중 `float32`로 남는지는 미확정이다.

## worker 수명과 upstream 완료

`v1/executor/multiproc_executor.py`는 `context.Process(...)`로 worker를 만든다. `get_mp_context()`의 기본 방법은 fork이고, 이미지 코드는 WSL이면 `VLLM_WORKER_MULTIPROC_METHOD`를 `spawn`으로 덮어쓴다. 그 executor 경로에서 `setsid`는 보이지 않았다. 이 프로세스 트리를 모델 load로 확인하지는 않았다.

컨트롤러는 `vllm serve`를 새 세션에서 띄우고, 종료 때 그 process group에 TERM 후 KILL을 보낸다. parent `wait()`만으로 `not_resident`를 선언하지 않는다. 종료 시점에 다시 잡은 자손 PID도 본다. 그룹 밖 PID에는 개별 signal을 보내지 않는다.

정상 upstream 응답을 끝까지 읽으면 active를 한 번 줄인다. transport 오류는 실행 종료로 보지 않고 active를 유지한 채 `EXECUTION_UNCONFIRMED`와 failed를 남긴다. pinned vLLM이 응답 EOF와 GPU 실행 종료를 같게 보는지는 미확정이다. vLLM `/health` 200이 engine 초기화 완료와 같은지도 미확정이므로, readiness는 `/health`와 `/v1/models`의 served model 이름을 함께 본다.

## CPU 테스트

`tests/outputs/cpu_control.json`, `tests/outputs/cpu_config.json`, `tests/outputs/cpu_prepare.json`은 fake worker와 fake Docker 결과이며 status는 `passed`다. `tests/outputs/gpu_api.json`은 전부 `not_run`이다. `tests/gpu_api.py`는 2단계 실행용이며 이 단계에서는 모델을 load하지 않는다.

## InferSwap 연결 초안

아래 값은 측정값이 아니다. `UNMEASURED`를 0이나 가중치 파일 크기로 바꾸지 않는다. `maxConcurrency`는 `max_num_seqs=4`만으로 4가 아니다.

```json
{
  "baseURL": "http://127.0.0.1:8000",
  "prepare": {"argv": ["./prepare-inferswap"]},
  "inferencePath": "/v1/chat/completions",
  "servedModelName": "qwen3.8-27b",
  "checkpointPath": "/home/kjh/workspace/models/llm/tf/qwen3.8-27b",
  "modelId": "UNMEASURED",
  "hostPort": "PUBLIC_PORT from .env.example is 8000; pick a port that does not collide",
  "resourceProfile": {
    "profileId": "UNMEASURED",
    "loadPeakBytes": "UNMEASURED",
    "inferencePeakBytes": "UNMEASURED",
    "unloadedResidualBytes": "UNMEASURED",
    "maxConcurrency": "UNMEASURED",
    "limits": {
      "maxBodyBytes": 33554432,
      "maxOutputTokens": 4096
    }
  }
}
```

`baseURL`의 8000은 `.env.example`의 예시 포트다. served name `qwen3.8-27b`는 위 로컬 스냅샷에 대응시킨다. 컨트롤러 한도 33554432와 4096은 접수 제한이지 GPU peak가 아니다.
