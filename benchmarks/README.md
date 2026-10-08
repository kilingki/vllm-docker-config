# 독립 GPU 측정과 InferSwap 연결 정보

이 파일은 설계 §11.3 실측과 §12 연결 정보만 담는다. 수치는 2026-10-08 이 머신에서 `tests/gpu_api.py --execute`와 BF16 비교 로드로 읽은 값이다. 모델 카드 수치는 넣지 않았다.

## 환경

| 항목 | 관측 |
|---|---|
| GPU | NVIDIA GeForce RTX 3090, 24576 MiB, driver 591.86, SM86 |
| 호스트 | WSL2 kernel 5.15.153.1-microsoft-standard-WSL2 |
| Docker | 29.2.1 |
| 이미지 | `vllm-docker-runtime:local`, id `sha256:936f9a6d71feb5179f0bed3275c28a0eb142d743a84e72e7e39d6ecd8c9b8402` |
| 런타임 | vLLM 0.31.0, torch 2.13.0+cu130, CUDA 13.0, transformers 5.17.0 |
| 체크포인트 | `/home/kjh/workspace/models/llm/tf/qwen3.8-27b` (컨테이너 `/models/qwen3.8-27b`) |
| 저장 형식 | compressed-tensors, pack-quantized, 4bit, group 128, symmetric. `kv_cache_scheme`은 없다 |
| 아키텍처 | `Qwen3_5ForConditionalGeneration` |

로드 전 GPU baseline은 `nvidia-smi` memory.used 1007 MiB이고 compute 프로세스는 없었다. 아래 peak와 residual은 같은 호스트 used 값이다. 런타임에 귀속할 때는 이 baseline을 빼서 적는다. 1 MiB는 1048576 bytes다.

## 기본 프로필 argv

`MODEL_PROFILE=qwen3.8-27b`. 로그에 찍힌 기동 명령은 다음과 같다.

```text
vllm serve /models/qwen3.8-27b --host 127.0.0.1 --port 8080 --served-model-name qwen3.8-27b --dtype bfloat16 --tensor-parallel-size 1 --max-model-len 32768 --max-num-seqs 4 --max-num-batched-tokens 2048 --gpu-memory-utilization 0.90 --kv-cache-dtype fp8_e4m3 --enable-prefix-caching --reasoning-parser qwen3 --limit-mm-per-prompt {"image":4} --mm-processor-kwargs {"max_pixels":262144}
```

`speculative_config=None`이었다. 체크포인트에 MTP 가중치가 있어도 기본 argv는 speculative를 켜지 않았고, 비활성 플래그를 추가하지 않았다.

## backend

| 항목 | 로그 |
|---|---|
| Linear kernel | `Using MarlinLinearKernel for CompressedTensorsWNA16`. 일부 shape은 Marlin thread-tile padding 경고가 있다 |
| KV | `Using fp8_e4m3 data type to store kv cache`. scaling factor가 없으면 정확도가 떨어질 수 있다는 경고가 함께 있다. 체크포인트에 KV scheme이 없으므로 calibrated scale이 아니다 |
| Decoder attention | FlashInfer. 후보 `FLASHINFER`, `TRITON_ATTN`. query는 bfloat16, `kv_cache_dtype=torch.float8_e4m3fn`, `arch=sm86` |
| Vision attention | FlashAttention (`vit`, `MMEncoderAttention`) |
| GDN | prefill `Triton/FLA` (`head_k_dim=128`), decode `cuda` |
| KV 용량 | 한 로드에서 72817 tokens, 32768 입력 기준 2.22x. 다른 로드에서 74031 tokens, 2.26x. 이것은 KV 풀 용량이고 외부 동시성 한도가 아니다 |
| Prefix | prefix caching on, Mamba cache mode `align`. attention block은 이 로드에서 1568 |

OOM은 없었다. 케이스 실패로 잡힌 preemption 카운터 증가는 없었다.

## 시간, peak, residual

측정은 호스트 `nvidia-smi`를 0.5초 간격으로 읽고, control status는 load/unload 동안 약 5초마다 호출했다. status 샘플은 1–8ms였고 10초 한도 안이었다.

| 항목 | 값 |
|---|---|
| compile cache가 비어 있던 첫 프로세스 기동 | 229.3s, 호스트 peak 22930 MiB |
| 통과한 실행의 load (compile cache 있음) | 51.4s, 호스트 peak 23035 MiB |
| 같은 실행의 다음 load | 54.9s, 53.2s. 중복 load 한 번은 0.002s no-op |
| unload | 세 번의 unload 뒤 호스트 used 970, 970, 970 MiB. worker 프로세스는 없었다 |
| 추론 구간 호스트 peak | 23480 MiB |

baseline 1007 MiB를 뺀 귀속값:

| 필드 | bytes |
|---|---|
| loadPeakBytes | 23098032128 (23035 − 1007 MiB) |
| inferencePeakBytes | 23564648448 (23480 − 1007 MiB) |
| unloadedResidualBytes | 0 |

unload 호스트 used 970 MiB는 baseline 1007 MiB보다 낮다. 이 런타임이 unload 뒤에 남긴 추가 점유로 0을 적는다. 호스트 전체 970 MiB를 residual로 쓰지 않는다.

## 요청

reasoning parser `qwen3` 때문에 본문이 `message.reasoning`에 있고 `content`가 비는 응답이 있다.

| 요청 | 관측 |
|---|---|
| 짧은 한국어, 비스트림 | HTTP 200, latency 1.55s, TTFT 1.55s, 32.3 output tok/s. `content`는 `확인` |
| 짧은 한국어, 스트림 | TTFT 0.51s, latency 0.80s. 스트림 output tok/s는 UNMEASURED |
| 1x1 이미지 1장 | latency 1.74s, 36.7 tok/s. reasoning은 단색 이미지로 서술 |
| 1x1 이미지 2장 | latency 1.72s, 37.2 tok/s |
| 표 이미지 | latency 1.72s, 64토큰이 모두 reasoning. 표의 행을 보려다 64토큰에서 끝났다. 네 숫자를 확정한 문장은 없다. 목표 argv는 바꾸지 않았다 |
| 32K에 가까운 입력 | 토크나이저 추정 30428. usage `prompt_tokens` 30494, `completion_tokens` 256, 합 30750. HTTP 200. 한도 32768을 넘기지 않았다 |
| prefix cache | 2303토큰 prefix를 두 번. usage `cached_tokens`는 0 (세부 usage 비활성). `vllm:prefix_cache_hits`는 1568 증가 |
| 동시성 1 wall | text 0.93s, image 1.82s, mixed 0.83s |
| 동시성 2 wall | text 1.12s, image 1.83s, mixed 1.75s. `vllm:num_requests_running` >= 2 |
| 동시성 4 wall | text 1.55s, image 2.03s, mixed 2.01s. running >= 2 |
| 최대 multimodal | 요청 4개, 각 이미지 4장, `max_pixels` 262144, `max_tokens` 256. HTTP 200, wall 7.86s |

동시성 wave의 요청별 TTFT와 prefill 시간은 UNMEASURED다. 긴 입력의 TTFT도 UNMEASURED다.

## BF16 KV 비교

비교 프로필 `qwen3.8-27b-kv-bf16`만 잠시 올렸다. `KV_CACHE_DTYPE`은 비어 있고 기본 프로필의 `fp8_e4m3`는 그대로 두었다. 비교가 끝난 뒤 컨테이너는 `MODEL_PROFILE=qwen3.8-27b`, `KV_CACHE_DTYPE=fp8_e4m3`, state `unloaded`로 돌아왔다.

| 항목 | 값 |
|---|---|
| load | 171.9s, 호스트 peak 22598 MiB, baseline 1001 MiB |
| loadPeakBytes (비교) | 22646095872 (22598 − 1001 MiB) |
| decoder | FlashAttention 2. 후보에 FLASH_ATTN, FLASHINFER, TRITON_ATTN, FLEX_ATTENTION |
| Marlin | CompressedTensorsWNA16에 MarlinLinearKernel |
| KV 용량 | 41642 tokens, 32768 기준 1.27x |
| 짧은 문장 | HTTP 200, latency 1.03s. 응답 문장 텍스트는 UNMEASURED (`content`가 비어 있었고 reasoning은 이 비교 기록에 남지 않음) |
| unload | 0.65s, 호스트 used 970 MiB |

## §12 연결 예시

InferSwap에는 등록하지 않았다. 이 환경에서 쓸 값은 아래다.

```yaml
baseURL: http://127.0.0.1:8010
inferencePath: /v1/chat/completions
servedModel: qwen3.8-27b
alias: qwen3.8-27b
prepare:
  argv:
    - vllm
    - serve
    - /models/qwen3.8-27b
    - --host
    - 127.0.0.1
    - --port
    - "8080"
    - --served-model-name
    - qwen3.8-27b
    - --dtype
    - bfloat16
    - --tensor-parallel-size
    - "1"
    - --max-model-len
    - "32768"
    - --max-num-seqs
    - "4"
    - --max-num-batched-tokens
    - "2048"
    - --gpu-memory-utilization
    - "0.90"
    - --kv-cache-dtype
    - fp8_e4m3
    - --enable-prefix-caching
    - --reasoning-parser
    - qwen3
    - --limit-mm-per-prompt
    - '{"image":4}'
    - --mm-processor-kwargs
    - '{"max_pixels":262144}'
resourceProfile:
  profileId: qwen3.8-27b-fp8-kv
  loadPeakBytes: 23098032128
  inferencePeakBytes: 23564648448
  unloadedResidualBytes: 0
  maxConcurrency: 4
  limits:
    maxBodyBytes: 33554432
    maxOutputTokens: 4096
    maxModelLen: 32768
    maxImagesPerPrompt: 4
    maxPixels: 262144
```

`maxConcurrency: 4`는 이미지 4장 × 동시 요청 4가 통과했기 때문이다. `max_num_seqs=4` 설정만으로 적은 값이 아니다. KV 풀은 32768 토큰 요청을 약 2.2개까지 겹칠 수 있다고 로그에 나왔고, 그 조건의 4-way는 이번 측정에 없다.

요청별 TTFT가 비어 있는 항목과 BF16 응답 문장은 UNMEASURED다.
