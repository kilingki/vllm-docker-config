# Independent GPU measurements and InferSwap connection

This file holds the §11.3 measurements and the §12 connection values. The numbers were read on this machine on 2026-10-08 from `tests/gpu_api.py --execute`, the BF16 comparison load, and a max-condition-only rerun with `tests/gpu_api.py --max-only`. Model-card numbers are not copied here.

## Environment

| Item | Observed |
|---|---|
| GPU | NVIDIA GeForce RTX 3090, 24576 MiB, driver 591.86, SM86 |
| Host | WSL2 kernel 5.15.153.1-microsoft-standard-WSL2 |
| Docker | 29.2.1 |
| Image | `vllm-docker-runtime:local`, id `sha256:936f9a6d71feb5179f0bed3275c28a0eb142d743a84e72e7e39d6ecd8c9b8402` |
| Runtime | vLLM 0.31.0, torch 2.13.0+cu130, CUDA 13.0, transformers 5.17.0 |
| Checkpoint | `/home/kjh/workspace/models/llm/tf/qwen3.8-27b` (container `/models/qwen3.8-27b`) |
| Storage format | compressed-tensors, pack-quantized, 4bit, group 128, symmetric. There is no `kv_cache_scheme` |
| Architecture | `Qwen3_5ForConditionalGeneration` |

The GPU baseline before the load measurement was `nvidia-smi` memory.used 1007 MiB, and no compute process was present. Load peak and residual below subtract that baseline. The max-condition rerun had its own pre-load baseline of 1120 MiB. The inference peak subtracts 1120 MiB from that window. 1 MiB is 1048576 bytes.

## Default profile argv

`MODEL_PROFILE=qwen3.8-27b`. The start command recorded in the log is:

```text
vllm serve /models/qwen3.8-27b --host 127.0.0.1 --port 8080 --served-model-name qwen3.8-27b --dtype bfloat16 --tensor-parallel-size 1 --max-model-len 32768 --max-num-seqs 4 --max-num-batched-tokens 2048 --gpu-memory-utilization 0.90 --kv-cache-dtype fp8_e4m3 --enable-prefix-caching --reasoning-parser qwen3 --limit-mm-per-prompt {"image":4} --mm-processor-kwargs {"max_pixels":262144}
```

`speculative_config=None`. The checkpoint may contain MTP weights, but the default argv did not enable speculative decoding and did not add a flag to turn it off.

## Backend

| Item | Log |
|---|---|
| Linear kernel | `Using MarlinLinearKernel for CompressedTensorsWNA16`. Some shapes warn about Marlin thread-tile padding |
| KV | `Using fp8_e4m3 data type to store kv cache`. A warning says accuracy may drop when no scaling factor is present. The checkpoint has no KV scheme, so this is not a calibrated scale |
| Decoder attention | FlashInfer. Candidates `FLASHINFER`, `TRITON_ATTN`. Query is bfloat16, `kv_cache_dtype=torch.float8_e4m3fn`, `arch=sm86` |
| Vision attention | FlashAttention (`vit`, `MMEncoderAttention`) |
| GDN | prefill `Triton/FLA` (`head_k_dim=128`), decode `cuda` |
| KV capacity | One load reported 72817 tokens, 2.22x for a 32768-token input. Another load reported 74031 tokens, 2.26x. This is KV pool capacity, not the external concurrency limit |
| Prefix | prefix caching on, Mamba cache mode `align`. The attention block on this load was 1568 |

There was no OOM. No case failure was a preemption-counter increase.

## Time, peak, and residual

The host `nvidia-smi` was sampled every 0.5s. Control status was called about every 5s during load and unload. Status samples were 1–8ms, inside the 10s limit.

| Item | Value |
|---|---|
| First process start with an empty compile cache | 229.3s, host peak 22930 MiB |
| Load on the passing run (compile cache present) | 51.4s, host peak 23035 MiB |
| Later loads in the same run | 54.9s, 53.2s. One duplicate load was a 0.002s no-op |
| Unload | After three unloads, host used was 970, 970, 970 MiB. No worker process remained |
| Host peak during inference | 23777 MiB. Four-request max window, `nvidia-smi` every 0.5s. Pre-load baseline 1120 MiB |

Load and residual subtract the 1007 MiB baseline. The inference peak subtracts the 1120 MiB baseline of the max-condition window.

| Field | bytes |
|---|---|
| loadPeakBytes | 23098032128 (23035 − 1007 MiB) |
| inferencePeakBytes | 23757586432 (23777 − 1120 MiB) |
| unloadedResidualBytes | 0 |

Host used after unload, 970 MiB, is below the 1007 MiB baseline. The extra occupancy this runtime left after unload is recorded as 0. The host-wide 970 MiB is not the residual.

## Requests

The `qwen3` reasoning parser can put the body in `message.reasoning` and leave `content` empty.

| Request | Observed |
|---|---|
| Short Korean, non-streaming | HTTP 200, latency 1.55s, 32.3 output tok/s. `content` is `확인` |
| Short Korean, streaming | TTFT 0.598s, latency 1.35s, remeasured with the per-event reader. Stream output tok/s is UNMEASURED |
| One 1x1 image | latency 1.74s, 36.7 tok/s. Reasoning describes a solid-color image |
| Two 1x1 images | latency 1.72s, 37.2 tok/s |
| Table image | latency 17.06s, `max_tokens` 2048, `finish_reason=stop`. `content` is `1 2 3 4` |
| Near-32K input | Tokenizer estimate 30428. Usage `prompt_tokens` 30494, `completion_tokens` 256, sum 30750. HTTP 200. Did not exceed the 32768 limit |
| Prefix cache | The same 2303-token prefix twice. Usage `cached_tokens` is 0 (detailed usage disabled). `vllm:prefix_cache_hits` increased by 1568 |
| Concurrency 1 wall | text 0.93s, image 1.82s, mixed 0.83s |
| Concurrency 2 wall | text 1.12s, image 1.83s, mixed 1.75s. `vllm:num_requests_running` >= 2 |
| Concurrency 4 wall | text 1.55s, image 2.03s, mixed 2.01s. running >= 2 |
| Max multimodal | 4 copies of one request, 4 images each, 262144 pixels after processing, `prompt_tokens` 27290, `max_tokens` and `min_tokens` 4096. Prefix caching was on, and a calibration request with the same prompt ran first. All four HTTP 200, `completion_tokens` 4096, `finish_reason=length`. Wall about 134s. Preemption 0. Host peak 23777 MiB |

Per-request TTFT and prefill time for the concurrency waves are UNMEASURED. TTFT for the long input is UNMEASURED.

## BF16 KV comparison

Only the comparison profile `qwen3.8-27b-kv-bf16` was loaded for this pass. `KV_CACHE_DTYPE` was empty, and the default profile's `fp8_e4m3` was left unchanged. After the comparison the container was back to `MODEL_PROFILE=qwen3.8-27b`, `KV_CACHE_DTYPE=fp8_e4m3`, state `unloaded`.

| Item | Value |
|---|---|
| Load | 171.9s, host peak 22598 MiB, baseline 1001 MiB |
| loadPeakBytes (comparison) | 22646095872 (22598 − 1001 MiB) |
| Decoder | FlashAttention 2. Candidates FLASH_ATTN, FLASHINFER, TRITON_ATTN, FLEX_ATTENTION |
| Marlin | MarlinLinearKernel for CompressedTensorsWNA16 |
| KV capacity | 41642 tokens, 1.27x at 32768 |
| Table image | Remeasured with the table-quality request. HTTP 200, latency 22.52s, `finish_reason=stop`. `content` is `1 2 3 4` |
| Unload | 0.65s, host used 970 MiB |

## §12 connection example

Nothing was registered in InferSwap. The values for this environment are:

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

`maxConcurrency: 4` is the value from the passing max-condition 4-way. The four requests were copies of one body: 4 images, 262144 pixels after processing, `prompt_tokens` 27290, 4096 output tokens, HTTP 200, `finish_reason=length`, and no preemption. A calibration request with that same prompt ran immediately before, and prefix caching was on. It is not taken from the `max_num_seqs=4` setting alone. The log said the KV pool can overlap about 2.2 requests of 32768 tokens. Four distinct prompts near that length are outside this observation. This pass is the shared-prefix four-request observation above, not that pool log.

`prepare.argv` is the absolute path of this repository's `prepare-inferswap`, and nothing else. The source of record is `tests/outputs/inferswap_connection.json`.

Per-request TTFT and prefill time for the concurrency waves, and TTFT for the long input, are UNMEASURED.
