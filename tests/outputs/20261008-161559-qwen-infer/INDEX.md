# Qwen3.8 추론 테스트

- base: `http://127.0.0.1:8010`
- model: `qwen3.8-27b`
- LLM prompt: What is 17 times 23? Reply with the number only in the final answer.
- VLM prompt: Describe this image in one sentence. Include the kind of animal and its color.

| case | HTTP | e2e_s | prompt_tok | completion_tok | prompt_tok/s | gen_tok/s | prompt_ms | predicted_ms | finish |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| llm-off | 200 | 0.26 | 32 | 4 |  |  |  |  | stop |
| llm-low | 200 | 1.86 | 60 | 63 |  |  |  |  | stop |
| llm-medium | 200 | 1.81 | 30 | 63 |  |  |  |  | stop |
| llm-xhigh | 200 | 1.62 | 72 | 57 |  |  |  |  | stop |
| vlm-cat-512 | 200 | 0.93 | 286 | 24 |  |  |  |  | stop |
| vlm-cat | 200 | 0.94 | 270 | 24 |  |  |  |  | stop |

## llm-off

- kwargs: `{"enable_thinking": false}`

### content

391

### reasoning_content

(empty)

## llm-low

- kwargs: `{"reasoning_effort": "low"}`

### content



391

### reasoning_content

The user asks me to calculate 17 × 23 and reply with the number only.

17 × 23 = 17 × 20 + 17 × 3 = 340 + 51 = 391


## llm-medium

- kwargs: `{"reasoning_effort": "medium"}`

### content



391

### reasoning_content

The user wants me to calculate 17 × 23 and reply with just the number.

17 × 23 = 17 × 20 + 17 × 3 = 340 + 51 = 391


## llm-xhigh

- kwargs: `{"reasoning_effort": "xhigh"}`

### content



391

### reasoning_content

We need answer user's request: "What is 17 times 23? Reply with the number only in the final answer." Need final only number. Compute 17*23 = 391. Ensure no extra.


## vlm-cat-512

- image: `test-cat-512.jpg`
- kwargs: `{"enable_thinking": false}`

### content

This image features a close-up of an orange tabby cat with striking green eyes, looking directly at the camera.

### reasoning_content

(empty)

## vlm-cat

- image: `test-cat.jpg`
- kwargs: `{"enable_thinking": false}`

### content

This is a close-up portrait of an orange tabby cat with striking green eyes, looking directly at the camera.

### reasoning_content

(empty)
