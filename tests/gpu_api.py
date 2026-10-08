"""Independent GPU/API executor for section 11.2.

``--execute`` runs the cases against one Compose service. This module does not
refuse execution because of a stage number. Importing it does not load a model.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import socket
import struct
import subprocess
import sys
import threading
import time
import tempfile
import urllib.error
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.results import utc_now, write_result

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "tests" / "outputs" / "gpu_api.json"
MEASUREMENTS = ROOT / "tests" / "outputs" / "gpu_measurements.json"
COMPARISON = ROOT / "tests" / "outputs" / "gpu_comparison.json"
CONNECTION = ROOT / "tests" / "outputs" / "inferswap_connection.json"
PREPARE = ROOT / "prepare-inferswap"
SERVICE = "vllm-runtime"
STATUS_RESPONSE_LIMIT_SEC = 10.0
LONG_CONTEXT_TARGET_TOKENS = 30000
CONTEXT_LIMIT_TOKENS = 32768
MAX_PIXELS = 262144
MAX_OUTPUT_BUDGET = 4096
MULTIMODAL_MAX_TIMEOUT_SEC = 3600
QUALITY_FIRST_TOKENS = 2048
CAP_IMAGE_WIDTH = 768
CAP_IMAGE_HEIGHT = 768
RESIDUAL_GROWTH_MIB = 256
_MIB = 1048576
_GDN_DTYPE_RE = re.compile(
    r"(?:mamba|gdn|ssm)(?:[\s_\-]+(?:cache|state|ssm)){0,3}[\s_\-]*dtype[\s:=]+"
    r"(?:torch\.)?(?:bfloat16|float32|float16|fp32|fp16|bf16)\b"
)
_TABLE_NUMBER_RE = re.compile(
    r"(?<!\d)1(?!\d)\D+(?<!\d)2(?!\d)\D+(?<!\d)3(?!\d)\D+(?<!\d)4(?!\d)",
    re.DOTALL,
)

# 1x1 PNG. Stage 2 uses this data URL so image checks do not fetch a host path.
_PNG = base64.b64encode(
    base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )
).decode("ascii")

CASES = [
    {
        "id": "container_startup",
        "pass_condition": "Container is up, /health and /control/status succeed, state is unloaded/not_resident/active 0, and no model load was started.",
    },
    {
        "id": "load",
        "pass_condition": "Owned vLLM worker becomes ready, /v1/models lists the expected served model, and status is ready/resident.",
    },
    {
        "id": "text_nonstream",
        "pass_condition": "Non-streaming chat completion returns OpenAI-shaped JSON and the request body is unchanged.",
    },
    {
        "id": "text_stream",
        "pass_condition": "Streaming chat completion yields SSE chunks and finishes without a lifecycle error.",
    },
    {
        "id": "image_single",
        "pass_condition": "One base64 image_url is interpreted; the run is not forced down a text-only path.",
    },
    {
        "id": "image_multi",
        "pass_condition": "Multiple base64 images in one chat request are accepted and answered.",
    },
    {
        "id": "backend_record",
        "pass_condition": "Logs or metrics record Marlin, KV dtype/scale, decoder attention, vision attention, and GDN state dtype. Recording is not a pass by itself.",
    },
    {
        "id": "concurrency_1",
        "pass_condition": "Text-only, image+text, and mixed requests each run at concurrency 1 with no error or OOM.",
    },
    {
        "id": "concurrency_2",
        "pass_condition": "The same three input kinds overlap at concurrency 2 and engine logs show batching.",
    },
    {
        "id": "concurrency_4",
        "pass_condition": "The same three input kinds overlap at concurrency 4, within the external limit, with batching observed and no OOM.",
    },
    {
        "id": "long_context",
        "pass_condition": "Prompt plus image tokens plus the output budget approach 32K without treating preemption as success.",
    },
    {
        "id": "multimodal_max",
        "pass_condition": "Declared image count, max_pixels, context, output budget, and concurrency are exercised together.",
    },
    {
        "id": "disconnect_busy",
        "pass_condition": "Unload during an in-flight request returns 409 BUSY, and active_requests is 0 only after upstream completion.",
    },
    {
        "id": "status_during_lifecycle",
        "pass_condition": "GET /control/status answers during load and unload without waiting on model initialization.",
    },
    {
        "id": "repeat_lifecycle",
        "pass_condition": "At least three load/unload cycles end with every owned worker gone and no growing residual allocation.",
    },
    {
        "id": "failure_bad_model",
        "pass_condition": "A bad model path or option becomes failed with a residency that matches worker ownership.",
    },
    {
        "id": "failure_load_timeout",
        "pass_condition": "A load deadline records failed and does not report not_resident before cleanup is confirmed.",
    },
    {
        "id": "failure_worker_crash",
        "pass_condition": "A worker crash becomes failed immediately and does not clear active until ownership says every worker is gone.",
    },
    {
        "id": "container_restart",
        "pass_condition": "Stopping the controller reaps workers; unless-stopped starts a new process that is unloaded and does not auto-load.",
    },
    {
        "id": "prefix_cache",
        "pass_condition": "A repeated token prefix shows cached tokens or a prefill drop. Response success alone is not enough.",
    },
    {
        "id": "control_duplicate",
        "pass_condition": "Two concurrent load calls share one operation and return the same ready result.",
    },
    {
        "id": "control_disconnect",
        "pass_condition": "Closing the control HTTP client does not cancel unload or load; status reaches the operation result.",
    },
    {
        "id": "client_disconnect",
        "pass_condition": "After the inference client disconnects, active_requests stays above 0 until upstream completion, then returns to 0.",
    },
    {
        "id": "quality_image",
        "pass_condition": "The table answer content contains 1, 2, 3, 4 in that order. HTTP success alone is not enough.",
    },
]

REQUIRED_CASE_IDS = [case["id"] for case in CASES]
EVIDENCE_KEYS = (
    "marlin",
    "kv_dtype",
    "kv_scale",
    "decoder_attention",
    "vision_attention",
    "gdn_dtype",
)


@dataclass
class RunConfig:
    base_url: str
    root: Path = ROOT
    service: str = SERVICE
    profile: str = "qwen3.8-27b"
    served_model: str = "qwen3.8-27b"
    load_timeout_sec: float = 600
    request_timeout_sec: float = 600
    image_count: int = 4
    max_output_tokens: int = 64
    max_context_tokens: int = 32768


@dataclass
class CaseResult:
    id: str
    status: str
    detail: str


def check_plan() -> None:
    required = [case["id"] for case in CASES]
    if len(required) != len(set(required)):
        raise SystemExit("duplicate gpu case id")
    missing = [case_id for case_id in REQUIRED_CASE_IDS if case_id not in required]
    if missing:
        raise SystemExit(f"gpu plan missing cases: {missing}")
    for case in CASES:
        if not case["pass_condition"].strip():
            raise SystemExit(f"case {case['id']} has no pass condition")


def image_data_url() -> str:
    return f"data:image/png;base64,{_PNG}"


def build_chat_request(
    served_model: str,
    *,
    kind: str,
    images: int = 0,
    text: str = "Reply with the single word ok.",
    stream: bool = False,
    max_tokens: int = 16,
    min_tokens: int | None = None,
    long_context_tokens: int = 0,
    image_url: str | None = None,
) -> bytes:
    """Return the original JSON bytes that the controller must forward unchanged."""
    if kind == "text":
        images = 0
    elif kind == "image":
        images = max(images, 1)
    elif kind == "mixed":
        images = max(images, 1)
    elif kind == "multi_image":
        images = max(images, 2)
    else:
        raise ValueError(f"unknown request kind: {kind}")
    if long_context_tokens > 0:
        # Repeated text is the prompt the runtime tokenizer will count. The
        # executor does not treat this character count as the token limit.
        text = ("한국어 문맥 확인용 문장입니다. " * (long_context_tokens * 2))[: long_context_tokens * 4]
    parts: list[dict[str, Any]] = []
    for _ in range(images):
        parts.append({"type": "image_url", "image_url": {"url": image_url or image_data_url()}})
    parts.append({"type": "text", "text": text})
    content: Any = text if images == 0 else parts
    payload: dict[str, Any] = {
        "model": served_model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "stream": stream,
    }
    if min_tokens is not None:
        payload["min_tokens"] = min_tokens
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def concurrency_bodies(config: RunConfig, level: int, kind: str) -> list[bytes]:
    """Build ``level`` requests. Mixed waves stay inside the external limit."""
    if level < 1:
        raise ValueError("concurrency level must be positive")
    if kind == "mixed":
        bodies = []
        for index in range(level):
            request_kind = "text" if index % 2 == 0 else "image"
            bodies.append(
                build_chat_request(
                    config.served_model,
                    kind=request_kind,
                    images=1,
                    max_tokens=config.max_output_tokens,
                    text=f"mixed item {index}. Reply ok.",
                )
            )
        return bodies
    return [
        build_chat_request(
            config.served_model,
            kind=kind,
            images=1 if kind == "image" else 0,
            max_tokens=config.max_output_tokens,
            text=f"{kind} item {index}. Reply ok.",
        )
        for index in range(level)
    ]


def evidence_from_text(text: str) -> dict[str, str]:
    """Pull backend observations out of engine logs or metrics. Empty means unseen."""
    lowered = text.lower()
    found: dict[str, str] = {}
    if "marlin" in lowered:
        found["marlin"] = "log"
    for label, needles in (
        ("kv_dtype", ("kv_cache_dtype", "kv cache dtype", "fp8_e4m3", "cache_dtype")),
        ("kv_scale", ("kv_scale", "k_scale", "v_scale", "calculate_kv_scales")),
        ("decoder_attention", ("flashinfer", "flash_attn", "triton_attn", "attention backend", "using flashattention")),
        ("vision_attention", ("vision attention", "mm_encoder_attn", "vit attention", "for vit attention")),
        ("prefix_cache", ("prefix cache hit", "cached tokens", "prefix_cache_hits")),
        ("oom", ("out of memory", "cuda oom")),
        ("preemption", ("preemptions:",)),
    ):
        if any(needle in lowered for needle in needles):
            found[label] = "log"
    if gdn_state_dtype_phrases(lowered):
        found["gdn_dtype"] = "log"
    return found


def gdn_state_dtype_phrases(text: str) -> list[str]:
    """Return state-dtype phrases. Kernel names and cache mode do not count."""
    return [match.group(0) for match in _GDN_DTYPE_RE.finditer(text.lower())]


def table_numbers_in_content(content: str | None) -> bool:
    return isinstance(content, str) and _TABLE_NUMBER_RE.search(content) is not None


def pixels_hit_cap(source_pixels: int, processed_pixels: int, cap: int = MAX_PIXELS) -> bool:
    """True when a larger image was resized onto the configured pixel cap."""
    if source_pixels <= cap or processed_pixels <= 0 or processed_pixels > cap:
        return False
    return processed_pixels >= (cap * 95) // 100


def output_budget_fits(
    prompt_tokens: int,
    output_budget: int = MAX_OUTPUT_BUDGET,
    floor: int = LONG_CONTEXT_TARGET_TOKENS,
    limit: int = CONTEXT_LIMIT_TOKENS,
) -> bool:
    total = prompt_tokens + output_budget
    return floor <= total <= limit


def sse_event_has_token(event: str) -> bool:
    """True when an SSE event carries a non-empty content or reasoning token."""
    for line in event.splitlines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            continue
        delta = choices[0].get("delta")
        if not isinstance(delta, dict):
            continue
        for key in ("content", "reasoning", "reasoning_content"):
            value = delta.get(key)
            if isinstance(value, str) and value != "":
                return True
    return False


def quality_should_retry(content: str | None, finish_reason: str | None) -> bool:
    blank = not isinstance(content, str) or not content.strip()
    return blank and finish_reason == "length"


def judge_case(case_id: str, *, http_ok: bool, evidence: dict[str, str], detail: str = "") -> tuple[str, str]:
    """Return status and detail. Response success alone cannot pass evidence cases."""
    if evidence.get("oom"):
        return "failed", detail or "OOM observed"
    if evidence.get("preemption") and case_id in {"long_context", "multimodal_max", "concurrency_4"}:
        return "failed", detail or "preemption or recompute is not a pass"
    if case_id == "backend_record":
        missing = [key for key in EVIDENCE_KEYS if not evidence.get(key)]
        if missing:
            return "failed", f"response success is not backend evidence; missing {', '.join(missing)}. {detail}".strip()
        return "passed", detail or "backend observations recorded"
    if case_id in {"concurrency_2", "concurrency_4"}:
        if not http_ok:
            return "failed", detail or "concurrent requests failed"
        if not evidence.get("batching"):
            return "failed", detail or "responses arrived but engine logs did not show batching"
        return "passed", detail or "overlapping requests and batching observed"
    if case_id == "long_context":
        if not http_ok:
            return "failed", detail or "long context request failed"
        if not evidence.get("context_tokens"):
            return "failed", detail or "usage tokens were not observed, so context length is unconfirmed"
        return "passed", detail or "combined context approached the profile limit"
    if case_id == "repeat_lifecycle":
        if not evidence.get("workers_gone"):
            return "failed", detail or "worker exit was not confirmed"
        if not evidence.get("residual_stable"):
            return "failed", detail or "unload residual was not measured or it grew"
        return "passed", detail or "workers exited and residual did not grow"
    if case_id == "prefix_cache":
        if not http_ok:
            return "failed", detail or "prefix cache requests failed"
        if not evidence.get("prefix_cache"):
            return "failed", detail or "repeated prefix did not show cached tokens or a prefill drop"
        return "passed", detail or "prefix reuse observed"
    if case_id == "quality_image":
        if not http_ok:
            return "failed", detail or "quality request failed"
        if not evidence.get("table_answer"):
            return "failed", detail or "content did not contain 1, 2, 3, 4 in order"
        return "passed", detail or "table numbers were in content"
    if case_id == "multimodal_max":
        if not http_ok:
            return "failed", detail or "max requests failed"
        missing = [
            key
            for key in ("pixels", "context_tokens", "output_budget", "concurrency", "preemption_clear")
            if not evidence.get(key)
        ]
        if missing:
            return "failed", f"max condition incomplete: {', '.join(missing)}. {detail}".strip()
        return "passed", detail or "declared max condition exercised"
    if not http_ok:
        return "failed", detail or "request failed"
    return "passed", detail or "observed"


def context_approached(usage: dict[str, Any] | None, output_budget: int, target: int = LONG_CONTEXT_TARGET_TOKENS) -> bool:
    if not isinstance(usage, dict):
        return False
    try:
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
    except (TypeError, ValueError):
        return False
    return prompt + max(completion, output_budget) >= target


def http_exchange(
    method: str,
    url: str,
    body: bytes | None = None,
    timeout: float = 30,
    headers: dict[str, str] | None = None,
) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=body, method=method)
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    if body is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return 0, json.dumps({"error": str(exc)}).encode("utf-8")


def _compose(root: Path, args: list[str], files: list[Path] | None = None) -> subprocess.CompletedProcess[str]:
    command = ["docker", "compose"]
    if files:
        command.extend(["-f", str(root / "docker-compose.yml")])
        for path in files:
            command.extend(["-f", str(path)])
    command.extend(args)
    return subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)


def _owned_container(root: Path, service: str) -> str:
    listed = _compose(root, ["ps", "-q", service])
    container_id = listed.stdout.strip().splitlines()[0] if listed.stdout.strip() else ""
    if listed.returncode != 0 or not container_id:
        raise RuntimeError(listed.stderr.strip() or f"compose service {service} is not running")
    inspected = subprocess.run(
        ["docker", "inspect", container_id],
        capture_output=True,
        text=True,
        check=False,
    )
    if inspected.returncode != 0:
        raise RuntimeError(inspected.stderr.strip() or "docker inspect failed")
    payload = json.loads(inspected.stdout)[0]
    labels = payload.get("Config", {}).get("Labels") or {}
    if labels.get("com.docker.compose.service") != service:
        raise RuntimeError("refusing to touch a container that is not the selected service")
    working = Path(str(labels.get("com.docker.compose.project.working_dir", ""))).resolve()
    if working != root.resolve():
        raise RuntimeError("refusing to touch a container from another compose project")
    return container_id


def _logs(container_id: str, since: str) -> str:
    result = subprocess.run(
        ["docker", "logs", "--since", since, container_id],
        capture_output=True,
        text=True,
        check=False,
    )
    return (result.stdout or "") + (result.stderr or "")


def _gpu_used_mib() -> int | None:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    try:
        return int(lines[0])
    except (IndexError, ValueError):
        return None


def _worker_cmds(container_id: str) -> list[str]:
    result = subprocess.run(
        ["docker", "exec", container_id, "ps", "-eo", "pid,cmd"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if "vllm" in line.lower()]


def _status(config: RunConfig, timeout: float = 10) -> tuple[int, dict[str, Any]]:
    code, raw = http_exchange("GET", f"{config.base_url}/control/status", timeout=timeout)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        payload = {}
    return code, payload if isinstance(payload, dict) else {}


def _health(config: RunConfig) -> tuple[int, dict[str, Any]]:
    code, raw = http_exchange("GET", f"{config.base_url}/health", timeout=10)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        payload = {}
    return code, payload if isinstance(payload, dict) else {}


def _post_control(config: RunConfig, action: str, timeout: float) -> tuple[int, dict[str, Any]]:
    code, raw = http_exchange("POST", f"{config.base_url}/control/{action}", body=b"{}", timeout=timeout)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        payload = {"raw": raw.decode("utf-8", errors="replace")}
    return code, payload if isinstance(payload, dict) else {}


def _chat(config: RunConfig, body: bytes) -> tuple[int, dict[str, Any] | None, str]:
    code, raw = http_exchange(
        "POST",
        f"{config.base_url}/v1/chat/completions",
        body=body,
        timeout=config.request_timeout_sec,
    )
    text = raw.decode("utf-8", errors="replace")
    if "text/event-stream" in text or text.startswith("data:"):
        return code, None, text
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    return code, payload if isinstance(payload, dict) else None, text


def _openai_ok(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    choices = payload.get("choices")
    return isinstance(choices, list) and bool(choices)


def _repro(config: RunConfig) -> str:
    return (
        f"python3 tests/gpu_api.py --execute --base-url {config.base_url} "
        f"--profile {config.profile} --service {config.service}"
    )


def _case(case_id: str, status: str, detail: str, config: RunConfig) -> CaseResult:
    text = detail.strip()
    repro = _repro(config)
    if repro not in text:
        text = f"{text} Repro: {repro}".strip()
    return CaseResult(case_id, status, text)


def _run_parallel(config: RunConfig, bodies: list[bytes]) -> tuple[bool, str]:
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, len(bodies))) as pool:
        futures = [pool.submit(_chat, config, body) for body in bodies]
        for future in as_completed(futures):
            code, payload, text = future.result()
            if code != 200 or (payload is not None and not _openai_ok(payload)):
                errors.append(text[:500])
            if "out of memory" in text.lower():
                errors.append("OOM")
    return not errors, "; ".join(errors)


def evidence_from_metrics(text: str) -> dict[str, str]:
    """Treat overlapping engine metrics as batching evidence. A zero gauge is not enough."""
    found: dict[str, str] = {}
    running = _metric_samples(text, "vllm:num_requests_running")
    if any(value >= 2 for value in running):
        found["batching"] = "metrics"
    preemptions = _metric_samples(text, "vllm:num_preemptions")
    if any(value > 0 for value in preemptions):
        found["preemption"] = "metrics"
    return found


def cached_token_count(usage: dict[str, Any] | None) -> int:
    if not isinstance(usage, dict):
        return 0
    details = usage.get("prompt_tokens_details")
    if not isinstance(details, dict):
        return 0
    try:
        return int(details.get("cached_tokens") or 0)
    except (TypeError, ValueError):
        return 0


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def table_png_bytes() -> bytes:
    """Small black-on-white table. Glyphs are a few pixels tall so the text is small."""
    glyph = {
        "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
        "B": ["11110", "10001", "11110", "10001", "10001", "10001", "11110"],
        "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
        "2": ["01110", "10001", "00001", "00110", "01000", "10000", "11111"],
        "3": ["11111", "00001", "00010", "00110", "00001", "10001", "01110"],
        "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
        "|": ["00100", "00100", "00100", "00100", "00100", "00100", "00100"],
        " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    }
    rows = ["| A | B |", "| 1 | 2 |", "| 3 | 4 |"]
    scale = 1
    cell_w = 6 * scale
    cell_h = 8 * scale
    width = max(len(row) for row in rows) * cell_w + 2
    height = len(rows) * cell_h + 2
    pixels = [[255 for _ in range(width)] for _ in range(height)]
    for y in range(height):
        pixels[y][0] = pixels[y][width - 1] = 0
    for x in range(width):
        pixels[0][x] = pixels[height - 1][x] = 0
    for row_index, row in enumerate(rows):
        for col_index, char in enumerate(row):
            bits = glyph.get(char, glyph[" "])
            origin_x = 1 + col_index * cell_w
            origin_y = 1 + row_index * cell_h
            for gy, line in enumerate(bits):
                for gx, bit in enumerate(line):
                    if bit == "1":
                        pixels[origin_y + gy][origin_x + gx] = 0
            if row_index == 0:
                for x in range(origin_x, min(width - 1, origin_x + cell_w)):
                    pixels[origin_y + 7][x] = 0
    raw = b"".join(b"\x00" + bytes(row) for row in pixels)
    return b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)),
            _png_chunk(b"IDAT", zlib.compress(raw, 9)),
            _png_chunk(b"IEND", b""),
        ]
    )


def table_image_data_url() -> str:
    return "data:image/png;base64," + base64.b64encode(table_png_bytes()).decode("ascii")


class GpuSampler:
    def __init__(self) -> None:
        self.samples: list[tuple[float, int]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        def loop() -> None:
            while not self._stop.is_set():
                used = _gpu_used_mib()
                if used is not None:
                    self.samples.append((time.time(), used))
                self._stop.wait(0.5)

        self._thread = threading.Thread(target=loop, name="gpu-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def peak_between(self, start: float, end: float) -> int | None:
        values = [used for stamp, used in self.samples if start <= stamp <= end]
        return max(values) if values else None


def _docker_python(
    container_id: str,
    script: str,
    args: list[str],
    timeout: float = 180,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    command = ["docker", "exec"]
    for key, value in (env or {}).items():
        command.extend(["-e", f"{key}={value}"])
    command.extend(["-i", container_id, "python3", "-", *args])
    return subprocess.run(
        command,
        input=script,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _metrics_text(container_id: str) -> str:
    script = (
        "import urllib.request\n"
        "print(urllib.request.urlopen('http://127.0.0.1:8080/metrics', timeout=2).read().decode('utf-8', 'replace'))\n"
    )
    result = _docker_python(container_id, script, [], timeout=8)
    return result.stdout or ""


def _relevant_lines(text: str) -> list[str]:
    keys = (
        "marlin",
        "kv_cache",
        "cache_dtype",
        "vit attention",
        "gdn",
        "mamba",
        "attention",
        "prefix",
        "fp8",
        "speculative",
        "mtp",
        "out of memory",
        "error",
        "compressedtensors",
    )
    kept: list[str] = []
    for line in text.splitlines():
        lowered = line.lower()
        if any(key in lowered for key in keys):
            kept.append(line[:400])
    return kept[-200:]


def _usage_of(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    usage = payload.get("usage")
    return usage if isinstance(usage, dict) else None


def _choice(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return {}
    return choices[0]


def _message_fields(payload: dict[str, Any] | None) -> tuple[str, str]:
    message = _choice(payload).get("message")
    if not isinstance(message, dict):
        return "", ""
    content = message.get("content")
    reasoning = message.get("reasoning")
    if not isinstance(reasoning, str):
        reasoning = message.get("reasoning_content")
    return (content if isinstance(content, str) else "", reasoning if isinstance(reasoning, str) else "")


def _finish_reason(payload: dict[str, Any] | None) -> str | None:
    reason = _choice(payload).get("finish_reason")
    return reason if isinstance(reason, str) else None


def _answer_text(payload: dict[str, Any] | None) -> str:
    content, reasoning = _message_fields(payload)
    parts = [part for part in (content, reasoning) if part]
    return "\n".join(parts)


def _empty_observation(code: int, text: str, latency: float) -> dict[str, Any]:
    return {
        "code": code,
        "payload": None,
        "text": text,
        "ttft_sec": None,
        "latency_sec": latency,
        "usage": None,
        "output_tok_s": None,
        "answer": "",
        "content": "",
        "reasoning": "",
        "finish_reason": None,
    }


def _chat_observed(config: RunConfig, body: bytes) -> dict[str, Any]:
    """Latency plus SSE time-to-first-token. Non-streaming TTFT stays unset."""
    request = urllib.request.Request(
        f"{config.base_url}/v1/chat/completions",
        data=body,
        method="POST",
    )
    request.add_header("Content-Type", "application/json")
    streaming = b'"stream":true' in body
    started = time.monotonic()
    ttft: float | None = None
    chunks: list[bytes] = []
    code = 0
    try:
        with urllib.request.urlopen(request, timeout=config.request_timeout_sec) as response:
            code = response.status
            consumed = 0
            while True:
                part = response.read(4096)
                if not part:
                    break
                chunks.append(part)
                if not streaming or ttft is not None:
                    continue
                pending = b"".join(chunks)
                while True:
                    split = pending.find(b"\n\n", consumed)
                    if split < 0:
                        break
                    event = pending[consumed:split].decode("utf-8", errors="replace")
                    consumed = split + 2
                    if sse_event_has_token(event):
                        ttft = time.monotonic() - started
                        break
    except urllib.error.HTTPError as exc:
        code = exc.code
        chunks.append(exc.read())
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        latency = time.monotonic() - started
        return _empty_observation(0, str(exc), latency)
    latency = time.monotonic() - started
    raw = b"".join(chunks)
    text = raw.decode("utf-8", errors="replace")
    payload: dict[str, Any] | None = None
    if not text.startswith("data:"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            payload = parsed
    usage = _usage_of(payload)
    completion = 0
    if isinstance(usage, dict):
        try:
            completion = int(usage.get("completion_tokens") or 0)
        except (TypeError, ValueError):
            completion = 0
    content, reasoning = _message_fields(payload)
    return {
        "code": code,
        "payload": payload,
        "text": text,
        "ttft_sec": ttft if streaming else None,
        "latency_sec": latency,
        "usage": usage,
        "output_tok_s": (completion / latency) if latency > 0 and completion else None,
        "answer": _answer_text(payload),
        "content": content,
        "reasoning": reasoning,
        "finish_reason": _finish_reason(payload),
    }


def _close_after_send(config: RunConfig, method: str, path: str, body: bytes) -> None:
    """Send one HTTP request and close before reading the response."""
    from urllib.parse import urlparse

    parsed = urlparse(config.base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 80
    sock = socket.create_connection((host, port), timeout=10)
    payload = (
        f"{method} {path} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Connection: close\r\n\r\n"
    ).encode("ascii") + body
    try:
        sock.sendall(payload)
        time.sleep(0.3)
    finally:
        sock.close()


def _poll_status_samples(config: RunConfig, stop: threading.Event, samples: list[float]) -> None:
    while not stop.is_set():
        started = time.monotonic()
        code, _payload = _status(config, timeout=STATUS_RESPONSE_LIMIT_SEC)
        elapsed = time.monotonic() - started
        samples.append(elapsed if code == 200 else STATUS_RESPONSE_LIMIT_SEC + 1)
        stop.wait(5)


def _long_user_text(container_id: str, max_tokens: int) -> tuple[str, str]:
    script = r"""
import json, sys
from transformers import AutoTokenizer
max_tokens = int(sys.argv[1])
floor = int(sys.argv[2])
limit = int(sys.argv[3])
tok = AutoTokenizer.from_pretrained("/models/qwen3.8-27b", trust_remote_code=True)
unit = "한국어 문맥 확인용 문장입니다. "

def token_count(text: str) -> int:
    ids = tok.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True,
        tokenize=True,
    )
    if hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    shape = getattr(ids, "shape", None)
    if shape is not None:
        return int(shape[-1])
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        return len(ids[0])
    return len(ids)

lo, hi = 1, 8000
best_text = unit
best_n = token_count(unit)
while lo <= hi:
    mid = (lo + hi) // 2
    text = unit * mid
    count = token_count(text)
    total = count + max_tokens
    if floor <= total <= limit:
        best_text = text
        best_n = count
        break
    if total < floor:
        best_text = text
        best_n = count
        lo = mid + 1
    else:
        hi = mid - 1
print(json.dumps({"prompt_tokens_estimate": best_n, "text": best_text}))
"""
    result = _docker_python(container_id, script, [str(max_tokens), "30000", "31000"], timeout=180)
    if result.returncode != 0:
        return "", (result.stderr or result.stdout or "tokenizer failed")[-500:]
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return "", result.stdout[-500:]
    text = payload.get("text")
    if not isinstance(text, str) or not text:
        return "", "tokenizer returned no text"
    return text, f"estimate={payload.get('prompt_tokens_estimate')}"


def _metric_samples(text: str, name: str) -> list[float]:
    values: list[float] = []
    names = (name, f"{name}_total")
    for line in text.splitlines():
        if not any(line.startswith(f"{item}{{") or line.startswith(f"{item} ") for item in names):
            continue
        try:
            values.append(float(line.split()[-1]))
        except ValueError:
            continue
    return values


def _metric_value(text: str, name: str) -> float | None:
    values = _metric_samples(text, name)
    return values[-1] if values else None


def _run_parallel_observed(
    config: RunConfig,
    bodies: list[bytes],
    container_id: str,
) -> tuple[bool, str, dict[str, str], float]:
    script = (
        "import time, urllib.request\n"
        "deadline = time.time() + 120\n"
        "while time.time() < deadline:\n"
        "    try:\n"
        "        data = urllib.request.urlopen('http://127.0.0.1:8080/metrics', timeout=1).read().decode()\n"
        "    except Exception as exc:\n"
        "        data = f'ERR {exc}'\n"
        "    print(data, flush=True)\n"
        "    print('---SAMPLE---', flush=True)\n"
        "    time.sleep(0.05)\n"
    )
    proc = subprocess.Popen(
        ["docker", "exec", "-i", container_id, "python3", "-"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdin is not None and proc.stdout is not None
    chunks: list[str] = []
    first_sample = threading.Event()

    def _drain() -> None:
        assert proc.stdout is not None
        collected: list[str] = []
        for line in proc.stdout:
            collected.append(line)
            if "---SAMPLE---" in line:
                first_sample.set()
        chunks.append("".join(collected))

    reader = threading.Thread(target=_drain, name="metrics-drain", daemon=True)
    reader.start()
    proc.stdin.write(script)
    proc.stdin.close()
    first_sample.wait(timeout=20)
    started = time.monotonic()
    ok, note = _run_parallel(config, bodies)
    elapsed = time.monotonic() - started
    time.sleep(0.3)
    proc.kill()
    reader.join(timeout=5)
    blob = chunks[0] if chunks else ""
    evidence = evidence_from_text(blob)
    evidence.update(evidence_from_metrics(blob))
    return ok, note, evidence, elapsed


def _note_kv_scale(evidence: dict[str, str], logs: str) -> None:
    """Checkpoint has no KV scheme. A default scale of 1.0 is not a calibrated KV cache."""
    lowered = logs.lower()
    if "fp8" not in lowered:
        return
    if any(token in lowered for token in ("k_scale", "v_scale", "kv_scale", "calculate_kv_scales")):
        evidence["kv_scale"] = "uncalibrated_default_scale" if "1.0" in lowered else "log"
        return
    evidence["kv_scale"] = "uncalibrated_no_checkpoint_scheme"


def _save_measurements(measurements: dict[str, Any]) -> None:
    MEASUREMENTS.parent.mkdir(parents=True, exist_ok=True)
    MEASUREMENTS.write_text(json.dumps(measurements, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def execute_suite(config: RunConfig) -> list[CaseResult]:
    """Run section 11.2 against the selected Compose service only."""
    results: list[CaseResult] = []
    started = _compose(config.root, ["up", "-d", "--no-build", "--no-recreate", config.service])
    if started.returncode != 0:
        reason = started.stderr.strip() or "docker compose up failed"
        return [_case(case["id"], "not_run", f"precondition failed: {reason}", config) for case in CASES]
    try:
        container_id = _owned_container(config.root, config.service)
    except Exception as exc:
        for case in CASES:
            results.append(_case(case["id"], "not_run", f"precondition failed: {exc}", config))
        return results

    code, health = _health(config)
    status_code, status = _status(config)
    workers = _worker_cmds(container_id)
    startup_ok = (
        code == 200
        and health.get("controller") == "ok"
        and status_code == 200
        and status.get("state") == "unloaded"
        and status.get("residency") == "not_resident"
        and status.get("active_requests") == 0
        and not any("serve" in line for line in workers)
    )
    results.append(
        _case(
            "container_startup",
            "passed" if startup_ok else "failed",
            f"health={health} status={status} workers={workers}",
            config,
        )
    )

    since = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(time.time() - 2))
    measurements: dict[str, Any] = {
        "baseline_mib": _gpu_used_mib(),
        "requests": [],
        "status_samples_during_load_sec": [],
        "status_samples_during_unload_sec": [],
    }
    sampler = GpuSampler()
    sampler.start()
    load_window = time.time()
    load_status_samples: list[float] = []
    load_stop = threading.Event()
    load_poller = threading.Thread(
        target=_poll_status_samples,
        args=(config, load_stop, load_status_samples),
        name="load-status",
        daemon=True,
    )
    load_poller.start()
    with ThreadPoolExecutor(max_workers=2) as pool:
        load_futures = [pool.submit(_post_control, config, "load", config.load_timeout_sec) for _ in range(2)]
        load_pair = [future.result() for future in load_futures]
    load_stop.set()
    load_poller.join(timeout=2)
    load_finished = time.time()
    measurements["cold_load_sec"] = load_finished - load_window
    measurements["cold_load_peak_mib"] = sampler.peak_between(load_window, load_finished)
    measurements["status_samples_during_load_sec"] = load_status_samples
    load_code, load_body = load_pair[0]
    status_code, status = _status(config)
    models_code, models_raw = http_exchange("GET", f"{config.base_url}/v1/models", timeout=30)
    load_ok = (
        load_code == 200
        and status.get("state") == "ready"
        and status.get("residency") == "resident"
        and config.served_model in models_raw.decode("utf-8", errors="replace")
    )
    duplicate_load_ok = all(code == 200 and body.get("state") == "ready" for code, body in load_pair)
    results.append(
        _case(
            "load",
            "passed" if load_ok else "failed",
            f"load={load_code} {load_body} status={status} models={models_code} duplicate={load_pair[1][0]}",
            config,
        )
    )
    logs = _logs(container_id, since)
    evidence = evidence_from_text(logs)
    _note_kv_scale(evidence, logs)
    measurements["backend_lines"] = _relevant_lines(logs)
    measurements["backend_evidence"] = evidence
    measurements["mtp_mentioned"] = "speculative" in logs.lower()
    version = _docker_python(
        container_id,
        "import torch, transformers, vllm\nprint(vllm.__version__)\nprint(torch.__version__)\nprint(torch.version.cuda)\nprint(transformers.__version__)\n",
        [],
        timeout=60,
    )
    measurements["runtime_versions"] = (version.stdout or version.stderr)[-500:]
    _save_measurements(measurements)

    def chat_case(case_id: str, body: bytes, *, require_image: bool = False) -> dict[str, Any] | None:
        if not load_ok:
            results.append(_case(case_id, "failed", f"load did not become ready: {load_body}", config))
            return None
        if require_image and b"image_url" not in body:
            results.append(_case(case_id, "failed", "request dropped image_url before send", config))
            return None
        observed = _chat_observed(config, body)
        measurements["requests"].append(
            {
                "id": case_id,
                "code": observed["code"],
                "ttft_sec": observed["ttft_sec"],
                "latency_sec": observed["latency_sec"],
                "output_tok_s": observed["output_tok_s"],
                "usage": observed["usage"],
                "answer": (observed["answer"] or "")[:500],
            }
        )
        ok = observed["code"] == 200 and (_openai_ok(observed["payload"]) or "data:" in observed["text"])
        local = dict(evidence)
        local.update(evidence_from_text(observed["text"] + logs))
        status_name, detail = judge_case(case_id, http_ok=ok, evidence=local, detail=observed["text"][:400])
        results.append(_case(case_id, status_name, detail, config))
        return observed

    infer_window = time.time()
    chat_case("text_nonstream", build_chat_request(config.served_model, kind="text", stream=False, max_tokens=64, text="한국어로 한 단어만 답하세요: 확인"))
    chat_case("text_stream", build_chat_request(config.served_model, kind="text", stream=True, max_tokens=64))
    chat_case(
        "image_single",
        build_chat_request(config.served_model, kind="image", images=1, text="Describe the image.", max_tokens=64),
        require_image=True,
    )
    chat_case(
        "image_multi",
        build_chat_request(config.served_model, kind="multi_image", images=2, text="Describe the images.", max_tokens=64),
        require_image=True,
    )
    if load_ok:
        results.append(_run_quality(config, measurements)["case"])
    else:
        results.append(_blocked("quality_image", load_body, config))

    backend_status, backend_detail = judge_case("backend_record", http_ok=load_ok, evidence=evidence, detail=logs[-1000:])
    results.append(_case("backend_record", backend_status, backend_detail, config))

    measurements["concurrency"] = []
    for level, case_id in ((1, "concurrency_1"), (2, "concurrency_2"), (4, "concurrency_4")):
        if not load_ok:
            results.append(_case(case_id, "failed", f"load did not become ready: {load_body}", config))
            continue
        wave_ok = True
        notes: list[str] = []
        wave_evidence: dict[str, str] = {}
        for kind in ("text", "image", "mixed"):
            bodies = concurrency_bodies(config, level, kind)
            if len(bodies) != level:
                wave_ok = False
                notes.append(f"{kind} generated {len(bodies)} bodies")
                continue
            ok, note, kind_evidence, elapsed = _run_parallel_observed(config, bodies, container_id)
            wave_ok = wave_ok and ok
            wave_evidence.update(kind_evidence)
            notes.append(f"{kind}:{note or 'ok'} {elapsed:.1f}s")
            measurements["concurrency"].append({"level": level, "kind": kind, "ok": ok, "elapsed_sec": elapsed})
        wave_evidence.update(evidence_from_text(_logs(container_id, since)))
        judged, detail = judge_case(case_id, http_ok=wave_ok, evidence=wave_evidence, detail="; ".join(notes))
        results.append(_case(case_id, judged, detail, config))
    _save_measurements(measurements)

    _run_prefix_cache(config, results, measurements, load_ok, load_body)
    _run_long_context(config, container_id, results, measurements, load_ok, load_body)
    _save_measurements(measurements)
    results.extend(_client_and_busy(config, load_ok, load_body))
    unload_duplicate_ok, unload_samples, warm_loads = _repeat_lifecycle(config, container_id, measurements, sampler)
    repeat_status, repeat_detail = measurements.pop("_repeat_case")
    results.append(_case("repeat_lifecycle", repeat_status, repeat_detail, config))
    disconnect_case = measurements.pop("control_disconnect_case")
    results.append(_case("control_disconnect", disconnect_case["status"], disconnect_case["detail"], config))
    results.append(
        _case(
            "control_duplicate",
            "passed" if duplicate_load_ok and unload_duplicate_ok else "failed",
            f"load_duplicate={duplicate_load_ok} unload_duplicate={unload_duplicate_ok}",
            config,
        )
    )
    combined_status = list(load_status_samples) + list(unload_samples)
    status_fast = bool(load_status_samples) and bool(unload_samples) and max(combined_status) <= STATUS_RESPONSE_LIMIT_SEC
    results.append(
        _case(
            "status_during_lifecycle",
            "passed" if status_fast else "failed",
            f"load_samples={load_status_samples[:12]} unload_samples={unload_samples[:12]} limit={STATUS_RESPONSE_LIMIT_SEC}",
            config,
        )
    )
    measurements["warm_load_sec"] = warm_loads
    measurements["status_samples_during_unload_sec"] = unload_samples
    _run_multimodal_max(config, container_id, results, measurements)
    measurements["inference_peak_mib"] = sampler.peak_between(infer_window, time.time())
    try:
        results.extend(_failure_cases(config))
    finally:
        sampler.stop()
        _save_measurements(measurements)
    return results


def _blocked(case_id: str, reason: object, config: RunConfig) -> CaseResult:
    return _case(case_id, "failed", f"load did not become ready: {reason}", config)


def _run_prefix_cache(
    config: RunConfig,
    results: list[CaseResult],
    measurements: dict[str, Any],
    load_ok: bool,
    load_body: object,
) -> None:
    if not load_ok:
        results.append(_blocked("prefix_cache", load_body, config))
        return
    # Attention blocks for this hybrid model are 784 tokens, so a shorter prefix cannot hit.
    text = "한국어 문맥 확인용 문장입니다. " * 250
    body = build_chat_request(config.served_model, kind="text", text=text, max_tokens=16)
    container_id = _owned_container(config.root, config.service)
    before_text = _metrics_text(container_id)
    before = _metric_value(before_text, "vllm:prefix_cache_hits")
    before_queries = _metric_value(before_text, "vllm:prefix_cache_queries")
    first = _chat_observed(config, body)
    second = _chat_observed(config, body)
    time.sleep(1)
    after_text = _metrics_text(container_id)
    after = _metric_value(after_text, "vllm:prefix_cache_hits")
    after_queries = _metric_value(after_text, "vllm:prefix_cache_queries")
    first_cached = cached_token_count(first["usage"])
    second_cached = cached_token_count(second["usage"])
    first_prompt = int((first["usage"] or {}).get("prompt_tokens") or 0) if isinstance(first["usage"], dict) else 0
    second_prompt = int((second["usage"] or {}).get("prompt_tokens") or 0) if isinstance(second["usage"], dict) else 0
    hit_delta = None if before is None or after is None else after - before
    query_delta = None if before_queries is None or after_queries is None else after_queries - before_queries
    reused = second_cached > 0 or (hit_delta is not None and hit_delta > 0) or (first_prompt > 0 and second_prompt < first_prompt)
    evidence = {"prefix_cache": "metrics" if reused else ""}
    ok = first["code"] == 200 and second["code"] == 200 and _openai_ok(second["payload"])
    status_name, detail = judge_case(
        "prefix_cache",
        http_ok=ok,
        evidence=evidence,
        detail=(
            f"first_cached={first_cached} second_cached={second_cached} "
            f"prompt={first_prompt}->{second_prompt} hit_delta={hit_delta} query_delta={query_delta}"
        ),
    )
    measurements["prefix_cache"] = {
        "first_cached": first_cached,
        "second_cached": second_cached,
        "first_prompt": first_prompt,
        "second_prompt": second_prompt,
        "hit_delta": hit_delta,
    }
    results.append(_case("prefix_cache", status_name, detail, config))


def _run_long_context(
    config: RunConfig,
    container_id: str,
    results: list[CaseResult],
    measurements: dict[str, Any],
    load_ok: bool,
    load_body: object,
) -> None:
    if not load_ok:
        results.append(_blocked("long_context", load_body, config))
        return
    output_budget = 256
    text, note = _long_user_text(container_id, output_budget)
    if not text:
        results.append(_case("long_context", "failed", f"tokenizer prompt was not built: {note}", config))
        return
    estimate = 0
    if "estimate=" in note:
        raw_estimate = note.split("estimate=", 1)[1].split()[0]
        if raw_estimate.isdigit():
            estimate = int(raw_estimate)
    if estimate < 1000:
        results.append(_case("long_context", "failed", f"tokenizer prompt was not near 32K: {note}", config))
        return
    body = build_chat_request(
        config.served_model,
        kind="image",
        images=1,
        text=text,
        max_tokens=output_budget,
    )
    observed = _chat_observed(config, body)
    usage = observed["usage"] if isinstance(observed["usage"], dict) else None
    evidence = evidence_from_text(observed["text"])
    if context_approached(usage, output_budget):
        evidence["context_tokens"] = "usage"
    status_name, detail = judge_case(
        "long_context",
        http_ok=observed["code"] == 200 and _openai_ok(observed["payload"]),
        evidence=evidence,
        detail=f"{note} usage={usage} body={observed['text'][:300]}",
    )
    measurements["long_context"] = {"usage": usage, "note": note, "code": observed["code"]}
    results.append(_case("long_context", status_name, detail, config))


def cap_png_bytes(width: int = CAP_IMAGE_WIDTH, height: int = CAP_IMAGE_HEIGHT) -> bytes:
    """Grid PNG larger than max_pixels. A regular grid compresses to a few kilobytes."""
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        row_on = (y % 32) < 2
        for x in range(width):
            raw.append(0 if row_on or (x % 32) < 2 else 255)
    return b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)),
            _png_chunk(b"IDAT", zlib.compress(bytes(raw), 9)),
            _png_chunk(b"IEND", b""),
        ]
    )


def cap_image_data_url() -> str:
    return "data:image/png;base64," + base64.b64encode(cap_png_bytes()).decode("ascii")


def _replace_request(measurements: dict[str, Any], record: dict[str, Any]) -> None:
    requests = measurements.setdefault("requests", [])
    for index, item in enumerate(requests):
        if isinstance(item, dict) and item.get("id") == record.get("id"):
            requests[index] = record
            return
    requests.append(record)


def _usage_int(usage: dict[str, Any] | None, key: str) -> int:
    if not isinstance(usage, dict):
        return 0
    try:
        return int(usage.get(key) or 0)
    except (TypeError, ValueError):
        return 0


_MAX_PROMPT_SCRIPT = r"""
import base64, io, json, sys
from PIL import Image
from transformers import AutoTokenizer
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

png_b64 = sys.argv[1]
max_pixels = int(sys.argv[2])
floor = int(sys.argv[3])
limit = int(sys.argv[4])
image_count = int(sys.argv[5])
target = int(sys.argv[6])
forced = int(sys.argv[7])
image = Image.open(io.BytesIO(base64.b64decode(png_b64)))
width, height = image.size
cfg = json.load(open("/models/qwen3.8-27b/config.json"))
vision = cfg["vision_config"]
pre = json.load(open("/models/qwen3.8-27b/preprocessor_config.json"))
factor = vision["patch_size"] * vision["spatial_merge_size"]
resized_h, resized_w = smart_resize(
    height=height,
    width=width,
    factor=factor,
    min_pixels=pre["size"]["shortest_edge"],
    max_pixels=max_pixels,
)
grid_h = resized_h // vision["patch_size"]
grid_w = resized_w // vision["patch_size"]
vision_tokens = (grid_h * grid_w) // (vision["spatial_merge_size"] ** 2)
tok = AutoTokenizer.from_pretrained("/models/qwen3.8-27b", trust_remote_code=True)
pad_id = tok.convert_tokens_to_ids("<|image_pad|>")
unit = "한국어 문맥 확인용 문장입니다. "
instruction = "\n출력 한도에 도달할 때까지 번호를 이어서 쓰세요."

def prompt_tokens(repeats: int) -> tuple[int, str]:
    text = (unit * repeats) + instruction
    content = [{"type": "image"} for _ in range(image_count)]
    content.append({"type": "text", "text": text})
    encoded = tok.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
        tokenize=True,
    )
    ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    pads = sum(1 for token in ids if token == pad_id)
    if pads != image_count:
        raise SystemExit(f"image pad count {pads} != {image_count}")
    return len(ids) + pads * (vision_tokens - 1), text

if forced > 0:
    count, text = prompt_tokens(forced)
    chosen = forced
else:
    lo, hi = 1, 12000
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        count, text = prompt_tokens(mid)
        if floor <= count <= limit and (best is None or abs(count - target) < abs(best[0] - target)):
            best = (count, text, mid)
        if count < target:
            lo = mid + 1
        elif count > target:
            hi = mid - 1
        else:
            break
    if best is None:
        raise SystemExit(f"no prompt landed in [{floor}, {limit}]")
    count, text, chosen = best
other, _other_text = prompt_tokens(chosen + 1)
print(json.dumps({
    "source_pixels": width * height,
    "processed_width": resized_w,
    "processed_height": resized_h,
    "processed_pixels": resized_w * resized_h,
    "vision_tokens": vision_tokens,
    "prompt_tokens_estimate": count,
    "repeats": chosen,
    "unit_tokens": other - count,
    "text": text,
}))
"""


_GDN_SCRIPT = r"""
import json, os
from pathlib import Path
import torch
from transformers import AutoConfig
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateDtypeCalculator

cmd = ""
for pid in os.listdir("/proc"):
    if not pid.isdigit():
        continue
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", "replace")
    except OSError:
        continue
    if "vllm" in raw and "serve" in raw:
        cmd = raw
        break
if not cmd:
    raise SystemExit("vllm serve process was not found")

def flag(name: str, default: str) -> str:
    parts = cmd.split()
    if name not in parts:
        return default
    index = parts.index(name)
    if index + 1 >= len(parts):
        return default
    return parts[index + 1]

model_dtype_name = flag("--dtype", "bfloat16")
cache_dtype = flag("--mamba-cache-dtype", "auto")
ssm_dtype = flag("--mamba-ssm-cache-dtype", "auto")
cfg = AutoConfig.from_pretrained("/models/qwen3.8-27b", trust_remote_code=True)
text_cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
hf_ssm = getattr(text_cfg, "mamba_ssm_dtype", None)
if ssm_dtype == "auto" and isinstance(hf_ssm, str) and hf_ssm:
    ssm_dtype = hf_ssm
torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[model_dtype_name]
conv, recurrent = MambaStateDtypeCalculator.gated_delta_net_state_dtype(torch_dtype, cache_dtype, ssm_dtype)
line = f"mamba cache dtype {conv} ssm state dtype {recurrent}"
print(line)
print(json.dumps({
    "line": line,
    "conv": str(conv),
    "recurrent": str(recurrent),
    "cache_dtype": cache_dtype,
    "ssm_dtype": ssm_dtype,
    "model_dtype": model_dtype_name,
}))
"""


def _load_json_stdout(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "container python failed")[-800:])
    lines = [line for line in result.stdout.splitlines() if line.strip().startswith("{")]
    if not lines:
        raise RuntimeError(result.stdout[-800:] or "container python returned no json")
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(result.stdout[-800:]) from exc
    if not isinstance(payload, dict):
        raise RuntimeError("container python returned a non-object")
    return payload


def _prepare_max_prompt(container_id: str, repeats: int = 0) -> dict[str, Any]:
    floor = LONG_CONTEXT_TARGET_TOKENS - MAX_OUTPUT_BUDGET
    limit = CONTEXT_LIMIT_TOKENS - MAX_OUTPUT_BUDGET
    target = (floor + limit) // 2
    result = _docker_python(
        container_id,
        _MAX_PROMPT_SCRIPT,
        [
            base64.b64encode(cap_png_bytes()).decode("ascii"),
            str(MAX_PIXELS),
            str(floor),
            str(limit),
            "4",
            str(target),
            str(repeats),
        ],
        timeout=180,
        env={"CUDA_VISIBLE_DEVICES": ""},
    )
    return _load_json_stdout(result)


def _probe_gdn_state_dtype(container_id: str) -> dict[str, Any]:
    result = subprocess.run(
        ["docker", "exec", "-e", "CUDA_VISIBLE_DEVICES=", "-i", container_id, "python3", "-"],
        input=_GDN_SCRIPT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    payload = _load_json_stdout(result)
    phrases = gdn_state_dtype_phrases(str(payload.get("line") or ""))
    if not phrases:
        raise RuntimeError(f"gdn probe did not report a state dtype: {payload}")
    payload["phrases"] = phrases
    return payload


def _quality_body(config: RunConfig, max_tokens: int) -> bytes:
    return build_chat_request(
        config.served_model,
        kind="image",
        images=1,
        text="Read the table. Reply with the four numbers in row-major order.",
        max_tokens=max_tokens,
        image_url=table_image_data_url(),
    )


def _run_quality(config: RunConfig, measurements: dict[str, Any]) -> dict[str, Any]:
    """Ask for the four table numbers. Retry once at 4096 when reasoning consumes 2048."""
    used_tokens = QUALITY_FIRST_TOKENS
    body = _quality_body(config, used_tokens)
    observed = _chat_observed(config, body)
    if quality_should_retry(observed.get("content"), observed.get("finish_reason")):
        used_tokens = MAX_OUTPUT_BUDGET
        body = _quality_body(config, used_tokens)
        observed = _chat_observed(config, body)
    content = observed.get("content") if isinstance(observed.get("content"), str) else ""
    accepted = table_numbers_in_content(content)
    http_ok = observed["code"] == 200 and _openai_ok(observed.get("payload"))
    evidence = {"table_answer": "content"} if accepted else {}
    detail = (
        f"max_tokens={used_tokens} finish={observed.get('finish_reason')} "
        f"content={content[:300]!r} reasoning={(observed.get('reasoning') or '')[:180]!r}"
    )
    status_name, judged = judge_case("quality_image", http_ok=http_ok, evidence=evidence, detail=detail)
    _replace_request(
        measurements,
        {
            "id": "quality_image",
            "code": observed["code"],
            "ttft_sec": None,
            "ttft_method": "not_applicable_non_streaming",
            "latency_sec": observed["latency_sec"],
            "output_tok_s": observed["output_tok_s"],
            "usage": observed["usage"],
            "finish_reason": observed.get("finish_reason"),
            "content": content[:2000],
            "reasoning": (observed.get("reasoning") or "")[:2000],
            "answer": content[:2000],
            "max_tokens": used_tokens,
            "table_answer": accepted,
        },
    )
    measurements["quality_answer"] = content[:2000]
    measurements["quality_request_body"] = body.decode("utf-8")
    return {
        "case": _case("quality_image", status_name, judged, config),
        "accepted": accepted,
        "body": body,
        "content": content,
        "finish_reason": observed.get("finish_reason"),
        "max_tokens": used_tokens,
    }


def _watch_metrics(container_id: str, seconds: float) -> tuple[subprocess.Popen[str], list[str], threading.Thread]:
    script = (
        "import time, urllib.request\n"
        f"deadline = time.time() + {seconds}\n"
        "while time.time() < deadline:\n"
        "    try:\n"
        "        data = urllib.request.urlopen('http://127.0.0.1:8080/metrics', timeout=2).read().decode()\n"
        "    except Exception as exc:\n"
        "        print(f'ERR {exc}', flush=True)\n"
        "        time.sleep(2)\n"
        "        continue\n"
        "    kept = ''\n"
        "    for line in data.splitlines():\n"
        "        if line.startswith('vllm:num_preemptions_total') or line.startswith('vllm:num_preemptions{') or line.startswith('vllm:num_preemptions '):\n"
        "            kept = line\n"
        "            break\n"
        "    print(kept or 'MISSING', flush=True)\n"
        "    time.sleep(2)\n"
    )
    proc = subprocess.Popen(
        ["docker", "exec", "-i", container_id, "python3", "-"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdin is not None and proc.stdout is not None
    chunks: list[str] = []

    def _drain() -> None:
        assert proc.stdout is not None
        collected: list[str] = []
        for line in proc.stdout:
            collected.append(line)
        chunks.append("".join(collected))

    proc.stdin.write(script)
    proc.stdin.close()
    reader = threading.Thread(target=_drain, name="max-metrics", daemon=True)
    reader.start()
    return proc, chunks, reader


def _run_parallel_chat(config: RunConfig, bodies: list[bytes]) -> list[dict[str, Any]]:
    ordered: list[dict[str, Any] | None] = [None] * len(bodies)
    with ThreadPoolExecutor(max_workers=max(1, len(bodies))) as pool:
        futures = {pool.submit(_chat_observed, config, body): index for index, body in enumerate(bodies)}
        for future in as_completed(futures):
            ordered[futures[future]] = future.result()
    return [item if item is not None else _empty_observation(0, "missing observation", 0.0) for item in ordered]


def _run_multimodal_max(
    config: RunConfig,
    container_id: str,
    results: list[CaseResult],
    measurements: dict[str, Any],
) -> None:
    """Exercise 4 images at the pixel cap, a near-32K budget, 4096 outputs, and concurrency 4."""
    _code, status = _status(config)
    if status.get("state") != "ready":
        load_code, load_body = _post_control(config, "load", config.load_timeout_sec)
        if load_code != 200:
            results.append(_case("multimodal_max", "failed", f"model was not ready for the max case: {load_body}", config))
            measurements["multimodal_max"] = {"ok": False, "reason": "not ready"}
            measurements["inference_peak_mib"] = None
            return
    try:
        prepared = _prepare_max_prompt(container_id)
    except RuntimeError as exc:
        results.append(_case("multimodal_max", "failed", f"max prompt was not built: {exc}", config))
        measurements["multimodal_max"] = {"ok": False, "reason": str(exc)[:500]}
        measurements["inference_peak_mib"] = None
        return
    if not pixels_hit_cap(int(prepared["source_pixels"]), int(prepared["processed_pixels"])):
        results.append(
            _case(
                "multimodal_max",
                "failed",
                f"processor did not bind max_pixels: {prepared['source_pixels']} -> {prepared['processed_pixels']}",
                config,
            )
        )
        measurements["multimodal_max"] = {"ok": False, "processor": prepared}
        measurements["inference_peak_mib"] = None
        return
    image_url = cap_image_data_url()
    calibrate_config = replace(config, request_timeout_sec=MULTIMODAL_MAX_TIMEOUT_SEC)

    def _one(text: str, max_tokens: int, *, min_tokens: int | None = None) -> bytes:
        return build_chat_request(
            config.served_model,
            kind="multi_image",
            images=4,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            text=text,
            image_url=image_url,
        )

    text = str(prepared["text"])
    probe = _chat_observed(calibrate_config, _one(text, 1))
    server_prompt = _usage_int(probe.get("usage"), "prompt_tokens")
    if probe["code"] != 200 or not output_budget_fits(server_prompt):
        unit = int(prepared.get("unit_tokens") or 0)
        repeats = int(prepared.get("repeats") or 0)
        if unit <= 0 or repeats <= 0 or server_prompt <= 0:
            results.append(
                _case(
                    "multimodal_max",
                    "failed",
                    f"calibration did not land in the context window: code={probe['code']} prompt={server_prompt} estimate={prepared.get('prompt_tokens_estimate')}",
                    config,
                )
            )
            measurements["multimodal_max"] = {"ok": False, "calibration": {"code": probe["code"], "prompt_tokens": server_prompt}}
            measurements["inference_peak_mib"] = None
            return
        target = (LONG_CONTEXT_TARGET_TOKENS + CONTEXT_LIMIT_TOKENS) // 2 - MAX_OUTPUT_BUDGET
        adjusted = max(1, repeats + round((target - server_prompt) / unit))
        try:
            prepared = _prepare_max_prompt(container_id, adjusted)
        except RuntimeError as exc:
            results.append(_case("multimodal_max", "failed", f"adjusted prompt was not built: {exc}", config))
            measurements["inference_peak_mib"] = None
            return
        text = str(prepared["text"])
        probe = _chat_observed(calibrate_config, _one(text, 1))
        server_prompt = _usage_int(probe.get("usage"), "prompt_tokens")
        if probe["code"] != 200 or not output_budget_fits(server_prompt):
            results.append(
                _case(
                    "multimodal_max",
                    "failed",
                    f"adjusted calibration is outside the context window: code={probe['code']} prompt={server_prompt}",
                    config,
                )
            )
            measurements["multimodal_max"] = {"ok": False, "calibration": {"code": probe["code"], "prompt_tokens": server_prompt}}
            measurements["inference_peak_mib"] = None
            return
    bodies = [_one(text, MAX_OUTPUT_BUDGET, min_tokens=MAX_OUTPUT_BUDGET) for _ in range(4)]
    if any(len(body) > 32 * 1024 * 1024 for body in bodies):
        results.append(_case("multimodal_max", "failed", "max request body exceeds 32MiB", config))
        measurements["inference_peak_mib"] = None
        return
    before_metrics = _metrics_text(container_id)
    before_preempt = _metric_value(before_metrics, "vllm:num_preemptions")
    watcher, chunks, reader = _watch_metrics(container_id, MULTIMODAL_MAX_TIMEOUT_SEC)
    sampler = GpuSampler()
    window_start = time.time()
    sampler.start()
    try:
        observations = _run_parallel_chat(calibrate_config, bodies)
    finally:
        window_end = time.time()
        sampler.stop()
        watcher.kill()
        reader.join(timeout=5)
    host_peak = sampler.peak_between(window_start, window_end)
    spot = _gpu_used_mib()
    if spot is not None:
        host_peak = spot if host_peak is None else max(host_peak, spot)
    blob = chunks[0] if chunks else ""
    samples = _metric_samples(blob, "vllm:num_preemptions")
    if before_preempt is None and samples:
        before_preempt = 0.0
    since = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(window_start - 2))
    logs = _logs(container_id, since)
    oom = "out of memory" in (blob + logs).lower() or any("out of memory" in item["text"].lower() for item in observations)
    preemption = before_preempt is not None and any(value > before_preempt for value in samples)
    preemption_clear = before_preempt is not None and bool(samples) and not preemption
    summaries = []
    output_ok = True
    context_ok = True
    http_ok = True
    for item in observations:
        prompt = _usage_int(item.get("usage"), "prompt_tokens")
        completion = _usage_int(item.get("usage"), "completion_tokens")
        finish = item.get("finish_reason")
        summaries.append(
            {
                "code": item["code"],
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "finish_reason": finish,
                "latency_sec": item["latency_sec"],
            }
        )
        http_ok = http_ok and item["code"] == 200 and _openai_ok(item.get("payload"))
        output_ok = output_ok and completion == MAX_OUTPUT_BUDGET and finish == "length"
        context_ok = context_ok and output_budget_fits(prompt)
    evidence: dict[str, str] = {}
    if oom:
        evidence["oom"] = "log"
    if preemption:
        evidence["preemption"] = "metrics"
    if pixels_hit_cap(int(prepared["source_pixels"]), int(prepared["processed_pixels"])):
        evidence["pixels"] = "processor"
    if context_ok:
        evidence["context_tokens"] = "usage"
    if output_ok:
        evidence["output_budget"] = "usage"
    if len(bodies) == 4 and all(body.count(b"image_url") >= 4 for body in bodies):
        evidence["concurrency"] = "requests"
    if preemption_clear:
        evidence["preemption_clear"] = "metrics"
    detail = (
        f"processed_pixels={prepared.get('processed_pixels')} source_pixels={prepared.get('source_pixels')} "
        f"calibration_prompt={server_prompt} preemption_before={before_preempt} "
        f"preemption_samples={samples[:8]} peak_mib={host_peak} requests={summaries}"
    )
    status_name, judged = judge_case(
        "multimodal_max",
        http_ok=http_ok and not oom,
        evidence=evidence,
        detail=detail,
    )
    measurements["multimodal_max"] = {
        "ok": status_name == "passed",
        "elapsed_sec": window_end - window_start,
        "processor": {
            "source_pixels": prepared.get("source_pixels"),
            "processed_width": prepared.get("processed_width"),
            "processed_height": prepared.get("processed_height"),
            "processed_pixels": prepared.get("processed_pixels"),
            "vision_tokens": prepared.get("vision_tokens"),
        },
        "calibration_prompt_tokens": server_prompt,
        "requests": summaries,
        "host_peak_mib": host_peak,
        "preemption_before": before_preempt,
        "preemption_clear": preemption_clear,
    }
    measurements["inference_peak_mib"] = host_peak
    measurements["inference_peak_note"] = "host nvidia-smi peak during the multimodal max window"
    results.append(_case("multimodal_max", status_name, judged, config))


def _client_and_busy(config: RunConfig, load_ok: bool, load_body: object) -> list[CaseResult]:
    if not load_ok:
        return [
            _blocked("client_disconnect", load_body, config),
            _blocked("disconnect_busy", load_body, config),
        ]
    disconnect_body = build_chat_request(
        config.served_model,
        kind="text",
        stream=True,
        max_tokens=512,
        text="Write a numbered list from 1 to 200.",
    )
    _close_after_send(config, "POST", "/v1/chat/completions", disconnect_body)
    saw_active = False
    deadline = time.monotonic() + config.request_timeout_sec
    last_status: dict[str, Any] = {}
    while time.monotonic() < deadline:
        _code, last_status = _status(config)
        active = int(last_status.get("active_requests") or 0)
        if active > 0:
            saw_active = True
        if saw_active and active == 0 and last_status.get("state") == "ready":
            break
        time.sleep(0.5)
    disconnect_ok = saw_active and last_status.get("active_requests") == 0 and last_status.get("state") == "ready"
    disconnect = _case(
        "client_disconnect",
        "passed" if disconnect_ok else "failed",
        f"saw_active={saw_active} status={last_status}",
        config,
    )

    body = build_chat_request(config.served_model, kind="text", stream=True, max_tokens=4096, text="Count slowly from 1 to 400.")
    holder: dict[str, Any] = {}

    def _call() -> None:
        holder["result"] = _chat(config, body)

    worker = threading.Thread(target=_call, name="gpu-stream")
    worker.start()
    inflight = False
    wait_deadline = time.monotonic() + 60
    while time.monotonic() < wait_deadline:
        _code, status = _status(config)
        if int(status.get("active_requests") or 0) > 0:
            inflight = True
            break
        if not worker.is_alive():
            break
        time.sleep(0.2)
    unload_code, unload_body = _post_control(config, "unload", 30)
    worker.join(timeout=config.request_timeout_sec)
    _, after = _status(config)
    busy_ok = inflight and unload_code == 409 and after.get("active_requests") == 0 and not worker.is_alive()
    busy = _case(
        "disconnect_busy",
        "passed" if busy_ok else "failed",
        f"inflight={inflight} unload={unload_code} {unload_body} after={after}",
        config,
    )
    return [disconnect, busy]


def _repeat_lifecycle(
    config: RunConfig,
    container_id: str,
    measurements: dict[str, Any],
    sampler: GpuSampler,
) -> tuple[bool, list[float], list[float]]:
    _code, status = _status(config)
    if status.get("state") == "failed":
        measurements["control_disconnect_case"] = {
            "status": "failed",
            "detail": f"cold load did not reach ready; extra loads were not repeated. status={status}",
        }
        measurements["_repeat_case"] = (
            "failed",
            f"cold load did not reach ready; lifecycle cycles were not repeated. status={status}",
        )
        return False, [], []
    if status.get("state") not in {"ready", "unloaded"}:
        _post_control(config, "load", config.load_timeout_sec)

    unload_samples: list[float] = []
    unload_stop = threading.Event()
    poller = threading.Thread(
        target=_poll_status_samples,
        args=(config, unload_stop, unload_samples),
        name="unload-status",
        daemon=True,
    )
    poller.start()
    with ThreadPoolExecutor(max_workers=2) as pool:
        unload_futures = [pool.submit(_post_control, config, "unload", 60) for _ in range(2)]
        unload_pair = [future.result() for future in unload_futures]
    unload_stop.set()
    poller.join(timeout=2)
    time.sleep(2)
    first_residual = _gpu_used_mib()
    unload_duplicate_ok = all(code == 200 and body.get("state") == "unloaded" for code, body in unload_pair)

    started = time.time()
    _close_after_send(config, "POST", "/control/load", b"{}")
    deadline = time.monotonic() + config.load_timeout_sec
    final: dict[str, Any] = {}
    while time.monotonic() < deadline:
        _code, final = _status(config)
        if final.get("state") in {"ready", "failed"}:
            break
        time.sleep(1)
    warm_loads = [time.time() - started]
    disconnect_ok = final.get("state") == "ready" and final.get("residency") == "resident"
    results_detail = f"disconnect_load status={final}"

    residuals: list[int | None] = [first_residual]
    for _ in range(2):
        window = time.time()
        _post_control(config, "load", config.load_timeout_sec)
        warm_loads.append(time.time() - window)
        _post_control(config, "unload", 60)
        time.sleep(2)
        residuals.append(_gpu_used_mib())
    numeric = [value for value in residuals if value is not None]
    workers = _worker_cmds(container_id)
    gone = not workers
    stable = len(numeric) >= 3 and max(numeric) - min(numeric) <= RESIDUAL_GROWTH_MIB
    evidence = {"workers_gone": "ps" if gone else "", "residual_stable": "nvidia-smi" if stable else ""}
    status_name, detail = judge_case(
        "repeat_lifecycle",
        http_ok=True,
        evidence=evidence,
        detail=f"residuals_mib={residuals} workers={workers} peak_during_cycles={sampler.peak_between(started, time.time())}",
    )
    measurements["residuals_mib"] = residuals
    measurements["control_disconnect"] = {"ok": disconnect_ok, "detail": results_detail}
    # The case list is appended by the caller for duplicate/status. Disconnect is recorded here.
    measurements["control_disconnect_case"] = {
        "status": "passed" if disconnect_ok else "failed",
        "detail": results_detail,
    }
    # Stash the repeat case on the measurement object via a side channel the caller reads.
    measurements["_repeat_case"] = (status_name, detail)
    return unload_duplicate_ok, unload_samples, warm_loads


def _failure_cases(config: RunConfig) -> list[CaseResult]:
    """Recreate only this compose service, then restore the original definition."""
    bad = _with_override(
        config,
        "services:\n  vllm-runtime:\n    environment:\n      MODEL_PATH: /models/does-not-exist-for-failure-check\n",
        lambda: _expect_failed_load(config, allow_not_resident=True),
    )
    timeout = _with_override(
        config,
        "services:\n  vllm-runtime:\n    environment:\n      LOAD_TIMEOUT_SEC: \"1\"\n",
        lambda: _expect_failed_load(config, allow_not_resident=False),
    )
    crash = _crash_worker(config)
    restarted = _restart_by_controller_exit(config)
    return [
        _case("failure_bad_model", bad[0], bad[1], config),
        _case("failure_load_timeout", timeout[0], timeout[1], config),
        _case("failure_worker_crash", crash[0], crash[1], config),
        _case("container_restart", restarted[0], restarted[1], config),
    ]


def _with_override(config: RunConfig, yaml_text: str, action: Callable[[], tuple[str, str]]) -> tuple[str, str]:
    try:
        owned = _owned_container(config.root, config.service)
    except Exception as exc:
        return "not_run", str(exc)
    del owned
    handle = tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False, encoding="utf-8")
    override = Path(handle.name)
    handle.write(yaml_text)
    handle.close()
    try:
        recreated = _compose(
            config.root,
            ["up", "-d", "--no-deps", "--force-recreate", config.service],
            files=[override],
        )
        if recreated.returncode != 0:
            return "failed", recreated.stderr.strip() or "override recreate failed"
        _owned_container(config.root, config.service)
        ready = _wait_http(config)
        if not ready:
            return "failed", "recreated controller did not answer /health"
        return action()
    finally:
        restored = _compose(config.root, ["up", "-d", "--no-deps", "--force-recreate", config.service])
        override.unlink(missing_ok=True)
        if restored.returncode != 0:
            return "failed", "failed to restore the original compose service"


def _expect_failed_load(config: RunConfig, *, allow_not_resident: bool) -> tuple[str, str]:
    code, body = _post_control(config, "load", max(30, config.load_timeout_sec))
    _status_code, status = _status(config)
    state = status.get("state")
    residency = status.get("residency")
    if code == 200 or state != "failed":
        return "failed", f"expected failed load, got {code} {body} {status}"
    if allow_not_resident:
        if residency not in {"not_resident", "unknown"}:
            return "failed", f"residency does not match ownership: {status}"
        return "passed", f"bad model failed with {status}"
    if residency == "not_resident" and state == "failed":
        workers = []
        try:
            workers = _worker_cmds(_owned_container(config.root, config.service))
        except RuntimeError:
            workers = ["inspect failed"]
        if workers:
            return "failed", f"not_resident reported while workers remain: {workers} {status}"
    return "passed", f"load deadline recorded {status}"


def _wait_http(config: RunConfig, timeout: float = 60) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        code, health = _health(config)
        if code == 200 and health.get("controller") == "ok":
            return True
        time.sleep(1)
    return False


def _container_started_at(container_id: str) -> str:
    inspected = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.StartedAt}}", container_id],
        capture_output=True,
        text=True,
        check=False,
    )
    return inspected.stdout.strip()


def _crash_worker(config: RunConfig) -> tuple[str, str]:
    try:
        container_id = _owned_container(config.root, config.service)
    except Exception as exc:
        return "not_run", str(exc)
    if not _wait_http(config):
        return "failed", "controller was not answering /health before the crash load"
    load_code, load_body = _post_control(config, "load", config.load_timeout_sec)
    if load_code != 200:
        return "failed", f"crash load did not become ready: {load_code} {load_body}"
    container_id = _owned_container(config.root, config.service)
    listed = subprocess.run(
        ["docker", "exec", container_id, "ps", "-eo", "pid,cmd"],
        capture_output=True,
        text=True,
        check=False,
    )
    pids = []
    for line in listed.stdout.splitlines():
        if "vllm" in line.lower() and "serve" in line.lower():
            pids.append(line.split()[0])
    if not pids:
        return "failed", f"no owned vllm process to crash: {listed.stdout}"
    killed = subprocess.run(
        ["docker", "exec", container_id, "kill", "-KILL", *pids],
        capture_output=True,
        text=True,
        check=False,
    )
    time.sleep(2)
    _code, status = _status(config)
    workers = _worker_cmds(container_id)
    if status.get("state") != "failed":
        return "failed", f"crash did not become failed: {status} kill={killed.stderr}"
    if workers and status.get("residency") == "not_resident":
        return "failed", f"not_resident while workers remain: {workers}"
    if status.get("active_requests", 0) and not workers:
        return "failed", f"active remained after every worker was gone: {status}"
    return "passed", f"crash status={status} workers={workers}"


def _restart_by_controller_exit(config: RunConfig) -> tuple[str, str]:
    try:
        container_id = _owned_container(config.root, config.service)
    except Exception as exc:
        return "not_run", str(exc)
    listed = subprocess.run(
        ["docker", "exec", container_id, "ps", "-eo", "pid,cmd"],
        capture_output=True,
        text=True,
        check=False,
    )
    pids = []
    for line in listed.stdout.splitlines():
        parts = line.split(None, 1)
        if len(parts) != 2 or parts[0] == "1":
            continue
        if "uvicorn" in parts[1] or "controller.main" in parts[1]:
            pids.append(parts[0])
    if not pids:
        return "failed", f"controller pid was not found: {listed.stdout}"
    started_at = _container_started_at(container_id)
    subprocess.run(["docker", "exec", container_id, "kill", "-TERM", *pids], check=False)
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        try:
            new_id = _owned_container(config.root, config.service)
        except RuntimeError:
            time.sleep(1)
            continue
        if _container_started_at(new_id) in {"", started_at}:
            time.sleep(1)
            continue
        if not _wait_http(config, timeout=20):
            time.sleep(1)
            continue
        code, health = _health(config)
        status_code, status = _status(config)
        workers = _worker_cmds(new_id)
        ok = (
            code == 200
            and health.get("controller") == "ok"
            and status_code == 200
            and status.get("state") == "unloaded"
            and status.get("active_requests") == 0
            and not workers
        )
        return ("passed" if ok else "failed", f"restarted={new_id} health={health} status={status} workers={workers}")
    return "failed", "unless-stopped did not start a new container for this service"


def write_execution(results: list[CaseResult], started: str) -> None:
    cases = [{"id": item.id, "status": item.status, "detail": item.detail} for item in results]
    statuses = [item.status for item in results]
    if statuses and all(status == "passed" for status in statuses):
        suite = "passed"
    elif any(status == "failed" for status in statuses):
        suite = "failed"
    else:
        suite = "not_run"
    payload_cases = cases
    write_result(OUTPUT, "gpu_api", payload_cases, started, utc_now())
    text = OUTPUT.read_text(encoding="utf-8")
    payload = json.loads(text)
    payload["status"] = suite
    payload["gpu_inference_success"] = suite == "passed"
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _compose_with_profile(root: Path, profile: str | None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if profile is None:
        env.pop("MODEL_PROFILE", None)
    else:
        env["MODEL_PROFILE"] = profile
    return subprocess.run(
        ["docker", "compose", "up", "-d", "--no-deps", "--force-recreate", "--no-build", SERVICE],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def run_comparison(
    config: RunConfig,
    quality_body: bytes | None = None,
    fp8_quality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure the BF16 KV profile without replacing the default FP8 profile."""
    record: dict[str, Any] = {"profile": "qwen3.8-27b-kv-bf16", "status": "not_run"}
    try:
        created = _compose_with_profile(config.root, "qwen3.8-27b-kv-bf16")
        record["recreate_rc"] = created.returncode
        if created.returncode != 0:
            record["status"] = "failed"
            record["error"] = (created.stderr or created.stdout)[-800:]
            return record
        deadline = time.monotonic() + 90
        status: dict[str, Any] = {}
        while time.monotonic() < deadline:
            _code, status = _status(config)
            if status.get("state") == "unloaded" and status.get("residency") == "not_resident":
                break
            time.sleep(2)
        record["startup_status"] = status
        if status.get("state") != "unloaded":
            record["status"] = "failed"
            record["error"] = "comparison container did not start unloaded"
            return record
        record["baseline_mib"] = _gpu_used_mib()
        started = time.time()
        load_code, load_body = _post_control(config, "load", config.load_timeout_sec)
        record["load_sec"] = time.time() - started
        record["load_peak_mib"] = _gpu_used_mib()
        record["load_code"] = load_code
        record["load_state"] = load_body.get("state") if isinstance(load_body, dict) else None
        container_id = _owned_container(config.root, config.service)
        logs = _logs(container_id, time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(started - 2)))
        record["backend_lines"] = _relevant_lines(logs)
        evidence = evidence_from_text(logs)
        _note_kv_scale(evidence, logs)
        record["kv_evidence"] = evidence
        if load_code == 200 and isinstance(load_body, dict) and load_body.get("state") == "ready":
            if not quality_body:
                record["quality"] = {
                    "status": "not_run",
                    "reason": "no quality request body",
                    "fp8_accepted": None if fp8_quality is None else fp8_quality.get("accepted"),
                }
            else:
                observed = _chat_observed(
                    replace(config, request_timeout_sec=max(config.request_timeout_sec, 900)),
                    quality_body,
                )
                bf16_content = observed.get("content") if isinstance(observed.get("content"), str) else ""
                record["quality"] = {
                    "status": "measured",
                    "fp8_content": None if fp8_quality is None else (fp8_quality.get("content") or "")[:2000],
                    "fp8_accepted": None if fp8_quality is None else bool(fp8_quality.get("accepted")),
                    "bf16_content": bf16_content[:2000],
                    "bf16_reasoning": (observed.get("reasoning") or "")[:2000],
                    "bf16_accepted": table_numbers_in_content(bf16_content),
                    "bf16_code": observed["code"],
                    "bf16_finish_reason": observed.get("finish_reason"),
                    "bf16_usage": observed.get("usage"),
                    "bf16_latency_sec": observed["latency_sec"],
                    "ttft_sec": None,
                }
            unload_started = time.time()
            unload_code, _unload_body = _post_control(config, "unload", 60)
            record["unload_sec"] = time.time() - unload_started
            record["unload_code"] = unload_code
            time.sleep(2)
            record["residual_mib"] = _gpu_used_mib()
            record["status"] = "measured"
        else:
            record["status"] = "load_failed"
            record["error"] = str(load_body)[:800]
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = str(exc)
    finally:
        restored = _compose_with_profile(config.root, None)
        record["restored_rc"] = restored.returncode
        record["restored_err"] = (restored.stderr or "")[-400:]
    return record


def _clear_nonstream_ttft(measurements: dict[str, Any]) -> None:
    for item in measurements.get("requests", []):
        if not isinstance(item, dict) or item.get("id") == "text_stream":
            continue
        item["ttft_sec"] = None
        item["ttft_method"] = "not_applicable_non_streaming"


def _remeasure_stream_ttft(config: RunConfig, measurements: dict[str, Any]) -> dict[str, Any]:
    observed = _chat_observed(
        config,
        build_chat_request(
            config.served_model,
            kind="text",
            stream=True,
            max_tokens=32,
            text="Reply with the single word ok.",
        ),
    )
    for item in measurements.get("requests", []):
        if isinstance(item, dict) and item.get("id") == "text_stream":
            item["ttft_sec"] = observed["ttft_sec"]
            item["ttft_method"] = "sse_first_token"
            item["latency_sec"] = observed["latency_sec"]
    measurements["ttft"] = {
        "method": "sse_first_token",
        "text_stream_sec": observed["ttft_sec"],
        "code": observed["code"],
    }
    return observed


def _merge_cases(updates: list[CaseResult]) -> None:
    payload = json.loads(OUTPUT.read_text(encoding="utf-8"))
    pending = {item.id: item for item in updates}
    for case in payload.get("cases", []):
        item = pending.pop(case.get("id"), None)
        if item is None:
            continue
        case["status"] = item.status
        case["detail"] = item.detail
    for item in pending.values():
        payload.setdefault("cases", []).append({"id": item.id, "status": item.status, "detail": item.detail})
    statuses = [case.get("status") for case in payload.get("cases", [])]
    if any(status == "failed" for status in statuses):
        suite = "failed"
    elif statuses and all(status == "passed" for status in statuses):
        suite = "passed"
    else:
        suite = "not_run"
    payload["status"] = suite
    payload["gpu_inference_success"] = suite == "passed"
    payload["finished_at"] = utc_now()
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_connection(
    config: RunConfig,
    *,
    confirmed: bool,
    host_peak: int | None,
    baseline: int | None,
    reason: str,
) -> None:
    profile: dict[str, Any] = {"profileId": "qwen3.8-27b-fp8-kv", "confirmed": confirmed}
    if confirmed and host_peak is not None and baseline is not None and host_peak >= baseline:
        profile["inferencePeakBytes"] = (host_peak - baseline) * _MIB
        profile["inferencePeakHostMiB"] = host_peak
        profile["baselineMiB"] = baseline
        profile["maxConcurrency"] = 4
        profile["limits"] = {
            "maxPixels": MAX_PIXELS,
            "maxOutputTokens": MAX_OUTPUT_BUDGET,
            "maxImagesPerPrompt": 4,
        }
    else:
        profile["confirmed"] = False
        profile["unconfirmed"] = ["maxConcurrency", "maxPixels", "maxOutputTokens", "inferencePeakBytes"]
        profile["reason"] = reason
    payload = {
        "baseURL": config.base_url,
        "inferencePath": "/v1/chat/completions",
        "servedModel": config.served_model,
        "prepare": {"argv": [str(PREPARE.resolve())]},
        "resourceProfile": profile,
    }
    CONNECTION.parent.mkdir(parents=True, exist_ok=True)
    CONNECTION.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_stage_notes(lines: list[str]) -> None:
    path = ROOT / "local" / "stage2_notes.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _append_stage_notes(lines: list[str]) -> None:
    path = ROOT / "local" / "stage2_notes.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if existing and not existing.endswith("\n"):
        existing += "\n"
    if existing and not existing.endswith("\n\n"):
        existing += "\n"
    path.write_text(existing + "\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _pre_load_baseline(config: RunConfig, measurements: dict[str, Any]) -> tuple[int | None, str]:
    """Use a fresh unloaded reading, or the stored pre-load value when the model is already up."""
    _code, status = _status(config)
    already_ready = status.get("state") == "ready" and status.get("residency") == "resident"
    stored = measurements.get("supplement_baseline_mib")
    if already_ready:
        if isinstance(stored, int):
            return stored, (
                f"모델이 이미 ready라 로드 전 baseline은 gpu_measurements.json의 "
                f"supplement_baseline_mib {stored} MiB를 사용했다."
            )
        return None, "모델이 이미 ready인데 로드 전 baseline 기록이 없다."
    measured = _gpu_used_mib()
    if measured is None:
        return None, "로드 전 nvidia-smi baseline을 읽지 못했다."
    return measured, f"로드 전 호스트 baseline은 {measured} MiB다."


def _ensure_model_ready(config: RunConfig) -> tuple[bool, str]:
    started = _compose(config.root, ["up", "-d", "--no-build", "--no-recreate", config.service])
    if started.returncode != 0:
        return False, (started.stderr or started.stdout or "compose up failed")[-500:]
    if not _wait_http(config, timeout=60):
        return False, "controller health did not answer"
    _code, status = _status(config)
    if status.get("state") == "ready" and status.get("residency") == "resident":
        return True, "already ready"
    load_code, load_body = _post_control(config, "load", config.load_timeout_sec)
    ready = load_code == 200 and isinstance(load_body, dict) and load_body.get("state") == "ready"
    return ready, str(load_body)


def run_supplement(config: RunConfig, *, skip_comparison: bool = False) -> int:
    """Rerun the missing stage-2 measurements without the lifecycle cases."""
    print("supplement: load existing measurements", flush=True)
    measurements: dict[str, Any] = {}
    loaded_measurements = False
    if MEASUREMENTS.exists():
        measurements = json.loads(MEASUREMENTS.read_text(encoding="utf-8"))
        loaded_measurements = True
    notes = [
        "# 2단계 보완 기록",
        "",
        "작성 시각은 이 실행이 끝난 시점이다. README와 benchmarks/README.md는 수정하지 않았다. git 명령은 실행하지 않았다.",
        "",
        "통과해 있던 load/unload 반복, failure, container restart는 다시 실행하지 않았다.",
        "",
        "품질 요청은 thinking을 끄지 않았다. 2048토큰에서 content가 비고 finish_reason이 length이면 4096으로 한 번 더 보낸다.",
        "",
        "연결 정보의 정본은 tests/outputs/inferswap_connection.json 이다. prepare.argv는 이 저장소 prepare-inferswap의 절대 경로 하나다.",
        "benchmarks/README.md의 prepare.argv는 vllm serve로 남아 있다. 그 파일은 고치지 말라는 지시를 따랐다.",
        "",
    ]
    updates: list[CaseResult] = []
    comparison: dict[str, Any] | None = None
    exit_code = 1
    try:
        baseline = _gpu_used_mib()
        measurements["supplement_baseline_mib"] = baseline
        print(f"supplement: baseline {baseline} MiB", flush=True)
        ready, ready_detail = _ensure_model_ready(config)
        notes.append(f"FP8 load: ready={ready} detail={ready_detail[:400]}")
        if not ready:
            notes.append("모델을 올리지 못해 GDN, TTFT, 품질, 최대 조건을 실행하지 못했다.")
            write_connection(config, confirmed=False, host_peak=None, baseline=baseline, reason="model did not become ready")
            return 1
        container_id = _owned_container(config.root, config.service)
        print("supplement: gdn state dtype", flush=True)
        try:
            probe = _probe_gdn_state_dtype(container_id)
        except RuntimeError as exc:
            probe = None
            notes.append(f"GDN state dtype probe 실패: {exc}")
            evidence = dict(measurements.get("backend_evidence") or {})
            evidence["gdn_dtype"] = ""
            measurements["backend_evidence"] = evidence
            status_name, detail = judge_case("backend_record", http_ok=True, evidence=evidence, detail=str(exc))
            updates.append(_case("backend_record", status_name, detail, config))
        else:
            evidence = dict(measurements.get("backend_evidence") or {})
            evidence["gdn_dtype"] = "probe"
            measurements["backend_evidence"] = evidence
            measurements["gdn_state_dtype"] = probe
            notes.append(
                f"GDN state dtype는 기동 로그에 없어 워커 argv와 Qwen3.5 설정으로 계산했다. "
                f"conv={probe.get('conv')} recurrent={probe.get('recurrent')} "
                f"cache={probe.get('cache_dtype')} ssm={probe.get('ssm_dtype')}"
            )
            status_name, detail = judge_case(
                "backend_record",
                http_ok=True,
                evidence=evidence,
                detail=str(probe.get("line")),
            )
            updates.append(_case("backend_record", status_name, detail, config))
        print("supplement: sse ttft", flush=True)
        _clear_nonstream_ttft(measurements)
        ttft = _remeasure_stream_ttft(config, measurements)
        notes.append(
            f"TTFT는 SSE의 첫 비어 있지 않은 content/reasoning 토큰만 기록한다. "
            f"text_stream={ttft.get('ttft_sec')} code={ttft.get('code')}. 비스트리밍 ttft_sec는 null이다."
        )
        print("supplement: quality image", flush=True)
        quality = _run_quality(config, measurements)
        updates.append(quality["case"])
        notes.append(
            f"품질: accepted={quality['accepted']} max_tokens={quality['max_tokens']} "
            f"finish={quality['finish_reason']} content={quality['content'][:240]!r}"
        )
        print("supplement: multimodal max", flush=True)
        before = len(updates)
        _run_multimodal_max(config, container_id, updates, measurements)
        max_case = updates[-1] if len(updates) > before else None
        max_passed = max_case is not None and max_case.status == "passed" and max_case.id == "multimodal_max"
        host_peak = measurements.get("inference_peak_mib")
        peak_value = host_peak if isinstance(host_peak, int) else None
        if max_passed:
            reason = "multimodal max passed"
        else:
            reason = max_case.detail if max_case is not None else "multimodal max did not run"
        notes.append(f"최대 조건: passed={max_passed} host_peak_mib={peak_value}")
        if not max_passed:
            notes.append(
                "최대 조건이 통과하지 않았다. max_model_len, max_pixels, 출력 4096, max_num_seqs는 낮추지 않았다. "
                "resourceProfile의 maxConcurrency, maxPixels, maxOutputTokens, inferencePeakBytes는 확정하지 않는다."
            )
        write_connection(
            config,
            confirmed=max_passed,
            host_peak=peak_value,
            baseline=baseline if isinstance(baseline, int) else None,
            reason=reason[:500],
        )
        _save_measurements(measurements)
        if OUTPUT.exists():
            _merge_cases(updates)
        if not skip_comparison:
            print("supplement: bf16 quality comparison", flush=True)
            comparison = run_comparison(config, quality["body"], {"content": quality["content"], "accepted": quality["accepted"]})
            COMPARISON.write_text(json.dumps(comparison, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            bf16 = comparison.get("quality") if isinstance(comparison.get("quality"), dict) else {}
            notes.append(
                f"BF16 비교 status={comparison.get('status')} "
                f"fp8_accepted={bf16.get('fp8_accepted')} bf16_accepted={bf16.get('bf16_accepted')} "
                f"bf16_content={str(bf16.get('bf16_content') or '')[:240]!r}"
            )
            notes.append("비교가 끝난 뒤 컨테이너는 기본 FP8 프로필로 되돌렸다. FP8 프로필을 BF16 결과로 바꾸지 않았다.")
        failed = [item.id for item in updates if item.status != "passed"]
        exit_code = 1 if failed else 0
        notes.append(f"이번 보완에서 통과하지 않은 케이스: {failed or '없음'}")
        return exit_code
    except Exception as exc:
        notes.append(f"보완 실행이 예외로 끝났다: {exc}")
        exit_code = 1
        return 1
    finally:
        notes.append("")
        notes.append("lifecycle 재실행을 하지 않은 것은 계획된 생략이다. 품질 thinking을 끄지 않은 것도 계획된 선택이다.")
        if comparison is None and not skip_comparison:
            notes.append("BF16 비교 기록이 없으면 그 단계 전에 실행이 멈춘 것이다.")
        if loaded_measurements:
            _save_measurements(measurements)
        _write_stage_notes(notes)


def run_max_only(config: RunConfig) -> int:
    """Rerun only the four-way multimodal max case. Other stage-2 checks stay as recorded."""
    print("max-only: load existing measurements", flush=True)
    measurements: dict[str, Any] = {}
    if MEASUREMENTS.exists():
        measurements = json.loads(MEASUREMENTS.read_text(encoding="utf-8"))
    baseline, baseline_note = _pre_load_baseline(config, measurements)
    notes = [
        "## 최대 조건만 재측정",
        "",
        baseline_note,
        "",
        "lifecycle, 품질 비교, TTFT, GDN probe는 다시 실행하지 않았다.",
        "채점 요청에만 min_tokens 4096을 넣었다. max_tokens 4096은 그대로다.",
        "",
    ]
    exit_code = 1
    try:
        ready, ready_detail = _ensure_model_ready(config)
        notes.append(f"FP8 load: ready={ready} detail={ready_detail[:400]}")
        if not ready:
            notes.append("모델을 올리지 못했다. 확정값은 넣지 않았다.")
            return 1
        container_id = _owned_container(config.root, config.service)
        updates: list[CaseResult] = []
        print("max-only: multimodal max", flush=True)
        _run_multimodal_max(config, container_id, updates, measurements)
        max_case = updates[-1] if updates else None
        max_passed = max_case is not None and max_case.status == "passed" and max_case.id == "multimodal_max"
        host_peak = measurements.get("inference_peak_mib")
        peak_value = host_peak if isinstance(host_peak, int) else None
        record = measurements.get("multimodal_max")
        record = record if isinstance(record, dict) else {}
        notes.append(f"최대 조건: passed={max_passed} host_peak_mib={peak_value}")
        notes.append(f"requests={record.get('requests')}")
        notes.append(
            f"preemption_before={record.get('preemption_before')} "
            f"preemption_clear={record.get('preemption_clear')}"
        )
        if not max_passed or max_case is None:
            notes.append(
                "조기 종료, preemption, OOM, 또는 그 밖의 실패라 확정값을 넣지 않았다. "
                "gpu_api.json과 inferswap_connection.json은 바꾸지 않았다."
            )
            return 1
        if not isinstance(baseline, int) or peak_value is None or peak_value < baseline:
            notes.append("peak 또는 로드 전 baseline이 없어 inferencePeakBytes를 확정하지 않았다.")
            return 1
        attributed = (peak_value - baseline) * _MIB
        _merge_cases([max_case])
        write_connection(
            config,
            confirmed=True,
            host_peak=peak_value,
            baseline=baseline,
            reason="multimodal max passed",
        )
        _save_measurements(measurements)
        notes.append(
            f"inferencePeakBytes={attributed} "
            f"(host {peak_value} MiB − baseline {baseline} MiB). "
            "maxConcurrency 4, maxPixels 262144, maxOutputTokens 4096을 확정했다."
        )
        notes.append("prepare.argv는 prepare-inferswap 절대 경로 하나다.")
        exit_code = 0
        return 0
    except Exception as exc:
        notes.append(f"최대 조건 재측정이 예외로 끝났다: {exc}")
        notes.append("확정값은 넣지 않았다.")
        exit_code = 1
        return 1
    finally:
        _append_stage_notes(notes)
        print(f"max-only: exit {exit_code}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Independent GPU/API verification")
    parser.add_argument("--check-plan", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Run the GPU/API cases against the compose service")
    parser.add_argument("--supplement", action="store_true", help="Rerun max-condition, quality, TTFT, and GDN checks only")
    parser.add_argument("--max-only", action="store_true", help="Rerun only the multimodal max case")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--service", default=SERVICE)
    parser.add_argument("--profile", default="qwen3.8-27b")
    parser.add_argument("--served-model", default="")
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--load-timeout-sec", type=float, default=600)
    parser.add_argument("--request-timeout-sec", type=float, default=600)
    parser.add_argument("--skip-comparison", action="store_true")
    args = parser.parse_args(argv)
    if args.check_plan:
        check_plan()
    if not args.execute and not args.supplement and not args.max_only:
        if not args.check_plan:
            print("GPU suite was not executed. Pass --execute to run it.", file=sys.stderr)
        return 0
    if not args.base_url:
        print("--base-url is required with --execute", file=sys.stderr)
        return 2
    started = utc_now()
    config = RunConfig(
        base_url=args.base_url.rstrip("/"),
        root=Path(args.root),
        service=args.service,
        profile=args.profile,
        served_model=args.served_model or args.profile,
        load_timeout_sec=args.load_timeout_sec,
        request_timeout_sec=args.request_timeout_sec,
    )
    if args.max_only:
        return run_max_only(config)
    if args.supplement:
        return run_supplement(config, skip_comparison=args.skip_comparison)
    results = execute_suite(config)
    write_execution(results, started)
    if not args.skip_comparison:
        comparison = run_comparison(config)
        COMPARISON.write_text(json.dumps(comparison, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if all(item.status == "passed" for item in results):
        return 0
    if any(item.status == "failed" for item in results):
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
