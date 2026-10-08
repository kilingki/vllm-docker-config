"""Independent GPU/API executor for section 11.2.

``--execute`` runs the cases against one Compose service. This module does not
refuse execution because of a stage number. Importing it does not load a model.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.results import utc_now, write_result

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "tests" / "outputs" / "gpu_api.json"
MEASUREMENTS = ROOT / "tests" / "outputs" / "gpu_measurements.json"
COMPARISON = ROOT / "tests" / "outputs" / "gpu_comparison.json"
SERVICE = "vllm-runtime"
STATUS_RESPONSE_LIMIT_SEC = 10.0
LONG_CONTEXT_TARGET_TOKENS = 30000
RESIDUAL_GROWTH_MIB = 256

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
        "pass_condition": "A table image with small text is accepted and the model answer is recorded for quality notes.",
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
    payload = {
        "model": served_model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "stream": stream,
    }
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
        ("gdn_dtype", ("mamba", "gdn", "linear_attention", "ssm cache", "gdn decode kernel")),
        ("prefix_cache", ("prefix cache hit", "cached tokens", "prefix_cache_hits")),
        ("oom", ("out of memory", "cuda oom")),
        ("preemption", ("preemptions:",)),
    ):
        if any(needle in lowered for needle in needles):
            found[label] = "log"
    return found


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


def _docker_python(container_id: str, script: str, args: list[str], timeout: float = 180) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", "exec", "-i", container_id, "python3", "-", *args],
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


def _answer_text(payload: dict[str, Any] | None) -> str:
    if not isinstance(payload, dict):
        return ""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    reasoning = message.get("reasoning")
    parts = [part for part in (content, reasoning) if isinstance(part, str) and part]
    return "\n".join(parts)


def _chat_observed(config: RunConfig, body: bytes) -> dict[str, Any]:
    """Time to first byte and full latency. Stream responses keep the SSE text."""
    request = urllib.request.Request(
        f"{config.base_url}/v1/chat/completions",
        data=body,
        method="POST",
    )
    request.add_header("Content-Type", "application/json")
    started = time.monotonic()
    ttft: float | None = None
    chunks: list[bytes] = []
    code = 0
    try:
        with urllib.request.urlopen(request, timeout=config.request_timeout_sec) as response:
            code = response.status
            while True:
                part = response.read(4096)
                if not part:
                    break
                if ttft is None:
                    ttft = time.monotonic() - started
                chunks.append(part)
    except urllib.error.HTTPError as exc:
        code = exc.code
        chunks.append(exc.read())
        if ttft is None:
            ttft = time.monotonic() - started
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
    return {
        "code": code,
        "payload": payload,
        "text": text,
        "ttft_sec": ttft,
        "latency_sec": latency,
        "usage": usage,
        "output_tok_s": (completion / latency) if latency > 0 and completion else None,
        "answer": _answer_text(payload),
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
    quality = chat_case(
        "quality_image",
        build_chat_request(
            config.served_model,
            kind="image",
            images=1,
            text="Read the table. Reply with the four numbers in row-major order.",
            max_tokens=64,
            image_url=table_image_data_url(),
        ),
        require_image=True,
    )
    if quality is not None:
        measurements["quality_answer"] = quality.get("answer") or quality.get("text", "")[:500]

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


def _run_multimodal_max(
    config: RunConfig,
    container_id: str,
    results: list[CaseResult],
    measurements: dict[str, Any],
) -> None:
    _code, status = _status(config)
    if status.get("state") != "ready":
        load_code, load_body = _post_control(config, "load", config.load_timeout_sec)
        if load_code != 200:
            results.append(_case("multimodal_max", "failed", f"model was not ready for the max case: {load_body}", config))
            return
    output_budget = 256
    bodies = [
        build_chat_request(
            config.served_model,
            kind="multi_image",
            images=config.image_count,
            max_tokens=output_budget,
            text="Read every image in this request.",
            image_url=table_image_data_url(),
        )
        for _ in range(4)
    ]
    ok, note, wave_evidence, elapsed = _run_parallel_observed(config, bodies, container_id)
    wave_evidence.update(evidence_from_text(_logs(container_id, "1s")))
    status_name, detail = judge_case(
        "multimodal_max",
        http_ok=ok and all(b"image_url" in body for body in bodies) and all(body.count(b"image_url") >= config.image_count for body in bodies),
        evidence=wave_evidence,
        detail=f"images={config.image_count} max_tokens={output_budget} concurrency=4 elapsed={elapsed:.1f}s {note}",
    )
    measurements["multimodal_max"] = {"ok": ok, "elapsed_sec": elapsed, "note": note[:500]}
    results.append(_case("multimodal_max", status_name, detail, config))


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


def run_comparison(config: RunConfig) -> dict[str, Any]:
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
            observed = _chat_observed(
                config,
                build_chat_request(
                    config.served_model,
                    kind="text",
                    max_tokens=32,
                    text="한국어로 한 문장만 답하세요.",
                ),
            )
            record["chat"] = {
                "code": observed["code"],
                "latency_sec": observed["latency_sec"],
                "ttft_sec": observed["ttft_sec"],
                "answer": (observed["answer"] or "")[:300],
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Independent GPU/API verification")
    parser.add_argument("--check-plan", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Run the GPU/API cases against the compose service")
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
    if not args.execute:
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
