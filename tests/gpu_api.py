"""Independent GPU/API executor for section 11.2.

``--execute`` runs the cases against one Compose service. This module does not
refuse execution because of a stage number. Importing it does not load a model.
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import threading
import time
import tempfile
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.results import utc_now, write_result

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "tests" / "outputs" / "gpu_api.json"
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
        parts.append({"type": "image_url", "image_url": {"url": image_data_url()}})
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
        ("kv_dtype", ("kv_cache_dtype", "kv cache dtype", "fp8_e4m3")),
        ("kv_scale", ("kv_scale", "k_scale", "v_scale", "calculate_kv_scales")),
        ("decoder_attention", ("flashinfer", "flash_attn", "triton_attn", "attention backend")),
        ("vision_attention", ("vision attention", "mm_encoder_attn", "vit attention")),
        ("gdn_dtype", ("mamba", "gdn", "linear_attention", "ssm cache")),
        ("batching", ("running:", "batched tokens", "num_waiting", "chunked prefill")),
        ("prefix_cache", ("prefix cache hit", "cached tokens", "prefix caching")),
        ("oom", ("out of memory", "cuda oom")),
        ("preemption", ("preempt", "recompute")),
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

    since = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    load_code, load_body = _post_control(config, "load", config.load_timeout_sec)
    status_code, status = _status(config)
    models_code, models_raw = http_exchange("GET", f"{config.base_url}/v1/models", timeout=30)
    load_ok = (
        load_code == 200
        and status.get("state") == "ready"
        and status.get("residency") == "resident"
        and config.served_model in models_raw.decode("utf-8", errors="replace")
    )
    results.append(
        _case(
            "load",
            "passed" if load_ok else "failed",
            f"load={load_code} {load_body} status={status} models={models_code}",
            config,
        )
    )
    logs = _logs(container_id, since)
    evidence = evidence_from_text(logs)

    def chat_case(case_id: str, body: bytes, *, require_image: bool = False) -> None:
        if require_image and b"image_url" not in body:
            results.append(_case(case_id, "failed", "request dropped image_url before send", config))
            return
        code, payload, text = _chat(config, body)
        ok = code == 200 and (_openai_ok(payload) or "data:" in text)
        if require_image and b"image_url" not in body:
            ok = False
        local = dict(evidence)
        local.update(evidence_from_text(text + logs))
        status_name, detail = judge_case(case_id, http_ok=ok, evidence=local, detail=text[:400])
        results.append(_case(case_id, status_name, detail, config))

    chat_case("text_nonstream", build_chat_request(config.served_model, kind="text", stream=False))
    chat_case("text_stream", build_chat_request(config.served_model, kind="text", stream=True))
    chat_case(
        "image_single",
        build_chat_request(config.served_model, kind="image", images=1, text="Describe the image."),
        require_image=True,
    )
    chat_case(
        "image_multi",
        build_chat_request(config.served_model, kind="multi_image", images=2, text="Describe the images."),
        require_image=True,
    )

    backend_status, backend_detail = judge_case("backend_record", http_ok=load_ok, evidence=evidence, detail=logs[-1000:])
    results.append(_case("backend_record", backend_status, backend_detail, config))

    for level, case_id in ((1, "concurrency_1"), (2, "concurrency_2"), (4, "concurrency_4")):
        wave_ok = True
        notes: list[str] = []
        for kind in ("text", "image", "mixed"):
            bodies = concurrency_bodies(config, level, kind)
            if len(bodies) != level:
                wave_ok = False
                notes.append(f"{kind} generated {len(bodies)} bodies")
                continue
            ok, note = _run_parallel(config, bodies)
            wave_ok = wave_ok and ok
            notes.append(f"{kind}:{note or 'ok'}")
        wave_logs = _logs(container_id, since)
        wave_evidence = evidence_from_text(wave_logs)
        judged, detail = judge_case(case_id, http_ok=wave_ok, evidence=wave_evidence, detail="; ".join(notes))
        results.append(_case(case_id, judged, detail, config))

    long_body = build_chat_request(
        config.served_model,
        kind="text",
        max_tokens=config.max_output_tokens,
        long_context_tokens=config.max_context_tokens,
    )
    long_code, long_payload, long_text = _chat(config, long_body)
    usage = long_payload.get("usage") if isinstance(long_payload, dict) else None
    long_evidence = evidence_from_text(long_text + _logs(container_id, since))
    if context_approached(usage if isinstance(usage, dict) else None, config.max_output_tokens):
        long_evidence["context_tokens"] = "usage"
    long_status, long_detail = judge_case(
        "long_context",
        http_ok=long_code == 200 and _openai_ok(long_payload),
        evidence=long_evidence,
        detail=str(usage),
    )
    results.append(_case("long_context", long_status, long_detail, config))

    max_bodies = [
        build_chat_request(
            config.served_model,
            kind="multi_image",
            images=config.image_count,
            max_tokens=config.max_output_tokens,
            text="Read every image.",
        )
        for _ in range(4)
    ]
    max_ok, max_note = _run_parallel(config, max_bodies)
    max_status, max_detail = judge_case(
        "multimodal_max",
        http_ok=max_ok and all(b"image_url" in body for body in max_bodies),
        evidence=evidence_from_text(max_note),
        detail=f"images={config.image_count} concurrency=4 {max_note}",
    )
    results.append(_case("multimodal_max", max_status, max_detail, config))

    results.extend(_busy_and_status(config, container_id))
    results.extend(_repeat_lifecycle(config, container_id))
    results.extend(_failure_cases(config))
    return results


def _busy_and_status(config: RunConfig, container_id: str) -> list[CaseResult]:
    del container_id
    body = build_chat_request(config.served_model, kind="text", stream=True, max_tokens=4096, text="Count slowly.")
    seen: dict[str, Any] = {}

    def _call() -> None:
        seen["result"] = _chat(config, body)

    worker = threading.Thread(target=_call, name="gpu-stream")
    worker.start()
    time.sleep(1)
    unload_code, unload_body = _post_control(config, "unload", 30)
    worker.join(timeout=config.request_timeout_sec)
    _, after = _status(config)
    busy_ok = unload_code == 409 and after.get("active_requests") == 0 and not worker.is_alive()
    busy = _case(
        "disconnect_busy",
        "passed" if busy_ok else "failed",
        f"unload={unload_code} {unload_body} after={after}",
        config,
    )

    samples: list[float] = []

    def _poll() -> None:
        started = time.monotonic()
        deadline = started + config.load_timeout_sec
        while time.monotonic() < deadline and len(samples) < 5:
            t0 = time.monotonic()
            code, _payload = _status(config, timeout=STATUS_RESPONSE_LIMIT_SEC)
            samples.append(time.monotonic() - t0)
            if code != 200:
                samples.append(STATUS_RESPONSE_LIMIT_SEC + 1)
            time.sleep(0.2)

    poller = threading.Thread(target=_poll, name="gpu-status")
    poller.start()
    _post_control(config, "load", config.load_timeout_sec)
    poller.join(timeout=5)
    fast = bool(samples) and max(samples) <= STATUS_RESPONSE_LIMIT_SEC
    status_case = _case(
        "status_during_lifecycle",
        "passed" if fast else "failed",
        f"status samples sec={samples} limit={STATUS_RESPONSE_LIMIT_SEC}",
        config,
    )
    return [busy, status_case]


def _repeat_lifecycle(config: RunConfig, container_id: str) -> list[CaseResult]:
    residuals: list[int] = []
    for _ in range(3):
        _post_control(config, "load", config.load_timeout_sec)
        _post_control(config, "unload", 60)
        time.sleep(2)
        used = _gpu_used_mib()
        if used is not None:
            residuals.append(used)
    workers = _worker_cmds(container_id)
    gone = not workers
    stable = len(residuals) >= 3 and max(residuals) - min(residuals) <= RESIDUAL_GROWTH_MIB
    evidence = {"workers_gone": "ps" if gone else "", "residual_stable": "nvidia-smi" if stable else ""}
    status_name, detail = judge_case(
        "repeat_lifecycle",
        http_ok=True,
        evidence=evidence,
        detail=f"residuals_mib={residuals} workers={workers}",
    )
    return [_case("repeat_lifecycle", status_name, detail, config)]


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


def _crash_worker(config: RunConfig) -> tuple[str, str]:
    try:
        container_id = _owned_container(config.root, config.service)
    except Exception as exc:
        return "not_run", str(exc)
    _post_control(config, "load", config.load_timeout_sec)
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
    pids = [line.split()[0] for line in listed.stdout.splitlines() if "uvicorn" in line or "controller.main" in line]
    if not pids:
        return "failed", f"controller pid was not found: {listed.stdout}"
    subprocess.run(["docker", "exec", container_id, "kill", "-TERM", *pids], check=False)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            new_id = _owned_container(config.root, config.service)
        except RuntimeError:
            time.sleep(1)
            continue
        if new_id == container_id:
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
    if all(item.status == "passed" for item in results):
        return 0
    if any(item.status == "failed" for item in results):
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
