#!/usr/bin/env python3
"""Qwen3.8 LLM reasoning levels and VLM image checks against the running runtime.

Talks to the controller at BASE_URL. Does not load weights inside this process.
Leaves the model loaded when the run finishes.
"""

import base64
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = os.environ.get("BASE_URL", "http://127.0.0.1:8010")
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen3.8-27b")
REQUEST_TIMEOUT_SEC = float(os.environ.get("REQUEST_TIMEOUT_SEC", "600"))
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "768"))
VLM_MAX_TOKENS = int(os.environ.get("VLM_MAX_TOKENS", "256"))

LLM_PROMPT = "What is 17 times 23? Reply with the number only in the final answer."
VLM_PROMPT = "Describe this image in one sentence. Include the kind of animal and its color."

LLM_CASES = (
    ("llm-off", {"enable_thinking": False}),
    ("llm-low", {"reasoning_effort": "low"}),
    ("llm-medium", {"reasoning_effort": "medium"}),
    ("llm-xhigh", {"reasoning_effort": "xhigh"}),
)

IMAGE_CASES = (
    ("vlm-cat-512", ROOT / "tests" / "test-cat-512.jpg"),
    ("vlm-cat", ROOT / "tests" / "test-cat.jpg"),
)


def request(method: str, path: str, body: dict | None = None, timeout: float = REQUEST_TIMEOUT_SEC) -> tuple[int, dict]:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data, method=method)
    if data is not None:
        req.add_header("content-type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = response.read()
            return response.status, json.loads(payload) if payload else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode()
        try:
            parsed = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            parsed = {"raw": raw}
        return exc.code, parsed


def ensure_ready(run_dir: Path) -> dict:
    status, body = request("GET", "/control/status", timeout=10)
    (run_dir / "00-status-before.json").write_text(
        json.dumps({"http_status": status, "body": body}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if status == 200 and body.get("state") == "ready":
        return body
    started = time.perf_counter()
    load_status, load_body = request("POST", "/control/load", {})
    record = {
        "http_status": load_status,
        "body": load_body,
        "e2e_sec": round(time.perf_counter() - started, 3),
    }
    (run_dir / "01-load.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if load_status != 200 or load_body.get("state") != "ready":
        raise SystemExit(f"load failed: HTTP {load_status} {load_body}")
    return load_body


def message_of(body: dict) -> dict:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return {}
    message = choices[0].get("message")
    return message if isinstance(message, dict) else {}


def text_field(message: dict, key: str) -> str:
    value = message.get(key)
    return value if isinstance(value, str) else ""


def reasoning_text(message: dict) -> str:
    """Prefer llama.cpp's field, then this runtime's reasoning parser field."""
    for key in ("reasoning_content", "reasoning"):
        value = message.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def finish_reason(body: dict) -> str | None:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    reason = choices[0].get("finish_reason")
    return reason if isinstance(reason, str) else None


def num(mapping: dict, key: str):
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def run_chat(name: str, payload: dict, meta: dict) -> dict:
    started = time.perf_counter()
    status, body = request("POST", "/v1/chat/completions", payload)
    elapsed = time.perf_counter() - started
    message = message_of(body)
    usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
    timings = body.get("timings") if isinstance(body.get("timings"), dict) else {}
    content = text_field(message, "content")
    reasoning = reasoning_text(message)
    return {
        "name": name,
        "http_status": status,
        "e2e_sec": round(elapsed, 3),
        "finish_reason": finish_reason(body),
        "content": content,
        "reasoning_content": reasoning,
        "content_chars": len(content),
        "reasoning_chars": len(reasoning),
        "usage": {
            "prompt_tokens": num(usage, "prompt_tokens"),
            "completion_tokens": num(usage, "completion_tokens"),
            "total_tokens": num(usage, "total_tokens"),
        },
        "timings": {
            "prompt_n": num(timings, "prompt_n"),
            "prompt_ms": num(timings, "prompt_ms"),
            "prompt_per_second": num(timings, "prompt_per_second"),
            "predicted_n": num(timings, "predicted_n"),
            "predicted_ms": num(timings, "predicted_ms"),
            "predicted_per_second": num(timings, "predicted_per_second"),
        },
        "request": meta,
        "error": None if status == 200 else body,
    }


def llm_case(name: str, kwargs: dict) -> dict:
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": LLM_PROMPT}],
        "max_tokens": LLM_MAX_TOKENS,
        "temperature": 0,
        "chat_template_kwargs": kwargs,
    }
    meta = {
        "kind": "llm",
        "prompt": LLM_PROMPT,
        "max_tokens": LLM_MAX_TOKENS,
        "temperature": 0,
        "chat_template_kwargs": kwargs,
    }
    return run_chat(name, payload, meta)


def vlm_case(name: str, image_path: Path) -> dict:
    raw = image_path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VLM_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }
        ],
        "max_tokens": VLM_MAX_TOKENS,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    meta = {
        "kind": "vlm",
        "prompt": VLM_PROMPT,
        "max_tokens": VLM_MAX_TOKENS,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        "image_file": image_path.name,
        "image_bytes": len(raw),
    }
    return run_chat(name, payload, meta)


def fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def write_index(run_dir: Path, rows: list[dict]) -> None:
    lines = [
        "# Qwen3.8 추론 테스트",
        "",
        f"- base: `{BASE}`",
        f"- model: `{MODEL_NAME}`",
        f"- LLM prompt: {LLM_PROMPT}",
        f"- VLM prompt: {VLM_PROMPT}",
        "",
        "| case | HTTP | e2e_s | prompt_tok | completion_tok | prompt_tok/s | gen_tok/s | prompt_ms | predicted_ms | finish |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        usage = row["usage"]
        timings = row["timings"]
        lines.append(
            "| {name} | {http} | {e2e} | {pt} | {ct} | {pps} | {gps} | {pms} | {dms} | {finish} |".format(
                name=row["name"],
                http=row["http_status"],
                e2e=fmt(row["e2e_sec"]),
                pt=fmt(usage.get("prompt_tokens")),
                ct=fmt(usage.get("completion_tokens")),
                pps=fmt(timings.get("prompt_per_second")),
                gps=fmt(timings.get("predicted_per_second")),
                pms=fmt(timings.get("prompt_ms")),
                dms=fmt(timings.get("predicted_ms")),
                finish=row.get("finish_reason") or "",
            )
        )
    lines.append("")
    for row in rows:
        lines.extend([f"## {row['name']}", ""])
        image_file = row["request"].get("image_file")
        if image_file:
            lines.append(f"- image: `{image_file}`")
        lines.extend(
            [
                f"- kwargs: `{json.dumps(row['request'].get('chat_template_kwargs'), ensure_ascii=False)}`",
                "",
                "### content",
                "",
                row["content"] or "(empty)",
                "",
                "### reasoning_content",
                "",
                row["reasoning_content"] or "(empty)",
                "",
            ]
        )
        if row.get("error") is not None:
            lines.extend(
                [
                    "### error",
                    "",
                    "```json",
                    json.dumps(row["error"], ensure_ascii=False, indent=2),
                    "```",
                    "",
                ]
            )
    (run_dir / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")


def save_case(run_dir: Path, row: dict) -> None:
    path = run_dir / f"{row['name']}.json"
    path.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(__file__).resolve().parent / "outputs" / f"{stamp}-qwen-infer"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"results: {run_dir}", flush=True)

    for _, image_path in IMAGE_CASES:
        if not image_path.is_file():
            raise SystemExit(f"missing image: {image_path}")

    ensure_ready(run_dir)
    rows: list[dict] = []
    failed = False

    for name, kwargs in LLM_CASES:
        print(f"== {name}", flush=True)
        row = llm_case(name, kwargs)
        rows.append(row)
        save_case(run_dir, row)
        write_index(run_dir, rows)
        print(
            f"   HTTP {row['http_status']} e2e {row['e2e_sec']}s "
            f"gen {row['timings'].get('predicted_per_second')} tok/s "
            f"finish {row['finish_reason']}",
            flush=True,
        )
        if row["http_status"] != 200:
            failed = True

    for name, image_path in IMAGE_CASES:
        print(f"== {name}", flush=True)
        row = vlm_case(name, image_path)
        rows.append(row)
        save_case(run_dir, row)
        write_index(run_dir, rows)
        print(
            f"   HTTP {row['http_status']} e2e {row['e2e_sec']}s "
            f"prompt {row['timings'].get('prompt_ms')} ms "
            f"finish {row['finish_reason']}",
            flush=True,
        )
        if row["http_status"] != 200 or not row["content"].strip():
            failed = True

    print(f"results: {run_dir / 'INDEX.md'}", flush=True)
    if failed:
        raise SystemExit("one or more cases failed")


if __name__ == "__main__":
    main()
