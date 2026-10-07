"""OpenAI proxy with a fixed rejection order and upstream-owned request lifetime.

Body limits are applied before readiness. Output checks, the admission limit,
and the active increment are committed under one lifecycle lock. A disconnected
client does not cancel upstream work or decrement active early.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from controller.lifecycle import Admit, Lifecycle, Ticket

logger = logging.getLogger("controller.proxy")

HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}

GENERATION_PATHS = {"chat/completions", "completions"}


class UpstreamError(Exception):
    pass


class ChunkQueue:
    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        self._items: deque[tuple[Any, int]] = deque()
        self._bytes = 0
        self._discarding = False

    @property
    def buffered_bytes(self) -> int:
        return self._bytes

    def discard(self) -> None:
        self._discarding = True
        self._items.clear()
        self._bytes = 0

    async def put(self, item: Any, size: int = 0) -> None:
        while True:
            if self._discarding:
                return
            blocked = bool(self._items) and size > 0 and self._bytes + size > self.max_bytes
            if not blocked:
                self._items.append((item, size))
                self._bytes += size
                return
            await asyncio.sleep(0.01)

    async def get(self) -> Any:
        while not self._items and not self._discarding:
            await asyncio.sleep(0.01)
        if not self._items:
            return None
        item, size = self._items.popleft()
        self._bytes -= size
        return item


def evaluate_output_limits(body: bytes, *, limit: int, require_explicit: bool) -> str | None:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return "invalid json"
    if not isinstance(payload, dict):
        return "invalid json"
    invalid = False
    values: list[int] = []
    for name in ("max_tokens", "max_completion_tokens"):
        if name not in payload:
            continue
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            invalid = True
            continue
        values.append(value)
    if invalid:
        return "output token limits must be positive integers"
    if len(values) == 2 and values[0] != values[1]:
        return "conflicting output token limits"
    if any(value > limit for value in values):
        return f"output token limit exceeds {limit}"
    if not values and require_explicit:
        return "an explicit output token limit is required"
    return None


def _counts_as_inference(method: str, path: str) -> bool:
    return not (method.upper() == "GET" and path.strip("/") == "models")


def _is_generation(method: str, path: str) -> bool:
    return method.upper() == "POST" and path.strip("/") in GENERATION_PATHS


async def read_limited_body(request: Request, limit: int) -> tuple[str, bytes]:
    header = request.headers.get("content-length")
    if header is not None:
        try:
            if int(header) > limit:
                return "too_large", b""
        except ValueError:
            pass
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            return "too_large", b""
        chunks.append(chunk)
    return "ok", b"".join(chunks)


def _filtered_headers(headers: Any) -> dict[str, str]:
    return {key: value for key, value in headers.items() if key.lower() not in HOP_BY_HOP}


class HttpxUpstream:
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(None))

    async def aclose(self) -> None:
        await self._client.aclose()

    async def open(
        self,
        method: str,
        path: str,
        headers: dict[str, str],
        content: bytes | None,
        params: Any,
    ) -> Any:
        target = f"{self._base_url}/v1/{path}" if path else f"{self._base_url}/v1"
        request = self._client.build_request(
            method,
            target,
            headers=headers,
            content=content,
            params=params,
        )
        response = await self._client.send(request, stream=True)
        return _HttpxStream(response)


class _HttpxStream:
    def __init__(self, response: httpx.Response) -> None:
        self.status_code = response.status_code
        self.headers = response.headers
        self._response = response

    async def aiter_raw(self) -> AsyncIterator[bytes]:
        async for chunk in self._response.aiter_raw():
            yield chunk

    async def aclose(self) -> None:
        await self._response.aclose()


def _detail(status_code: int, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": detail})


async def proxy_openai(
    request: Request,
    path: str,
    lifecycle: Lifecycle,
    upstream: Any,
) -> JSONResponse | StreamingResponse:
    kind, body = await read_limited_body(request, lifecycle.settings.max_body_bytes)
    if kind == "too_large":
        return _detail(413, "request body too large")

    counted = _counts_as_inference(request.method, path)
    output_error = None
    if _is_generation(request.method, path):
        output_error = evaluate_output_limits(
            body,
            limit=lifecycle.settings.max_output_tokens,
            require_explicit=lifecycle.settings.require_explicit_output_limit,
        )

    queue = ChunkQueue(lifecycle.settings.stream_buffer_max_bytes)
    disconnected = asyncio.Event()

    async def run(ticket: Ticket | None) -> None:
        stream = None
        try:
            stream = await upstream.open(
                request.method,
                path,
                _filtered_headers(request.headers),
                body if body else None,
                request.query_params,
            )
            if disconnected.is_set():
                async for _chunk in stream.aiter_raw():
                    pass
            else:
                await queue.put(
                    (
                        "head",
                        stream.status_code,
                        _filtered_headers(stream.headers),
                        stream.headers.get("content-type"),
                    )
                )
                async for chunk in stream.aiter_raw():
                    if disconnected.is_set() or queue._discarding:
                        continue
                    await queue.put(("chunk", chunk), len(chunk))
                await queue.put(("end",))
            if ticket is not None:
                await lifecycle.release(ticket)
        except Exception:
            logger.exception("upstream request failed")
            if ticket is not None:
                await lifecycle.mark_unconfirmed(ticket)
            if not disconnected.is_set():
                await queue.put(("error",))
        finally:
            if stream is not None:
                await stream.aclose()

    decision: Admit = await lifecycle.admit(
        counted=counted,
        output_error=output_error,
        start_task=lambda ticket: asyncio.create_task(run(ticket), name="vllm-proxy"),
    )
    if not decision.ok:
        assert decision.detail is not None
        return _detail(decision.status_code, decision.detail)

    try:
        head = await queue.get()
    except asyncio.CancelledError:
        disconnected.set()
        queue.discard()
        raise
    if head is None or head[0] == "error":
        return _detail(502, "upstream connection failed")

    _kind, status_code, response_headers, media_type = head

    async def stream_body() -> AsyncIterator[bytes]:
        try:
            while True:
                item = await queue.get()
                if item is None or item[0] in {"end", "error"}:
                    return
                yield item[1]
        finally:
            disconnected.set()
            queue.discard()

    return StreamingResponse(
        stream_body(),
        status_code=status_code,
        headers=response_headers,
        media_type=media_type,
    )
