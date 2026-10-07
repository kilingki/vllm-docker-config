import asyncio

import httpx

from controller.config import Settings
from controller.lifecycle import WorkerFailure
from controller.main import create_app


class FakeStream:
    def __init__(self, body: bytes = b"{}", status: int = 200, release: asyncio.Event | None = None):
        self.status_code = status
        self.headers = {"content-type": "application/json"}
        self._body = body
        self.release = release
        self.drained = asyncio.Event()

    async def aiter_raw(self):
        if self.release is not None:
            await self.release.wait()
        yield self._body
        self.drained.set()

    async def aclose(self) -> None:
        return None


class FakeUpstream:
    def __init__(self) -> None:
        self.bodies: list[bytes] = []
        self.hold: asyncio.Event | None = None
        self.fail: BaseException | None = None
        self.opened = asyncio.Event()
        self.calls = 0

    async def open(self, method, path, headers, content, params):
        self.calls += 1
        self.bodies.append(b"" if content is None else bytes(content))
        self.opened.set()
        if self.fail is not None:
            raise self.fail
        if path.strip("/") == "models":
            return FakeStream(b'{"object":"list","data":[]}')
        return FakeStream(release=self.hold)


class FakeWorker:
    def __init__(self) -> None:
        self.alive = False
        self.start_gate: asyncio.Event | None = None
        self.stop_gate: asyncio.Event | None = None
        self.fail_start: BaseException | None = None
        self.stop_failure: BaseException | None = None
        self.port_busy = False
        self.entered_start = asyncio.Event()
        self.entered_stop = asyncio.Event()
        self.start_count = 0
        self.stop_count = 0
        self.probes: list[str] = []
        self.ownership_value = "resident"
        self.fatal: str | None = None
        self._handler = None

    async def start(self, generation: int) -> None:
        self.start_count += 1
        self.entered_start.set()
        if self.port_busy:
            raise WorkerFailure("port already has a listener", "not_resident")
        if self.start_gate is not None:
            await self.start_gate.wait()
        if self.fail_start is not None:
            raise self.fail_start
        self.alive = True
        self.ownership_value = "resident"

    async def stop(self) -> str:
        self.stop_count += 1
        self.entered_stop.set()
        if self.stop_gate is not None:
            await self.stop_gate.wait()
        if self.stop_failure is not None:
            raise self.stop_failure
        self.alive = False
        self.ownership_value = "all_dead"
        return "not_resident"

    def is_alive(self) -> bool:
        return self.alive

    def ownership(self) -> str:
        return self.ownership_value

    async def probe(self, timeout: float) -> str:
        if self.probes:
            return self.probes.pop(0)
        return "ok"

    def fatal_error(self) -> str | None:
        return self.fatal

    def arm_watch(self, handler) -> None:
        self._handler = handler


def settings(**overrides) -> Settings:
    values = dict(
        load_timeout_sec=2,
        unload_timeout_sec=2,
        kill_grace_sec=0.2,
        model_path="/models/qwen3.8-27b",
        served_model_name="qwen3.8-27b",
        kv_cache_dtype="",
        prefix_caching=False,
        reasoning_parser="",
        limit_mm_per_prompt="",
        mm_processor_kwargs="",
        health_probe_interval_sec=0.02,
        health_probe_timeout_sec=0.05,
        health_probe_failures=3,
        require_explicit_output_limit=True,
    )
    values.update(overrides)
    return Settings(**values)


def build(worker: FakeWorker, upstream: FakeUpstream | None = None, **overrides):
    app = create_app(settings(**overrides), worker, upstream or FakeUpstream())
    return app, app.state.lifecycle


def client_for(app):
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://runtime")


CHAT = {"max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]}
