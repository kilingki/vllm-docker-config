import asyncio
import os
import signal
import sys

import pytest

from controller.lifecycle import WorkerFailure, terminate_owned_group
from controller.lifecycle import _group_alive
from controller.proxy import ChunkQueue, evaluate_output_limits, proxy_openai, read_limited_body
from tests.fakes import CHAT, FakeUpstream, FakeWorker, build, client_for

pytestmark = pytest.mark.cpu_control


def run(coro):
    return asyncio.run(coro)


async def _ready_app(**overrides):
    worker = FakeWorker()
    upstream = FakeUpstream()
    app, lifecycle = build(worker, upstream, **overrides)
    async with client_for(app) as client:
        loaded = await client.post("/control/load")
        assert loaded.status_code == 200, loaded.text
        assert loaded.json()["state"] == "ready"
        return app, lifecycle, worker, upstream, client


class _Body:
    def __init__(self, headers, chunks, pulled):
        self.headers = headers
        self._chunks = chunks
        self._pulled = pulled

    async def stream(self):
        for chunk in self._chunks:
            self._pulled.append(chunk)
            yield chunk


def test_initial_unloaded_health_and_inference_503():
    async def scenario():
        worker = FakeWorker()
        app, lifecycle = build(worker, FakeUpstream())
        async with client_for(app) as client:
            health = await client.get("/health")
            assert health.status_code == 200
            assert health.json() == {"controller": "ok"}
            status = await client.get("/control/status")
            assert status.status_code == 200
            body = status.json()
            assert list(body) == ["state", "residency", "active_requests", "last_error"]
            assert body == {
                "state": "unloaded",
                "residency": "not_resident",
                "active_requests": 0,
                "last_error": None,
            }
            rejected = await client.post("/v1/chat/completions", json=CHAT)
            assert rejected.status_code == 503
            assert rejected.json() == {"detail": "model is not ready"}
            assert worker.start_count == 0
            assert lifecycle._active == 0

    run(scenario())


def test_load_unload_success_noop_shared_and_conflict():
    async def scenario():
        worker = FakeWorker()
        worker.start_gate = asyncio.Event()
        app, lifecycle = build(worker, FakeUpstream())
        async with client_for(app) as client:
            first = asyncio.create_task(lifecycle.load())
            await worker.entered_start.wait()
            second = asyncio.create_task(lifecycle.load())
            await asyncio.sleep(0.05)
            assert worker.start_count == 1
            assert not second.done()
            progress = await client.get("/control/status")
            assert progress.json()["state"] == "loading"
            conflict = await client.post("/control/unload")
            assert conflict.status_code == 409
            assert conflict.json()["error"]["code"] == "LIFECYCLE_CONFLICT"
            worker.start_gate.set()
            assert (await first)["state"] == "ready"
            assert (await second)["state"] == "ready"
            again = await client.post("/control/load")
            assert again.status_code == 200
            assert again.json()["state"] == "ready"
            assert worker.start_count == 1
            worker.stop_gate = asyncio.Event()
            unloading = asyncio.create_task(client.post("/control/unload"))
            await worker.entered_stop.wait()
            during = await client.get("/control/status")
            assert during.json()["state"] == "unloading"
            opposite = await client.post("/control/load")
            assert opposite.status_code == 409
            assert opposite.json()["error"]["code"] == "LIFECYCLE_CONFLICT"
            worker.stop_gate.set()
            unloaded = await unloading
            assert unloaded.status_code == 200
            assert unloaded.json()["state"] == "unloaded"
            noop = await client.post("/control/unload")
            assert noop.status_code == 200
            assert noop.json()["residency"] == "not_resident"
            assert worker.stop_count == 1

    run(scenario())


def test_control_disconnect_keeps_task_and_status_responds():
    async def scenario():
        worker = FakeWorker()
        worker.start_gate = asyncio.Event()
        _app, lifecycle = build(worker, FakeUpstream())
        task = asyncio.create_task(lifecycle.load())
        await worker.entered_start.wait()
        status = await lifecycle.status()
        assert status["state"] == "loading"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert worker.start_count == 1
        worker.start_gate.set()
        for _ in range(50):
            status = await lifecycle.status()
            if status["state"] == "ready":
                break
            await asyncio.sleep(0.01)
        assert status["state"] == "ready"
        assert status["active_requests"] == 0

    run(scenario())


def test_operation_result_stays_with_its_waiter():
    async def scenario():
        worker = FakeWorker()
        worker.start_gate = asyncio.Event()
        _app, lifecycle = build(worker, FakeUpstream())
        reached = asyncio.Event()
        release = asyncio.Event()

        async def paused():
            snapshot = await lifecycle.load()
            reached.set()
            await release.wait()
            return snapshot

        task = asyncio.create_task(paused())
        await worker.entered_start.wait()
        worker.start_gate.set()
        await reached.wait()
        unloaded = await lifecycle.unload()
        assert unloaded["state"] == "unloaded"
        release.set()
        assert (await task)["state"] == "ready"

    run(scenario())


def test_admit_unload_races_and_busy():
    async def scenario():
        worker = FakeWorker()
        upstream = FakeUpstream()
        upstream.hold = asyncio.Event()
        app, lifecycle = build(worker, upstream)
        async with client_for(app) as client:
            assert (await client.post("/control/load")).status_code == 200
            held = asyncio.create_task(client.post("/v1/chat/completions", json=CHAT))
            await upstream.opened.wait()
            await asyncio.sleep(0.05)
            busy = await client.post("/control/unload")
            assert busy.status_code == 409
            assert busy.json()["error"]["code"] == "BUSY"
            assert (await client.get("/control/status")).json()["active_requests"] == 1
            upstream.hold.set()
            assert (await held).status_code == 200
            assert (await client.get("/control/status")).json()["active_requests"] == 0
            worker.stop_gate = asyncio.Event()
            unloading = asyncio.create_task(client.post("/control/unload"))
            await worker.entered_stop.wait()
            rejected = await client.post("/v1/chat/completions", json=CHAT)
            assert rejected.status_code == 503
            assert rejected.json()["detail"] == "model is not ready"
            assert (await client.get("/control/status")).json()["active_requests"] == 0
            worker.stop_gate.set()
            assert (await unloading).status_code == 200

            failed = FakeWorker()
            failed.alive = False
            failed_app, failed_life = build(failed, FakeUpstream())
            failed_life._state = "unloaded"
            failed_life._residency = "not_resident"
            failed_life._active = 1
            with pytest.raises(Exception) as caught:
                await failed_life.unload()
            assert caught.value.code == "BUSY"

    run(scenario())


def test_external_limit_models_not_counted_and_not_ready_body():
    async def scenario():
        worker = FakeWorker()
        upstream = FakeUpstream()
        upstream.hold = asyncio.Event()
        app, lifecycle = build(worker, upstream, max_active_requests=4)
        async with client_for(app) as client:
            rejected = await client.post("/v1/chat/completions", json=CHAT)
            assert rejected.status_code == 503
            assert rejected.json() == {"detail": "model is not ready"}
            assert (await client.post("/control/load")).status_code == 200
            tasks = [
                asyncio.create_task(client.post("/v1/chat/completions", json=CHAT))
                for _ in range(4)
            ]
            for _ in range(50):
                if lifecycle._active == 4:
                    break
                await asyncio.sleep(0.01)
            assert lifecycle._active == 4
            models = await client.get("/v1/models")
            assert models.status_code == 200
            assert lifecycle._active == 4
            extra = await client.post("/v1/chat/completions", json=CHAT)
            assert extra.status_code == 429
            assert extra.json() == {"detail": "too many active requests"}
            assert lifecycle._active == 4
            upstream.hold.set()
            for task in tasks:
                assert (await task).status_code == 200
            assert (await client.get("/control/status")).json()["active_requests"] == 0

    run(scenario())


def test_rejection_priority_and_atomic_active():
    async def scenario():
        worker = FakeWorker()
        upstream = FakeUpstream()
        app, lifecycle = build(worker, upstream, max_body_bytes=32, max_active_requests=1)
        async with client_for(app) as client:
            huge = await client.post("/v1/chat/completions", content=b"x" * 33)
            assert huge.status_code == 413
            assert huge.json() == {"detail": "request body too large"}
            assert lifecycle._active == 0
            early = await client.post(
                "/v1/chat/completions",
                content=b'{"max_tokens": true}',
            )
            assert early.status_code == 503
            assert early.json()["detail"] == "model is not ready"
            assert (await client.post("/control/load")).status_code == 200
            upstream.hold = asyncio.Event()
            held = asyncio.create_task(
                client.post("/v1/chat/completions", content=b'{"max_tokens": 8}')
            )
            await upstream.opened.wait()
            typed = await client.post(
                "/v1/chat/completions",
                content=b'{"max_tokens": true}',
            )
            assert typed.status_code == 400
            assert typed.json()["detail"] == "output token limits must be positive integers"
            limited = await client.post(
                "/v1/chat/completions",
                content=b'{"max_tokens": 8}',
            )
            assert limited.status_code == 429
            assert lifecycle._active == 1
            upstream.hold.set()
            assert (await held).status_code == 200

        entered = asyncio.Event()
        release = asyncio.Event()

        class Blocking:
            headers: dict[str, str] = {}
            method = "POST"
            query_params = {}

            async def stream(self):
                entered.set()
                await release.wait()
                yield b'{"max_tokens": 8}'

        worker2 = FakeWorker()
        upstream2 = FakeUpstream()
        app2, lifecycle2 = build(worker2, upstream2)
        async with client_for(app2) as client:
            assert (await client.post("/control/load")).status_code == 200
        proxy_task = asyncio.create_task(
            proxy_openai(Blocking(), "chat/completions", lifecycle2, upstream2)
        )
        await entered.wait()
        assert (await lifecycle2.unload())["state"] == "unloaded"
        release.set()
        response = await proxy_task
        assert response.status_code == 503
        assert lifecycle2._active == 0
        assert upstream2.calls == 0

    run(scenario())


@pytest.mark.parametrize(
    ("raw", "status", "detail"),
    [
        (b'{"max_tokens": 16}', 200, None),
        (b'{"max_completion_tokens": 16}', 200, None),
        (b'{"max_tokens": 16, "max_completion_tokens": 16}', 200, None),
        (b'{"max_tokens": 16, "max_completion_tokens": 32}', 400, "conflicting output token limits"),
        (b'{"max_tokens": null}', 400, "output token limits must be positive integers"),
        (b'{"max_tokens": true}', 400, "output token limits must be positive integers"),
        (b'{"max_tokens": false}', 400, "output token limits must be positive integers"),
        (b'{"max_tokens": 1.0}', 400, "output token limits must be positive integers"),
        (b'{"max_tokens": 0}', 400, "output token limits must be positive integers"),
        (b'{"max_tokens": -3}', 400, "output token limits must be positive integers"),
        (b'{"max_tokens": "16"}', 400, "output token limits must be positive integers"),
        (b'{"max_tokens": 4097}', 400, "output token limit exceeds 4096"),
        (b'{"messages": []}', 400, "an explicit output token limit is required"),
        (b'{"max_tokens": true, "max_completion_tokens": 99999}', 400, "output token limits must be positive integers"),
        (b'{"max_tokens": 1, "max_completion_tokens": 5000}', 400, "conflicting output token limits"),
        (b'{"max_tokens": 5000, "max_completion_tokens": 5000}', 400, "output token limit exceeds 4096"),
    ],
)
def test_output_limits_and_preserved_bytes(raw, status, detail):
    async def scenario():
        for path in ("chat/completions", "completions"):
            worker = FakeWorker()
            upstream = FakeUpstream()
            app, _lifecycle = build(worker, upstream)
            async with client_for(app) as client:
                assert (await client.post("/control/load")).status_code == 200
                response = await client.post(f"/v1/{path}", content=raw)
                assert response.status_code == status, response.text
                if detail is None:
                    assert upstream.bodies[-1] == raw
                else:
                    assert response.json()["detail"] == detail
                    assert raw not in upstream.bodies

    run(scenario())


def test_omitted_limit_can_pass_without_rewriting_body():
    async def scenario():
        worker = FakeWorker()
        upstream = FakeUpstream()
        app, _lifecycle = build(worker, upstream, require_explicit_output_limit=False)
        raw = b'{"messages":[{"role":"user","content":"hi"}]}'
        async with client_for(app) as client:
            assert (await client.post("/control/load")).status_code == 200
            response = await client.post("/v1/chat/completions", content=raw)
            assert response.status_code == 200
            assert upstream.bodies[-1] == raw

    run(scenario())


def test_thinking_fields_are_forwarded_unchanged():
    async def scenario():
        worker = FakeWorker()
        upstream = FakeUpstream()
        app, _lifecycle = build(worker, FakeUpstream())
        app, _lifecycle = build(worker, upstream)
        raw = (
            b'{"max_tokens":32,"messages":[{"role":"user","content":"hi"}],'
            b'"chat_template_kwargs":{"enable_thinking":false,"reasoning_effort":"low"},'
            b'"reasoning_effort":"high"}'
        )
        async with client_for(app) as client:
            assert (await client.post("/control/load")).status_code == 200
            response = await client.post("/v1/chat/completions", content=raw)
            assert response.status_code == 200
            assert upstream.bodies[-1] == raw

    run(scenario())


def test_other_post_counts_active_without_output_check_and_chunked_limit():
    async def scenario():
        worker = FakeWorker()
        upstream = FakeUpstream()
        upstream.hold = asyncio.Event()
        app, lifecycle = build(worker, upstream, max_body_bytes=128)
        async with client_for(app) as client:
            assert (await client.post("/control/load")).status_code == 200
            held = asyncio.create_task(
                client.post("/v1/embeddings", content=b'{"max_tokens": 99999}')
            )
            await upstream.opened.wait()
            assert lifecycle._active == 1
            upstream.hold.set()
            assert (await held).status_code == 200
        pulled: list[bytes] = []
        request = _Body({}, [b"a" * 10, b"b" * 10, b"tail"], pulled)
        kind, _body = await read_limited_body(request, 16)
        assert kind == "too_large"
        assert b"tail" not in pulled
        header_pulled: list[bytes] = []
        header = _Body({"content-length": "33554433"}, [b"unread"], header_pulled)
        kind, _body = await read_limited_body(header, 33554432)
        assert kind == "too_large"
        assert header_pulled == []

    run(scenario())


def test_bounded_downstream_buffer():
    async def scenario():
        queue = ChunkQueue(8)
        await queue.put(("chunk", b"12345678"), 8)
        started = asyncio.Event()

        async def overflow():
            started.set()
            await queue.put(("chunk", b"x"), 1)

        task = asyncio.create_task(overflow())
        await started.wait()
        await asyncio.sleep(0.05)
        assert not task.done()
        assert queue.buffered_bytes == 8
        await queue.get()
        await task
        queue.discard()
        await queue.put(("chunk", b"yyyyyyyyyyyy"), 12)
        assert queue.buffered_bytes == 0

    run(scenario())


def test_health_failures_busy_timeout_and_worker_death():
    async def scenario():
        worker = FakeWorker()
        upstream = FakeUpstream()
        app, lifecycle = build(worker, upstream)
        async with client_for(app) as client:
            assert (await client.post("/control/load")).status_code == 200
            worker.probes = ["http_error", "http_error", "http_error"]
            failed = None
            for _ in range(100):
                failed = await lifecycle.status()
                if failed["state"] == "failed":
                    break
                await asyncio.sleep(0.02)
            assert failed["state"] == "failed"
            assert failed["residency"] == "resident"
            assert failed["active_requests"] == 0
            assert failed["last_error"]["code"] == "WORKER_UNHEALTHY"
            blocked = await client.post("/v1/chat/completions", json=CHAT)
            assert blocked.status_code == 503

        worker = FakeWorker()
        upstream = FakeUpstream()
        upstream.hold = asyncio.Event()
        app, lifecycle = build(worker, upstream)
        async with client_for(app) as client:
            assert (await client.post("/control/load")).status_code == 200
            held = asyncio.create_task(client.post("/v1/chat/completions", json=CHAT))
            await upstream.opened.wait()
            worker.probes = ["read_timeout"] * 40
            await asyncio.sleep(0.12)
            assert (await lifecycle.status())["state"] == "ready"
            assert lifecycle.busy_probe_timeouts >= 1
            upstream.hold.set()
            assert (await held).status_code == 200
            for _ in range(100):
                if (await lifecycle.status())["state"] == "failed":
                    break
                await asyncio.sleep(0.02)
            assert (await lifecycle.status())["state"] == "failed"

        worker = FakeWorker()
        upstream = FakeUpstream()
        upstream.hold = asyncio.Event()
        app, lifecycle = build(worker, upstream)
        async with client_for(app) as client:
            assert (await client.post("/control/load")).status_code == 200
            held = asyncio.create_task(client.post("/v1/chat/completions", json=CHAT))
            await upstream.opened.wait()
            await lifecycle.note_worker_exit(lifecycle._generation, "resident")
            death = await lifecycle.status()
            assert death["state"] == "failed"
            assert death["residency"] == "resident"
            assert death["active_requests"] == 1
            busy = await client.post("/control/unload")
            assert busy.status_code == 409
            assert busy.json()["error"]["code"] == "BUSY"
            upstream.hold.set()
            await held
            worker.fatal = "engine fatal"
            worker.alive = True
            worker.ownership_value = "resident"
            lifecycle._state = "ready"
            lifecycle._residency = "resident"
            lifecycle._last_error = None
            lifecycle._start_monitor(lifecycle._generation)
            for _ in range(100):
                if (await lifecycle.status())["state"] == "failed":
                    break
                await asyncio.sleep(0.02)
            fatal = await lifecycle.status()
            assert fatal["state"] == "failed"
            assert fatal["active_requests"] == 0

    run(scenario())


def test_stream_and_nonstream_disconnect_hold_active_until_upstream_done():
    async def once(chunks: bytes):
        worker = FakeWorker()
        upstream = FakeUpstream()
        upstream.hold = asyncio.Event()
        app, lifecycle = build(worker, upstream)
        async with client_for(app) as client:
            assert (await client.post("/control/load")).status_code == 200
            task = asyncio.create_task(
                client.post("/v1/chat/completions", content=b'{"max_tokens": 8}')
            )
            await upstream.opened.wait()
            await asyncio.sleep(0.05)
            assert lifecycle._active == 1
            task.cancel()
            await asyncio.sleep(0.05)
            assert lifecycle._active == 1
            upstream.hold.set()
            for _ in range(50):
                if lifecycle._active == 0:
                    break
                await asyncio.sleep(0.02)
            assert lifecycle._active == 0
            assert chunks

    run(once(b"{}"))
    run(once(b"data: {}\n\n"))


def test_unconfirmed_upstream_does_not_clear_active():
    async def scenario():
        worker = FakeWorker()
        worker.ownership_value = "resident"
        upstream = FakeUpstream()
        upstream.fail = RuntimeError("connection reset")
        app, lifecycle = build(worker, upstream)
        async with client_for(app) as client:
            assert (await client.post("/control/load")).status_code == 200
            failed = await client.post("/v1/chat/completions", json=CHAT)
            assert failed.status_code == 502
            status = (await client.get("/control/status")).json()
            assert status["active_requests"] == 1
            assert status["state"] == "failed"
            assert status["residency"] == "resident"
            nxt = await client.post("/v1/chat/completions", json=CHAT)
            assert nxt.status_code == 503
            busy = await client.post("/control/unload")
            assert busy.status_code == 409
            assert busy.json()["error"]["code"] == "BUSY"
            await lifecycle.note_worker_exit(lifecycle._generation, "all_dead")
            cleared = (await client.get("/control/status")).json()
            assert cleared["active_requests"] == 0
            assert cleared["residency"] == "not_resident"
            recovered = await client.post("/control/unload")
            assert recovered.status_code == 200
            assert recovered.json()["state"] == "unloaded"
            assert recovered.json()["last_error"] is None

    run(scenario())


def test_load_failure_cleanup_ownership_and_unexpected_exit():
    async def scenario():
        worker = FakeWorker()
        worker.fail_start = WorkerFailure("engine init failed", "resident")
        app, lifecycle = build(worker, FakeUpstream())
        async with client_for(app) as client:
            failed = await client.post("/control/load")
            assert failed.status_code == 500
            assert failed.json()["error"]["code"] == "LOAD_FAILED"
            body = (await client.get("/control/status")).json()
            assert body["state"] == "failed"
            assert body["residency"] == "not_resident"
            assert worker.stop_count >= 1
            recovered = await client.post("/control/unload")
            assert recovered.json()["state"] == "unloaded"

        worker = FakeWorker()
        worker.fail_start = WorkerFailure("engine init failed", "not_resident")
        worker.stop_failure = WorkerFailure("workers remain", "resident")
        app, _lifecycle = build(worker, FakeUpstream())
        async with client_for(app) as client:
            assert (await client.post("/control/load")).status_code == 500
            body = (await client.get("/control/status")).json()
            assert body["state"] == "failed"
            assert body["residency"] == "resident"

        worker = FakeWorker()
        worker.start_gate = asyncio.Event()
        worker.stop_failure = WorkerFailure("ownership unknown", "unknown")
        app, _lifecycle = build(worker, FakeUpstream(), load_timeout_sec=0.2)
        async with client_for(app) as client:
            assert (await client.post("/control/load")).status_code == 500
            body = (await client.get("/control/status")).json()
            assert body["state"] == "failed"
            assert body["residency"] == "unknown"
            assert "not_resident" != body["residency"]

        worker = FakeWorker()
        upstream = FakeUpstream()
        app, lifecycle = build(worker, upstream)
        async with client_for(app) as client:
            assert (await client.post("/control/load")).status_code == 200
            await lifecycle.note_worker_exit(lifecycle._generation, "unknown")
            body = (await client.get("/control/status")).json()
            assert body["state"] == "failed"
            assert body["residency"] == "unknown"
            assert body["active_requests"] == 0

        worker = FakeWorker()
        worker.port_busy = True
        app, _lifecycle = build(worker, FakeUpstream())
        async with client_for(app) as client:
            assert (await client.post("/control/load")).status_code == 500
            body = (await client.get("/control/status")).json()
            assert body["state"] == "failed"
            assert body["residency"] == "not_resident"
            assert worker.alive is False

    run(scenario())


def test_previous_generation_callback_does_not_overwrite():
    async def scenario():
        worker = FakeWorker()
        _app, lifecycle = build(worker, FakeUpstream())
        await lifecycle.load()
        first = lifecycle._generation
        await lifecycle.unload()
        await lifecycle.load()
        assert lifecycle._generation != first
        await lifecycle.note_worker_exit(first, "unknown")
        assert (await lifecycle.status())["state"] == "ready"

    run(scenario())


def test_load_timeout_attempts_cleanup_before_not_resident():
    async def scenario():
        worker = FakeWorker()
        worker.start_gate = asyncio.Event()
        _app, lifecycle = build(worker, FakeUpstream(), load_timeout_sec=0.2)
        failed = None
        try:
            await lifecycle.load()
        except Exception as exc:
            failed = exc
        assert failed is not None
        assert failed.code == "LOAD_FAILED"
        assert worker.stop_count >= 1
        body = await lifecycle.status()
        assert body["state"] == "failed"
        assert body["residency"] == "not_resident"

    run(scenario())


def test_upstream_http_error_does_not_fail_lifecycle():
    async def scenario():
        worker = FakeWorker()
        upstream = FakeUpstream()

        class Bad(FakeUpstream):
            async def open(self, method, path, headers, content, params):
                await super().open(method, path, headers, content, params)
                stream = await FakeUpstream.open(self, method, path, headers, content, params)
                stream.status_code = 400
                stream._body = b'{"error":"bad request"}'
                return stream

        # Use a one-shot upstream that returns 400 after a normal open path.
        class Rejecting:
            def __init__(self):
                self.bodies = []

            async def open(self, method, path, headers, content, params):
                self.bodies.append(content)
                stream = __import__("tests.fakes", fromlist=["FakeStream"]).FakeStream(
                    b'{"error":"bad request"}',
                    status=400,
                )
                return stream

        rejecting = Rejecting()
        app, lifecycle = build(worker, rejecting)
        async with client_for(app) as client:
            assert (await client.post("/control/load")).status_code == 200
            response = await client.post("/v1/chat/completions", json=CHAT)
            assert response.status_code == 400
            assert response.content == b'{"error":"bad request"}'
            body = (await client.get("/control/status")).json()
            assert body["state"] == "ready"
            assert body["active_requests"] == 0
            assert body["last_error"] is None

    run(scenario())


def test_status_failure_and_bad_control_body_and_shutdown():
    async def scenario():
        worker = FakeWorker()
        app, lifecycle = build(worker, FakeUpstream())
        async with client_for(app) as client:
            lifecycle.fail_status()
            status = await client.get("/control/status")
            assert status.status_code == 500
            assert status.json()["error"]["code"] == "STATUS_FAILED"
            lifecycle._status_fault = None
            bad = await client.post("/control/load", json={"model": "x"})
            assert bad.status_code == 400
            assert bad.json()["error"]["code"] == "BAD_REQUEST"
            assert (await client.post("/control/load", json={})).status_code == 200
            await lifecycle.shutdown()
            assert worker.stop_count >= 1
            assert (await lifecycle.status())["state"] == "unloaded"

    run(scenario())


def test_output_limit_helper_rejects_non_integers():
    assert evaluate_output_limits(b'{"max_tokens": true}', limit=4096, require_explicit=True)
    assert (
        evaluate_output_limits(b'{"max_tokens": 16}', limit=4096, require_explicit=True) is None
    )


def test_owned_process_group_dies_without_signaling_another_runtime():
    async def scenario():
        bystander = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
            start_new_session=True,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        code = "import os, time\npid = os.fork()\nif pid == 0:\n    time.sleep(30)\nelse:\n    os._exit(0)\n"
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            code,
            start_new_session=True,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        pgid = proc.pid
        try:
            await proc.wait()
            await asyncio.sleep(0.1)
            assert _group_alive(pgid) is True
            result = await terminate_owned_group(pgid, set(), 2, 1)
            assert result == "not_resident"
            assert _group_alive(pgid) is False
            os.kill(bystander.pid, 0)
        finally:
            if _group_alive(pgid):
                os.killpg(pgid, signal.SIGKILL)
            try:
                os.kill(bystander.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await bystander.wait()

    run(scenario())
