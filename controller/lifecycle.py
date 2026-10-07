"""Model lifecycle, worker ownership, and readiness.

load/unload are shared tasks. Waiters await that task, so a later operation
cannot change the result they observe. Client disconnect does not cancel the
task. not_resident is reported only after owned workers are confirmed gone.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import socket
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from controller.config import ConfigError, Settings

logger = logging.getLogger("controller.lifecycle")


class ControlError(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


class WorkerFailure(Exception):
    def __init__(self, message: str, residency: str) -> None:
        super().__init__(message)
        self.message = message
        self.residency = residency


@dataclass
class OpResult:
    ok: bool
    snapshot: dict[str, Any] | None = None
    error: ControlError | None = None


@dataclass
class Ticket:
    generation: int
    settled: bool = False


@dataclass
class Admit:
    ok: bool
    status_code: int = 200
    detail: str | None = None
    ticket: Ticket | None = None


class ModelWorker(Protocol):
    async def start(self, generation: int) -> None: ...

    async def stop(self) -> str: ...

    def is_alive(self) -> bool: ...

    def ownership(self) -> str: ...

    async def probe(self, timeout: float) -> str: ...

    def fatal_error(self) -> str | None: ...

    def arm_watch(self, handler: Callable[[int, str], Awaitable[None]]) -> None: ...


def port_is_open(host: str, port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            return sock.connect_ex((host, port)) == 0
    except OSError:
        return False


def _group_alive(pgid: int) -> bool | None:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return None
    return True


def _pid_alive(pid: int) -> bool | None:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return None
    return True


def snapshot_descendants(root_pid: int) -> set[int] | None:
    """PIDs whose ancestor is root_pid, from a single /proc scan."""
    try:
        entries = [name for name in os.listdir("/proc") if name.isdigit()]
    except OSError:
        return None
    parents: dict[int, int] = {}
    for name in entries:
        pid = int(name)
        try:
            with open(f"/proc/{pid}/stat", encoding="utf-8", errors="replace") as handle:
                stat = handle.read()
        except OSError:
            continue
        end = stat.rfind(")")
        if end < 0:
            continue
        fields = stat[end + 2 :].split()
        if len(fields) < 2:
            continue
        try:
            parents[pid] = int(fields[1])
        except ValueError:
            continue
    descendants: set[int] = set()
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if pid in descendants:
                continue
            if parent == root_pid or parent in descendants:
                descendants.add(pid)
                changed = True
    return descendants


def _signal_group(pgid: int, sig: int) -> bool | None:
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        return False
    except OSError:
        return None
    return True


def _signal_pids(pids: set[int], sig: int) -> None:
    for pid in pids:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            continue
        except OSError:
            logger.warning("failed to signal owned descendant %s", pid)


async def terminate_owned_group(
    pgid: int,
    extra_pids: set[int] | None,
    term_timeout: float,
    kill_timeout: float,
) -> str:
    """Stop one owned process group and the descendants snapshotted for it.

    Returns not_resident, resident, or unknown. PIDs outside extra_pids are
    not signaled individually; the process group signal covers the session.
    """
    if extra_pids is None:
        return "unknown"
    owned = set(extra_pids)

    def gone() -> bool | None:
        group = _group_alive(pgid)
        if group is None:
            return None
        for pid in owned:
            alive = _pid_alive(pid)
            if alive is None:
                return None
            if alive:
                return False
        return not group

    signaled = _signal_group(pgid, signal.SIGTERM)
    if signaled is None:
        return "unknown"
    if await _wait_until(gone, term_timeout):
        return "not_resident"
    signaled = _signal_group(pgid, signal.SIGKILL)
    _signal_pids(owned, signal.SIGKILL)
    if signaled is None:
        return "unknown"
    if await _wait_until(gone, kill_timeout):
        return "not_resident"
    final = gone()
    if final is True:
        return "not_resident"
    if final is False:
        return "resident"
    return "unknown"


async def _wait_until(predicate: Callable[[], bool | None], timeout: float) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        state = predicate()
        if state is True:
            return True
        if loop.time() >= deadline:
            return False
        await asyncio.sleep(0.05)


class VllmServeWorker:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._proc: asyncio.subprocess.Process | None = None
        self._pgid: int | None = None
        self._descendants: set[int] | None = None
        self._log_task: asyncio.Task[None] | None = None
        self._watch_task: asyncio.Task[None] | None = None
        self._fatal: str | None = None
        self._handler: Callable[[int, str], Awaitable[None]] | None = None
        self._generation = 0
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(None))

    def is_alive(self) -> bool:
        proc = self._proc
        if proc is not None and proc.returncode is None:
            return True
        if self._pgid is None:
            return False
        group = _group_alive(self._pgid)
        if group:
            return True
        if self._descendants:
            return any(_pid_alive(pid) is True for pid in self._descendants)
        return False

    def ownership(self) -> str:
        if self._descendants is None and self._pgid is None:
            return "unknown"
        if self.is_alive():
            return "resident"
        if self._descendants is None:
            return "unknown"
        return "all_dead"

    def fatal_error(self) -> str | None:
        return self._fatal

    def arm_watch(self, handler: Callable[[int, str], Awaitable[None]]) -> None:
        self._handler = handler
        proc = self._proc
        if proc is None:
            return
        if proc.returncode is not None:
            asyncio.create_task(self._emit_exit(), name="vllm-exit")
            return
        if self._watch_task is None:
            self._watch_task = asyncio.create_task(self._watch(), name="vllm-watch")

    async def start(self, generation: int) -> None:
        self._generation = generation
        self._fatal = None
        if port_is_open(self.settings.host, self.settings.port):
            raise WorkerFailure(
                f"{self.settings.host}:{self.settings.port} already has a listener",
                "not_resident",
            )
        if not self.settings.model_path or not os.path.isdir(self.settings.model_path):
            raise WorkerFailure(
                f"MODEL_PATH is not a directory: {self.settings.model_path}",
                "not_resident",
            )
        try:
            cmd = self.settings.vllm_serve_argv()
        except ConfigError as exc:
            raise WorkerFailure(str(exc), "not_resident") from exc
        logger.info("Starting vLLM: %s", cmd)
        try:
            self._proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            self._proc = None
            raise WorkerFailure(str(exc), "not_resident") from exc
        self._pgid = self._proc.pid
        self._descendants = snapshot_descendants(self._proc.pid)
        self._log_task = asyncio.create_task(self._pump_logs(self._proc), name="vllm-logs")
        await self._wait_until_ready()
        current = snapshot_descendants(self._proc.pid)
        if current is not None:
            self._descendants = set(self._descendants or set()) | current

    async def stop(self) -> str:
        await self._cancel_watch()
        proc = self._proc
        pgid = self._pgid
        if proc is None or pgid is None:
            self._proc = None
            self._pgid = None
            return "not_resident"
        if proc.pid is not None:
            current = snapshot_descendants(proc.pid)
            if current is None:
                self._descendants = None
            else:
                self._descendants = set(self._descendants or set()) | current
        if not self.is_alive() and self._descendants is not None and not any(
            _pid_alive(pid) is True for pid in self._descendants
        ):
            self._proc = None
            self._pgid = None
            await self._reap_logs()
            return "not_resident"
        result = await terminate_owned_group(
            pgid,
            self._descendants,
            self.settings.unload_timeout_sec,
            self.settings.kill_grace_sec,
        )
        await self._reap_logs()
        if result != "not_resident":
            raise WorkerFailure("failed to release model workers", result)
        self._proc = None
        self._pgid = None
        self._descendants = set()
        return "not_resident"

    async def probe(self, timeout: float) -> str:
        url = f"{self.settings.vllm_base_url}/health"
        try:
            response = await self._client.get(url, timeout=timeout)
        except httpx.ReadTimeout:
            return "read_timeout"
        except httpx.TimeoutException:
            return "read_timeout"
        except httpx.HTTPError:
            return "connect_error"
        if response.status_code != 200:
            return "http_error"
        return "ok"

    async def _wait_until_ready(self) -> None:
        deadline = asyncio.get_running_loop().time() + self.settings.load_timeout_sec
        health_url = f"{self.settings.vllm_base_url}/health"
        models_url = f"{self.settings.vllm_base_url}/v1/models"
        expected = self.settings.served_model_name
        while True:
            proc = self._proc
            if proc is not None and proc.returncode is not None:
                raise WorkerFailure(
                    f"vLLM exited during load with code {proc.returncode}",
                    "not_resident" if not self.is_alive() else "resident",
                )
            if asyncio.get_running_loop().time() >= deadline:
                raise WorkerFailure(
                    f"vLLM did not become ready within {self.settings.load_timeout_sec}s",
                    "resident" if self.is_alive() else "not_resident",
                )
            try:
                health = await self._client.get(health_url, timeout=2.0)
                if health.status_code == 200:
                    models = await self._client.get(models_url, timeout=2.0)
                    if models.status_code == 200 and _serves_model(models.json(), expected):
                        logger.info("vLLM ready for served model %s", expected)
                        return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.5)

    async def _watch(self) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            await proc.wait()
        except asyncio.CancelledError:
            raise
        await self._emit_exit()

    async def _emit_exit(self) -> None:
        handler = self._handler
        if handler is None:
            return
        await handler(self._generation, self.ownership())

    async def _cancel_watch(self) -> None:
        watch = self._watch_task
        self._watch_task = None
        if watch is None:
            return
        watch.cancel()
        try:
            await watch
        except asyncio.CancelledError:
            pass

    async def _reap_logs(self) -> None:
        task = self._log_task
        self._log_task = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _pump_logs(self, proc: asyncio.subprocess.Process) -> None:
        if proc.stdout is None:
            return
        try:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip()
                logger.info("[vllm] %s", text)
                lowered = text.lower()
                if "cuda out of memory" in lowered or "engine core" in lowered and "fatal" in lowered:
                    self._fatal = text
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("failed while forwarding vLLM logs")


def _serves_model(payload: Any, expected: str) -> bool:
    if not isinstance(payload, dict):
        return False
    data = payload.get("data")
    if not isinstance(data, list):
        return False
    for item in data:
        if isinstance(item, dict) and item.get("id") == expected:
            return True
    return False


class Lifecycle:
    def __init__(self, settings: Settings, worker: ModelWorker | None = None) -> None:
        self.settings = settings
        self._worker = worker if worker is not None else VllmServeWorker(settings)
        self._lock = asyncio.Lock()
        self._state = "unloaded"
        self._residency = "not_resident"
        self._active = 0
        self._last_error: dict[str, str] | None = None
        self._generation = 0
        self._operation: asyncio.Task[OpResult] | None = None
        self._operation_kind: str | None = None
        self._stopping = False
        self._admission_blocked = False
        self._status_fault: str | None = None
        self._monitor: asyncio.Task[None] | None = None
        self.busy_probe_timeouts = 0
        self._background: set[asyncio.Task[Any]] = set()

    @property
    def lock(self) -> asyncio.Lock:
        return self._lock

    def worker_alive(self) -> bool:
        return self._worker.is_alive()

    def fail_status(self, message: str = "status is unavailable") -> None:
        self._status_fault = message

    async def status(self) -> dict[str, Any]:
        async with self._lock:
            if self._status_fault is not None:
                raise ControlError(500, "STATUS_FAILED", self._status_fault)
            return self._snapshot()

    async def load(self) -> dict[str, Any]:
        async with self._lock:
            prepared = self._prepare_load_locked()
        if isinstance(prepared, dict):
            return prepared
        return await self._await_operation(prepared)

    async def unload(self) -> dict[str, Any]:
        async with self._lock:
            prepared = self._prepare_unload_locked()
        if isinstance(prepared, dict):
            return prepared
        return await self._await_operation(prepared)

    async def admit(
        self,
        *,
        counted: bool,
        output_error: str | None,
        start_task: Callable[[Ticket | None], asyncio.Task[Any]],
    ) -> Admit:
        async with self._lock:
            if not self._is_ready_locked():
                return Admit(False, 503, "model is not ready")
            if output_error:
                return Admit(False, 400, output_error)
            ticket: Ticket | None = None
            if counted:
                if self._active >= self.settings.max_active_requests:
                    return Admit(False, 429, "too many active requests")
                self._active += 1
                ticket = Ticket(self._generation)
            try:
                task = start_task(ticket)
            except Exception:
                if ticket is not None and self._active > 0:
                    self._active -= 1
                raise
            self._track(task)
            return Admit(True, ticket=ticket)

    async def release(self, ticket: Ticket) -> None:
        async with self._lock:
            if ticket.settled or ticket.generation != self._generation:
                ticket.settled = True
                return
            ticket.settled = True
            if self._active > 0:
                self._active -= 1

    async def mark_unconfirmed(self, ticket: Ticket) -> None:
        async with self._lock:
            if ticket.settled or ticket.generation != self._generation:
                ticket.settled = True
                return
            ticket.settled = True
            self._admission_blocked = True
            self._state = "failed"
            ownership = self._worker.ownership()
            if ownership == "resident":
                self._residency = "resident"
            elif ownership == "all_dead":
                self._residency = "unknown"
            else:
                self._residency = "unknown"
            self._last_error = {
                "code": "EXECUTION_UNCONFIRMED",
                "message": "upstream ended without proof that model execution finished",
            }
            logger.error(self._last_error["message"])

    async def note_worker_exit(self, generation: int, ownership: str) -> None:
        async with self._lock:
            if generation != self._generation or self._stopping:
                return
            if self._state == "unloading":
                return
            self._state = "failed"
            self._last_error = {
                "code": "WORKER_EXITED",
                "message": "vLLM worker exited unexpectedly",
            }
            if ownership == "all_dead":
                self._residency = "not_resident"
                self._active = 0
                self._admission_blocked = False
            elif ownership == "resident":
                self._residency = "resident"
            else:
                self._residency = "unknown"
            logger.error(self._last_error["message"])

    async def shutdown(self) -> None:
        logger.info("Controller shutting down; stopping vLLM if it is running")
        self._cancel_monitor()
        if self.worker_alive():
            try:
                await self._worker.stop()
            except WorkerFailure:
                logger.exception("failed to stop vLLM during shutdown")
                return
        async with self._lock:
            self._state = "unloaded"
            self._residency = "not_resident"
            self._active = 0
            self._last_error = None

    def _prepare_load_locked(self) -> asyncio.Task[OpResult] | dict[str, Any]:
        if self._state == "loading" and self._operation is not None and self._operation_kind == "load":
            return self._operation
        if self._state == "unloading":
            raise ControlError(409, "LIFECYCLE_CONFLICT", "unload is in progress")
        if self._state == "failed" and self._residency != "not_resident":
            raise ControlError(
                409,
                "LIFECYCLE_CONFLICT",
                "load is not allowed while residency remains",
            )
        if self._state == "ready":
            return self._snapshot()
        if self._active > 0:
            raise ControlError(409, "BUSY", "runtime has active inference requests")
        if self._residency != "not_resident":
            raise ControlError(
                409,
                "LIFECYCLE_CONFLICT",
                "load is not allowed while residency remains",
            )
        self._generation += 1
        generation = self._generation
        self._state = "loading"
        self._stopping = False
        self._admission_blocked = False
        self._operation_kind = "load"
        task = asyncio.create_task(self._run_load(generation), name="load")
        self._operation = task
        return task

    def _prepare_unload_locked(self) -> asyncio.Task[OpResult] | dict[str, Any]:
        if (
            self._state == "unloading"
            and self._operation is not None
            and self._operation_kind == "unload"
        ):
            return self._operation
        if self._active > 0:
            raise ControlError(409, "BUSY", "runtime has active inference requests")
        if self._state == "loading":
            raise ControlError(409, "LIFECYCLE_CONFLICT", "load is in progress")
        if self._state == "unloaded" and self._residency == "not_resident":
            return self._snapshot()
        if self._state == "failed" and self._residency == "not_resident":
            self._state = "unloaded"
            self._residency = "not_resident"
            self._last_error = None
            self._admission_blocked = False
            return self._snapshot()
        if self._state == "ready" or (
            self._state == "failed" and self._residency in {"resident", "unknown"}
        ):
            self._state = "unloading"
            self._stopping = True
            self._cancel_monitor()
            self._operation_kind = "unload"
            task = asyncio.create_task(self._run_unload(), name="unload")
            self._operation = task
            return task
        raise ControlError(409, "LIFECYCLE_CONFLICT", "unload is not allowed in the current state")

    async def _await_operation(self, task: asyncio.Task[OpResult]) -> dict[str, Any]:
        result = await asyncio.shield(task)
        if not result.ok:
            assert result.error is not None
            raise result.error
        assert result.snapshot is not None
        return result.snapshot

    async def _run_load(self, generation: int) -> OpResult:
        try:
            await asyncio.wait_for(
                self._worker.start(generation),
                timeout=self.settings.load_timeout_sec,
            )
        except TimeoutError:
            residency = await self._cleanup_after_failure()
            return await self._record_failed_load(
                generation,
                f"vLLM did not become ready within {self.settings.load_timeout_sec}s",
                residency,
            )
        except WorkerFailure as exc:
            residency = await self._cleanup_after_failure()
            return await self._record_failed_load(generation, exc.message, residency)
        except Exception as exc:
            residency = await self._cleanup_after_failure()
            return await self._record_failed_load(generation, str(exc), residency)
        async with self._lock:
            if generation != self._generation:
                return OpResult(
                    False,
                    error=ControlError(500, "LOAD_FAILED", "stale load generation"),
                )
            if not self._worker.is_alive():
                return self._fail_load_locked(
                    generation,
                    "vLLM exited during load",
                    "not_resident" if self._worker.ownership() == "all_dead" else "unknown",
                )
            self._state = "ready"
            self._residency = "resident"
            self._last_error = None
            self._admission_blocked = False
            self._stopping = False
            if self._operation is asyncio.current_task():
                self._operation = None
                self._operation_kind = None
            snapshot = self._snapshot()
        self._worker.arm_watch(self.note_worker_exit)
        self._start_monitor(generation)
        return OpResult(True, snapshot)

    async def _run_unload(self) -> OpResult:
        try:
            residency = await self._worker.stop()
        except WorkerFailure as exc:
            async with self._lock:
                self._state = "failed"
                self._residency = exc.residency
                self._last_error = {"code": "UNLOAD_FAILED", "message": exc.message}
                self._stopping = False
                if self._operation is asyncio.current_task():
                    self._operation = None
                    self._operation_kind = None
                return OpResult(
                    False,
                    error=ControlError(500, "UNLOAD_FAILED", exc.message),
                )
        except Exception as exc:
            async with self._lock:
                self._state = "failed"
                self._residency = "unknown"
                self._last_error = {"code": "UNLOAD_FAILED", "message": str(exc)}
                self._stopping = False
                if self._operation is asyncio.current_task():
                    self._operation = None
                    self._operation_kind = None
                return OpResult(False, error=ControlError(500, "UNLOAD_FAILED", str(exc)))
        async with self._lock:
            if residency != "not_resident":
                self._state = "failed"
                self._residency = residency
                self._last_error = {
                    "code": "UNLOAD_FAILED",
                    "message": "worker ownership was not confirmed",
                }
                self._stopping = False
                if self._operation is asyncio.current_task():
                    self._operation = None
                    self._operation_kind = None
                return OpResult(
                    False,
                    error=ControlError(500, "UNLOAD_FAILED", "worker ownership was not confirmed"),
                )
            self._state = "unloaded"
            self._residency = "not_resident"
            self._active = 0
            self._last_error = None
            self._admission_blocked = False
            self._stopping = False
            if self._operation is asyncio.current_task():
                self._operation = None
                self._operation_kind = None
            return OpResult(True, self._snapshot())

    def _fail_load_locked(self, generation: int, message: str, residency: str) -> OpResult:
        if generation == self._generation:
            self._state = "failed"
            self._residency = residency
            self._last_error = {"code": "LOAD_FAILED", "message": message}
            self._stopping = False
            if self._operation is asyncio.current_task():
                self._operation = None
                self._operation_kind = None
        return OpResult(False, error=ControlError(500, "LOAD_FAILED", message))

    async def _cleanup_after_failure(self) -> str:
        self._stopping = True
        try:
            return await self._worker.stop()
        except WorkerFailure as exc:
            return exc.residency
        except Exception:
            logger.exception("cleanup after failed load raised")
            return "unknown"

    async def _record_failed_load(self, generation: int, message: str, residency: str) -> OpResult:
        async with self._lock:
            return self._fail_load_locked(generation, message, residency)

    def _start_monitor(self, generation: int) -> None:
        self._cancel_monitor()
        self._monitor = asyncio.create_task(self._monitor_loop(generation), name="vllm-monitor")

    def _cancel_monitor(self) -> None:
        monitor = self._monitor
        self._monitor = None
        if monitor is not None:
            monitor.cancel()

    async def _monitor_loop(self, generation: int) -> None:
        failures = 0
        try:
            while True:
                await asyncio.sleep(self.settings.health_probe_interval_sec)
                async with self._lock:
                    if generation != self._generation or self._state != "ready":
                        return
                if not self._worker.is_alive() or self._worker.fatal_error():
                    await self.note_worker_exit(generation, self._worker.ownership())
                    return
                result = await self._worker.probe(self.settings.health_probe_timeout_sec)
                async with self._lock:
                    if generation != self._generation or self._state != "ready":
                        return
                    if result == "ok":
                        failures = 0
                        continue
                    if result == "read_timeout" and self._active > 0:
                        self.busy_probe_timeouts += 1
                        continue
                    if result in {"http_error", "connect_error", "read_timeout"}:
                        failures += 1
                    if failures >= self.settings.health_probe_failures:
                        self._state = "failed"
                        self._last_error = {
                            "code": "WORKER_UNHEALTHY",
                            "message": "vLLM health failed repeatedly",
                        }
                        return
        except asyncio.CancelledError:
            raise

    def _is_ready_locked(self) -> bool:
        return (
            self._state == "ready"
            and self._residency == "resident"
            and not self._admission_blocked
        )

    def _track(self, task: asyncio.Task[Any]) -> None:
        self._background.add(task)

        def _done(done: asyncio.Task[Any]) -> None:
            self._background.discard(done)
            if done.cancelled():
                return
            error = done.exception()
            if error is not None:
                logger.error("background task failed", exc_info=error)

        task.add_done_callback(_done)

    def _snapshot(self) -> dict[str, Any]:
        last_error = None if self._last_error is None else dict(self._last_error)
        return {
            "state": self._state,
            "residency": self._residency,
            "active_requests": self._active,
            "last_error": last_error,
        }


