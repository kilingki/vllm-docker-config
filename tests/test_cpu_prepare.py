import threading
import time
from pathlib import Path

import pytest

from controller.prepare import Identity, PrepareError, prepare

pytestmark = pytest.mark.cpu_prepare


class FakeDocker:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.running: str | None = None
        self.identity_obj: Identity | None = None
        self.up_calls = 0
        self.max_depth = 0
        self._depth = 0
        self._guard = threading.Lock()
        self.sleep = 0.0

    def running_id(self) -> str | None:
        return self.running

    def identity(self, container_id: str) -> Identity | None:
        return self.identity_obj

    def up(self) -> None:
        with self._guard:
            self._depth += 1
            self.max_depth = max(self.max_depth, self._depth)
        if self.sleep:
            time.sleep(self.sleep)
        with self._guard:
            self._depth -= 1
            self.up_calls += 1
            self.running = "cid-new"
            self.identity_obj = Identity(
                container_id="cid-new",
                service="vllm-runtime",
                working_dir=str(self.root),
                published_port=8000,
                running=True,
            )


def _identity(root: Path, **overrides) -> Identity:
    values = dict(
        container_id="cid",
        service="vllm-runtime",
        working_dir=str(root),
        published_port=8000,
        running=True,
    )
    values.update(overrides)
    return Identity(**values)


def _unloaded():
    return {
        "state": "unloaded",
        "residency": "not_resident",
        "active_requests": 0,
        "last_error": None,
    }


@pytest.mark.parametrize(
    ("state", "active"),
    [
        ("ready", 2),
        ("loading", 0),
        ("unloading", 0),
        ("failed", 1),
    ],
)
def test_running_container_preserves_status(tmp_path: Path, state: str, active: int):
    docker = FakeDocker(tmp_path)
    docker.running = "cid"
    docker.identity_obj = _identity(tmp_path)
    seen = {
        "state": state,
        "residency": "resident",
        "active_requests": active,
        "last_error": {"code": "LOAD_FAILED", "message": "x"} if state == "failed" else None,
    }

    def getter(_url: str):
        return dict(seen)

    assert (
        prepare(
            tmp_path,
            8000,
            docker=docker,
            status_getter=getter,
            lock_path=tmp_path / ".lock",
            running_timeout=0.5,
            new_timeout=0.5,
        )
        == 0
    )
    assert docker.up_calls == 0
    assert seen["state"] == state
    assert seen["active_requests"] == active


def test_running_endpoint_error_does_not_restart(tmp_path: Path):
    docker = FakeDocker(tmp_path)
    docker.running = "cid"
    docker.identity_obj = _identity(tmp_path)

    def getter(_url: str):
        raise TimeoutError("status down")

    with pytest.raises(PrepareError, match="not restarting"):
        prepare(
            tmp_path,
            8000,
            docker=docker,
            status_getter=getter,
            lock_path=tmp_path / ".lock",
            running_timeout=0.4,
            new_timeout=0.4,
        )
    assert docker.up_calls == 0


def test_new_container_must_report_unloaded(tmp_path: Path):
    docker = FakeDocker(tmp_path)

    def getter(_url: str):
        return _unloaded()

    assert (
        prepare(
            tmp_path,
            8000,
            docker=docker,
            status_getter=getter,
            lock_path=tmp_path / ".lock",
            new_timeout=1,
        )
        == 0
    )
    assert docker.up_calls == 1


def test_other_service_status_is_not_success(tmp_path: Path):
    docker = FakeDocker(tmp_path)
    calls = {"n": 0}

    def getter(_url: str):
        calls["n"] += 1
        if docker.up_calls == 0:
            return {
                "state": "ready",
                "residency": "resident",
                "active_requests": 3,
                "last_error": None,
            }
        return _unloaded()

    assert (
        prepare(
            tmp_path,
            8000,
            docker=docker,
            status_getter=getter,
            lock_path=tmp_path / ".lock",
            new_timeout=1,
        )
        == 0
    )
    assert docker.up_calls == 1
    assert calls["n"] >= 1


def test_running_identity_mismatch_fails_without_restart(tmp_path: Path):
    docker = FakeDocker(tmp_path)
    docker.running = "cid"
    docker.identity_obj = _identity(tmp_path, service="other-runtime", published_port=8000)
    with pytest.raises(PrepareError, match="not this Compose service"):
        prepare(
            tmp_path,
            8000,
            docker=docker,
            status_getter=lambda _url: _unloaded(),
            lock_path=tmp_path / ".lock",
        )
    assert docker.up_calls == 0


def test_concurrent_prepare_is_serialized(tmp_path: Path):
    docker = FakeDocker(tmp_path)
    docker.sleep = 0.2
    lock = tmp_path / ".lock"
    results: list[int] = []

    def getter(_url: str):
        return _unloaded()

    def call():
        results.append(
            prepare(
                tmp_path,
                8000,
                docker=docker,
                status_getter=getter,
                lock_path=lock,
                new_timeout=2,
            )
        )

    threads = [threading.Thread(target=call) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results == [0, 0]
    assert docker.max_depth == 1
    assert docker.up_calls == 1
