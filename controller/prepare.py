"""Bring the Compose service to a usable control endpoint without loading a model."""

from __future__ import annotations

import fcntl
import sys
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


SERVICE = "vllm-runtime"


class PrepareError(Exception):
    pass


@dataclass
class Identity:
    container_id: str
    service: str
    working_dir: str
    published_port: int | None
    running: bool


class DockerBackend:
    def __init__(self, root: Path, service: str = SERVICE) -> None:
        self.root = root
        self.service = service

    def running_id(self) -> str | None:
        result = subprocess.run(
            ["docker", "compose", "ps", "-q", self.service],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=False,
        )
        container_id = result.stdout.strip()
        return container_id or None

    def identity(self, container_id: str) -> Identity | None:
        result = subprocess.run(
            ["docker", "inspect", container_id],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        payload = json.loads(result.stdout)[0]
        labels = payload.get("Config", {}).get("Labels") or {}
        ports = payload.get("NetworkSettings", {}).get("Ports") or {}
        published = _published_port(ports)
        return Identity(
            container_id=container_id,
            service=str(labels.get("com.docker.compose.service", "")),
            working_dir=str(labels.get("com.docker.compose.project.working_dir", "")),
            published_port=published,
            running=payload.get("State", {}).get("Status") == "running",
        )

    def up(self) -> None:
        result = subprocess.run(
            ["docker", "compose", "up", "-d", "--no-build", "--no-recreate"],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise PrepareError(result.stderr.strip() or "docker compose up failed")


def _published_port(ports: dict[str, Any]) -> int | None:
    binding = ports.get("8000/tcp")
    if not binding:
        return None
    try:
        return int(binding[0]["HostPort"])
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def fetch_status(url: str, timeout: float = 3) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise PrepareError("status payload is not an object")
    return payload


def _identity_ok(identity: Identity | None, root: Path, port: int) -> bool:
    if identity is None or not identity.running:
        return False
    if identity.service != SERVICE:
        return False
    if Path(identity.working_dir).resolve() != root.resolve():
        return False
    if identity.published_port != port:
        return False
    return True


def _status_fields(payload: dict[str, Any]) -> bool:
    return set(payload) >= {"state", "residency", "active_requests", "last_error"}


def prepare(
    root: Path,
    port: int,
    *,
    docker: DockerBackend | None = None,
    status_getter: Callable[[str], dict[str, Any]] | None = None,
    lock_path: Path | None = None,
    running_timeout: float = 5,
    new_timeout: float = 60,
) -> int:
    """Return 0 when the control endpoint is usable. Do not build, download, or load."""
    root = root.resolve()
    backend = docker if docker is not None else DockerBackend(root)
    getter = status_getter if status_getter is not None else fetch_status
    lock_file = lock_path if lock_path is not None else root / ".prepare.lock"
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    with lock_file.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            return _prepare_locked(root, port, backend, getter, running_timeout, new_timeout)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _prepare_locked(
    root: Path,
    port: int,
    docker: DockerBackend,
    status_getter: Callable[[str], dict[str, Any]],
    running_timeout: float,
    new_timeout: float,
) -> int:
    url = f"http://127.0.0.1:{port}/control/status"
    running = docker.running_id()
    if running:
        identity = docker.identity(running)
        if not _identity_ok(identity, root, port):
            raise PrepareError("running container is not this Compose service")
        payload = _poll_status(status_getter, url, running_timeout)
        if payload is None or not _status_fields(payload):
            raise PrepareError(
                "vllm-runtime is running but GET /control/status failed; not restarting"
            )
        return 0

    docker.up()
    deadline = time.monotonic() + new_timeout
    while time.monotonic() < deadline:
        current = docker.running_id()
        identity = docker.identity(current) if current else None
        if _identity_ok(identity, root, port):
            try:
                payload = status_getter(url)
            except (OSError, urllib.error.URLError, PrepareError, json.JSONDecodeError, TimeoutError):
                payload = None
            if (
                payload
                and _status_fields(payload)
                and payload.get("state") == "unloaded"
                and payload.get("residency") == "not_resident"
                and payload.get("active_requests") == 0
            ):
                return 0
        time.sleep(0.2)
    raise PrepareError("new runtime did not report unloaded/not_resident")


def _poll_status(
    status_getter: Callable[[str], dict[str, Any]],
    url: str,
    timeout: float,
) -> dict[str, Any] | None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            return status_getter(url)
        except (OSError, urllib.error.URLError, PrepareError, json.JSONDecodeError, TimeoutError):
            if time.monotonic() >= deadline:
                return None
        time.sleep(0.2)


def _port_from_env(root: Path) -> int:
    if os.environ.get("PUBLIC_PORT", "").strip():
        return int(os.environ["PUBLIC_PORT"])
    env_file = root / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("PUBLIC_PORT="):
                return int(line.split("=", 1)[1].strip().strip("'\""))
    return 8000


def main() -> int:
    root = Path.cwd()
    try:
        return prepare(root, _port_from_env(root))
    except PrepareError as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
