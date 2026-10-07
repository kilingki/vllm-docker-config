import json
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def suite_status(cases: list[dict[str, str]]) -> str:
    statuses = [case["status"] for case in cases]
    if any(status == "failed" for status in statuses):
        return "failed"
    if statuses and all(status == "not_run" for status in statuses):
        return "not_run"
    if statuses and all(status == "skipped" for status in statuses):
        return "skipped"
    return "passed"


def write_result(
    path: Path,
    suite: str,
    cases: list[dict[str, str]],
    started_at: str,
    finished_at: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "suite": suite,
        "started_at": started_at,
        "finished_at": finished_at,
        "status": suite_status(cases),
        "cases": cases,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
