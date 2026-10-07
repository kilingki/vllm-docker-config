import json
from pathlib import Path

import pytest

from tests.results import utc_now, write_result

_STARTED = utc_now()
_CASES: dict[str, list[dict[str, str]]] = {
    "cpu_control": [],
    "cpu_config": [],
    "cpu_prepare": [],
}
_SUITES = {
    "test_cpu_control.py": "cpu_control",
    "test_cpu_config.py": "cpu_config",
    "test_cpu_prepare.py": "cpu_prepare",
}
_OUTPUT = Path(__file__).resolve().parent / "outputs"


def _suite_for(nodeid: str) -> str | None:
    for filename, suite in _SUITES.items():
        if filename in nodeid:
            return suite
    return None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    suite = _suite_for(item.nodeid)
    if suite is None:
        return
    if report.when != "call" and not (report.when == "setup" and report.failed):
        return
    if report.passed:
        status = "passed"
        detail = "ok"
    elif report.skipped:
        status = "skipped"
        detail = str(report.longrepr)
    else:
        status = "failed"
        detail = str(report.longrepr)[-4000:]
    case_id = item.name
    _CASES[suite] = [case for case in _CASES[suite] if case["id"] != case_id]
    _CASES[suite].append({"id": case_id, "status": status, "detail": detail})


def pytest_sessionfinish(session, exitstatus):
    finished = utc_now()
    for suite, cases in _CASES.items():
        if not cases and exitstatus == 0:
            continue
        write_result(_OUTPUT / f"{suite}.json", suite, cases, _STARTED, finished)
