"""Executable stage-2 GPU/API checks.

Running this file records not_run. It does not load a model or claim a GPU pass.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.results import utc_now, write_result

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


def check_plan() -> None:
    required = [case["id"] for case in CASES]
    if len(required) != len(set(required)):
        raise SystemExit("duplicate gpu case id")
    missing = [
        case_id
        for case_id in (
            "container_startup",
            "load",
            "text_nonstream",
            "text_stream",
            "image_single",
            "image_multi",
            "backend_record",
            "concurrency_1",
            "concurrency_2",
            "concurrency_4",
            "long_context",
            "multimodal_max",
            "disconnect_busy",
            "status_during_lifecycle",
            "repeat_lifecycle",
            "failure_bad_model",
            "failure_load_timeout",
            "failure_worker_crash",
            "container_restart",
        )
        if case_id not in required
    ]
    if missing:
        raise SystemExit(f"gpu plan missing cases: {missing}")
    for case in CASES:
        if not case["pass_condition"].strip():
            raise SystemExit(f"case {case['id']} has no pass condition")


def write_not_run(reason: str) -> None:
    now = utc_now()
    write_result(
        Path(__file__).resolve().parent / "outputs" / "gpu_api.json",
        "gpu_api",
        [
            {
                "id": case["id"],
                "status": "not_run",
                "detail": f"{reason} Pass condition: {case['pass_condition']}",
            }
            for case in CASES
        ],
        now,
        now,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Independent GPU/API verification")
    parser.add_argument("--check-plan", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Reserved for stage 2. Stage 1 refuses this switch.",
    )
    args = parser.parse_args(argv)
    if args.check_plan:
        check_plan()
    if args.execute:
        print("GPU execution belongs to stage 2 and was not run", file=sys.stderr)
        write_not_run("stage 2 execution was requested during stage 1 and refused")
        return 2
    write_not_run("stage 1 prepares this script and does not execute GPU load or inference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
