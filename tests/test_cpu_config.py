import json
from pathlib import Path

import pytest

from controller.config import CliAllowlist, ConfigError, load_settings, settings_from_mapping

pytestmark = pytest.mark.cpu_config

ROOT = Path(__file__).resolve().parents[1]
FLAGS = (
    "--host",
    "--port",
    "--served-model-name",
    "--dtype",
    "--tensor-parallel-size",
    "--max-model-len",
    "--max-num-seqs",
    "--max-num-batched-tokens",
    "--gpu-memory-utilization",
    "--kv-cache-dtype",
    "--enable-prefix-caching",
    "--reasoning-parser",
    "--limit-mm-per-prompt",
    "--mm-processor-kwargs",
    "--quantization",
    "--max-logprobs",
)


def allowlist(parsers=("qwen3",)):
    return CliAllowlist(frozenset(FLAGS), parsers)


def test_model_env_cannot_override_host_or_port(tmp_path: Path):
    common = tmp_path / "common.env"
    model = tmp_path / "model.env"
    common.write_text("HOST=127.0.0.1\nPORT=8080\n", encoding="utf-8")
    model.write_text("HOST=10.0.0.1\nMODEL_PATH=/models/x\nSERVED_MODEL_NAME=m\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="HOST"):
        load_settings(common, model)


def test_extra_argv_rejects_ownership_and_non_arrays():
    with pytest.raises(ConfigError, match="--host"):
        settings_from_mapping({"EXTRA_ARGV_JSON": '["--host", "0.0.0.0"]'})
    with pytest.raises(ConfigError, match="--port"):
        settings_from_mapping({"EXTRA_ARGV_JSON": '["--port=9"]'})
    with pytest.raises(ConfigError, match="--served-model-name"):
        settings_from_mapping({"EXTRA_ARGV_JSON": '["--served-model-name", "other"]'})
    with pytest.raises(ConfigError, match="JSON array"):
        settings_from_mapping({"EXTRA_ARGV_JSON": '{"flag": "--host"}'})
    with pytest.raises(ConfigError, match="JSON array"):
        settings_from_mapping({"EXTRA_ARGV_JSON": '"--host"'})


def test_profile_argv_uses_only_allowlisted_flags():
    settings = load_settings(ROOT / "configs/common.env", ROOT / "configs/models/qwen3.8-27b.env")
    argv = settings.vllm_serve_argv(allowlist())
    assert argv[:3] == ["vllm", "serve", "/models/qwen3.8-27b"]
    assert "--kv-cache-dtype" in argv
    assert "fp8_e4m3" in argv
    assert "--enable-prefix-caching" in argv
    assert "--reasoning-parser" in argv
    assert "qwen3" in argv
    assert "--limit-mm-per-prompt" in argv
    assert "--quantization" not in argv
    assert "auto" not in argv
    assert "--host" in argv and "127.0.0.1" in argv
    assert "--max-model-len" in argv and "32768" in argv
    assert "--max-num-seqs" in argv and "4" in argv
    joined = json.dumps(argv)
    assert "262144" in joined
    with pytest.raises(ConfigError, match="unconfirmed"):
        settings.vllm_serve_argv(CliAllowlist(frozenset({"--host"}), ("qwen3",)))


def test_repository_allowlist_builds_both_profile_argvs():
    default = load_settings(ROOT / "configs/common.env", ROOT / "configs/models/qwen3.8-27b.env")
    comparison = load_settings(
        ROOT / "configs/common.env",
        ROOT / "configs/models/qwen3.8-27b-kv-bf16.env",
    )
    default_argv = default.vllm_serve_argv()
    comparison_argv = comparison.vllm_serve_argv()
    assert default_argv[:3] == ["vllm", "serve", "/models/qwen3.8-27b"]
    assert "--reasoning-parser" in default_argv and "qwen3" in default_argv
    assert "--kv-cache-dtype" in default_argv and "fp8_e4m3" in default_argv
    assert "--kv-cache-dtype" not in comparison_argv
    assert "--reasoning-parser" in comparison_argv and "qwen3" in comparison_argv
    assert "auto" not in default_argv and "auto" not in comparison_argv


def test_unconfirmed_reasoning_parser_is_rejected():
    settings = load_settings(ROOT / "configs/common.env", ROOT / "configs/models/qwen3.8-27b.env")
    with pytest.raises(ConfigError, match="unconfirmed"):
        settings.vllm_serve_argv(CliAllowlist(frozenset(FLAGS), None))
    with pytest.raises(ConfigError, match="not supported"):
        settings.vllm_serve_argv(CliAllowlist(frozenset(FLAGS), ("other",)))


def test_missing_allowlist_fails_before_launch():
    settings = load_settings(ROOT / "configs/common.env", ROOT / "configs/models/qwen3.8-27b.env")
    settings = settings_from_mapping(
        {
            "MODEL_PATH": "/models/qwen3.8-27b",
            "SERVED_MODEL_NAME": "qwen3.8-27b",
            "ALLOWLIST_PATH": "/tmp/does-not-exist-vllm-allowlist.json",
        }
    )
    with pytest.raises(ConfigError, match="allowlist is not available"):
        settings.vllm_serve_argv()


def test_comparison_profile_is_not_default_and_omits_fp8():
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "MODEL_PROFILE=qwen3.8-27b\n" in example
    assert "qwen3.8-27b-kv-bf16" not in example
    comparison = load_settings(
        ROOT / "configs/common.env",
        ROOT / "configs/models/qwen3.8-27b-kv-bf16.env",
    )
    assert comparison.kv_cache_dtype == ""
    argv = comparison.vllm_serve_argv(allowlist())
    assert "--kv-cache-dtype" not in argv
    assert "auto" not in argv
    assert "--dtype" in argv and "bfloat16" in argv


def test_common_limits_and_model_profile_do_not_own_bind():
    common = (ROOT / "configs/common.env").read_text(encoding="utf-8")
    assert "LOAD_TIMEOUT_SEC=600" in common
    assert "UNLOAD_TIMEOUT_SEC=30" in common
    assert "MAX_ACTIVE_REQUESTS=4" in common
    assert "MAX_BODY_BYTES=33554432" in common
    assert "MAX_OUTPUT_TOKENS=4096" in common
    assert "REQUIRE_EXPLICIT_OUTPUT_LIMIT=1" in common
    model = (ROOT / "configs/models/qwen3.8-27b.env").read_text(encoding="utf-8")
    for line in model.splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        assert not line.startswith("HOST=")
        assert not line.startswith("PORT=")
    settings = load_settings(ROOT / "configs/common.env", ROOT / "configs/models/qwen3.8-27b.env")
    assert settings.max_body_bytes == 33_554_432
    assert settings.max_active_requests == 4
    assert settings.require_explicit_output_limit is True
    assert settings.stream_buffer_max_bytes == 4_194_304


def test_model_path_with_spaces_is_one_argument():
    settings = settings_from_mapping(
        {
            "MODEL_PATH": "/models/my snapshot",
            "SERVED_MODEL_NAME": "qwen3.8-27b",
            "KV_CACHE_DTYPE": "",
            "PREFIX_CACHING": "off",
            "REASONING_PARSER": "",
            "LIMIT_MM_PER_PROMPT": "",
            "MM_PROCESSOR_KWARGS": "",
        }
    )
    argv = settings.vllm_serve_argv(
        CliAllowlist(
            frozenset(
                flag
                for flag in FLAGS
                if flag
                not in {
                    "--kv-cache-dtype",
                    "--enable-prefix-caching",
                    "--reasoning-parser",
                    "--limit-mm-per-prompt",
                    "--mm-processor-kwargs",
                    "--quantization",
                }
            ),
            (),
        )
    )
    assert argv[2] == "/models/my snapshot"
    assert isinstance(argv, list)


def test_from_env_process_environment_overrides_profile_file(tmp_path, monkeypatch):
    from controller.config import from_env

    configs = tmp_path / "configs"
    models = configs / "models"
    models.mkdir(parents=True)
    (configs / "common.env").write_text("LOAD_TIMEOUT_SEC=600\nHOST=127.0.0.1\nPORT=8080\n", encoding="utf-8")
    (models / "qwen3.8-27b.env").write_text(
        "MODEL_PATH=/models/real\nSERVED_MODEL_NAME=qwen3.8-27b\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CONFIG_DIR", str(configs))
    monkeypatch.setenv("MODEL_PROFILE", "qwen3.8-27b")
    monkeypatch.setenv("MODEL_PATH", "/models/does-not-exist-for-failure-check")
    monkeypatch.setenv("LOAD_TIMEOUT_SEC", "1")
    settings = from_env()
    assert settings.model_path == "/models/does-not-exist-for-failure-check"
    assert settings.load_timeout_sec == 1.0


def test_gpu_script_check_plan_does_not_claim_gpu_success():
    import subprocess
    import sys

    script = ROOT / "tests" / "gpu_api.py"
    output = ROOT / "tests" / "outputs" / "gpu_api.json"
    original = output.read_text(encoding="utf-8")
    output.write_text('{"suite":"gpu_api","status":"sentinel"}\n', encoding="utf-8")
    try:
        completed = subprocess.run(
            [sys.executable, str(script), "--check-plan"],
            check=False,
            text=True,
            capture_output=True,
        )
        assert completed.returncode == 0, completed.stderr
        bare = subprocess.run(
            [sys.executable, str(script)],
            check=False,
            text=True,
            capture_output=True,
        )
        assert bare.returncode == 0, bare.stderr
        assert "not executed" in bare.stderr
        assert output.read_text(encoding="utf-8") == '{"suite":"gpu_api","status":"sentinel"}\n'
        source = script.read_text(encoding="utf-8")
        assert "stage 2 execution was requested" not in source
        assert "def execute_suite" in source
    finally:
        output.write_text(original, encoding="utf-8")


def test_gpu_requests_and_evidence_rules():
    from tests.gpu_api import (
        RunConfig,
        build_chat_request,
        concurrency_bodies,
        context_approached,
        evidence_from_text,
        judge_case,
        pixels_hit_cap,
        quality_should_retry,
        sse_event_has_token,
        table_numbers_in_content,
    )

    text = build_chat_request("qwen3.8-27b", kind="text", max_tokens=32)
    image = build_chat_request("qwen3.8-27b", kind="image", images=1, text="Describe the image.")
    assert b"image_url" not in text
    assert b'"max_tokens":32' in text
    assert b"image_url" in image
    assert b"data:image/png;base64," in image
    config = RunConfig(base_url="http://127.0.0.1:9")
    mixed = concurrency_bodies(config, 4, "mixed")
    assert len(mixed) == 4
    assert sum(b"image_url" in body for body in mixed) == 2
    status, detail = judge_case("backend_record", http_ok=True, evidence={})
    assert status == "failed"
    assert "not backend evidence" in detail
    status, detail = judge_case("concurrency_4", http_ok=True, evidence={})
    assert status == "failed"
    assert "batching" in detail
    logs = "selected marlin kernel kv_cache_dtype=fp8_e4m3 k_scale=1.0 attention backend flashinfer vision attention sdpa mamba cache dtype float32"
    found = evidence_from_text(logs)
    status, _detail = judge_case("backend_record", http_ok=True, evidence=found)
    assert status == "passed"
    kernel_only = (
        "Using Triton/FLA GDN prefill kernel. GDN decode kernel: cuda. "
        "Mamba cache mode is set to align. Warmed Mamba batch_memcpy_kernel."
    )
    kernel_found = evidence_from_text(kernel_only)
    assert "gdn_dtype" not in kernel_found
    status, detail = judge_case("backend_record", http_ok=True, evidence=kernel_found)
    assert status == "failed"
    assert "gdn_dtype" in detail
    assert table_numbers_in_content("1, 2, 3, 4")
    assert table_numbers_in_content("the numbers are 1 2 3 4.")
    assert not table_numbers_in_content("12, 3, 4")
    assert not table_numbers_in_content(None)
    assert quality_should_retry(None, "length")
    assert quality_should_retry("  ", "length")
    assert not quality_should_retry("1, 2, 3, 4", "length")
    assert not quality_should_retry(None, "stop")
    role_event = 'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}'
    token_event = 'data: {"choices":[{"delta":{"reasoning":"The"}}]}'
    assert not sse_event_has_token(role_event)
    assert sse_event_has_token(token_event)
    assert pixels_hit_cap(768 * 768, 262144)
    assert not pixels_hit_cap(56 * 26, 56 * 26)
    status, detail = judge_case("quality_image", http_ok=True, evidence={})
    assert status == "failed"
    assert "1, 2, 3, 4" in detail
    status, detail = judge_case("multimodal_max", http_ok=True, evidence={})
    assert status == "failed"
    assert "max condition incomplete" in detail
    assert context_approached({"prompt_tokens": 32000, "completion_tokens": 16}, 64) is True
    assert context_approached({"prompt_tokens": 10, "completion_tokens": 2}, 64) is False
    status, detail = judge_case(
        "long_context",
        http_ok=True,
        evidence={"preemption": "log", "context_tokens": "usage"},
    )
    assert status == "failed"
    assert "preemption" in detail
