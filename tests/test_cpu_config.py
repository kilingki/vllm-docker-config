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


def test_gpu_script_check_plan_does_not_claim_gpu_success():
    import subprocess
    import sys

    script = ROOT / "tests" / "gpu_api.py"
    completed = subprocess.run([sys.executable, str(script), "--check-plan"], check=False, text=True)
    assert completed.returncode == 0, completed.stderr
    payload = json.loads((ROOT / "tests" / "outputs" / "gpu_api.json").read_text(encoding="utf-8"))
    assert payload["status"] == "not_run"
    assert payload["cases"]
    assert all(case["status"] == "not_run" for case in payload["cases"])
