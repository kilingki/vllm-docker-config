"""Profile parsing and vLLM argv construction.

HOST/PORT belong to configs/common.env. A model profile or EXTRA_ARGV_JSON
that overrides them, the model path, or the served model name is rejected.
Flags are emitted only when they appear in the probed CLI allowlist.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


class ConfigError(ValueError):
    pass


OWNERSHIP_FLAGS = ("--host", "--port", "--model", "--served-model-name")
_AUTO = "auto"


def parse_env_file(path: str | Path) -> dict[str, str]:
    values: dict[str, str] = {}
    file_path = Path(path)
    if not file_path.is_file():
        raise ConfigError(f"env file does not exist: {file_path}")
    for lineno, raw in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ConfigError(f"{file_path}:{lineno}: expected KEY=VALUE")
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or not key.replace("_", "").isalnum():
            raise ConfigError(f"{file_path}:{lineno}: invalid key")
        values[key] = _unquote(value.strip())
    return values


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


@dataclass(frozen=True)
class CliAllowlist:
    flags: frozenset[str]
    reasoning_parsers: tuple[str, ...] | None


def load_allowlist(path: str | Path | None) -> CliAllowlist | None:
    if path is None or str(path).strip() == "":
        return None
    file_path = Path(path)
    if not file_path.is_file():
        return None
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"allowlist is not valid JSON: {file_path}") from exc
    flags = payload.get("flags")
    if not isinstance(flags, list) or not all(isinstance(item, str) for item in flags):
        raise ConfigError("allowlist flags must be a list of strings")
    parsers = payload.get("reasoning_parsers", None)
    parsed: tuple[str, ...] | None
    if parsers is None:
        parsed = None
    elif isinstance(parsers, list) and all(isinstance(item, str) for item in parsers):
        parsed = tuple(parsers)
    else:
        raise ConfigError("allowlist reasoning_parsers must be a list of strings or null")
    return CliAllowlist(flags=frozenset(flags), reasoning_parsers=parsed)


@dataclass(frozen=True)
class Settings:
    vllm_bin: str = "vllm"
    load_timeout_sec: float = 600
    unload_timeout_sec: float = 30
    kill_grace_sec: float = 5
    host: str = "127.0.0.1"
    port: int = 8080
    model_path: str = ""
    served_model_name: str = ""
    dtype: str = "bfloat16"
    tensor_parallel_size: str = "1"
    max_model_len: str = "32768"
    max_num_seqs: str = "4"
    max_num_batched_tokens: str = "2048"
    gpu_memory_utilization: str = "0.90"
    kv_cache_dtype: str = ""
    prefix_caching: bool = True
    reasoning_parser: str = ""
    limit_mm_per_prompt: str = ""
    mm_processor_kwargs: str = ""
    quantization: str = ""
    extra_argv: tuple[str, ...] = ()
    max_active_requests: int = 4
    max_body_bytes: int = 33_554_432
    max_output_tokens: int = 4096
    require_explicit_output_limit: bool = True
    health_probe_interval_sec: float = 5
    health_probe_timeout_sec: float = 10
    health_probe_failures: int = 3
    stream_buffer_max_bytes: int = 4_194_304
    allowlist_path: str = ""

    @property
    def vllm_base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def vllm_serve_argv(self, allowlist: CliAllowlist | None = None) -> list[str]:
        loaded = load_allowlist(self.allowlist_path) if allowlist is None else allowlist
        if loaded is None:
            raise ConfigError("vLLM CLI allowlist is not available")
        if not self.model_path:
            raise ConfigError("MODEL_PATH is not set")
        if not self.served_model_name:
            raise ConfigError("SERVED_MODEL_NAME is not set")
        _reject_ownership(self.extra_argv)
        argv = [self.vllm_bin, "serve", self.model_path]
        emitted: list[str] = []

        def add(flag: str, value: str | None = None) -> None:
            if flag not in loaded.flags:
                raise ConfigError(f"unconfirmed or unsupported flag: {flag}")
            if value is not None and value.strip().lower() == _AUTO:
                raise ConfigError(f"refusing to pass {_AUTO!r} for {flag}")
            argv.append(flag)
            emitted.append(flag)
            if value is not None:
                argv.append(value)

        add("--host", self.host)
        add("--port", str(self.port))
        add("--served-model-name", self.served_model_name)
        add("--dtype", self.dtype)
        add("--tensor-parallel-size", self.tensor_parallel_size)
        add("--max-model-len", self.max_model_len)
        add("--max-num-seqs", self.max_num_seqs)
        add("--max-num-batched-tokens", self.max_num_batched_tokens)
        add("--gpu-memory-utilization", self.gpu_memory_utilization)
        if self.kv_cache_dtype.strip() and self.kv_cache_dtype.strip().lower() != _AUTO:
            add("--kv-cache-dtype", self.kv_cache_dtype)
        if self.prefix_caching:
            add("--enable-prefix-caching")
        if self.reasoning_parser:
            if loaded.reasoning_parsers is None:
                raise ConfigError(
                    f"reasoning parser {self.reasoning_parser!r} is unconfirmed for this image"
                )
            if self.reasoning_parser not in loaded.reasoning_parsers:
                raise ConfigError(
                    f"reasoning parser {self.reasoning_parser!r} is not supported by this image"
                )
            add("--reasoning-parser", self.reasoning_parser)
        if self.limit_mm_per_prompt:
            add("--limit-mm-per-prompt", self.limit_mm_per_prompt)
        if self.mm_processor_kwargs:
            add("--mm-processor-kwargs", self.mm_processor_kwargs)
        if self.quantization:
            add("--quantization", self.quantization)

        for token in self.extra_argv:
            flag = token.split("=", 1)[0]
            if flag.startswith("--"):
                if flag in emitted or flag in OWNERSHIP_FLAGS:
                    raise ConfigError(f"extra argv overrides owned flag: {flag}")
                if flag not in loaded.flags:
                    raise ConfigError(f"unconfirmed or unsupported flag: {flag}")
            argv.append(token)
        if _AUTO in argv:
            raise ConfigError(f"refusing to pass {_AUTO!r}")
        return argv


def load_settings(common_path: str | Path, model_path: str | Path) -> Settings:
    common = parse_env_file(common_path)
    model = parse_env_file(model_path)
    forbidden = sorted(key for key in ("HOST", "PORT") if key in model)
    if forbidden:
        raise ConfigError(
            "model env must not set " + ", ".join(forbidden) + "; configs/common.env owns them"
        )
    merged = dict(common)
    merged.update(model)
    return settings_from_mapping(merged)


def settings_from_mapping(values: dict[str, str]) -> Settings:
    def pick(name: str, default: str) -> str:
        raw = values.get(name, default)
        return raw.strip() if raw is not None else default

    extra_raw = pick("EXTRA_ARGV_JSON", "[]")
    try:
        extra = json.loads(extra_raw) if extra_raw else []
    except json.JSONDecodeError as exc:
        raise ConfigError("EXTRA_ARGV_JSON must be a JSON array") from exc
    if not isinstance(extra, list) or not all(isinstance(item, str) for item in extra):
        raise ConfigError("EXTRA_ARGV_JSON must be a JSON array of strings")
    _reject_ownership(extra)

    limit_mm = pick("LIMIT_MM_PER_PROMPT", "")
    if not limit_mm and pick("LIMIT_MM_IMAGE", ""):
        limit_mm = json.dumps({"image": int(pick("LIMIT_MM_IMAGE", "0"))}, separators=(",", ":"))

    return Settings(
        vllm_bin=pick("VLLM_BIN", "vllm"),
        load_timeout_sec=float(pick("LOAD_TIMEOUT_SEC", "600")),
        unload_timeout_sec=float(pick("UNLOAD_TIMEOUT_SEC", "30")),
        kill_grace_sec=float(pick("KILL_GRACE_SEC", "5")),
        host=pick("HOST", "127.0.0.1"),
        port=int(pick("PORT", "8080")),
        model_path=pick("MODEL_PATH", ""),
        served_model_name=pick("SERVED_MODEL_NAME", ""),
        dtype=pick("DTYPE", "bfloat16"),
        tensor_parallel_size=pick("TENSOR_PARALLEL_SIZE", "1"),
        max_model_len=pick("MAX_MODEL_LEN", "32768"),
        max_num_seqs=pick("MAX_NUM_SEQS", "4"),
        max_num_batched_tokens=pick("MAX_NUM_BATCHED_TOKENS", "2048"),
        gpu_memory_utilization=pick("GPU_MEMORY_UTILIZATION", "0.90"),
        kv_cache_dtype=pick("KV_CACHE_DTYPE", ""),
        prefix_caching=_truthy(pick("PREFIX_CACHING", "on")),
        reasoning_parser=pick("REASONING_PARSER", ""),
        limit_mm_per_prompt=limit_mm,
        mm_processor_kwargs=pick("MM_PROCESSOR_KWARGS", ""),
        quantization=pick("QUANTIZATION", ""),
        extra_argv=tuple(extra),
        max_active_requests=int(pick("MAX_ACTIVE_REQUESTS", "4")),
        max_body_bytes=int(pick("MAX_BODY_BYTES", "33554432")),
        max_output_tokens=int(pick("MAX_OUTPUT_TOKENS", "4096")),
        require_explicit_output_limit=_truthy(pick("REQUIRE_EXPLICIT_OUTPUT_LIMIT", "1")),
        health_probe_interval_sec=float(pick("HEALTH_PROBE_INTERVAL_SEC", "5")),
        health_probe_timeout_sec=float(pick("HEALTH_PROBE_TIMEOUT_SEC", "10")),
        health_probe_failures=int(pick("HEALTH_PROBE_FAILURES", "3")),
        stream_buffer_max_bytes=int(pick("STREAM_BUFFER_MAX_BYTES", "4194304")),
        allowlist_path=pick("ALLOWLIST_PATH", ""),
    )


def from_env() -> Settings:
    config_dir = os.environ.get("CONFIG_DIR", "").strip()
    if config_dir:
        profile = os.environ.get("MODEL_PROFILE", "qwen3.8-27b").strip() or "qwen3.8-27b"
        root = Path(config_dir)
        merged = dict(parse_env_file(root / "common.env"))
        merged.update(parse_env_file(root / "models" / f"{profile}.env"))
        # Compose environment overrides the profile file, matching process env as the runtime source.
        for key, value in os.environ.items():
            if key in merged:
                merged[key] = value
        return settings_from_mapping(merged)
    return settings_from_mapping(dict(os.environ))


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _reject_ownership(argv: list[str] | tuple[str, ...]) -> None:
    for token in argv:
        flag = token.split("=", 1)[0]
        if flag in OWNERSHIP_FLAGS:
            raise ConfigError(f"extra argv must not set {flag}")
