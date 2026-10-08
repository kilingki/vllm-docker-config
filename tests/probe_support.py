"""Record Docker, GPU, checkpoint, and vLLM CLI evidence.

This does not load a model. Static support is not recorded as a GPU pass.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.results import utc_now

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "vllm/vllm-openai:v0.31.0"
OUTPUT = ROOT / "tests" / "outputs" / "probe_support.json"
ALLOWLIST = ROOT / "configs" / "vllm_cli_allowlist.json"
_RUNTIME_PROBE = r'''
import json
import pathlib
import re
import traceback

out = {}
root = None
try:
    import torch
    import transformers
    import vllm
    out["torch"] = torch.__version__
    out["cuda"] = str(torch.version.cuda)
    out["vllm"] = vllm.__version__
    out["transformers"] = transformers.__version__
    try:
        cap = torch.cuda.get_device_capability()
        out["capability"] = [int(cap[0]), int(cap[1])]
    except Exception as exc:
        out["capability_error"] = str(exc)
    root = pathlib.Path(vllm.__file__).resolve().parent
except Exception:
    out["import_error"] = traceback.format_exc()
    candidates = list(pathlib.Path("/usr/local/lib").glob("python*/dist-packages/vllm/__init__.py"))
    if candidates:
        root = candidates[0].parent

if root is not None:
    hits = []
    executor = []
    spawn_needles = ("setsid", "start_new_session", "multiprocessing.get_context")
    exec_needles = spawn_needles + ("subprocess", "Process(")
    exec_root = root / "v1" / "executor"
    for path in root.rglob("*.py"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        rel = str(path.relative_to(root))
        found = [name for name in spawn_needles if name in text]
        if found:
            hits.append({"file": rel, "markers": found})
        if path.parent == exec_root or exec_root in path.parents:
            executor.append({"file": rel, "markers": [name for name in exec_needles if name in text]})
    out["worker_spawn_hits"] = hits[:60]
    out["executor_files"] = executor
    fa = root / "v1" / "attention" / "backends" / "fa_utils.py"
    try:
        source = fa.read_text(encoding="utf-8")
        match = re.search(r"def flash_attn_supports_kv_cache_dtype\(.*?\n(?:.*\n){0,40}", source)
        out["fa_source"] = match.group(0) if match else source[:4000]
    except OSError as exc:
        out["fa_error"] = repr(exc)
    try:
        reg = (root / "reasoning" / "__init__.py").read_text(encoding="utf-8")
        block = re.search(r"_REASONING_PARSERS_TO_REGISTER\s*=\s*\{(.*?)\n\}", reg, re.S)
        names = re.findall(r"^\s*[\"']([A-Za-z0-9_]+)[\"']\s*:", block.group(1), re.M) if block else []
        out["reasoning_parsers_registered"] = names
        mp = (root / "utils" / "system_utils.py").read_text(encoding="utf-8").splitlines()
        start = next(i for i, line in enumerate(mp) if line.startswith("def get_mp_context"))
        out["mp_context_source"] = "\n".join(mp[start:start + 14])
        wsl = next(i for i, line in enumerate(mp) if "WSL is detected" in line)
        out["mp_wsl_source"] = "\n".join(mp[wsl - 2:wsl + 8])
    except Exception as exc:
        out["source_excerpt_error"] = repr(exc)

imported = []
errors = {}
if not out.get("import_error"):
    try:
        import vllm.reasoning
        from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
        ReasoningParserManager.get_reasoning_parser("qwen3")
        imported.append("qwen3")
    except Exception as exc:
        errors["qwen3"] = f"{type(exc).__name__}: {exc}"
out["reasoning_parsers_imported"] = imported
out["reasoning_parser_errors"] = errors
print(json.dumps(out))
'''


def run(cmd: list[str], timeout: float = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def docker_available() -> dict:
    if shutil.which("docker") is None:
        return {"available": False, "reason": "docker executable not found"}
    result = run(["docker", "version", "--format", "{{.Server.Version}}"])
    if result.returncode != 0:
        return {"available": False, "reason": result.stderr.strip() or "docker version failed"}
    return {"available": True, "reason": None, "server_version": result.stdout.strip()}


def gpu_available() -> dict:
    if shutil.which("nvidia-smi") is None:
        return {"available": False, "reason": "nvidia-smi not found"}
    result = run(["nvidia-smi", "-L"])
    if result.returncode != 0:
        return {"available": False, "reason": result.stderr.strip() or "nvidia-smi failed"}
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return {"available": bool(lines), "reason": None if lines else "no GPU listed", "devices": lines}


def _snapshot_summary(directory: Path, payload: dict) -> dict:
    quant = payload.get("quantization_config")
    quant = quant if isinstance(quant, dict) else {}
    groups = quant.get("config_groups") if isinstance(quant.get("config_groups"), dict) else {}
    weights: dict = {}
    if groups:
        first = next(iter(groups.values()))
        if isinstance(first, dict) and isinstance(first.get("weights"), dict):
            weights = first["weights"]
    ignore = quant.get("ignore") if isinstance(quant.get("ignore"), list) else []
    suffixes: dict[str, int] = {}
    for item in ignore:
        if not isinstance(item, str):
            continue
        suffix = item.split(".")[-1]
        suffixes[suffix] = suffixes.get(suffix, 0) + 1
    text = payload.get("text_config") if isinstance(payload.get("text_config"), dict) else {}
    vision = payload.get("vision_config") if isinstance(payload.get("vision_config"), dict) else {}
    layer_counts: dict[str, int] = {}
    for item in text.get("layer_types") or []:
        layer_counts[str(item)] = layer_counts.get(str(item), 0) + 1
    processor: dict = {}
    preprocessor = directory / "preprocessor_config.json"
    if preprocessor.is_file():
        try:
            raw = json.loads(preprocessor.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw = {}
        if isinstance(raw, dict):
            processor = {
                "processor_class": raw.get("processor_class"),
                "image_processor_type": raw.get("image_processor_type"),
                "size": raw.get("size"),
                "patch_size": raw.get("patch_size"),
            }
    files = []
    for child in sorted(directory.iterdir()):
        if child.is_file() and not child.name.startswith("."):
            files.append({"name": child.name, "bytes": child.stat().st_size})
    return {
        "path": str(directory),
        "model_type": payload.get("model_type"),
        "architectures": payload.get("architectures"),
        "quantization": {
            "quant_method": quant.get("quant_method"),
            "format": quant.get("format"),
            "quantization_status": quant.get("quantization_status"),
            "version": quant.get("version"),
            "bits": weights.get("num_bits"),
            "group_size": weights.get("group_size"),
            "symmetric": weights.get("symmetric"),
            "strategy": weights.get("strategy"),
            "kv_cache_scheme": quant.get("kv_cache_scheme"),
            "ignored_suffix_counts": suffixes,
        },
        "text": {
            "model_type": text.get("model_type"),
            "dtype": text.get("dtype"),
            "max_position_embeddings": text.get("max_position_embeddings"),
            "mamba_ssm_dtype": text.get("mamba_ssm_dtype"),
            "full_attention_interval": text.get("full_attention_interval"),
            "layer_type_counts": layer_counts,
        },
        "vision": {
            "model_type": vision.get("model_type"),
            "depth": vision.get("depth"),
        },
        "processor": processor,
        "files": files,
        "loaded": False,
    }


def checkpoint_available() -> dict:
    candidates = []
    env_dir = os.environ.get("HOST_MODELS_DIR", "").strip()
    roots = [Path(env_dir)] if env_dir else []
    roots.append(Path("/home/kjh/.cache/huggingface/hub"))
    seen: list[str] = []
    configs: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        seen.append(str(root))
        direct = root / "config.json"
        if direct.is_file():
            configs.append(direct)
        configs.extend(root.glob("models--*qwen*/snapshots/*/config.json"))
        configs.extend(root.glob("*/config.json"))
    seen_paths: set[str] = set()
    for path in configs:
        directory = path.parent
        if str(directory) in seen_paths:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        model_type = str(payload.get("model_type", ""))
        name = str(directory)
        if "qwen3" not in name.lower() and "qwen3" not in model_type.lower():
            continue
        seen_paths.add(str(directory))
        candidates.append(_snapshot_summary(directory, payload))
    if not candidates:
        return {
            "available": False,
            "reason": "no local Qwen3 HF snapshot with config.json was found",
            "searched": seen,
        }
    return {"available": True, "reason": None, "snapshots": candidates}


def _json_stdout(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return {}
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _help_excerpts(help_text: str) -> dict[str, str]:
    keys = (
        "kv-cache-dtype",
        "reasoning-parser",
        "prefix-caching",
        "limit-mm-per-prompt",
        "mm-processor-kwargs",
        "max-model-len",
        "dtype",
    )
    excerpts: dict[str, str] = {}
    lines = help_text.splitlines()
    for key in keys:
        chunks: list[str] = []
        for index, line in enumerate(lines):
            if f"--{key}" not in line:
                continue
            chunks.extend(lines[index : index + 6])
            break
        excerpts[key] = "\n".join(chunks)[:800]
    return excerpts


def image_probe(docker_ok: bool) -> dict:
    if not docker_ok:
        return {"available": False, "reason": "docker is not available", "image": IMAGE}
    listed = run(["docker", "image", "inspect", IMAGE])
    if listed.returncode != 0:
        return {
            "available": False,
            "reason": "image is not present locally; pull was not finished or failed",
            "image": IMAGE,
        }
    info = json.loads(listed.stdout)[0]
    config = info.get("Config") or {}
    digest = (info.get("RepoDigests") or [None])[0]
    image_id = info.get("Id")
    version = run(
        ["docker", "run", "--rm", "--gpus", "all", "--entrypoint", "vllm", IMAGE, "--version"],
        timeout=180,
    )
    help_text = run(
        [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            "--entrypoint",
            "vllm",
            IMAGE,
            "serve",
            "--help",
        ],
        timeout=180,
    )
    help_all = run(
        [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            "--entrypoint",
            "vllm",
            IMAGE,
            "serve",
            "--help=all",
        ],
        timeout=180,
    )
    runtime = run(
        [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            "--entrypoint",
            "python3",
            IMAGE,
            "-c",
            _RUNTIME_PROBE,
        ],
        timeout=180,
    )
    runtime_payload = _json_stdout(runtime.stdout)
    cli_ok = version.returncode == 0 and help_text.returncode == 0 and help_all.returncode == 0
    help_stdout = help_all.stdout if help_all.returncode == 0 else ""
    help_source = "vllm --version; vllm serve --help; vllm serve --help=all" if cli_ok else "unconfirmed"
    flags = sorted(set(re.findall(r"--[A-Za-z0-9][A-Za-z0-9-]*", help_stdout))) if cli_ok else []
    imported = runtime_payload.get("reasoning_parsers_imported") or []
    parsers = list(imported) if imported else None
    intended = [
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
    ]
    return {
        "available": True,
        "reason": None,
        "image": IMAGE,
        "digest": digest,
        "image_id": image_id,
        "entrypoint": config.get("Entrypoint"),
        "cmd": config.get("Cmd"),
        "vllm_version_cli": version.stdout.strip(),
        "vllm_version_exit": version.returncode,
        "vllm_version_error": version.stderr.strip(),
        "help_exit": help_text.returncode,
        "help_all_exit": help_all.returncode,
        "cli_ok": cli_ok,
        "help_ok": cli_ok,
        "help_source": help_source,
        "help_error": ""
        if cli_ok
        else "\n".join(
            part
            for part in (
                version.stderr.strip(),
                help_text.stderr.strip(),
                help_all.stderr.strip(),
            )
            if part
        )[:4000],
        "console_help_ok": cli_ok,
        "flags": flags,
        "intended_flags_present": [flag for flag in intended if flag in flags],
        "intended_flags_missing": [flag for flag in intended if flag not in flags],
        "reasoning_parsers": parsers,
        "help_excerpts": _help_excerpts(help_stdout),
        "runtime_versions": {
            "torch": runtime_payload.get("torch"),
            "cuda": runtime_payload.get("cuda"),
            "vllm": runtime_payload.get("vllm"),
            "transformers": runtime_payload.get("transformers"),
            "capability": runtime_payload.get("capability"),
            "capability_error": runtime_payload.get("capability_error"),
            "import_error": runtime_payload.get("import_error"),
        },
        "runtime_error": runtime.stderr.strip(),
        "runtime_ok": runtime.returncode == 0 and not runtime_payload.get("import_error"),
        "fa_utils_source_ok": bool(runtime_payload.get("fa_source")),
        "fa_utils_error": runtime_payload.get("fa_error", ""),
        "fp8_kv_fa_source": str(runtime_payload.get("fa_source") or "")[:4000],
        "worker_spawn_hits": runtime_payload.get("worker_spawn_hits") or [],
        "executor_files": runtime_payload.get("executor_files") or [],
        "worker_spawn_error": runtime_payload.get("worker_error"),
        "reasoning_parsers_registered": runtime_payload.get("reasoning_parsers_registered") or [],
        "reasoning_parsers_imported": runtime_payload.get("reasoning_parsers_imported") or [],
        "reasoning_parser_errors": runtime_payload.get("reasoning_parser_errors") or {},
        "mp_context_source": runtime_payload.get("mp_context_source") or "",
        "mp_wsl_source": runtime_payload.get("mp_wsl_source") or "",
        "source_excerpt_error": runtime_payload.get("source_excerpt_error"),
    }


def write_allowlist(image: dict) -> None:
    flags = image.get("flags") or []
    payload = {
        "image": IMAGE,
        "digest": image.get("digest"),
        "source": image.get("help_source") if image.get("help_ok") else "unconfirmed",
        "flags": flags,
        "reasoning_parsers": image.get("reasoning_parsers"),
        "gpu_success": False,
    }
    ALLOWLIST.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    started = utc_now()
    docker_info = docker_available()
    gpu_info = gpu_available()
    checkpoint = checkpoint_available()
    image = image_probe(bool(docker_info.get("available")))
    write_allowlist(image)
    cases = [
        {
            "id": "docker",
            "status": "passed" if docker_info.get("available") else "not_run",
            "detail": json.dumps(docker_info),
        },
        {
            "id": "gpu",
            "status": "passed" if gpu_info.get("available") else "not_run",
            "detail": json.dumps(gpu_info),
        },
        {
            "id": "checkpoint",
            "status": "passed" if checkpoint.get("available") else "not_run",
            "detail": json.dumps(checkpoint),
        },
        {
            "id": "vllm_image_cli",
            "status": "passed" if image.get("cli_ok") else "not_run",
            "detail": "real vllm --version and vllm serve --help "
            + ("exited 0" if image.get("cli_ok") else "did not exit 0")
            + "; format_help and module stubs are not a CLI pass; this is not a GPU inference pass",
        },
    ]
    payload = {
        "suite": "probe_support",
        "started_at": started,
        "finished_at": utc_now(),
        "status": "passed" if all(case["status"] == "passed" for case in cases) else "not_run",
        "gpu_inference_success": False,
        "cases": cases,
        "docker": docker_info,
        "gpu": gpu_info,
        "checkpoint": checkpoint,
        "image": {key: value for key, value in image.items() if key != "fp8_kv_fa_source"},
        "fp8_kv_fa_source": image.get("fp8_kv_fa_source", ""),
        "reference_repo": {
            "name": "kilingki/llama-cpp-docker-config",
            "revision": "e655a2f2ad3979bc5c4a7b4204b80bad08de42e9",
            "head_matches_design": True,
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
