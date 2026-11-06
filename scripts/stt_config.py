"""Shared local/API STT configuration. Persist environment-variable names, never keys."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import re
from urllib.parse import urlsplit


@dataclass
class STTConfig:
    backend: str = "local"
    local: dict = field(default_factory=lambda: {"model": "small", "device": "auto", "compute_type": "int8"})
    api: dict = field(default_factory=lambda: {
        "base_url": "", "model": "", "api_key_env": "STT_API_KEY",
        "response_format": "verbose_json", "timeout": 120.0,
        "chunk_seconds": 300.0, "max_upload_mb": 20.0,
    })

    def endpoint(self) -> str:
        base = self.api["base_url"].rstrip("/")
        return base if base.endswith("/audio/transcriptions") else base + "/audio/transcriptions"

    def api_key(self) -> str:
        return os.environ.get(self.api["api_key_env"], "").strip()

    def validate_api(self) -> None:
        parsed = urlsplit(self.api["base_url"])
        if (parsed.scheme not in ("https", "http") or not parsed.hostname or
                parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("STT API base_url 需为 http(s) 地址，不含账号密码、查询参数或片段")
        # Validate malformed ports without echoing a possibly sensitive URL.
        try:
            parsed.port
        except ValueError:
            raise ValueError("STT API base_url 端口无效") from None
        if not self.api["model"].strip():
            raise ValueError("请配置 STT_API_MODEL 或 api.model")
        if not self.api_key() and parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError(f"缺少密钥环境变量 {self.api['api_key_env']}；请在运行环境设置，不写入配置文件")

    def public_summary(self) -> dict:
        if self.backend == "local":
            return {"backend": "local", **self.local}
        self.validate_api()
        return {"backend": "api", **self.api, "api_key_present": bool(self.api_key())}


def load_stt_config(config_file=None, backend=None, api_options=None, local_options=None) -> STTConfig:
    """Explicit arguments > STT_* environment > JSON > built-in defaults."""
    config = STTConfig()
    configured_path = config_file or os.environ.get("STT_CONFIG_FILE")
    path = Path(configured_path).expanduser() if configured_path else Path(__file__).resolve().parents[1] / "stt.local.json"
    if configured_path or path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            raise ValueError("STT 配置文件不存在、不可读或不是合法 JSON") from None
        if not isinstance(data, dict) or set(data) - {"backend", "local", "api"}:
            raise ValueError("STT 配置仅支持 backend、local 和 api；密钥需通过环境变量提供")
        config.backend = data.get("backend", config.backend)
        for section in ("local", "api"):
            values = data.get(section, {})
            defaults = getattr(config, section)
            if not isinstance(values, dict) or set(values) - set(defaults):
                raise ValueError(f"STT {section} 配置含未知字段；API 密钥只配置 api_key_env")
            defaults.update(values)
    config.backend = backend or os.environ.get("STT_BACKEND", config.backend)
    for section, options in (("local", local_options or {}), ("api", api_options or {})):
        values = getattr(config, section)
        if set(options) - set(values):
            raise ValueError(f"未知 STT {section} 参数")
        for key in values:
            env_name = "STT_API_KEY_ENV" if section == "api" and key == "api_key_env" else f"STT_{section.upper()}_{key.upper()}"
            env = os.environ.get(env_name)
            if env is not None:
                values[key] = env
            if options.get(key) is not None:
                values[key] = options[key]
    if config.backend not in ("local", "api"):
        raise ValueError("STT backend 仅支持 local 或 api")
    for name in ("model", "device", "compute_type"):
        if not isinstance(config.local[name], str) or not config.local[name].strip():
            raise ValueError(f"local.{name} 必须为非空字符串")
    for name in ("base_url", "model", "api_key_env", "response_format"):
        if not isinstance(config.api[name], str) or any(ord(c) < 32 for c in config.api[name]):
            raise ValueError(f"api.{name} 必须为不含控制字符的字符串")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", config.api["api_key_env"]):
        raise ValueError("api_key_env 必须是有效环境变量名")
    if config.api["response_format"] not in ("verbose_json", "json", "text", "diarized_json"):
        raise ValueError("API response_format 仅支持 verbose_json/json/text/diarized_json")
    for name, upper in (("timeout", 3600), ("chunk_seconds", 3600), ("max_upload_mb", 128)):
        try:
            value = float(config.api[name])
        except (TypeError, ValueError):
            raise ValueError(f"api.{name} 必须为有限正数") from None
        if isinstance(config.api[name], bool) or not math.isfinite(value) or not 0 < value <= upper:
            raise ValueError(f"api.{name} 必须在 (0, {upper}] 内")
        config.api[name] = value
    if config.backend == "api":
        config.validate_api()
    return config


def add_stt_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stt-backend", choices=("local", "api"), default=None,
                        help="STT 后端；未指定时从 STT_BACKEND/配置读取，默认 local")
    parser.add_argument("--stt-config", default=None, help="STT JSON 配置路径（密钥只写环境变量）")
    for name in ("base-url", "model", "key-env", "response-format", "timeout", "chunk-seconds", "max-upload-mb"):
        parser.add_argument(f"--stt-api-{name}", default=None,
                            type=float if name in ("timeout", "chunk-seconds", "max-upload-mb") else str)


def api_options_from_args(args) -> dict:
    return {key: getattr(args, f"stt_api_{arg}") for key, arg in (
        ("base_url", "base_url"), ("model", "model"), ("api_key_env", "key_env"),
        ("response_format", "response_format"), ("timeout", "timeout"),
        ("chunk_seconds", "chunk_seconds"), ("max_upload_mb", "max_upload_mb"))}
