"""Settings loaded from .env."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    base_url: str
    api_key: str
    model: str
    temperature: float
    max_tokens: int
    request_timeout: float
    max_iterations: int
    data_dir: Path
    history_turns: int
    weather_timeout: float
    log_level: str
    log_format: str
    observability_enabled: bool
    observability_host: str
    observability_port: int


def _str(key: str, default: str) -> str:
    value = os.getenv(key)
    return default if value is None or not value.strip() else value.strip()


def _num(key: str, default: float, cast):
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return cast(default)
    try:
        return cast(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{key} in .env is not a valid number: {raw!r}") from exc


def _bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{key} in .env must be true or false, got {raw!r}")


def _choice(key: str, default: str, choices: set[str]) -> str:
    value = _str(key, default).lower()
    if value not in choices:
        raise ValueError(f"{key} in .env must be one of {', '.join(sorted(choices))}")
    return value


def load_settings() -> Settings:
    data_dir = Path(_str("DATA_DIR", "./data")).expanduser()
    return Settings(
        base_url=_str("OPENAI_BASE_URL", "http://localhost:11434/v1"),
        # Local runtimes ignore the key but the OpenAI client insists on one.
        api_key=_str("OPENAI_API_KEY", "not-needed"),
        model=_str("MODEL", "llama3.1:8b"),
        temperature=_num("TEMPERATURE", 0.2, float),
        max_tokens=_num("MAX_TOKENS", 1024, int),
        request_timeout=_num("REQUEST_TIMEOUT", 180, float),
        max_iterations=_num("MAX_ITERATIONS", 14, int),
        data_dir=data_dir,
        history_turns=_num("HISTORY_TURNS", 6, int),
        weather_timeout=_num("WEATHER_TIMEOUT", 20, float),
        log_level=_choice("LOG_LEVEL", "INFO", {"debug", "info", "warning", "error", "critical"}),
        log_format=_choice("LOG_FORMAT", "json", {"json", "text"}),
        observability_enabled=_bool("OBSERVABILITY_ENABLED", False),
        observability_host=_str("OBSERVABILITY_HOST", "0.0.0.0"),
        observability_port=_num("OBSERVABILITY_PORT", 8080, int),
    )
