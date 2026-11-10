from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.store import Store


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        base_url="http://model.invalid/v1",
        api_key="test-key",
        model="test-model",
        temperature=0.2,
        max_tokens=128,
        request_timeout=5,
        max_iterations=3,
        data_dir=tmp_path,
        history_turns=2,
        weather_timeout=2,
        log_level="info",
        log_format="json",
        observability_enabled=False,
        observability_host="127.0.0.1",
        observability_port=0,
    )


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path)
