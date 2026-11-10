from __future__ import annotations

import pytest

from app.config import load_settings


def test_settings_defaults(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    settings = load_settings()
    assert settings.base_url == "http://localhost:11434/v1"
    assert settings.log_format == "json"
    assert settings.observability_enabled is False


@pytest.mark.parametrize("key,value", [("MAX_TOKENS", "nope"), ("LOG_FORMAT", "xml"), ("OBSERVABILITY_ENABLED", "perhaps")])
def test_settings_reject_invalid_values(monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError):
        load_settings()
