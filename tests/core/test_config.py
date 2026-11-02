from __future__ import annotations

import logging
from pathlib import Path

import pytest
from pydantic import ValidationError

from nimlab.core import NIMConfig
from nimlab.core import config as config_module
from nimlab.core.config import DEFAULT_BASE_URL, DEFAULT_RERANK_URL, quiet_config


def test_defaults_without_key_enable_mock() -> None:
    cfg = NIMConfig()
    assert cfg.api_key is None
    assert cfg.mock is True
    assert cfg.mock_reason == "no_api_key"
    assert cfg.base_url == DEFAULT_BASE_URL == "https://integrate.api.nvidia.com/v1"
    assert cfg.rerank_base_url == DEFAULT_RERANK_URL == "https://ai.api.nvidia.com/v1/retrieval"
    assert (cfg.timeout, cfg.max_retries, cfg.max_concurrency) == (120, 5, 8)


def test_key_disables_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-abc")
    cfg = NIMConfig.from_env()
    assert cfg.api_key == "nvapi-abc"
    assert cfg.mock is False
    assert cfg.mock_reason is None


def test_blank_key_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "   ")
    assert NIMConfig().api_key is None
    assert NIMConfig(api_key="").mock is True


def test_nimlab_mock_env_forces_mock_even_with_key(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-abc")
    monkeypatch.setenv("NIMLAB_MOCK", "1")
    with caplog.at_level(logging.WARNING):
        cfg = NIMConfig.from_env()
    assert cfg.mock is True
    assert cfg.mock_reason == "explicit"
    assert not caplog.records  # explicit mock does not warn


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_truthy_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "k")
    monkeypatch.setenv("NIMLAB_MOCK", value)
    assert NIMConfig().mock is True


def test_nimlab_mock_zero_forces_real_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NIMLAB_MOCK", "0")
    cfg = NIMConfig()
    assert cfg.mock is False and cfg.api_key is None


def test_explicit_mock_false_without_key_is_respected() -> None:
    assert NIMConfig(mock=False).mock is False


def test_one_time_warning_when_no_key(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="nimlab"):
        NIMConfig()
        NIMConfig()
        NIMConfig.from_env()
    warnings = [r for r in caplog.records if "MOCK mode" in r.getMessage()]
    assert len(warnings) == 1


def test_quiet_config_suppresses_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="nimlab"), quiet_config():
        NIMConfig()
    assert not caplog.records
    with caplog.at_level(logging.WARNING, logger="nimlab"):
        NIMConfig()  # not consumed by the quiet one
    assert len(caplog.records) == 1


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "k")
    monkeypatch.setenv("NIM_BASE_URL", "http://localhost:8000/v1/")
    monkeypatch.setenv("NIM_RERANK_URL", "http://localhost:8001/v1/retrieval")
    monkeypatch.setenv("NIM_TIMEOUT", "30.5")
    monkeypatch.setenv("NIM_MAX_RETRIES", "2")
    monkeypatch.setenv("NIM_MAX_CONCURRENCY", "3")
    cfg = NIMConfig.from_env()
    assert cfg.base_url == "http://localhost:8000/v1"  # trailing slash stripped
    assert cfg.rerank_base_url == "http://localhost:8001/v1/retrieval"
    assert (cfg.timeout, cfg.max_retries, cfg.max_concurrency) == (30.5, 2, 3)


def test_explicit_args_beat_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "env-key")
    monkeypatch.setenv("NIM_BASE_URL", "http://env")
    cfg = NIMConfig.from_env(api_key="arg-key", base_url="http://arg/", max_retries=0)
    assert (cfg.api_key, cfg.base_url, cfg.max_retries) == ("arg-key", "http://arg", 0)


def test_api_key_not_in_repr_or_str() -> None:
    cfg = NIMConfig(api_key="nvapi-SECRET", mock=False)
    assert "SECRET" not in repr(cfg)
    assert "SECRET" not in str(cfg)


@pytest.mark.parametrize(
    "kwargs", [{"max_concurrency": 0}, {"max_retries": -1}, {"timeout": 0}, {"bogus": 1}]
)
def test_validation(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        NIMConfig(**kwargs)  # type: ignore[arg-type]


def test_dotenv_loading(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # register undo so values loaded by the dotenv reader do not leak into other tests
    for name in ("NVIDIA_API_KEY", "NIM_TIMEOUT", "NIMLAB_MOCK", "OTHER_SECRET"):
        monkeypatch.setenv(name, "placeholder")
        monkeypatch.delenv(name)
    monkeypatch.delenv("NIMLAB_NO_DOTENV")
    (tmp_path / ".env").write_text(
        "# comment\n"
        'NVIDIA_API_KEY="nvapi-from-dotenv"\n'
        "export NIM_TIMEOUT=45 # inline comment\n"
        "OTHER_SECRET=nope\n"
        "\n"
        "garbage line\n"
    )
    config_module._load_dotenv.cache_clear()
    cfg = NIMConfig()
    assert cfg.api_key == "nvapi-from-dotenv"
    assert cfg.timeout == 45
    assert cfg.mock is False
    import os

    assert "OTHER_SECRET" not in os.environ  # only NVIDIA_/NIM_/NIMLAB_ keys are loaded


def test_dotenv_does_not_override_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("NIMLAB_NO_DOTENV")
    monkeypatch.setenv("NVIDIA_API_KEY", "from-env")
    (tmp_path / ".env").write_text("NVIDIA_API_KEY=from-file\n")
    config_module._load_dotenv.cache_clear()
    assert NIMConfig().api_key == "from-env"
