"""Runtime configuration for the NIM client."""

from __future__ import annotations

import contextlib
import contextvars
import logging
import os
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_RERANK_URL = "https://ai.api.nvidia.com/v1/retrieval"

_log = logging.getLogger("nimlab.config")
_warned_no_key = False
_quiet: contextvars.ContextVar[bool] = contextvars.ContextVar("nimlab_quiet", default=False)

_DOTENV_PREFIXES = ("NVIDIA_", "NIM_", "NIMLAB_")


@lru_cache(maxsize=16)
def _load_dotenv(directory: str) -> None:
    """Load ``NVIDIA_*`` / ``NIM_*`` / ``NIMLAB_*`` keys from ``<directory>/.env`` (no override)."""
    path = Path(directory) / ".env"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].strip()
        if key.startswith(_DOTENV_PREFIXES) and key not in os.environ:
            os.environ[key] = value


def _env(name: str) -> str | None:
    if not os.environ.get("NIMLAB_NO_DOTENV"):
        _load_dotenv(os.getcwd())
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return value.strip()


def _env_bool(name: str) -> bool | None:
    value = _env(name)
    if value is None:
        return None
    return value.lower() in {"1", "true", "yes", "on"}


@contextlib.contextmanager
def quiet_config() -> Iterator[None]:
    """Suppress the one-time 'no API key, using mock' warning inside the block."""
    token = _quiet.set(True)
    try:
        yield
    finally:
        _quiet.reset(token)


def _reset_warning() -> None:
    """Re-arm the one-time warning (tests)."""
    global _warned_no_key
    _warned_no_key = False


class NIMConfig(BaseModel):
    """Configuration; unspecified fields are read from the environment (and ``./.env``).

    =================  ====================  ==========================================
    field              env var               default
    =================  ====================  ==========================================
    api_key            NVIDIA_API_KEY        None
    base_url           NIM_BASE_URL          https://integrate.api.nvidia.com/v1
    rerank_base_url    NIM_RERANK_URL        https://ai.api.nvidia.com/v1/retrieval
    timeout            NIM_TIMEOUT           120 (seconds)
    max_retries        NIM_MAX_RETRIES       5
    max_concurrency    NIM_MAX_CONCURRENCY   8
    mock               NIMLAB_MOCK           True if no api_key, else False
    =================  ====================  ==========================================

    ``mock`` is true when ``NIMLAB_MOCK`` is truthy **or** when there is no API key (a one-time
    warning is logged in that second case). Passing ``mock=False`` / ``NIMLAB_MOCK=0`` forces the
    real backend even without a key (useful for self-hosted NIMs that need no auth).
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=False)

    api_key: str | None = Field(default=None, repr=False)
    base_url: str = DEFAULT_BASE_URL
    rerank_base_url: str = DEFAULT_RERANK_URL
    timeout: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=5, ge=0)
    max_concurrency: int = Field(default=8, ge=1)
    mock: bool = False
    # Why mock is on: "explicit" (arg/env), "no_api_key", or None when mock is off.
    mock_reason: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _from_env(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        d = dict(data)

        def fill(field: str, env_name: str) -> None:
            if d.get(field) is None:
                value = _env(env_name)
                if value is not None:
                    d[field] = value
                else:
                    d.pop(field, None)

        fill("api_key", "NVIDIA_API_KEY")
        fill("base_url", "NIM_BASE_URL")
        fill("rerank_base_url", "NIM_RERANK_URL")
        fill("timeout", "NIM_TIMEOUT")
        fill("max_retries", "NIM_MAX_RETRIES")
        fill("max_concurrency", "NIM_MAX_CONCURRENCY")

        key = d.get("api_key")
        if isinstance(key, str) and not key.strip():
            d["api_key"] = key = None

        if d.get("mock") is None:
            forced = _env_bool("NIMLAB_MOCK")
            if forced is not None:
                d["mock"] = forced
                d["mock_reason"] = "explicit" if forced else None
            elif not key:
                d["mock"] = True
                d["mock_reason"] = "no_api_key"
            else:
                d["mock"] = False
        elif d["mock"] and d.get("mock_reason") is None:
            d["mock_reason"] = "explicit"
        return d

    @field_validator("base_url", "rerank_base_url")
    @classmethod
    def _strip_slash(cls, v: str) -> str:
        return v.rstrip("/")

    def model_post_init(self, context: Any, /) -> None:
        global _warned_no_key
        if self.mock_reason == "no_api_key" and not _warned_no_key and not _quiet.get():
            _warned_no_key = True
            _log.warning(
                "NVIDIA_API_KEY is not set: running in MOCK mode (offline, deterministic "
                "responses). Set NVIDIA_API_KEY for real NIM calls, or NIMLAB_MOCK=1 to "
                "silence this warning."
            )

    @classmethod
    def from_env(cls, **overrides: Any) -> NIMConfig:
        """Build a config from environment variables; keyword overrides win."""
        return cls(**overrides)
