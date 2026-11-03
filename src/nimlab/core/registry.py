"""Curated catalog of NVIDIA NIM models with role-based defaults, live discovery and overrides.

``pick(role)`` precedence (highest first)::

    per-request / per-app override  >  global role override (Studio settings)  >  env
    NIMLAB_MODEL_<ROLE>  >  nimlab.toml [models]  >  first *available* preferred default
    (live ``/v1/models`` + session blacklist)  >  best heuristic match in the live catalog
    >  curated default
"""

from __future__ import annotations

import contextlib
import contextvars
import copy
import json
import os
import sys
import threading
from collections.abc import Iterable, Iterator, Mapping
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .catalog import (
    CACHE_FILENAME,
    CatalogSnapshot,
    LiveCatalog,
    heuristic_score,
    rank_for_role,
    studio_home,
)
from .config import NIMConfig, quiet_config
from .logging import get_logger

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on 3.10 only
    import tomli as tomllib

__all__ = [
    "DEFAULTS",
    "PREFERRED",
    "RETIRED",
    "ModelInfo",
    "ModelRegistry",
    "Role",
    "default_registry",
    "model_scope",
    "reset_default_registry",
    "set_default_registry",
]

_log = get_logger("nimlab.registry")


class Role(str, Enum):
    CHAT = "chat"
    REASONING = "reasoning"
    CODER = "coder"
    VISION = "vision"
    EMBED = "embed"
    RERANK = "rerank"
    REWARD = "reward"
    SAFETY = "safety"
    FAST = "fast"

    def __str__(self) -> str:
        return self.value


class ModelInfo(BaseModel):
    """One catalog entry. ``cost_tier`` is relative: 1 (cheapest/fastest) to 5 (largest).

    ``deprecated`` marks models NVIDIA still lists but has flagged for retirement; they are kept
    only as last-resort fallbacks and are never role defaults unless nothing else exists.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    roles: frozenset[Role]
    context_window: int
    notes: str = ""
    cost_tier: int = Field(ge=1, le=5)
    deprecated: bool = False


def _m(
    id_: str,
    roles: Iterable[Role],
    ctx: int,
    tier: int,
    notes: str = "",
    *,
    deprecated: bool = False,
) -> ModelInfo:
    return ModelInfo(
        id=id_,
        roles=frozenset(roles),
        context_window=ctx,
        cost_tier=tier,
        notes=notes,
        deprecated=deprecated,
    )


_C, _R, _K, _V = Role.CHAT, Role.REASONING, Role.CODER, Role.VISION
_E, _RR, _RW, _S, _F = Role.EMBED, Role.RERANK, Role.REWARD, Role.SAFETY, Role.FAST

# Refreshed 2026-11-03 against NVIDIA's live catalog. Sources (all fetched 2026-11-03):
#   [L] https://integrate.api.nvidia.com/v1/models (id is served by the hosted API)
#   [B] https://build.nvidia.com/<org>/<model> model pages (labels such as "Deprecated")
#   [D] https://docs.api.nvidia.com/nim/reference/... API reference pages
#   [G] https://big-agi.com/nvidia (third-party tracker of live-verified NVIDIA models: context
#       windows, retirement dates). Contexts marked "unverified" have no first-party source.
# NVIDIA lists some retired ids in /v1/models and its catalog turns over every few weeks, so
# `nimlab models --live` and the runtime fallback (see NIMClient) are the source of truth.
CURATED_MODELS: tuple[ModelInfo, ...] = (
    # --- Nemotron 3 / 3.5 (NVIDIA flagship open models) -------------------------------------------
    # [L][B] https://build.nvidia.com/nvidia/nemotron-3-ultra-550b-a55b ; ctx [G]
    _m(
        "nvidia/nemotron-3-ultra-550b-a55b",
        [_R, _C],
        1048576,
        5,
        "Flagship hybrid Mamba-Transformer MoE (550B, 55B active); reasoning toggle: "
        "chat_template_kwargs.enable_thinking",
    ),
    # [L] [G] https://big-agi.com/models/nemotron-3-super-120b-a12b
    _m(
        "nvidia/nemotron-3-super-120b-a12b",
        [_C, _R],
        1048576,
        3,
        "Open MoE (120B, 12B active); strong general default, reasoning on by default",
    ),
    # [L][B] https://build.nvidia.com/nvidia/nemotron-3.5-super-vl-120b-a12b ("early access")
    _m(
        "nvidia/nemotron-3.5-super-vl-120b-a12b",
        [_V, _C, _R],
        1048576,
        3,
        "Early-access multimodal MoE (image input, reasoning)",
    ),
    # [L][B] https://build.nvidia.com/nvidia/nemotron-3.5-lightning-30b-a3b ; ctx [G]
    _m(
        "nvidia/nemotron-3.5-lightning-30b-a3b",
        [_F, _C],
        1048576,
        1,
        "Fastest Nemotron MoE (30B, 3B active); text only",
    ),
    # [L][B] https://build.nvidia.com/nvidia/nemotron-3.5-bolt-e4b ; context UNVERIFIED
    _m(
        "nvidia/nemotron-3.5-bolt-e4b",
        [_F, _V],
        131072,
        1,
        "Compact vision-language model (4B effective); context unverified",
    ),
    # [L][B] https://build.nvidia.com/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning
    # context 256K-1M depending on source ([G], OpenRouter)
    _m(
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        [_V, _R, _C],
        262144,
        2,
        "Omni-modal Nemotron Nano (image/video/audio input), reasoning",
    ),
    # [L] older but still served
    _m(
        "nvidia/llama-3.1-nemotron-ultra-253b-v1",
        [_R, _C],
        131072,
        5,
        "Previous-generation Nemotron reasoning flagship (still listed)",
    ),
    _m(
        "nvidia/llama-3.1-nemotron-70b-instruct",
        [_C],
        131072,
        2,
        "Previous-generation helpful-tuned Llama 3.1 70B (still listed)",
    ),
    # --- other current flagship open models ([L]; details [G]) -----------------------------------
    _m(
        "deepseek-ai/deepseek-v4.1-flash",
        [_R, _C, _K],
        1048576,
        2,
        "DeepSeek V4.1 Flash MoE (552B, 8B active), reasoning",
    ),
    _m(
        "google/gemma-4-31b-it",
        [_C, _V, _F],
        131072,
        2,
        "Google Gemma 4; NVIDIA silently truncates inputs beyond 131K",
    ),
    _m("google/gemma-3-12b-it", [_F, _V, _C], 131072, 1, "Gemma 3 12B (multimodal)"),
    _m("google/gemma-3-4b-it", [_F, _V], 131072, 1, "Gemma 3 4B, tiny and fast"),
    _m(
        "openai/gpt-oss-20b",
        [_R, _F, _C],
        131072,
        1,
        "Small open-weight reasoning MoE (adjustable effort)",
    ),
    _m("moonshotai/kimi-k2.6", [_R, _C], 262144, 3, "Moonshot Kimi K2.6; context unverified"),
    _m(
        "z-ai/glm-5.3-flash",
        [_R, _C, _V],
        1048576,
        2,
        "Z.ai GLM 5.3 Flash (320B MoE, 18B active), reasoning",
    ),
    _m(
        "minimaxai/minimax-m3.1-flash",
        [_R, _C],
        262144,
        2,
        "MiniMax M3.1 Flash; context unverified",
    ),
    _m(
        "mistralai/mistral-large-4-1t-a52b",
        [_C],
        262144,
        5,
        "Mistral Large 4 (1T MoE, 52B active); context unverified",
    ),
    _m("mistralai/mistral-large-2-instruct", [_C], 131072, 4, "Mistral Large 2 (123B, older)"),
    _m(
        "meta/muse-glimmer-30b",
        [_V, _R, _C],
        131072,
        2,
        "Meta open multimodal reasoning model (30B dense)",
    ),
    # --- coding ---------------------------------------------------------------------------------
    # [L][B] https://build.nvidia.com/poolside/laguna-xs-2.1 ; description/ctx [G]
    _m(
        "poolside/laguna-xs-2.1",
        [_K],
        262144,
        2,
        "Poolside coding-focused reasoning model; can be slow to cold-start",
    ),
    _m(
        "mistralai/codestral-22b-instruct-v0.1",
        [_K],
        32768,
        2,
        "Older code model with fill-in-middle (still listed)",
    ),
    # --- vision ([L]; ctx [G]) ------------------------------------------------------------------
    _m(
        "meta/llama-3.2-90b-vision-instruct",
        [_V, _C],
        32768,
        4,
        "Image + text input; NVIDIA serves a reduced 32K context",
    ),
    _m("meta/llama-3.2-11b-vision-instruct", [_V], 131072, 2, "Smaller Llama vision model"),
    # --- embeddings: all use POST /v1/embeddings with input_type=query|passage -------------------
    # [D] https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3_5-embed-1b ; ctx UNVERIFIED
    _m(
        "nvidia/nemotron-3.5-embed-1b",
        [_E],
        8192,
        1,
        "Multimodal/multilingual retrieval embeddings (Oct 2026); requires input_type",
    ),
    _m(
        "nvidia/nemotron-3.5-embed-8b",
        [_E],
        8192,
        2,
        "8B multimodal embeddings, 4096-d (Oct 2026); requires input_type",
    ),
    # [D] https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3-embed-1b
    _m(
        "nvidia/nemotron-3-embed-1b",
        [_E],
        8192,
        1,
        "1B text embeddings for retrieval and code retrieval (Jul 2026)",
    ),
    _m(
        "nvidia/llama-nemotron-embed-vl-1b-v2",
        [_E],
        8192,
        1,
        "Multimodal Q&A retrieval embeddings (Feb 2026)",
    ),
    _m(
        "nvidia/llama-3.2-nv-embedqa-1b-v1",
        [_E],
        8192,
        1,
        "Older multilingual retrieval embeddings (still listed)",
    ),
    _m(
        "nvidia/nv-embedqa-mistral-7b-v2",
        [_E],
        512,
        2,
        "Older English retrieval embeddings (still listed)",
    ),
    # --- reranking: POST {rerank_base_url}/{model}/reranking (NOT listed by /v1/models) ----------
    # [D] https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3_5-rerank-8b-infer
    _m("nvidia/nemotron-3.5-rerank-8b", [_RR], 8192, 2, "8B multimodal reranker (Oct 2026)"),
    # [D] https://docs.api.nvidia.com/nim/reference/nvidia-llama-nemotron-rerank-vl-1b-v2-infer
    _m(
        "nvidia/llama-nemotron-rerank-vl-1b-v2",
        [_RR],
        8192,
        1,
        "1.7B multimodal reranker (Mar 2026)",
    ),
    # [B] https://build.nvidia.com/nvidia/llama-nemotron-rerank-1b-v2 is labelled "Deprecated"
    _m(
        "nvidia/llama-nemotron-rerank-1b-v2",
        [_RR],
        8192,
        1,
        "Text reranker; hosted endpoint deprecated",
        deprecated=True,
    ),
    _m(
        "nvidia/llama-3.2-nv-rerankqa-1b-v2",
        [_RR],
        8192,
        1,
        "Previous reranker; UNVERIFIED but likely retired on the hosted API",
        deprecated=True,
    ),
    # --- reward: NVIDIA has deprecated every hosted reward model ---------------------------------
    # [B] https://build.nvidia.com/nvidia/nemotron-4-340b-reward is "Deprecated" (still in [L])
    _m(
        "nvidia/nemotron-4-340b-reward",
        [_RW],
        4096,
        4,
        "Scores helpfulness/correctness/coherence/complexity/verbosity; DEPRECATED",
        deprecated=True,
    ),
    # [B] https://build.nvidia.com/nvidia/llama-3_1-nemotron-70b-reward is "Deprecated" (not in [L])
    _m(
        "nvidia/llama-3.1-nemotron-70b-reward",
        [_RW],
        4096,
        2,
        "Returns reward:<float>; DEPRECATED and no longer listed",
        deprecated=True,
    ),
    # --- safety ----------------------------------------------------------------------------------
    # [D] https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3-5-content-safety (plain text)
    _m(
        "nvidia/nemotron-3.5-content-safety",
        [_S],
        131072,
        1,
        "Multilingual multimodal content safety (Gemma-3-4B base); plain 'User Safety: ...' reply",
    ),
    _m(
        "nvidia/llama-3.1-nemotron-safety-guard-8b-v3",
        [_S],
        131072,
        1,
        "Multilingual content-safety classifier (Oct 2025)",
    ),
    _m(
        "nvidia/llama-3.1-nemoguard-8b-content-safety",
        [_S],
        131072,
        1,
        "Content-safety classifier (JSON verdict)",
    ),
    _m("meta/llama-guard-4-12b", [_S], 65536, 1, "Llama Guard 4; replies 'safe' / 'unsafe\\nS1'"),
    _m(
        "nvidia/llama-3.1-nemoguard-8b-topic-control",
        [_S],
        131072,
        1,
        "Topic-control guardrail (not a content-safety model)",
    ),
)

#: Ordered preferences per role. ``pick`` returns the first one the endpoint serves.
PREFERRED: dict[Role, tuple[str, ...]] = {
    Role.CHAT: (
        "nvidia/nemotron-3-super-120b-a12b",
        "google/gemma-4-31b-it",
        "mistralai/mistral-large-4-1t-a52b",
        "nvidia/llama-3.1-nemotron-70b-instruct",
        "deepseek-ai/deepseek-v4.1-flash",
        "nvidia/nemotron-3-ultra-550b-a55b",
    ),
    Role.REASONING: (
        "nvidia/nemotron-3-ultra-550b-a55b",
        "nvidia/nemotron-3-super-120b-a12b",
        "deepseek-ai/deepseek-v4.1-flash",
        "moonshotai/kimi-k2.6",
        "z-ai/glm-5.3-flash",
        "openai/gpt-oss-20b",
        "nvidia/llama-3.1-nemotron-ultra-253b-v1",
    ),
    Role.CODER: (
        "poolside/laguna-xs-2.1",
        "nvidia/nemotron-3-super-120b-a12b",
        "deepseek-ai/deepseek-v4.1-flash",
        "nvidia/nemotron-3-ultra-550b-a55b",
        "mistralai/codestral-22b-instruct-v0.1",
    ),
    Role.VISION: (
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        "google/gemma-4-31b-it",
        "nvidia/nemotron-3.5-super-vl-120b-a12b",
        "meta/llama-3.2-90b-vision-instruct",
        "meta/llama-3.2-11b-vision-instruct",
        "meta/muse-glimmer-30b",
    ),
    Role.EMBED: (
        "nvidia/nemotron-3.5-embed-1b",
        "nvidia/nemotron-3-embed-1b",
        "nvidia/nemotron-3.5-embed-8b",
        "nvidia/llama-nemotron-embed-vl-1b-v2",
        "nvidia/llama-3.2-nv-embedqa-1b-v1",
        "nvidia/nv-embedqa-mistral-7b-v2",
    ),
    Role.RERANK: (
        "nvidia/nemotron-3.5-rerank-8b",
        "nvidia/llama-nemotron-rerank-vl-1b-v2",
        "nvidia/llama-nemotron-rerank-1b-v2",
        "nvidia/llama-3.2-nv-rerankqa-1b-v2",
    ),
    Role.REWARD: (
        "nvidia/nemotron-4-340b-reward",
        "nvidia/llama-3.1-nemotron-70b-reward",
    ),
    Role.SAFETY: (
        "nvidia/nemotron-3.5-content-safety",
        "nvidia/llama-3.1-nemotron-safety-guard-8b-v3",
        "nvidia/llama-3.1-nemoguard-8b-content-safety",
        "meta/llama-guard-4-12b",
    ),
    Role.FAST: (
        "nvidia/nemotron-3.5-lightning-30b-a3b",
        "nvidia/nemotron-3.5-bolt-e4b",
        "openai/gpt-oss-20b",
        "google/gemma-3-4b-it",
        "google/gemma-3-12b-it",
    ),
}

DEFAULTS: dict[Role, str] = {role: ids[0] for role, ids in PREFERRED.items()}

#: Ids that earlier releases of NIM Lab used and that are absent from the live catalog (or retired
#: per https://big-agi.com/nvidia) mapped to the suggested replacement. Used by ``nimlab models
#: --live`` and by the runtime fallback hint.
RETIRED: dict[str, str] = {
    "meta/llama-3.3-70b-instruct": "nvidia/nemotron-3-super-120b-a12b",
    "meta/llama-3.1-405b-instruct": "nvidia/nemotron-3-ultra-550b-a55b",
    "meta/llama-3.1-8b-instruct": "nvidia/nemotron-3.5-lightning-30b-a3b",
    "meta/llama-3.2-3b-instruct": "nvidia/nemotron-3.5-bolt-e4b",
    "meta/llama-3.2-1b-instruct": "nvidia/nemotron-3.5-bolt-e4b",
    "meta/llama-4-maverick-17b-128e-instruct": "nvidia/nemotron-3-super-120b-a12b",
    "meta/llama-4-scout-17b-16e-instruct": "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/llama-3.3-nemotron-super-49b-v1": "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/llama-3.3-nemotron-super-49b-v1.5": "nvidia/nemotron-3-super-120b-a12b",
    "deepseek-ai/deepseek-r1": "nvidia/nemotron-3-ultra-550b-a55b",
    "deepseek-ai/deepseek-r1-distill-qwen-32b": "openai/gpt-oss-20b",
    "qwen/qwq-32b": "nvidia/nemotron-3-ultra-550b-a55b",
    "qwen/qwen2.5-coder-32b-instruct": "poolside/laguna-xs-2.1",
    "mistralai/mixtral-8x22b-instruct-v0.1": "mistralai/mistral-large-4-1t-a52b",
    "microsoft/phi-4-mini-instruct": "nvidia/nemotron-3.5-bolt-e4b",
    "google/gemma-2-27b-it": "google/gemma-4-31b-it",
    "openai/gpt-oss-120b": "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/llama-3.2-nv-embedqa-1b-v2": "nvidia/nemotron-3.5-embed-1b",
    "nvidia/nv-embedqa-e5-v5": "nvidia/nemotron-3.5-embed-1b",
    "baai/bge-m3": "nvidia/nemotron-3.5-embed-1b",
}

LIVE_MODELS_FILE = Path(__file__).parent / "data" / "live_models.txt"
STUDIO_SETTINGS_FILE = "studio_settings.json"
#: Roles whose models are not served from ``/v1/models`` (reranking lives on the retrieval API).
UNLISTED_ROLES: frozenset[Role] = frozenset({Role.RERANK})
STALE_RETRY_S = 600.0


def _coerce_role(role: Role | str) -> Role:
    if isinstance(role, Role):
        return role
    try:
        return Role(str(role).strip().lower())
    except ValueError:
        valid = ", ".join(r.value for r in Role)
        raise ValueError(f"unknown role {role!r}; valid roles: {valid}") from None


class _Shared:
    """State shared by a registry and every scoped view of it."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.snapshot: CatalogSnapshot | None = None
        self.loaded_at: float | None = None
        self.config: NIMConfig | None = None
        self.unavailable: dict[str, str] = {}
        self.auto: dict[str, Role] = {}
        self.substitutions: list[dict[str, str]] = []
        self.settings_path: Path | None = None
        self.settings_cache: tuple[float, dict[str, Any]] | None = None


class ModelRegistry:
    """Catalog + role selection (see the module docstring for the precedence order).

    ``live`` forces live discovery on/off (``None``: on when there is an API key and
    ``NIMLAB_DISCOVERY`` is not ``0``). Overrides may name models that are not in the catalog.
    """

    def __init__(
        self,
        models: Iterable[ModelInfo] | None = None,
        defaults: Mapping[Role, str] | None = None,
        *,
        config_path: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
        live: bool | None = None,
        nim_config: NIMConfig | None = None,
        catalog: LiveCatalog | None = None,
        settings_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self._models: dict[str, ModelInfo] = {
            m.id: m for m in (CURATED_MODELS if models is None else models)
        }
        if defaults is None:
            self._preferred: dict[Role, tuple[str, ...]] = dict(PREFERRED)
        else:
            self._preferred = {_coerce_role(r): (m,) for r, m in defaults.items()}
        self._config_path = Path(config_path) if config_path is not None else None
        self._env = env
        self._live = live
        self._catalog = catalog
        self._sh = _Shared()
        self._sh.config = nim_config
        if settings_path is not None:
            self._sh.settings_path = Path(settings_path)
        self._app: str | None = None
        self._request: dict[Role, str] = {}
        self.picks: list[dict[str, str]] | None = None  # recorded inside a ``model_scope``

    @property
    def _defaults(self) -> dict[Role, str]:
        return {r: ids[0] for r, ids in self._preferred.items() if ids}

    # -- catalog access
    def get(self, model_id: str) -> ModelInfo:
        """Catalog entry for ``model_id``; raises ``KeyError`` if it is not curated."""
        try:
            return self._models[model_id]
        except KeyError:
            raise KeyError(f"model {model_id!r} is not in the curated catalog") from None

    def find(self, model_id: str) -> ModelInfo | None:
        return self._models.get(model_id)

    def __contains__(self, model_id: object) -> bool:
        return model_id in self._models

    def __iter__(self) -> Iterator[ModelInfo]:
        return iter(self._models.values())

    def __len__(self) -> int:
        return len(self._models)

    def all(self) -> list[ModelInfo]:
        return list(self._models.values())

    def ids(self) -> list[str]:
        return list(self._models)

    def register(self, info: ModelInfo, *, default_for: Role | str | None = None) -> None:
        """Add or replace a catalog entry (optionally making it a role default)."""
        self._models[info.id] = info
        if default_for is not None:
            role = _coerce_role(default_for)
            rest = tuple(m for m in self._preferred.get(role, ()) if m != info.id)
            self._preferred[role] = (info.id, *rest)

    def by_role(self, role: Role | str) -> list[ModelInfo]:
        """Models with ``role``: the current pick first (if curated), then by cost tier, id."""
        r = _coerce_role(role)
        picked = self._safe_pick(r)
        found = [m for m in self._models.values() if r in m.roles]
        found.sort(key=lambda m: (m.id != picked, m.deprecated, m.cost_tier, m.id))
        return found

    def _safe_pick(self, role: Role) -> str | None:
        try:
            return self.pick(role)
        except KeyError:
            return None

    # -- environment / files
    def _environ(self) -> Mapping[str, str]:
        return self._env if self._env is not None else os.environ

    def _toml_models(self) -> Mapping[str, Any]:
        path = self._config_path
        if path is None:
            env_path = self._environ().get("NIMLAB_CONFIG")
            path = Path(env_path) if env_path else Path.cwd() / "nimlab.toml"
        try:
            with path.open("rb") as fh:
                data = tomllib.load(fh)
        except (OSError, tomllib.TOMLDecodeError):
            return {}
        table = data.get("models", {})
        return table if isinstance(table, dict) else {}

    # -- Studio settings (global role overrides and per-app overrides)
    @property
    def settings_path(self) -> Path:
        sh = self._sh
        return sh.settings_path or studio_home(self._environ()) / STUDIO_SETTINGS_FILE

    def use_settings_path(self, path: str | os.PathLike[str] | None) -> None:
        """Read/write Studio overrides at ``path`` (``<home>/studio_settings.json``)."""
        with self._sh.lock:
            self._sh.settings_path = Path(path) if path is not None else None
            self._sh.settings_cache = None

    def _read_settings(self) -> dict[str, Any]:
        path = self.settings_path
        try:
            mtime = path.stat().st_mtime_ns
        except OSError:
            return {}
        cached = self._sh.settings_cache
        if cached is not None and cached[0] == mtime:
            return cached[1]
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(data, dict):
            return {}
        self._sh.settings_cache = (mtime, data)
        return data

    @staticmethod
    def _clean_overrides(raw: Any) -> dict[Role, str]:
        out: dict[Role, str] = {}
        if isinstance(raw, dict):
            for key, value in raw.items():
                try:
                    role = _coerce_role(str(key))
                except ValueError:
                    continue
                if isinstance(value, str) and value.strip():
                    out[role] = value.strip()
        return out

    def studio_overrides(self) -> dict[Role, str]:
        """Global per-role overrides saved by Studio's Settings -> Models."""
        return self._clean_overrides(self._read_settings().get("model_overrides"))

    def app_overrides(self, app: str | None = None) -> dict[str, dict[Role, str]]:
        """Per-app overrides ``{app: {role: model}}`` (one app when ``app`` is given)."""
        raw = self._read_settings().get("app_model_overrides")
        out: dict[str, dict[Role, str]] = {}
        if isinstance(raw, dict):
            for name, table in raw.items():
                clean = self._clean_overrides(table)
                if clean and (app is None or str(name) == app):
                    out[str(name)] = clean
        return out

    def _write_settings(self, mutate: Any) -> None:
        with self._sh.lock:
            path = self.settings_path
            data = dict(self._read_settings())
            mutate(data)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")
            os.replace(tmp, path)
            self._sh.settings_cache = None

    def set_studio_overrides(self, overrides: Mapping[Role | str, str | None]) -> None:
        """Merge global overrides; a falsy / ``"auto"`` value clears that role."""

        def mutate(data: dict[str, Any]) -> None:
            table = {
                r.value: m for r, m in self._clean_overrides(data.get("model_overrides")).items()
            }
            for key, value in overrides.items():
                role = _coerce_role(key).value
                if value and value.strip().lower() != "auto":
                    table[role] = value.strip()
                else:
                    table.pop(role, None)
            data["model_overrides"] = table

        self._write_settings(mutate)

    def set_app_overrides(self, app: str, overrides: Mapping[Role | str, str | None]) -> None:
        """Merge overrides for one app; a falsy / ``"auto"`` value resets that role."""

        def mutate(data: dict[str, Any]) -> None:
            apps = dict(data.get("app_model_overrides") or {})
            table = {r.value: m for r, m in self._clean_overrides(apps.get(app)).items()}
            for key, value in overrides.items():
                role = _coerce_role(key).value
                if value and value.strip().lower() != "auto":
                    table[role] = value.strip()
                else:
                    table.pop(role, None)
            if table:
                apps[app] = table
            else:
                apps.pop(app, None)
            data["app_model_overrides"] = apps

        self._write_settings(mutate)

    # -- scopes
    def scoped(
        self, app: str | None = None, overrides: Mapping[Role | str, str] | None = None
    ) -> ModelRegistry:
        """A view applying ``overrides`` (this request) and the saved overrides of ``app``."""
        view = copy.copy(self)
        view._app = app
        view._request = {
            _coerce_role(k): v.strip()
            for k, v in (overrides or {}).items()
            if isinstance(v, str) and v.strip() and v.strip().lower() != "auto"
        }
        view.picks = []
        return view

    def override_layers(self, role: Role | str, app: str | None = None) -> dict[str, str]:
        """Every explicit override present for ``role``, keyed by source, highest first.

        Sources: ``request`` > ``app`` > ``studio`` > ``env`` > ``toml``.
        """
        r = _coerce_role(role)
        layers: dict[str, str] = {}
        if r in self._request:
            layers["request"] = self._request[r]
        name = app or self._app
        if name:
            value = self.app_overrides(name).get(name, {}).get(r)
            if value:
                layers["app"] = value
        value = self.studio_overrides().get(r)
        if value:
            layers["studio"] = value
        env_value = self._environ().get(f"NIMLAB_MODEL_{r.name}", "").strip()
        if env_value:
            layers["env"] = env_value
        table = self._toml_models()
        for key in (r.value, r.name, r.value.upper()):
            toml_value = table.get(key)
            if isinstance(toml_value, str) and toml_value.strip():
                layers["toml"] = toml_value.strip()
                break
        return layers

    def override_for(self, role: Role | str, app: str | None = None) -> tuple[str, str] | None:
        """``(model, source)`` of the winning explicit override, or ``None``."""
        layers = self.override_layers(role, app)
        return next(iter(layers.items()))[::-1] if layers else None

    # -- live discovery
    def _live_enabled(self) -> bool:
        if self._live is not None:
            return self._live
        if self._environ().get("NIMLAB_DISCOVERY", "").strip().lower() in {
            "0",
            "false",
            "no",
            "off",
        }:
            return False
        return not self._config().mock

    def _config(self) -> NIMConfig:
        sh = self._sh
        if sh.config is None:
            with quiet_config():
                sh.config = NIMConfig.from_env()
        return sh.config

    def _live_catalog(self) -> LiveCatalog:
        if self._catalog is None:
            self._catalog = LiveCatalog(Path(studio_home(self._environ())) / CACHE_FILENAME)
        return self._catalog

    def snapshot(self, *, fetch: bool = True) -> CatalogSnapshot | None:
        """The live model list (loaded once, then cached on disk for 24 h), or ``None``."""
        sh = self._sh
        snap = sh.snapshot
        if snap is not None:
            if not fetch or snap.age_s() < self._live_catalog().ttl_s:
                return snap
        elif not fetch:
            return None
        with sh.lock:
            if not self._live_enabled():
                return sh.snapshot
            now = self._now()
            if (
                sh.snapshot is None
                and sh.loaded_at is not None
                and now - sh.loaded_at < STALE_RETRY_S
            ):
                return None
            sh.snapshot = self._live_catalog().load(self._config(), force=False) or sh.snapshot
            sh.loaded_at = now
            return sh.snapshot

    @staticmethod
    def _now() -> float:
        import time

        return time.time()

    def refresh_catalog(
        self, *, config: NIMConfig | None = None, force: bool = True
    ) -> CatalogSnapshot | None:
        """Refetch ``/v1/models`` now (after a key change, or ``nimlab models --refresh``)."""
        sh = self._sh
        with sh.lock:
            if config is not None:
                sh.config = config
            if not self._live_enabled():
                return None
            snap = self._live_catalog().load(self._config(), force=force)
            sh.loaded_at = self._now()
            if snap is not None:
                sh.snapshot = snap
            return sh.snapshot

    def forget_catalog(self) -> None:
        """Drop the in-memory list (tests, key removal)."""
        with self._sh.lock:
            self._sh.snapshot = None
            self._sh.loaded_at = None

    def _snapshot_for(self, role: Role, snap: CatalogSnapshot | None) -> CatalogSnapshot | None:
        """The snapshot if it can vouch for ``role`` (reranking is not listed by /v1/models)."""
        if snap is None:
            return None
        if role in UNLISTED_ROLES and not any(
            heuristic_score(role.value, i) is not None for i in snap.ids
        ):
            return None
        return snap

    # -- availability
    def mark_unavailable(self, model: str, reason: str = "") -> None:
        """Remember (for this process) that ``model`` answered 404 / 'not available'."""
        self._sh.unavailable[model] = reason or "unavailable"

    def is_unavailable(self, model: str) -> bool:
        return model in self._sh.unavailable

    def unavailable(self) -> dict[str, str]:
        return dict(self._sh.unavailable)

    def role_of_pick(self, model: str) -> Role | None:
        """The role ``model`` was auto-picked for (``None`` if it was an explicit choice)."""
        return self._sh.auto.get(model)

    def record_substitution(self, requested: str, used: str, role: Role, reason: str = "") -> None:
        entry = {"requested": requested, "used": used, "role": role.value, "reason": reason[:200]}
        subs = self._sh.substitutions
        subs.append(entry)
        del subs[:-50]
        if self.picks is not None:
            self.picks.append(
                {"role": role.value, "model": used, "source": "fallback", "replaced": requested}
            )

    def substitutions(self) -> list[dict[str, str]]:
        return list(self._sh.substitutions)

    def candidates(self, role: Role | str, *, fetch: bool = True) -> list[str]:
        """Auto-selection order for ``role``: available preferred ids, then heuristic matches."""
        r = _coerce_role(role)
        pref = self._preferred.get(r, ())
        snap = self._snapshot_for(r, self.snapshot(fetch=fetch))
        gone = self._sh.unavailable
        out = [m for m in pref if m not in gone and (snap is None or m in snap.ids)]
        if snap is not None:
            out += [
                m
                for m in rank_for_role(r.value, snap.ids)
                if m not in pref and m not in gone and not self._is_deprecated(m)
            ]
        if not out and pref:
            out = [m for m in pref if m not in gone] or [pref[0]]
        return out

    def _is_deprecated(self, model: str) -> bool:
        info = self._models.get(model)
        return bool(info and info.deprecated)

    def next_candidate(self, role: Role | str, *, exclude: Iterable[str] = ()) -> str | None:
        """The next auto-selectable model for ``role`` (never fetches; skips unavailable ones)."""
        skip = set(exclude)
        return next((m for m in self.candidates(role, fetch=False) if m not in skip), None)

    def pick(self, role: Role | str, *, app: str | None = None) -> str:
        """Model id for ``role`` (see the module docstring for the precedence order)."""
        r = _coerce_role(role)
        override = self.override_for(r, app)
        if override is not None:
            self._note(r, override[0], override[1])
            return override[0]
        cands = self.candidates(r)
        if not cands:
            raise KeyError(f"no default model configured for role {r.value!r}")
        choice = cands[0]
        self._sh.auto[choice] = r
        self._note(r, choice, "auto" if choice != self._preferred.get(r, ("",))[0] else "default")
        return choice

    def _note(self, role: Role, model: str, source: str) -> None:
        if self.picks is not None and not any(
            p["role"] == role.value and p["model"] == model for p in self.picks
        ):
            self.picks.append({"role": role.value, "model": model, "source": source})

    # -- introspection (CLI and Studio)
    def stale_defaults(self, *, fetch: bool = False) -> list[dict[str, str]]:
        """Roles whose curated default is not served any more, with the substitute in use."""
        snap = self.snapshot(fetch=fetch)
        if snap is None:
            return []
        out: list[dict[str, str]] = []
        for role, ids in self._preferred.items():
            vouched = self._snapshot_for(role, snap)
            if vouched is None or not ids:
                continue
            default = ids[0]
            if default in vouched.ids and default not in self._sh.unavailable:
                continue
            subs = self.candidates(role, fetch=False)
            out.append(
                {"role": role.value, "default": default, "substitute": subs[0] if subs else ""}
            )
        return out

    def describe(self, role: Role | str, *, app: str | None = None) -> dict[str, Any]:
        """Effective model for ``role`` and its source (request, app, studio, env, toml, auto)."""
        r = _coerce_role(role)
        pref = self._preferred.get(r, ())
        override = self.override_for(r, app)
        if override is not None:
            model, source = override
        else:
            cands = self.candidates(r, fetch=False)
            model = cands[0] if cands else (pref[0] if pref else "")
            source = "default" if pref and model == pref[0] else "auto"
        return {
            "role": r.value,
            "model": model,
            "source": source,
            "default": pref[0] if pref else None,
            "substituted": source == "auto",
        }

    def models_for_role(
        self, role: Role | str, *, include_all: bool = False, fetch: bool = False
    ) -> list[dict[str, Any]]:
        """Picker options: live models matching ``role`` by name (or every live model)."""
        r = _coerce_role(role)
        snap = self.snapshot(fetch=fetch)
        gone = self._sh.unavailable
        pref = self._preferred.get(r, ())
        if snap is not None:
            if include_all:
                ids = sorted(snap.ids)
            else:
                matched = rank_for_role(r.value, snap.ids)
                ids = [m for m in pref if m in snap.ids] + [m for m in matched if m not in pref]
        else:
            ids = [m.id for m in self.by_role(r)] if not include_all else sorted(self._models)
        return [
            {
                "id": m,
                "live": snap is not None and m in snap.ids,
                "curated": m in self._models,
                "preferred": m in pref,
                "deprecated": self._is_deprecated(m),
                "unavailable": m in gone,
            }
            for m in dict.fromkeys(ids)
        ]

    # -- live availability (snapshot file)
    @staticmethod
    def load_live_ids(path: str | os.PathLike[str] | None = None) -> set[str]:
        """IDs from ``data/live_models.txt`` (one per line), if that snapshot file exists."""
        p = Path(path) if path is not None else LIVE_MODELS_FILE
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
        except OSError:
            return set()
        return {ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")}

    def missing_from(self, live_ids: Iterable[str]) -> list[str]:
        """Curated ids that are absent from ``live_ids``."""
        live = set(live_ids)
        return [i for i in self._models if i not in live]

    def replacement_for(self, model_id: str) -> str | None:
        """Suggested replacement for a retired / missing id (known mapping, else role match)."""
        if model_id in RETIRED:
            return RETIRED[model_id]
        info = self._models.get(model_id)
        if info is not None:
            for role in sorted(info.roles, key=lambda x: x.value):
                alt = next((c for c in self.candidates(role, fetch=False) if c != model_id), None)
                if alt:
                    return alt
        return None


_scope: contextvars.ContextVar[ModelRegistry | None] = contextvars.ContextVar(
    "nimlab_model_scope", default=None
)


@lru_cache(maxsize=1)
def _created_registry() -> ModelRegistry:
    return ModelRegistry()


_installed: ModelRegistry | None = None


def _process_registry() -> ModelRegistry:
    return _installed if _installed is not None else _created_registry()


def set_default_registry(registry: ModelRegistry | None) -> None:
    """Use ``registry`` as the process-wide default (``None`` restores the built-in one)."""
    global _installed
    _installed = registry


def default_registry() -> ModelRegistry:
    """Process-wide registry, or the active ``model_scope`` view."""
    scoped = _scope.get()
    return scoped if scoped is not None else _process_registry()


def reset_default_registry() -> None:
    global _installed
    _installed = None
    _created_registry.cache_clear()


default_registry.cache_clear = reset_default_registry  # type: ignore[attr-defined]


@contextlib.contextmanager
def model_scope(
    app: str | None = None, overrides: Mapping[Role | str, str] | None = None
) -> Iterator[ModelRegistry]:
    """Make ``default_registry()`` return a view with per-app / per-request model overrides.

    Everything inside the block (including tasks it creates) sees the overrides; the view records
    which model was picked for which role in ``view.picks``.
    """
    view = _process_registry().scoped(app, overrides)
    token = _scope.set(view)
    try:
        yield view
    finally:
        _scope.reset(token)
