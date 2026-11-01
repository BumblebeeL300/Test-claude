"""Curated catalog of NVIDIA NIM models with role-based defaults and user overrides."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable, Iterator, Mapping
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on 3.10 only
    import tomli as tomllib

__all__ = ["ModelInfo", "ModelRegistry", "Role", "default_registry", "reset_default_registry"]


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
    """One catalog entry. ``cost_tier`` is relative: 1 (cheapest/fastest) to 5 (largest)."""

    model_config = ConfigDict(frozen=True)

    id: str
    roles: frozenset[Role]
    context_window: int
    notes: str = ""
    cost_tier: int = Field(ge=1, le=5)


def _m(id_: str, roles: Iterable[Role], ctx: int, tier: int, notes: str = "") -> ModelInfo:
    return ModelInfo(
        id=id_, roles=frozenset(roles), context_window=ctx, cost_tier=tier, notes=notes
    )


_C, _R, _K, _V = Role.CHAT, Role.REASONING, Role.CODER, Role.VISION
_E, _RR, _RW, _S, _F = Role.EMBED, Role.RERANK, Role.REWARD, Role.SAFETY, Role.FAST

# Curated from the public NIM catalog (build.nvidia.com). Not every model is enabled for every
# account/endpoint: use `nimlab models --live` to see which of these your endpoint serves.
CURATED_MODELS: tuple[ModelInfo, ...] = (
    _m("meta/llama-3.3-70b-instruct", [_C], 131072, 2, "Strong general-purpose default"),
    _m("meta/llama-3.1-405b-instruct", [_C], 131072, 5, "Largest Llama; best quality, slowest"),
    _m("meta/llama-3.1-8b-instruct", [_C, _F], 131072, 1, "Small, fast, cheap"),
    _m("meta/llama-3.2-3b-instruct", [_F], 131072, 1, "Tiny and very fast"),
    _m("meta/llama-3.2-1b-instruct", [_F], 131072, 1, "Smallest Llama"),
    _m("meta/llama-3.2-90b-vision-instruct", [_V, _C], 131072, 4, "Image + text input"),
    _m("meta/llama-3.2-11b-vision-instruct", [_V], 131072, 2, "Smaller vision model"),
    _m("meta/llama-4-maverick-17b-128e-instruct", [_C, _V], 1048576, 3, "MoE, multimodal"),
    _m("meta/llama-4-scout-17b-16e-instruct", [_C, _V], 1048576, 2, "MoE, very long context"),
    _m(
        "nvidia/llama-3.3-nemotron-super-49b-v1",
        [_C, _R],
        131072,
        3,
        "Reasoning toggled by system prompt 'detailed thinking on/off'",
    ),
    _m(
        "nvidia/llama-3.1-nemotron-ultra-253b-v1",
        [_C, _R],
        131072,
        5,
        "Flagship Nemotron reasoning model",
    ),
    _m("nvidia/llama-3.1-nemotron-70b-instruct", [_C], 131072, 2, "Helpful-tuned Llama 3.1 70B"),
    _m("deepseek-ai/deepseek-r1", [_R], 131072, 4, "Emits <think> blocks (stripped by the client)"),
    _m("deepseek-ai/deepseek-r1-distill-qwen-32b", [_R], 131072, 2, "Distilled R1; emits <think>"),
    _m("qwen/qwq-32b", [_R], 131072, 3, "Reasoning model; emits <think>"),
    _m("qwen/qwen2.5-coder-32b-instruct", [_K], 32768, 2, "Code generation and review"),
    _m("mistralai/codestral-22b-instruct-v0.1", [_K], 32768, 2, "Code model with fill-in-middle"),
    _m("mistralai/mixtral-8x22b-instruct-v0.1", [_C], 65536, 3, "MoE general model"),
    _m("mistralai/mistral-large-2-instruct", [_C], 131072, 4, "Mistral flagship (123B)"),
    _m("microsoft/phi-4-mini-instruct", [_F, _C], 131072, 1, "3.8B, fast, good for routing"),
    _m("google/gemma-2-27b-it", [_C], 8192, 2, "Short context"),
    _m("openai/gpt-oss-120b", [_C, _R], 131072, 3, "Open-weight reasoning MoE"),
    _m("openai/gpt-oss-20b", [_C, _R, _F], 131072, 1, "Small open-weight reasoning MoE"),
    _m(
        "nvidia/llama-3.2-nv-embedqa-1b-v2",
        [_E],
        8192,
        1,
        "Multilingual retrieval embeddings (use input_type=query|passage)",
    ),
    _m("nvidia/nv-embedqa-e5-v5", [_E], 512, 1, "English retrieval embeddings, 1024-d"),
    _m("baai/bge-m3", [_E], 8192, 1, "Multilingual embeddings, long inputs"),
    _m(
        "nvidia/llama-3.2-nv-rerankqa-1b-v2",
        [_RR],
        8192,
        1,
        "Passage reranker; served from the retrieval API",
    ),
    _m(
        "nvidia/llama-3.1-nemotron-70b-reward",
        [_RW],
        4096,
        2,
        "Scores a conversation; returns a reward as the assistant message",
    ),
    _m("nvidia/llama-3.1-nemoguard-8b-content-safety", [_S], 8192, 1, "Content-safety classifier"),
    _m("nvidia/llama-3.1-nemoguard-8b-topic-control", [_S], 8192, 1, "Topic-control guardrail"),
)

DEFAULTS: dict[Role, str] = {
    Role.CHAT: "meta/llama-3.3-70b-instruct",
    Role.REASONING: "deepseek-ai/deepseek-r1",
    Role.CODER: "qwen/qwen2.5-coder-32b-instruct",
    Role.VISION: "meta/llama-3.2-90b-vision-instruct",
    Role.EMBED: "nvidia/llama-3.2-nv-embedqa-1b-v2",
    Role.RERANK: "nvidia/llama-3.2-nv-rerankqa-1b-v2",
    Role.REWARD: "nvidia/llama-3.1-nemotron-70b-reward",
    Role.SAFETY: "nvidia/llama-3.1-nemoguard-8b-content-safety",
    Role.FAST: "meta/llama-3.1-8b-instruct",
}

LIVE_MODELS_FILE = Path(__file__).parent / "data" / "live_models.txt"


def _coerce_role(role: Role | str) -> Role:
    if isinstance(role, Role):
        return role
    try:
        return Role(str(role).strip().lower())
    except ValueError:
        valid = ", ".join(r.value for r in Role)
        raise ValueError(f"unknown role {role!r}; valid roles: {valid}") from None


class ModelRegistry:
    """Catalog + role defaults.

    ``pick(role)`` precedence: env ``NIMLAB_MODEL_<ROLE>`` > ``[models]`` table of ``nimlab.toml``
    (``./nimlab.toml`` or ``$NIMLAB_CONFIG``) > built-in default. Overrides may name models that are
    not in the catalog.
    """

    def __init__(
        self,
        models: Iterable[ModelInfo] | None = None,
        defaults: Mapping[Role, str] | None = None,
        *,
        config_path: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._models: dict[str, ModelInfo] = {
            m.id: m for m in (CURATED_MODELS if models is None else models)
        }
        self._defaults: dict[Role, str] = dict(DEFAULTS if defaults is None else defaults)
        self._config_path = Path(config_path) if config_path is not None else None
        self._env = env

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
            self._defaults[_coerce_role(default_for)] = info.id

    # -- role selection
    def by_role(self, role: Role | str) -> list[ModelInfo]:
        """Models with ``role``: the current pick first (if curated), then by cost tier, id."""
        r = _coerce_role(role)
        picked = self._safe_pick(r)
        found = [m for m in self._models.values() if r in m.roles]
        found.sort(key=lambda m: (m.id != picked, m.cost_tier, m.id))
        return found

    def _safe_pick(self, role: Role) -> str | None:
        try:
            return self.pick(role)
        except KeyError:
            return None

    def _toml_models(self) -> Mapping[str, Any]:
        path = self._config_path
        if path is None:
            env_path = (self._env if self._env is not None else os.environ).get("NIMLAB_CONFIG")
            path = Path(env_path) if env_path else Path.cwd() / "nimlab.toml"
        try:
            with path.open("rb") as fh:
                data = tomllib.load(fh)
        except (OSError, tomllib.TOMLDecodeError):
            return {}
        table = data.get("models", {})
        return table if isinstance(table, dict) else {}

    def pick(self, role: Role | str) -> str:
        """Default model id for ``role`` (see class docstring for override precedence)."""
        r = _coerce_role(role)
        env = self._env if self._env is not None else os.environ
        override = env.get(f"NIMLAB_MODEL_{r.name}", "").strip()
        if override:
            return override
        table = self._toml_models()
        for key in (r.value, r.name, r.value.upper()):
            value = table.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        try:
            return self._defaults[r]
        except KeyError:
            raise KeyError(f"no default model configured for role {r.value!r}") from None

    # -- live availability
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


@lru_cache(maxsize=1)
def default_registry() -> ModelRegistry:
    """Process-wide registry (env / toml overrides are re-read on every ``pick``)."""
    return ModelRegistry()


def reset_default_registry() -> None:
    default_registry.cache_clear()
