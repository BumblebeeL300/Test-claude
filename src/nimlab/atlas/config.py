"""Tunable settings for Atlas. Every knob has a sensible default; see ``docs/atlas.md``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nimlab.core import ModelRegistry, Role, default_registry

DEFAULT_DB_PATH = Path(".nimlab") / "atlas.db"
DEFAULT_COLLECTION = "default"


class AtlasConfig(BaseModel):
    """Settings for ingestion, retrieval, generation and verification."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # -- chunking
    chunk_max_chars: int = Field(1200, ge=100, description="Maximum characters per chunk")
    chunk_overlap: int = Field(150, ge=0, description="Characters carried over between chunks")
    chunk_min_chars: int = Field(200, ge=0, description="Chunks smaller than this are merged")

    # -- embedding
    embed_batch_size: int = Field(64, ge=1, le=512, description="Texts per embed() call")

    # -- BM25
    bm25_k1: float = Field(1.5, ge=0.0)
    bm25_b: float = Field(0.75, ge=0.0, le=1.0)

    # -- retrieval
    k: int = Field(6, ge=1, le=50, description="Passages handed to the model")
    candidates: int = Field(30, ge=1, description="Per-retriever candidate pool (dense and BM25)")
    rerank_candidates: int = Field(20, ge=1, description="Fused candidates sent to the reranker")
    rrf_k: int = Field(60, ge=1, description="Reciprocal Rank Fusion constant")
    dense_weight: float = Field(1.0, ge=0.0)
    bm25_weight: float = Field(1.0, ge=0.0)
    use_rerank: bool = True
    use_mmr: bool = True
    mmr_lambda: float = Field(0.7, ge=0.0, le=1.0, description="1 = pure relevance, 0 = diversity")
    context_chars: int = Field(6000, ge=200, description="Character budget for packed context")

    # -- agentic
    max_subquestions: int = Field(4, ge=1, le=8)
    max_hops: int = Field(2, ge=0, le=5, description="Follow-up search rounds after the first")
    hyde: bool = False
    hyde_weight: float = Field(0.5, ge=0.0, le=1.0, description="Weight of the HyDE vector")
    agentic_context_factor: float = Field(1.5, ge=1.0, description="Extra context for agentic mode")

    # -- generation
    temperature: float = Field(0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(1024, ge=64)

    # -- verification
    verify_concurrency: int = Field(4, ge=1)
    max_claims: int = Field(12, ge=1, description="Claims verified per answer (rest are skipped)")
    revise_rounds: int = Field(1, ge=1, le=3)

    # -- guard
    guard_fail_open: bool | None = Field(
        None, description="Allow content when the safety reply is unparseable (None: only in mock)"
    )

    # -- model overrides by role name ("chat", "fast", "embed", "rerank", "safety", ...)
    models: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> AtlasConfig:
        if self.chunk_overlap >= self.chunk_max_chars:
            raise ValueError("chunk_overlap must be smaller than chunk_max_chars")
        if self.chunk_min_chars >= self.chunk_max_chars:
            raise ValueError("chunk_min_chars must be smaller than chunk_max_chars")
        return self

    def chunk_signature(self, embed_tag: str) -> str:
        """Fingerprint of everything that changes a document's chunks / vectors."""
        return f"c{self.chunk_max_chars}:{self.chunk_overlap}:{self.chunk_min_chars}|{embed_tag}"


@dataclass(frozen=True)
class ModelSet:
    """The concrete model ids Atlas uses, one per job."""

    chat: str  # grounded answering, verification, revision
    fast: str  # query planning, HyDE, sufficiency checks
    embed: str
    rerank: str
    safety: str

    @classmethod
    def resolve(cls, config: AtlasConfig, registry: ModelRegistry | None = None) -> ModelSet:
        """Pick models from the registry; ``config.models`` (keyed by role name) wins."""
        reg = registry or default_registry()

        def pick(role: Role) -> str:
            return config.models.get(role.value) or reg.pick(role)

        return cls(
            chat=pick(Role.CHAT),
            fast=pick(Role.FAST),
            embed=pick(Role.EMBED),
            rerank=pick(Role.RERANK),
            safety=pick(Role.SAFETY),
        )
