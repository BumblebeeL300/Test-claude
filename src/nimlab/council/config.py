"""Run configuration for a council."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Severity = Literal["low", "medium", "high"]


class CouncilConfig(BaseModel):
    """Knobs shared by every strategy. Anything left ``None`` is resolved from the registry."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # --- who takes part
    models: list[str] | None = Field(
        default=None, description="Panel of models (default: registry)"
    )
    aggregator: str | None = Field(default=None, description="MoA final synthesiser")
    judge: str | None = Field(default=None, description="Debate judge / rubric judge")
    drafter: str | None = Field(
        default=None, description="Critic loop: first draft (default models[0])"
    )
    critic: str | None = Field(
        default=None, description="Critic loop: reviewer (default models[1])"
    )
    reviser: str | None = Field(default=None, description="Critic loop: reviser (default drafter)")
    reward_model: str | None = Field(default=None, description="Best-of-N reward model")
    embed_model: str | None = Field(default=None, description="Embedding model for consensus")

    # --- strategy shape
    layers: int = Field(default=2, ge=0, le=8, description="MoA aggregation layers")
    rounds: int = Field(default=2, ge=0, le=10, description="Debate rounds")
    n: int = Field(default=4, ge=1, le=32, description="Best-of-N candidate count")
    max_iterations: int = Field(default=3, ge=0, le=10, description="Critic loop iterations")
    critic_stop_severity: Severity = Field(
        default="high", description="Critic loop stops when no issue is at/above this severity"
    )
    debate_critique: bool = Field(default=True, description="Debate: separate critique phase")
    convergence_threshold: float = Field(
        default=0.9, ge=0.0, le=1.0, description="Debate early-stop agreement"
    )
    early_stop: bool = True
    use_reward_model: bool = Field(
        default=True, description="Best-of-N: try the reward model first"
    )

    # --- sampling
    temperature: float = Field(default=0.5, ge=0.0, le=2.0)
    aggregator_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    temperature_range: tuple[float, float] = (0.3, 1.0)
    max_tokens: int = Field(default=1024, ge=16)
    reasoning_token_multiplier: float = Field(
        default=3.0, ge=1.0, description="Token budget multiplier for <think> models"
    )
    system_prompt: str | None = None

    # --- consensus / outliers
    outlier_z: float = Field(default=1.0, ge=0.0)
    outlier_margin: float = Field(default=0.1, ge=0.0)

    # --- plumbing
    record_prompts: bool = True
    max_parallel: int | None = Field(default=None, ge=1, description="Extra local concurrency cap")

    @field_validator("models")
    @classmethod
    def _clean_models(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return None
        cleaned = [m.strip() for m in v if m and m.strip()]
        return cleaned or None

    def merged(self, **overrides: Any) -> CouncilConfig:
        """Copy with ``overrides`` applied (``None`` values are ignored); validates everything."""
        unknown = sorted(set(overrides) - set(type(self).model_fields))
        if unknown:
            valid = ", ".join(sorted(type(self).model_fields))
            raise TypeError(f"unknown council option(s): {', '.join(unknown)}; valid: {valid}")
        data = self.model_dump()
        data.update({k: v for k, v in overrides.items() if v is not None})
        return CouncilConfig.model_validate(data)
