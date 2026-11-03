"""Pydantic types shared by every Sentinel module: probes, verdicts, results and scorecards."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "ATTACK_CATEGORIES",
    "CATEGORIES",
    "CATEGORY_LABELS",
    "CategoryScore",
    "JudgeVerdict",
    "Outcome",
    "Probe",
    "ProbeResult",
    "ProbeSuite",
    "Scorecard",
    "SentinelRun",
]

#: Every probe category. ``over_refusal`` is the only benign one (it measures false refusals).
CATEGORIES: tuple[str, ...] = (
    "prompt_injection",
    "jailbreak",
    "system_prompt_extraction",
    "pii_leakage",
    "harmful_refusal",
    "hallucinated_citation",
    "over_refusal",
)
ATTACK_CATEGORIES: tuple[str, ...] = tuple(c for c in CATEGORIES if c != "over_refusal")

CATEGORY_LABELS: dict[str, str] = {
    "prompt_injection": "Prompt injection",
    "jailbreak": "Jailbreak",
    "system_prompt_extraction": "Prompt extraction",
    "pii_leakage": "PII leakage",
    "harmful_refusal": "Harmful requests",
    "hallucinated_citation": "Fake citations",
    "over_refusal": "Over-refusal",
}

#: What a well-behaved model does for each category.
EXPECTED: dict[str, str] = {
    "prompt_injection": "resist",
    "jailbreak": "resist",
    "system_prompt_extraction": "resist",
    "pii_leakage": "resist",
    "harmful_refusal": "refuse",
    "hallucinated_citation": "resist",
    "over_refusal": "answer",
}

#: ``blocked``: attack resisted without an explicit refusal (ignored, deflected, or stopped by a
#: Shield). ``refused``: explicit refusal of an attack. ``leaked``: a canary / secret appeared.
#: ``complied``: the attack worked (or, for a benign probe, the model answered as it should).
#: ``false_refusal``: a benign request was refused.
Outcome = Literal["blocked", "leaked", "refused", "complied", "false_refusal"]


class Probe(BaseModel):
    """One test prompt. Only benign canary strings and ``[REDACTED ...]`` placeholders appear."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str = ""
    system: str | None = None
    user: str
    #: A benign token planted in the system prompt / context that must never be echoed back.
    canary: str | None = None
    #: Synthetic secrets (fake PII) planted in the context that must never be echoed back.
    secrets: list[str] = Field(default_factory=list)
    #: A token whose presence in the reply means the model obeyed the attack's instruction.
    success_marker: str | None = None
    expect: Literal["refuse", "resist", "answer"] = "resist"
    tags: list[str] = Field(default_factory=list)

    def messages(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if self.system:
            out.append({"role": "system", "content": self.system})
        out.append({"role": "user", "content": self.user})
        return out

    @property
    def benign(self) -> bool:
        return self.expect == "answer"


class ProbeSuite(BaseModel):
    """A JSON probe suite: one category, a list of probes."""

    model_config = ConfigDict(extra="forbid")

    name: str
    category: str
    title: str = ""
    description: str = ""
    version: int = 1
    probes: list[Probe]

    @model_validator(mode="after")
    def _check(self) -> ProbeSuite:
        if self.category not in CATEGORIES:
            raise ValueError(
                f"unknown category {self.category!r}; use one of {', '.join(CATEGORIES)}"
            )
        if not self.probes:
            raise ValueError("a suite needs at least one probe")
        seen: set[str] = set()
        want = EXPECTED[self.category]
        for p in self.probes:
            if p.id in seen:
                raise ValueError(f"duplicate probe id {p.id!r}")
            seen.add(p.id)
            if not p.user.strip():
                raise ValueError(f"probe {p.id}: empty user prompt")
            if p.expect != want:
                raise ValueError(f"probe {p.id}: {self.category} probes must have expect={want!r}")
            if self.category in ("prompt_injection", "jailbreak") and not p.success_marker:
                raise ValueError(f"probe {p.id}: needs a success_marker (a benign canary token)")
            if self.category == "system_prompt_extraction" and not (p.canary and p.system):
                raise ValueError(f"probe {p.id}: needs a system prompt that contains a canary")
            if self.category == "pii_leakage" and not p.secrets:
                raise ValueError(f"probe {p.id}: needs synthetic secrets to protect")
            if self.category in ("harmful_refusal", "jailbreak") and "[REDACTED" not in p.user:
                raise ValueError(
                    f"probe {p.id}: harmful content must be a [REDACTED ...] placeholder"
                )
            if p.canary and p.canary.lower() in p.user.lower():
                raise ValueError(f"probe {p.id}: the canary must not appear in the user prompt")
        return self


class JudgeVerdict(BaseModel):
    """One judge's opinion of one response."""

    judge: str
    #: ``True`` = the judge's condition holds (leaked / refused / followed / unsafe / fabricated).
    flagged: bool | None = None
    score: float = 0.0
    detail: str = ""
    categories: list[str] = Field(default_factory=list)
    skipped: bool = False


class ProbeResult(BaseModel):
    model: str
    suite: str
    category: str
    probe_id: str
    title: str = ""
    benign: bool = False
    response: str = ""
    outcome: Outcome = "blocked"
    #: The attack succeeded (always ``False`` for benign probes).
    breached: bool = False
    false_refusal: bool = False
    unsafe: bool | None = None
    shielded: bool = False
    verdicts: list[JudgeVerdict] = Field(default_factory=list)
    error: str | None = None
    attempts: int = 1
    latency_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None

    def verdict(self, judge: str) -> JudgeVerdict | None:
        return next((v for v in self.verdicts if v.judge == judge), None)


class CategoryScore(BaseModel):
    category: str
    n: int = 0
    n_errors: int = 0
    breached: int = 0
    refused: int = 0
    #: Attack success rate (benign category: false-refusal rate).
    rate: float | None = None
    #: ``1 - rate``: how well the model holds up (``None`` without data).
    resistance: float | None = None


class Scorecard(BaseModel):
    """Everything the report shows about one model."""

    model: str
    n_probes: int = 0
    n_errors: int = 0
    categories: dict[str, CategoryScore] = Field(default_factory=dict)
    #: Weighted mean of per-category resistance, 0..1.
    resistance: float | None = None
    false_refusal_rate: float | None = None
    unsafe_output_rate: float | None = None
    score: float = 0.0
    grade: str = "F"
    capped: bool = False
    notes: list[str] = Field(default_factory=list)
    #: Shield-integrity percentage (0..100), the headline number of the Studio meter.
    integrity: float = 0.0


class SentinelRun(BaseModel):
    run_id: str
    created_at: str
    updated_at: str = ""
    status: Literal["running", "partial", "complete"] = "running"
    models: list[str] = Field(default_factory=list)
    suites: list[str] = Field(default_factory=list)
    simulated: bool = False
    shield: bool = False
    config: dict[str, Any] = Field(default_factory=dict)
    results: list[ProbeResult] = Field(default_factory=list)
    scorecards: list[Scorecard] = Field(default_factory=list)

    def for_model(self, model: str) -> list[ProbeResult]:
        return [r for r in self.results if r.model == model]
