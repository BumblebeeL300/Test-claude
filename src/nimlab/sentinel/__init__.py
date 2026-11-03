"""Sentinel: a safety and robustness lab for NVIDIA NIM models (defensive use).

Public API::

    from nimlab.sentinel import Sentinel, Shield, Redactor

    async with Sentinel() as lab:                      # offline simulation without NVIDIA_API_KEY
        run = await lab.scan(["meta/llama-3.3-70b-instruct"])
        print(run.scorecards[0].grade)

    async with Shield(client) as shield:               # screen any chat call
        reply = await shield.chat(model, messages)

    redactor = Redactor()                              # reversible PII tokenisation
    safe = redactor.redact("mail me at a@b.example").text
    original = redactor.restore(safe)

The Typer CLI lives in :mod:`nimlab.sentinel.cli` (``sentinel`` command / ``nimlab sentinel``).
Probes contain only benign canary tokens and ``[REDACTED ...]`` placeholders.
"""

from __future__ import annotations

from .judges import (
    CanaryJudge,
    CitationJudge,
    InjectionJudge,
    Judge,
    JudgeContext,
    NemoGuardJudge,
    RefusalJudge,
    available_judges,
    build_judges,
    register_judge,
)
from .lab import Sentinel
from .models import (
    ATTACK_CATEGORIES,
    CATEGORIES,
    JudgeVerdict,
    Probe,
    ProbeResult,
    ProbeSuite,
    Scorecard,
    SentinelRun,
)
from .pii import PIIMatch, Redactor, find_pii, iban_valid, luhn_valid, redact, restore
from .report import render_html, render_terminal, write_html
from .runner import Runner, ScanSettings
from .scoring import build_scorecard, classify, grade_for
from .shield import Action, CheckResult, Shield, ShieldConfig, ShieldResult
from .simulate import SimulatedClient
from .store import RunStore
from .suites import SuiteError, list_suites, load_suite, resolve_suites

__all__ = [
    "ATTACK_CATEGORIES",
    "CATEGORIES",
    "Action",
    "CanaryJudge",
    "CheckResult",
    "CitationJudge",
    "InjectionJudge",
    "Judge",
    "JudgeContext",
    "JudgeVerdict",
    "NemoGuardJudge",
    "PIIMatch",
    "Probe",
    "ProbeResult",
    "ProbeSuite",
    "Redactor",
    "RefusalJudge",
    "RunStore",
    "Runner",
    "ScanSettings",
    "Scorecard",
    "Sentinel",
    "SentinelRun",
    "Shield",
    "ShieldConfig",
    "ShieldResult",
    "SimulatedClient",
    "SuiteError",
    "available_judges",
    "build_judges",
    "build_scorecard",
    "classify",
    "find_pii",
    "grade_for",
    "iban_valid",
    "list_suites",
    "load_suite",
    "luhn_valid",
    "redact",
    "register_judge",
    "render_html",
    "render_terminal",
    "resolve_suites",
    "restore",
    "write_html",
]
