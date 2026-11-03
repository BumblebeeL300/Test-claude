"""Shield: middleware that wraps a chat call with input and output safety checks.

::

    async with Shield(client) as shield:
        reply = await shield.chat("nvidia/nemotron-3-super-120b-a12b", messages)
        print(reply.content, reply.blocked, [c.name for c in reply.checks if c.triggered])

Checks (each one can be ``block``, ``redact``, ``annotate`` or ``off``):

=================  ======  ==========================================================
``input_injection``  in   injection heuristics on user / tool messages
``input_pii``        in   emails, phones, cards (Luhn), IBANs, IPs and API keys
``input_guard``      in   NemoGuard content safety on the user's text
``output_guard``     out  NemoGuard content safety on the reply
``output_leak``      out  configured canaries, or a verbatim stretch of the system prompt
``output_pii``       out  the same PII detectors on the reply
=================  ======  ==========================================================

``redact`` on ``input_pii`` swaps PII for tokens before the model sees it and restores them in the
reply (``restore_input_pii``), so the provider never receives the raw values. For the NemoGuard
checks ``redact`` behaves like ``block`` (a flagged text cannot be partially redacted).
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from nimlab.core import NIMConfig
from nimlab.core.registry import ModelRegistry
from nimlab.core.results import ChatResult

from .guard import check_safety
from .injection import expand_span, injection_scan
from .leaks import canary_hits, prompt_overlap
from .llm import guard_model as default_guard_model
from .llm import make_client, message_text
from .pii import Redactor, find_pii

__all__ = ["Action", "CheckResult", "Shield", "ShieldConfig", "ShieldResult"]

INJECTION_MARK = "[removed: suspected prompt injection]"
LEAK_MARK = "[REDACTED]"
PROMPT_MARK = "[REDACTED: system prompt]"
MAX_GUARD_CHARS = 6000


Applied = Literal["none", "blocked", "redacted", "annotated"]


class Action(str, Enum):
    OFF = "off"
    ANNOTATE = "annotate"
    REDACT = "redact"
    BLOCK = "block"


class ShieldConfig(BaseModel):
    input_injection: Action = Action.BLOCK
    input_pii: Action = Action.ANNOTATE
    input_guard: Action = Action.BLOCK
    output_guard: Action = Action.BLOCK
    output_leak: Action = Action.REDACT
    output_pii: Action = Action.REDACT
    #: Injection score (0..1) from which the input counts as an injection attempt.
    injection_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    #: Strings that must never appear in a reply (planted canaries, secrets).
    canaries: list[str] = Field(default_factory=list)
    #: PII kinds to look for (default: all, see ``nimlab.sentinel.pii.PII_KINDS``).
    pii_kinds: list[str] | None = None
    #: Do not flag PII in a reply that the user already typed in the request.
    allow_echo_input_pii: bool = True
    #: With ``input_pii=redact``: put the original values back into the reply.
    restore_input_pii: bool = True
    input_roles: list[str] = Field(default_factory=lambda: ["user", "tool"])
    guard_model: str | None = None
    block_message: str = "This request was blocked by Shield ({reasons})."

    @classmethod
    def all(cls, action: Action) -> ShieldConfig:
        """Every check set to ``action``."""
        return cls(
            input_injection=action,
            input_pii=action,
            input_guard=action,
            output_guard=action,
            output_leak=action,
            output_pii=action,
        )

    @classmethod
    def detect_only(cls) -> ShieldConfig:
        """Annotate everything, change nothing (a dry run to see what Shield would catch)."""
        return cls.all(Action.ANNOTATE)


class CheckResult(BaseModel):
    name: str
    stage: Literal["input", "output"]
    action: Action
    triggered: bool = False
    skipped: bool = False
    #: What was done about it: ``none`` | ``blocked`` | ``redacted`` | ``annotated``.
    applied: Applied = "none"
    detail: str = ""
    findings: list[str] = Field(default_factory=list)


class ShieldResult(ChatResult):
    """A :class:`~nimlab.core.ChatResult` plus what Shield did."""

    blocked: bool = False
    blocked_by: list[str] = Field(default_factory=list)
    checks: list[CheckResult] = Field(default_factory=list)
    annotations: list[str] = Field(default_factory=list)
    redactions: int = 0

    def check(self, name: str) -> CheckResult | None:
        return next((c for c in self.checks if c.name == name), None)


def _map_text(messages: list[dict[str, Any]], roles: set[str], fn: Any) -> list[dict[str, Any]]:
    """Copy of ``messages`` with ``fn(text)`` applied to the text of messages with these roles."""
    out: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") not in roles:
            out.append(msg)
            continue
        content = msg.get("content")
        if isinstance(content, str):
            out.append({**msg, "content": fn(content)})
        elif isinstance(content, list):
            parts = [
                {**p, "text": fn(str(p.get("text", "")))}
                if isinstance(p, dict) and p.get("type", "text") == "text"
                else p
                for p in content
            ]
            out.append({**msg, "content": parts})
        else:
            out.append(msg)
    return out


def _replace_spans(text: str, spans: list[tuple[int, int]], mark: str) -> str:
    merged: list[tuple[int, int]] = []
    for s, e in sorted(spans):
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    out, pos = [], 0
    for s, e in merged:
        out += [text[pos:s], mark]
        pos = e
    out.append(text[pos:])
    return "".join(out)


class Shield:
    """Wraps ``client.chat`` with the checks in :class:`ShieldConfig`.

    ``client`` is anything with ``async chat(model, messages, **kw)``; without one, Shield builds
    its own (the offline simulator in mock mode) and closes it on exit.
    """

    def __init__(
        self,
        client: Any = None,
        config: ShieldConfig | None = None,
        *,
        mock: bool | None = None,
        nim_config: NIMConfig | None = None,
        registry: ModelRegistry | None = None,
        **overrides: Any,
    ) -> None:
        base = config or ShieldConfig()
        self.config = base.model_copy(update=overrides) if overrides else base
        self._owns_client = client is None
        self.client = client if client is not None else make_client(mock=mock, config=nim_config)
        self._registry = registry
        self.stats: dict[str, int] = {"calls": 0, "blocked": 0, "redactions": 0}

    async def __aenter__(self) -> Shield:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client and hasattr(self.client, "aclose"):
            await self.client.aclose()

    @property
    def guard_model(self) -> str:
        return self.config.guard_model or default_guard_model(self._registry)

    # ------------------------------------------------------------------ input checks
    def _input_text(self, messages: list[dict[str, Any]]) -> str:
        roles = set(self.config.input_roles)
        return "\n".join(message_text(m) for m in messages if m.get("role") in roles)

    async def check_input(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[CheckResult], dict[str, str]]:
        """Run the input checks. Returns ``(messages to send, checks, token mapping)``."""
        cfg = self.config
        roles = set(cfg.input_roles)
        checks: list[CheckResult] = []
        mapping: dict[str, str] = {}
        msgs = messages

        # -- injection heuristics
        if cfg.input_injection is not Action.OFF:
            score, rules = 0.0, []
            for m in msgs:
                if m.get("role") in roles:
                    scan = injection_scan(message_text(m))
                    score = max(score, scan.score)
                    rules += [r for r in scan.rules if r not in rules]
            hit = score >= cfg.injection_threshold
            res = CheckResult(
                name="input_injection",
                stage="input",
                action=cfg.input_injection,
                triggered=hit,
                detail=f"injection score {score:.2f}" + (f" ({', '.join(rules)})" if hit else ""),
                findings=rules if hit else [],
            )
            if hit:
                if cfg.input_injection is Action.REDACT:

                    def strip(text: str) -> str:
                        scan = injection_scan(text)
                        if scan.score < cfg.injection_threshold:
                            return text
                        spans = [expand_span(text, h.start, h.end) for h in scan.hits]
                        return _replace_spans(text, spans, INJECTION_MARK)

                    msgs = _map_text(msgs, roles, strip)
                    res.applied = "redacted"
                else:
                    res.applied = _applied(cfg.input_injection)
            checks.append(res)

        # -- PII in the request
        if cfg.input_pii is not Action.OFF:
            found = find_pii(self._input_text(msgs), cfg.pii_kinds)
            kinds = sorted({m.kind for m in found})
            res = CheckResult(
                name="input_pii",
                stage="input",
                action=cfg.input_pii,
                triggered=bool(found),
                detail=f"{len(found)} PII item(s)" if found else "no PII found",
                findings=kinds,
            )
            if found:
                if cfg.input_pii is Action.REDACT:
                    redactor = Redactor(style="token", kinds=cfg.pii_kinds)
                    msgs = _map_text(msgs, roles, lambda t: redactor.redact(t).text)
                    mapping = dict(redactor.mapping)
                    res.applied = "redacted"
                else:
                    res.applied = _applied(cfg.input_pii)
            checks.append(res)

        # -- NemoGuard on the user's text
        already_blocked = any(c.applied == "blocked" for c in checks)
        if cfg.input_guard is not Action.OFF and not already_blocked:
            text = self._input_text(msgs)[:MAX_GUARD_CHARS]
            g = await check_safety(self.client, self.guard_model, text)
            res = CheckResult(
                name="input_guard",
                stage="input",
                action=cfg.input_guard,
                skipped=g.safe is None,
                triggered=g.safe is False,
                detail=g.detail
                or ("unsafe: " + ", ".join(g.categories) if g.safe is False else "safe"),
                findings=g.categories,
            )
            if g.safe is False:
                res.applied = "blocked" if cfg.input_guard is not Action.ANNOTATE else "annotated"
            checks.append(res)
        return msgs, checks, mapping

    # ------------------------------------------------------------------ output checks
    async def check_output(
        self, text: str, messages: list[dict[str, Any]]
    ) -> tuple[str, list[CheckResult], int]:
        """Run the output checks on a reply. Returns ``(text, checks, redaction count)``."""
        cfg = self.config
        checks: list[CheckResult] = []
        redactions = 0

        if cfg.output_guard is not Action.OFF:
            g = await check_safety(
                self.client, self.guard_model, self._input_text(messages)[:MAX_GUARD_CHARS], text
            )
            res = CheckResult(
                name="output_guard",
                stage="output",
                action=cfg.output_guard,
                skipped=g.safe is None,
                triggered=g.safe is False,
                detail=g.detail
                or ("unsafe: " + ", ".join(g.categories) if g.safe is False else "safe"),
                findings=g.categories,
            )
            if g.safe is False:
                res.applied = "blocked" if cfg.output_guard is not Action.ANNOTATE else "annotated"
            checks.append(res)

        if cfg.output_leak is not Action.OFF:
            system = "\n".join(message_text(m) for m in messages if m.get("role") == "system")
            hits = canary_hits(text, cfg.canaries)
            spans = prompt_overlap(system, text) if system else []
            findings = [*hits, *(["<system prompt>"] if spans else [])]
            res = CheckResult(
                name="output_leak",
                stage="output",
                action=cfg.output_leak,
                triggered=bool(findings),
                detail="leak: " + ", ".join(findings) if findings else "no canary or prompt leak",
                findings=findings,
            )
            if findings:
                if cfg.output_leak is Action.REDACT:
                    for c in hits:
                        text, n = re.subn(re.escape(c), LEAK_MARK, text, flags=re.I)
                        redactions += n
                    spans = prompt_overlap(system, text) if system else []
                    if spans:
                        text = _replace_spans(text, spans, PROMPT_MARK)
                        redactions += len(spans)
                    res.applied = "redacted"
                else:
                    res.applied = _applied(cfg.output_leak)
            checks.append(res)

        if cfg.output_pii is not Action.OFF:
            echoed = self._input_text(messages) if cfg.allow_echo_input_pii else ""
            found = [m for m in find_pii(text, cfg.pii_kinds) if not (echoed and m.text in echoed)]
            res = CheckResult(
                name="output_pii",
                stage="output",
                action=cfg.output_pii,
                triggered=bool(found),
                detail=f"{len(found)} PII item(s) in the reply" if found else "no PII in the reply",
                findings=sorted({m.kind for m in found}),
            )
            if found:
                if cfg.output_pii is Action.REDACT:
                    pieces, pos = [], 0
                    for m in found:
                        pieces += [text[pos : m.start], f"[REDACTED:{m.kind.upper()}]"]
                        pos = m.end
                    pieces.append(text[pos:])
                    text = "".join(pieces)
                    redactions += len(found)
                    res.applied = "redacted"
                else:
                    res.applied = _applied(cfg.output_pii)
            checks.append(res)
        return text, checks, redactions

    # ------------------------------------------------------------------ the wrapped call
    def _blocked(self, model: str, checks: list[CheckResult]) -> ShieldResult:
        reasons = [c.name for c in checks if c.applied == "blocked"]
        self.stats["blocked"] += 1
        return ShieldResult(
            model=model,
            content=self.config.block_message.format(reasons=", ".join(reasons) or "policy"),
            finish_reason="shield_blocked",
            blocked=True,
            blocked_by=reasons,
            checks=checks,
        )

    async def chat(self, model: str, messages: list[dict[str, Any]], **kw: Any) -> ShieldResult:
        """Like ``client.chat`` but screened. Check ``result.blocked`` before trusting the text."""
        self.stats["calls"] += 1
        sent, in_checks, mapping = await self.check_input(messages)
        if any(c.applied == "blocked" for c in in_checks):
            return self._blocked(model, in_checks)
        result = await self.client.chat(model, sent, **kw)
        text, out_checks, redactions = await self.check_output(result.content, messages)
        checks = [*in_checks, *out_checks]
        if any(c.applied == "blocked" for c in out_checks):
            blocked = self._blocked(model, checks)
            return blocked.model_copy(update={"usage": result.usage, "latency_s": result.latency_s})
        if mapping and self.config.restore_input_pii:
            text = Redactor().restore(text, mapping)
        self.stats["redactions"] += redactions
        annotations = [f"{c.name}: {c.detail}" for c in checks if c.applied == "annotated"]
        return ShieldResult(
            **{**result.model_dump(), "content": text},
            checks=checks,
            annotations=annotations,
            redactions=redactions,
        )


_APPLIED: dict[Action, Applied] = {
    Action.BLOCK: "blocked",
    Action.REDACT: "redacted",
    Action.ANNOTATE: "annotated",
    Action.OFF: "none",
}


def _applied(action: Action) -> Applied:
    return _APPLIED[action]
