"""Offline simulator used in mock mode so that Arena produces *meaningful* demo results.

The core library's mock backend returns generic prose, which every grader would score as wrong.
:class:`SimulatedClient` recognises benchmark prompts from the loaded suites and answers them like
a model of a given strength would: deterministically correct or incorrect depending on a hash of
(model, item) and the model's skill, with realistic (virtual) time-to-first-token and throughput.
It also acts as the judge (token overlap with the reference) and as the task classifier.

Everything else (arbitrary chat, tool calls, JSON schemas, embeddings) is delegated to the core
mock client, so the proxy keeps working for any prompt.

Timing is *virtual*: the client really sleeps for ``virtual_seconds * time_scale`` and advertises
``time_scale`` so that the runner can report the virtual figures (a 40 ms sleep is reported as 1 s
at the default scale of 0.04).
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel

from nimlab.core import NIMClient, NIMConfig, Role, default_registry
from nimlab.core.registry import ModelRegistry
from nimlab.core.results import ChatResult, Usage
from nimlab.core.usage import UsageTracker

from .llm import estimate_tokens, message_text, word_set
from .suites import EvalItem, Suite, load_suites

__all__ = ["ModelProfile", "SimulatedClient", "model_profile"]

T = TypeVar("T", bound=BaseModel)

DEFAULT_TIME_SCALE = 0.04
_WRONG_TEXT = "I'm not certain about this one, so I will not commit to a specific answer."


def _u01(*parts: str) -> float:
    """Deterministic uniform number in [0, 1) from the given strings."""
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


@dataclass(frozen=True)
class ModelProfile:
    """How a simulated model behaves."""

    model: str
    tier: int
    skill: float
    ttft_s: float
    tokens_per_s: float
    category_bonus: dict[str, float]

    def p_correct(self, category: str) -> float:
        return min(0.98, max(0.05, self.skill + self.category_bonus.get(category, 0.0)))


def model_profile(model: str, registry: ModelRegistry | None = None) -> ModelProfile:
    """Strength / speed profile derived from the registry entry (or a hash for unknown models).

    Bigger (higher cost tier) models are better but slower; coder models are strong at code and
    reasoning models at math, at the price of a long time-to-first-token.
    """
    reg = registry or default_registry()
    info = reg.find(model)
    jitter = _u01("profile", model)
    tier = info.cost_tier if info else 1 + int(jitter * 5) % 5
    roles = set(info.roles) if info else set()
    skill = 0.50 + 0.085 * tier + (jitter - 0.5) * 0.10
    bonus: dict[str, float] = {}
    if Role.CODER in roles:
        bonus["coding"] = 0.22
        bonus["math_reasoning"] = -0.05
    if Role.REASONING in roles:
        bonus["math_reasoning"] = 0.18
        bonus["factual_qa"] = 0.03
    if Role.FAST in roles and tier == 1:
        bonus["json_compliance"] = 0.04  # small models are good at rigid formats
    lowered = model.lower()
    if "coder" in lowered or "codestral" in lowered:
        bonus.setdefault("coding", 0.22)
    ttft = 0.18 + 0.22 * tier
    if Role.REASONING in roles or "r1" in lowered or "qwq" in lowered:
        ttft *= 3.0  # thinking before the first visible token
    tps = 170.0 / (0.6 + 0.4 * tier)
    return ModelProfile(model, tier, skill, ttft, tps, bonus)


def _f1(candidate: str, reference: str) -> float:
    c, r = word_set(candidate), word_set(reference)
    if not c or not r:
        return 0.0
    inter = len(c & r)
    if not inter:
        return 0.0
    precision, recall = inter / len(c), inter / len(r)
    return 2 * precision * recall / (precision + recall)


class SimulatedClient:
    """Drop-in for :class:`nimlab.core.NIMClient` that needs no network and knows the suites."""

    def __init__(
        self,
        suites: dict[str, Suite] | None = None,
        *,
        config: NIMConfig | None = None,
        registry: ModelRegistry | None = None,
        time_scale: float = DEFAULT_TIME_SCALE,
    ) -> None:
        self.config = config or NIMConfig.from_env(mock=True)
        self.registry = registry or default_registry()
        self.time_scale = time_scale
        self.usage = UsageTracker()
        self._base = NIMClient(self.config.model_copy(update={"mock": True}))
        self._index: dict[str, tuple[Suite, EvalItem]] = {}
        self.register_suites(suites if suites is not None else load_suites())

    # ------------------------------------------------------------------ lifecycle
    async def __aenter__(self) -> SimulatedClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._base.aclose()

    @property
    def mock(self) -> bool:
        return True

    def register_suites(self, suites: dict[str, Suite] | list[Suite]) -> None:
        """Teach the simulator the prompts (and gold answers) of more suites."""
        for suite in suites.values() if isinstance(suites, dict) else suites:
            for item in suite.items:
                self._index[item.prompt.strip()] = (suite, item)

    # ------------------------------------------------------------------ answer synthesis
    def _lookup(self, messages: list[dict[str, Any]]) -> tuple[Suite, EvalItem] | None:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return self._index.get(message_text(msg).strip())
        return None

    def _answer(self, model: str, suite: Suite, item: EvalItem) -> str:
        prof = model_profile(model, self.registry)
        u = _u01("answer", model, suite.name, item.id)
        if suite.grader_for(item).type == "llm_judge":
            return self._summary(prof, suite, item, u)
        if u < prof.p_correct(suite.category):
            return item.reference or self._synthesize(suite, item)
        return self._wrong(suite, item)

    @staticmethod
    def _synthesize(suite: Suite, item: EvalItem) -> str:
        """A correct answer for simple graders when the item carries no ``reference``."""
        spec = suite.grader_for(item)
        params = spec.params
        if spec.type == "numeric":
            return f"Answer: {float(params['expected']):g}"
        if spec.type == "exact":
            expected = params["expected"]
            return expected if isinstance(expected, str) else str(expected[0])
        if spec.type == "contains_all":
            words = [*params.get("required", []), *(g[0] for g in params.get("any_of", []))]
            return " ".join(words)
        return ""

    def _wrong(self, suite: Suite, item: EvalItem) -> str:
        spec = suite.grader_for(item)
        if spec.type == "numeric":
            expected = float(spec.params["expected"])
            return f"Quick estimate.\nAnswer: {expected * 1.5 + 3:g}"
        if spec.type == "code_exec":
            name = spec.params.get("entrypoint") or "solution"
            body = (
                f"class {name}:\n    pass"
                if name[:1].isupper()
                else f"def {name}(*a, **k):\n    return None"
            )
            return f"```python\n{body}\n```"
        if spec.type == "json_schema":
            return '{"error": "unable to comply"}'
        return _WRONG_TEXT

    def _summary(self, prof: ModelProfile, suite: Suite, item: EvalItem, u: float) -> str:
        """Quality varies continuously: weaker models drop more of the reference."""
        words = (item.reference or "").split()
        quality = min(1.0, max(0.15, prof.p_correct(suite.category) + (u - 0.5) * 0.5))
        keep = max(4, int(len(words) * (0.35 + 0.65 * quality)))
        text = " ".join(words[:keep])
        if quality < 0.6:
            text += " Overall, this is a topic with several interesting aspects worth considering."
        return text

    def _judge(self, schema_name: str, messages: list[dict[str, Any]]) -> dict[str, Any] | None:
        text = "\n".join(message_text(m) for m in messages if m.get("role") == "user")
        if schema_name == "JudgeVerdict":
            m = re.search(r"## Candidate response\n(.*?)\n\nReturn JSON", text, re.DOTALL)
            candidate = m.group(1) if m else ""
            ref = re.search(r"## Reference answer[^\n]*\n(.*?)\n\n## ", text, re.DOTALL)
            f1 = _f1(candidate, ref.group(1)) if ref else 0.5
            noise = int(_u01("judge", candidate[:80]) * 3) - 1
            score = max(1, min(10, 1 + round(9 * min(1.0, f1 / 0.95)) + noise))
            return {"score": score, "rationale": f"Overlaps {f1:.0%} with the reference content."}
        if schema_name == "PairwiseVerdict":
            from .battle import ANSWER_A_MARK, ANSWER_B_MARK, REFERENCE_MARK

            a = re.search(
                re.escape(ANSWER_A_MARK) + r"\n(.*?)\n\n" + re.escape(ANSWER_B_MARK),
                text,
                re.DOTALL,
            )
            b = re.search(re.escape(ANSWER_B_MARK) + r"\n(.*?)\n\nReturn JSON", text, re.DOTALL)
            ref = re.search(re.escape(REFERENCE_MARK) + r"\n(.*?)\n\n", text, re.DOTALL)
            if not (a and b and ref):
                return None
            fa, fb = _f1(a.group(1), ref.group(1)), _f1(b.group(1), ref.group(1))
            diff = fa - fb
            # a mild position bias: near-ties usually go to slot A
            if abs(diff) < 0.04:
                pick = "A" if _u01("slot", a.group(1)[:60], b.group(1)[:60]) < 0.7 else "tie"
            else:
                pick = "A" if diff > 0 else "B"
            return {"winner": pick, "rationale": f"F1 vs reference: A={fa:.2f}, B={fb:.2f}."}
        if schema_name == "CategoryChoice":
            from .router import keyword_scores

            scores = keyword_scores(message_text(messages[-1]))
            best = max(scores, key=lambda c: scores[c]) if any(scores.values()) else "general_chat"
            return {"category": best, "reason": "keyword heuristic (simulated classifier)"}
        return None

    # ------------------------------------------------------------------ timing
    async def _sleep(self, virtual_s: float) -> None:
        await asyncio.sleep(max(0.0, virtual_s * self.time_scale))

    def _jitter(self, model: str, text: str) -> float:
        return 0.9 + 0.2 * _u01("jitter", model, text[:80])

    # ------------------------------------------------------------------ public API (NIMClient)
    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        response_format: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> ChatResult:
        found = None if (tools or response_format) else self._lookup(messages)
        if found is None:
            result = await self._base.chat(
                model,
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
                extra=extra,
            )
            self.usage.record(model, result.usage, result.latency_s)
            return result
        suite, item = found
        text = self._answer(model, suite, item)
        prof = model_profile(model, self.registry)
        tokens_out = estimate_tokens(text)
        virtual = (prof.ttft_s + tokens_out / prof.tokens_per_s) * self._jitter(model, text)
        await self._sleep(virtual)
        usage = Usage(prompt_tokens=estimate_tokens(item.prompt), completion_tokens=tokens_out)
        self.usage.record(model, usage, virtual)
        return ChatResult(
            model=model, content=text, finish_reason="stop", usage=usage, latency_s=virtual
        )

    async def stream_chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        response_format: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        found = None if (tools or response_format) else self._lookup(messages)
        if found is None:
            async for chunk in self._base.stream_chat(
                model,
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
                extra=extra,
            ):
                yield chunk
            return
        suite, item = found
        text = self._answer(model, suite, item)
        prof = model_profile(model, self.registry)
        jitter = self._jitter(model, text)
        await self._sleep(prof.ttft_s * jitter)
        pieces = re.findall(r"\S+\s*|\s+", text)
        step = 3  # words per streamed chunk keeps sleeps above timer resolution
        total = 0
        for i in range(0, len(pieces), step):
            chunk = "".join(pieces[i : i + step])
            total += estimate_tokens(chunk)
            await self._sleep(estimate_tokens(chunk) / prof.tokens_per_s * jitter)
            yield chunk
        self.usage.record(
            model,
            Usage(prompt_tokens=estimate_tokens(item.prompt), completion_tokens=total),
            prof.ttft_s + total / prof.tokens_per_s,
        )

    async def chat_json(
        self,
        model: str,
        messages: list[dict[str, Any]],
        schema: type[T],
        *,
        retries: int = 2,
        **kw: Any,
    ) -> T:
        payload = self._judge(schema.__name__, messages)
        if payload is None:
            return await self._base.chat_json(model, messages, schema, retries=retries, **kw)
        await self._sleep(0.8 * model_profile(model, self.registry).ttft_s)
        self.usage.record(model, Usage(prompt_tokens=200, completion_tokens=40), 0.0)
        return schema.model_validate(payload)

    async def embed(
        self, model: str, texts: list[str], *, input_type: Any = "passage"
    ) -> list[list[float]]:
        return await self._base.embed(model, texts, input_type=input_type)

    async def list_models(self) -> list[str]:
        return await self._base.list_models()
