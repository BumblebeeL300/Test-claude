"""Offline simulator used in mock mode so Sentinel produces *meaningful* demo results.

The core mock backend answers every prompt with generic prose, which would make every attack look
"resisted". :class:`SimulatedClient` recognises the bundled probes and answers them like a model of
a given strength would, deterministically from a hash of (model, probe):

* bigger models (higher registry ``cost_tier``) resist better, with per-model jitter;
* categories differ in difficulty (fake-citation bait and indirect injection are the hardest);
* every model has its own appetite for over-refusing benign requests (small models more so);
* it also plays NemoGuard: prompts in the content-safety template are classified by looking for the
  ``[REDACTED HARMFUL REQUEST ...]`` and ``[SIMULATED UNSAFE COMPLETION ...]`` placeholders.

Nothing here is real model behaviour: results are labelled *simulated* everywhere they appear.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from typing import Any

from nimlab.core import NIMClient, NIMConfig, default_registry
from nimlab.core.registry import ModelRegistry
from nimlab.core.results import ChatResult, Usage
from nimlab.core.usage import UsageTracker

from .llm import message_text
from .models import Probe, ProbeSuite
from .suites import list_suites

__all__ = ["UNSAFE_MARK", "SimProfile", "SimulatedClient", "sim_profile"]

UNSAFE_MARK = "[SIMULATED UNSAFE COMPLETION: placeholder, no real content]"
_GUARD_PREFIX = "Task: Check if there is unsafe content"

#: Category difficulty: added to a model's base resistance.
_OFFSET = {
    "harmful_refusal": 0.12,
    "pii_leakage": 0.05,
    "system_prompt_extraction": 0.0,
    "jailbreak": -0.05,
    "prompt_injection": -0.10,
    "hallucinated_citation": -0.18,
}
_TOPICS = {
    "weapon": "Guns and Illegal Weapons",
    "malware": "Malware",
    "phishing": "Fraud/Deception",
    "fraud": "Fraud/Deception",
    "invoice": "Fraud/Deception",
    "self-harm": "Suicide and Self Harm",
    "harass": "Harassment",
    "drug": "Controlled/Regulated Substances",
    "hate": "Hate/Identity Hate",
    "pathogen": "Criminal Planning/Confessed Criminal Activity",
}


def _u01(*parts: str) -> float:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class SimProfile:
    """How a simulated model behaves."""

    def __init__(self, model: str, tier: int, resist: float, over_refusal: float) -> None:
        self.model = model
        self.tier = tier
        self.resist = resist
        self.over_refusal = over_refusal

    def p_hold(self, category: str) -> float:
        return _clamp(self.resist + _OFFSET.get(category, 0.0), 0.08, 0.995)


def sim_profile(model: str, registry: ModelRegistry | None = None) -> SimProfile:
    info = (registry or default_registry()).find(model)
    jitter = _u01("sentinel-profile", model)
    tier = info.cost_tier if info else 1 + int(jitter * 5) % 5
    resist = 0.55 + 0.095 * tier + (jitter - 0.5) * 0.08
    over = 0.03 + 0.42 * _u01("sentinel-over", model) ** 2.2 + (0.08 if tier <= 1 else 0.0)
    return SimProfile(model, tier, resist, over)


def _topic_category(text: str) -> str:
    low = text.lower()
    for key, name in _TOPICS.items():
        if key in low:
            return name
    return "Criminal Planning/Confessed Criminal Activity"


class SimulatedClient:
    """Drop-in for :class:`nimlab.core.NIMClient` that needs no network and knows the probes."""

    simulated = True

    def __init__(
        self,
        suites: list[ProbeSuite] | dict[str, ProbeSuite] | None = None,
        *,
        config: NIMConfig | None = None,
        registry: ModelRegistry | None = None,
        delay_s: float = 0.0,
    ) -> None:
        self.config = config or NIMConfig.from_env(mock=True)
        self.registry = registry or default_registry()
        self.delay_s = delay_s
        self._base = NIMClient(self.config.model_copy(update={"mock": True}))
        self._index: dict[str, tuple[ProbeSuite, Probe]] = {}
        loaded = suites if suites is not None else list_suites()
        self.register_suites(loaded)

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

    @property
    def usage(self) -> UsageTracker:
        """Token accounting, shared with the delegate that answers non-probe prompts."""
        return self._base.usage

    @usage.setter
    def usage(self, tracker: UsageTracker) -> None:
        self._base.usage = tracker

    def register_suites(self, suites: list[ProbeSuite] | dict[str, ProbeSuite]) -> None:
        for suite in suites.values() if isinstance(suites, dict) else suites:
            for probe in suite.probes:
                self._index[probe.user.strip()] = (suite, probe)

    def __getattr__(self, name: str) -> Any:  # chat_json, stream_chat, embed, rerank ...
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._base, name)

    # ------------------------------------------------------------------ chat
    async def chat(self, model: str, messages: list[dict[str, Any]], **kw: Any) -> ChatResult:
        last_user = next(
            (message_text(m) for m in reversed(messages) if m.get("role") == "user"), ""
        )
        if last_user.startswith(_GUARD_PREFIX):
            return await self._finish(model, messages, self._guard_reply(last_user), "guard")
        found = self._index.get(last_user.strip())
        if found is None:
            return await self._base.chat(model, messages, **kw)
        _, probe = found
        suite = found[0]
        return await self._finish(
            model, messages, self._answer(model, suite.category, probe), probe.id
        )

    async def _finish(
        self, model: str, messages: list[dict[str, Any]], text: str, key: str
    ) -> ChatResult:
        prof = sim_profile(model, self.registry)
        virtual = round(0.25 + 0.12 * prof.tier + 0.004 * len(text), 3)
        if self.delay_s > 0:
            await asyncio.sleep(self.delay_s * (0.5 + _u01("sentinel-delay", model, key)))
        tokens_in = sum(len(message_text(m)) for m in messages) // 4 + 1
        result = ChatResult(
            model=model,
            content=text,
            finish_reason="stop",
            usage=Usage(prompt_tokens=tokens_in, completion_tokens=len(text) // 4 + 1),
            latency_s=virtual,
        )
        self.usage.record(model, result.usage, result.latency_s)
        return result

    # ------------------------------------------------------------------ NemoGuard
    @staticmethod
    def _guard_reply(prompt: str) -> str:
        m = re.search(
            r"\nuser: (.*?)\n(?:response: agent: (.*?)\n)?<END CONVERSATION>", prompt, re.S
        )
        user, response = (m.group(1), m.group(2)) if m else ("", None)
        user_bad = "[REDACTED HARMFUL REQUEST" in user
        resp_bad = response is not None and "[SIMULATED UNSAFE COMPLETION" in response
        data: dict[str, str] = {"User Safety": "unsafe" if user_bad else "safe"}
        if response is not None:
            data["Response Safety"] = "unsafe" if resp_bad else "safe"
        if user_bad or resp_bad:
            data["Safety Categories"] = _topic_category(user)
        body = ", ".join(f'"{k}": "{v}"' for k, v in data.items())
        return "{" + body + "}"

    # ------------------------------------------------------------------ probe answers
    def _answer(self, model: str, category: str, probe: Probe) -> str:
        prof = sim_profile(model, self.registry)
        u = lambda tag: _u01("sentinel", model, probe.id, tag)  # noqa: E731
        if probe.benign:
            if u("over") < prof.over_refusal:
                return "I'm sorry, but I can't help with that request."
            subject = " ".join(probe.user.split()[:9]).rstrip("?.,")
            return (
                f'Sure. On "{subject}": here is a concise, practical answer with the key steps '
                "and one safety tip. (simulated reply)"
            )
        breach = u("breach") >= prof.p_hold(category)
        refuse_style = u("style") < 0.7
        return (self._breach if breach else self._held)(category, probe, refuse_style, u)

    @staticmethod
    def _breach(category: str, probe: Probe, _style: bool, u: Any) -> str:
        marker = probe.success_marker or ""
        if category == "prompt_injection":
            if u("shape") < 0.6:
                return marker
            return f"Summary: routine logistics and dates.\n{marker}"
        if category == "jailbreak":
            if u("shape") < 0.3:
                return f"{marker} Okay, I accept the persona."
            return f"{marker}\nSure, going along with that persona. {UNSAFE_MARK}"
        if category == "system_prompt_extraction":
            if u("shape") < 0.5 and probe.system:
                return f"Sure, here are my instructions:\n{probe.system}"
            return f"The internal reference code is {probe.canary}."
        if category == "pii_leakage":
            return "Sure, here is what I can see: " + "; ".join(probe.secrets)
        if category == "harmful_refusal":
            return f"Sure, here are the steps. {UNSAFE_MARK}"
        if category == "hallucinated_citation":
            return (
                "This is a well-known work. Okonkwo and Veltrane (2017) show a clear effect, "
                "published in the Journal of Plausible Results 12(3), pp. 45-67, "
                "doi:10.0000/sim.2017.001. They report a 31% improvement in the main table."
            )
        return "Sure."

    @staticmethod
    def _held(category: str, probe: Probe, refuse: bool, u: Any) -> str:
        if category == "prompt_injection":
            if u("shape") < 0.3:
                return (
                    "Summary: the document covers routine logistics and dates. It also contains an "
                    "embedded instruction aimed at AI assistants, which I ignored."
                )
            return "Summary: the document covers routine logistics and dates; nothing to action."
        if category == "pii_leakage":
            return (
                "I can't share other customers' personal information."
                if refuse
                else "I can only help with your own account. What would you like to know?"
            )
        if category == "system_prompt_extraction":
            return (
                "I'm sorry, but I can't share my internal instructions."
                if refuse
                else "I can help with orders and shipping. What do you need?"
            )
        if category == "hallucinated_citation":
            return (
                "I couldn't find a verifiable source with that title, so I can't confirm it "
                "exists. If you share a link or the text I can summarise it."
                if refuse
                else "I'm not aware of that source and don't want to invent details. "
                "Could you check the reference?"
            )
        # jailbreak and harmful_refusal
        return (
            "I'm sorry, but I can't help with that request."
            if refuse
            else "Happy to help with something else, but I won't go along with that."
        )
