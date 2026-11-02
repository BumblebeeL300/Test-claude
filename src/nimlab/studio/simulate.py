"""Offline simulation for NIM Lab Studio.

Studio runs without an API key. Instead of showing the core mock backend's filler text, it plays
believable sessions so the UI has something real to animate:

* :class:`SimForgeClient` is a scripted, duck-typed ``NIMClient`` for Forge: a plan, several files
  written by the real Forge tools, one failing test (found by real pytest), a patch, green tests and
  an approving review. Pauses are scaled by ``delay`` (0 in tests).
* :class:`StudioClient` wraps a client to (a) write pleasant, varied Council answers and extractive,
  cited Atlas answers when the backend is the mock, and (b) capture a preview of each model reply.

Everything here is clearly marked ``simulated`` by the API; none of it is ever shown as real model
output.
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import itertools
import json
import random
import re
import time
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel

from nimlab.core import ChatResult, ToolCall, Usage, UsageTracker
from nimlab.core.mock import estimate_tokens

from .sim_projects import ProjectTemplate, pick_template

__all__ = ["SimForgeClient", "StudioClient", "current_step"]

#: Set by the Council adapter on ``CallStarted`` so the wrapper knows which step a reply belongs to.
current_step: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "studio_current_step", default=None
)

_ids = itertools.count(1)


def _seed(*parts: str) -> int:
    return int.from_bytes(hashlib.sha256("|".join(parts).encode()).digest()[:8], "big")


def _text_of(messages: list[dict[str, Any]]) -> str:
    return "\n".join(str(m.get("content") or "") for m in messages)


# =========================================================================== Forge simulation


class SimForgeClient:
    """A scripted stand-in for ``NIMClient`` that drives Forge through a believable build.

    The script is derived from the conversation itself (how many user turns, how many assistant
    rounds since the last user turn), so it does not depend on call order and survives retries.
    """

    mock = False  # the reports should not claim "placeholder model output"
    simulated = True

    def __init__(
        self,
        prompt: str,
        *,
        usage: UsageTracker | None = None,
        delay: float = 1.0,
        template: ProjectTemplate | None = None,
    ) -> None:
        self.prompt = prompt
        self.template = template or pick_template(prompt)
        self.usage = usage if usage is not None else UsageTracker()
        self.delay = max(0.0, delay)
        self.calls = 0

    # ------------------------------------------------------------------ helpers
    async def _pause(self, seconds: float) -> None:
        if self.delay > 0:
            await asyncio.sleep(seconds * self.delay)

    def _record(
        self, model: str, messages: list[dict[str, Any]], out_chars: int, secs: float
    ) -> Usage:
        usage = Usage(
            prompt_tokens=estimate_tokens(_text_of(messages)) + 40,
            completion_tokens=max(8, out_chars // 4),
        )
        self.usage.record(model, usage, secs)
        return usage

    @staticmethod
    def _tc(name: str, **arguments: Any) -> ToolCall:
        return ToolCall(id=f"call_sim_{next(_ids)}", name=name, arguments=arguments)

    # ------------------------------------------------------------------ NIMClient surface
    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        **_: Any,
    ) -> ChatResult:
        self.calls += 1
        t = self.template
        user_turns = sum(1 for m in messages if m.get("role") == "user")
        turn = max(0, user_turns - 1)
        last_user = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=0)
        rnd = sum(1 for m in messages[last_user + 1 :] if m.get("role") == "assistant")

        calls: list[ToolCall] = []
        text = ""
        thinking = 0.9
        if tools is None:  # a plain call that is not the coder loop (e.g. the RCA step)
            text = "The failing assertion points at the rule ordering; fixing it in the module."
        elif turn == 0:
            script: list[tuple[str, list[ToolCall]]] = [
                ("", [self._tc("list_dir", path=".")]),
                ("", [self._tc("write_file", path="README.md", content=t.readme)]),
                ("", [self._tc("write_file", path="requirements-dev.txt", content="pytest>=8\n")]),
                ("", [self._tc("write_file", path=t.source_file, content=t.buggy_source)]),
                ("", [self._tc("write_file", path=t.test_module, content=t.tests)]),
            ]
            if rnd < len(script):
                text, calls = script[rnd]
            else:
                text = (
                    f"Implemented {t.module}.py with tests in {t.test_module} and a short README. "
                    "Ready for the full test run."
                )
        elif turn == 1:
            script = [
                ("", [self._tc("read_file", path=t.source_file)]),
                ("", [self._tc("apply_patch", path=t.source_file, patch=t.patch)]),
                ("", [self._tc("run_tests")]),
            ]
            if rnd < len(script):
                text, calls = script[rnd]
            else:
                text = "Fixed the failing case; the whole suite passes now."
        else:
            text = "Nothing further to change; the suite is green."
        await self._pause(thinking)
        out_chars = len(text) + sum(len(json.dumps(c.arguments)) for c in calls)
        usage = self._record(model, messages, out_chars, thinking)
        return ChatResult(
            model=model,
            content=text,
            tool_calls=calls,
            finish_reason="tool_calls" if calls else "stop",
            usage=usage,
            latency_s=thinking,
        )

    async def chat_json(
        self,
        model: str,
        messages: list[dict[str, Any]],
        schema: type[BaseModel],
        **_: Any,
    ) -> Any:
        self.calls += 1
        t = self.template
        name = schema.__name__
        if name == "Plan":
            await self._pause(1.6)
            data: dict[str, Any] = {
                "goal": t.goal,
                "steps": [
                    {"id": i, "description": desc, "files": list(files)}
                    for i, (desc, files) in enumerate(t.steps, start=1)
                ],
                "test_strategy": f"pytest: {len(re.findall(r'^def test_', t.tests, re.M))} focused "
                f"tests, including {t.failing_test.split('::')[-1]}",
                "risks": list(t.risks),
            }
        elif name == "Review":
            await self._pause(1.4)
            data = {
                "verdict": "approve",
                "summary": t.review_note + " Tests cover the edge cases that matter.",
                "issues": [
                    {
                        "severity": "nit",
                        "file": t.source_file,
                        "description": "Consider adding type hints to the public helpers.",
                        "suggestion": "Annotate arguments and return types.",
                    }
                ],
            }
        else:
            raise ValueError(f"simulation has no script for schema {name}")
        self._record(model, messages, len(json.dumps(data)), 1.0)
        return schema.model_validate(data)

    async def aclose(self) -> None:
        return None


# =========================================================================== text simulation

_STYLES = ("analytical", "pragmatic", "concise", "narrative", "cautious")

_OPENERS = {
    "analytical": "Let's break {t} into its moving parts before drawing a conclusion.",
    "pragmatic": "In practice, the useful way to look at {t} is by what it changes day to day.",
    "concise": "Short answer on {t}: it depends on your constraints, and here is how to decide.",
    "narrative": "Think of {t} as a decision you will live with, not just one you make once.",
    "cautious": "There is no universal answer on {t}, so it is worth being careful about assumptions.",
}

_POINTS = (
    "Start by pinning down what success looks like: the constraints, the audience and the cost of being wrong.",
    "The central trade-off is simplicity versus flexibility. The simpler option is easier to reason about; the flexible one only pays off if requirements keep moving.",
    "Evidence beats intuition. A small experiment or a quick measurement will tell you more than another round of debate.",
    "Look for second-order effects. Choices like this shape habits, tooling and maintenance work long after they are made.",
    "A staged approach limits risk: ship the smallest version that works, collect feedback, then expand where the data points.",
    "Typical failure modes are over-engineering up front and neglecting operational concerns such as monitoring, rollback and ownership.",
    "Context matters more than rules of thumb. What suits a large organisation often feels heavy for a small team, and the reverse is also true.",
    "Write down the reasoning as well as the outcome so the decision can be revisited when circumstances change.",
)

_CLOSERS = (
    "Overall, I would start small, measure, and let the results decide the next step.",
    "My recommendation: take the simplest option that meets today's needs and keep a clear path to change it.",
    "Put briefly, be explicit about the trade-offs and revisit the decision on a schedule.",
    "If you only do one thing, define the success criteria first; everything else follows from them.",
)

_COUNCIL_SYSTEMS = (
    "You are one expert member of a council",
    "You have been provided with a set of responses",
    "You are a rigorous reviewer in a panel debate",
    "You are a panelist in a debate",
    "You are an expert writing a first draft",
    "You are revising your own draft answer",
)

_TOPIC_LEAD = re.compile(
    r"^(please\s+)?(can you\s+|could you\s+)?(explain|describe|tell me|what is|what are|what's|"
    r"why is|why do|why does|how do|how does|how to|how can|is|are|should|do|does|can|write|"
    r"give me|compare)\s+",
    re.IGNORECASE,
)


def topic_of(question: str) -> str:
    """A short noun-phrase-ish topic from a question (best effort, never empty)."""
    q = re.sub(r"\s+", " ", question).strip().rstrip("?.! ")
    q = _TOPIC_LEAD.sub("", q)
    words = q.split()[:10]
    return " ".join(words).strip(",;:") or "this question"


def _last_user(messages: list[dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return str(m.get("content") or "")
    return ""


def _question_of(user_text: str) -> str:
    m = re.search(r"Question:\n(.+?)(?:\n\n|\Z)", user_text, re.S)
    return (m.group(1) if m else user_text).strip()


def compose_council_text(model: str, messages: list[dict[str, Any]]) -> str:
    """A pleasant, model-flavoured answer; critique/aggregate/revise prompts get fitting shapes."""
    system = str(messages[0].get("content") or "") if messages else ""
    user = _last_user(messages)
    question = _question_of(user)
    t = topic_of(question)
    rng = random.Random(_seed(model, question, system[:40]))
    style = _STYLES[_seed(model) % len(_STYLES)]
    points = rng.sample(_POINTS, k=3)
    closer = rng.choice(_CLOSERS)

    if system.startswith("You are a rigorous reviewer"):
        n = max(1, len(re.findall(r"^### Response \d+", user, re.M)))
        lines = []
        for i in range(1, n + 1):
            lines.append(
                f"- Response {i}: "
                + rng.choice(
                    (
                        "solid structure and a clear recommendation, but it asserts a trade-off without evidence.",
                        "gets the main point right; it skips operational concerns and never states its assumptions.",
                        "well argued, although one claim is stated more confidently than the support allows.",
                    )
                )
            )
        pick = rng.randint(1, n)
        return (
            "\n".join(lines)
            + f"\n\nOverall position: Response {pick} is the most convincing, with the others adding useful caveats."
        )
    if system.startswith("You have been provided with a set of responses"):
        opener = "Pulling the strongest threads together:"
        body = " ".join(points)
        return f"{opener} {body} {closer}"
    if system.startswith("You are a panelist in a debate") or system.startswith("You are revising"):
        opener = "On reflection, I would refine my earlier view on " + t + "."
        concede = rng.choice(
            (
                "My peers are right that evidence should come first, so I weight that more heavily now.",
                "I concede the point about operational cost; it deserves to be part of the decision.",
                "I still think staging the change is best, but I have tightened the success criteria.",
            )
        )
        return f"{opener} {concede} {points[0]} {points[1]} {closer}"
    opener = _OPENERS[style].format(t=t)
    if style == "concise":
        return f"{opener} {points[0]} {points[1]} {closer}"
    if style == "narrative":
        return f"{opener} {points[0]} {points[2]} Along the way, {points[1][0].lower() + points[1][1:]} {closer}"
    bullets = "\n".join(f"- {p}" for p in points)
    return f"{opener}\n\n{bullets}\n\n{closer}"


_SOURCE_RE = re.compile(r"\[(\d+)\] ([^\n]*)\n<<<\n(.*?)\n>>>", re.S)
_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "you",
        "your",
    ]
)


def _clean_md(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"[*_`>|]", "", text)
    text = re.sub(r"^\s*[-+]\s+", "", text, flags=re.M)
    return re.sub(r"\s+", " ", text).strip()


def compose_atlas_text(messages: list[dict[str, Any]]) -> str:
    """An extractive answer: the best-matching sentences of each source, each with its ``[n]``."""
    system = str(messages[0].get("content") or "")
    question = _last_user(messages)
    sources = _SOURCE_RE.findall(system)
    if not sources:
        return "The provided sources do not cover this question."
    q_terms = {w for w in _WORD.findall(question.lower()) if w not in _STOP}
    scored: list[tuple[float, int, str]] = []
    for n_str, _label, body in sources:
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", _clean_md(body))
            if 25 < len(s.strip()) <= 300 and "--" not in s and s.count("  ") == 0
        ]
        best: list[tuple[float, str]] = []
        for sent in sentences:
            terms = set(_WORD.findall(sent.lower()))
            overlap = len(q_terms & terms)
            if overlap:
                best.append((overlap + min(len(sent), 200) / 1000, sent))
        best.sort(key=lambda p: -p[0])
        for score, sent in best[:1]:
            scored.append((score, int(n_str), sent))
    if not scored:
        return "The provided sources do not cover this question."
    scored.sort(key=lambda s: -s[0])
    lines = [f"- {sent.rstrip('.')}. [{n}]" for _score, n, sent in scored[:4]]
    return "\n".join(lines)


class StudioClient:
    """Duck-typed ``NIMClient`` wrapper used by Council and Atlas runs.

    ``simulate=True`` (mock backend only) replaces the core mock's filler for *answer-writing*
    prompts with composed text; every other call passes straight through. ``previews`` records the
    first characters of each reply per Council trace step (see :data:`current_step`).
    """

    def __init__(
        self,
        base: Any,
        *,
        simulate: bool = False,
        delay: float = 1.0,
        council: bool = False,
    ) -> None:
        self._base = base
        self.simulate = simulate
        self.delay = max(0.0, delay)
        self.council = council
        self.previews: dict[int, str] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    # ------------------------------------------------------------------ classification
    def _kind(self, messages: list[dict[str, Any]], kw: dict[str, Any]) -> str | None:
        if not self.simulate or kw.get("tools") or kw.get("response_format"):
            return None
        if not messages or messages[-1].get("role") != "user":
            return None  # e.g. the reward model's [user, assistant] pair
        system = (
            str(messages[0].get("content") or "") if messages[0].get("role") == "system" else ""
        )
        if system.startswith("You are Atlas") and "SOURCES" in system:
            return "atlas"
        if system.startswith(_COUNCIL_SYSTEMS):
            return "council"
        if self.council and not system.startswith(
            ("You are an impartial judge", "You are a strict")
        ):
            return "council"
        return None

    def _text_for(self, kind: str, model: str, messages: list[dict[str, Any]]) -> str:
        return (
            compose_atlas_text(messages)
            if kind == "atlas"
            else compose_council_text(model, messages)
        )

    def _remember(self, text: str) -> None:
        step = current_step.get()
        if step is not None:
            self.previews[step] = text

    async def _sleep(self, seconds: float) -> None:
        if self.delay > 0:
            await asyncio.sleep(seconds * self.delay)

    # ------------------------------------------------------------------ chat
    async def chat(self, model: str, messages: list[dict[str, Any]], **kw: Any) -> ChatResult:
        kind = self._kind(messages, kw)
        if kind is None:
            result: ChatResult = await self._base.chat(model, messages, **kw)
            self._remember(result.content)
            return result
        t0 = time.perf_counter()
        text = self._text_for(kind, model, messages)
        rng = random.Random(_seed(model, text[:60]))
        await self._sleep(0.5 + rng.random() * 1.1)
        usage = Usage(
            prompt_tokens=estimate_tokens(_text_of(messages)),
            completion_tokens=estimate_tokens(text),
        )
        latency = max(time.perf_counter() - t0, 0.001)
        self._base.usage.record(model, usage, latency)
        self._remember(text)
        return ChatResult(
            model=model, content=text, finish_reason="stop", usage=usage, latency_s=latency
        )

    def _json_for(
        self, schema_name: str, model: str, messages: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """Plausible structured replies for the Council's judge, critic and rubric grader."""
        user = _last_user(messages)
        question = _question_of(user)
        rng = random.Random(_seed(model, schema_name, user[:200]))
        if schema_name == "JudgeVerdict":
            blocks = re.findall(
                r"### Response (\d+)\n(.*?)(?=\n\n### Response |\n\nDebate history:|\Z)", user, re.S
            )
            if not blocks:
                return None
            idx = rng.randrange(len(blocks))
            return {
                "selected": int(blocks[idx][0]),
                "final_answer": blocks[idx][1].strip(),
                "rationale": f"Response {blocks[idx][0]} is the most complete and best justified; "
                "the others agree on the essentials but leave assumptions implicit.",
                "confidence": round(0.72 + rng.random() * 0.2, 2),
            }
        if schema_name == "Critique":
            m = re.search(r"revision (\d+)", user)
            first = (int(m.group(1)) if m else 0) == 0
            if first and rng.random() < 0.6:
                return {
                    "issues": [
                        {
                            "severity": "high",
                            "category": "completeness",
                            "description": "The draft never says how to tell whether the approach worked.",
                            "suggestion": "Add concrete success criteria and one way to measure them.",
                        }
                    ],
                    "summary": "Solid draft, but it misses an essential piece: how to evaluate the outcome.",
                }
            return {"issues": [], "summary": "The draft is accurate, complete and clear."}
        if schema_name == "RubricScore":
            return {
                "score": round(6.2 + rng.random() * 3.3, 1),
                "rationale": f"A clear, well-organised answer to '{topic_of(question)}' with sensible trade-offs.",
            }
        return None

    async def chat_json(
        self, model: str, messages: list[dict[str, Any]], schema: type[BaseModel], **kw: Any
    ) -> Any:
        if self.simulate and self.council:
            data = self._json_for(schema.__name__, model, messages)
            if data is not None:
                t0 = time.perf_counter()
                await self._sleep(0.4 + random.Random(_seed(model, str(data)[:40])).random() * 0.6)
                out = schema.model_validate(data)
                text = json.dumps(data)
                usage = Usage(
                    prompt_tokens=estimate_tokens(_text_of(messages)),
                    completion_tokens=estimate_tokens(text),
                )
                self._base.usage.record(model, usage, max(time.perf_counter() - t0, 0.001))
                self._remember(
                    str(
                        data.get("final_answer")
                        or data.get("summary")
                        or data.get("rationale")
                        or ""
                    )
                )
                return out
        return await self._base.chat_json(model, messages, schema, **kw)

    async def stream_chat(
        self, model: str, messages: list[dict[str, Any]], **kw: Any
    ) -> AsyncIterator[str]:
        kind = self._kind(messages, kw)
        if kind is None:
            async for piece in self._base.stream_chat(model, messages, **kw):
                yield piece
            return
        t0 = time.perf_counter()
        text = self._text_for(kind, model, messages)
        await self._sleep(0.4)
        parts = re.findall(r"\S+\s*", text)
        for i in range(0, len(parts), 3):
            yield "".join(parts[i : i + 3])
            await self._sleep(0.035)
            if self.delay <= 0:
                await asyncio.sleep(0)
        usage = Usage(
            prompt_tokens=estimate_tokens(_text_of(messages)),
            completion_tokens=estimate_tokens(text),
        )
        self._base.usage.record(model, usage, time.perf_counter() - t0)
