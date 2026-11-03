"""Deterministic offline simulator used in mock mode.

The core mock backend produces generic filler, which would make every Distill stage look the same.
:class:`SimClient` instead plays a believable synthetic-data session: topic trees, varied
instructions, grounded QA pairs from the document excerpt it is given, answers whose quality depends
on the (simulated) model, and reward-model scores computed from the answer text. It also injects
realistic defects (refusals, fragments, repetition, non-English text, duplicates and
near-duplicates) so the filter and dedup stages visibly do something.

Everything is a pure function of the inputs plus ``seed``: the same call always returns the same
result, whatever the concurrency, which keeps resumed runs identical to uninterrupted ones.
Output is **simulated**; it is never real model text.
"""

# ruff: noqa: E501
from __future__ import annotations

import asyncio
import hashlib
import random
import re
from typing import Any, TypeVar

from pydantic import BaseModel

from nimlab.core import ChatResult, ModelRegistry, Usage, UsageTracker, default_registry
from nimlab.core.mock import embed_text, estimate_tokens

from .prompts import (
    InstructionList,
    JudgeScore,
    QAPair,
    QuestionList,
    SubtopicList,
    TopicList,
    task_of,
)
from .textutil import (
    content_words,
    looks_english,
    repeat_ratio,
    sentences,
    word_tokens,
)

__all__ = ["SimClient", "answer_quality"]

T = TypeVar("T", bound=BaseModel)

_DOMAIN_TOPICS: dict[str, list[str]] = {
    "software engineering": [
        "unit testing",
        "code review",
        "API design",
        "database indexing",
        "continuous integration",
        "error handling",
        "dependency management",
        "refactoring legacy code",
    ],
    "data science": [
        "feature engineering",
        "model evaluation",
        "data cleaning",
        "A/B testing",
        "time series forecasting",
        "dimensionality reduction",
        "experiment tracking",
        "data labeling",
    ],
    "personal finance": [
        "emergency funds",
        "index fund investing",
        "debt repayment",
        "retirement accounts",
        "budgeting",
        "credit scores",
        "tax planning",
        "insurance basics",
    ],
    "health and fitness": [
        "strength training",
        "sleep hygiene",
        "meal planning",
        "running for beginners",
        "stress management",
        "mobility work",
        "hydration",
        "recovery between workouts",
    ],
    "cooking": [
        "knife skills",
        "sauce making",
        "bread baking",
        "meal prep",
        "fermentation",
        "seasoning",
        "food safety",
        "cooking with legumes",
    ],
    "climate and energy": [
        "solar power",
        "home insulation",
        "electric vehicles",
        "carbon accounting",
        "grid storage",
        "heat pumps",
        "wind energy",
        "energy efficiency audits",
    ],
}
_GENERIC_TOPIC_PATTERNS = (
    "foundations of {d}",
    "tools used in {d}",
    "common mistakes in {d}",
    "measuring results in {d}",
    "planning projects in {d}",
    "ethics and risk in {d}",
    "case studies from {d}",
    "careers in {d}",
    "beginner questions about {d}",
    "advanced techniques in {d}",
)
_ASPECTS = (
    "core concepts",
    "common mistakes",
    "step-by-step workflows",
    "tooling and automation",
    "performance and efficiency",
    "risk and safety considerations",
    "real-world case studies",
    "ways of measuring success",
)
_INSTRUCTION_PATTERNS = (
    "Explain {s} to a new team member in under 150 words.",
    "Write a step-by-step checklist for handling {s}.",
    "What are the three biggest trade-offs to weigh when dealing with {s}? Justify each briefly.",
    "Draft a short email to a colleague summarizing the key points about {s}.",
    "Give a concrete example that illustrates {s}, then explain what it teaches.",
    "Create a quiz of three questions about {s}, with answers.",
    "Compare two common approaches to {s} and recommend one for a small team.",
    "List five warning signs that {s} is going wrong and how to react to each.",
    "Summarize {s} as a table with columns for goal, method and pitfall.",
    "Write a brief FAQ with four questions on {s}.",
    "Explain why {s} matters, using an analogy from everyday life.",
    "Outline a one-week plan to get better at {s}.",
    "Describe the most common misconception about {s} and correct it.",
    "Suggest a simple experiment to test whether your approach to {s} works.",
    "How would you teach {s} to a skeptical beginner? Give a short script.",
    "Write a short retrospective on a project where {s} went badly.",
)
_HARD_PATTERNS = (
    "Under what conditions does {s} break down, and what would you change to fix it? Reason step by step.",
    "Two experts disagree about {s}: one prioritises speed, the other safety. Build the strongest case for each and decide which should win in a regulated setting.",
    "A team applies {s} and sees the opposite of the expected result. Enumerate three plausible causes and a decisive test for each.",
    "Derive a rule of thumb for {s} from first principles and state where it fails.",
    "How would you quantify the cost of getting {s} wrong, and which assumptions dominate the estimate?",
    "Design a minimal audit for {s} that a skeptical reviewer would accept, and explain what it cannot detect.",
    "Compare the long-term and short-term consequences of neglecting {s}, and propose a decision rule.",
    "Which common advice about {s} is actually harmful? Argue your position with a concrete scenario.",
)
_SWAPS = {
    "Explain": "Describe",
    "Write": "Draft",
    "Give": "Provide",
    "List": "Name",
    "Create": "Build",
    "Summarize": "Sum up",
    "Outline": "Sketch",
    "Compare": "Contrast",
    "Suggest": "Propose",
}
_INTROS = (
    "Here is a practical way to think about {t}.",
    "Let's work through {t} step by step.",
    "A clear approach to {t} comes down to a few ideas.",
    "The short version on {t}: keep it simple and keep it measurable.",
)
_POINTS = (
    "Start by pinning down what success looks like for {t}; without a clear target every later decision is guesswork.",
    "Break {t} into small, checkable steps so that progress is visible and mistakes surface early.",
    "Most problems with {t} come from skipped fundamentals, so revisit the basics before reaching for advanced tricks.",
    "Measure the outcome of {t} with one or two simple indicators rather than a dashboard nobody reads.",
    "Write down the assumptions behind your approach to {t}; they are the first thing to question when results surprise you.",
    "Keep a short feedback loop: try a change, observe its effect on {t}, and keep only what demonstrably helps.",
    "Share what you learn about {t} with teammates, because a second pair of eyes catches blind spots quickly.",
    "Automate the repetitive parts of {t} once they are stable, and leave the judgement calls to people.",
    "Watch the trade-off between speed and care in {t}; the right balance depends on the cost of being wrong.",
    "Document one worked example of {t} so newcomers can see the reasoning and not only the result.",
    "Review the results for {t} on a fixed schedule, and retire practices that no longer pull their weight.",
    "Expect {t} to feel slow at first; consistency matters more than intensity in the early weeks.",
)
_CLOSERS = (
    "Revisit this after your first attempt and adjust based on what you observe.",
    "If you only do one thing, make the first step small enough to finish today.",
    "Keep notes as you go; they turn a one-off effort into a repeatable method.",
)
_REFUSALS = (
    "I'm sorry, but I can't help with that request.",
    "As an AI language model, I cannot provide an answer to this question.",
    "I'm unable to assist with that. Please consult a qualified professional instead.",
)
_SPANISH = (
    "Esta es una guia practica sobre {t}. Es importante definir objetivos claros y revisar los "
    "resultados con frecuencia para mejorar el proceso poco a poco."
)
_FRAGMENTS = ("It depends.", "Sure, here is", "Good question!", "See above.")
_HALLUCINATIONS = (
    "It is widely believed that this topic has many facets and experts continue to debate them at length.",
    "The answer relates to a broader pattern that is common across many unrelated fields of study.",
)
_QA_TEMPLATES = (
    "What does the text say about {a} and {b}?",
    "How does the document describe {a}?",
    "Why does the passage connect {a} with {b}?",
    "Explain what the excerpt states regarding {a}.",
    "According to the document, what is the role of {a}?",
)
_JUDGE_WORDS = ("Clear and on topic.", "Reasonable but generic.", "Thin or off target.")


def _h(*parts: object) -> int:
    digest = hashlib.sha256("\x1f".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _u(*parts: object) -> float:
    return _h(*parts) / 2**64


def _rng(*parts: object) -> random.Random:
    return random.Random(_h(*parts))


def _pattern_regexes(patterns: tuple[str, ...]) -> list[re.Pattern[str]]:
    out = []
    for p in patterns:
        pre, post = p.split("{s}")
        out.append(re.compile("^" + re.escape(pre) + r"(?P<s>.+?)" + re.escape(post) + "$"))
    return out


_PROMPT_RES = _pattern_regexes(_INSTRUCTION_PATTERNS + _HARD_PATTERNS)


def _subject_of(prompt: str) -> str:
    """The topic phrase a simulated answer talks about."""
    flat = " ".join(prompt.split())
    for rx in _PROMPT_RES:
        m = rx.match(flat)
        if m:
            return m.group("s")
    words = content_words(flat)[:3]
    return " ".join(words) if words else "this topic"


def _clip_words(text: str, n: int) -> str:
    words = text.split()
    return " ".join(words[:n]) + ("..." if len(words) > n else "")


def answer_quality(prompt: str, response: str) -> float:
    """A 0-1 heuristic quality of ``response`` to ``prompt`` (length, relevance, structure, fluency)."""
    words = word_tokens(response)
    n = len(words)
    if n == 0:
        return 0.0
    length = min(1.0, n / 140.0) * (1.0 if n <= 300 else max(0.4, 300 / n))
    wanted = set(content_words(prompt))
    overlap = len(wanted & set(words)) / len(wanted) if wanted else 0.5
    structure = 1.0 if re.search(r"^\s*(?:[-*]|\d+\.|\||Q\d|Subject:)", response, re.M) else 0.55
    fluency = 1.0 if looks_english(response) else 0.2
    q = 0.35 * length + 0.30 * min(1.0, overlap * 1.6) + 0.15 * structure + 0.20 * fluency
    q -= 0.6 * min(1.0, repeat_ratio(response) * 2)
    if re.search(
        r"\b(?:i'm sorry|as an ai language model|i'm unable|i can't help)\b", response, re.I
    ):
        q -= 0.45
    return max(0.0, min(1.0, q))


class SimClient:
    """A duck-typed stand-in for :class:`nimlab.core.NIMClient` that simulates Distill's calls."""

    mock = True
    simulated = True

    def __init__(
        self,
        *,
        seed: int = 0,
        registry: ModelRegistry | None = None,
        usage: UsageTracker | None = None,
        delay: float = 0.0,
        fail_after: int | None = None,
    ) -> None:
        self.seed = seed
        self.registry = registry or default_registry()
        self.usage = usage if usage is not None else UsageTracker()
        self.delay = delay
        #: Raise ``RuntimeError`` on every call after this many calls (tests: simulate a crash).
        self.fail_after = fail_after
        self.calls = 0

    async def aclose(self) -> None:
        return None

    async def __aenter__(self) -> SimClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    # ------------------------------------------------------------------ plumbing
    async def _tick(self) -> None:
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("simulated crash")
        if self.delay:
            await asyncio.sleep(self.delay)

    def skill(self, model: str) -> float:
        """How good the simulated model is, 0.35-0.95 (bigger/pricier models score higher)."""
        info = self.registry.find(model)
        tier = info.cost_tier if info is not None else 2 + _h(model) % 3
        jitter = (_u("skill", model) - 0.5) * 0.08
        return max(0.35, min(0.95, 0.30 + 0.17 * tier + jitter))

    def _result(self, model: str, messages: list[dict[str, Any]], text: str) -> ChatResult:
        prompt_tokens = sum(estimate_tokens(str(m.get("content") or "")) for m in messages)
        usage = Usage(prompt_tokens=prompt_tokens, completion_tokens=estimate_tokens(text))
        self.usage.record(model, usage, 0.0)
        return ChatResult(model=model, content=text, finish_reason="stop", usage=usage)

    # ------------------------------------------------------------------ public API
    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        **_: Any,
    ) -> ChatResult:
        await self._tick()
        if messages and messages[-1].get("role") == "assistant":
            text = self._reward(model, messages)
        else:
            task = task_of(messages)
            user = str(messages[-1].get("content") or "")
            if task in ("answer", "reference"):
                text = self._answer(model, user, temperature, strong=task == "reference")
            else:
                text = "Simulated reply: " + " ".join(word_tokens(user)[:12])
        return self._result(model, messages, text)

    async def chat_json(
        self,
        model: str,
        messages: list[dict[str, Any]],
        schema: type[T],
        *,
        retries: int = 2,
        **kw: Any,
    ) -> T:
        await self._tick()
        task = task_of(messages)
        user = str(messages[-1].get("content") or "")
        data: dict[str, Any]
        if task == "topics":
            data = {"topics": self._topics(user)}
        elif task == "subtopics":
            data = {"subtopics": self._subtopics(user)}
        elif task == "instructions":
            data = {"instructions": self._prompts(user, _INSTRUCTION_PATTERNS)}
        elif task == "hard_questions":
            data = {"questions": self._prompts(user, _HARD_PATTERNS)}
        elif task == "qa_pair":
            data = self._qa(model, user, float(kw.get("temperature", 0.5)))
        elif task == "judge":
            data = self._judge(user)
        else:
            raise ValueError(f"simulator does not know task {task!r}")
        self._result(model, messages, str(data))
        return schema.model_validate(data)

    async def embed(self, model: str, texts: list[str], **_: Any) -> list[list[float]]:
        await self._tick()
        self.usage.record(model, Usage(prompt_tokens=sum(estimate_tokens(t) for t in texts)), 0.0)
        return [embed_text(t) for t in texts]

    # ------------------------------------------------------------------ seed tasks
    @staticmethod
    def _field(user: str, name: str) -> str:
        m = re.search(rf"^{name}: (.+)$", user, re.M)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _count(user: str) -> int:
        m = re.search(r"(?:List|Write) (\d+)", user)
        return max(1, int(m.group(1))) if m else 3

    def _topics(self, user: str) -> list[str]:
        domain, k = self._field(user, "Domain"), self._count(user)
        bank = list(_DOMAIN_TOPICS.get(domain.lower(), []))
        if not bank:
            bank = [p.format(d=domain) for p in _GENERIC_TOPIC_PATTERNS]
        _rng(self.seed, "topics", domain).shuffle(bank)
        return [
            bank[i % len(bank)] + ("" if i < len(bank) else f" (part {i // len(bank) + 1})")
            for i in range(k)
        ]

    def _subtopics(self, user: str) -> list[str]:
        topic, k = self._field(user, "Topic"), self._count(user)
        aspects = list(_ASPECTS)
        _rng(self.seed, "subtopics", topic).shuffle(aspects)
        return [
            f"{aspects[i % len(aspects)]} in {topic}"
            + ("" if i < len(aspects) else f" (level {i // len(aspects) + 1})")
            for i in range(k)
        ]

    def _prompts(self, user: str, patterns: tuple[str, ...]) -> list[str]:
        sub, topic, k = self._field(user, "Subtopic"), self._field(user, "Topic"), self._count(user)
        rng = _rng(self.seed, "prompts", topic, sub, k)
        order = list(range(len(patterns)))
        rng.shuffle(order)
        out: list[str] = []
        for i in range(k):
            roll = rng.random()
            if out and roll < 0.10:
                out.append(self._near_variant(rng.choice(out), rng))
            elif out and roll < 0.15:
                out.append(rng.choice(out))
            else:
                out.append(patterns[order[i % len(order)]].format(s=sub))
        return out

    @staticmethod
    def _near_variant(text: str, rng: random.Random) -> str:
        first, _, rest = text.partition(" ")
        if first in _SWAPS and rng.random() < 0.6:
            return f"{_SWAPS[first]} {rest}"
        if rng.random() < 0.5:
            return text.lower().rstrip(".?!") + " ?"
        return text + " "

    # ------------------------------------------------------------------ grounded QA
    def _qa(self, model: str, user: str, temperature: float) -> dict[str, Any]:
        variant = int(self._field(user, "Variant") or 0)
        m = re.search(r"Excerpt:\n(.*?)\n\nWrite question variant", user, re.S)
        excerpt = m.group(1) if m else user
        sents = [s for s in sentences(excerpt) if len(word_tokens(s)) >= 5] or [excerpt.strip()]
        rng = _rng(self.seed, "qa", model, excerpt, variant, temperature)
        sent = sents[(variant + rng.randrange(2)) % len(sents)]
        kws = sorted(set(content_words(sent)), key=lambda w: (-len(w), w))[:2] or ["the topic"]
        a, b = kws[0], kws[-1]
        question = _QA_TEMPLATES[(variant + _h(sent) % 3) % len(_QA_TEMPLATES)].format(a=a, b=b)
        answer = sent
        if len(sents) > 1 and rng.random() < 0.4:
            answer += " " + sents[(sents.index(sent) + 1) % len(sents)]
        flaw = rng.random() + (0.6 - self.skill(model)) * 0.15
        if flaw < 0.05:
            answer = "I don't know based on the provided text."
        elif flaw < 0.09:
            answer = "Yes."
        elif flaw < 0.14:
            answer = rng.choice(_HALLUCINATIONS)
        return {"question": question, "answer": answer}

    # ------------------------------------------------------------------ answers
    def _answer(self, model: str, prompt: str, temperature: float, *, strong: bool) -> str:
        rng = _rng(self.seed, "answer", model, prompt, round(temperature, 2))
        subject = _subject_of(prompt)
        q = self.skill(model) + (rng.random() - 0.5) * 0.2 - 0.10 * max(0.0, temperature - 0.8)
        q = max(0.1, min(1.0, q + (0.15 if strong else 0.0)))
        roll = rng.random() + (0.6 - self.skill(model)) * 0.12
        if roll < 0.035:
            return rng.choice(_REFUSALS)
        if roll < 0.065:
            return rng.choice(_FRAGMENTS)
        if roll < 0.085:
            return _SPANISH.format(t=subject)
        points = [
            p.format(t=subject if i % 2 == 0 else "this topic") for i, p in enumerate(_POINTS)
        ]
        rng.shuffle(points)
        count = 1 + round(q * 5)
        if roll < 0.11:
            points = [points[0]] * (count + 3)
        picked = points[:count] if roll >= 0.11 else points
        intro = rng.choice(_INTROS).format(t=subject)
        if q >= 0.6:  # stronger models restate the request, which keeps them on topic
            intro = f"On your request ({_clip_words(prompt, 10)}): {intro}"
        closer = rng.choice(_CLOSERS)
        low = prompt.lower()
        if "table" in low:
            rows = "\n".join(
                f"| {p.split(' ')[0].strip(',;')} | {' '.join(p.split(' ')[1:8])} | {' '.join(p.split(' ')[-4:])} |"
                for p in picked
            )
            return f"{intro}\n\n| Goal | Method | Pitfall |\n|---|---|---|\n{rows}\n\n{closer}"
        if "email" in low:
            body = "\n\n".join(picked)
            return f"Subject: Notes on {subject}\n\nHi team,\n\n{intro}\n\n{body}\n\n{closer}\n\nBest,\nAlex"
        if any(w in low for w in ("quiz", "faq", "questions")):
            qa = "\n".join(
                f"Q{i + 1}: What should you remember about {subject} (point {i + 1})?\nA{i + 1}: {p}"
                for i, p in enumerate(picked)
            )
            return f"{intro}\n\n{qa}"
        if any(w in low for w in ("checklist", "step", "plan", "outline")):
            steps = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(picked))
            return f"{intro}\n\n{steps}\n\n{closer}"
        bullets = "\n".join(f"- {p}" for p in picked)
        return f"{intro}\n\n{bullets}\n\n{closer}"

    # ------------------------------------------------------------------ scoring
    def _noise(self, model: str, *parts: object) -> float:
        return _u(self.seed, "noise", model, *parts) - 0.5

    def _reward(self, model: str, messages: list[dict[str, Any]]) -> str:
        prompt = str(messages[0].get("content") or "")
        response = str(messages[-1].get("content") or "")
        if _u(self.seed, "unparseable", prompt, response) < 0.03:
            return "I am unable to rate this conversation."
        q = answer_quality(prompt, response)
        value = -16.0 + 20.0 * q + 1.6 * self._noise(model, prompt, response)
        return f"reward:{value:.4f}"

    def _judge(self, user: str) -> dict[str, Any]:
        m = re.search(r"Question:\n(.*?)\n\nAnswer:\n(.*)$", user, re.S)
        prompt, response = (m.group(1), m.group(2)) if m else ("", user)
        q = answer_quality(prompt, response)
        score = max(0.0, min(10.0, round(10 * q + 2.0 * self._noise("judge", prompt, response), 1)))
        idx = 0 if score >= 7 else 1 if score >= 4.5 else 2
        return {"score": score, "rationale": _JUDGE_WORDS[idx]}


# Re-exported so ``JudgeScore`` & friends stay importable from one place for tests.
_SCHEMAS = (TopicList, SubtopicList, InstructionList, QuestionList, QAPair, JudgeScore)
