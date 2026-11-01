"""SmartRouter: classify a prompt into a task category, then pick the best model for it.

Classification (``classifier="auto"``)
    1. **Embedding kNN** over bundled labelled exemplar prompts (``exemplars.json``): the
       ``k`` nearest exemplars vote with weight ``exp(sim / 0.1)``.
    2. **Keyword heuristics** when the kNN is not confident (low similarity or split vote) or
       embeddings are unavailable.
    3. Optional **LLM classification** (``Role.FAST`` model through ``chat_json``) via
       ``classifier="llm"`` or ``use_llm=True`` (used when the cheap methods stay uncertain).

Selection
    ``utility(m) = quality(m, category) - lambda_latency * latency_s(m, category)
    - lambda_cost * cost_tier(m)``; the model with the highest utility wins. Quality and latency
    come from the latest leaderboard data; with no data the registry's role default is used.
    Every decision lists all candidates, the three terms of each utility and plain-language
    reasons, so a routing choice can always be explained.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from nimlab.core import Role, default_registry
from nimlab.core.errors import NIMError
from nimlab.core.mock import mock_embeddings
from nimlab.core.registry import ModelRegistry

from .leaderboard import Leaderboard
from .llm import message_text
from .results import RunStore
from .suites import CATEGORIES

__all__ = [
    "CATEGORY_ROLES",
    "PROFILES",
    "Candidate",
    "CategoryChoice",
    "Classification",
    "RoutingDecision",
    "SmartRouter",
    "keyword_scores",
    "load_exemplars",
    "prompt_from_messages",
]

EXEMPLARS_PATH = Path(__file__).parent / "exemplars.json"

#: Registry role used when there is no leaderboard data for a category.
CATEGORY_ROLES: dict[str, Role] = {
    "math_reasoning": Role.REASONING,
    "coding": Role.CODER,
    "factual_qa": Role.CHAT,
    "instruction_following": Role.CHAT,
    "json_compliance": Role.CHAT,
    "summarization": Role.CHAT,
    "general_chat": Role.CHAT,
    "vision": Role.VISION,
}

#: ``name -> (lambda_latency per second, lambda_cost per cost tier)``
PROFILES: dict[str, tuple[float, float]] = {
    "quality": (0.002, 0.0),
    "balanced": (0.02, 0.03),
    "speed": (0.08, 0.02),
    "economy": (0.02, 0.10),
}
DEFAULT_LATENCY_S = 5.0
KNN_TEMPERATURE = 0.1


class CategoryChoice(BaseModel):
    """Structured output of the LLM classifier."""

    category: Literal[
        "math_reasoning",
        "coding",
        "factual_qa",
        "instruction_following",
        "json_compliance",
        "summarization",
        "general_chat",
    ]
    reason: str = ""


class Neighbor(BaseModel):
    category: str
    similarity: float
    text: str


class Classification(BaseModel):
    category: str
    method: Literal["knn", "keywords", "llm", "default", "override", "vision"]
    confidence: float = 0.0
    scores: dict[str, float] = Field(default_factory=dict)
    neighbors: list[Neighbor] = Field(default_factory=list)
    keyword_scores: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class Candidate(BaseModel):
    model: str
    quality: float | None = None
    quality_source: Literal["category", "overall", "none"] = "none"
    latency_s: float | None = None
    cost_tier: int | None = None
    utility: float | None = None
    quality_term: float = 0.0
    latency_penalty: float = 0.0
    cost_penalty: float = 0.0


class RoutingDecision(BaseModel):
    prompt: str
    category: str
    classifier: str
    confidence: float
    model: str
    source: Literal["leaderboard", "registry"]
    profile: str
    latency_weight: float
    cost_weight: float
    run_id: str | None = None
    candidates: list[Candidate] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    classification: Classification | None = None

    def explain(self) -> str:
        lines = [f"Routed to {self.model}", *[f"  - {r}" for r in self.reasons]]
        return "\n".join(lines)


# ------------------------------------------------------------------------------ keywords


def _any(*alternatives: str) -> str:
    """Whole-word alternation regex."""
    return r"\b(" + "|".join(alternatives) + r")\b"


_COUNT = r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "math_reasoning": [
        (
            _any(
                "calculate",
                "compute",
                "solve",
                "equation",
                "probability",
                "derivative",
                "integral",
                "factorial",
                "prime",
            ),
            2,
        ),
        (
            _any(
                "percent(age)?",
                "ratio",
                "fraction",
                "remainder",
                "average",
                "median",
                "sum of",
                "product of",
                "how many",
                "how much",
            ),
            1.5,
        ),
        (
            _any(
                "area",
                "perimeter",
                "volume",
                "speed",
                "km/h",
                "interest",
                "angle",
                "triangle",
                "algebra",
                "geometry",
            ),
            1,
        ),
        (r"\d+\s*[-+*/^x\xd7\xf7]\s*\d+", 1.5),
        (r"\d+\s*%", 1),
    ],
    "coding": [
        (r"```|\bdef \w+\(|\bfunction\s+\w*\(|=>|\bimport \w+|#include|;\s*$", 2),
        (
            _any(
                "python",
                "javascript",
                "typescript",
                "java",
                r"c\+\+",
                "rust",
                "golang",
                "kotlin",
                "swift",
                "sql",
                "bash",
                "regex",
                "html",
                "css",
            ),
            2,
        ),
        (
            _any(
                "function",
                "class",
                "method",
                "bug",
                "compile",
                "stack ?trace",
                "exception",
                "algorithm",
                "code",
                "script",
                "unit tests?",
                "refactor",
                "implement",
                "debug",
                "segfault",
                "api",
                "library",
                "snippet",
            ),
            1.5,
        ),
    ],
    "factual_qa": [
        (r"^\s*(who|what|when|where|which|how (tall|many|long|far|old|big))\b", 1.5),
        (
            _any(
                "capital of",
                "invented",
                "discovered",
                "founded",
                "born in",
                "in what year",
                "largest",
                "smallest",
                "longest",
                "tallest",
                "first (person|woman|man)",
                "how many (bones|people|countries)",
            ),
            2,
        ),
    ],
    "instruction_following": [
        (
            _any(
                "exactly",
                "at least",
                "at most",
                "no more than",
                "fewer than",
                "less than",
                "more than",
            )
            + r".{0,30}"
            + _any("words?", "sentences?", "bullets?", "paragraphs?", "lines?", "items?"),
            3,
        ),
        (
            _any(
                "all caps",
                "capital letters",
                "uppercase",
                "lowercase",
                "title case",
                "bullet points?",
                "numbered list",
                "double angle",
                "without (using|any)",
                "do not (use|include)",
                "don't use",
                "end (your|with)",
                "start (with|each)",
            ),
            2,
        ),
        (_COUNT + r" " + _any("words?", "sentences?", "paragraphs?", "bullets?"), 2),
    ],
    "json_compliance": [
        (r"\bjson\b", 3),
        (
            _any(
                "schema",
                "valid json",
                "keys?",
                "fields?",
                "array of",
                "object with",
                "structured output",
                "yaml",
                "csv",
            ),
            1,
        ),
    ],
    "summarization": [
        (
            _any(
                "summari[sz]e",
                "summary",
                "tl;?dr",
                "condense",
                "key points",
                "main points",
                "recap",
                "synopsis",
                "abstract",
                "executive summary",
                "boil .* down",
                "shorten",
            ),
            3,
        ),
    ],
    "general_chat": [
        (r"^\s*(hi|hello|hey|good (morning|evening))\b", 3),
        (
            _any(
                "thanks",
                "thank you",
                "recommend",
                "advice",
                "brainstorm",
                "ideas?",
                "opinion",
                "help me",
                "should i",
                "tips",
                "joke",
                "poem",
                "story",
                "plan a",
                "feel",
            ),
            1.5,
        ),
    ],
}
_COMPILED = {
    cat: [(re.compile(p, re.IGNORECASE | re.MULTILINE), w) for p, w in rules]
    for cat, rules in _KEYWORDS.items()
}


def keyword_scores(text: str) -> dict[str, float]:
    """Heuristic evidence per category (0 = none); long text leans towards summarization."""
    scores = {c: 0.0 for c in CATEGORIES}
    for cat, rules in _COMPILED.items():
        for rx, weight in rules:
            if rx.search(text):
                scores[cat] += weight
    if (
        len(text) > 700
        and scores["summarization"] == 0
        and not any(scores[c] for c in ("coding", "json_compliance"))
    ):
        scores["summarization"] += 1.0
    return scores


def _best(scores: dict[str, float]) -> tuple[str, float] | None:
    top = max(scores.items(), key=lambda kv: kv[1], default=None)
    return top if top and top[1] > 0 else None


def load_exemplars(path: str | Path | None = None) -> list[tuple[str, str]]:
    """``[(category, prompt), ...]`` from the bundled (or a custom) exemplar file."""
    data = json.loads(Path(path or EXEMPLARS_PATH).read_text(encoding="utf-8"))
    return [(cat, text) for cat, texts in data.items() for text in texts]


def prompt_from_messages(messages: list[dict[str, Any]]) -> tuple[str, bool]:
    """Text to classify (the last user message) and whether any message carries an image."""
    has_image = any(
        isinstance(m.get("content"), list)
        and any(
            isinstance(p, dict) and p.get("type") in ("image_url", "image") for p in m["content"]
        )
        for m in messages
    )
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return message_text(msg), has_image
    return (message_text(messages[-1]) if messages else ""), has_image


# ------------------------------------------------------------------------------ router


class SmartRouter:
    """Chooses a model per prompt from leaderboard data. See the module docstring."""

    def __init__(
        self,
        leaderboard: Leaderboard | None = None,
        *,
        client: Any = None,
        registry: ModelRegistry | None = None,
        runs_dir: str | Path | None = None,
        profile: str = "balanced",
        latency_weight: float | None = None,
        cost_weight: float | None = None,
        classifier: Literal["auto", "knn", "keywords", "llm"] = "auto",
        use_llm: bool = False,
        embed_model: str | None = None,
        fast_model: str | None = None,
        k: int = 5,
        allow: Iterable[str] | None = None,
        min_confidence: float = 0.5,
        min_similarity: float = 0.15,
        auto_reload: bool = True,
        exemplars_path: str | Path | None = None,
    ) -> None:
        self.client = client
        self.registry = registry or default_registry()
        self.store = RunStore(runs_dir)
        self.profile = profile
        self._latency_weight = latency_weight
        self._cost_weight = cost_weight
        self.classifier = classifier
        self.use_llm = use_llm
        self.embed_model = embed_model
        self.fast_model = fast_model
        self.k = max(1, k)
        self.allow = set(allow) if allow else None
        self.min_confidence = min_confidence
        self.min_similarity = min_similarity
        self.auto_reload = auto_reload and leaderboard is None
        self._leaderboard = leaderboard
        self._signature: tuple[Any, ...] | None = None
        self._exemplars = load_exemplars(exemplars_path)
        self._matrix: np.ndarray | None = None
        self._matrix_model: str | None = None
        self._lock: asyncio.Lock | None = None
        self._embed_retry_at = 0.0
        if leaderboard is None:
            self._load_leaderboard()

    # ------------------------------------------------------------------ leaderboard data
    def _dir_signature(self) -> tuple[Any, ...]:
        root = self.store.root
        if not root.is_dir():
            return ()
        return tuple(sorted((p.name, p.stat().st_mtime_ns) for p in root.glob("*.json")))

    def _load_leaderboard(self) -> None:
        self._signature = self._dir_signature()
        try:
            merged = self.store.merged()
        except (OSError, ValueError):
            merged = None
        self._leaderboard = (
            Leaderboard.from_run(merged, registry=self.registry, n_boot=0) if merged else None
        )

    @property
    def leaderboard(self) -> Leaderboard | None:
        if self.auto_reload and self._dir_signature() != self._signature:
            self._load_leaderboard()
        return self._leaderboard

    def set_leaderboard(self, leaderboard: Leaderboard | None) -> None:
        self._leaderboard = leaderboard
        self.auto_reload = False

    def weights(self, profile: str | None = None) -> tuple[str, float, float]:
        name = profile or self.profile
        if name not in PROFILES:
            raise ValueError(f"unknown profile {name!r}; choose from {', '.join(PROFILES)}")
        lat, cost = PROFILES[name]
        if profile is None:
            lat = self._latency_weight if self._latency_weight is not None else lat
            cost = self._cost_weight if self._cost_weight is not None else cost
        return name, lat, cost

    # ------------------------------------------------------------------ classification
    async def _exemplar_matrix(self) -> np.ndarray:
        """Unit-normalised exemplar embeddings (computed once per embedding model)."""
        model = self.embed_model or self.registry.pick(Role.EMBED)
        if self._matrix is not None and self._matrix_model == model:
            return self._matrix
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            if self._matrix is None or self._matrix_model != model:
                texts = [t for _, t in self._exemplars]
                if self.client is None:
                    vecs = mock_embeddings(texts)
                else:
                    vecs = await self.client.embed(model, texts, input_type="passage")
                self._matrix = _unit(np.asarray(vecs, dtype=float))
                self._matrix_model = model
        return self._matrix

    async def _embed_query(self, text: str) -> np.ndarray:
        model = self.embed_model or self.registry.pick(Role.EMBED)
        if self.client is None:
            vec = mock_embeddings([text])[0]
        else:
            vec = (await self.client.embed(model, [text], input_type="query"))[0]
        return _unit(np.asarray(vec, dtype=float)[None, :])[0]

    async def classify_knn(self, text: str) -> Classification | None:
        if time.monotonic() < self._embed_retry_at:
            return None
        try:
            matrix = await self._exemplar_matrix()
            query = await self._embed_query(text)
        except (NIMError, OSError, ValueError, KeyError) as exc:
            self._embed_retry_at = time.monotonic() + 60.0
            return Classification(
                category="general_chat", method="default", notes=[f"embedding unavailable: {exc}"]
            )
        if matrix.shape[1] != query.shape[0]:
            self._matrix = None  # embedding model changed under us
            return None
        sims = matrix @ query
        order = np.argsort(-sims)[: self.k]
        weights = np.exp(sims[order] / KNN_TEMPERATURE)
        votes: dict[str, float] = {}
        for idx, w in zip(order, weights, strict=True):
            cat = self._exemplars[int(idx)][0]
            votes[cat] = votes.get(cat, 0.0) + float(w)
        total = sum(votes.values())
        scores = {c: round(v / total, 4) for c, v in sorted(votes.items(), key=lambda kv: -kv[1])}
        best = next(iter(scores))
        neighbors = [
            Neighbor(
                category=self._exemplars[int(i)][0],
                similarity=round(float(sims[i]), 4),
                text=self._exemplars[int(i)][1],
            )
            for i in order[:3]
        ]
        return Classification(
            category=best,
            method="knn",
            confidence=scores[best],
            scores=scores,
            neighbors=neighbors,
        )

    def classify_keywords(self, text: str) -> Classification:
        scores = keyword_scores(text)
        top = _best(scores)
        total = sum(scores.values())
        if top is None:
            return Classification(
                category="general_chat",
                method="default",
                confidence=0.0,
                keyword_scores=scores,
                notes=["no keyword matched; defaulting to general_chat"],
            )
        return Classification(
            category=top[0],
            method="keywords",
            confidence=round(top[1] / total, 4),
            scores={c: round(v / total, 4) for c, v in scores.items() if v > 0},
            keyword_scores={c: v for c, v in scores.items() if v > 0},
        )

    async def classify_llm(self, text: str) -> Classification | None:
        if self.client is None:
            return None
        model = self.fast_model or self.registry.pick(Role.FAST)
        messages = [
            {
                "role": "system",
                "content": (
                    "Classify the user's request into exactly one task category: "
                    + ", ".join(CATEGORIES)
                    + ". math_reasoning = calculations and word problems; coding = writing or "
                    "debugging code; factual_qa = short factual questions; instruction_following = "
                    "strict formatting or length constraints; json_compliance = structured JSON "
                    "output; summarization = condensing a given text; general_chat = everything "
                    "else. Reply with JSON only."
                ),
            },
            {"role": "user", "content": text[:2000]},
        ]
        try:
            choice = await self.client.chat_json(model, messages, CategoryChoice, temperature=0.0)
        except NIMError as exc:
            return Classification(
                category="general_chat", method="default", notes=[f"LLM classifier failed: {exc}"]
            )
        return Classification(
            category=choice.category,
            method="llm",
            confidence=0.9,
            scores={choice.category: 1.0},
            notes=[f"{model}: {choice.reason}"] if choice.reason else [],
        )

    async def classify(self, text: str) -> Classification:
        """Pick a task category for ``text`` (see the module docstring for the strategy)."""
        text = text.strip()
        kw = self.classify_keywords(text)
        if self.classifier == "keywords":
            return kw
        if self.classifier == "llm":
            llm = await self.classify_llm(text)
            if llm and llm.method == "llm":
                llm.keyword_scores = kw.keyword_scores
                return llm
        knn = await self.classify_knn(text) if self.classifier in ("auto", "knn", "llm") else None
        notes: list[str] = []
        if knn and knn.method == "knn":
            top_sim = knn.neighbors[0].similarity if knn.neighbors else 0.0
            if knn.confidence >= self.min_confidence and top_sim >= self.min_similarity:
                knn.keyword_scores = kw.keyword_scores
                if kw.method == "keywords" and kw.category != knn.category:
                    knn.notes.append(
                        f"keyword heuristic disagreed ({kw.category}); kNN vote was confident"
                    )
                return knn
            notes.append(
                f"kNN unsure (vote share {knn.confidence:.2f}, top similarity {top_sim:.2f}); "
                "falling back to keywords"
            )
        elif knn:
            notes.extend(knn.notes)
        if self.use_llm and kw.confidence < 0.6:
            llm = await self.classify_llm(text)
            if llm and llm.method == "llm":
                llm.notes = [*notes, *llm.notes]
                llm.keyword_scores = kw.keyword_scores
                return llm
        kw.notes = [*notes, *kw.notes]
        return kw

    # ------------------------------------------------------------------ selection
    def choose(
        self,
        category: str,
        *,
        profile: str | None = None,
        allow: Iterable[str] | None = None,
    ) -> tuple[str, Literal["leaderboard", "registry"], list[Candidate], list[str], str | None]:
        """Pick a model; returns ``(model, source, candidates, reasons, run_id)``."""
        name, lam_lat, lam_cost = self.weights(profile)
        allowed = set(allow) if allow else self.allow
        lb = self.leaderboard
        reasons: list[str] = []
        candidates: list[Candidate] = []

        if lb and category != "vision":
            pool = [s.model for s in lb.standings if not allowed or s.model in allowed]
            raw: list[Candidate] = []
            have_cat = [m for m in pool if lb.category_quality(m, category) is not None]
            for m in pool:
                q = lb.category_quality(m, category)
                src: Literal["category", "overall", "none"] = "category"
                if q is None:  # no suite of this category for the model: fall back to its overall
                    standing = lb.get(m)
                    q, src = (standing.overall if standing else None), "overall"
                if q is None:
                    continue
                info = self.registry.find(m)
                raw.append(
                    Candidate(
                        model=m,
                        quality=round(q, 4),
                        quality_source=src,
                        latency_s=lb.category_latency(m, category),
                        cost_tier=info.cost_tier if info else None,
                    )
                )
            if raw:
                known = [c.latency_s for c in raw if c.latency_s is not None]
                fallback_lat = float(np.median(known)) if known else DEFAULT_LATENCY_S
                for c in raw:
                    lat = c.latency_s if c.latency_s is not None else fallback_lat
                    tier = c.cost_tier if c.cost_tier is not None else 3
                    c.latency_s = round(lat, 4)
                    c.quality_term = c.quality or 0.0
                    c.latency_penalty = round(lam_lat * lat, 4)
                    c.cost_penalty = round(lam_cost * tier, 4)
                    c.utility = round(c.quality_term - c.latency_penalty - c.cost_penalty, 4)
                candidates = sorted(raw, key=lambda c: (-(c.utility or 0.0), c.model))
                best = candidates[0]
                suites = lb.suites_in(category)
                basis = (
                    f"{category} suite{'s' if len(suites) != 1 else ''} ({', '.join(suites)})"
                    if suites and category != "general_chat"
                    else "the overall score across all suites"
                )
                reasons.append(
                    f"Using leaderboard run {lb.run_id}: quality is measured on {basis}"
                    + (" (mock/simulated data)" if lb.mock else "")
                    + "."
                )
                reasons.append(
                    f"Profile '{name}': utility = quality - {lam_lat:g} x latency(s) - "
                    f"{lam_cost:g} x cost_tier."
                )
                reasons.append(
                    f"{best.model} has the highest utility {best.utility:.3f} "
                    f"(quality {best.quality:.3f}, latency {best.latency_s:.2f}s, "
                    f"cost tier {best.cost_tier if best.cost_tier is not None else '?'})."
                )
                if len(candidates) > 1:
                    runner = candidates[1]
                    reasons.append(
                        f"Runner-up {runner.model} scored {runner.utility:.3f} "
                        f"(margin {best.utility - runner.utility:.3f})."  # type: ignore[operator]
                    )
                if len(have_cat) < len(pool) and category != "general_chat":
                    reasons.append(
                        f"{len(pool) - len(have_cat)} model(s) lack {category} data and were "
                        "scored by their overall result."
                    )
                return best.model, "leaderboard", candidates, reasons, lb.run_id

        role = CATEGORY_ROLES.get(category, Role.CHAT)
        picked = self.registry.pick(role)
        if allowed and picked not in allowed:
            for info in self.registry.by_role(role):
                if info.id in allowed:
                    picked = info.id
                    break
            else:
                picked = sorted(allowed)[0]
        candidates = [
            Candidate(model=info.id, cost_tier=info.cost_tier)
            for info in self.registry.by_role(role)[:5]
            if not allowed or info.id in allowed
        ]
        if not any(c.model == picked for c in candidates):
            info = self.registry.find(picked)
            candidates.insert(
                0, Candidate(model=picked, cost_tier=info.cost_tier if info else None)
            )
        reason = (
            "no leaderboard data for the image/vision category"
            if category == "vision"
            else "no leaderboard data found (run `nimlab arena run` first)"
        )
        reasons.append(f"Fallback to the registry default for role '{role.value}': {reason}.")
        return picked, "registry", candidates, reasons, None

    # ------------------------------------------------------------------ public routing API
    async def route(
        self,
        prompt: str | list[dict[str, Any]],
        *,
        category: str | None = None,
        profile: str | None = None,
        allow: Iterable[str] | None = None,
    ) -> RoutingDecision:
        """Classify ``prompt`` (text or chat messages) and choose a model."""
        if isinstance(prompt, str):
            text, has_image = prompt, False
        else:
            text, has_image = prompt_from_messages(prompt)
        if category is not None and category not in CATEGORY_ROLES:
            raise ValueError(
                f"unknown category {category!r}; use one of {', '.join(CATEGORY_ROLES)}"
            )
        if category is not None:
            cls = Classification(category=category, method="override", confidence=1.0)
        elif has_image:
            cls = Classification(
                category="vision",
                method="vision",
                confidence=1.0,
                notes=["request contains an image"],
            )
        else:
            cls = await self.classify(text)
        model, source, candidates, reasons, run_id = self.choose(
            cls.category, profile=profile, allow=allow
        )
        name, lam_lat, lam_cost = self.weights(profile)
        head = _classification_reason(cls)
        return RoutingDecision(
            prompt=text[:300],
            category=cls.category,
            classifier=cls.method,
            confidence=round(cls.confidence, 4),
            model=model,
            source=source,
            profile=name,
            latency_weight=lam_lat,
            cost_weight=lam_cost,
            run_id=run_id,
            candidates=candidates,
            reasons=[head, *cls.notes, *reasons],
            classification=cls,
        )

    def route_sync(self, prompt: str | list[dict[str, Any]], **kw: Any) -> RoutingDecision:
        """Blocking wrapper around :meth:`route` (do not call from a running event loop)."""
        return asyncio.run(self.route(prompt, **kw))


def _unit(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1.0, norms)


def _classification_reason(cls: Classification) -> str:
    if cls.method == "knn":
        near = cls.neighbors[0] if cls.neighbors else None
        extra = (
            f'; nearest exemplar: "{near.text[:70]}" (similarity {near.similarity:.2f})'
            if near
            else ""
        )
        return (
            f"Classified as '{cls.category}' by embedding kNN "
            f"(vote share {cls.confidence:.0%}){extra}."
        )
    if cls.method == "keywords":
        hits = ", ".join(
            f"{c}={v:g}" for c, v in sorted(cls.keyword_scores.items(), key=lambda kv: -kv[1])
        )
        return f"Classified as '{cls.category}' by keyword heuristics ({hits})."
    if cls.method == "llm":
        return f"Classified as '{cls.category}' by the LLM classifier."
    if cls.method == "override":
        return f"Category '{cls.category}' was set explicitly."
    if cls.method == "vision":
        return "The request contains an image, so a vision-capable model is required."
    return f"No strong signal: defaulting to '{cls.category}'."
