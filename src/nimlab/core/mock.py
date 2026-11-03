"""Deterministic offline backend used when ``NIMConfig.mock`` is true.

Everything here is pure and synchronous; :class:`nimlab.core.client.NIMClient` wraps it so the
mock goes through the same parsing / usage-tracking code as real HTTP responses.

* chat: plausible text derived from a hash of (model, last user message); valid JSON for a
  schema (found in a ``<json_schema>`` block of a system message, or in ``response_format``); a
  schema-valid tool call on the first turn when ``tools`` are given, plain text once a ``tool``
  message follows.
* embeddings: hashed bag-of-words unit vectors (dim 1024), so shared words => higher cosine.
* rerank: token-overlap scores.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import re
import uuid
from collections import Counter
from functools import lru_cache
from typing import Any

import numpy as np

EMBED_DIM = 1024
JSON_SCHEMA_OPEN = "<json_schema>"
JSON_SCHEMA_CLOSE = "</json_schema>"

# ---------------------------------------------------------------------------- helpers


def _h(*parts: str) -> int:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def message_text(msg: dict[str, Any]) -> str:
    """Text of a chat message whose content is a string or a list of content parts."""
    content = msg.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type", "text") == "text":
                parts.append(str(part.get("text", "")))
        return "\n".join(parts)
    return str(content)


def last_user_text(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return message_text(msg)
    return ""


def tool_results_since_user(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``tool`` messages that come after the most recent ``user`` message."""
    out: list[dict[str, Any]] = []
    for msg in reversed(messages):
        if msg.get("role") == "user":
            break
        if msg.get("role") == "tool":
            out.append(msg)
    out.reverse()
    return out


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4)) if text else 0


def mock_delay() -> float:
    """Optional artificial latency in seconds (``NIMLAB_MOCK_LATENCY``) for demos."""
    try:
        return max(0.0, float(os.environ.get("NIMLAB_MOCK_LATENCY", "0")))
    except ValueError:
        return 0.0


_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    """
    a an the of and or to in on for with is are was were be been being it its this that these
    those as at by from but not no do does did can could should would will shall may might have
    has had i you he she we they them his her our your their my me what which who whom how why
    when where if then than so such into about over under
    """.split()  # noqa: SIM905
)


def _stem(tok: str) -> str:
    for suffix in ("ing", "ed", "s"):
        if tok.endswith(suffix) and len(tok) - len(suffix) >= 3:
            return tok[: -len(suffix)]
    return tok


def tokens(text: str) -> list[str]:
    """Lower-cased, stop-word-free, lightly stemmed tokens."""
    return [_stem(t) for t in _WORD.findall(text.lower()) if t not in _STOP]


def keywords(text: str, n: int = 3) -> list[str]:
    seen: list[str] = []
    for w in _WORD.findall(text.lower()):
        if len(w) > 3 and w not in _STOP and w not in seen:
            seen.append(w)
            if len(seen) == n:
                break
    return seen


def _join_words(words: list[str]) -> str:
    if not words:
        return "your request"
    if len(words) == 1:
        return words[0]
    return ", ".join(words[:-1]) + " and " + words[-1]


# ---------------------------------------------------------------------------- embeddings


@lru_cache(maxsize=100_000)
def _feature_index(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % EMBED_DIM


def embed_text(text: str) -> list[float]:
    """Deterministic hashed bag-of-words (+ bigrams) unit vector of dimension 1024."""
    toks = tokens(text) or ["<empty>"]
    vec = np.zeros(EMBED_DIM, dtype=np.float64)
    for tok, count in Counter(toks).items():
        vec[_feature_index(tok)] += 1.0 + math.log(count)
    for a, b in itertools.pairwise(toks):
        vec[_feature_index(f"{a}\x00{b}")] += 0.5
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:  # pragma: no cover - cannot happen, every token adds weight
        vec[0] = 1.0
        norm = 1.0
    return (vec / norm).tolist()


def mock_embeddings(texts: list[str]) -> list[list[float]]:
    return [embed_text(t) for t in texts]


def mock_rerank_scores(query: str, passages: list[str]) -> list[float]:
    """Token-overlap relevance: query coverage plus a small passage-precision term."""
    q = set(tokens(query))
    scores: list[float] = []
    for passage in passages:
        p = set(tokens(passage))
        inter = len(q & p)
        coverage = inter / len(q) if q else 0.0
        precision = inter / len(p) if p else 0.0
        scores.append(round(coverage + 0.1 * precision, 6))
    return scores


# ---------------------------------------------------------------------------- JSON from schema

_QUERYISH = frozenset(
    {
        "query", "q", "question", "prompt", "input", "search", "search_query", "text",
        "message", "task", "expression", "instruction",
    }
)  # fmt: skip
_STRING_FORMATS = {
    "date-time": "2024-01-01T00:00:00Z",
    "date": "2024-01-01",
    "time": "12:00:00",
    "email": "user@example.com",
    "uri": "https://example.com/resource",
    "url": "https://example.com/resource",
    "hostname": "example.com",
    "ipv4": "192.0.2.1",
    "ipv6": "2001:db8::1",
    "duration": "PT1H",
}
_PATH_TAIL = re.compile(r"([A-Za-z0-9_\-]+)((?:\[\d+\])*)$")


def _humanize(name: str) -> str:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name).replace("_", " ").replace("-", " ")
    spaced = spaced.strip() or "value"
    return spaced[0].upper() + spaced[1:]


class SchemaMocker:
    """Generates a deterministic, schema-valid JSON value from a JSON Schema dict."""

    MAX_DEPTH = 3

    def __init__(self, root: dict[str, Any], seed: str = "", user_text: str = "") -> None:
        self.root = root
        self.seed = seed
        self.user_text = user_text
        self.topic = _join_words(keywords(user_text))

    def generate(self) -> Any:
        return self._gen(self.root, "$", 0)

    # -- internals
    def _idx(self, path: str, n: int) -> int:
        return _h(self.seed, path) % max(n, 1)

    def _lookup(self, ref: str) -> dict[str, Any]:
        node: Any = self.root
        if not ref.startswith("#"):
            return {}
        for part in ref[1:].strip("/").split("/"):
            if not part:
                continue
            part = part.replace("~1", "/").replace("~0", "~")
            node = node.get(part, {}) if isinstance(node, dict) else {}
        return node if isinstance(node, dict) else {}

    @staticmethod
    def _is_null(schema: Any) -> bool:
        return isinstance(schema, dict) and schema.get("type") == "null"

    def _gen(self, schema: Any, path: str, depth: int) -> Any:
        if not isinstance(schema, dict):
            return self._string(path, {})
        if "$ref" in schema:
            target = self._lookup(str(schema["$ref"]))
            merged = {**target, **{k: v for k, v in schema.items() if k != "$ref"}}
            return self._gen(merged, path, depth)
        if "const" in schema:
            return schema["const"]
        if schema.get("enum"):
            values = schema["enum"]
            return values[self._idx(path, len(values))]
        if "allOf" in schema:
            merged = {k: v for k, v in schema.items() if k != "allOf"}
            for part in schema["allOf"]:
                part = (
                    self._lookup(part["$ref"])
                    if isinstance(part, dict) and "$ref" in part
                    else part
                )
                if isinstance(part, dict):
                    props = {**merged.get("properties", {}), **part.get("properties", {})}
                    merged = {**merged, **part}
                    if props:
                        merged["properties"] = props
            return self._gen(merged, path, depth)
        for key in ("anyOf", "oneOf"):
            if key in schema:
                variants = schema[key]
                non_null = [v for v in variants if not self._is_null(v)]
                if not non_null:
                    return None
                if len(non_null) != len(variants) and depth > self.MAX_DEPTH:
                    return None
                return self._gen(non_null[0], path, depth)

        type_ = schema.get("type")
        if isinstance(type_, list):
            non_null_types = [t for t in type_ if t != "null"]
            type_ = non_null_types[0] if non_null_types else "null"
        if type_ is None:
            if "properties" in schema or "additionalProperties" in schema:
                type_ = "object"
            elif "items" in schema or "prefixItems" in schema:
                type_ = "array"
            else:
                type_ = "string"

        if type_ == "null":
            return None
        if type_ == "boolean":
            return bool((_h(self.seed, path) >> 3) & 1)
        if type_ == "integer":
            return self._integer(path, schema)
        if type_ == "number":
            return self._number(path, schema)
        if type_ == "array":
            return self._array(path, schema, depth)
        if type_ == "object":
            return self._object(path, schema, depth)
        return self._string(path, schema)

    def _string(self, path: str, schema: dict[str, Any]) -> str:
        fmt = schema.get("format")
        m = _PATH_TAIL.search(path)
        name, idx = (m.group(1), m.group(2)) if m else ("value", "")
        if fmt == "uuid":
            return str(uuid.UUID(int=_h(self.seed, path, "uuid") << 64 | _h(path, self.seed)))
        if fmt in _STRING_FORMATS:
            value = _STRING_FORMATS[fmt]
        elif name.lower() in _QUERYISH and not idx:
            value = self.user_text.strip()[:200] or self.topic
        else:
            label = _humanize(name)
            if idx:
                label += " " + str(int(idx.strip("[]").split("][")[-1]) + 1)
            joiner = ("about", "regarding", "on")[self._idx(path, 3)]
            value = f"{label} {joiner} {self.topic}"
        max_len = schema.get("maxLength")
        min_len = schema.get("minLength", 0)
        if isinstance(max_len, int) and len(value) > max_len:
            value = value[:max_len]
        if isinstance(min_len, int) and len(value) < min_len:
            value = (value + " " + "x" * min_len)[: max(min_len, len(value))]
        return value

    @staticmethod
    def _bounds(schema: dict[str, Any], integer: bool) -> tuple[float, float, bool, bool]:
        """(low, high, low_exclusive, high_exclusive) with sensible defaults when unbounded."""
        lo: float | None = schema.get("minimum")
        hi: float | None = schema.get("maximum")
        lo_ex = hi_ex = False
        ex_min, ex_max = schema.get("exclusiveMinimum"), schema.get("exclusiveMaximum")
        if lo is None and isinstance(ex_min, (int, float)) and not isinstance(ex_min, bool):
            lo, lo_ex = ex_min, True
        if hi is None and isinstance(ex_max, (int, float)) and not isinstance(ex_max, bool):
            hi, hi_ex = ex_max, True
        default_lo, default_hi = (1.0, 10.0) if integer else (0.0, 1.0)
        if lo is None and hi is None:
            return default_lo, default_hi, False, False
        if lo is None:
            assert hi is not None
            return min(default_lo, hi - (1 if integer else 0.5)), hi, False, hi_ex
        if hi is None:
            return lo, lo + (9 if integer else 1.0), lo_ex, False
        return lo, hi, lo_ex, hi_ex

    def _integer(self, path: str, schema: dict[str, Any]) -> int:
        lo, hi, lo_ex, hi_ex = self._bounds(schema, True)
        lo_i = math.floor(lo) + 1 if lo_ex else math.ceil(lo)
        hi_i = math.ceil(hi) - 1 if hi_ex else math.floor(hi)
        if hi_i < lo_i:
            hi_i = lo_i
        step = schema.get("multipleOf")
        if isinstance(step, (int, float)) and step >= 1 and float(step).is_integer():
            step = int(step)
            first = math.ceil(lo_i / step) * step
            count = (hi_i - first) // step + 1
            if count >= 1:
                return int(first + (_h(self.seed, path) % count) * step)
        return int(lo_i + _h(self.seed, path) % (hi_i - lo_i + 1))

    def _number(self, path: str, schema: dict[str, Any]) -> float:
        lo, hi, lo_ex, hi_ex = self._bounds(schema, False)
        if hi < lo:
            hi = lo
        frac = (_h(self.seed, path) % 1000) / 1000.0
        if lo_ex or hi_ex:
            frac = 0.001 + frac * 0.998
        value = round(lo + frac * (hi - lo), 3)
        if (lo_ex and value <= lo) or (hi_ex and value >= hi):
            value = lo + (hi - lo) / 2
        return float(min(max(value, lo), hi))

    def _array(self, path: str, schema: dict[str, Any], depth: int) -> list[Any]:
        prefix = schema.get("prefixItems")
        if isinstance(prefix, list):
            return [self._gen(s, f"{path}[{i}]", depth + 1) for i, s in enumerate(prefix)]
        min_items = int(schema.get("minItems", 0) or 0)
        max_items = schema.get("maxItems")
        count = max(min_items, 2)
        if isinstance(max_items, int):
            count = min(count, max_items)
        if depth >= self.MAX_DEPTH and min_items == 0:
            count = 0
        items = schema.get("items", {})
        out: list[Any] = []
        seen: set[str] = set()
        for i in range(count):
            value = self._gen(items, f"{path}[{i}]", depth + 1)
            if schema.get("uniqueItems"):
                key = json.dumps(value, sort_keys=True, default=str)
                if key in seen:
                    continue
                seen.add(key)
            out.append(value)
        return out

    def _object(self, path: str, schema: dict[str, Any], depth: int) -> dict[str, Any]:
        props = schema.get("properties") or {}
        out: dict[str, Any] = {
            name: self._gen(sub, f"{path}.{name}", depth + 1) for name, sub in props.items()
        }
        extra = schema.get("additionalProperties")
        if not props and isinstance(extra, dict) and depth < self.MAX_DEPTH:
            for i in range(2):
                out[f"key_{i + 1}"] = self._gen(extra, f"{path}.key_{i + 1}", depth + 1)
        return out


def generate_json(schema: dict[str, Any], seed: str = "", user_text: str = "") -> Any:
    """Deterministic placeholder value that validates against ``schema``."""
    return SchemaMocker(schema, seed=seed, user_text=user_text).generate()


# ---------------------------------------------------------------------------- chat

_OPENERS = (
    "Here is a concise answer regarding {topic}.",
    "Let's take {topic} step by step.",
    "Short version: {topic} comes down to a handful of key points.",
    "Considering {topic}, the main points are as follows.",
)
_MIDDLES = (
    "First, it helps to define the terms clearly and agree on the goal.",
    "The most important factor is how {kw1} interacts with {kw2}.",
    "In practice, the trade-offs between {kw1} and {kw2} dominate the decision.",
    "There are usually three things to check: scope, constraints, and evidence.",
    "A sensible default is to start simple and measure before optimizing.",
    "Edge cases around {kw2} deserve extra attention.",
    "Evidence from similar situations suggests keeping {kw1} explicit and testable.",
)
_CLOSERS = (
    "Overall, a measured approach to {topic} tends to work best.",
    "In summary, {topic} is manageable once the key trade-offs are explicit.",
    "That covers the essentials; I can go deeper on any part.",
)
_REASONING_MODELS = ("deepseek-r1", "qwq", "/r1", "nemotron-3-ultra", "deepseek-v4")


def is_reasoning_model(model: str) -> bool:
    low = model.lower()
    return any(s in low for s in _REASONING_MODELS)


def mock_text(model: str, messages: list[dict[str, Any]]) -> str:
    """Plausible prose answer; deterministic in (model, last user message, tool results)."""
    user = last_user_text(messages)
    tool_msgs = tool_results_since_user(messages)
    if tool_msgs:
        snippet = " ".join(message_text(m).strip() for m in tool_msgs)
        snippet = re.sub(r"\s+", " ", snippet)[:400]
        return f"Based on the tool results, here is the answer: {snippet}"
    kws = keywords(user)
    topic = _join_words(kws)
    kw1 = kws[0] if kws else "the inputs"
    kw2 = kws[1] if len(kws) > 1 else "the constraints"
    seed = _h(model, user)

    def pick(options: tuple[str, ...], salt: int) -> str:
        return options[(seed >> (salt * 5)) % len(options)].format(topic=topic, kw1=kw1, kw2=kw2)

    middles = [pick(_MIDDLES, 1)]
    second = pick(_MIDDLES, 2)
    if second not in middles:
        middles.append(second)
    return " ".join([pick(_OPENERS, 0), *middles, pick(_CLOSERS, 3)])


def _think_block(model: str, messages: list[dict[str, Any]]) -> str:
    topic = _join_words(keywords(last_user_text(messages)))
    return (
        f"<think>\nThe user is asking about {topic}. I should weigh the key factors and answer "
        "concisely.\n</think>\n\n"
    )


def find_schema(
    messages: list[dict[str, Any]], response_format: dict[str, Any] | None
) -> dict[str, Any] | None:
    """JSON schema requested via a ``<json_schema>`` block or ``response_format``."""
    pattern = re.compile(
        re.escape(JSON_SCHEMA_OPEN) + r"\s*(.*?)\s*" + re.escape(JSON_SCHEMA_CLOSE), re.DOTALL
    )
    for msg in messages:
        found = pattern.search(message_text(msg))
        if found:
            try:
                parsed = json.loads(found.group(1))
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    if isinstance(response_format, dict) and response_format.get("type") == "json_schema":
        js = response_format.get("json_schema")
        if isinstance(js, dict):
            schema = js.get("schema", js)
            if isinstance(schema, dict):
                return schema
    return None


def _tool_def(tool: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    fn = tool.get("function", tool)
    return str(fn.get("name", "tool")), fn.get("parameters") or {"type": "object", "properties": {}}


def _pick_tool(tools: list[dict[str, Any]], tool_choice: Any) -> dict[str, Any] | None:
    if not tools or tool_choice == "none":
        return None
    if isinstance(tool_choice, dict):
        wanted = tool_choice.get("function", {}).get("name") or tool_choice.get("name")
        for tool in tools:
            if _tool_def(tool)[0] == wanted:
                return tool
    return tools[0]


def mock_chat_response(
    model: str,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
    response_format: dict[str, Any] | None = None,
    max_tokens: int = 1024,
) -> dict[str, Any]:
    """An OpenAI-style ``chat.completion`` payload."""
    user = last_user_text(messages)
    prompt_tokens = sum(estimate_tokens(message_text(m)) for m in messages)
    seed = f"{model}|{user}"
    finish = "stop"
    message: dict[str, Any]

    chosen = (
        _pick_tool(tools, tool_choice) if tools and not tool_results_since_user(messages) else None
    )
    if chosen is not None:
        name, params = _tool_def(chosen)
        args = generate_json(params, seed=f"{seed}|{name}", user_text=user)
        if not isinstance(args, dict):
            args = {}
        arguments = json.dumps(args)
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"call_{_h(seed, name):016x}"[:21],
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            ],
        }
        finish = "tool_calls"
        completion_tokens = estimate_tokens(arguments)
    else:
        schema = find_schema(messages, response_format)
        wants_json = isinstance(response_format, dict) and response_format.get("type") in {
            "json_object",
            "json_schema",
        }
        if schema is not None:
            content = json.dumps(generate_json(schema, seed=seed, user_text=user), indent=2)
        elif wants_json:
            content = json.dumps({"answer": mock_text(model, messages)})
        else:
            content = mock_text(model, messages)
            words = content.split(" ")
            budget = max(1, int(max_tokens / 1.3))
            if len(words) > budget:
                content = " ".join(words[:budget])
                finish = "length"
        if is_reasoning_model(model):
            content = _think_block(model, messages) + content
        message = {"role": "assistant", "content": content}
        completion_tokens = estimate_tokens(content)

    return {
        "id": f"chatcmpl-mock-{_h(seed):x}",
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def mock_stream_chunks(text: str) -> list[str]:
    """Split text into word chunks (whitespace attached to the preceding word)."""
    return re.findall(r"\S+\s*|\s+", text)
