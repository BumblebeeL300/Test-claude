"""Unit tests for the deterministic mock backend (pure functions)."""

from __future__ import annotations

import json
import math
import uuid
from datetime import date, datetime
from enum import Enum
from typing import Annotated, Any, Literal, Optional, Union

import pytest
from pydantic import BaseModel, Field

from nimlab.core import cosine, extract_json
from nimlab.core.mock import (
    EMBED_DIM,
    JSON_SCHEMA_CLOSE,
    JSON_SCHEMA_OPEN,
    generate_json,
    mock_chat_response,
    mock_embeddings,
    mock_rerank_scores,
    mock_stream_chunks,
    mock_text,
)


class Color(str, Enum):
    RED = "red"
    BLUE = "blue"


class Level(int, Enum):
    LOW = 1
    HIGH = 2


class Leaf(BaseModel):
    name: str
    score: float = Field(ge=0, le=1)


class Branch(BaseModel):
    title: str
    leaves: list[Leaf]
    best: Optional[Leaf] = None  # noqa: UP045


class Everything(BaseModel):
    s: str
    i: int
    f: float
    b: bool
    lit: Literal["x", "y", "z"]
    single: Literal["only"]
    color: Color
    level: Level
    opt_int: Optional[int] = None  # noqa: UP045
    union: Union[int, str] = 1  # noqa: UP007
    strs: list[str]
    nested: Branch
    many: list[Branch]
    mapping: dict[str, int]
    bounded_i: Annotated[int, Field(ge=5, le=7)]
    gt_i: Annotated[int, Field(gt=10, lt=13)]
    even: Annotated[int, Field(multiple_of=4, ge=3, le=30)]
    frac: Annotated[float, Field(gt=0.0, lt=1.0)]
    short: Annotated[str, Field(min_length=3, max_length=5)]
    padded: Annotated[str, Field(min_length=40)]
    at_least_three: Annotated[list[int], Field(min_length=3)]
    at_most_one: Annotated[list[int], Field(max_length=1)]
    unique: set[Color]
    when: datetime
    day: date
    ident: uuid.UUID
    pair: tuple[int, str]
    any_val: Any = None


class Node(BaseModel):
    value: int
    children: list[Node] = []
    parent: Optional[Node] = None  # noqa: UP045


class Wrapper(BaseModel):
    node: Node


@pytest.mark.parametrize("model", [Leaf, Branch, Everything, Node, Wrapper])
@pytest.mark.parametrize("seed", ["", "hello", "a much longer seed about caching strategies"])
def test_generated_json_validates(model: type[BaseModel], seed: str) -> None:
    value = generate_json(model.model_json_schema(), seed=seed, user_text=seed)
    json.dumps(value)  # JSON serialisable
    model.model_validate(value)


def test_bounds_are_respected() -> None:
    for seed in map(str, range(25)):
        v = Everything.model_validate(
            generate_json(Everything.model_json_schema(), seed=seed, user_text="topic words here")
        )
        assert 5 <= v.bounded_i <= 7
        assert 10 < v.gt_i < 13
        assert v.even % 4 == 0 and 3 <= v.even <= 30
        assert 0 < v.frac < 1
        assert 3 <= len(v.short) <= 5
        assert len(v.padded) >= 40
        assert len(v.at_least_three) >= 3
        assert len(v.at_most_one) <= 1


def test_recursive_models_terminate() -> None:
    value = generate_json(Node.model_json_schema(), seed="x")
    assert Node.model_validate(value)
    assert len(json.dumps(value)) < 20_000


def test_generation_is_deterministic_and_seed_sensitive() -> None:
    schema = Everything.model_json_schema()
    a = generate_json(schema, seed="one", user_text="alpha beta gamma")
    b = generate_json(schema, seed="one", user_text="alpha beta gamma")
    c = generate_json(schema, seed="two", user_text="alpha beta gamma")
    assert a == b
    assert a != c


def test_strings_use_topic_and_query_fields_echo_user_text() -> None:
    class Q(BaseModel):
        query: str
        summary: str

    value = generate_json(Q.model_json_schema(), seed="s", user_text="Explain database indexing")
    assert value["query"] == "Explain database indexing"
    assert "database" in value["summary"].lower()


def test_enum_and_literal_values_are_members() -> None:
    for seed in map(str, range(10)):
        v = Everything.model_validate(generate_json(Everything.model_json_schema(), seed=seed))
        assert v.lit in {"x", "y", "z"} and v.single == "only"


def test_optional_fields_are_populated() -> None:
    v = Branch.model_validate(generate_json(Branch.model_json_schema(), seed="s"))
    assert v.best is not None


# ---------------------------------------------------------------------------- chat payloads
USER = [{"role": "user", "content": "Explain caching strategies for web applications"}]


def content_of(resp: dict[str, Any]) -> str:
    return resp["choices"][0]["message"]["content"]


def test_chat_response_shape_and_determinism() -> None:
    a = mock_chat_response("nvidia/nemotron-3-super-120b-a12b", USER)
    b = mock_chat_response("nvidia/nemotron-3-super-120b-a12b", USER)
    assert a == b
    assert a["choices"][0]["finish_reason"] == "stop"
    assert (
        a["usage"]["total_tokens"] == a["usage"]["prompt_tokens"] + a["usage"]["completion_tokens"]
    )
    assert "caching" in content_of(a).lower()


def test_chat_text_depends_on_model_and_message() -> None:
    texts = {
        content_of(
            mock_chat_response(f"model-{i}", [{"role": "user", "content": f"topic number {i}"}])
        )
        for i in range(12)
    }
    assert len(texts) > 3


def test_only_last_user_message_matters_and_content_parts_work() -> None:
    msgs: list[dict[str, Any]] = [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "first question about rockets"},
        {"role": "assistant", "content": "ok"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Explain rockets"},
                {"type": "image_url", "image_url": {"url": "x"}},
            ],
        },
    ]
    assert "rockets" in content_of(mock_chat_response("m", msgs))


def test_reasoning_models_emit_think_block() -> None:
    text = content_of(mock_chat_response("nvidia/nemotron-3-ultra-550b-a55b", USER))
    assert text.startswith("<think>") and "</think>" in text
    assert "<think>" not in content_of(
        mock_chat_response("nvidia/nemotron-3-super-120b-a12b", USER)
    )


def test_max_tokens_truncates_plain_text() -> None:
    resp = mock_chat_response("m", USER, max_tokens=8)
    assert resp["choices"][0]["finish_reason"] == "length"
    assert len(content_of(resp).split()) <= 8


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "days": {"type": "integer", "minimum": 1, "maximum": 5},
                "unit": {"type": "string", "enum": ["c", "f"]},
            },
            "required": ["city", "days"],
        },
    },
}
OTHER_TOOL = {
    "type": "function",
    "function": {
        "name": "other",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
    },
}


def test_tool_call_on_first_turn() -> None:
    resp = mock_chat_response("m", USER, tools=[WEATHER_TOOL, OTHER_TOOL])
    choice = resp["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    call = choice["message"]["tool_calls"][0]
    assert call["type"] == "function" and call["id"]
    assert call["function"]["name"] == "get_weather"
    args = json.loads(call["function"]["arguments"])
    assert isinstance(args["city"], str) and 1 <= args["days"] <= 5 and args["unit"] in {"c", "f"}


def test_plain_text_once_tool_message_present() -> None:
    msgs = [
        *USER,
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "Sunny, 21C"},
    ]
    resp = mock_chat_response("m", msgs, tools=[WEATHER_TOOL])
    assert resp["choices"][0]["finish_reason"] == "stop"
    assert "tool_calls" not in resp["choices"][0]["message"]
    assert "Sunny, 21C" in content_of(resp)


def test_tool_called_again_for_a_new_user_turn() -> None:
    msgs = [
        {"role": "user", "content": "q1"},
        {"role": "tool", "tool_call_id": "c1", "content": "old result"},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "q2"},
    ]
    assert (
        mock_chat_response("m", msgs, tools=[WEATHER_TOOL])["choices"][0]["finish_reason"]
        == "tool_calls"
    )


def test_tool_choice_variants() -> None:
    tools = [WEATHER_TOOL, OTHER_TOOL]
    none = mock_chat_response("m", USER, tools=tools, tool_choice="none")
    assert none["choices"][0]["finish_reason"] == "stop"
    named = mock_chat_response(
        "m", USER, tools=tools, tool_choice={"type": "function", "function": {"name": "other"}}
    )
    assert named["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "other"
    auto = mock_chat_response("m", USER, tools=tools, tool_choice="auto")
    assert auto["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_weather"


class Report(BaseModel):
    headline: str
    items: list[Leaf]


def test_schema_marker_in_system_message_yields_valid_json() -> None:
    marker = f"{JSON_SCHEMA_OPEN}\n{json.dumps(Report.model_json_schema())}\n{JSON_SCHEMA_CLOSE}"
    msgs = [{"role": "system", "content": f"Be helpful.\n\n{marker}"}, *USER]
    text = content_of(mock_chat_response("m", msgs))
    Report.model_validate(extract_json(text))


def test_response_format_json_schema_yields_valid_json() -> None:
    rf = {"type": "json_schema", "json_schema": {"name": "r", "schema": Report.model_json_schema()}}
    Report.model_validate(
        extract_json(content_of(mock_chat_response("m", USER, response_format=rf)))
    )


def test_response_format_json_object_without_schema() -> None:
    resp = mock_chat_response("m", USER, response_format={"type": "json_object"})
    assert "answer" in json.loads(content_of(resp))


def test_reasoning_model_json_still_extractable() -> None:
    marker = f"{JSON_SCHEMA_OPEN}\n{json.dumps(Report.model_json_schema())}\n{JSON_SCHEMA_CLOSE}"
    msgs = [{"role": "system", "content": marker}, *USER]
    text = content_of(mock_chat_response("nvidia/nemotron-3-ultra-550b-a55b", msgs))
    assert text.startswith("<think>")
    Report.model_validate(extract_json(text))


def test_mock_text_without_user_message() -> None:
    assert mock_text("m", [{"role": "system", "content": "x"}])


def test_stream_chunks_are_lossless() -> None:
    text = "Here is  a test.\nWith lines and  spaces. "
    chunks = mock_stream_chunks(text)
    assert "".join(chunks) == text
    assert len(chunks) > 5
    assert mock_stream_chunks("") == []


# ---------------------------------------------------------------------------- embeddings
def test_embeddings_are_unit_vectors_with_dim_1024() -> None:
    vecs = mock_embeddings(["hello world", "", "a b c d e f g"])
    assert EMBED_DIM == 1024
    for v in vecs:
        assert len(v) == 1024
        assert math.isclose(math.sqrt(sum(x * x for x in v)), 1.0, rel_tol=1e-9)


def test_embeddings_deterministic() -> None:
    assert mock_embeddings(["same text"]) == mock_embeddings(["same text"])


def test_shared_words_mean_higher_cosine() -> None:
    q, near, far = mock_embeddings(
        [
            "how do I reset my password",
            "steps to reset a forgotten password",
            "the migration patterns of arctic terns",
        ]
    )
    assert cosine(q, near) > cosine(q, far)
    assert cosine(q, near) > 0.3
    assert cosine(q, far) < 0.1
    assert cosine(q, q) == pytest.approx(1.0)


def test_stemming_and_stopwords() -> None:
    a, b = mock_embeddings(["cats chased mice", "the cat is chasing a mouse"])
    assert cosine(a, b) > 0.3  # cats/cat, chased/chasing share stems


def test_rerank_scores_by_overlap() -> None:
    passages = ["the sky is blue", "python asyncio tutorial for beginners", "asyncio event loop"]
    scores = mock_rerank_scores("python asyncio", passages)
    assert scores[1] > scores[2] > scores[0] == 0.0
    assert mock_rerank_scores("", passages) == [0.0, 0.0, 0.0]
    assert mock_rerank_scores("query", []) == []
