from __future__ import annotations

import asyncio
import time
from enum import Enum
from typing import Annotated, Any, Literal

import pytest
from pydantic import BaseModel, Field

from nimlab.core import (
    AgentEvent,
    ChatResult,
    NIMClient,
    Tool,
    ToolCall,
    run_agent,
    tool,
)
from nimlab.core.tools import parse_docstring

CHAT = "meta/llama-3.3-70b-instruct"
USER = [{"role": "user", "content": "do the thing"}]


class Unit(str, Enum):
    C = "c"
    F = "f"


class Point(BaseModel):
    x: int
    y: int


# --------------------------------------------------------------------------- schema generation
def test_schema_from_signature_and_docstring() -> None:
    @tool
    def get_weather(city: str, days: int = 3, unit: Unit = Unit.C, verbose: bool = False) -> str:
        """Get the weather forecast.

        Longer explanation that must not leak into the description.

        Args:
            city: Name of the city.
            days: How many days ahead
                to forecast.
            unit: Temperature unit.

        Returns:
            A forecast string.
        """
        return "sunny"

    assert isinstance(get_weather, Tool)
    schema = get_weather.schema()
    assert schema["type"] == "function"
    fn = schema["function"]
    assert fn["name"] == "get_weather"
    assert fn["description"] == "Get the weather forecast."
    params = fn["parameters"]
    assert params["type"] == "object"
    assert params["required"] == ["city"]
    props = params["properties"]
    assert props["city"] == {"type": "string", "description": "Name of the city."}
    assert props["days"]["type"] == "integer" and props["days"]["default"] == 3
    assert props["days"]["description"] == "How many days ahead to forecast."
    assert props["unit"]["description"] == "Temperature unit."
    assert props["verbose"]["type"] == "boolean"
    assert "title" not in str(params)  # pydantic title noise removed


def test_property_named_title_is_kept() -> None:
    @tool
    def make_post(title: str, body: str) -> str:
        """Create a post."""
        return title

    assert set(make_post.parameters["properties"]) == {"title", "body"}
    assert make_post.parameters["required"] == ["title", "body"]


def test_complex_types() -> None:
    @tool
    def plan(
        points: list[Point],
        mode: Literal["fast", "slow"],
        tags: list[str] | None = None,
        weights: dict[str, float] | None = None,
        note: Annotated[str, Field(description="Free text", max_length=20)] = "",
    ) -> int:
        """Plan a route."""
        return len(points)

    params = plan.parameters
    assert params["properties"]["mode"]["enum"] == ["fast", "slow"]
    assert "$defs" in params and "Point" in params["$defs"]
    assert params["properties"]["note"]["description"] == "Free text"
    assert params["properties"]["note"]["maxLength"] == 20
    assert params["required"] == ["points", "mode"]


def test_decorator_with_arguments_and_default_description() -> None:
    @tool(name="search_kb", description="Search the knowledge base")
    def search(query: str) -> list[str]:
        return [query]

    @tool
    def no_doc(a: int) -> int:
        return a

    assert search.name == "search_kb"
    assert search.schema()["function"]["description"] == "Search the knowledge base"
    assert no_doc.description == "no_doc"  # falls back to the name


def test_untyped_parameter_and_ignored_varargs() -> None:
    @tool
    def odd(x, *args, **kwargs):  # type: ignore[no-untyped-def]
        """Odd one."""
        return x

    assert list(odd.parameters["properties"]) == ["x"]


def test_param_names_shadowing_basemodel_attrs() -> None:
    @tool
    def f(schema: str, json: int = 1, copy: bool = False) -> str:
        """Shadowing names."""
        return schema

    assert set(f.parameters["properties"]) == {"schema", "json", "copy"}


def test_docstring_styles() -> None:
    summary, params = parse_docstring(
        """Do it.

        :param a: first
        :param int b: second
        """
    )
    assert summary == "Do it." and params == {"a": "first", "b": "second"}
    assert parse_docstring(None) == ("", {})
    assert parse_docstring("Just a summary\nspanning lines.\n\nSecond paragraph.") == (
        "Just a summary spanning lines.",
        {},
    )


def test_tool_stays_callable() -> None:
    @tool
    def add(a: int, b: int) -> int:
        """Add."""
        return a + b

    assert add(2, 3) == 5


# --------------------------------------------------------------------------- invoke
async def test_invoke_sync_and_async() -> None:
    @tool
    def add(a: int, b: int = 10) -> int:
        """Add."""
        return a + b

    @tool
    async def aadd(a: int, b: int = 10) -> int:
        """Add async."""
        await asyncio.sleep(0)
        return a + b

    assert await add.invoke({"a": 1}) == 11
    assert await aadd.invoke({"a": 1, "b": 2}) == 3


async def test_invoke_coerces_and_builds_nested_models() -> None:
    @tool
    def dist(p: Point, unit: Unit) -> str:
        """Distance."""
        return f"{type(p).__name__}:{p.x + p.y}:{unit.value}"

    assert await dist.invoke({"p": {"x": "3", "y": 4}, "unit": "f"}) == "Point:7:f"


async def test_invoke_validation_error() -> None:
    from pydantic import ValidationError

    @tool
    def f(n: int) -> int:
        """F."""
        return n

    with pytest.raises(ValidationError):
        await f.invoke({"n": "not a number"})
    with pytest.raises(ValidationError):
        await f.invoke({})


async def test_sync_tools_do_not_block_the_event_loop() -> None:
    @tool
    def slow() -> str:
        """Blocks a worker thread."""
        time.sleep(0.15)
        return "done"

    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    t = asyncio.create_task(ticker())
    assert await slow.invoke({}) == "done"
    t.cancel()
    assert ticks >= 5


# --------------------------------------------------------------------- run_agent, scripted model
class ScriptedClient:
    """Duck-typed stand-in for NIMClient.chat: returns pre-baked ChatResults."""

    def __init__(self, *results: ChatResult) -> None:
        self.results = list(results)
        self.requests: list[dict[str, Any]] = []

    async def chat(self, model: str, messages: list[dict[str, Any]], **kw: Any) -> ChatResult:
        self.requests.append({"model": model, "messages": [dict(m) for m in messages], **kw})
        return self.results.pop(0)


def call(name: str, id_: str = "c1", **args: Any) -> ToolCall:
    return ToolCall(id=id_, name=name, arguments=args)


def turn(content: str = "", *calls: ToolCall) -> ChatResult:
    return ChatResult(model=CHAT, content=content, tool_calls=list(calls))


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool
def boom() -> str:
    """Always fails."""
    raise RuntimeError("kaput")


async def run(client: ScriptedClient, tools: list[Tool], **kw: Any) -> Any:
    return await run_agent(client, CHAT, USER, tools, **kw)  # type: ignore[arg-type]


async def test_agent_executes_tools_and_stops_on_text() -> None:
    client = ScriptedClient(turn("", call("add", a=2, b=3)), turn("The sum is 5."))
    result = await run(client, [add])
    assert result.final == "The sum is 5." and not result.truncated
    assert [m["role"] for m in result.messages] == ["user", "assistant", "tool", "assistant"]
    tool_msg = result.messages[2]
    assert tool_msg == {"role": "tool", "tool_call_id": "c1", "name": "add", "content": "5"}
    assistant = result.messages[1]
    assert assistant["tool_calls"][0]["function"] == {
        "name": "add",
        "arguments": '{"a": 2, "b": 3}',
    }
    assert len(result.steps) == 2
    assert (
        result.steps[0].tool_results[0].output == "5"
        and not result.steps[0].tool_results[0].is_error
    )
    assert result.tool_calls[0].name == "add"
    # tool schemas were offered on every request; model and history passed through
    assert client.requests[0]["tools"][0]["function"]["name"] == "add"
    assert client.requests[1]["messages"][-1]["role"] == "tool"


async def test_agent_does_not_mutate_input_messages() -> None:
    msgs = [{"role": "user", "content": "hi"}]
    client = ScriptedClient(turn("hello"))
    await run_agent(client, CHAT, msgs, [add])  # type: ignore[arg-type]
    assert msgs == [{"role": "user", "content": "hi"}]


async def test_agent_without_tool_calls_returns_immediately() -> None:
    client = ScriptedClient(turn("just text"))
    result = await run(client, [])
    assert result.final == "just text" and len(result.steps) == 1
    assert client.requests[0]["tools"] is None


async def test_unknown_tool_error_is_reported_to_model() -> None:
    client = ScriptedClient(turn("", call("nope")), turn("sorry"))
    result = await run(client, [add])
    tr = result.steps[0].tool_results[0]
    assert tr.is_error and "unknown tool 'nope'" in tr.output and "add" in tr.output
    assert result.messages[2]["content"] == tr.output


async def test_invalid_arguments_are_reported_not_raised() -> None:
    client = ScriptedClient(turn("", call("add", a="x")), turn("ok"))
    result = await run(client, [add])
    tr = result.steps[0].tool_results[0]
    assert tr.is_error and tr.output.startswith("Error: invalid arguments for add")
    assert "a:" in tr.output and "b:" in tr.output  # both problems listed


async def test_tool_exception_is_reported_not_raised() -> None:
    client = ScriptedClient(turn("", call("boom")), turn("recovered"))
    result = await run(client, [boom])
    tr = result.steps[0].tool_results[0]
    assert tr.is_error and tr.output == "Error: RuntimeError: kaput"
    assert result.final == "recovered"


async def test_max_steps_truncates() -> None:
    client = ScriptedClient(*[turn("thinking", call("add", f"c{i}", a=i, b=1)) for i in range(5)])
    result = await run(client, [add], max_steps=3)
    assert result.truncated and len(result.steps) == 3
    assert result.final == "thinking"
    assert len(client.results) == 2  # exactly 3 model calls were made
    # pending calls of the last step are not executed
    assert result.steps[-1].tool_results == []
    assert result.messages[-1]["role"] == "assistant"


async def test_serialization_of_tool_results() -> None:
    @tool
    def structured() -> dict[str, Any]:
        """Dict."""
        return {"a": [1, 2], "p": Point(x=1, y=2)}

    @tool
    def model() -> Point:
        """Model."""
        return Point(x=1, y=2)

    @tool
    def nothing() -> None:
        """None."""

    client = ScriptedClient(
        turn("", call("structured", "1"), call("model", "2"), call("nothing", "3")), turn("done")
    )
    result = await run(client, [structured, model, nothing])
    outputs = [t.output for t in result.steps[0].tool_results]
    assert outputs == ['{"a": [1, 2], "p": {"x": 1, "y": 2}}', '{"x":1,"y":2}', "null"]


async def test_parallel_tool_calls_run_concurrently_and_keep_order() -> None:
    @tool
    async def nap(label: str, seconds: float) -> str:
        """Sleep."""
        await asyncio.sleep(seconds)
        return label

    client = ScriptedClient(
        turn(
            "",
            call("nap", "a", label="a", seconds=0.2),
            call("nap", "b", label="b", seconds=0.05),
            call("nap", "c", label="c", seconds=0.2),
        ),
        turn("done"),
    )
    start = time.perf_counter()
    result = await run(client, [nap])
    elapsed = time.perf_counter() - start
    assert elapsed < 0.35  # sequential execution would take >= 0.45s
    assert [m["tool_call_id"] for m in result.messages if m["role"] == "tool"] == ["a", "b", "c"]
    assert [t.output for t in result.steps[0].tool_results] == ["a", "b", "c"]


async def test_events_sync_handler() -> None:
    events: list[AgentEvent] = []
    client = ScriptedClient(turn("", call("add", a=1, b=2)), turn("3"))
    await run(client, [add], on_event=events.append)
    assert [e.kind for e in events] == ["model", "tool_call", "tool_result", "model", "final"]
    assert events[1].tool_call is not None and events[1].tool_call.name == "add"
    assert events[2].tool_result is not None and events[2].tool_result.output == "3"
    assert [e.step for e in events] == [0, 0, 0, 1, 1]


async def test_events_async_handler_and_handler_errors_are_contained() -> None:
    seen: list[str] = []

    async def handler(event: AgentEvent) -> None:
        seen.append(event.kind)
        if event.kind == "tool_call":
            raise ValueError("handler bug")

    client = ScriptedClient(turn("", call("add", a=1, b=2)), turn("fine"))
    result = await run(client, [add], on_event=handler)
    assert result.final == "fine"
    assert seen == ["model", "tool_call", "tool_result", "model", "final"]


async def test_chat_kwargs_are_forwarded() -> None:
    client = ScriptedClient(turn("x"))
    await run(client, [add], temperature=0.0, max_tokens=55, tool_choice="auto", extra={"top_p": 1})
    req = client.requests[0]
    assert (req["temperature"], req["max_tokens"], req["tool_choice"], req["extra"]) == (
        0.0,
        55,
        "auto",
        {"top_p": 1},
    )


async def test_agent_against_real_client_mock_backend(mock_client: NIMClient) -> None:
    @tool
    def convert(amount: float, currency: Literal["usd", "eur"]) -> str:
        """Convert money."""
        return f"{amount:.2f} {currency}"

    result = await run_agent(mock_client, CHAT, USER, [convert])
    assert result.steps[0].tool_results[0].name == "convert"
    assert not result.steps[0].tool_results[0].is_error
    assert result.final.startswith("Based on the tool results")
