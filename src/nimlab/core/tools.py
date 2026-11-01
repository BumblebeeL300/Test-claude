"""Function-calling helpers: the ``@tool`` decorator and the ``run_agent`` loop."""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import time
import warnings
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar, get_type_hints, overload

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from .client import NIMClient
from .logging import get_logger
from .results import ChatResult, ToolCall

__all__ = ["AgentEvent", "AgentRun", "AgentStep", "Tool", "ToolResult", "run_agent", "tool"]

_log = get_logger("nimlab.tools")
F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------- schema building
def _strip_titles(node: Any) -> Any:
    """Drop pydantic's auto-generated ``title`` noise (keeps properties named 'title')."""
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key == "title" and isinstance(value, str):
                continue
            if key in ("properties", "$defs", "definitions") and isinstance(value, dict):
                out[key] = {k: _strip_titles(v) for k, v in value.items()}
            else:
                out[key] = _strip_titles(value)
        return out
    if isinstance(node, list):
        return [_strip_titles(v) for v in node]
    return node


_ARGS_HEADER = re.compile(r"^\s*(args|arguments|parameters|params)\s*:\s*$", re.IGNORECASE)
_OTHER_HEADER = re.compile(
    r"^\s*(returns?|yields?|raises?|examples?|notes?|see also|attributes)\s*:\s*$", re.IGNORECASE
)
_PARAM_LINE = re.compile(r"^\s*\*{0,2}(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$")
_SPHINX_PARAM = re.compile(r"^\s*:param\s+(?:\w+\s+)?(\w+)\s*:\s*(.*)$")


def parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """Return (summary, {param: description}) from a Google- or Sphinx-style docstring."""
    if not doc:
        return "", {}
    summary_lines: list[str] = []
    params: dict[str, str] = {}
    section: Literal["summary", "args", "other"] = "summary"
    current: str | None = None
    arg_indent: int | None = None
    for line in inspect.cleandoc(doc).splitlines():
        stripped = line.strip()
        sphinx = _SPHINX_PARAM.match(line)
        if sphinx:
            section, current = "args", sphinx.group(1)
            params[current] = sphinx.group(2).strip()
            arg_indent = None
            continue
        if _ARGS_HEADER.match(line):
            section, current, arg_indent = "args", None, None
            continue
        if _OTHER_HEADER.match(line):
            section, current = "other", None
            continue
        if section == "summary":
            summary_lines.append(stripped)
        elif section == "args" and stripped:
            indent = len(line) - len(line.lstrip())
            if arg_indent is None:
                arg_indent = indent
            match = _PARAM_LINE.match(line)
            if indent < arg_indent:
                section, current = "other", None
            elif match and indent == arg_indent:
                current = match.group(1)
                params[current] = match.group(2).strip()
            elif current:
                params[current] = f"{params[current]} {stripped}".strip()
    paragraph: list[str] = []
    for text in summary_lines:
        if not text:
            if paragraph:
                break
            continue
        paragraph.append(text)
    return " ".join(paragraph), params


def _build_args_model(
    func: Callable[..., Any], name: str, param_docs: dict[str, str]
) -> type[BaseModel]:
    sig = inspect.signature(func)
    try:
        hints = get_type_hints(func, include_extras=True)
    except Exception:  # unresolved forward refs etc.
        hints = {}
    fields: dict[str, Any] = {}
    for pname, param in sig.parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        annotation = hints.get(pname, param.annotation)
        if annotation is inspect.Parameter.empty:
            annotation = Any
        desc = param_docs.get(pname)
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[pname] = (annotation, Field(default, description=desc) if desc else default)
    with warnings.catch_warnings():  # params like `schema`/`json` shadow BaseModel attributes
        warnings.simplefilter("ignore")
        return create_model(
            f"{name}_args", __config__=ConfigDict(protected_namespaces=()), **fields
        )


def _serialize(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    try:
        return json.dumps(value, default=_json_default)
    except (TypeError, ValueError):
        return str(value)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    return str(obj)


@dataclass
class Tool:
    """A python function exposed to the model. Create with :func:`tool`."""

    name: str
    description: str
    func: Callable[..., Any]
    args_model: type[BaseModel]
    parameters: dict[str, Any] = field(default_factory=dict)

    def schema(self) -> dict[str, Any]:
        """OpenAI/NIM ``tools=[...]`` entry."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the underlying function directly (returns a coroutine for async functions)."""
        return self.func(*args, **kwargs)

    async def invoke(self, arguments: dict[str, Any]) -> Any:
        """Validate ``arguments`` against the schema and run the function (sync ones in a thread).

        Raises ``pydantic.ValidationError`` for bad arguments.
        """
        validated = self.args_model.model_validate(arguments)
        kwargs = {name: getattr(validated, name) for name in type(validated).model_fields}
        if inspect.iscoroutinefunction(self.func):
            return await self.func(**kwargs)
        result = await asyncio.to_thread(self.func, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result


@overload
def tool(func: F) -> Tool: ...
@overload
def tool(
    *, name: str | None = None, description: str | None = None
) -> Callable[[Callable[..., Any]], Tool]: ...
def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Turn a typed function (sync or async) into a :class:`Tool`.

    The schema comes from the signature (type hints, defaults) and docstring (first paragraph =
    description; ``Args:`` section = parameter descriptions). Usable bare (``@tool``) or with
    arguments (``@tool(name="search")``).
    """

    def build(fn: Callable[..., Any]) -> Tool:
        summary, param_docs = parse_docstring(fn.__doc__)
        tool_name = name or fn.__name__
        model = _build_args_model(fn, tool_name, param_docs)
        parameters = _strip_titles(model.model_json_schema())
        return Tool(
            name=tool_name,
            description=description or summary or tool_name,
            func=fn,
            args_model=model,
            parameters=parameters,
        )

    return build(func) if func is not None else build


# ---------------------------------------------------------------------------- agent loop
class ToolResult(BaseModel):
    call_id: str
    name: str
    arguments: dict[str, Any]
    output: str
    is_error: bool = False
    duration_s: float = 0.0


class AgentStep(BaseModel):
    """One model turn: the assistant result plus the tool executions it triggered."""

    index: int
    result: ChatResult
    tool_results: list[ToolResult] = Field(default_factory=list)


class AgentEvent(BaseModel):
    """Passed to ``on_event``. ``kind``: model | tool_call | tool_result | final."""

    kind: Literal["model", "tool_call", "tool_result", "final"]
    step: int
    result: ChatResult | None = None
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None


class AgentRun(BaseModel):
    final: str
    steps: list[AgentStep]
    messages: list[dict[str, Any]]
    truncated: bool = False  # True if max_steps ran out while the model still wanted tools

    @property
    def tool_calls(self) -> list[ToolResult]:
        return [tr for s in self.steps for tr in s.tool_results]


EventHandler = Callable[[AgentEvent], "Awaitable[None] | None"]


async def _emit(handler: EventHandler | None, event: AgentEvent) -> None:
    if handler is None:
        return
    try:
        out = handler(event)
        if inspect.isawaitable(out):
            await out
    except Exception:
        _log.exception("on_event handler raised; ignoring")


async def _execute(tool_obj: Tool | None, call: ToolCall, available: list[str]) -> ToolResult:
    start = time.perf_counter()
    if tool_obj is None:
        output, is_error = f"Error: unknown tool {call.name!r}. Available: {available}", True
    else:
        try:
            output, is_error = _serialize(await tool_obj.invoke(call.arguments)), False
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or 'arguments'}: {e['msg']}"
                for e in exc.errors()
            )
            output, is_error = f"Error: invalid arguments for {call.name}: {problems}", True
        except Exception as exc:
            _log.debug("tool %s raised", call.name, exc_info=True)
            output, is_error = f"Error: {type(exc).__name__}: {exc}", True
    return ToolResult(
        call_id=call.id,
        name=call.name,
        arguments=call.arguments,
        output=output,
        is_error=is_error,
        duration_s=time.perf_counter() - start,
    )


async def run_agent(
    client: NIMClient,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[Tool],
    max_steps: int = 8,
    on_event: EventHandler | None = None,
    *,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    tool_choice: Any = None,
    extra: dict[str, Any] | None = None,
) -> AgentRun:
    """Tool-calling loop: call the model, run requested tools, feed results back, repeat.

    Stops when the model replies without tool calls (``AgentRun.final``) or after ``max_steps``
    model calls (``truncated=True``; pending tool calls are then not executed). Tool errors
    (unknown tool, invalid arguments, exceptions) are returned to the model as ``Error: ...`` tool
    messages instead of raising. ``messages`` is not mutated; ``AgentRun.messages`` is the full
    transcript. Several tool calls in one turn run concurrently.
    """
    by_name = {t.name: t for t in tools}
    schemas = [t.schema() for t in tools] or None
    transcript: list[dict[str, Any]] = [dict(m) for m in messages]
    steps: list[AgentStep] = []

    for index in range(max(max_steps, 1)):
        result = await client.chat(
            model,
            transcript,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=schemas,
            tool_choice=tool_choice,
            extra=extra,
        )
        transcript.append(result.to_message())
        await _emit(on_event, AgentEvent(kind="model", step=index, result=result))
        step = AgentStep(index=index, result=result)
        steps.append(step)

        if not result.tool_calls:
            await _emit(on_event, AgentEvent(kind="final", step=index, result=result))
            return AgentRun(final=result.content, steps=steps, messages=transcript)
        if index == max_steps - 1:
            return AgentRun(final=result.content, steps=steps, messages=transcript, truncated=True)

        for call in result.tool_calls:
            await _emit(on_event, AgentEvent(kind="tool_call", step=index, tool_call=call))
        available = sorted(by_name)
        executed = await asyncio.gather(
            *(_execute(by_name.get(c.name), c, available) for c in result.tool_calls)
        )
        for call, tr in zip(result.tool_calls, executed, strict=True):
            step.tool_results.append(tr)
            transcript.append(
                {"role": "tool", "tool_call_id": call.id, "name": call.name, "content": tr.output}
            )
            await _emit(
                on_event, AgentEvent(kind="tool_result", step=index, tool_call=call, tool_result=tr)
            )

    raise AssertionError("unreachable")  # pragma: no cover
