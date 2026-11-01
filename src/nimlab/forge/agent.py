"""The coder's tool-calling loop.

Deliberately independent of ``nimlab.core.tools.run_agent``: Forge needs per-call events, repeat
detection, argument coercion and tolerance for the offline mock's nonsense tool calls, and it keeps
one long conversation alive across fix rounds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from nimlab.core import NIMAuthError, NIMError

from .events import EventBus
from .tools import WorkspaceTools

__all__ = ["LoopResult", "assistant_message", "compact_messages", "run_tool_loop"]

_REPEAT_HINT = (
    "\nNOTE: you have made this exact call {n} times in a row with the same result. "
    "Change your approach (read the failing code, edit it) or finish with a summary."
)


@dataclass
class LoopResult:
    """Outcome of one coder turn (a tool loop that ends when the model stops calling tools)."""

    final_text: str = ""
    steps: int = 0
    tool_calls: int = 0
    failed_tool_calls: int = 0
    changed_files: list[str] = field(default_factory=list)
    ended_reason: str = "done"  # done | max_steps | llm_error | stuck
    error: str | None = None


def assistant_message(content: str, calls: list[tuple[str, str, dict[str, Any]]]) -> dict[str, Any]:
    """OpenAI-format assistant message; ``calls`` are ``(id, name, arguments)`` triples."""
    msg: dict[str, Any] = {"role": "assistant", "content": content or ""}
    if calls:
        msg["tool_calls"] = [
            {
                "id": cid,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args, default=str)},
            }
            for cid, name, args in calls
        ]
    return msg


def _size(msg: dict[str, Any]) -> int:
    total = len(str(msg.get("content") or ""))
    for tc in msg.get("tool_calls") or []:
        total += len(str(tc.get("function", {}).get("arguments", "")))
    return total


def compact_messages(
    messages: list[dict[str, Any]], *, max_chars: int = 60_000, keep_tail: int = 14
) -> None:
    """Shrink a long conversation in place.

    The system prompt and the first user message (the task) are always kept, as are the last
    ``keep_tail`` messages. Older tool outputs and big ``write_file`` arguments are elided first;
    if that is not enough, whole old assistant/tool turns are dropped.
    """
    if sum(_size(m) for m in messages) <= max_chars or len(messages) <= keep_tail + 2:
        return
    boundary = len(messages) - keep_tail
    for i in range(2, boundary):
        msg = messages[i]
        if msg.get("role") == "tool" and len(str(msg.get("content") or "")) > 240:
            msg["content"] = str(msg["content"])[:200] + "\n... [older tool output elided]"
        elif msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", {})
                if len(str(fn.get("arguments", ""))) > 400:
                    fn["arguments"] = json.dumps({"_elided": "large argument removed from history"})
    # Still too big: drop the oldest complete turns between the head and the kept tail.
    while sum(_size(m) for m in messages) > max_chars and len(messages) - 2 > keep_tail:
        del messages[2]
        while len(messages) - 2 > keep_tail and messages[2].get("role") == "tool":
            del messages[2]


async def run_tool_loop(
    *,
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    tools: WorkspaceTools,
    bus: EventBus,
    phase: str,
    max_steps: int = 14,
    max_tokens: int = 4096,
    temperature: float = 0.2,
) -> LoopResult:
    """Drive ``model`` with ``tools`` until it answers without tool calls.

    ``messages`` is mutated in place (assistant and tool messages are appended). The loop ends
    when the model stops calling tools, after ``max_steps`` rounds, when the LLM errors (non-auth
    errors only; auth errors propagate), or when the model repeats the same call 5 times.
    """
    result = LoopResult()
    recent: list[str] = []
    changed: set[str] = set()
    schemas = tools.schemas()
    tools.phase = phase
    for step in range(1, max_steps + 1):
        compact_messages(messages)
        try:
            res = await client.chat(
                model, messages, tools=schemas, temperature=temperature, max_tokens=max_tokens
            )
        except NIMAuthError:
            raise
        except NIMError as exc:
            result.ended_reason, result.error = "llm_error", str(exc)
            bus.emit("warning", phase, f"LLM call failed: {exc}", model=model)
            break
        result.steps = step
        raw_calls = list(getattr(res, "tool_calls", None) or [])
        calls = [
            (
                getattr(tc, "id", "") or f"call_{step}_{i}",
                str(getattr(tc, "name", "")),
                getattr(tc, "arguments", {}),
            )
            for i, tc in enumerate(raw_calls)
        ]
        usage = getattr(res, "usage", None)
        bus.emit(
            "llm",
            phase,
            f"{model}: {len(calls)} tool call(s)" if calls else f"{model}: answered",
            model=model,
            tokens_in=getattr(usage, "prompt_tokens", 0),
            tokens_out=getattr(usage, "completion_tokens", 0),
            latency_s=getattr(res, "latency_s", 0.0),
            tool_calls=len(calls),
        )
        content = getattr(res, "content", "") or ""
        messages.append(assistant_message(content, calls))
        if not calls:
            result.final_text = content.strip()
            result.ended_reason = "done"
            break
        for cid, name, args in calls:
            outcome = await tools.call(name, args)
            result.tool_calls += 1
            changed.update(outcome.changed)
            if not outcome.ok and name != "run_tests":
                result.failed_tool_calls += 1
            key = f"{name}:{json.dumps(args, sort_keys=True, default=str)}"
            recent.append(key)
            repeats = sum(1 for k in recent[-6:] if k == key)
            text = outcome.output + (_REPEAT_HINT.format(n=repeats) if repeats >= 3 else "")
            messages.append({"role": "tool", "tool_call_id": cid, "name": name, "content": text})
            if repeats >= 5:
                result.ended_reason = "stuck"
        if result.ended_reason == "stuck":
            bus.emit("warning", phase, "coder repeated the same tool call; ending this turn")
            break
    else:
        result.ended_reason = "max_steps"
        bus.emit("warning", phase, f"coder used all {max_steps} tool rounds")
    result.changed_files = sorted(changed)
    return result
