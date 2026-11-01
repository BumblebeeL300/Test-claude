"""The coder's tool loop: id pairing, robustness to nonsense, repeat guard, compaction."""

# ruff: noqa: ASYNC240

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from forge_fakes import FakeClient, calls, say, tc

from nimlab.core import NIMAuthError, NIMRateLimitError, NIMResponseError
from nimlab.forge import EventBus, Sandbox
from nimlab.forge.agent import assistant_message, compact_messages, run_tool_loop
from nimlab.forge.tools import WorkspaceTools


def start_messages() -> list[dict[str, Any]]:
    return [{"role": "system", "content": "sys"}, {"role": "user", "content": "do it"}]


async def drive(client: FakeClient, ws: Path, **kw: Any):
    bus = kw.pop("bus", EventBus())
    tools = WorkspaceTools(Sandbox(ws), bus)
    messages = kw.pop("messages", start_messages())
    result = await run_tool_loop(
        client=client,
        model="coder",
        messages=messages,
        tools=tools,
        bus=bus,
        phase="implement",
        **kw,
    )
    return result, messages, tools, bus


async def test_executes_tools_and_pairs_ids(ws: Path) -> None:
    client = FakeClient(
        chat=[
            calls(
                tc("write_file", path="a.py", content="x = 1\n"),
                tc("write_file", path="b.py", content="y = 2\n"),
            ),
            calls(tc("read_file", path="a.py")),
            say("All done."),
        ]
    )
    result, messages, _, _ = await drive(client, ws)
    assert result.ended_reason == "done" and result.final_text == "All done."
    assert result.steps == 3 and result.tool_calls == 3
    assert result.changed_files == ["a.py", "b.py"]
    assert (ws / "b.py").read_text() == "y = 2\n"
    # Every assistant tool_call id has exactly one matching tool message right after the turn.
    assistant = [m for m in messages if m["role"] == "assistant" and m.get("tool_calls")]
    tool_msgs = [m for m in messages if m["role"] == "tool"]
    assert [c["id"] for m in assistant for c in m["tool_calls"]] == [
        m["tool_call_id"] for m in tool_msgs
    ]
    assert all(m["name"] for m in tool_msgs)
    assert json.loads(assistant[0]["tool_calls"][0]["function"]["arguments"]) == {
        "path": "a.py",
        "content": "x = 1\n",
    }
    # The model is offered the tool schemas on every call.
    assert client.calls[0].tool_names[0] == "list_dir" and len(client.calls[0].tool_names) == 7


async def test_survives_nonsense_tool_calls(ws: Path) -> None:
    """What the offline mock does: tool calls with placeholder / invalid arguments."""
    client = FakeClient(
        chat=[
            calls(
                tc("does_not_exist", foo="bar"),
                tc("write_file", path="a.py"),  # missing content
                tc("write_file", path="../../escape.py", content="boom"),
                tc("run_python", script="Script about tests"),
                tc("apply_patch", path="a.py", patch="no markers"),
                tc("read_file", path=None),
                tc("list_dir", path=["not", "a", "string"]),
            ),
            say(""),
        ]
    )
    result, messages, tools, _ = await drive(client, ws)
    assert result.ended_reason == "done" and result.final_text == ""
    assert result.tool_calls == 7 and result.failed_tool_calls == 7
    outputs = [m["content"] for m in messages if m["role"] == "tool"]
    assert len(outputs) == 7 and all(o.startswith("Error") for o in outputs)
    assert tools.changed_files == set()
    assert list(ws.iterdir()) == []  # nothing was written anywhere in the workspace
    assert not (ws.parent.parent / "escape.py").exists()


async def test_repeat_guard_adds_hint_then_stops(ws: Path) -> None:
    same = lambda _call: calls(tc("list_dir", path="."))  # noqa: E731
    client = FakeClient(chat=same)
    result, messages, _, bus = await drive(client, ws, max_steps=20)
    assert result.ended_reason == "stuck"
    assert result.steps == 5
    tool_outputs = [m["content"] for m in messages if m["role"] == "tool"]
    assert "NOTE: you have made this exact call 3 times" in tool_outputs[2]
    assert "NOTE" not in tool_outputs[1]
    assert any(e.kind == "warning" and "repeated" in e.message for e in bus.history)


async def test_max_steps(ws: Path) -> None:
    counter = iter(range(1000))
    client = FakeClient(chat=lambda _c: calls(tc("list_dir", path=f"d{next(counter)}")))
    result, _, _, bus = await drive(client, ws, max_steps=4)
    assert result.ended_reason == "max_steps" and result.steps == 4
    assert any("all 4 tool rounds" in e.message for e in bus.history)


async def test_llm_error_ends_turn_gracefully(ws: Path) -> None:
    client = FakeClient(chat=[calls(tc("list_dir")), NIMResponseError("HTTP 500")])
    result, messages, _, bus = await drive(client, ws)
    assert result.ended_reason == "llm_error" and "HTTP 500" in (result.error or "")
    assert messages[-1]["role"] == "tool"  # history stays well-formed
    assert any(e.kind == "warning" for e in bus.history)


async def test_rate_limit_error_is_not_fatal(ws: Path) -> None:
    client = FakeClient(chat=[NIMRateLimitError("slow down")])
    result, _, _, _ = await drive(client, ws)
    assert result.ended_reason == "llm_error"


async def test_auth_error_propagates(ws: Path) -> None:
    client = FakeClient(chat=[NIMAuthError("bad key", status_code=401)])
    with pytest.raises(NIMAuthError):
        await drive(client, ws)


async def test_missing_ids_get_synthesised(ws: Path) -> None:
    result = say("")
    result.tool_calls = [tc("list_dir")]
    result.tool_calls[0].id = ""
    client = FakeClient(chat=[result, say("fine")])
    _, messages, _, _ = await drive(client, ws)
    assistant = next(m for m in messages if m.get("tool_calls"))
    tool_msg = next(m for m in messages if m["role"] == "tool")
    assert (
        assistant["tool_calls"][0]["id"]
        and assistant["tool_calls"][0]["id"] == tool_msg["tool_call_id"]
    )


async def test_emits_llm_events_with_usage(ws: Path) -> None:
    client = FakeClient(chat=[calls(tc("list_dir")), say("ok")])
    _, _, _, bus = await drive(client, ws)
    llm = [e for e in bus.history if e.kind == "llm"]
    assert len(llm) == 2 and llm[0].data["tool_calls"] == 1 and llm[0].data["tokens_in"] == 100


async def test_conversation_continues_across_turns(ws: Path) -> None:
    client = FakeClient(chat=[say("first"), say("second")])
    messages = start_messages()
    await drive(client, ws, messages=messages)
    messages.append({"role": "user", "content": "now fix it"})
    result, _, _, _ = await drive(client, ws, messages=messages)
    assert result.final_text == "second"
    assert [m["role"] for m in client.calls[1].messages] == ["system", "user", "assistant", "user"]


class TestMessages:
    def test_assistant_message_shape(self) -> None:
        msg = assistant_message("hi", [("c1", "write_file", {"path": "a"})])
        assert msg["role"] == "assistant" and msg["content"] == "hi"
        assert msg["tool_calls"][0] == {
            "id": "c1",
            "type": "function",
            "function": {"name": "write_file", "arguments": '{"path": "a"}'},
        }
        assert "tool_calls" not in assistant_message(None, [])  # type: ignore[arg-type]
        assert assistant_message(None, [])["content"] == ""  # type: ignore[arg-type]

    def test_compact_noop_when_small(self) -> None:
        msgs = [*start_messages(), {"role": "user", "content": "x"}]
        before = json.dumps(msgs)
        compact_messages(msgs)
        assert json.dumps(msgs) == before

    def _long_history(self, turns: int, size: int) -> list[dict[str, Any]]:
        msgs = start_messages()
        for i in range(turns):
            msgs.append(
                assistant_message(
                    "", [(f"c{i}", "write_file", {"path": f"f{i}.py", "content": "x" * size})]
                )
            )
            msgs.append(
                {
                    "role": "tool",
                    "tool_call_id": f"c{i}",
                    "name": "write_file",
                    "content": "y" * size,
                }
            )
        return msgs

    def test_compact_elides_old_outputs_and_big_arguments(self) -> None:
        msgs = self._long_history(30, 4000)
        original_tail = json.dumps(msgs[-14:])
        compact_messages(msgs, max_chars=60_000)
        assert json.dumps(msgs[-14:]) == original_tail  # recent turns untouched
        assert msgs[0]["content"] == "sys" and msgs[1]["content"] == "do it"
        old_tool = next(m for m in msgs[2:20] if m["role"] == "tool")
        assert "elided" in old_tool["content"] and len(old_tool["content"]) < 300
        old_call = next(m for m in msgs[2:20] if m["role"] == "assistant")
        json.loads(old_call["tool_calls"][0]["function"]["arguments"])  # still valid JSON
        assert sum(len(str(m.get("content") or "")) for m in msgs) < 60_000

    def test_compact_drops_oldest_turns_without_orphaning_tool_messages(self) -> None:
        msgs = self._long_history(60, 3000)
        compact_messages(msgs, max_chars=20_000, keep_tail=6)
        assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user"
        ids = set()
        for m in msgs:
            if m["role"] == "assistant":
                ids |= {c["id"] for c in m.get("tool_calls", [])}
            if m["role"] == "tool":
                assert m["tool_call_id"] in ids, "tool message lost its assistant tool_call"
        assert len(msgs) < 62 * 2
