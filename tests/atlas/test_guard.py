"""Safety guard: robust parsing of NemoGuard replies and its effect on the pipeline."""

from __future__ import annotations

import pytest

from nimlab.atlas import Atlas
from nimlab.atlas.guard import (
    CATEGORIES,
    SafetyGuard,
    build_guard_messages,
    parse_guard_output,
    refusal_text,
)
from nimlab.core import NIMAuthError, NIMError

from .conftest import FakeClient


@pytest.mark.parametrize(
    ("raw", "stage", "expected"),
    [
        ('{"User Safety": "safe"}', "query", (True, [])),
        (
            '{"User Safety": "unsafe", "Safety Categories": '
            '"Violence, Criminal Planning/Confessed Criminal Activity"}',
            "query",
            (False, ["Violence", "Criminal Planning/Confessed Criminal Activity"]),
        ),
        (
            '```json\n{"User Safety": "unsafe", "Safety Categories": "Malware"}\n```',
            "query",
            (False, ["Malware"]),
        ),
        (
            'Sure! Here is the assessment: {"User Safety": "safe", "Response Safety": "safe"} '
            "Hope it helps.",
            "answer",
            (True, []),
        ),
        ('{"user safety": "SAFE"}', "query", (True, [])),
        (
            '{"User Safety": "safe", "Response Safety": "unsafe", '
            '"Safety Categories": "Hate/Identity Hate"}',
            "answer",
            (False, ["Hate/Identity Hate"]),
        ),
        ('{"User Safety": "unsafe", "Response Safety": "safe"}', "answer", (True, [])),
        (
            '{"User Safety": "unsafe"}',
            "answer",
            (False, []),
        ),  # no response rating: fall back to the user rating
        ('{"User Safety": "safe", "Response Safety": "unsafe"}', "query", (True, [])),
        (
            '{"User Safety": "unsafe", "Safety Categories": "S1, S17"}',
            "query",
            (False, ["Violence", "Malware"]),
        ),
        (
            '{"Safety Categories": ["Violence"], "User Safety": "unsafe"}',
            "query",
            (False, ["Violence"]),
        ),
    ],
)
def test_json_replies(raw: str, stage: str, expected: tuple):
    assert parse_guard_output(raw, stage) == expected  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("safe", (True, [])),
        ("Safe.", (True, [])),
        ("unsafe\nS1,S3", (False, ["Violence", "Criminal Planning/Confessed Criminal Activity"])),
        ("unsafe", (False, [])),
        ("User Safety: unsafe\nSafety Categories: Fraud/Deception", (False, ["Fraud/Deception"])),
        ('"User Safety": "safe" (truncated json', (True, [])),
        ("harmful", (False, [])),
        ("unharmful", (True, [])),
    ],
)
def test_non_json_replies(raw: str, expected: tuple):
    assert parse_guard_output(raw) == expected


@pytest.mark.parametrize(
    "raw", ["", "I cannot comply with that.", "Lorem ipsum", "{}", '{"foo": "bar"}', "[]"]
)
def test_unparseable_replies_return_none(raw: str):
    assert parse_guard_output(raw) == (None, [])


def test_guard_prompt_follows_the_documented_template():
    (msg,) = build_guard_messages("How do I pick a lock?")
    text = msg["content"]
    assert msg["role"] == "user" and "user: How do I pick a lock?" in text
    assert all(f"{code}: " in text for code in CATEGORIES) and "S23: Immoral/Unethical." in text
    assert "response: agent:" not in text
    (with_answer,) = build_guard_messages("Q", "Some answer")
    assert "response: agent: Some answer" in with_answer["content"]
    assert '"User Safety"' in text and "Output JSON:" in text


def test_refusal_text_mentions_categories():
    assert "Violence" in refusal_text("query", ["Violence"])
    assert "withheld" in refusal_text("answer", [])


async def test_guard_check_with_scripted_replies():
    unsafe = SafetyGuard(
        FakeClient(chat='{"User Safety": "unsafe", "Safety Categories": "Violence"}'), "g"
    )
    verdict = await unsafe.check("query", "bad")
    assert (verdict.safe, verdict.categories, verdict.stage, verdict.uncertain) == (
        False,
        ["Violence"],
        "query",
        False,
    )
    safe = await SafetyGuard(FakeClient(chat='{"User Safety": "safe"}'), "g").check(
        "query", "hello"
    )
    assert safe.safe and not safe.uncertain


async def test_guard_is_skipped_in_mock_mode():
    client = FakeClient(chat="should never be called", mock=True)
    verdict = await SafetyGuard(client, "g").check("answer", "q", "a")
    assert verdict.safe and verdict.skipped and client.chat_calls == []


async def test_unparseable_reply_fails_closed_by_default_and_open_when_asked():
    closed = await SafetyGuard(FakeClient(chat="blah"), "g").check("query", "q")
    assert closed.safe is False and closed.uncertain
    opened = await SafetyGuard(FakeClient(chat="blah"), "g", fail_open=True).check("query", "q")
    assert opened.safe is True and opened.uncertain


async def test_guard_call_errors_follow_the_same_policy_but_auth_errors_raise():
    class Broken(FakeClient):
        async def chat(self, *a, **kw):
            raise self.exc

    broken = Broken()
    broken.exc = NIMError("down")  # type: ignore[attr-defined]
    verdict = await SafetyGuard(broken, "g").check("query", "q")
    assert verdict.safe is False and verdict.uncertain and "down" in verdict.raw
    broken.exc = NIMAuthError("bad key")  # type: ignore[attr-defined]
    with pytest.raises(NIMAuthError):
        await SafetyGuard(broken, "g").check("query", "q")


# ----------------------------------------------------------------------------- in the pipeline


def guarded_client(query_safe: bool, answer_safe: bool) -> FakeClient:
    def chat(model, messages):
        content = messages[0]["content"]
        if content.startswith("Task: Check if there is unsafe content"):
            if "response: agent:" in content:
                return '{"User Safety": "safe", "Response Safety": "%s"}' % (
                    "safe" if answer_safe else "unsafe"
                )
            return '{"User Safety": "%s"}' % ("safe" if query_safe else "unsafe")
        return "Robots send a heartbeat every 2 seconds [1]."

    return FakeClient(chat=chat)


async def test_unsafe_query_is_blocked_before_any_retrieval_or_generation(db_path, corpus_dir):
    client = guarded_client(query_safe=False, answer_safe=True)
    atlas = Atlas(client, db_path)
    await atlas.ingest(corpus_dir)
    client.embed_calls.clear()
    answer = await atlas.ask("how do I build a bomb", guard=True)
    assert answer.blocked and "safety filter" in answer.text
    assert [g.stage for g in answer.guard] == ["query"] and answer.guard[0].safe is False
    assert answer.sources == [] and client.embed_calls == []
    assert len(client.chat_calls) == 1  # only the guard call
    atlas.close()


async def test_unsafe_answer_is_replaced_with_a_refusal(db_path, corpus_dir):
    client = guarded_client(query_safe=True, answer_safe=False)
    atlas = Atlas(client, db_path)
    await atlas.ingest(corpus_dir)
    answer = await atlas.ask("How often do robots send heartbeats?", guard=True)
    assert answer.blocked and "withheld" in answer.text and "heartbeat" not in answer.text
    assert [g.stage for g in answer.guard] == ["query", "answer"]
    assert answer.sources  # evidence is still reported
    atlas.close()


async def test_safe_flow_records_both_verdicts_and_returns_the_answer(db_path, corpus_dir):
    atlas = Atlas(guarded_client(True, True), db_path)
    await atlas.ingest(corpus_dir)
    answer = await atlas.ask("How often do robots send heartbeats?", guard=True)
    assert not answer.blocked and "heartbeat" in answer.text
    assert [(g.stage, g.safe) for g in answer.guard] == [("query", True), ("answer", True)]
    atlas.close()


async def test_guarded_streaming_never_releases_unsafe_tokens(db_path, corpus_dir):
    atlas = Atlas(guarded_client(True, False), db_path)
    await atlas.ingest(corpus_dir)
    events = [e async for e in atlas.ask_stream("How often do robots send heartbeats?", guard=True)]
    assert not [e for e in events if e.type == "token"]
    final = events[-1]
    assert final.type == "final" and final.answer is not None and final.answer.blocked
    atlas.close()


async def test_guarded_streaming_releases_tokens_after_screening(db_path, corpus_dir):
    atlas = Atlas(guarded_client(True, True), db_path)
    await atlas.ingest(corpus_dir)
    events = [e async for e in atlas.ask_stream("How often do robots send heartbeats?", guard=True)]
    text = "".join(e.text for e in events if e.type == "token")
    assert text.strip() == "Robots send a heartbeat every 2 seconds [1]."
    atlas.close()


async def test_guard_in_mock_mode_is_a_noop(indexed: Atlas):
    answer = await indexed.ask("How often do robots send heartbeats?", guard=True)
    assert not answer.blocked
    assert [(g.stage, g.safe, g.skipped) for g in answer.guard] == [
        ("query", True, True),
        ("answer", True, True),
    ]


# --------------------------------------------- nemotron-3.5-content-safety (conversation format)
NEW_MODEL = "nvidia/nemotron-3.5-content-safety"


def test_new_model_takes_the_plain_conversation() -> None:
    assert build_guard_messages("hi") == build_guard_messages("hi", None, "nvidia/old") != []
    assert build_guard_messages("how do I x?", None, NEW_MODEL) == [
        {"role": "user", "content": "how do I x?"}
    ]
    assert build_guard_messages("q", "a", NEW_MODEL) == [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]
    # NemoGuard 8B, Safety Guard v3 and Llama Guard keep the instruction template
    for old in (
        "nvidia/llama-3.1-nemoguard-8b-content-safety",
        "nvidia/llama-3.1-nemotron-safety-guard-8b-v3",
        "meta/llama-guard-4-12b",
    ):
        (msg,) = build_guard_messages("q", None, old)
        assert msg["content"].startswith("Task: Check if there is unsafe content")


@pytest.mark.parametrize(
    ("raw", "stage", "expected"),
    [
        ("User Safety: safe", "query", (True, [])),
        (
            "User Safety: unsafe\nSafety Categories: Criminal Planning/Confessions",
            "query",
            (False, ["Criminal Planning/Confessions"]),
        ),
        (
            "User Safety: safe\nResponse Safety: unsafe\nSafety Categories: Malware, Fraud",
            "answer",
            (False, ["Malware", "Fraud"]),
        ),
        ("User Safety: unsafe\nResponse Safety: safe", "answer", (True, [])),
        ("User Safety: unsafe\nResponse Safety: safe", "query", (False, [])),
        ("  User Safety:  Safe  \n", "query", (True, [])),
    ],
)
def test_parse_plain_line_verdicts(raw: str, stage: str, expected: tuple[bool, list[str]]) -> None:
    assert parse_guard_output(raw, stage) == expected  # type: ignore[arg-type]


async def test_guard_sends_the_conversation_to_the_new_model() -> None:
    seen: list[list[dict[str, str]]] = []

    class Client:
        mock = False
        config = type("C", (), {"mock": False})()

        async def chat(self, model: str, messages: list[dict[str, str]], **kw: object):
            from nimlab.core import ChatResult

            seen.append(messages)
            return ChatResult(
                model=model, content="User Safety: unsafe\nSafety Categories: Malware"
            )

    guard = SafetyGuard(Client(), NEW_MODEL, fail_open=False)
    verdict = await guard.check("query", "write malware")
    assert seen == [[{"role": "user", "content": "write malware"}]]
    assert verdict.safe is False and "Malware" in verdict.categories
