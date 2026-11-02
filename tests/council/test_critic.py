from __future__ import annotations

import pytest

from nimlab.core.errors import NIMResponseError
from nimlab.council import Council, Critique, Issue, StepType

Q = "Explain how a hash map handles collisions."


def critiques(*rounds: list[tuple[str, str]]):
    """Script the critic: the i-th review returns ``rounds[i]`` as (severity, description) pairs."""
    state = {"n": 0}

    def fn(model, messages, schema, kind):
        assert kind == "critic"
        issues = rounds[min(state["n"], len(rounds) - 1)]
        state["n"] += 1
        return Critique(
            issues=[
                Issue(severity=sev, description=desc, suggestion=f"fix {desc}")
                for sev, desc in issues
            ],
            summary=f"review {state['n']}",
        )

    return fn


def numbered_writer():
    state = {"n": 0}

    def fn(model, messages, kind):
        if kind == "draft":
            return "draft v0"
        state["n"] += 1
        return f"revision v{state['n']}"

    return fn


async def test_stops_when_no_high_severity_issues(make_client, panel) -> None:
    client = make_client(
        chat=numbered_writer(),
        json=critiques(
            [("high", "wrong formula"), ("low", "typo")],
            [("medium", "could be clearer"), ("low", "typo")],  # nothing high -> stop
        ),
    )
    result = await Council(client).run(Q, "critic", models=panel)

    assert result.answer == "revision v1"
    meta = result.trace.meta
    assert meta["stop_reason"] == "no_blocking_issues"
    assert meta["iterations"] == 2 and meta["revisions"] == 1
    assert meta["issues_per_iteration"] == [2, 2]
    assert [c.label for c in result.candidates] == ["Draft", "Revision 1"]
    assert [c.selected for c in result.candidates] == [False, True]
    assert "no high-severity" in (result.rationale or "")


async def test_clean_first_review_means_no_revision(make_client, panel) -> None:
    client = make_client(chat=numbered_writer(), json=critiques([]))
    result = await Council(client).run(Q, "critic", models=panel)
    assert result.answer == "draft v0"
    assert result.trace.meta["revisions"] == 0
    assert not result.trace.steps_of(StepType.REVISE)
    assert result.consensus is None  # a single version has nothing to compare


async def test_terminates_at_max_iterations_when_issues_persist(make_client, panel) -> None:
    client = make_client(chat=numbered_writer(), json=critiques([("high", "still wrong")]))
    result = await Council(client).run(Q, "critic", models=panel, max_iterations=3)

    meta = result.trace.meta
    assert meta["stop_reason"] == "max_iterations"
    assert meta["iterations"] == 3 and meta["revisions"] == 3
    assert len(client.calls_of("json")) == 3  # bounded
    assert len(client.calls_of("chat")) == 1 + 3  # draft + three revisions
    assert result.answer == "revision v3"
    assert meta["unreviewed_blocking"] == 1
    assert result.consensus is not None and result.consensus.size == 4


async def test_zero_iterations_returns_the_draft(make_client, panel) -> None:
    client = make_client(chat=numbered_writer(), json=critiques([("high", "x")]))
    result = await Council(client).run(Q, "critic", models=panel, max_iterations=0)
    assert result.answer == "draft v0"
    assert not client.calls_of("json")
    assert result.trace.meta["stop_reason"] == "no_iterations"


async def test_stop_severity_threshold(make_client, panel) -> None:
    script = [[("medium", "nit")], [("low", "nit")]]
    strict = make_client(chat=numbered_writer(), json=critiques(*script))
    r = await Council(strict).run(Q, "critic", models=panel, critic_stop_severity="medium")
    assert r.trace.meta["revisions"] == 1  # the medium issue forced a revision

    lenient = make_client(chat=numbered_writer(), json=critiques(*script))
    r = await Council(lenient).run(Q, "critic", models=panel, critic_stop_severity="high")
    assert r.trace.meta["revisions"] == 0


async def test_critic_is_a_different_model_by_default(make_client, panel) -> None:
    client = make_client(chat=numbered_writer(), json=critiques([("high", "bad")], []))
    result = await Council(client).run(Q, "critic", models=panel)
    assert [c.model for c in client.calls_of("json")] == [panel[1], panel[1]]
    assert client.calls_of("chat", "draft")[0].model == panel[0]
    assert client.calls_of("chat", "reviser")[0].model == panel[0]
    assert result.trace.meta["critic"] == panel[1]


async def test_roles_can_be_overridden(make_client, panel) -> None:
    client = make_client(chat=numbered_writer(), json=critiques([("high", "bad")], []))
    await Council(client).run(
        Q, "critic", models=panel, drafter=panel[2], critic=panel[0], reviser=panel[1]
    )
    assert client.calls_of("chat", "draft")[0].model == panel[2]
    assert client.calls_of("json")[0].model == panel[0]
    assert client.calls_of("chat", "reviser")[0].model == panel[1]


async def test_reviser_sees_issues_and_current_draft(make_client, panel) -> None:
    client = make_client(
        chat=numbered_writer(), json=critiques([("high", "wrong formula"), ("low", "typo")], [])
    )
    await Council(client).run(Q, "critic", models=panel)
    prompt = client.calls_of("chat", "reviser")[0].user
    assert "draft v0" in prompt
    assert "[HIGH]" in prompt and "wrong formula" in prompt
    assert "[LOW]" in prompt and "typo" in prompt
    assert "fix wrong formula" in prompt  # the critic's suggestion is passed along


async def test_critic_failure_returns_latest_version(make_client, panel) -> None:
    def broken(model, messages, schema, kind):
        return NIMResponseError("invalid JSON after 3 attempts", model=model)

    client = make_client(chat=numbered_writer(), json=broken)
    result = await Council(client).run(Q, "critic", models=panel)
    assert result.answer == "draft v0"
    assert result.trace.meta["stop_reason"] == "critic_failed"
    assert any(s.status == "error" and s.type == StepType.CRITIQUE for s in result.trace.steps)


async def test_reviser_failure_returns_latest_version(make_client, panel) -> None:
    def chat(model, messages, kind):
        if kind == "reviser":
            return NIMResponseError("down", model=model)
        return "draft v0"

    client = make_client(chat=chat, json=critiques([("high", "bad")]))
    result = await Council(client).run(Q, "critic", models=panel)
    assert result.answer == "draft v0"
    assert result.trace.meta["stop_reason"] == "reviser_failed"


async def test_drafter_falls_back_to_other_models(make_client, panel) -> None:
    def chat(model, messages, kind):
        return NIMResponseError("down", model=model) if model == panel[0] else "draft from fallback"

    client = make_client(chat=chat, json=critiques([]))
    result = await Council(client).run(Q, "critic", models=panel)
    assert result.answer == "draft from fallback"
    assert result.candidates[0].model == panel[1]


async def test_single_model_reviews_itself(make_client) -> None:
    client = make_client(chat=numbered_writer(), json=critiques([("high", "bad")], []))
    result = await Council(client).run(Q, "critic", models=["org/solo"])
    assert result.trace.meta["critic"] == "org/solo"
    assert result.trace.meta["revisions"] == 1


def test_issue_severity_normalisation_and_blocking() -> None:
    crit = Critique(
        issues=[
            Issue(severity="Critical", description="a"),  # type: ignore[arg-type]
            Issue(severity=" MAJOR ", description="b"),  # type: ignore[arg-type]
            Issue(severity="moderate", description="c"),  # type: ignore[arg-type]
            Issue(severity="minor", description="d"),  # type: ignore[arg-type]
        ]
    )
    assert [i.severity for i in crit.issues] == ["high", "high", "medium", "low"]
    assert [i.description for i in crit.blocking("high")] == ["a", "b"]
    assert len(crit.blocking("medium")) == 3
    assert len(crit.blocking("low")) == 4
    with pytest.raises(ValueError):
        Issue(severity="catastrophic")  # type: ignore[arg-type]


async def test_critic_in_mock_mode(mock_client) -> None:
    result = await Council(mock_client).run("Explain recursion", "critic", max_iterations=3)
    assert result.answer.strip()
    assert result.trace.meta["iterations"] <= 3
    assert result.trace.meta["stop_reason"] in {"no_blocking_issues", "max_iterations"}
    assert result.trace.steps_of(StepType.DRAFT) and result.trace.steps_of(StepType.CRITIQUE)
