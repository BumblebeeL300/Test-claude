from __future__ import annotations

import asyncio
from typing import ClassVar

import pytest

from nimlab.core.registry import ModelInfo, ModelRegistry, Role
from nimlab.council import (
    CallFinished,
    CallStarted,
    ConsensusComputed,
    Council,
    CouncilConfig,
    CouncilError,
    PhaseStarted,
    RunFinished,
    RunStarted,
    Strategy,
    StrategyOutcome,
    UnknownStrategyError,
    get_strategy,
    list_strategies,
    register_strategy,
)
from nimlab.council.selection import default_panel, diverse_subset, family, resolve
from nimlab.council.strategies import BestOfN, CriticLoop, Debate, MixtureOfAgents
from nimlab.council.strategies.base import unregister_strategy

Q = "Why is the sky blue?"


# ------------------------------------------------------------------------ strategy registry
def test_builtin_strategies_are_registered_in_order() -> None:
    names = [s.name for s in list_strategies()]
    assert names[:4] == ["moa", "debate", "best-of-n", "critic"]
    for s in list_strategies():
        assert s.description and isinstance(s, Strategy)


@pytest.mark.parametrize(
    ("alias", "cls"),
    [
        ("moa", MixtureOfAgents),
        ("MoA", MixtureOfAgents),
        ("mixture_of_agents", MixtureOfAgents),
        ("debate", Debate),
        ("best-of-n", BestOfN),
        ("best_of_n", BestOfN),
        ("BoN", BestOfN),
        ("critic", CriticLoop),
        ("critic loop", CriticLoop),
    ],
)
def test_get_strategy_resolves_names_and_aliases(alias: str, cls: type) -> None:
    assert isinstance(get_strategy(alias), cls)


def test_unknown_strategy_lists_available() -> None:
    with pytest.raises(UnknownStrategyError) as info:
        get_strategy("telepathy")
    assert "moa" in str(info.value) and "telepathy" in str(info.value)
    assert isinstance(info.value, KeyError) and isinstance(info.value, CouncilError)


async def test_custom_strategy_can_be_registered_and_run(make_client, panel) -> None:
    @register_strategy
    class Longest:
        name: ClassVar[str] = "longest"
        description: ClassVar[str] = "Pick the longest answer."

        async def run(self, ctx):
            from nimlab.council import StepType
            from nimlab.council.context import agent_labels

            answers = await ctx.gather(
                *(
                    ctx.ask(
                        m,
                        [{"role": "user", "content": ctx.question}],
                        label=lab,
                        type=StepType.PROPOSE,
                    )
                    for m, lab in zip(ctx.models, agent_labels(len(ctx.models)), strict=True)
                )
            )
            best = max((a for a in answers if a), key=lambda a: len(a.text))
            return StrategyOutcome(answer=best.text, rationale="longest wins")

    try:
        texts = {panel[0]: "short", panel[1]: "the longest answer here", panel[2]: "medium one"}
        client = make_client(chat=lambda m, msgs, k: texts[m])
        result = await Council(client).run(Q, "longest", models=panel)
        assert result.answer == "the longest answer here"
        assert result.strategy == "longest"
        assert any(s.name == "longest" for s in Council.strategies())
        # a strategy instance can also be passed directly
        direct = await Council(client).run(Q, get_strategy("longest"), models=panel)
        assert direct.answer == result.answer
    finally:
        unregister_strategy("longest")
    with pytest.raises(UnknownStrategyError):
        get_strategy("longest")


# ------------------------------------------------------------------------ run() contract
async def test_run_validates_inputs(make_client) -> None:
    council = Council(make_client())
    with pytest.raises(ValueError, match="non-empty"):
        await council.run("   ")
    with pytest.raises(UnknownStrategyError):
        await council.run(Q, "nope")
    with pytest.raises(TypeError, match="unknown council option"):
        await council.run(Q, "moa", roudns=3)
    with pytest.raises(ValueError):
        await council.run(Q, "moa", layers=-1)  # pydantic range validation


async def test_overrides_merge_over_base_config(make_client, panel) -> None:
    base = CouncilConfig(models=panel, layers=0, temperature=0.9)
    council = Council(make_client(), config=base, max_tokens=256)
    result = await council.run(Q, "moa")
    assert result.trace.models == panel
    assert result.trace.config["layers"] == 0 and result.trace.config["max_tokens"] == 256
    result2 = await council.run(Q, "moa", models=panel[:2], layers=1)
    assert result2.trace.models == panel[:2] and result2.trace.config["layers"] == 1
    assert council.config.layers == 0  # per-call overrides never mutate the council


async def test_sampling_parameters_reach_the_client(make_client, panel) -> None:
    client = make_client()
    await Council(client).run(
        Q, "moa", models=panel[:1], layers=0, temperature=0.77, max_tokens=321
    )
    proposer = client.calls_of("chat", "proposer")[0]
    assert proposer.kwargs["temperature"] == 0.77 and proposer.kwargs["max_tokens"] == 321


async def test_reasoning_models_get_a_larger_token_budget(make_client) -> None:
    client = make_client()
    await Council(client).run(
        Q,
        "moa",
        models=["deepseek-ai/deepseek-r1", "meta/llama-3.3-70b-instruct"],
        layers=0,
        max_tokens=500,
    )
    budgets = {c.model: c.kwargs["max_tokens"] for c in client.calls_of("chat", "proposer")}
    assert budgets == {"deepseek-ai/deepseek-r1": 1500, "meta/llama-3.3-70b-instruct": 500}


async def test_result_exposes_trace_candidates_and_usage(make_client, panel, tmp_path) -> None:
    result = await Council(make_client()).run(Q, "moa", models=panel, layers=1)
    assert result.question == Q and result.models == panel
    assert result.agreement is not None and result.agreement == result.consensus.agreement
    assert result.usage and sum(u.calls for u in result.usage) == len(result.trace.steps)
    assert result.trace.wall_time_s > 0 and result.trace.started_at
    json_path = result.save_json(tmp_path / "t.json")
    html_path = result.save_html(tmp_path / "t.html")
    assert json_path.stat().st_size > 0 and html_path.stat().st_size > 0


async def test_failure_attaches_partial_trace(make_client, panel) -> None:
    with pytest.raises(CouncilError) as info:
        await Council(make_client()).run(Q, "debate", models=["only/one"])
    assert info.value.trace is not None
    assert info.value.trace.strategy == "debate" and "at least 2 models" in info.value.trace.error


async def test_run_sync_wrapper(make_client, panel) -> None:
    council = Council(make_client())
    result = await asyncio.to_thread(council.run_sync, Q, "moa", models=panel, layers=0)
    assert result.answer


async def test_council_without_client_creates_one(monkeypatch, panel) -> None:
    monkeypatch.setenv("NIMLAB_MOCK", "1")
    result = await Council().run(Q, "moa", models=panel, layers=0)
    assert result.trace.mock is True and result.answer


# ------------------------------------------------------------------------ observers
async def test_observers_receive_ordered_events(make_client, panel) -> None:
    events: list = []
    client = make_client()
    result = await Council(client).run(Q, "moa", models=panel, layers=1, observers=[events.append])
    assert isinstance(events[0], RunStarted) and events[0].models == tuple(panel)
    assert isinstance(events[-1], RunFinished) and events[-1].trace is result.trace
    starts = [e for e in events if isinstance(e, CallStarted)]
    ends = [e for e in events if isinstance(e, CallFinished)]
    assert len(starts) == len(ends) == len(result.trace.steps)
    assert {e.step_index for e in starts} == {e.step_index for e in ends}
    assert [e.phase for e in events if isinstance(e, PhaseStarted)] == [
        "Proposers",
        "Layer 1",
        "Final aggregation",
    ]
    assert len([e for e in events if isinstance(e, ConsensusComputed)]) == 2
    # every call's CallStarted precedes its CallFinished
    seen: set[int] = set()
    for e in events:
        if isinstance(e, CallStarted):
            seen.add(e.step_index)
        elif isinstance(e, CallFinished):
            assert e.step_index in seen


async def test_a_broken_observer_cannot_break_the_run(make_client, panel) -> None:
    def explode(event) -> None:
        raise RuntimeError("observer bug")

    result = await Council(make_client(), observers=[explode]).run(Q, "moa", models=panel, layers=0)
    assert result.answer


async def test_observer_sees_run_finished_on_failure(make_client) -> None:
    events: list = []
    with pytest.raises(CouncilError):
        await Council(make_client()).run(Q, "debate", models=["x/y"], observers=[events.append])
    assert isinstance(events[-1], RunFinished) and events[-1].error


# ------------------------------------------------------------------------ model selection
def test_default_panel_is_diverse_and_deterministic() -> None:
    panel = default_panel()
    assert len(panel) == 3 and len({family(m) for m in panel}) == 3
    assert panel == default_panel()
    assert len(default_panel(count=5)) == 5


def test_diverse_subset_prefers_new_families_then_fills() -> None:
    pool = ["a/1", "a/2", "b/1", "c/1", "c/2"]
    assert diverse_subset(pool, 3) == ["a/1", "b/1", "c/1"]
    assert diverse_subset(pool, 5) == ["a/1", "b/1", "c/1", "a/2", "c/2"]
    assert diverse_subset(["a/1", "a/1"], 2) == ["a/1"]


def test_default_panel_respects_registry_overrides() -> None:
    def info(id_: str, *roles: Role, tier: int = 3) -> ModelInfo:
        return ModelInfo(id=id_, roles=frozenset(roles), context_window=8192, cost_tier=tier)

    reg = ModelRegistry(
        models=[
            info("x/think", Role.REASONING),
            info("y/chat", Role.CHAT),
            info("y/other", Role.CHAT),
            info("z/chat", Role.CHAT),
        ],
        defaults={Role.REASONING: "x/think", Role.CHAT: "y/chat"},
        env={},
        config_path="/nonexistent.toml",
    )
    assert default_panel(reg) == ["x/think", "y/chat", "z/chat"]


def test_resolve_roles_with_explicit_and_default_panels() -> None:
    explicit = resolve(CouncilConfig(), models=["a/x", "b/y"])
    assert explicit.models == ["a/x", "b/y"]
    assert explicit.aggregator == explicit.judge == "a/x"  # nothing outside the requested set
    assert explicit.critic == "b/y" and explicit.reviser == explicit.drafter == "a/x"
    assert explicit.reward and explicit.embed  # from the registry

    default = resolve(CouncilConfig())
    assert len(default.models) == 3
    assert default.aggregator not in {None, ""} and default.judge not in {None, ""}

    custom = resolve(
        CouncilConfig(aggregator="agg/m", judge="jud/m", reward_model="rw/m", embed_model="em/m")
    )
    assert (custom.aggregator, custom.judge, custom.reward, custom.embed) == (
        "agg/m",
        "jud/m",
        "rw/m",
        "em/m",
    )


def test_config_cleans_model_list() -> None:
    assert CouncilConfig(models=[" a/x ", "", "b/y"]).models == ["a/x", "b/y"]
    assert CouncilConfig(models=[" ", ""]).models is None
    with pytest.raises(ValueError):
        CouncilConfig(bogus=1)  # type: ignore[call-arg]
