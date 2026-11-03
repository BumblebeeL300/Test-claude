"""The public Arena API end to end (simulator): runs, resume, leaderboard, report and routing."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from nimlab.arena import Arena, SmartRouter
from nimlab.arena.results import RunResult, RunStore
from nimlab.arena.suites import load_suites

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "arena" / "custom_suite.json"
MODELS = [
    "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/nemotron-3-ultra-550b-a55b",
    "poolside/laguna-xs-2.1",
    "nvidia/nemotron-3.5-lightning-30b-a3b",
]


def install_example_suite(directory: Path) -> None:
    directory.mkdir(exist_ok=True)
    (directory / "triage.json").write_text(EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")


@pytest.fixture
async def full_run(runs_dir: Path) -> RunResult:
    async with Arena(mock=True, runs_dir=runs_dir, concurrency=16) as arena:
        return await arena.run(MODELS, "all")


def test_package_exports() -> None:
    import nimlab.arena as pkg

    assert pkg.Arena is Arena and pkg.SmartRouter is SmartRouter
    assert callable(pkg.create_app)
    with pytest.raises(AttributeError):
        pkg.does_not_exist  # noqa: B018


async def test_full_benchmark_over_all_suites(full_run: RunResult, runs_dir: Path) -> None:
    total = sum(len(s.items) for s in load_suites().values())
    assert len(full_run.items) == total * len(MODELS)
    assert full_run.status == "complete" and full_run.mock
    assert set(full_run.suites) == set(load_suites())
    assert all(r.scored and r.latency_s > 0 and r.ttft_s is not None for r in full_run.items)
    assert RunStore(runs_dir).list_ids() == [full_run.run_id]


async def test_leaderboard_html_and_router_from_a_run(
    full_run: RunResult, runs_dir: Path, tmp_path: Path
) -> None:
    async with Arena(mock=True, runs_dir=runs_dir) as arena:
        lb = arena.leaderboard(full_run, n_boot=40)
        assert len(lb.standings) == 4 and lb.elo_source == "scores"
        assert lb.standings[0].overall >= lb.standings[-1].overall  # type: ignore[operator]
        by = {s.model: s for s in lb.standings}
        assert by[MODELS[0]].overall > by[MODELS[3]].overall  # 405B beats 8B
        assert by[MODELS[3]].latency_p50_s < by[MODELS[0]].latency_p50_s  # but the 8B is faster

        out = tmp_path / "board.html"
        page = arena.report(full_run, html_out=out, title="Mine")
        assert out.read_text(encoding="utf-8") == page and "<title>Mine</title>" in page
        assert arena.report("latest") and arena.load_run().run_id == full_run.run_id

        router = arena.router()
        decisions = {
            "coding": await router.route("Write a Python function that merges intervals"),
            "math": await router.route("Solve 4x + 9 = 33"),
        }
        for d in decisions.values():
            assert d.source == "leaderboard" and d.model in MODELS
            assert d.model == d.candidates[0].model
        assert (
            decisions["coding"].category == "coding"
            and decisions["math"].category == "math_reasoning"
        )


async def test_router_prefers_specialists_on_a_quality_first_profile(
    full_run: RunResult, runs_dir: Path
) -> None:
    async with Arena(mock=True, runs_dir=runs_dir) as arena:
        lb = arena.leaderboard(full_run, n_boot=0)
        router = arena.router(profile="quality")
        for category in ("coding", "math_reasoning", "factual_qa"):
            d = await router.route("x", category=category)
            best = max(lb.standings, key=lambda s: lb.category_quality(s.model, category) or 0.0)
            chosen_q = lb.category_quality(d.model, category)
            assert (
                chosen_q is not None
                and chosen_q >= (lb.category_quality(best.model, category) or 0) - 0.12
            )


async def test_resume_and_limit(runs_dir: Path) -> None:
    async with Arena(mock=True, runs_dir=runs_dir) as arena:
        first = await arena.run(MODELS[:2], "coding,factual_qa", limit=3)
        assert len(first.items) == 2 * 2 * 3
        again = await arena.run(MODELS[:2], "coding,factual_qa", limit=3, resume=first.run_id)
        assert again.run_id == first.run_id and len(again.items) == len(first.items)
        more = await arena.run(MODELS[:3], "coding,factual_qa", limit=3, resume=first.run_id)
        assert len(more.items) == 3 * 2 * 3 and len(RunStore(runs_dir).list_ids()) == 1


async def test_run_accepts_suite_objects_and_custom_suite_dirs(tmp_path: Path) -> None:
    suites_dir = tmp_path / "suites"
    install_example_suite(suites_dir)
    async with Arena(mock=True, runs_dir=tmp_path / "r", suites_dirs=[suites_dir]) as arena:
        assert "support_triage" in arena.suites() and "coding" in arena.suites()
        run = await arena.run(MODELS[:2], "support_triage")
        assert {r.suite for r in run.items} == {"support_triage"} and len(run.items) == 10
        run2 = await arena.run(MODELS[:1], [arena.suites()["coding"]], limit=2)
        assert len(run2.items) == 2


async def test_persist_false_keeps_everything_in_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    async with Arena(mock=True, persist=False) as arena:
        run = await arena.run(MODELS[:2], "factual_qa", limit=2)
        assert len(run.items) == 4 and arena.store is None
        assert arena.leaderboard(run, n_boot=0).standings
    assert not (tmp_path / ".nimlab").exists()


async def test_runs_go_to_the_default_location_under_the_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    async with Arena(mock=True) as arena:
        run = await arena.run(MODELS[:2], "factual_qa", limit=2)
    path = tmp_path / ".nimlab" / "arena" / "runs" / f"{run.run_id}.json"
    assert path.is_file() and json.loads(path.read_text())["run_id"] == run.run_id
    monkeypatch.setenv("NIMLAB_HOME", str(tmp_path / "elsewhere"))
    async with Arena(mock=True) as arena:
        other = await arena.run(MODELS[:1], "factual_qa", limit=1)
    assert (tmp_path / "elsewhere" / "arena" / "runs" / f"{other.run_id}.json").is_file()


async def test_arena_uses_a_supplied_client_and_does_not_close_it() -> None:
    from arena_fakes import ScriptedClient, make_suite

    closed = []

    class Closing(ScriptedClient):
        async def aclose(self) -> None:
            closed.append(True)

    suite = make_suite("one", [("q", "Q?", 3)])
    client = Closing(lambda m, p: "Answer: 3")
    async with Arena(client, persist=False) as arena:
        run = await arena.run(["m"], [suite])
        assert run.items[0].score == 1.0 and not arena.mock
    assert closed == []  # not ours to close


async def test_missing_run_reference_raises(runs_dir: Path) -> None:
    async with Arena(mock=True, runs_dir=runs_dir) as arena:
        with pytest.raises(FileNotFoundError):
            arena.load_run("latest")
        with pytest.raises(FileNotFoundError):
            arena.leaderboard("20250101T000000Z")


def test_run_store_edge_cases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frozen = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    monkeypatch.setattr("nimlab.arena.results.utcnow", lambda: frozen)
    store = RunStore(tmp_path / "runs")
    assert store.list_ids() == [] and store.latest() is None and store.merged() is None
    with pytest.raises(ValueError, match="invalid run id"):
        store.path_for("../evil")
    first = store.new_run()
    assert first.run_id == "20260102T030405Z"
    store.save(first)
    second = store.new_run()
    assert second.run_id == "20260102T030405Z-2"  # same-second runs never overwrite each other
    (store.root / "garbage.json").write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not a valid Arena run"):
        store.load("garbage")
