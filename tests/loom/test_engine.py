from __future__ import annotations

import asyncio
from typing import Any

import pytest

from nimlab.loom import (
    CycleError,
    InputError,
    NodeTimeout,
    PipelineError,
    register_function,
    run_pipeline,
)
from nimlab.loom import engine as engine_module

from .conftest import FakeClient, inp, make, out, prompt


def kinds(events: list[dict[str, Any]], kind: str) -> list[str]:
    return [e["node"] for e in events if e["type"] == kind]


# ------------------------------------------------------------------ basics & events
async def test_linear_run_and_event_order(fake: FakeClient) -> None:
    pipe = make(
        [inp(default="cats"), prompt("a", "Tell me about {{q}}"), out("res", "result")],
        [("q.value", "a.q"), ("a.text", "res.value")],
    )
    events: list[dict[str, Any]] = []
    result = await run_pipeline(pipe, client=fake, on_event=events.append)
    assert result.ok and result.outputs == {"result": "reply(Tell me about cats)"}
    assert events[0]["type"] == "run.plan" and events[0]["nodes"] == ["q", "a", "res"]
    assert events[-1]["type"] == "run.done" and events[-1]["ok"] is True
    for node in ("q", "a", "res"):
        types = [e["type"] for e in events if e.get("node") == node]
        assert types[0] == "node.start" and types[-1] == "node.end"
    end = next(e for e in events if e["type"] == "node.end" and e["node"] == "a")
    assert end["preview"] == "reply(Tell me about cats)"
    assert end["emitted"] == ["text"] and end["latency_ms"] >= 0 and end["cached"] is False
    assert end["tokens"] > 0
    assert all("t" in e for e in events)


async def test_tokens_stream_when_listening(fake: FakeClient) -> None:
    pipe = make([inp(), prompt("a", "{{q}}"), out()], [("q.value", "a.q"), ("a.text", "out.value")])
    events: list[dict[str, Any]] = []
    result = await run_pipeline(pipe, client=fake, on_event=events.append)
    streamed = "".join(e["text"] for e in events if e["type"] == "node.token")
    assert streamed.strip() == result.outputs["out"]
    assert ("stream", "hello") in fake.calls
    # without a listener the plain chat endpoint is used
    fake.calls.clear()
    await run_pipeline(pipe, client=fake)
    assert fake.calls == [("chat", "hello")]


async def test_inputs_are_coerced_and_validated(fake: FakeClient) -> None:
    pipe = make(
        [
            {"id": "n", "type": "input", "params": {"type": "number"}},
            {"id": "flag", "type": "input", "params": {"type": "boolean", "default": False}},
            {"id": "t", "type": "template", "params": {"template": "{{n}}-{{flag}}"}},
            out(),
        ],
        [("n.value", "t.n"), ("flag.value", "t.flag"), ("t.text", "out.value")],
    )
    result = await run_pipeline(pipe, {"n": "42", "flag": "true"}, client=fake)
    assert result.outputs["out"] == "42-true"
    with pytest.raises(InputError, match="missing required input 'n'"):
        await run_pipeline(pipe, {}, client=fake)
    with pytest.raises(InputError, match="unknown input"):
        await run_pipeline(pipe, {"n": 1, "zzz": 2}, client=fake)
    with pytest.raises(InputError, match="input 'n'"):
        await run_pipeline(pipe, {"n": "abc"}, client=fake)


async def test_cycle_is_rejected_before_any_call(fake: FakeClient) -> None:
    pipe = make(
        [prompt("a", "{{x}}"), prompt("b", "{{x}}"), out()],
        [("a.text", "b.x"), ("b.text", "a.x"), ("b.text", "out.value")],
    )
    with pytest.raises(CycleError, match="a -> b -> a"):
        await run_pipeline(pipe, client=fake)
    assert fake.calls == []


async def test_lint_errors_block_a_run(fake: FakeClient) -> None:
    pipe = make([prompt("a", "{{missing}}"), out()], [("a.text", "out.value")])
    with pytest.raises(PipelineError, match="unknown variable"):
        await run_pipeline(pipe, client=fake)
    assert fake.calls == []


# ------------------------------------------------------------------ parallelism
async def test_independent_nodes_run_in_parallel() -> None:
    fake = FakeClient(delay=0.1)
    nodes: list[dict[str, Any]] = [inp()]
    edges: list[tuple[str, str]] = []
    for i in range(5):
        nodes.append(prompt(f"p{i}", f"branch {i}: {{{{q}}}}"))
        edges += [("q.value", f"p{i}.q"), (f"p{i}.text", "j.in")]
    nodes += [{"id": "j", "type": "join", "params": {"separator": " | "}}, out()]
    edges.append(("j.text", "out.value"))
    result = await run_pipeline(make(nodes, edges), client=fake)
    assert result.ok
    assert fake.max_active == 5  # all five at once
    assert result.elapsed_s < 0.35  # not 5 x 0.1 s


async def test_dependent_nodes_wait_for_each_other() -> None:
    fake = FakeClient(delay=0.05)
    pipe = make(
        [inp(), prompt("a", "{{q}}"), prompt("b", "second {{a}}"), out()],
        [("q.value", "a.q"), ("a.text", "b.a"), ("b.text", "out.value")],
    )
    await run_pipeline(pipe, client=fake)
    assert fake.max_active == 1
    first_end = next(t for label, t in fake.ends if label == "hello")
    second_start = next(t for label, t in fake.starts if label.startswith("second"))
    assert second_start >= first_end


async def test_diamond_runs_middle_in_parallel_then_joins() -> None:
    fake = FakeClient(delay=0.05)
    pipe = make(
        [
            inp(),
            prompt("l", "left {{q}}"),
            prompt("r", "right {{q}}"),
            prompt("m", "mix {{l}} {{r}}"),
            out(),
        ],
        [
            ("q.value", "l.q"),
            ("q.value", "r.q"),
            ("l.text", "m.l"),
            ("r.text", "m.r"),
            ("m.text", "out.value"),
        ],
    )
    result = await run_pipeline(pipe, client=fake)
    assert result.ok and fake.max_active == 2


async def test_max_parallel_bounds_node_concurrency() -> None:
    fake = FakeClient(delay=0.03)
    nodes: list[dict[str, Any]] = [inp()]
    edges = []
    for i in range(6):
        nodes.append(prompt(f"p{i}", f"x{i} {{{{q}}}}"))
        edges += [("q.value", f"p{i}.q"), (f"p{i}.text", "j.in")]
    nodes += [{"id": "j", "type": "join"}, out()]
    edges.append(("j.text", "out.value"))
    await run_pipeline(make(nodes, edges), client=fake, max_parallel=2)
    assert fake.max_active == 2


# ------------------------------------------------------------------ branch, map, merge
def branchy(condition: str) -> Any:
    return make(
        [
            {"id": "n", "type": "input", "params": {"type": "number", "default": 5}},
            {"id": "b", "type": "branch", "params": {"condition": condition}},
            {"id": "big", "type": "template", "params": {"template": "big:{{v}}"}},
            {"id": "small", "type": "template", "params": {"template": "small:{{v}}"}},
            {"id": "pick", "type": "merge"},
            {"id": "after_big", "type": "template", "params": {"template": "[{{x}}]"}},
            out("res", "result"),
            out("big_only", "big_only"),
        ],
        [
            ("n.value", "b.value"),
            ("b.then", "big.v"),
            ("b.else", "small.v"),
            ("big.text", "pick.in"),
            ("small.text", "pick.in"),
            ("pick.out", "res.value"),
            ("big.text", "after_big.x"),
            ("after_big.text", "big_only.value"),
        ],
    )


@pytest.mark.parametrize(
    "n, taken, skipped",
    [(5, "big:5", ["small"]), (1, "small:1", ["big", "after_big", "big_only"])],
)
async def test_branch_takes_one_side_and_skips_the_other(
    fake: FakeClient, n, taken, skipped
) -> None:
    events: list[dict[str, Any]] = []
    result = await run_pipeline(branchy("value > 3"), {"n": n}, client=fake, on_event=events.append)
    assert result.ok and result.outputs["result"] == taken
    assert sorted(kinds(events, "node.skipped")) == sorted(skipped)  # skip propagates downstream
    for node in skipped:
        assert result.nodes[node].status == "skipped" and result.nodes[node].reason
        assert node not in kinds(events, "node.start")
    assert ("big_only" in result.outputs) == (n == 5)  # an output behind a dead arm is absent


async def test_branch_condition_error_fails_the_node(fake: FakeClient) -> None:
    result = await run_pipeline(branchy("value['x']"), {"n": 1}, client=fake, validate=False)
    assert (
        not result.ok and result.failed_node == "b" and "condition failed" in (result.error or "")
    )


async def test_branch_condition_rejects_attribute_access(fake: FakeClient) -> None:
    with pytest.raises(PipelineError, match="attribute access"):
        await run_pipeline(branchy("value.real"), client=fake)


async def test_join_collects_only_live_inputs(fake: FakeClient) -> None:
    pipe = make(
        [
            inp("n", 1, "number"),
            {"id": "b", "type": "branch", "params": {"condition": "value > 3"}},
            {"id": "x", "type": "template", "params": {"template": "X{{v}}"}},
            {"id": "y", "type": "template", "params": {"template": "Y{{v}}"}},
            {"id": "z", "type": "template", "params": {"template": "Z"}},
            {"id": "j", "type": "join", "params": {"separator": "+"}},
            out(),
        ],
        [
            ("n.value", "b.value"),
            ("b.then", "x.v"),
            ("b.else", "y.v"),
            ("x.text", "j.in"),
            ("y.text", "j.in"),
            ("z.text", "j.in"),
            ("j.text", "out.value"),
        ],
    )
    result = await run_pipeline(pipe, client=fake)
    assert result.outputs["out"] == "Y1+Z"


async def test_merge_with_every_arm_dead_is_skipped(fake: FakeClient) -> None:
    pipe = make(
        [
            inp("n", 1, "number"),
            {"id": "b", "type": "branch", "params": {"condition": "value > 3"}},
            {"id": "x", "type": "template", "params": {"template": "X{{v}}"}},
            {"id": "m", "type": "merge"},
            out(),
        ],
        [("n.value", "b.value"), ("b.then", "x.v"), ("x.text", "m.in"), ("m.out", "out.value")],
    )
    result = await run_pipeline(pipe, client=fake)
    assert result.ok and result.outputs == {} and result.nodes["m"].status == "skipped"


def map_pipeline(items: list[str], **map_params: Any) -> Any:
    return make(
        [
            {"id": "items", "type": "input", "params": {"type": "list", "default": items}},
            inp("topic", "birds"),
            {
                "id": "m",
                "type": "map",
                "params": {
                    "step": {
                        "type": "prompt",
                        "params": {"template": "{{topic}}: {{item}} #{{index}}"},
                    },
                    **map_params,
                },
            },
            out(),
        ],
        [("items.value", "m.items"), ("topic.value", "m.topic"), ("m.results", "out.value")],
    )


async def test_map_fans_out_in_order_with_context() -> None:
    fake = FakeClient(delay=0.02)
    result = await run_pipeline(map_pipeline(["a", "b", "c", "d"]), client=fake)
    assert result.outputs["out"] == [f"reply(birds: {c} #{i})" for i, c in enumerate("abcd")]
    assert result.nodes["m"].tokens > 0


@pytest.mark.parametrize("limit", [1, 2, 3])
async def test_map_concurrency_is_bounded(limit: int) -> None:
    fake = FakeClient(delay=0.03)
    await run_pipeline(map_pipeline([str(i) for i in range(7)], concurrency=limit), client=fake)
    assert fake.max_active == limit


async def test_map_reports_progress(fake: FakeClient) -> None:
    events: list[dict[str, Any]] = []
    await run_pipeline(map_pipeline(["a", "b", "c"]), client=fake, on_event=events.append)
    progress = [e for e in events if e["type"] == "node.progress"]
    assert [p["done"] for p in progress] == [0, 1, 2, 3] and progress[-1]["total"] == 3


async def test_map_item_errors_follow_policy(fake: FakeClient) -> None:
    def reply(text: str) -> str:
        if "#1" in text:
            raise RuntimeError("bad item")
        return text

    fake.reply = reply
    failed = await run_pipeline(map_pipeline(["a", "b", "c"]), client=fake)
    assert not failed.ok and "item 1" in (failed.error or "")
    skipped = await run_pipeline(map_pipeline(["a", "b", "c"], on_error="skip"), client=fake)
    assert skipped.outputs["out"] == ["birds: a #0", "birds: c #2"]
    nulls = await run_pipeline(map_pipeline(["a", "b", "c"], on_error="null"), client=fake)
    assert nulls.outputs["out"] == ["birds: a #0", None, "birds: c #2"]


async def test_map_limits_item_count(fake: FakeClient) -> None:
    result = await run_pipeline(map_pipeline(["a", "b", "c"], max_items=2), client=fake)
    assert not result.ok and "limit is 2" in (result.error or "")


async def test_map_with_a_json_step() -> None:
    fake = FakeClient()
    fake.json_reply = {"tags": ["x", "y"]}
    schema = {
        "type": "object",
        "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
        "required": ["tags"],
    }
    pipe = make(
        [
            {"id": "items", "type": "input", "params": {"type": "list", "default": ["p", "q"]}},
            {
                "id": "m",
                "type": "map",
                "params": {
                    "step": {"type": "json", "params": {"template": "{{item}}", "schema": schema}}
                },
            },
            {
                "id": "flat",
                "type": "python_fn",
                "params": {"fn": "flatten_field", "args": {"field": "tags"}},
            },
            out(),
        ],
        [("items.value", "m.items"), ("m.results", "flat.value"), ("flat.result", "out.value")],
    )
    result = await run_pipeline(pipe, client=fake)
    assert result.outputs["out"] == ["x", "y", "x", "y"]


# ------------------------------------------------------------------ other node types
async def test_text_nodes(fake: FakeClient) -> None:
    pipe = make(
        [
            inp("t", "Order #1234 and #5678 shipped"),
            {
                "id": "ids",
                "type": "regex",
                "params": {"pattern": "#(\\d+)", "mode": "findall", "group": 1},
            },
            {"id": "first", "type": "regex", "params": {"pattern": "#(\\d+)", "group": 1}},
            {"id": "cnt", "type": "python_fn", "params": {"fn": "count"}},
            {
                "id": "fmt",
                "type": "template",
                "params": {"template": "{{ids | join}} / {{first}} / {{n}}"},
            },
            out(),
        ],
        [
            ("t.value", "ids.text"),
            ("t.value", "first.text"),
            ("ids.matches", "cnt.value"),
            ("ids.matches", "fmt.ids"),
            ("first.match", "fmt.first"),
            ("cnt.result", "fmt.n"),
            ("fmt.text", "out.value"),
        ],
    )
    result = await run_pipeline(pipe, client=fake)
    assert result.outputs["out"] == "1234, 5678 / 1234 / 2"


async def test_chunk_embed_dedupe_and_rerank(fake: FakeClient) -> None:
    pipe = make(
        [
            {
                "id": "texts",
                "type": "input",
                "params": {"type": "list", "default": ["apple pie", "apple tart", "banana split"]},
            },
            {"id": "e", "type": "embed", "params": {"dedupe_threshold": 0.9}},
            {"id": "q", "type": "input", "params": {"type": "string", "default": "banana"}},
            {"id": "r", "type": "rerank", "params": {"top_n": 1}},
            out("uniq"),
            out("best"),
            out("n"),
        ],
        [
            ("texts.value", "e.texts"),
            ("e.unique", "uniq.value"),
            ("e.count", "n.value"),
            ("q.value", "r.query"),
            ("texts.value", "r.passages"),
            ("r.ranked", "best.value"),
        ],
    )
    result = await run_pipeline(pipe, client=fake)
    assert result.outputs["uniq"] == ["apple pie", "banana split"] and result.outputs["n"] == 2
    assert result.outputs["best"] == ["banana split"]


async def test_chunk_node(fake: FakeClient) -> None:
    text = "\n\n".join(f"Paragraph number {i} has a few words in it." for i in range(10))
    pipe = make(
        [
            inp("doc", text),
            {"id": "c", "type": "chunk", "params": {"max_chars": 120, "overlap": 10}},
            out("chunks"),
            out("count"),
        ],
        [("doc.value", "c.text"), ("c.chunks", "chunks.value"), ("c.count", "count.value")],
    )
    result = await run_pipeline(pipe, client=fake)
    assert len(result.outputs["chunks"]) == result.outputs["count"] > 1
    assert all(len(c) <= 130 for c in result.outputs["chunks"])


async def test_python_fn_only_runs_registered_functions(fake: FakeClient) -> None:
    register_function(
        "shout_test", lambda value, **kw: f"{value}{kw.get('mark', '!')}", replace=True
    )
    ok = make(
        [
            inp(),
            {"id": "f", "type": "python_fn", "params": {"fn": "shout_test", "args": {"mark": "?"}}},
            out(),
        ],
        [("q.value", "f.value"), ("f.result", "out.value")],
    )
    assert (await run_pipeline(ok, client=fake)).outputs["out"] == "hello?"
    evil = make(
        [
            inp(),
            {"id": "f", "type": "python_fn", "params": {"fn": "__import__('os').system"}},
            out(),
        ],
        [("q.value", "f.value"), ("f.result", "out.value")],
    )
    with pytest.raises(PipelineError, match="not registered"):
        await run_pipeline(evil, client=fake)


async def test_http_get_is_disabled_by_default_and_allowlisted(fake: FakeClient) -> None:
    import httpx

    pipe = make(
        [
            inp("u", "https://example.com/data"),
            {"id": "g", "type": "http_get", "params": {"url": "{{u}}"}},
            out("body"),
        ],
        [("u.value", "g.u"), ("g.body", "body.value")],
    )
    off = await run_pipeline(pipe, client=fake)
    assert not off.ok and "disabled" in (off.error or "")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=f"hello from {request.url.host}")

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ok = await run_pipeline(pipe, client=fake, http_allow=["example.com"], http_client=http)
    assert ok.ok and ok.outputs["body"] == "hello from example.com"
    other = await run_pipeline(
        pipe,
        {"u": "https://evil.test/x"},
        client=fake,
        http_allow=["example.com"],
        http_client=http,
    )
    assert not other.ok and "not on the allowlist" in (other.error or "")
    scheme = await run_pipeline(
        pipe, {"u": "file:///etc/passwd"}, client=fake, http_allow=["example.com"], http_client=http
    )
    assert not scheme.ok and "only http" in (scheme.error or "")
    await http.aclose()


async def test_env_allowlist(monkeypatch: pytest.MonkeyPatch, fake: FakeClient) -> None:
    import httpx

    monkeypatch.setenv("NIMLAB_LOOM_HTTP_ALLOW", "*.example.org")
    pipe = make(
        [
            inp("u", "https://docs.example.org/x"),
            {"id": "g", "type": "http_get", "params": {"url": "{{u}}"}},
            out("body"),
        ],
        [("u.value", "g.u"), ("g.body", "body.value")],
    )
    http = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, text="ok"))
    )
    result = await run_pipeline(pipe, client=fake, http_client=http)
    assert result.ok
    await http.aclose()


async def test_json_node_validates_against_the_inline_schema() -> None:
    fake = FakeClient()
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}
    pipe = make(
        [
            inp(),
            {"id": "j", "type": "json", "params": {"template": "{{q}}", "schema": schema}},
            out(),
        ],
        [("q.value", "j.q"), ("j.data", "out.value")],
    )
    fake.json_reply = {"n": 3}
    assert (await run_pipeline(pipe, client=fake)).outputs["out"] == {"n": 3}
    fake.json_reply = {"n": "three"}
    bad = await run_pipeline(pipe, client=fake)
    assert not bad.ok and "expected integer" in (bad.error or "")


async def test_council_and_atlas_nodes_in_mock_mode(tmp_path) -> None:
    pipe = make(
        [
            inp("q", "Is a tomato a fruit?"),
            {"id": "c", "type": "council", "params": {"strategy": "best-of-n", "n": 2}},
            {"id": "a", "type": "atlas_ask", "params": {"db": str(tmp_path / "atlas.db")}},
            out("verdict"),
            out("docs"),
        ],
        [
            ("q.value", "c.question"),
            ("c.answer", "verdict.value"),
            ("q.value", "a.question"),
            ("a.answer", "docs.value"),
        ],
    )
    result = await run_pipeline(pipe, mock=True)
    assert result.ok, result.error
    assert result.outputs["verdict"].strip() and isinstance(result.outputs["docs"], str)


# ------------------------------------------------------------------ retries & timeouts
@pytest.fixture
def sleeps_loom(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        recorded.append(seconds)

    monkeypatch.setattr(engine_module, "_sleep", fake_sleep)
    return recorded


async def test_retries_recover_from_transient_errors(sleeps_loom: list[float]) -> None:
    fake = FakeClient()
    fake.fail_first = 2
    pipe = make(
        [inp(), prompt("a", "{{q}}", retries=2), out()],
        [("q.value", "a.q"), ("a.text", "out.value")],
    )
    events: list[dict[str, Any]] = []
    result = await run_pipeline(pipe, client=fake, on_event=events.append, stream=False)
    assert result.ok and result.nodes["a"].attempts == 3
    assert [e["attempt"] for e in events if e["type"] == "node.retry"] == [1, 2]
    assert sleeps_loom == [0.5, 1.0]  # exponential backoff


async def test_retries_exhausted_fails_the_run(sleeps_loom: list[float]) -> None:
    fake = FakeClient()
    fake.fail_first = 5
    pipe = make(
        [inp(), prompt("a", "{{q}}", retries=1), prompt("b", "{{a}}"), out()],
        [("q.value", "a.q"), ("a.text", "b.a"), ("b.text", "out.value")],
    )
    events: list[dict[str, Any]] = []
    result = await run_pipeline(pipe, client=fake, on_event=events.append, stream=False)
    assert not result.ok and result.failed_node == "a" and result.error == "boom"
    assert result.nodes["a"].attempts == 2 and result.nodes["a"].status == "error"
    assert result.nodes["b"].status == "pending"  # never started
    assert (
        kinds(events, "node.error") == ["a"]
        and events[-1]["type"] == "run.done"
        and events[-1]["ok"] is False
    )
    with pytest.raises(Exception, match="boom"):
        result.raise_for_error()


async def test_timeout_fails_with_a_clear_message(sleeps_loom: list[float]) -> None:
    fake = FakeClient(delay=1.0)
    pipe = make(
        [inp(), prompt("a", "{{q}}", timeout=0.05), out()],
        [("q.value", "a.q"), ("a.text", "out.value")],
    )
    result = await run_pipeline(pipe, client=fake, stream=False)
    assert not result.ok and "timed out after 0.05s" in (result.error or "")
    assert isinstance(NodeTimeout("x"), Exception)


async def test_timeout_then_retry_can_succeed(sleeps_loom: list[float]) -> None:
    fake = FakeClient()
    slow_once = {"left": 1}
    original = fake.chat

    async def chat(model: str, messages: list[dict[str, Any]], **kw: Any) -> Any:
        if slow_once["left"]:
            slow_once["left"] -= 1
            await asyncio.sleep(1)
        return await original(model, messages, **kw)

    fake.chat = chat  # type: ignore[method-assign]
    pipe = make(
        [inp(), prompt("a", "{{q}}", timeout=0.05, retries=1), out()],
        [("q.value", "a.q"), ("a.text", "out.value")],
    )
    result = await run_pipeline(pipe, client=fake, stream=False)
    assert result.ok and result.nodes["a"].attempts == 2


async def test_failure_cancels_running_siblings() -> None:
    fake = FakeClient(delay=0.5)
    fake.fail_first = 1

    pipe = make(
        [inp(), prompt("a", "one {{q}}"), prompt("b", "two {{q}}"), out("o1"), out("o2")],
        [("q.value", "a.q"), ("q.value", "b.q"), ("a.text", "o1.value"), ("b.text", "o2.value")],
    )
    result = await run_pipeline(pipe, client=fake, stream=False)
    assert not result.ok and result.elapsed_s < 0.4  # the sibling was cancelled, not awaited


# ------------------------------------------------------------------ caching
def cached_pipeline(template: str = "Echo {{q}}") -> Any:
    return make(
        [inp(), prompt("a", template), out()], [("q.value", "a.q"), ("a.text", "out.value")]
    )


async def test_cache_hit_skips_the_model(tmp_path, fake: FakeClient) -> None:
    cache = tmp_path / "cache"
    first = await run_pipeline(cached_pipeline(), client=fake, cache=cache, stream=False)
    assert first.cache_hits == 0 and len(fake.calls) == 1 and any(cache.glob("*.json"))
    events: list[dict[str, Any]] = []
    second = await run_pipeline(cached_pipeline(), client=fake, cache=cache, on_event=events.append)
    assert len(fake.calls) == 1  # no new model call
    assert second.outputs == first.outputs and second.cache_hits == 1
    end = next(e for e in events if e["type"] == "node.end" and e["node"] == "a")
    assert end["cached"] is True and end["tokens"] == 0
    assert second.nodes["a"].status == "cached"


async def test_cache_key_covers_config_and_inputs(tmp_path, fake: FakeClient) -> None:
    cache = tmp_path / "cache"
    await run_pipeline(cached_pipeline(), client=fake, cache=cache, stream=False)
    await run_pipeline(
        cached_pipeline(), {"q": "other"}, client=fake, cache=cache, stream=False
    )  # new input
    await run_pipeline(
        cached_pipeline("Changed {{q}}"), client=fake, cache=cache, stream=False
    )  # new config
    assert len(fake.calls) == 3
    await run_pipeline(cached_pipeline(), {"q": "other"}, client=fake, cache=cache, stream=False)
    assert len(fake.calls) == 3


async def test_cache_is_off_by_default_and_per_node(tmp_path, fake: FakeClient) -> None:
    await run_pipeline(cached_pipeline(), client=fake, stream=False)
    await run_pipeline(cached_pipeline(), client=fake, stream=False)
    assert len(fake.calls) == 2  # default: no cache
    pipe = make(
        [inp(), prompt("a", "Echo {{q}}", cache=False), out()],
        [("q.value", "a.q"), ("a.text", "out.value")],
    )
    for _ in range(2):
        await run_pipeline(pipe, client=fake, cache=tmp_path / "c", stream=False)
    assert len(fake.calls) == 4


async def test_cache_default_directory_and_backend_separation(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    mock_client, live_client = FakeClient(mock=True), FakeClient(mock=False)
    await run_pipeline(cached_pipeline(), client=mock_client, cache=True, stream=False)
    assert any((tmp_path / ".nimlab" / "loom" / "cache").glob("*.json"))
    await run_pipeline(cached_pipeline(), client=live_client, cache=True, stream=False)
    assert len(live_client.calls) == 1  # mock results never answer live runs
    monkeypatch.setenv("NIMLAB_LOOM_CACHE", str(tmp_path / "elsewhere"))
    await run_pipeline(cached_pipeline(), client=mock_client, cache=True, stream=False)
    assert any((tmp_path / "elsewhere").glob("*.json"))


async def test_corrupt_cache_entries_are_misses(tmp_path, fake: FakeClient) -> None:
    cache = tmp_path / "cache"
    await run_pipeline(cached_pipeline(), client=fake, cache=cache, stream=False)
    for f in cache.glob("*.json"):
        f.write_text("{not json")
    result = await run_pipeline(cached_pipeline(), client=fake, cache=cache, stream=False)
    assert result.ok and len(fake.calls) == 2


async def test_owned_client_is_created_and_closed_in_mock_mode() -> None:
    result = await run_pipeline(cached_pipeline(), mock=True)
    assert result.ok and result.outputs["out"]


async def test_regex_limits(fake: FakeClient) -> None:
    pipe = make(
        [inp("t", "abc"), {"id": "r", "type": "regex", "params": {"pattern": "a" * 600}}, out()],
        [("t.value", "r.text"), ("r.match", "out.value")],
    )
    result = await run_pipeline(pipe, client=fake)
    assert not result.ok and "too long" in (result.error or "")
