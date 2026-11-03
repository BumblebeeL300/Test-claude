from __future__ import annotations

from typing import Any

import pytest

from nimlab.loom import lint, list_pipelines, load_pipeline

from .conftest import inp, make, out, prompt


def codes(report: Any, severity: str | None = None) -> list[str]:
    return [i.code for i in report.issues if severity in (None, i.severity)]


def test_clean_pipeline() -> None:
    pipe = make([inp(), prompt("a", "{{q}}"), out()], [("q.value", "a.q"), ("a.text", "out.value")])
    report = lint(pipe)
    assert report.ok and report.issues == []
    assert report.ports["a"]["inputs"][0]["name"] == "q"  # ports are reported for editors
    assert report.ports["a"]["outputs"] == [{"name": "text", "type": "text"}]


@pytest.mark.parametrize("bundled", [i.id for i in list_pipelines()])
def test_bundled_pipelines_are_clean(bundled: str) -> None:
    assert lint(load_pipeline(bundled)).issues == []


def test_unknown_variable() -> None:
    pipe = make([prompt("a", "Hello {{who}}"), out()], [("a.text", "out.value")])
    report = lint(pipe)
    issue = next(i for i in report.issues if i.code == "unbound")
    assert issue.severity == "error" and issue.node == "a" and issue.port == "who"
    assert "unknown variable" in issue.message
    # a default silences it
    pipe.node("a").defaults["who"] = "world"
    assert lint(pipe).ok


def test_type_mismatch_error_and_warning() -> None:
    pipe = make(
        [
            {"id": "t", "type": "input", "params": {"type": "list", "default": ["a"]}},
            {"id": "e", "type": "embed"},
            {"id": "r", "type": "rerank"},
            {"id": "n", "type": "input", "params": {"type": "number", "default": 1}},
            {"id": "c", "type": "chunk"},
            out("o1"),
            out("o2"),
        ],
        [
            ("t.value", "e.texts"),
            ("e.vectors", "r.query"),  # vectors -> text: incompatible
            ("t.value", "r.passages"),
            ("n.value", "c.text"),  # number -> text: fine
            ("r.ranked", "o1.value"),
            ("c.chunks", "o2.value"),
        ],
    )
    report = lint(pipe)
    mismatch = [i for i in report.issues if i.code == "type-mismatch"]
    assert [(i.severity, i.node, i.port) for i in mismatch] == [("error", "r", "query")]
    warn = make(
        [inp("s", "x"), {"id": "r", "type": "rerank"}, out()],
        [("s.value", "r.query"), ("s.value", "r.passages"), ("r.ranked", "out.value")],
    )
    got = [(i.severity, i.code) for i in lint(warn).issues]
    assert got == [("warning", "type-mismatch")]  # text -> list converts at run time


def test_unreachable_and_missing_output() -> None:
    pipe = make(
        [inp(), prompt("used", "{{q}}"), prompt("dead", "{{q}}"), out()],
        [("q.value", "used.q"), ("q.value", "dead.q"), ("used.text", "out.value")],
    )
    report = lint(pipe)
    assert [(i.code, i.node) for i in report.issues] == [("unreachable", "dead")]
    assert report.ok  # a warning, not an error
    none = make([inp(), prompt("a", "{{q}}")], [("q.value", "a.q")])
    assert "no-output" in codes(lint(none), "error")


def test_cycle_is_reported() -> None:
    pipe = make(
        [prompt("a", "{{x}}"), prompt("b", "{{x}}"), out()],
        [("a.text", "b.x"), ("b.text", "a.x"), ("b.text", "out.value")],
    )
    cyc = next(i for i in lint(pipe).issues if i.code == "cycle")
    assert "a -> b -> a" in cyc.message and cyc.severity == "error"


def test_edge_problems() -> None:
    pipe = make(
        [inp(), inp("q2", "x"), prompt("a", "{{q}}"), out()],
        [("q.value", "a.q"), ("q2.value", "a.q"), ("a.nope", "out.value"), ("q.value", "a.ghost")],
    )
    got = codes(lint(pipe), "error")
    assert got.count("multiple-edges") == 1 and got.count("unknown-port") == 2


def test_param_checks() -> None:
    pipe = make(
        [
            inp(),
            {
                "id": "p",
                "type": "prompt",
                "params": {"template": "{{q}}", "temperature": 9, "color": "red"},
            },
            {"id": "b", "type": "branch", "params": {"condition": "value.real"}},
            {"id": "t", "type": "template", "params": {"template": "{{q | shout}}"}},
            {"id": "f", "type": "python_fn", "params": {"fn": "no_such_function"}},
            {"id": "r", "type": "regex", "params": {}},
            out(),
        ],
        [
            ("q.value", "p.q"),
            ("p.text", "b.value"),
            ("q.value", "t.q"),
            ("t.text", "f.value"),
            ("f.result", "r.text"),
            ("b.then", "out.value"),
        ],
    )
    report = lint(pipe)
    text = " | ".join(i.format() for i in report.issues)
    assert "above the maximum" in text
    assert "unknown parameter 'color'" in text
    assert "attribute access" in text
    assert "unknown filter 'shout'" in text
    assert "no_such_function" in text
    assert "missing required parameter 'pattern'" in text


def test_unknown_node_type_and_schema_errors() -> None:
    report = lint({"name": "x", "nodes": [{"id": "a", "type": "teleporter"}]})
    assert "unknown-type" in codes(report, "error")
    broken = lint(
        {"name": "x", "nodes": [{"id": "a", "type": "input"}, {"id": "a", "type": "input"}]}
    )
    assert codes(broken) == ["schema"] and "duplicate node id" in broken.issues[0].message


def test_path_into_scalar_warns() -> None:
    pipe = make(
        [inp(), prompt("a", "{{q.name}}"), out()],
        [("q.value", "a.q"), ("a.text", "out.value")],
    )
    assert codes(lint(pipe)) == ["path-type"]


def test_http_get_warns_until_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    pipe = make(
        [
            inp("u", "https://example.com"),
            {"id": "g", "type": "http_get", "params": {"url": "{{u}}"}},
            out(),
        ],
        [("u.value", "g.u"), ("g.body", "out.value")],
    )
    assert codes(lint(pipe)) == ["http-disabled"]
    monkeypatch.setenv("NIMLAB_LOOM_HTTP_ALLOW", "example.com")
    assert lint(pipe).issues == []


def test_duplicate_io_names_and_unused_defaults() -> None:
    pipe = make(
        [
            inp("a", "x"),
            inp("b", "y"),
            prompt("p", "{{a}}{{b}}"),
            out("o1", "same"),
            out("o2", "same"),
        ],
        [("a.value", "p.a"), ("b.value", "p.b"), ("p.text", "o1.value"), ("p.text", "o2.value")],
    )
    pipe.node("p").defaults["zzz"] = 1
    got = codes(lint(pipe))
    assert "duplicate-name" in got and "unused-default" in got


def test_map_step_is_checked() -> None:
    pipe = make(
        [
            {"id": "i", "type": "input", "params": {"type": "list", "default": ["a"]}},
            {
                "id": "m",
                "type": "map",
                "params": {"step": {"type": "branch", "params": {"condition": "value"}}},
            },
            out(),
        ],
        [("i.value", "m.items"), ("m.results", "out.value")],
    )
    assert "cannot be a 'branch'" in " ".join(i.message for i in lint(pipe).issues)
