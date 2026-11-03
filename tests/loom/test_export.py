from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from nimlab.loom import Pipeline, list_pipelines, load_pipeline, run_pipeline, to_mermaid, to_python

from .conftest import inp, make, out, prompt

BUNDLED = [i.id for i in list_pipelines()]


def test_mermaid_structure() -> None:
    pipe = load_pipeline("support-triage")
    text = to_mermaid(pipe)
    assert text.startswith("%% Support triage\nflowchart LR")
    assert 'n_is_billing{"Billing?<br/><i>branch</i>"}' in text  # branch is a diamond
    assert 'n_ticket(["ticket<br/><i>input</i>"])' in text
    assert 'n_is_billing -->|"then → info"| n_billing_reply' in text  # branch arms are labelled
    assert text.count("-->") == len(pipe.edges)
    assert "classDef cat_model" in text


def test_mermaid_escapes_quotes_and_dashes() -> None:
    pipe = make(
        [inp("in-1"), {"id": "x", "type": "template", "label": 'say "hi"', "params": {"template": "{{a}}"}}, out()],
        [("in-1.value", "x.a"), ("x.text", "out.value")],
    )
    text = to_mermaid(pipe)
    assert "n_in_1" in text and "#quot;hi#quot;" in text and '"hi"' not in text


@pytest.mark.parametrize("name", BUNDLED)
def test_python_export_compiles(name: str) -> None:
    source = to_python(load_pipeline(name))
    compile(source, f"{name}.py", "exec")
    assert source.startswith("#!/usr/bin/env python3")
    assert "from nimlab.core import" in source and "async def n_" in source
    assert "eval(" not in source.replace("evaluate(", "").replace("def evaluate", "")  # no raw eval


@pytest.mark.parametrize("name", BUNDLED)
def test_exported_script_runs_in_mock_mode(name: str, tmp_path: Path) -> None:
    script = tmp_path / f"{name}.py"
    script.write_text(to_python(load_pipeline(name)), encoding="utf-8")
    env = {**os.environ, "NIMLAB_MOCK": "1", "NIMLAB_NO_DOTENV": "1"}
    proc = subprocess.run(
        [sys.executable, str(script), "--json"],
        capture_output=True, text=True, env=env, cwd=tmp_path, timeout=120,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stderr
    outputs = json.loads(proc.stdout)
    expected = Pipeline.model_validate(load_pipeline(name).to_data())
    names = {str(n.params.get("name") or n.id) for n in expected.nodes if n.type == "output"}
    assert set(outputs) <= names and outputs  # skipped arms are absent, the rest is present


def test_exported_script_matches_the_engine(tmp_path: Path) -> None:
    """Same pipeline, same deterministic mock backend: identical outputs from engine and script."""
    pipe = make(
        [inp("topic", "tea"), prompt("a", "About {{topic}}"), {"id": "t", "type": "template", "params": {"template": "<{{x}}>"}}, out()],
        [("topic.value", "a.topic"), ("a.text", "t.x"), ("t.text", "out.value")],
    )
    import asyncio

    expected = asyncio.run(run_pipeline(pipe, {"topic": "coffee"}, mock=True)).outputs
    script = tmp_path / "p.py"
    script.write_text(to_python(pipe), encoding="utf-8")
    env = {**os.environ, "NIMLAB_MOCK": "1", "NIMLAB_NO_DOTENV": "1"}
    proc = subprocess.run(
        [sys.executable, str(script), "--json", "--set", "topic=coffee"],
        capture_output=True, text=True, env=env, cwd=tmp_path, timeout=120,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == expected


def test_exported_script_reports_missing_inputs(tmp_path: Path) -> None:
    pipe = make([{"id": "q", "type": "input", "params": {"type": "string"}}, out()], [("q.value", "out.value")])
    script = tmp_path / "p.py"
    script.write_text(to_python(pipe), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(script), "--mock"], capture_output=True, text=True, cwd=tmp_path,
        env={**os.environ, "NIMLAB_NO_DOTENV": "1"}, timeout=60,
    )  # fmt: skip
    assert proc.returncode != 0 and "missing input 'q'" in proc.stderr
