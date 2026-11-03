from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from nimlab.loom import lint, load_pipeline

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "loom"


def test_example_pipelines_lint_clean() -> None:
    files = sorted(EXAMPLES.glob("*.json"))
    assert files
    for path in files:
        assert lint(load_pipeline(path)).ok, path


@pytest.mark.parametrize("script", sorted(EXAMPLES.glob("*.py")), ids=lambda p: p.name)
def test_example_scripts_run_offline(script: Path, tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=120,
        env={**os.environ, "NIMLAB_MOCK": "1", "NIMLAB_NO_DOTENV": "1", "HOME": str(tmp_path)},
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip()
