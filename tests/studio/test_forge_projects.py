"""The Foundry builder (simulated and scripted-live) and the project browsing endpoints."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from studio_helpers import HOST, events_of, finish, make_settings

from nimlab.studio import projects as projects_module
from nimlab.studio.server import create_app

sys.path.insert(0, str(Path(__file__).parents[1] / "forge"))
from forge_fakes import (
    APPROVE_JSON,
    FIZZBUZZ_BUGGY,
    FIZZBUZZ_PATCH,
    FIZZBUZZ_TESTS,
    PLAN_JSON,
    FakeClient,
    calls,
    say,
    tc,
)

EXPECTED_ORDER = [
    "run.start",
    "plan",
    "file.write",
    "test.run",
    "test.result",
    "review",
    "run.done",
]
STUDIO_TYPES = {
    "phase.start",
    "phase.end",
    "plan",
    "tool.call",
    "tool.result",
    "file.write",
    "test.run",
    "test.result",
    "review",
    "log",
    "usage",
    "run.start",
    "run.done",
}


def build(
    c: TestClient, prompt: str = "Build a todo list manager", **extra: Any
) -> tuple[str, str]:
    body = c.post("/api/forge/build", json={"prompt": prompt, **extra}).json()
    return body["run_id"], body["project"]


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """One simulated build shared by the assertions below (it runs real pytest twice)."""
    tmp = tmp_path_factory.mktemp("studio-forge")
    app = create_app(make_settings(tmp))
    with TestClient(app, base_url=HOST) as c:
        rid, project = build(c)
        events = events_of(c, rid)
        return {
            "tmp": tmp,
            "run_id": rid,
            "project": project,
            "events": events,
            "detail": c.get(f"/api/runs/{rid}").json(),
            "tree": c.get(f"/api/projects/{project}/tree").json(),
            "listing": c.get("/api/projects").json(),
            "diff": c.get(f"/api/projects/{project}/diff").json(),
        }


class TestSimulatedBuild:
    def test_returns_run_and_project(self, built: dict[str, Any]) -> None:
        assert built["project"] == "build-a-todo-list-manager"
        assert built["detail"]["simulated"] is True and built["detail"]["status"] == "done"

    def test_event_types_are_the_documented_ones(self, built: dict[str, Any]) -> None:
        types = [e["type"] for e in built["events"]]
        assert set(types) <= STUDIO_TYPES
        assert {
            "phase.start",
            "phase.end",
            "plan",
            "tool.call",
            "tool.result",
            "file.write",
            "test.run",
            "test.result",
            "review",
            "log",
            "usage",
        } <= set(types)
        assert [e["seq"] for e in built["events"]] == list(range(1, len(types) + 1))

    def test_sequence(self, built: dict[str, Any]) -> None:
        types = [e["type"] for e in built["events"]]
        positions = [types.index(t) for t in EXPECTED_ORDER]
        assert positions == sorted(positions)
        assert types[-1] == "run.done"
        first_test = types.index("test.result")
        assert types.count("file.write") >= 4 and types.index("file.write") < first_test
        assert types.count("test.run") == 2 and types.count("test.result") == 2

    def test_one_failing_test_then_green(self, built: dict[str, Any]) -> None:
        results = [e["data"] for e in built["events"] if e["type"] == "test.result"]
        first, second = results
        assert (first["failed"], first["ok"]) == (1, False) and first["passed"] >= 4
        assert first["failing"] == ["test_todo.py::test_remove_unknown_id_raises"]
        assert any(r["status"] == "failed" for r in first["results"])
        assert first["names"] and "test_todo.py::test_add_assigns_increasing_ids" in first["names"]
        assert (second["failed"], second["ok"], second["failing"]) == (0, True, [])
        assert all(r["status"] == "passed" for r in second["results"])

    def test_file_write_payload(self, built: dict[str, Any]) -> None:
        writes = [e["data"] for e in built["events"] if e["type"] == "file.write"]
        paths = [w["path"] for w in writes]
        assert paths[:3] == ["README.md", "requirements-dev.txt", "todo.py"]
        assert "test_todo.py" in paths
        assert paths.count("todo.py") == 2  # created, then patched
        patched = [w for w in writes if w["path"] == "todo.py"][-1]
        assert patched["action"] == "modified"
        created = writes[0]
        assert created["action"] == "created" and created["bytes"] > 0 and created["lines"] > 0

    def test_plan_review_and_usage(self, built: dict[str, Any]) -> None:
        plan = next(e for e in built["events"] if e["type"] == "plan")["data"]
        assert plan["goal"] and [s["id"] for s in plan["steps"]] == [1, 2, 3]
        review = next(e for e in built["events"] if e["type"] == "review")["data"]
        assert review["verdict"] == "approve" and review["blocking"] == 0
        usage = [e["data"] for e in built["events"] if e["type"] == "usage"]
        assert usage and usage[-1]["totals"]["calls"] == len(usage)
        assert usage[0]["model"] and usage[0]["tokens_in"] > 0
        phases = [e["data"]["phase"] for e in built["events"] if e["type"] == "phase.start"]
        assert phases[:4] == ["init", "explore", "plan", "implement"] and "review" in phases
        ends = [e["data"] for e in built["events"] if e["type"] == "phase.end"]
        assert all("seconds" in e for e in ends)

    def test_final_result(self, built: dict[str, Any]) -> None:
        result = built["detail"]["result"]
        assert result["status"] == "success" and result["ok"] is True
        assert result["project"]["id"] == built["project"] and result["project"]["files"] == 4
        assert result["simulated"] is True and result["iterations"] == 1
        assert result["tests"]["failed"] == 0 and result["tests"]["passed"] >= 5
        assert {c["path"] for c in result["files_changed"]} == {
            "README.md",
            "requirements-dev.txt",
            "todo.py",
            "test_todo.py",
        }
        assert "diff --git" in result["diff"]
        assert result["review"]["verdict"] == "approve"
        assert built["events"][-1]["data"]["result"]["status"] == "success"

    def test_real_files_in_the_projects_dir(self, built: dict[str, Any]) -> None:
        root = built["tmp"] / "home" / "Projects" / built["project"]
        assert {p.name for p in root.iterdir() if p.is_file()} == {
            "README.md",
            "requirements-dev.txt",
            "todo.py",
            "test_todo.py",
        }
        source = (root / "todo.py").read_text()
        assert "raise KeyError(task_id)" in source  # the patched, correct version
        assert (root / ".forge" / "forge_report.json").is_file()

    def test_project_listing_and_tree(self, built: dict[str, Any]) -> None:
        (row,) = built["listing"]
        assert set(row) == {"id", "name", "path", "files", "updated_at", "last_status"}
        assert (
            row["id"] == built["project"] and row["files"] == 4 and row["last_status"] == "success"
        )
        assert row["path"].endswith("/Projects/build-a-todo-list-manager")
        tree = built["tree"]
        names = {n["name"] for n in tree["tree"]}
        assert names == {"README.md", "requirements-dev.txt", "todo.py", "test_todo.py"}
        assert {f["path"] for f in tree["files"]} == names
        assert all(n["type"] == "file" and n["size"] > 0 for n in tree["tree"])
        assert ".forge" not in str(tree)

    def test_diff_endpoint(self, built: dict[str, Any]) -> None:
        diff = built["diff"]
        assert diff["has_report"] is True and diff["status"] == "success"
        assert "diff --git a/todo.py b/todo.py" in diff["diff"]
        assert {f["path"] for f in diff["files"]} >= {"todo.py", "test_todo.py"}

    def test_projects_dir_env_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NIMLAB_PROJECTS_DIR", str(tmp_path / "elsewhere"))
        app = create_app(make_settings(tmp_path))
        with TestClient(app, base_url=HOST) as c:
            rid, project = build(c, "slugify some titles")
            assert finish(c, rid)["status"] == "done"
        assert (tmp_path / "elsewhere" / project / "slugify.py").is_file()


class TestBuildRequests:
    def test_named_project_is_reused_and_default_names_are_unique(self, client: TestClient) -> None:
        r1, p1 = build(client, "fizzbuzz please", project="My Demo!")
        r3, p3 = build(client, "fizzbuzz please")
        r4, p4 = build(client, "fizzbuzz please")  # same prompt, started back to back
        assert (p1, p3, p4) == ("my-demo", "fizzbuzz-please", "fizzbuzz-please-2")
        for rid in (r1, r3, r4):
            assert finish(client, rid)["status"] == "done"
        r2, p2 = build(client, "fizzbuzz please", project="My Demo!")  # extends the same folder
        assert p2 == "my-demo"
        finish(client, r2)
        assert {p["id"] for p in client.get("/api/projects").json()} == {"my-demo", p3, p4}

    def test_a_project_cannot_be_built_twice_at_once(self, tmp_path: Path) -> None:
        app = create_app(make_settings(tmp_path, sim_delay=0.2))
        with TestClient(app, base_url=HOST) as c:
            rid, _ = build(c, "todo", project="busy")
            resp = c.post("/api/forge/build", json={"prompt": "todo", "project": "busy"})
            assert resp.status_code == 409 and "already being built" in resp.json()["detail"]
            c.post(f"/api/runs/{rid}/cancel")
            finish(c, rid)

    def test_prompt_keywords_pick_the_project(self, client: TestClient, settings: Any) -> None:
        rid, project = build(client, "an LRU cache")
        assert finish(client, rid)["result"]["status"] == "success"
        assert (settings.projects_dir / project / "lru_cache.py").is_file()

    def test_validation(self, client: TestClient) -> None:
        assert client.post("/api/forge/build", json={"prompt": ""}).status_code == 422
        assert client.post("/api/forge/build", json={}).status_code == 422
        assert (
            client.post("/api/forge/build", json={"prompt": "x", "max_iters": 0}).status_code == 422
        )

    def test_max_iters_is_applied(self, client: TestClient) -> None:
        rid, _ = build(client, "fizzbuzz", max_iters=1)
        # one fix round is exactly enough for the scripted bug
        assert finish(client, rid)["result"]["status"] in ("success", "failed")


class TestLiveScriptedForge:
    """The non-simulated path: Forge runs on the shared client (here a scripted fake)."""

    def test_real_client_events_are_mapped(self, tmp_path: Path) -> None:
        fake = FakeClient(
            chat=[
                calls(
                    tc("write_file", path="fizzbuzz.py", content=FIZZBUZZ_BUGGY),
                    tc("write_file", path="test_fizzbuzz.py", content=FIZZBUZZ_TESTS),
                ),
                say("Implemented."),
                calls(tc("apply_patch", path="fizzbuzz.py", patch=FIZZBUZZ_PATCH)),
                say("Fixed."),
            ],
            json=[PLAN_JSON, APPROVE_JSON],
        )
        app = create_app(make_settings(tmp_path), client=fake)
        with TestClient(app, base_url=HOST) as c:
            assert c.get("/api/status").json()["mode"] == "live"
            rid, _ = build(c, "fizzbuzz")
            detail = finish(c, rid)
            events = events_of(c, rid)
        assert detail["simulated"] is False
        assert detail["result"]["status"] == "success" and detail["result"]["simulated"] is False
        types = [e["type"] for e in events]
        assert types.count("file.write") == 3  # two creates and one patch
        failing = next(e["data"] for e in events if e["type"] == "test.result")
        assert failing["failed"] == 4 and "test_fizzbuzz.py::test_fizzbuzz_15" in failing["failing"]
        assert fake.unexpected == []


@pytest.fixture(scope="module")
def shared(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[TestClient, str, Path]]:
    """A client with one finished project; module-scoped because every build runs real pytest."""
    tmp = tmp_path_factory.mktemp("studio-files")
    settings = make_settings(tmp)
    with TestClient(create_app(settings), base_url=HOST) as c:
        rid, project = build(c, "Build a todo list manager")
        finish(c, rid)
        yield c, project, tmp


class TestProjectFiles:
    @pytest.fixture
    def project(self, shared: tuple[TestClient, str, Path]) -> str:
        return shared[1]

    @pytest.fixture
    def client(self, shared: tuple[TestClient, str, Path]) -> TestClient:  # type: ignore[override]
        return shared[0]

    @pytest.fixture
    def tmp_path(self, shared: tuple[TestClient, str, Path]) -> Path:
        return shared[2]

    def test_read_file(self, client: TestClient, project: str) -> None:
        data = client.get(f"/api/projects/{project}/file", params={"path": "todo.py"}).json()
        assert data["path"] == "todo.py" and data["language"] == "python"
        assert "class TodoList" in data["content"] and data["lines"] > 20 and data["size"] > 0

    @pytest.mark.parametrize(
        "path",
        [
            "../../etc/passwd",
            "/etc/passwd",
            "..",
            "../",
            "a/../../x",
            "..\\..\\x",
            "C:\\Windows\\win.ini",
            "todo.py\x00.txt",
            "",
            "..",
        ],
    )
    def test_path_traversal_is_rejected(self, client: TestClient, project: str, path: str) -> None:
        resp = client.get(f"/api/projects/{project}/file", params={"path": path})
        assert resp.status_code in (400, 404), (path, resp.status_code, resp.text)
        assert "root:" not in resp.text

    def test_symlink_escape_is_rejected(
        self, client: TestClient, project: str, tmp_path: Path
    ) -> None:
        root = Path(
            next(p["path"] for p in client.get("/api/projects").json() if p["id"] == project)
        )
        secret = tmp_path / "secret.txt"
        secret.write_text("TOP SECRET")
        (root / "link.txt").symlink_to(secret)
        resp = client.get(f"/api/projects/{project}/file", params={"path": "link.txt"})
        assert resp.status_code == 400 and "TOP SECRET" not in resp.text

    def test_internal_folders_are_not_browsable(self, client: TestClient, project: str) -> None:
        resp = client.get(
            f"/api/projects/{project}/file", params={"path": ".forge/forge_report.json"}
        )
        assert resp.status_code == 403

    def test_binary_and_oversized_files(self, client: TestClient, project: str) -> None:
        root = Path(
            next(p["path"] for p in client.get("/api/projects").json() if p["id"] == project)
        )
        (root / "blob.bin").write_bytes(b"\x00\x01\x02binary")
        (root / "big.txt").write_text("x" * (projects_module.MAX_FILE_BYTES + 1))
        (root / "latin.txt").write_bytes("caf\xe9".encode("latin-1"))
        assert (
            client.get(f"/api/projects/{project}/file", params={"path": "blob.bin"}).status_code
            == 415
        )
        assert (
            client.get(f"/api/projects/{project}/file", params={"path": "big.txt"}).status_code
            == 413
        )
        assert (
            client.get(f"/api/projects/{project}/file", params={"path": "latin.txt"}).status_code
            == 415
        )
        assert (
            client.get(f"/api/projects/{project}/file", params={"path": "nope.py"}).status_code
            == 404
        )
        assert client.get(f"/api/projects/{project}/file", params={"path": "."}).status_code == 404

    @pytest.mark.parametrize("pid", ["nope", "..", "a b", "UPPER", ".hidden", "x" * 80])
    def test_bad_or_unknown_project_ids(self, client: TestClient, pid: str) -> None:
        for suffix in ("tree", "diff"):
            resp = client.get(f"/api/projects/{pid}/{suffix}")
            assert resp.status_code in (400, 404)
        assert client.post(f"/api/projects/{pid}/reveal").status_code in (400, 404)

    def test_project_id_cannot_walk_out_via_symlink(
        self, client: TestClient, project: str, tmp_path: Path
    ) -> None:
        root = tmp_path / "home" / "Projects"
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "x.txt").write_text("hi")
        (root / "evil").symlink_to(outside, target_is_directory=True)
        assert client.get("/api/projects/evil/tree").status_code == 404
        assert "evil" not in [p["id"] for p in client.get("/api/projects").json()]

    def test_diff_without_a_report(self, client: TestClient, tmp_path: Path) -> None:
        (tmp_path / "home" / "Projects" / "manual").mkdir(parents=True)
        (tmp_path / "home" / "Projects" / "manual" / "a.txt").write_text("hi")
        diff = client.get("/api/projects/manual/diff").json()
        assert diff == {
            "project": "manual",
            "has_report": False,
            "status": None,
            "diff": "",
            "files": [],
        }
        (row,) = [p for p in client.get("/api/projects").json() if p["id"] == "manual"]
        assert row["last_status"] is None and row["files"] == 1

    def test_reveal_is_a_noop_off_macos(self, client: TestClient, project: str) -> None:
        body = client.post(f"/api/projects/{project}/reveal").json()
        assert body["ok"] is True and body["path"].endswith(project) and body["revealed"] is False

    def test_reveal_runs_open_dash_r_on_macos(
        self, client: TestClient, project: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[list[str]] = []
        monkeypatch.setattr(projects_module.sys, "platform", "darwin")
        monkeypatch.setattr(
            projects_module.subprocess, "run", lambda cmd, **kw: seen.append(list(cmd))
        )
        body = client.post(f"/api/projects/{project}/reveal").json()
        assert body["revealed"] is True
        assert seen == [["open", "-R", body["path"]]]
