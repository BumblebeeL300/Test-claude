"""POST /api/uninstall and the update endpoints, always against tmp dirs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from studio_helpers import HOST, make_settings

from nimlab.studio import server as server_module
from nimlab.studio import uninstall
from nimlab.studio.server import create_app
from nimlab.studio.uninstall import InstallInfo


@pytest.fixture
def populated(settings: Any, tmp_path: Path) -> Path:
    home: Path = settings.home
    (home / "Projects" / "p1").mkdir(parents=True)
    (home / "Projects" / "p1" / "a.py").write_text("print(1)")
    (home / "loom").mkdir()
    (home / "loom" / "x.json").write_text("{}")
    (home / "atlas.db").write_bytes(b"db")
    (home / "studio_history.json").write_text("{}")
    cfg = tmp_path / "nimlab-config"
    cfg.mkdir(exist_ok=True)
    (cfg / "credentials.json").write_text("{}")
    return home


def test_dry_run_returns_the_plan_and_removes_nothing(client: TestClient, populated: Path) -> None:
    resp = client.post("/api/uninstall", json={"dry_run": True})
    assert resp.status_code == 200, resp.text
    plan = resp.json()
    assert plan["dry_run"] is True and plan["home"] == str(populated.resolve())
    items = {i["id"]: i for i in plan["items"]}
    assert items["key"]["selected"] and items["data"]["selected"]
    assert not items["projects"]["selected"] and not items["atlas"]["selected"]
    assert items["projects"]["count"] == 2 and items["projects"]["bytes"] > 0
    assert items["atlas"]["bytes"] == 2
    assert (populated / "studio_history.json").exists() and (populated / "Projects").exists()


def test_dry_run_reflects_the_checkboxes(client: TestClient, populated: Path) -> None:
    plan = client.post(
        "/api/uninstall", json={"dry_run": True, "remove_projects": True, "remove_app": True}
    ).json()
    items = {i["id"]: i for i in plan["items"]}
    assert items["projects"]["selected"] and items["app"]["selected"]
    assert plan["install"]["kind"] in {"dev", "editable", "pip", "unknown", "frozen-app"}


def test_real_uninstall_requires_the_typed_confirmation(
    client: TestClient, populated: Path
) -> None:
    for confirm in ("", "uninstall", "yes"):
        resp = client.post("/api/uninstall", json={"dry_run": False, "confirm": confirm})
        assert resp.status_code == 422
    assert (populated / "studio_history.json").exists()


def test_uninstall_removes_data_but_keeps_projects_by_default(
    client: TestClient, populated: Path, tmp_path: Path
) -> None:
    resp = client.post("/api/uninstall", json={"dry_run": False, "confirm": "UNINSTALL"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["errors"] == [] and body["exiting"] is False  # tests never quit the process
    assert not (populated / "studio_history.json").exists()
    assert not (tmp_path / "nimlab-config" / "credentials.json").exists()
    assert (populated / "Projects" / "p1" / "a.py").exists()
    assert (populated / "loom" / "x.json").exists() and (populated / "atlas.db").exists()


def test_uninstall_everything(client: TestClient, populated: Path) -> None:
    resp = client.post(
        "/api/uninstall",
        json={
            "dry_run": False,
            "confirm": "UNINSTALL",
            "remove_projects": True,
            "remove_atlas": True,
        },
    )
    assert resp.status_code == 200
    assert not populated.exists()


def test_removing_the_app_schedules_the_detached_helper(
    client: TestClient, populated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launched: list[InstallInfo] = []
    info = InstallInfo("pip", removable=True, instructions="pip uninstall -y nimlab")
    monkeypatch.setattr(uninstall, "detect_install", lambda **kw: info)

    def fake_schedule(self: Any, i: InstallInfo) -> bool:
        launched.append(i)
        return True

    monkeypatch.setattr(server_module.StudioState, "schedule_app_removal", fake_schedule)
    body = client.post(
        "/api/uninstall", json={"dry_run": False, "confirm": "UNINSTALL", "remove_app": True}
    ).json()
    assert launched == [info] and body["app_removal_scheduled"] is True


def test_editable_installs_get_instructions_instead_of_a_helper(
    client: TestClient, populated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    info = InstallInfo("editable", removable=False, instructions="pip uninstall nimlab by hand")
    monkeypatch.setattr(uninstall, "detect_install", lambda **kw: info)
    body = client.post(
        "/api/uninstall", json={"dry_run": False, "confirm": "UNINSTALL", "remove_app": True}
    ).json()
    assert body["app_removal_scheduled"] is False
    assert body["app_instructions"] == "pip uninstall nimlab by hand"


def test_the_exit_hook_runs_only_when_the_launcher_allows_it(tmp_path: Path) -> None:
    calls: list[int] = []
    settings = make_settings(tmp_path, allow_exit=True)
    app = create_app(settings)
    app.state.studio.exit_hook = lambda: calls.append(1)
    app.state.studio.schedule_exit(delay=0)
    import time

    time.sleep(0.2)
    assert calls == [1]
    quiet = create_app(make_settings(tmp_path / "other"))
    quiet.state.studio.exit_hook = lambda: calls.append(2)
    quiet.state.studio.schedule_exit(delay=0)
    time.sleep(0.2)
    assert calls == [1]


def test_a_dangerous_home_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = tmp_path / "me"
    fake.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: fake))
    with TestClient(create_app(make_settings(tmp_path, home=fake)), base_url=HOST) as c:
        assert c.post("/api/uninstall", json={"dry_run": True}).status_code == 422
        resp = c.post("/api/uninstall", json={"dry_run": False, "confirm": "UNINSTALL"})
        assert resp.status_code == 422
    assert fake.exists()


# ------------------------------------------------------------------------------- updates
def test_update_check_endpoint(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from nimlab.studio import updater

    monkeypatch.setattr(
        updater,
        "check_for_updates",
        lambda **kw: updater.UpdateStatus(
            kind="git", state="behind", behind=3, can_update=True, message="3 new commits"
        ),
    )
    data = client.get("/api/update/check").json()
    assert data["state"] == "behind" and data["behind"] == 3 and data["can_update"] is True
    assert data["version"]


def test_update_apply_endpoint(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from nimlab.studio import updater

    monkeypatch.setattr(
        updater,
        "apply_update",
        lambda **kw: {"ok": True, "restart": True, "steps": [], "message": "m"},
    )
    assert client.post("/api/update/apply").json()["restart"] is True
