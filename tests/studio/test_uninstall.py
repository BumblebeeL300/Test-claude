"""Uninstall planning and removal. Everything happens inside tmp dirs, never the real home."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest

from nimlab.studio import credentials, uninstall
from nimlab.studio.uninstall import InstallInfo, UninstallOptions


@pytest.fixture
def home(tmp_path: Path) -> Path:
    root = tmp_path / "fake-home" / "NIMLab"
    (root / "Projects" / "todo-app").mkdir(parents=True)
    (root / "Projects" / "todo-app" / "main.py").write_text("print('hi')\n" * 10)
    (root / "Projects" / "blog").mkdir()
    (root / "Projects" / "blog" / "x.py").write_text("x = 1\n")
    (root / "loom").mkdir()
    (root / "loom" / "mine.json").write_text("{}")
    (root / "atlas.db").write_bytes(b"A" * 2048)
    (root / "atlas.db-wal").write_bytes(b"W" * 64)
    (root / "studio_history.json").write_text("{}")
    (root / "studio_settings.json").write_text('{"model_overrides": {}}')
    (root / "models_cache.json").write_text("{}")
    (root / "arena" / "runs").mkdir(parents=True)
    (root / "arena" / "runs" / "r1.json").write_text("{}")
    return root


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    # the conftest points NIMLAB_CONFIG_DIR at this folder
    path = tmp_path / "nimlab-config"
    path.mkdir(exist_ok=True)
    (path / "credentials.json").write_text('{"api_key": "nvapi-x"}')
    (path / "other.json").write_text("{}")
    return path


DEV = InstallInfo("dev", instructions="dev checkout")


def run(opts: UninstallOptions, home: Path, **kw: Any) -> dict[str, Any]:
    return uninstall.execute(opts, home=home, install=DEV, delete_key=lambda: True, **kw)


def names(paths: list[str]) -> set[str]:
    return {Path(p).name for p in paths}


# ------------------------------------------------------------------------------- plan
def test_plan_lists_exact_paths_counts_and_sizes(home: Path, config_dir: Path) -> None:
    plan = uninstall.plan(UninstallOptions(), home=home, install=DEV)
    items = {i["id"]: i for i in plan["items"]}
    assert plan["home"] == str(home.resolve())
    assert items["key"]["selected"] and items["data"]["selected"]
    assert not items["projects"]["selected"] and not items["atlas"]["selected"]
    assert not items["app"]["selected"]
    assert names(items["data"]["paths"]) == {
        "studio_history.json",
        "studio_settings.json",
        "models_cache.json",
        "arena",
        "other.json",
    }
    assert items["projects"]["count"] == 3  # two projects + one saved pipeline folder entry
    assert names(items["projects"]["paths"]) == {"Projects", "loom"}
    assert items["projects"]["bytes"] > 0
    assert names(items["atlas"]["paths"]) == {"atlas.db", "atlas.db-wal"}
    assert items["atlas"]["bytes"] == 2048 + 64
    assert items["key"]["paths"] == [str(config_dir / "credentials.json")]


def test_plan_touches_nothing(home: Path, config_dir: Path) -> None:
    before = sorted(str(p) for p in home.rglob("*"))
    uninstall.plan(UninstallOptions(remove_projects=True, remove_atlas=True), home=home)
    assert sorted(str(p) for p in home.rglob("*")) == before


# ------------------------------------------------------------------------------- execute
def test_defaults_remove_key_and_data_but_keep_projects_and_atlas(
    home: Path, config_dir: Path
) -> None:
    report = run(UninstallOptions(dry_run=False), home)
    assert not (config_dir / "credentials.json").exists() and not config_dir.exists()
    assert not (home / "studio_history.json").exists() and not (home / "arena").exists()
    assert (home / "Projects" / "todo-app" / "main.py").is_file()
    assert (home / "loom" / "mine.json").is_file()
    assert (home / "atlas.db").is_file()
    assert report["errors"] == [] and report["refused"] == []
    assert "API key" in report["removed"]


def test_remove_everything_also_removes_the_empty_folders(home: Path, config_dir: Path) -> None:
    opts = UninstallOptions(remove_projects=True, remove_atlas=True, dry_run=False)
    run(opts, home)
    assert not home.exists() and not config_dir.exists()


def test_projects_are_only_removed_when_asked(home: Path, config_dir: Path) -> None:
    run(
        UninstallOptions(remove_data=False, remove_key=False, remove_projects=True, dry_run=False),
        home,
    )
    assert not (home / "Projects").exists() and not (home / "loom").exists()
    assert (home / "studio_history.json").exists() and (home / "atlas.db").exists()
    assert (config_dir / "credentials.json").exists()


def test_atlas_only(home: Path, config_dir: Path) -> None:
    run(
        UninstallOptions(remove_data=False, remove_key=False, remove_atlas=True, dry_run=False),
        home,
    )
    assert not (home / "atlas.db").exists() and not (home / "atlas.db-wal").exists()
    assert (home / "Projects").exists()


def test_key_removal_uses_the_credential_store(
    home: Path, config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    credentials.save_api_key("nvapi-" + "k" * 40)  # file backend (keyring disabled in tests)
    assert credentials.load_api_key()
    uninstall.execute(UninstallOptions(remove_data=False, dry_run=False), home=home, install=DEV)
    assert credentials.load_api_key() is None


# ------------------------------------------------------------------------------- safety
def test_symlinks_are_removed_as_links_and_never_followed(
    home: Path, config_dir: Path, tmp_path: Path
) -> None:
    precious = tmp_path / "precious"
    precious.mkdir()
    (precious / "keep.txt").write_text("do not delete")
    (home / "cache-link").symlink_to(precious, target_is_directory=True)
    (home / "file-link").symlink_to(precious / "keep.txt")
    run(UninstallOptions(dry_run=False), home)
    assert not (home / "cache-link").exists() and not (home / "file-link").is_symlink()
    assert (precious / "keep.txt").read_text() == "do not delete"


def test_a_projects_symlink_escaping_the_folder_only_loses_the_link(
    home: Path, config_dir: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "my-documents"
    (outside / "thesis").mkdir(parents=True)
    (outside / "thesis" / "final.docx").write_text("important")
    for p in sorted((home / "Projects").rglob("*"), reverse=True):
        p.unlink() if p.is_file() else p.rmdir()
    (home / "Projects").rmdir()
    (home / "Projects").symlink_to(outside, target_is_directory=True)
    run(UninstallOptions(remove_projects=True, dry_run=False), home)
    assert not (home / "Projects").exists()
    assert (outside / "thesis" / "final.docx").read_text() == "important"


def test_refuses_dangerous_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_user_home = tmp_path / "u"
    fake_user_home.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_user_home))
    for bad in (fake_user_home, Path("/"), Path("/tmp")):
        with pytest.raises(ValueError):
            uninstall.plan(UninstallOptions(), home=bad)
        with pytest.raises(ValueError):
            uninstall.execute(UninstallOptions(dry_run=False), home=bad)
    assert fake_user_home.exists()


def test_paths_outside_the_studio_folder_are_reported_never_deleted(
    home: Path, config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    docs = tmp_path / "Documents"
    docs.mkdir()
    (docs / "report.txt").write_text("mine")
    monkeypatch.setenv("NIMLAB_PROJECTS_DIR", str(docs))
    plan = uninstall.plan(UninstallOptions(remove_projects=True), home=home, install=DEV)
    assert any(
        "NIMLAB_PROJECTS_DIR" in line and "never removed" in line for line in plan["outside"]
    )
    run(UninstallOptions(remove_projects=True, dry_run=False), home)
    assert (docs / "report.txt").read_text() == "mine"


def test_unrelated_files_next_to_the_folder_survive(
    home: Path, config_dir: Path, tmp_path: Path
) -> None:
    sibling = home.parent / "Documents"
    sibling.mkdir()
    (sibling / "a.txt").write_text("a")
    run(UninstallOptions(remove_projects=True, remove_atlas=True, dry_run=False), home)
    assert (sibling / "a.txt").exists()


def test_options_from_mapping_ignores_unknown_keys() -> None:
    opts = UninstallOptions.from_mapping({"remove_projects": 1, "bogus": True})
    assert opts.remove_projects and opts.remove_key and not opts.remove_app


# ------------------------------------------------------------------------------- the app itself
def test_detect_frozen_app_bundle(tmp_path: Path) -> None:
    exe = tmp_path / "NIM Lab Studio.app" / "Contents" / "MacOS" / "NIM Lab Studio"
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    info = uninstall.detect_install(frozen=True, executable=str(exe))
    assert info.kind == "frozen-app" and info.removable
    assert info.app_bundle == (tmp_path / "NIM Lab Studio.app").resolve()


def test_detect_editable_and_pip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(uninstall, "_direct_url_editable", lambda: True)
    editable = uninstall.detect_install(frozen=False)
    assert (
        editable.kind == "editable"
        and not editable.removable
        and "pip uninstall" in editable.instructions
    )
    monkeypatch.setattr(uninstall, "_direct_url_editable", lambda: False)
    assert uninstall.detect_install(frozen=False).kind == "pip"
    monkeypatch.setattr(uninstall, "_direct_url_editable", lambda: None)
    assert uninstall.detect_install(frozen=False).kind == "dev"


def test_frozen_app_is_moved_to_the_trash_after_the_process_exits(tmp_path: Path) -> None:
    bundle = tmp_path / "NIM Lab Studio.app"
    info = InstallInfo("frozen-app", app_bundle=bundle, location=bundle, removable=True)
    cmd = uninstall.build_removal_command(info, 4242)
    assert cmd is not None and cmd[:2] == ["/bin/sh", "-c"]
    script = cmd[2]
    assert "kill -0 4242" in script  # waits for the app to exit first
    assert "osascript" in script and "Finder" in script and "delete POSIX file" in script
    assert str(bundle) in script and script.index("kill -0") < script.index("osascript")


def test_pip_install_is_uninstalled_by_a_detached_helper() -> None:
    info = InstallInfo("pip", removable=True)
    cmd = uninstall.build_removal_command(info, 7)
    assert cmd is not None and f"{sys.executable} -m pip uninstall -y nimlab" in cmd[2]
    launched: list[tuple[list[str], dict[str, Any]]] = []

    def fake_popen(argv: list[str], **kw: Any) -> None:
        launched.append((argv, kw))

    assert uninstall.schedule_app_removal(info, 7, popen=fake_popen) == cmd
    assert launched[0][1]["start_new_session"] is True  # detached: survives our exit


@pytest.mark.parametrize("kind", ["editable", "dev", "unknown"])
def test_nothing_is_scheduled_for_editable_or_dev_installs(kind: str) -> None:
    info = InstallInfo(kind, removable=False, instructions="do it by hand")
    assert uninstall.build_removal_command(info, 1) is None
    assert uninstall.schedule_app_removal(info, 1, popen=lambda *a, **k: pytest.fail("ran")) is None


def test_removal_never_touches_the_environment(home: Path, config_dir: Path) -> None:
    before = dict(os.environ)
    run(UninstallOptions(dry_run=False), home)
    assert dict(os.environ) == before
