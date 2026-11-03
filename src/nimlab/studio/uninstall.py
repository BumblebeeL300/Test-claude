"""Uninstall NIM Lab: plan and remove exactly the files it created, nothing else.

Known locations only (all resolved before use):

* the API key: macOS Keychain entry ``nimlab`` and ``<config dir>/credentials.json``
* settings, history and caches: everything directly inside the Studio folder (``~/NIMLab`` or
  ``$NIMLAB_STUDIO_HOME``) except the projects, Loom pipelines and Atlas database below, plus the
  config folder (``~/.config/nimlab``)
* Forge projects (``<home>/Projects``) and saved Loom pipelines (``<home>/loom``)
* the Atlas database (``<home>/atlas.db`` and its ``-wal`` / ``-shm`` files)
* the application itself: the ``.app`` bundle (frozen build) or the pip package

Paths outside the Studio folder (for example ``NIMLAB_PROJECTS_DIR`` pointing at ``~/Documents``)
are reported but never deleted. Symlinks are removed as links and never followed, and a target
whose resolved path escapes its root is refused.
"""

from __future__ import annotations

import contextlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

from . import credentials
from .frozen import is_frozen

__all__ = [
    "InstallInfo",
    "UninstallOptions",
    "build_removal_command",
    "detect_install",
    "execute",
    "plan",
    "schedule_app_removal",
]

PROJECTS_DIRNAME = "Projects"
LOOM_DIRNAME = "loom"
ATLAS_NAMES = ("atlas.db", "atlas.db-wal", "atlas.db-shm", "atlas.db-journal")


@dataclass
class UninstallOptions:
    remove_key: bool = True
    remove_data: bool = True
    remove_projects: bool = False
    remove_atlas: bool = False
    remove_app: bool = False
    dry_run: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> UninstallOptions:
        base = cls()
        return cls(**{k: bool(data.get(k, getattr(base, k))) for k in base.__dict__})


@dataclass
class InstallInfo:
    """How NIM Lab is installed, which decides how the app itself is removed."""

    kind: str  # "frozen-app" | "pip" | "editable" | "dev" | "unknown"
    app_bundle: Path | None = None
    location: Path | None = None
    removable: bool = False
    instructions: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "app_bundle": str(self.app_bundle) if self.app_bundle else None,
            "location": str(self.location) if self.location else None,
            "removable": self.removable,
            "instructions": self.instructions,
        }


@dataclass
class Item:
    id: str
    label: str
    paths: list[Path] = field(default_factory=list)
    count: int = 0
    bytes: int = 0
    selected: bool = False
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "paths": [str(p) for p in self.paths],
            "count": self.count,
            "bytes": self.bytes,
            "selected": self.selected,
            "note": self.note,
        }


# --------------------------------------------------------------------------- safety helpers
def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (ValueError, OSError):
        return False
    return True


def validate_home(home: Path) -> Path:
    """The Studio folder, refusing anything that is obviously not a dedicated app folder."""
    resolved = home.expanduser().resolve()
    user_home = Path.home().resolve()
    if resolved in (Path(resolved.anchor), user_home) or len(resolved.parts) < 3:
        raise ValueError(f"refusing to treat {resolved} as the NIM Lab folder")
    return resolved


def _size(path: Path) -> tuple[int, int]:
    """(file count, bytes) of ``path`` without following symlinks."""
    if path.is_symlink() or path.is_file():
        try:
            return 1, path.lstat().st_size
        except OSError:
            return 1, 0
    files = total = 0
    for root, dirs, names in os.walk(path, followlinks=False):
        for name in [*names, *[d for d in dirs if (Path(root) / d).is_symlink()]]:
            files += 1
            with contextlib.suppress(OSError):
                total += (Path(root) / name).lstat().st_size
    return files, total


def _entry_count(path: Path) -> int:
    try:
        return sum(1 for _ in path.iterdir())
    except OSError:
        return 0


# --------------------------------------------------------------------------- install detection
def _direct_url_editable() -> bool | None:
    try:
        dist = metadata.distribution("nimlab")
        raw = dist.read_text("direct_url.json")
    except (metadata.PackageNotFoundError, OSError):
        return None
    if not raw:
        return False
    try:
        data = json.loads(raw)
    except ValueError:
        return False
    return bool(data.get("dir_info", {}).get("editable"))


def detect_install(*, frozen: bool | None = None, executable: str | None = None) -> InstallInfo:
    """Frozen ``.app`` bundle, regular pip install, editable/dev checkout, or unknown."""
    if is_frozen() if frozen is None else frozen:
        exe = Path(executable or sys.executable).resolve()
        bundle = next((p for p in [exe, *exe.parents] if p.suffix == ".app"), None)
        if bundle is not None:
            return InstallInfo(
                "frozen-app",
                app_bundle=bundle,
                location=bundle,
                removable=True,
                instructions=f"Move {bundle} to the Trash.",
            )
        return InstallInfo("unknown", instructions="Delete the NIM Lab application manually.")
    package = Path(__file__).resolve().parents[1]
    editable = _direct_url_editable()
    if editable:
        return InstallInfo(
            "editable",
            location=package.parent,
            instructions=(
                "This is an editable (development) install. Remove it with "
                f"`{sys.executable} -m pip uninstall nimlab` and delete the checkout "
                f"({package.parent.parent}) yourself if you no longer want it."
            ),
        )
    if editable is None:
        return InstallInfo(
            "dev",
            location=package.parent,
            instructions="Running from a source checkout: delete the checkout folder to remove it.",
        )
    return InstallInfo(
        "pip",
        location=package.parent,
        removable=True,
        instructions=f"{sys.executable} -m pip uninstall -y nimlab",
    )


def build_removal_command(info: InstallInfo, pid: int) -> list[str] | None:
    """A shell command that waits for process ``pid`` to exit, then removes the app."""
    if not info.removable:
        return None
    if info.kind == "frozen-app" and info.app_bundle is not None:
        script = (
            f'tell application "Finder" to delete POSIX file {json.dumps(str(info.app_bundle))}'
        )
        action = f"osascript -e {shlex.quote(script)}"
    elif info.kind == "pip":
        action = shlex.join([sys.executable, "-m", "pip", "uninstall", "-y", "nimlab"])
    else:
        return None
    wait = f"while kill -0 {int(pid)} 2>/dev/null; do sleep 0.5; done; sleep 1"
    return ["/bin/sh", "-c", f"{wait}; {action}"]


def schedule_app_removal(
    info: InstallInfo,
    pid: int | None = None,
    *,
    popen: Callable[..., Any] = subprocess.Popen,
) -> list[str] | None:
    """Start the detached helper (it outlives this process). Returns its argv, or ``None``."""
    cmd = build_removal_command(info, pid if pid is not None else os.getpid())
    if cmd is None:
        return None
    popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )
    return cmd


# --------------------------------------------------------------------------- planning
def _atlas_paths(home: Path) -> list[Path]:
    return [p for n in ATLAS_NAMES if (p := home / n).exists() or (p := home / n).is_symlink()]


def _preserved(home: Path, opts: UninstallOptions) -> set[str]:
    keep: set[str] = set()
    if not opts.remove_projects:
        keep |= {PROJECTS_DIRNAME, LOOM_DIRNAME}
    if not opts.remove_atlas:
        keep |= set(ATLAS_NAMES)
    return keep


def plan(
    opts: UninstallOptions,
    *,
    home: Path,
    config_dir: Path | None = None,
    install: InstallInfo | None = None,
) -> dict[str, Any]:
    """What would be removed: items with paths, counts and sizes. Touches nothing."""
    root = validate_home(home)
    cfg = (config_dir if config_dir is not None else credentials.config_dir()).expanduser()
    info = install or detect_install()
    items: list[Item] = []

    source = credentials.stored_source()
    key = Item(
        "key",
        "API key (Keychain and credentials file)",
        selected=opts.remove_key,
        count=1 if source else 0,
        note=f"stored in: {source}" if source else "no stored key",
    )
    cred_file = credentials.credentials_file()
    if cred_file.exists():
        key.paths.append(cred_file)
        key.bytes = cred_file.stat().st_size
    items.append(key)

    keep = _preserved(root, opts)
    data = Item("data", "Settings, history and caches", selected=opts.remove_data)
    if root.is_dir():
        for entry in sorted(root.iterdir()):
            if entry.name in keep or entry.name in {PROJECTS_DIRNAME, LOOM_DIRNAME, *ATLAS_NAMES}:
                continue
            data.paths.append(entry)
            n, b = _size(entry)
            data.count += n
            data.bytes += b
    if cfg.is_dir() and _is_within(cfg, cfg):
        extra = [p for p in sorted(cfg.iterdir()) if p != cred_file]
        for entry in extra:
            data.paths.append(entry)
            n, b = _size(entry)
            data.count += n
            data.bytes += b
    data.note = f"{root} and {cfg}"
    items.append(data)

    projects = Item(
        "projects", "Forge projects and saved Loom pipelines", selected=opts.remove_projects
    )
    for name in (PROJECTS_DIRNAME, LOOM_DIRNAME):
        p = root / name
        if p.exists() or p.is_symlink():
            projects.paths.append(p)
            n, b = _size(p)
            projects.bytes += b
            projects.count += _entry_count(p) if p.is_dir() and not p.is_symlink() else 1
    projects.note = "projects and pipelines are your work: off by default"
    items.append(projects)

    atlas = Item("atlas", "Atlas database", selected=opts.remove_atlas)
    for p in _atlas_paths(root):
        atlas.paths.append(p)
        n, b = _size(p)
        atlas.bytes += b
        atlas.count = 1
    items.append(atlas)

    app = Item(
        "app",
        "The application itself",
        selected=opts.remove_app,
        note=info.instructions,
    )
    if info.app_bundle:
        app.paths.append(info.app_bundle)
        app.bytes = _size(info.app_bundle)[1] if info.app_bundle.exists() else 0
    items.append(app)

    outside: list[str] = []
    for env in ("NIMLAB_PROJECTS_DIR", "NIMLAB_ATLAS_DB", "NIMLAB_LOOM_DIR"):
        value = os.environ.get(env, "").strip()
        if value and not _is_within(Path(value).expanduser(), root):
            outside.append(f"{env}={value} (outside {root}; never removed)")

    return {
        "home": str(root),
        "config_dir": str(cfg),
        "install": info.as_dict(),
        "items": [i.as_dict() for i in items],
        "outside": outside,
        "options": {k: getattr(opts, k) for k in opts.__dict__},
    }


# --------------------------------------------------------------------------- execution
def _remove(path: Path, root: Path, removed: list[str], refused: list[str]) -> None:
    """Delete one known path; symlinks go as links, escaping targets are refused."""
    if path.is_symlink():
        path.unlink()
        removed.append(str(path))
        return
    if not path.exists():
        return
    if not _is_within(path, root):
        refused.append(f"{path} (outside {root})")
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    removed.append(str(path))


def execute(
    opts: UninstallOptions,
    *,
    home: Path,
    config_dir: Path | None = None,
    install: InstallInfo | None = None,
    delete_key: Callable[[], bool] = credentials.delete_api_key,
) -> dict[str, Any]:
    """Remove the selected items. Returns the plan plus ``removed`` / ``refused`` / ``errors``."""
    root = validate_home(home)
    cfg = (config_dir if config_dir is not None else credentials.config_dir()).expanduser()
    report = plan(opts, home=root, config_dir=cfg, install=install)
    removed: list[str] = []
    refused: list[str] = []
    errors: list[str] = []

    def guarded(path: Path, base: Path) -> None:
        try:
            _remove(path, base, removed, refused)
        except OSError as exc:
            errors.append(f"{path}: {exc}")

    by_id = {i["id"]: i for i in report["items"]}
    if opts.remove_key:
        try:
            if delete_key():
                removed.append("API key")
        except Exception as exc:
            errors.append(f"API key: {type(exc).__name__}: {exc}")
        if cfg.is_dir():
            cred = credentials.credentials_file()
            if cred.exists() and _is_within(cfg, cfg):
                guarded(cred, cfg)
    if opts.remove_data:
        for p in by_id["data"]["paths"]:
            path = Path(p)
            base = cfg if _is_within(path, cfg) and not _is_within(path, root) else root
            guarded(path, base)
    if opts.remove_projects:
        for p in by_id["projects"]["paths"]:
            guarded(Path(p), root)
    if opts.remove_atlas:
        for p in by_id["atlas"]["paths"]:
            guarded(Path(p), root)
    # tidy up the now-empty folders (never non-empty ones)
    for folder in (root, cfg):
        with contextlib.suppress(OSError):
            if folder.is_dir() and not folder.is_symlink() and not any(folder.iterdir()):
                folder.rmdir()
                removed.append(str(folder))
    report.update(removed=removed, refused=refused, errors=errors)
    return report


def human_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{n} B"  # pragma: no cover


def summarize(report: Mapping[str, Any], selected_only: bool = True) -> Iterable[str]:
    for item in report["items"]:
        if selected_only and not item["selected"]:
            continue
        yield f"{item['label']}: {item['count']} item(s), {human_size(item['bytes'])}"
