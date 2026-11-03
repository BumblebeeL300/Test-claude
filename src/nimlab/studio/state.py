"""Process-wide Studio state: settings, the shared NIM client, usage, runs and projects."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

from nimlab import __version__
from nimlab.core import NIMAuthError, NIMClient, NIMConfig, NIMError, UsageTracker
from nimlab.core.config import quiet_config

from . import credentials
from .projects import ProjectStore
from .runs import APP_IDS, RunManager
from .settings import StudioSettings
from .simulate import StudioClient

__all__ = ["APPS", "StudioState"]

_log = logging.getLogger("nimlab.studio")

#: Static presentation metadata for the eight apps.
APPS: dict[str, dict[str, str]] = {
    "council": {
        "name": "Council",
        "tagline": "Many models deliberate, one better answer",
        "color": "#FF3EA5",
        "glyph": "◎",
    },
    "atlas": {
        "name": "Atlas",
        "tagline": "Ask your documents, with citations you can check",
        "color": "#2DE2E6",
        "glyph": "◈",
    },
    "forge": {
        "name": "Forge",
        "tagline": "A test-driven agent that builds real projects",
        "color": "#C6FF3D",
        "glyph": "▲",
    },
    "arena": {
        "name": "Arena",
        "tagline": "Benchmark, rank and route across models",
        "color": "#FFA62B",
        "glyph": "◆",
    },
    "prism": {
        "name": "Prism",
        "tagline": "Put an image on the light table: describe, read, extract, rebuild",
        "color": "#B07CFF",
        "glyph": "△",
    },
    "sentinel": {
        "name": "Sentinel",
        "tagline": "Blast-test your models for leaks, injections and jailbreaks",
        "color": "#FF4D4D",
        "glyph": "⬢",
    },
    "loom": {
        "name": "Loom",
        "tagline": "Patch models into pipelines",
        "color": "#3DFFB4",
        "glyph": "≋",
    },
    "distill": {
        "name": "Distill",
        "tagline": "Refine model output into clean, deduplicated datasets",
        "color": "#FFC83D",
        "glyph": "⬡",
    },
}

VERIFY_TIMEOUT_S = 6.0


class StudioState:
    """Owns everything shared between requests. One instance per Studio app."""

    def __init__(self, settings: StudioSettings, *, client: Any | None = None) -> None:
        self.settings = settings
        self.usage = UsageTracker()
        self.runs = RunManager(
            settings.history_path,
            max_history=settings.max_history,
            max_memory_runs=settings.max_memory_runs,
        )
        self.projects = ProjectStore(settings.projects_dir)  # type: ignore[arg-type]
        self.network = "unknown"
        self._client: Any | None = client
        self._injected = client is not None
        self._retired: list[Any] = []
        self._atlas: Any | None = None
        self._atlas_for: Any | None = None
        #: Called (once) to quit the app after an uninstall; set by the launcher.
        self.exit_hook: Callable[[], None] | None = None

    # ------------------------------------------------------------------ client
    def base_config(self) -> NIMConfig:
        """Config from the environment + stored key (no 'no key' warning in the log)."""
        with quiet_config():
            return NIMConfig.from_env()

    def client(self) -> Any:
        """The shared client, created lazily and replaced when the key changes."""
        if self._client is None:
            client = NIMClient(self.base_config())
            client.usage = self.usage
            self._client = client
        return self._client

    @property
    def mock(self) -> bool:
        return bool(getattr(self.client(), "mock", False))

    def studio_client(self, *, council: bool = False) -> StudioClient:
        """The shared client wrapped for simulation (mock backend only) and reply previews."""
        base = self.client()
        return StudioClient(
            base,
            simulate=bool(getattr(base, "mock", False)),
            delay=self.settings.sim_delay,
            council=council,
        )

    def reset_client(self) -> None:
        """Forget the shared client (the key changed). In-flight runs keep the old one."""
        if self._client is not None and not self._injected:
            self._retired.append(self._client)
        if not self._injected:
            self._client = None
        self.close_atlas()

    def close_atlas(self) -> None:
        if self._atlas is not None:
            try:
                self._atlas.close()
            except Exception:  # pragma: no cover - best effort
                _log.debug("closing atlas failed", exc_info=True)
        self._atlas = None
        self._atlas_for = None

    def atlas(self) -> Any:
        """The Atlas engine for the current client (recreated when the client changes)."""
        from nimlab.atlas import Atlas

        base = self.client()
        if self._atlas is None or self._atlas_for is not base:
            self.close_atlas()
            path = self.settings.atlas_db
            assert path is not None
            path.parent.mkdir(parents=True, exist_ok=True)
            self._atlas = Atlas(self.studio_client(), path)
            self._atlas_for = base
        return self._atlas

    async def shutdown(self) -> None:
        await self.runs.shutdown()
        self.close_atlas()
        clients = [*self._retired, *([self._client] if self._client is not None else [])]
        for c in clients:
            aclose = getattr(c, "aclose", None)
            if aclose is not None and not self._injected:
                try:
                    await aclose()
                except Exception:  # pragma: no cover - best effort
                    _log.debug("closing client failed", exc_info=True)
        self._retired.clear()

    # ------------------------------------------------------------------ key
    def key_info(self) -> dict[str, Any]:
        key = self.base_config().api_key
        return {
            "set": bool(key),
            "source": credentials.key_source() if key else None,
            "masked": credentials.mask_key(key),
        }

    async def verify_key(self, key: str | None) -> dict[str, Any]:
        """Call ``list_models`` with ``key`` (6 s budget). Never raises."""
        if not key:
            return {
                "verified": False,
                "model_count": None,
                "error": "No API key is set. Paste an nvapi- key to use live models.",
            }
        cfg = self.base_config().model_copy(
            update={
                "api_key": key,
                "mock": False,
                "mock_reason": None,
                "max_retries": 0,
                "timeout": VERIFY_TIMEOUT_S,
            }
        )
        host = urlparse(cfg.base_url).hostname or cfg.base_url
        client = NIMClient(cfg)
        try:
            ids = await asyncio.wait_for(client.list_models(), timeout=VERIFY_TIMEOUT_S + 1)
        except NIMAuthError:
            self.network = "ok"
            return {
                "verified": False,
                "model_count": None,
                "error": "NVIDIA rejected this key (authentication failed). Check that it is "
                "a current nvapi- key from build.nvidia.com.",
            }
        except (TimeoutError, asyncio.TimeoutError):
            self.network = "blocked"
            return {
                "verified": False,
                "model_count": None,
                "error": f"Timed out after {VERIFY_TIMEOUT_S:.0f}s reaching {host}. The key was "
                "saved but could not be verified; check your network or proxy.",
            }
        except NIMError as exc:
            reachable = getattr(exc, "status_code", None) is not None
            self.network = "ok" if reachable else "blocked"
            if reachable:
                msg = f"{host} answered with an error: {exc}"
            else:
                msg = (
                    f"Could not reach {host} ({exc}). The key was saved but could not be "
                    "verified; check your network, VPN or proxy."
                )
            return {"verified": False, "model_count": None, "error": msg}
        finally:
            await client.aclose()
        self.network = "ok"
        return {"verified": True, "model_count": len(ids), "error": None}

    # ------------------------------------------------------------------ status
    def app_infos(self) -> list[dict[str, Any]]:
        no_key = self.settings.require_key and self.mock
        out = []
        for app_id in APP_IDS:
            meta = APPS[app_id]
            busy = bool(self.runs.active(app_id))
            out.append(
                {
                    "id": app_id,
                    **meta,
                    "status": "busy" if busy else ("needs-key" if no_key else "ready"),
                    "runs": self.runs.counts.get(app_id, 0),
                    "last_run_at": self.runs.last_run_at.get(app_id),
                }
            )
        return out

    def schedule_app_removal(self, info: Any) -> bool:
        from .uninstall import schedule_app_removal

        try:
            return schedule_app_removal(info) is not None
        except OSError:  # pragma: no cover - the helper could not be started
            _log.warning("could not start the app-removal helper", exc_info=True)
            return False

    def schedule_exit(self, delay: float = 1.5) -> None:
        """Quit shortly after the response is sent (window close + server stop)."""
        hook = self.exit_hook
        if not self.settings.allow_exit or hook is None:
            return
        timer = threading.Timer(delay, hook)
        timer.daemon = True
        timer.start()

    def models_info(self) -> dict[str, Any]:
        """Stale-default summary for the amber notice (never fetches)."""
        from nimlab.core import default_registry

        reg = default_registry()
        stale = reg.stale_defaults(fetch=False)
        return {
            "live": reg.snapshot(fetch=False) is not None,
            "stale": stale,
            "stale_count": len(stale),
        }

    def status(self) -> dict[str, Any]:
        cfg = self.base_config()
        return {
            "models": self.models_info(),
            "version": __version__,
            "mode": "mock" if self.mock else "live",
            "key": self.key_info(),
            "base_url": cfg.base_url,
            "network": self.network,
            "apps": self.app_infos(),
        }
