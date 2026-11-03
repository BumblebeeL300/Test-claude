"""Where the NVIDIA API key lives when it is not in the environment.

Backends, in order of preference:

1. the macOS Keychain (or any real OS keyring) through the optional ``keyring`` package,
   service ``nimlab``, account ``nvidia_api_key``;
2. ``<config dir>/credentials.json`` (``~/.config/nimlab`` or ``$NIMLAB_CONFIG_DIR``), written with
   mode 0600 through an atomic temp-file + rename.

``keyring`` is never required: when it is missing, only has the "fail" backend, or raises, the file
backend is used. Set ``NIMLAB_DISABLE_KEYRING=1`` to skip the keyring entirely (the test-suite
does).

The key is only ever returned by :func:`load_api_key`; everything that reaches a UI or a log goes
through :func:`mask_key`.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, NamedTuple

__all__ = [
    "ACCOUNT",
    "SERVICE",
    "KeySaveResult",
    "config_dir",
    "credentials_file",
    "delete_api_key",
    "key_source",
    "keyring_available",
    "load_api_key",
    "mask_key",
    "save_api_key",
    "stored_source",
    "validate_key_format",
]

SERVICE = "nimlab"
ACCOUNT = "nvidia_api_key"
FILE_NAME = "credentials.json"
KEY_PREFIX = "nvapi-"
MIN_KEY_LENGTH = 20
MAX_KEY_LENGTH = 256

_log = logging.getLogger("nimlab.credentials")


class KeySaveResult(NamedTuple):
    """Where a key was stored and, if its format looked unusual, a warning to show."""

    source: str  # "keychain" | "file"
    warning: str | None = None


# --------------------------------------------------------------------------- locations


def config_dir() -> Path:
    """``$NIMLAB_CONFIG_DIR`` or ``~/.config/nimlab`` (not created here)."""
    override = os.environ.get("NIMLAB_CONFIG_DIR", "").strip()
    return Path(override).expanduser() if override else Path.home() / ".config" / "nimlab"


def credentials_file() -> Path:
    return config_dir() / FILE_NAME


# --------------------------------------------------------------------------- formatting


def mask_key(key: str | None) -> str | None:
    """``nvapi-…a1b2``: the prefix and the last four characters only."""
    if not key:
        return None
    tail = key[-4:] if len(key) >= 12 else ""
    prefix = KEY_PREFIX if key.startswith(KEY_PREFIX) else ""
    return f"{prefix}…{tail}" if tail else f"{prefix}…"


def validate_key_format(key: str) -> str | None:
    """A warning when ``key`` does not look like an NVIDIA key, else ``None``.

    Raises ``ValueError`` for input that can never be a key (empty, whitespace, control chars).
    """
    if not key or not key.strip():
        raise ValueError("the API key is empty")
    if any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in key.strip()):
        raise ValueError("the API key contains whitespace or control characters")
    key = key.strip()
    problems: list[str] = []
    if not key.startswith(KEY_PREFIX):
        problems.append(f"does not start with {KEY_PREFIX!r}")
    if not MIN_KEY_LENGTH <= len(key) <= MAX_KEY_LENGTH:
        problems.append(f"has an unusual length ({len(key)} characters)")
    if problems:
        return "This key " + " and ".join(problems) + "; saving it anyway."
    return None


# --------------------------------------------------------------------------- keyring


def _keyring_disabled() -> bool:
    return os.environ.get("NIMLAB_DISABLE_KEYRING", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _keyring() -> Any | None:
    """The ``keyring`` module when a usable (non-fail) backend exists, else ``None``."""
    if _keyring_disabled():
        return None
    try:
        import keyring  # type: ignore[import-not-found,unused-ignore]
    except Exception:
        return None
    try:
        backend = keyring.get_keyring()
    except Exception:
        return None
    name = f"{type(backend).__module__}.{type(backend).__name__}".lower()
    if "fail" in name or "null" in name:
        return None
    priority = getattr(backend, "priority", 1)
    try:
        if float(priority) <= 0:
            return None
    except (TypeError, ValueError):
        pass
    return keyring


def keyring_available() -> bool:
    return _keyring() is not None


# --------------------------------------------------------------------------- file backend


def _read_file() -> str | None:
    try:
        data = json.loads(credentials_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = data.get(ACCOUNT) if isinstance(data, dict) else None
    return value.strip() if isinstance(value, str) and value.strip() else None


def _write_file(key: str) -> None:
    directory = config_dir()
    directory.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        os.chmod(directory, 0o700)
    payload = json.dumps({ACCOUNT: key}, indent=2) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=".credentials-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, credentials_file())
        with contextlib.suppress(OSError):
            os.chmod(credentials_file(), 0o600)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def _delete_file() -> bool:
    path = credentials_file()
    existed = path.exists()
    with contextlib.suppress(OSError):
        path.unlink()
    return existed


# --------------------------------------------------------------------------- public API


def save_api_key(key: str) -> KeySaveResult:
    """Store ``key`` (Keychain first, file as the fallback) and say where it went.

    Raises ``ValueError`` for an empty/garbled key. An odd-looking key (no ``nvapi-`` prefix,
    strange length) is accepted with a warning in the result and in the log.
    """
    warning = validate_key_format(key)
    key = key.strip()
    if warning:
        _log.warning("%s", warning)
    kr = _keyring()
    if kr is not None:
        try:
            kr.set_password(SERVICE, ACCOUNT, key)
        except Exception as exc:
            _log.warning(
                "keyring write failed (%s); using the credentials file", type(exc).__name__
            )
        else:
            _delete_file()  # never leave a stale copy next to the Keychain entry
            return KeySaveResult("keychain", warning)
    _write_file(key)
    return KeySaveResult("file", warning)


def _load_keyring() -> str | None:
    kr = _keyring()
    if kr is None:
        return None
    try:
        value = kr.get_password(SERVICE, ACCOUNT)
    except Exception:
        return None
    return value.strip() if isinstance(value, str) and value.strip() else None


def load_api_key() -> str | None:
    """The stored key (Keychain, then file), or ``None``. Never looks at the environment."""
    return _load_keyring() or _read_file()


def stored_source() -> str | None:
    """``"keychain"`` / ``"file"`` for the stored key, ``None`` when nothing is stored."""
    if _load_keyring():
        return "keychain"
    return "file" if _read_file() else None


def delete_api_key() -> bool:
    """Remove the key from every backend; ``True`` if something was removed."""
    removed = False
    kr = _keyring()
    if kr is not None:
        try:
            if kr.get_password(SERVICE, ACCOUNT) is not None:
                kr.delete_password(SERVICE, ACCOUNT)
                removed = True
        except Exception as exc:
            _log.warning("keyring delete failed (%s)", type(exc).__name__)
    return _delete_file() or removed


def _dotenv_value(name: str) -> str | None:
    if os.environ.get("NIMLAB_NO_DOTENV"):
        return None
    try:
        lines = (Path.cwd() / ".env").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for raw in lines:
        line = raw.strip()
        if line.startswith("export "):
            line = line[len("export ") :]
        key, sep, value = line.partition("=")
        if sep and key.strip() == name:
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            return value.strip() or None
    return None


def key_source() -> str | None:
    """Where the key NIM Lab would use comes from.

    ``"env"`` (``NVIDIA_API_KEY``), ``"dotenv"`` (``./.env``), ``"keychain"``, ``"file"`` or
    ``None`` when there is no key. Same precedence as :meth:`NIMConfig.from_env`.
    """
    from nimlab.core.config import _env

    value = _env("NVIDIA_API_KEY")  # also loads ./.env into the environment
    if value:
        return "dotenv" if _dotenv_value("NVIDIA_API_KEY") == value else "env"
    return stored_source()
