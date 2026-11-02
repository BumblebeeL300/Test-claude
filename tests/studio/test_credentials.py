"""The credential store and its place in NIMConfig's key precedence."""

from __future__ import annotations

import json
import os
import stat
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from nimlab.core import NIMConfig
from nimlab.studio import credentials as cred

KEY = "nvapi-" + "a" * 60 + "a1b2"


@pytest.fixture
def config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "cfg"
    monkeypatch.setenv("NIMLAB_CONFIG_DIR", str(path))
    return path


class TestFileBackend:
    def test_roundtrip_and_delete(self, config_dir: Path) -> None:
        assert cred.load_api_key() is None
        result = cred.save_api_key(KEY)
        assert result == cred.KeySaveResult("file", None)
        assert cred.load_api_key() == KEY
        assert cred.stored_source() == "file"
        assert cred.delete_api_key() is True
        assert cred.load_api_key() is None
        assert cred.delete_api_key() is False

    def test_permissions_and_no_temp_files(self, config_dir: Path) -> None:
        cred.save_api_key(KEY)
        cred.save_api_key(KEY[:-4] + "zzzz")  # overwrite goes through the same atomic path
        file = config_dir / "credentials.json"
        assert stat.S_IMODE(file.stat().st_mode) == 0o600
        assert stat.S_IMODE(config_dir.stat().st_mode) == 0o700
        assert [p.name for p in config_dir.iterdir()] == ["credentials.json"]
        assert json.loads(file.read_text())["nvidia_api_key"].endswith("zzzz")

    def test_default_location_is_dot_config(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("NIMLAB_CONFIG_DIR")
        monkeypatch.setenv("HOME", str(tmp_path))
        assert cred.config_dir() == tmp_path / ".config" / "nimlab"

    def test_corrupt_file_is_ignored(self, config_dir: Path) -> None:
        config_dir.mkdir(parents=True)
        (config_dir / "credentials.json").write_text("{not json")
        assert cred.load_api_key() is None
        assert cred.stored_source() is None

    def test_whitespace_is_trimmed(self, config_dir: Path) -> None:
        cred.save_api_key(f"  {KEY}\n")
        assert cred.load_api_key() == KEY


class TestValidation:
    def test_good_key_has_no_warning(self) -> None:
        assert cred.validate_key_format(KEY) is None

    @pytest.mark.parametrize("bad", ["", "   ", "nvapi-with space-inside-it-1234", "nvapi-\x00x"])
    def test_garbage_is_rejected(self, bad: str, config_dir: Path) -> None:
        with pytest.raises(ValueError):
            cred.save_api_key(bad)
        assert cred.load_api_key() is None

    def test_odd_keys_warn_but_are_saved(self, config_dir: Path) -> None:
        result = cred.save_api_key("sk-not-an-nvidia-key-but-long-enough")
        assert result.source == "file"
        assert result.warning and "nvapi-" in result.warning
        assert cred.load_api_key() == "sk-not-an-nvidia-key-but-long-enough"
        assert "unusual length" in (cred.validate_key_format("nvapi-short") or "")

    def test_mask(self) -> None:
        assert cred.mask_key(KEY) == "nvapi-…a1b2"
        assert cred.mask_key(None) is None
        assert cred.mask_key("") is None
        assert "a" * 10 not in (cred.mask_key(KEY) or "")


def fake_keyring(
    store: dict[str, str], *, backend: str = "Keyring", fail: bool = False
) -> types.ModuleType:
    mod = types.ModuleType("keyring")

    class Backend:
        priority = 5

    Backend.__name__ = backend
    Backend.__module__ = (
        "keyring.backends.macOS" if backend == "Keyring" else "keyring.backends.fail"
    )
    if backend != "Keyring":
        Backend.priority = 0  # type: ignore[assignment]

    def set_password(service: str, account: str, value: str) -> None:
        if fail:
            raise RuntimeError("locked")
        store[f"{service}/{account}"] = value

    def get_password(service: str, account: str) -> str | None:
        if fail:
            raise RuntimeError("locked")
        return store.get(f"{service}/{account}")

    def delete_password(service: str, account: str) -> None:
        store.pop(f"{service}/{account}")

    mod.get_keyring = lambda: Backend()  # type: ignore[attr-defined]
    mod.set_password = set_password  # type: ignore[attr-defined]
    mod.get_password = get_password  # type: ignore[attr-defined]
    mod.delete_password = delete_password  # type: ignore[attr-defined]
    return mod


class TestKeyring:
    def test_disabled_flag_skips_keyring(
        self, config_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store: dict[str, str] = {}
        monkeypatch.setitem(sys.modules, "keyring", fake_keyring(store))
        assert cred.keyring_available() is False  # NIMLAB_DISABLE_KEYRING=1 from conftest
        assert cred.save_api_key(KEY).source == "file"
        assert store == {}

    def test_keychain_is_preferred_and_file_removed(
        self, config_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store: dict[str, str] = {}
        monkeypatch.delenv("NIMLAB_DISABLE_KEYRING")
        monkeypatch.setitem(sys.modules, "keyring", fake_keyring(store))
        cred.save_api_key("nvapi-" + "f" * 40)  # starts in the file backend? no: keychain
        assert store == {"nimlab/nvidia_api_key": "nvapi-" + "f" * 40}
        assert cred.stored_source() == "keychain"
        assert cred.load_api_key() == "nvapi-" + "f" * 40
        assert not (config_dir / "credentials.json").exists()
        assert cred.delete_api_key() is True
        assert store == {}

    def test_fail_backend_falls_back_to_file(
        self, config_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("NIMLAB_DISABLE_KEYRING")
        monkeypatch.setitem(sys.modules, "keyring", fake_keyring({}, backend="Keyring_fail"))
        # the 'fail' backend has priority 0 and a fail module name
        assert cred.keyring_available() is False
        assert cred.save_api_key(KEY).source == "file"

    def test_erroring_keychain_falls_back_to_file(
        self, config_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("NIMLAB_DISABLE_KEYRING")
        monkeypatch.setitem(sys.modules, "keyring", fake_keyring({}, fail=True))
        assert cred.save_api_key(KEY).source == "file"
        assert cred.load_api_key() == KEY

    def test_missing_keyring_package(
        self, config_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("NIMLAB_DISABLE_KEYRING")
        monkeypatch.setitem(
            sys.modules, "keyring", None
        )  # makes `import keyring` raise ImportError
        assert cred.keyring_available() is False
        assert cred.save_api_key(KEY).source == "file"


class TestPrecedence:
    """explicit arg > NVIDIA_API_KEY > ./.env > stored credential."""

    def test_stored_key_is_used_when_nothing_else_is_set(self, config_dir: Path) -> None:
        cred.save_api_key(KEY)
        cfg = NIMConfig.from_env()
        assert cfg.api_key == KEY
        assert cfg.mock is False and cfg.mock_reason is None
        assert cred.key_source() == "file"

    def test_no_key_means_mock(self, config_dir: Path) -> None:
        cfg = NIMConfig.from_env()
        assert cfg.api_key is None and cfg.mock is True and cfg.mock_reason == "no_api_key"
        assert cred.key_source() is None

    def test_env_beats_stored(self, config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cred.save_api_key(KEY)
        monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-from-env")
        assert NIMConfig.from_env().api_key == "nvapi-from-env"
        assert cred.key_source() == "env"

    def test_explicit_beats_everything(
        self, config_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cred.save_api_key(KEY)
        monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-from-env")
        assert NIMConfig.from_env(api_key="nvapi-explicit").api_key == "nvapi-explicit"

    def test_dotenv_beats_stored(
        self, config_dir: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from nimlab.core import config as config_module

        cred.save_api_key(KEY)
        (tmp_path / ".env").write_text("NVIDIA_API_KEY=nvapi-from-dotenv\n")
        monkeypatch.delenv("NIMLAB_NO_DOTENV")
        config_module._load_dotenv.cache_clear()
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        try:
            assert NIMConfig.from_env().api_key == "nvapi-from-dotenv"
            assert cred.key_source() == "dotenv"
        finally:
            os.environ.pop("NVIDIA_API_KEY", None)
            config_module._load_dotenv.cache_clear()

    def test_nimlab_mock_still_forces_mock(
        self, config_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cred.save_api_key(KEY)
        monkeypatch.setenv("NIMLAB_MOCK", "1")
        cfg = NIMConfig.from_env()
        assert cfg.api_key == KEY and cfg.mock is True and cfg.mock_reason == "explicit"

    def test_broken_store_never_breaks_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom() -> Any:
            raise RuntimeError("keychain exploded")

        monkeypatch.setattr(cred, "load_api_key", boom)
        cfg = NIMConfig.from_env()
        assert cfg.api_key is None and cfg.mock is True

    def test_test_isolation_does_not_see_real_home(self) -> None:
        # conftest points NIMLAB_CONFIG_DIR at a tmp dir and disables the keyring for every test
        assert "nimlab-config" in str(cred.config_dir())
        assert cred.keyring_available() is False
