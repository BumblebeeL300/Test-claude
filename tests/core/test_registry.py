from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from nimlab.core import ModelInfo, ModelRegistry, Role, default_registry
from nimlab.core.registry import CURATED_MODELS, DEFAULTS

REQUIRED_IDS = [
    "meta/llama-3.3-70b-instruct",
    "meta/llama-3.1-405b-instruct",
    "nvidia/llama-3.3-nemotron-super-49b-v1",
    "nvidia/llama-3.1-nemotron-ultra-253b-v1",
    "deepseek-ai/deepseek-r1",
    "qwen/qwen2.5-coder-32b-instruct",
    "qwen/qwq-32b",
    "mistralai/mixtral-8x22b-instruct-v0.1",
    "microsoft/phi-4-mini-instruct",
    "meta/llama-3.1-8b-instruct",
    "meta/llama-3.2-90b-vision-instruct",
    "nvidia/llama-3.2-nv-embedqa-1b-v2",
    "nvidia/nv-embedqa-e5-v5",
    "nvidia/llama-3.2-nv-rerankqa-1b-v2",
    "nvidia/llama-3.1-nemotron-70b-reward",
    "nvidia/llama-3.1-nemoguard-8b-content-safety",
]


def test_role_is_str_enum() -> None:
    assert Role.CODER == "coder"
    assert {r.name for r in Role} == {
        "CHAT", "REASONING", "CODER", "VISION", "EMBED", "RERANK", "REWARD", "SAFETY", "FAST",
    }  # fmt: skip


def test_required_models_present() -> None:
    reg = default_registry()
    for model_id in REQUIRED_IDS:
        assert model_id in reg, model_id
        info = reg.get(model_id)
        assert info.id == model_id and info.roles and info.context_window > 0
        assert 1 <= info.cost_tier <= 5


def test_ids_unique() -> None:
    ids = [m.id for m in CURATED_MODELS]
    assert len(ids) == len(set(ids))


def test_every_role_has_a_catalog_default() -> None:
    reg = default_registry()
    for role in Role:
        picked = reg.pick(role)
        assert picked in reg
        assert role in reg.get(picked).roles
        assert picked == DEFAULTS[role]


def test_get_unknown_raises_and_find_returns_none() -> None:
    reg = default_registry()
    with pytest.raises(KeyError):
        reg.get("nope/model")
    assert reg.find("nope/model") is None


def test_by_role_default_first_then_cost() -> None:
    reg = default_registry()
    coders = reg.by_role(Role.CODER)
    assert coders[0].id == reg.pick(Role.CODER)
    assert all(Role.CODER in m.roles for m in coders)
    chat = reg.by_role("chat")  # str accepted
    assert chat[0].id == "meta/llama-3.3-70b-instruct"
    rest = chat[1:]
    assert [m.cost_tier for m in rest] == sorted(m.cost_tier for m in rest)


def test_by_role_moves_override_to_front(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NIMLAB_MODEL_CHAT", "meta/llama-3.1-405b-instruct")
    assert default_registry().by_role(Role.CHAT)[0].id == "meta/llama-3.1-405b-instruct"


def test_pick_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NIMLAB_MODEL_CODER", "acme/custom-coder")
    reg = default_registry()
    assert reg.pick(Role.CODER) == "acme/custom-coder"  # not in the catalog: still allowed
    assert "acme/custom-coder" not in reg
    assert reg.pick(Role.CHAT) == "meta/llama-3.3-70b-instruct"  # other roles unaffected


def test_pick_accepts_string_roles() -> None:
    reg = default_registry()
    assert reg.pick("coder") == reg.pick("CODER") == reg.pick(Role.CODER)
    with pytest.raises(ValueError, match="valid roles"):
        reg.pick("telepathy")


def test_toml_override(tmp_path: Path) -> None:
    toml = tmp_path / "nimlab.toml"
    toml.write_text('[models]\ncoder = "toml/coder"\nREASONING = "toml/reasoner"\n')
    reg = ModelRegistry(config_path=toml)
    assert reg.pick(Role.CODER) == "toml/coder"
    assert reg.pick(Role.REASONING) == "toml/reasoner"
    assert reg.pick(Role.CHAT) == "meta/llama-3.3-70b-instruct"


def test_toml_discovered_in_cwd() -> None:
    Path("nimlab.toml").write_text('[models]\nfast = "toml/fast"\n')  # cwd is a tmp dir
    assert default_registry().pick(Role.FAST) == "toml/fast"


def test_toml_path_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    custom = tmp_path / "elsewhere.toml"
    custom.write_text('[models]\nchat = "elsewhere/chat"\n')
    monkeypatch.setenv("NIMLAB_CONFIG", str(custom))
    assert default_registry().pick(Role.CHAT) == "elsewhere/chat"


def test_env_beats_toml(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    toml = tmp_path / "nimlab.toml"
    toml.write_text('[models]\ncoder = "toml/coder"\n')
    monkeypatch.setenv("NIMLAB_MODEL_CODER", "env/coder")
    assert ModelRegistry(config_path=toml).pick(Role.CODER) == "env/coder"


def test_bad_toml_is_ignored(tmp_path: Path) -> None:
    toml = tmp_path / "nimlab.toml"
    toml.write_text("this is [not valid toml")
    assert ModelRegistry(config_path=toml).pick(Role.CODER) == DEFAULTS[Role.CODER]


def test_explicit_env_mapping() -> None:
    reg = ModelRegistry(env={"NIMLAB_MODEL_VISION": "x/vision"})
    assert reg.pick(Role.VISION) == "x/vision"


def test_default_registry_is_singleton() -> None:
    assert default_registry() is default_registry()


def test_register_custom_model() -> None:
    reg = ModelRegistry()
    info = ModelInfo(
        id="acme/coder-9b", roles=frozenset({Role.CODER}), context_window=8192, cost_tier=1
    )
    reg.register(info, default_for="coder")
    assert reg.get("acme/coder-9b") is info
    assert reg.pick(Role.CODER) == "acme/coder-9b"


def test_missing_role_default() -> None:
    reg = ModelRegistry(models=[], defaults={})
    with pytest.raises(KeyError):
        reg.pick(Role.CHAT)
    assert reg.by_role(Role.CHAT) == []


@pytest.mark.parametrize("tier", [0, 6])
def test_cost_tier_validated(tier: int) -> None:
    with pytest.raises(ValidationError):
        ModelInfo(id="x", roles=frozenset({Role.CHAT}), context_window=1, cost_tier=tier)


def test_iteration_and_len() -> None:
    reg = default_registry()
    assert len(reg) == len(list(reg)) == len(reg.ids()) == len(reg.all())


def test_live_snapshot_helpers(tmp_path: Path) -> None:
    snap = tmp_path / "live.txt"
    snap.write_text("meta/llama-3.3-70b-instruct\n\n# comment\nsome/other\n")
    live = ModelRegistry.load_live_ids(snap)
    assert live == {"meta/llama-3.3-70b-instruct", "some/other"}
    missing = default_registry().missing_from(live)
    assert "meta/llama-3.3-70b-instruct" not in missing and "qwen/qwq-32b" in missing
    assert ModelRegistry.load_live_ids(tmp_path / "absent.txt") == set()
