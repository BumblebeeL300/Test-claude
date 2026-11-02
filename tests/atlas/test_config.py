"""AtlasConfig validation, model resolution and small helpers."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nimlab.atlas import AtlasConfig, AtlasError, ModelSet
from nimlab.atlas.compat import is_mock, usage_totals
from nimlab.core import NIMClient, NIMConfig

from .conftest import FakeClient


def test_defaults_are_sensible():
    cfg = AtlasConfig()
    assert (cfg.k, cfg.rrf_k, cfg.mmr_lambda) == (6, 60, 0.7)
    assert cfg.chunk_overlap < cfg.chunk_min_chars < cfg.chunk_max_chars
    assert cfg.bm25_k1 == 1.5 and cfg.bm25_b == 0.75 and cfg.hyde is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chunk_overlap": 1200},
        {"chunk_min_chars": 1200},
        {"mmr_lambda": 1.5},
        {"bm25_b": -0.1},
        {"k": 0},
        {"unknown_option": 1},
    ],
)
def test_invalid_settings_are_rejected(kwargs: dict):
    with pytest.raises(ValidationError):
        AtlasConfig(**kwargs)


def test_chunk_signature_changes_with_anything_that_alters_chunks_or_vectors():
    base = AtlasConfig().chunk_signature("m1")
    assert base == AtlasConfig().chunk_signature("m1")
    assert base != AtlasConfig().chunk_signature("m2")
    assert base != AtlasConfig(chunk_max_chars=800, chunk_overlap=100).chunk_signature("m1")
    assert base == AtlasConfig(k=3, mmr_lambda=0.2).chunk_signature(
        "m1"
    )  # retrieval knobs do not matter


def test_model_set_uses_registry_defaults_and_overrides(monkeypatch: pytest.MonkeyPatch):
    models = ModelSet.resolve(AtlasConfig())
    assert models.embed == "nvidia/llama-3.2-nv-embedqa-1b-v2"
    assert models.rerank == "nvidia/llama-3.2-nv-rerankqa-1b-v2"
    assert models.safety == "nvidia/llama-3.1-nemoguard-8b-content-safety"
    assert models.fast != models.chat
    monkeypatch.setenv("NIMLAB_MODEL_CHAT", "env/chat-model")
    assert ModelSet.resolve(AtlasConfig()).chat == "env/chat-model"
    overridden = ModelSet.resolve(AtlasConfig(models={"chat": "cfg/chat", "fast": "cfg/fast"}))
    assert (overridden.chat, overridden.fast) == ("cfg/chat", "cfg/fast")


async def test_is_mock_and_usage_totals_follow_the_client():
    async with NIMClient(NIMConfig(mock=True)) as client:
        assert is_mock(client) is True
        assert usage_totals(client) == (0, 0)
        await client.embed("m", ["hello"])
        calls, tokens = usage_totals(client)
        assert calls == 1 and tokens > 0
    assert is_mock(FakeClient()) is False and is_mock(FakeClient(mock=True)) is True
    assert is_mock(object()) is False and usage_totals(object()) == (0, 0)


async def test_atlas_rejects_bad_config_through_the_api(db_path):
    from nimlab.atlas import Atlas

    async with NIMClient(NIMConfig(mock=True)) as client:
        atlas = Atlas(client, db_path)
        with pytest.raises(AtlasError, match="unknown collection"):
            await atlas.reindex("ghost")
        assert atlas.delete_document("default", "nothing.md") is False
        atlas.close()
