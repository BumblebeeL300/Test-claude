"""Atlas: agentic, citation-verified retrieval-augmented QA on NVIDIA NIM.

>>> from nimlab.atlas import Atlas                      # doctest: +SKIP
>>> async with Atlas.open(".nimlab/atlas.db") as atlas:  # doctest: +SKIP
...     await atlas.ingest("docs/")
...     answer = await atlas.ask("How do I rotate keys?", verify=True)
"""

from __future__ import annotations

from .bm25 import BM25Index
from .chunking import chunk_sections
from .config import AtlasConfig, ModelSet
from .engine import Atlas
from .evaluation import EvalReport, QAItem, load_qa, run_eval
from .fusion import mmr_select, reciprocal_rank_fusion
from .models import (
    Answer,
    AtlasError,
    AtlasEvent,
    ClaimCheck,
    FileReport,
    GuardVerdict,
    IngestReport,
    SourceRef,
    TraceStep,
    VerificationReport,
)
from .store import AtlasStore

__all__ = [
    "Answer",
    "Atlas",
    "AtlasConfig",
    "AtlasError",
    "AtlasEvent",
    "AtlasStore",
    "BM25Index",
    "ClaimCheck",
    "EvalReport",
    "FileReport",
    "GuardVerdict",
    "IngestReport",
    "ModelSet",
    "QAItem",
    "SourceRef",
    "TraceStep",
    "VerificationReport",
    "chunk_sections",
    "load_qa",
    "mmr_select",
    "reciprocal_rank_fusion",
    "run_eval",
]
