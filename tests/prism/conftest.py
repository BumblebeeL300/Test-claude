"""Fixtures for the Prism tests: tiny stdlib-built PNGs, a mock client and a Prism on top of it."""

from __future__ import annotations

import struct
import zlib
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest

from nimlab.core import NIMClient, NIMConfig
from nimlab.prism import Prism

EXAMPLES = Path(__file__).parents[2] / "examples" / "prism"
Color = tuple[int, int, int]


def make_png(
    width: int, height: int, fill: Color = (200, 30, 30), stripe: Color | None = None
) -> bytes:
    """A valid truecolour PNG; with ``stripe`` the right half uses that colour."""

    def chunk(kind: bytes, body: bytes) -> bytes:
        return (
            struct.pack(">I", len(body))
            + kind
            + body
            + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)
        )

    rows = b""
    for _ in range(height):
        row = b"".join(bytes(stripe if stripe and x >= width // 2 else fill) for x in range(width))
        rows += b"\x00" + row
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


@pytest.fixture
def png() -> Callable[..., bytes]:
    return make_png


@pytest.fixture
def sample_ui() -> Path:
    return EXAMPLES / "sample_ui.png"


@pytest.fixture
def sample_chart() -> Path:
    return EXAMPLES / "sample_chart.png"


@pytest.fixture
def png_file(tmp_path: Path) -> Callable[..., Path]:
    def make(name: str = "a.png", width: int = 16, height: int = 12, **kw: Any) -> Path:
        path = tmp_path / name
        path.write_bytes(make_png(width, height, **kw))
        return path

    return make


@pytest.fixture
async def prism() -> AsyncIterator[Prism]:
    async with NIMClient(NIMConfig(mock=True)) as client:
        yield Prism(client)


@pytest.fixture
def prism_factory() -> Callable[..., Prism]:
    def make(**kw: Any) -> Prism:
        return Prism(NIMClient(NIMConfig(mock=True)), **kw)

    return make
