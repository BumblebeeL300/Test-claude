"""Generate the tiny sample images in this folder with the standard library only.

    python examples/prism/make_samples.py

* ``sample_ui.png``    a 256x160 mock app screen (header, sidebar, cards)
* ``sample_chart.png`` a 240x150 bar chart
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

HERE = Path(__file__).parent
Color = tuple[int, int, int]


def png(width: int, height: int, pixels: list[list[Color]]) -> bytes:
    def chunk(kind: bytes, body: bytes) -> bytes:
        crc = zlib.crc32(kind + body) & 0xFFFFFFFF
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", crc)

    raw = b"".join(b"\x00" + bytes(c for px in row for c in px) for row in pixels)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def canvas(width: int, height: int, fill: Color) -> list[list[Color]]:
    return [[fill] * width for _ in range(height)]


def rect(px: list[list[Color]], x: int, y: int, w: int, h: int, color: Color) -> None:
    for yy in range(max(0, y), min(len(px), y + h)):
        row = px[yy]
        for xx in range(max(0, x), min(len(row), x + w)):
            row[xx] = color


def sample_ui() -> bytes:
    w, h = 256, 160
    px = canvas(w, h, (244, 245, 250))
    rect(px, 0, 0, w, 24, (38, 50, 120))  # header
    rect(px, 8, 8, 40, 8, (240, 244, 255))  # logo
    rect(px, 0, 24, 56, h - 24, (28, 36, 84))  # sidebar
    for i in range(4):
        rect(px, 10, 38 + i * 22, 36, 8, (120, 140, 230))
    for i in range(3):  # cards
        x = 68 + i * 62
        rect(px, x, 36, 54, 48, (255, 255, 255))
        rect(px, x, 36, 54, 6, (255, 138, 61) if i == 1 else (110, 124, 255))
        rect(px, x + 6, 52, 30, 5, (180, 186, 210))
        rect(px, x + 6, 62, 40, 5, (210, 214, 230))
    rect(px, 68, 96, 182, 52, (255, 255, 255))  # table
    for i in range(4):
        rect(px, 76, 102 + i * 11, 166, 5, (222, 225, 238) if i % 2 else (200, 206, 232))
    return png(w, h, px)


def sample_chart() -> bytes:
    w, h = 240, 150
    px = canvas(w, h, (255, 255, 255))
    rect(px, 28, 10, 2, 120, (60, 60, 70))  # y axis
    rect(px, 28, 130, 200, 2, (60, 60, 70))  # x axis
    for i in range(1, 5):
        rect(px, 30, 130 - i * 26, 196, 1, (225, 225, 232))
    for i, value in enumerate((38, 64, 52, 96, 80)):
        rect(px, 40 + i * 38, 130 - value, 26, value, (108, 92, 255) if i != 3 else (255, 61, 154))
    return png(w, h, px)


if __name__ == "__main__":
    (HERE / "sample_ui.png").write_bytes(sample_ui())
    (HERE / "sample_chart.png").write_bytes(sample_chart())
    print("wrote sample_ui.png and sample_chart.png")
