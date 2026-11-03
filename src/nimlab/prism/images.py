"""Image loading, normalisation and encoding.

Everything works on the standard library; Pillow (optional) adds downscaling, re-encoding and
orientation handling. Without Pillow Prism still reads image sizes, strips JPEG EXIF (location
data) and computes dominant colours of PNGs, and it passes large images through unchanged as long
as they fit the byte cap.
"""

from __future__ import annotations

import base64
import hashlib
import io
import re
import struct
import zlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import ImageError

__all__ = [
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_SIDE",
    "EXTENSIONS",
    "PrismImage",
    "content_parts",
    "dominant_colors",
    "have_pillow",
    "image_size",
    "list_images",
    "load_image",
    "sniff_format",
    "strip_jpeg_exif",
]

DEFAULT_MAX_SIDE = 1568
DEFAULT_MAX_BYTES = 4 * 1024 * 1024
EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".gif")
MIME = {"png": "image/png", "jpeg": "image/jpeg", "webp": "image/webp", "gif": "image/gif"}
_PNG_SIG = b"\x89PNG\r\n\x1a\n"
_MAX_DECODE_PIXELS = 4_000_000


def _pil() -> Any | None:
    """The Pillow ``Image`` module, or ``None`` (patched in tests to exercise the fallback)."""
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - depends on the environment
        return None
    return Image


def have_pillow() -> bool:
    return _pil() is not None


# --------------------------------------------------------------------------- format & size


def sniff_format(data: bytes) -> str | None:
    """``png`` / ``jpeg`` / ``webp`` / ``gif`` from magic bytes, else ``None``."""
    if data.startswith(_PNG_SIG):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def _png_size(data: bytes) -> tuple[int, int]:
    if len(data) < 24 or data[12:16] != b"IHDR":
        raise ImageError("corrupt PNG: missing IHDR chunk")
    w, h = struct.unpack(">II", data[16:24])
    return w, h


def _gif_size(data: bytes) -> tuple[int, int]:
    if len(data) < 10:
        raise ImageError("corrupt GIF: header too short")
    w, h = struct.unpack("<HH", data[6:10])
    return w, h


def _jpeg_size(data: bytes) -> tuple[int, int]:
    i, n = 2, len(data)
    while i + 4 <= n:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker == 0xFF:
            i += 1
            continue
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        (length,) = struct.unpack(">H", data[i + 2 : i + 4])
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            if i + 9 > n:
                break
            h, w = struct.unpack(">HH", data[i + 5 : i + 9])
            return w, h
        i += 2 + length
    raise ImageError("corrupt JPEG: no frame header found")


def _webp_size(data: bytes) -> tuple[int, int]:
    kind = data[12:16]
    try:
        if kind == b"VP8X":
            w = 1 + int.from_bytes(data[24:27], "little")
            h = 1 + int.from_bytes(data[27:30], "little")
            return w, h
        if kind == b"VP8L":
            bits = int.from_bytes(data[21:25], "little")
            return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
        if kind == b"VP8 ":
            w, h = struct.unpack("<HH", data[26:30])
            return w & 0x3FFF, h & 0x3FFF
    except (struct.error, ValueError):
        pass
    raise ImageError("corrupt WebP: unreadable header")


def image_size(data: bytes, fmt: str | None = None) -> tuple[int, int]:
    """``(width, height)`` read from the file header (no decoding)."""
    fmt = fmt or sniff_format(data)
    readers = {"png": _png_size, "gif": _gif_size, "jpeg": _jpeg_size, "webp": _webp_size}
    if fmt not in readers:
        raise ImageError("unsupported image type (use png, jpg, webp or gif)")
    w, h = readers[fmt](data)
    if w <= 0 or h <= 0:
        raise ImageError("image has zero width or height")
    return w, h


# --------------------------------------------------------------------------- EXIF


def strip_jpeg_exif(data: bytes) -> bytes:
    """Remove EXIF (``APP1``) segments from a JPEG without re-encoding it.

    EXIF is where cameras store GPS coordinates. XMP and other metadata are kept.
    """
    if not data.startswith(b"\xff\xd8"):
        return data
    out = bytearray(b"\xff\xd8")
    i, n = 2, len(data)
    while i + 4 <= n:
        if data[i] != 0xFF:
            break
        marker = data[i + 1]
        if marker == 0xDA:  # start of scan: the rest is entropy-coded data
            break
        if marker in (0x01, 0xFF) or 0xD0 <= marker <= 0xD8:
            out += data[i : i + 2]
            i += 2
            continue
        (length,) = struct.unpack(">H", data[i + 2 : i + 4])
        segment = data[i : i + 2 + length]
        if not (marker == 0xE1 and segment[4:10] == b"Exif\x00\x00"):
            out += segment
        i += 2 + length
    out += data[i:]
    return bytes(out)


def _has_jpeg_exif(data: bytes) -> bool:
    return strip_jpeg_exif(data) != data


# --------------------------------------------------------------------------- colours


def _png_pixels(data: bytes, limit: int = 4096) -> list[tuple[int, int, int]]:
    """Sample RGB pixels from an 8-bit, non-interlaced PNG with the standard library."""
    w, h = _png_size(data)
    if w * h > _MAX_DECODE_PIXELS:
        raise ImageError("too large to sample")
    pos, idat, plte = 8, bytearray(), b""
    depth = ctype = interlace = -1
    while pos + 8 <= len(data):
        length, kind = struct.unpack(">I4s", data[pos : pos + 8])
        body = data[pos + 8 : pos + 8 + length]
        if kind == b"IHDR":
            depth, ctype, _, _, interlace = struct.unpack(">BBBBB", body[8:13])
        elif kind == b"PLTE":
            plte = body
        elif kind == b"IDAT":
            idat += body
        elif kind == b"IEND":
            break
        pos += 12 + length
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(ctype)
    if depth != 8 or interlace != 0 or channels is None:
        raise ImageError("unsupported PNG layout for sampling")
    raw = zlib.decompress(bytes(idat))
    stride = w * channels
    prev = bytearray(stride)
    rows: list[bytearray] = []
    p = 0
    for _ in range(h):
        ft = raw[p]
        line = bytearray(raw[p + 1 : p + 1 + stride])
        p += 1 + stride
        for x in range(stride):
            a = line[x - channels] if x >= channels else 0
            b = prev[x]
            c = prev[x - channels] if x >= channels else 0
            if ft == 1:
                line[x] = (line[x] + a) & 255
            elif ft == 2:
                line[x] = (line[x] + b) & 255
            elif ft == 3:
                line[x] = (line[x] + ((a + b) >> 1)) & 255
            elif ft == 4:
                pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                pred = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                line[x] = (line[x] + pred) & 255
        rows.append(line)
        prev = line
    step = max(1, int((w * h / limit) ** 0.5))
    pixels: list[tuple[int, int, int]] = []
    for y in range(0, h, step):
        line = rows[y]
        for x in range(0, w, step):
            o = x * channels
            if ctype == 3:
                i = line[o] * 3
                pixels.append((plte[i], plte[i + 1], plte[i + 2]))
            elif ctype in (0, 4):
                pixels.append((line[o], line[o], line[o]))
            else:
                pixels.append((line[o], line[o + 1], line[o + 2]))
    return pixels


def _palette(pixels: Iterable[tuple[int, int, int]], n: int) -> list[str]:
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for r, g, b in pixels:
        key = (r >> 5, g >> 5, b >> 5)
        acc = buckets.setdefault(key, [0, 0, 0, 0])
        acc[0] += r
        acc[1] += g
        acc[2] += b
        acc[3] += 1
    ranked = sorted(buckets.values(), key=lambda a: (-a[3], a[0], a[1], a[2]))
    colors: list[tuple[int, int, int]] = []
    for r, g, b, c in ranked:
        cand = (r // c, g // c, b // c)
        if all(sum(abs(x - y) for x, y in zip(cand, other, strict=True)) > 60 for other in colors):
            colors.append(cand)
        if len(colors) == n:
            break
    return [f"#{r:02X}{g:02X}{b:02X}" for r, g, b in colors]


def _hash_palette(data: bytes, n: int) -> list[str]:
    digest = hashlib.sha256(data).digest()
    return [f"#{digest[3 * i]:02X}{digest[3 * i + 1]:02X}{digest[3 * i + 2]:02X}" for i in range(n)]


def dominant_colors(data: bytes, n: int = 3) -> tuple[list[str], str]:
    """``(hex colours, source)``; source is ``pillow``, ``png`` or ``hash`` (a stable stand-in)."""
    pil = _pil()
    if pil is not None:
        try:
            with pil.open(io.BytesIO(data)) as im:
                small = im.convert("RGB")
                small.thumbnail((64, 64))
                raw = small.tobytes()
                px = [(raw[i], raw[i + 1], raw[i + 2]) for i in range(0, len(raw) - 2, 3)]
            colors = _palette(px, n)
            if colors:
                return colors, "pillow"
        except Exception:  # corrupt or exotic file: fall through
            pass
    elif sniff_format(data) == "png":
        try:
            colors = _palette(_png_pixels(data), n)
            if colors:
                return colors, "png"
        except (ImageError, zlib.error, IndexError, struct.error):
            pass
    return _hash_palette(data, n), "hash"


# --------------------------------------------------------------------------- the image object


@dataclass
class PrismImage:
    """A validated image, ready to send: normalised bytes plus what was done to them."""

    name: str
    data: bytes
    format: str
    width: int
    height: int
    original_size: tuple[int, int]
    original_bytes: int
    notes: list[str] = field(default_factory=list)

    @property
    def mime(self) -> str:
        return MIME[self.format]

    @property
    def sha(self) -> str:
        return hashlib.sha256(self.data).hexdigest()

    @property
    def data_url(self) -> str:
        return f"data:{self.mime};base64,{base64.b64encode(self.data).decode('ascii')}"

    def part(self) -> dict[str, Any]:
        """An OpenAI-style ``image_url`` content part."""
        return {"type": "image_url", "image_url": {"url": self.data_url}}

    def colors(self, n: int = 3) -> list[str]:
        return dominant_colors(self.data, n)[0]

    def info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "format": self.format,
            "width": self.width,
            "height": self.height,
            "bytes": len(self.data),
            "original_width": self.original_size[0],
            "original_height": self.original_size[1],
            "original_bytes": self.original_bytes,
            "sha256": self.sha[:16],
            "notes": list(self.notes),
        }


def content_parts(text: str, images: Iterable[PrismImage]) -> list[dict[str, Any]]:
    """User-message content: the text first, then one ``image_url`` part per image."""
    return [{"type": "text", "text": text}, *(img.part() for img in images)]


# --------------------------------------------------------------------------- loading

_SAFE_NAME = re.compile(r"[^\w.\- ]+")


def _clean_name(name: str) -> str:
    base = Path(name.replace("\\", "/")).name.strip()
    return _SAFE_NAME.sub("_", base)[:120] or "image"


def _normalise_pillow(
    pil: Any, data: bytes, fmt: str, max_side: int, max_bytes: int, notes: list[str]
) -> tuple[bytes, str]:
    from PIL import ImageOps

    try:
        im = pil.open(io.BytesIO(data))
        im.load()
    except Exception as exc:
        raise ImageError(f"could not decode the image: {exc}") from exc
    exif = im.getexif() if hasattr(im, "getexif") else {}
    has_exif = bool(exif) or (fmt == "jpeg" and _has_jpeg_exif(data))
    animated = bool(getattr(im, "is_animated", False))
    oversize = max(im.size) > max_side
    if not (has_exif or oversize or animated or fmt == "gif" or len(data) > max_bytes):
        return data, fmt
    if has_exif:
        im = ImageOps.exif_transpose(im)
        notes.append("EXIF metadata removed")
    if oversize:
        scale = max_side / max(im.size)
        im = im.resize(
            (max(1, round(im.width * scale)), max(1, round(im.height * scale))), pil.LANCZOS
        )
        notes.append(f"downscaled to max side {max_side}px")
    if animated or fmt == "gif":
        notes.append("GIF flattened to its first frame (PNG)")
        fmt = "png"
    buf = io.BytesIO()
    if fmt == "jpeg":
        im.convert("RGB").save(buf, "JPEG", quality=90)
    elif fmt == "webp":
        im.save(buf, "WEBP", quality=90)
    else:
        im.save(buf, "PNG", optimize=True)
    out = buf.getvalue()
    if len(out) > max_bytes:
        flat = im.convert("RGB")
        for factor in (1.0, 0.8, 0.64, 0.5, 0.4, 0.3):
            cand = flat
            if factor < 1.0:
                cand = flat.resize(
                    (max(1, round(flat.width * factor)), max(1, round(flat.height * factor))),
                    pil.LANCZOS,
                )
            for quality in (85, 70, 55):
                buf = io.BytesIO()
                cand.save(buf, "JPEG", quality=quality)
                if len(buf.getvalue()) <= max_bytes:
                    notes.append("re-encoded as JPEG to fit the size cap")
                    if factor < 1.0:
                        notes.append(f"scaled a further {round(factor * 100)}% for the size cap")
                    return buf.getvalue(), "jpeg"
    return out, fmt


def load_image(
    source: str | Path | bytes | bytearray | PrismImage,
    *,
    name: str | None = None,
    max_side: int = DEFAULT_MAX_SIDE,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> PrismImage:
    """Read, validate and normalise an image.

    * accepts a path, raw bytes or an existing :class:`PrismImage`;
    * png / jpg / webp / gif only, detected by content rather than extension;
    * with Pillow: downscales so the longest side is at most ``max_side``, drops EXIF (applying
      the orientation first) and flattens GIFs to PNG; without it, JPEG EXIF is still stripped but
      oversized images are passed through if they fit ``max_bytes`` and rejected otherwise.
    """
    if isinstance(source, PrismImage):
        return source
    if isinstance(source, (bytes, bytearray)):
        data, label = bytes(source), name or "image"
    else:
        path = Path(source).expanduser()
        if not path.is_file():
            raise ImageError(f"image not found: {path}")
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise ImageError(f"could not read {path}: {exc}") from exc
        label = name or path.name
    if not data:
        raise ImageError(f"{label}: the file is empty")
    raw_len = len(data)
    fmt = sniff_format(data)
    if fmt is None:
        raise ImageError(f"{label}: unsupported image type (use png, jpg, webp or gif)")
    original = image_size(data, fmt)
    notes: list[str] = []
    pil = _pil()
    if pil is not None:
        data, fmt = _normalise_pillow(pil, data, fmt, max_side, max_bytes, notes)
    else:
        if fmt == "jpeg" and _has_jpeg_exif(data):
            data = strip_jpeg_exif(data)
            notes.append("EXIF metadata removed")
        if max(original) > max_side:
            if len(data) > max_bytes:
                raise ImageError(
                    f"{label}: {original[0]}x{original[1]} px, {len(data) // 1024} KiB is over "
                    f"the {max_bytes // 1024} KiB cap and cannot be downscaled because Pillow "
                    "is not installed (pip install pillow, or resize the image first)"
                )
            notes.append(
                f"not downscaled ({max(original)}px > {max_side}px): Pillow is not installed"
            )
    if len(data) > max_bytes:
        raise ImageError(
            f"{label}: {len(data) // 1024} KiB is over the {max_bytes // 1024} KiB size cap"
        )
    width, height = image_size(data, fmt)
    return PrismImage(
        name=_clean_name(label),
        data=data,
        format=fmt,
        width=width,
        height=height,
        original_size=original,
        original_bytes=raw_len,
        notes=notes,
    )


def list_images(directory: str | Path, *, recursive: bool = False) -> list[Path]:
    """Image files (by extension) in ``directory``, sorted by name."""
    root = Path(directory).expanduser()
    if not root.is_dir():
        raise ImageError(f"not a directory: {root}")
    walk = root.rglob("*") if recursive else root.iterdir()
    return sorted(p for p in walk if p.is_file() and p.suffix.lower() in EXTENSIONS)
