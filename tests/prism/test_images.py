"""Image loading, normalisation and encoding."""

from __future__ import annotations

import base64
import io
import struct
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from nimlab.prism import ImageError, PrismImage, dominant_colors, list_images, load_image
from nimlab.prism import images as images_mod
from nimlab.prism.images import content_parts, image_size, sniff_format, strip_jpeg_exif


def pil() -> Any:
    return pytest.importorskip("PIL.Image")


def encoded(fmt: str, size: tuple[int, int] = (64, 48), **save: Any) -> bytes:
    Image = pil()
    im = Image.new("RGB", size, (30, 120, 220))
    buf = io.BytesIO()
    im.save(buf, fmt, **save)
    return buf.getvalue()


def jpeg_with_gps() -> bytes:
    Image = pil()
    exif = Image.Exif()
    exif[0x010F] = "TestCam"
    gps = exif.get_ifd(0x8825)
    gps[1], gps[2], gps[3], gps[4] = "N", (37.0, 46.0, 29.0), "W", (122.0, 25.0, 10.0)
    return encoded("JPEG", exif=exif)


@pytest.fixture
def no_pillow(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(images_mod, "_pil", lambda: None)


class TestFormats:
    def test_png_from_path_and_bytes(self, png: Callable[..., bytes], tmp_path: Path) -> None:
        data = png(30, 20)
        path = tmp_path / "shot.png"
        path.write_bytes(data)
        for source in (path, str(path), data):
            img = load_image(source)
            assert (img.format, img.width, img.height, img.mime) == ("png", 30, 20, "image/png")
        assert load_image(path).name == "shot.png"
        assert load_image(data, name="dir/evil name!.png").name == "evil name_.png"

    @pytest.mark.parametrize(
        ("fmt", "mime"), [("JPEG", "image/jpeg"), ("WEBP", "image/webp"), ("PNG", "image/png")]
    )
    def test_pillow_made_formats(self, fmt: str, mime: str) -> None:
        img = load_image(encoded(fmt, (64, 48)))
        assert img.mime == mime and (img.width, img.height) == (64, 48)

    def test_gif_is_read_and_flattened_to_png(self) -> None:
        img = load_image(encoded("GIF"))
        assert img.format == "png" and img.mime == "image/png"
        assert any("GIF" in n for n in img.notes)

    @pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP", "GIF"])
    def test_header_size_reader_matches_pillow(self, fmt: str) -> None:
        data = encoded(fmt, (77, 31))
        assert image_size(data) == (77, 31)

    def test_lossless_webp_header(self) -> None:
        assert image_size(encoded("WEBP", (50, 40), lossless=True)) == (50, 40)

    def test_sniffing_ignores_the_extension(
        self, png: Callable[..., bytes], tmp_path: Path
    ) -> None:
        path = tmp_path / "actually.jpg"
        path.write_bytes(png(8, 8))
        assert load_image(path).format == "png"
        assert sniff_format(b"GIF89a....") == "gif" and sniff_format(b"hello") is None

    @pytest.mark.parametrize(
        ("data", "message"),
        [(b"", "empty"), (b"%PDF-1.7 ...", "unsupported"), (b"\x89PNG\r\n\x1a\nbad", "corrupt")],
    )
    def test_bad_input_has_a_clear_error(self, data: bytes, message: str) -> None:
        with pytest.raises(ImageError, match=message):
            load_image(data)

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ImageError, match="not found"):
            load_image(tmp_path / "nope.png")


class TestEncoding:
    def test_data_url_round_trips(self, png: Callable[..., bytes]) -> None:
        data = png(10, 10)
        img = load_image(data)
        prefix, _, b64 = img.data_url.partition(",")
        assert prefix == "data:image/png;base64"
        assert base64.b64decode(b64) == img.data

    def test_content_parts_put_text_first_then_images(self, png: Callable[..., bytes]) -> None:
        a, b = load_image(png(4, 4)), load_image(png(5, 5, fill=(0, 0, 255)))
        parts = content_parts("compare", [a, b])
        assert parts[0] == {"type": "text", "text": "compare"}
        assert [p["type"] for p in parts] == ["text", "image_url", "image_url"]
        assert parts[1]["image_url"]["url"] == a.data_url
        assert a.part() == parts[1]

    def test_info_is_json_friendly(self, png: Callable[..., bytes]) -> None:
        info = load_image(png(4, 4)).info()
        assert info["width"] == 4 and info["format"] == "png" and len(info["sha256"]) == 16


class TestDownscale:
    def test_large_image_is_scaled_to_the_max_side(self) -> None:
        img = load_image(encoded("PNG", (3000, 1500)), max_side=1568)
        assert (img.width, img.height) == (1568, 784)
        assert img.original_size == (3000, 1500)
        assert any("downscaled" in n for n in img.notes)

    def test_small_image_is_untouched(self, png: Callable[..., bytes]) -> None:
        data = png(100, 80)
        img = load_image(data)
        assert img.data == data and img.notes == []

    def test_custom_max_side(self) -> None:
        assert load_image(encoded("PNG", (400, 200)), max_side=100).width == 100

    def test_oversize_bytes_are_re_encoded_as_jpeg(self) -> None:
        Image = pil()
        noisy = Image.effect_noise((900, 900), 80).convert("RGB")
        buf = io.BytesIO()
        noisy.save(buf, "PNG")
        img = load_image(buf.getvalue(), max_bytes=200_000)
        assert img.format == "jpeg" and len(img.data) <= 200_000

    def test_hopeless_size_cap_is_an_error(self) -> None:
        Image = pil()
        buf = io.BytesIO()
        Image.effect_noise((400, 400), 90).convert("RGB").save(buf, "PNG")
        with pytest.raises(ImageError, match="size cap"):
            load_image(buf.getvalue(), max_bytes=2_000)


class TestNoPillow:
    def test_small_images_still_load(self, no_pillow: None, png: Callable[..., bytes]) -> None:
        img = load_image(png(20, 10))
        assert (img.width, img.height) == (20, 10)

    def test_large_image_passes_through_with_a_note(
        self, no_pillow: None, png: Callable[..., bytes]
    ) -> None:
        data = png(1700, 4)
        img = load_image(data, max_side=1568)
        assert img.data == data and img.width == 1700
        assert any("Pillow" in n for n in img.notes)

    def test_large_and_over_cap_is_a_clear_error(
        self, no_pillow: None, png: Callable[..., bytes]
    ) -> None:
        with pytest.raises(ImageError, match="pip install pillow"):
            load_image(png(1700, 40), max_side=1000, max_bytes=100)

    def test_over_cap_without_resize_need(self, no_pillow: None, png: Callable[..., bytes]) -> None:
        with pytest.raises(ImageError, match="size cap"):
            load_image(png(20, 20), max_bytes=10)


class TestExif:
    def test_pillow_path_removes_gps(self) -> None:
        Image = pil()
        data = jpeg_with_gps()
        assert Image.open(io.BytesIO(data)).getexif().get_ifd(0x8825)  # the fixture has GPS
        img = load_image(data)
        assert not Image.open(io.BytesIO(img.data)).getexif().get_ifd(0x8825)
        assert b"Exif" not in img.data[:200]
        assert any("EXIF" in n for n in img.notes)

    def test_stdlib_path_removes_gps(self, no_pillow: None) -> None:
        data = jpeg_with_gps()
        assert b"Exif\x00\x00" in data
        img = load_image(data)
        assert b"Exif\x00\x00" not in img.data and any("EXIF" in n for n in img.notes)
        assert image_size(img.data) == (64, 48)  # still a valid JPEG

    def test_strip_is_a_noop_without_exif(self) -> None:
        data = encoded("JPEG")
        assert strip_jpeg_exif(data) == data
        assert strip_jpeg_exif(b"not a jpeg") == b"not a jpeg"

    def test_orientation_is_applied_before_stripping(self) -> None:
        Image = pil()
        exif = Image.Exif()
        exif[0x0112] = 6  # rotate 90 degrees clockwise to display
        img = load_image(encoded("JPEG", (80, 40), exif=exif))
        assert (img.width, img.height) == (40, 80)


class TestColours:
    def test_png_without_pillow_uses_the_png_decoder(
        self, no_pillow: None, png: Callable[..., bytes]
    ) -> None:
        colours, source = dominant_colors(png(40, 40, fill=(250, 10, 10), stripe=(10, 10, 250)))
        assert source == "png"
        assert set(colours) == {"#FA0A0A", "#0A0AFA"}

    def test_pillow_agrees(self, png: Callable[..., bytes]) -> None:
        pil()
        colours, source = dominant_colors(png(40, 40, fill=(250, 10, 10)))
        assert source == "pillow" and colours == ["#FA0A0A"]

    def test_png_filters_are_decoded(self, no_pillow: None) -> None:
        Image = pil()
        im = Image.new("RGB", (64, 64))
        px = im.load()
        for y in range(64):
            for x in range(64):
                px[x, y] = (x * 4, y * 4, 128)
        buf = io.BytesIO()
        im.save(buf, "PNG", optimize=True)
        colours, source = dominant_colors(buf.getvalue(), 3)
        assert source == "png" and len(colours) == 3

    def test_hash_fallback_is_stable(self, no_pillow: None) -> None:
        data = encoded("JPEG")
        first, source = dominant_colors(data)
        assert source == "hash" and first == dominant_colors(data)[0]

    def test_helper_on_image(self, png: Callable[..., bytes]) -> None:
        assert load_image(png(8, 8, fill=(0, 200, 0))).colors(1) == ["#00C800"]


class TestListing:
    def test_lists_only_images_sorted(self, tmp_path: Path, png: Callable[..., bytes]) -> None:
        for name in ("b.png", "a.JPG", "c.txt", "d.webp"):
            (tmp_path / name).write_bytes(png(2, 2))
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "e.png").write_bytes(png(2, 2))
        assert [p.name for p in list_images(tmp_path)] == ["a.JPG", "b.png", "d.webp"]
        assert [p.name for p in list_images(tmp_path, recursive=True)][-1] == "e.png"

    def test_not_a_directory(self, tmp_path: Path) -> None:
        with pytest.raises(ImageError, match="not a directory"):
            list_images(tmp_path / "missing")


def test_png_with_bad_ihdr_is_rejected() -> None:
    with pytest.raises(ImageError, match="IHDR"):
        load_image(b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"XXXX" + b"\0" * 20)


def test_prism_image_is_exported() -> None:
    assert PrismImage is images_mod.PrismImage
