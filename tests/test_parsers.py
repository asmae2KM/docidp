import base64
from io import BytesIO

import pytest
from PIL import Image

from docidp.config import Settings
from docidp.errors import FileTooLargeError, ParsingError, ScannedPDFError, UnsupportedFileError
from docidp.parsers import load_document, normalize_image

PYPDF = Settings(pdf_parser="pypdf")


def png_bytes(img: Image.Image, fmt="PNG") -> bytes:
    buf = BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def decode(b64: str) -> Image.Image:
    img = Image.open(BytesIO(base64.b64decode(b64)))
    img.load()
    return img


def test_transparent_png_is_flattened_onto_white_not_black():
    """Regression: v1 used convert('RGB'), which turns transparent pixels black."""
    img = Image.new("RGBA", (40, 40), (0, 0, 0, 0))
    out = decode(normalize_image(png_bytes(img)))
    assert out.mode == "RGB" and out.getpixel((5, 5))[0] > 240


@pytest.mark.parametrize("mode", ["CMYK", "L", "P", "1"])
def test_other_color_modes_are_supported(mode):
    img = Image.new("RGB", (30, 30), (200, 10, 10)).convert(mode)
    fmt = "JPEG" if mode == "CMYK" else "PNG"  # PNG cannot store CMYK
    assert decode(normalize_image(png_bytes(img, fmt))).mode == "RGB"


def test_large_images_are_downscaled():
    out = decode(normalize_image(png_bytes(Image.new("RGB", (5000, 3000), "white"))))
    assert max(out.size) == 2048


def test_exif_orientation_is_applied():
    img = Image.new("RGB", (60, 20), "white")
    exif = Image.Exif()
    exif[274] = 6  # rotate 90 degrees clockwise to display
    buf = BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    assert decode(normalize_image(buf.getvalue())).size == (20, 60)


def test_corrupt_image_raises_parsing_error():
    with pytest.raises(ParsingError):
        normalize_image(b"definitely not an image")


def test_unsupported_extension():
    with pytest.raises(UnsupportedFileError):
        load_document(b"x", "notes.docx", PYPDF)


def test_size_limit():
    with pytest.raises(FileTooLargeError):
        load_document(b"x" * 2_100_000, "a.pdf", Settings(max_upload_mb=2))


def test_fake_pdf_is_rejected():
    with pytest.raises(ParsingError):
        load_document(b"hello world", "a.pdf", PYPDF)


def test_pdf_text_is_extracted_with_layout(dataset_dir):
    data = (dataset_dir / "files" / "inv-002.pdf").read_bytes()
    doc = load_document(data, "inv-002.pdf", PYPDF)
    assert doc.kind == "text" and "FACTURE" in doc.content and "Net à payer" in doc.content


def test_multi_page_pdf_marks_pages(dataset_dir):
    doc = load_document((dataset_dir / "files" / "inv-020.pdf").read_bytes(), "inv-020.pdf", PYPDF)
    assert "[page 1]" in doc.content and "[page 2]" in doc.content


def test_text_is_truncated_and_flagged(dataset_dir):
    data = (dataset_dir / "files" / "inv-020.pdf").read_bytes()
    doc = load_document(data, "x.pdf", Settings(pdf_parser="pypdf", max_text_chars=100))
    assert doc.truncated and len(doc.content) == 100


def test_pdf_page_limit(dataset_dir):
    data = (dataset_dir / "files" / "inv-020.pdf").read_bytes()
    with pytest.raises(ParsingError, match="pages"):
        load_document(data, "x.pdf", Settings(pdf_parser="pypdf", max_pdf_pages=1))


def test_pdf_without_text_layer_is_reported_as_scanned():
    from pypdf import PdfWriter

    w = PdfWriter()
    w.add_blank_page(200, 200)
    buf = BytesIO()
    w.write(buf)
    with pytest.raises(ScannedPDFError):
        load_document(buf.getvalue(), "scan.pdf", PYPDF)


def test_marker_requested_but_missing(monkeypatch, dataset_dir):
    monkeypatch.setattr("shutil.which", lambda _: None)
    data = (dataset_dir / "files" / "inv-002.pdf").read_bytes()
    with pytest.raises(ParsingError, match="marker"):
        load_document(data, "x.pdf", Settings(pdf_parser="marker"))


# ---------------------------------------------------------------- marker subprocess path (hermetic, any OS)
@pytest.fixture
def fake_marker(monkeypatch):
    """Pretend `marker_single` is installed and intercept the subprocess call.

    No executable is created or run, so this behaves identically on Linux, macOS and Windows,
    whether or not the real marker-pdf happens to be installed.
    """
    import subprocess

    from docidp import parsers

    monkeypatch.setattr(
        parsers.shutil, "which", lambda name, *a, **k: "marker_single" if name == "marker_single" else None
    )
    calls = []

    def install(behaviour):
        def fake_run(cmd, **kwargs):
            calls.append((cmd, kwargs))
            return behaviour(cmd)

        monkeypatch.setattr(parsers.subprocess, "run", fake_run)
        return calls

    install.subprocess = subprocess
    return install


def _write_markdown(cmd):
    from pathlib import Path

    out = Path(cmd[cmd.index("--output_dir") + 1]) / "input"
    out.mkdir(parents=True)
    (out / "input.md").write_text("# Parsed by marker\n\n| Total | 120.00 |", encoding="utf-8")


def test_marker_output_is_read_and_invoked_safely(fake_marker, dataset_dir):
    calls = fake_marker(_write_markdown)
    data = (dataset_dir / "files" / "inv-002.pdf").read_bytes()
    doc = load_document(data, "inv-002.pdf", Settings(pdf_parser="auto", marker_timeout_s=42))  # auto picks marker
    assert "Parsed by marker" in doc.content
    cmd, kwargs = calls[0]
    assert isinstance(cmd, list) and cmd[0] == "marker_single"  # argument list: no shell interpolation
    assert cmd[1].endswith("input.pdf")  # fixed file name, never the user's
    assert not kwargs.get("shell") and kwargs["timeout"] == 42


def test_marker_failure_becomes_parsing_error(fake_marker, dataset_dir):
    def fail(cmd):
        raise fake_marker.subprocess.CalledProcessError(1, cmd, stderr="CUDA out of memory")

    fake_marker(fail)
    data = (dataset_dir / "files" / "inv-002.pdf").read_bytes()
    with pytest.raises(ParsingError, match="CUDA out of memory"):
        load_document(data, "inv-002.pdf", Settings(pdf_parser="marker"))


def test_marker_timeout_becomes_parsing_error(fake_marker, dataset_dir):
    def hang(cmd):
        raise fake_marker.subprocess.TimeoutExpired(cmd, 1)

    fake_marker(hang)
    data = (dataset_dir / "files" / "inv-002.pdf").read_bytes()
    with pytest.raises(ParsingError, match="timed out"):
        load_document(data, "inv-002.pdf", Settings(pdf_parser="marker", marker_timeout_s=1))


def test_marker_without_output_file_is_an_error(fake_marker, dataset_dir):
    fake_marker(lambda cmd: None)  # "succeeds" but writes nothing
    data = (dataset_dir / "files" / "inv-002.pdf").read_bytes()
    with pytest.raises(ParsingError, match="no markdown"):
        load_document(data, "inv-002.pdf", Settings(pdf_parser="marker"))


def test_auto_falls_back_to_pypdf_when_marker_is_absent(dataset_dir):
    data = (dataset_dir / "files" / "inv-002.pdf").read_bytes()
    assert "FACTURE" in load_document(data, "inv-002.pdf", Settings(pdf_parser="auto")).content


def test_huge_noisy_images_are_compressed_under_the_provider_payload_limit():
    """Groq rejects base64 images over 4 MB. Random noise is the worst case for JPEG."""
    import os

    from docidp.parsers import MAX_IMAGE_B64_BYTES

    noise = Image.frombytes("RGB", (2400, 2400), os.urandom(2400 * 2400 * 3))
    encoded = normalize_image(png_bytes(noise))
    assert len(encoded) <= MAX_IMAGE_B64_BYTES
    assert decode(encoded).size[0] <= 2048