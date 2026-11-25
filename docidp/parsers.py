"""Turn uploaded bytes into something an LLM can read: text for PDFs, base64 JPEG for images."""

from __future__ import annotations

import base64
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Literal, Optional

from PIL import Image, ImageOps, UnidentifiedImageError

from .config import Settings
from .errors import (
    FileTooLargeError,
    ParsingError,
    ScannedPDFError,
    UnsupportedFileError,
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
PDF_EXTS = {".pdf"}
SUPPORTED_EXTS = sorted(IMAGE_EXTS | PDF_EXTS)
MIN_TEXT_CHARS = 20
MAX_IMAGE_SIDE = 2048
MAX_IMAGE_B64_BYTES = 3_500_000  # Groq rejects base64 images above 4 MB; stay safely below


@dataclass(frozen=True)
class Document:
    name: str
    kind: Literal["text", "image"]
    content: str  # markdown/plain text, or base64 JPEG
    truncated: bool = False
    parse_seconds: float = 0.0

    @property
    def is_image(self) -> bool:
        return self.kind == "image"


def load_document(data: bytes, filename: str, settings: Optional[Settings] = None) -> Document:
    settings = settings or Settings()
    name = os.path.basename(filename) or "upload"
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise FileTooLargeError(f"{name} exceeds the {settings.max_upload_mb} MB limit.")
    ext = os.path.splitext(name)[1].lower()
    t0 = time.perf_counter()

    if ext in PDF_EXTS:
        if b"%PDF" not in data[:1024]:
            raise ParsingError(f"{name} is not a valid PDF.")
        text = extract_pdf_text(data, settings)
        truncated = len(text) > settings.max_text_chars
        return Document(name, "text", text[: settings.max_text_chars], truncated, time.perf_counter() - t0)

    if ext in IMAGE_EXTS:
        return Document(name, "image", normalize_image(data), False, time.perf_counter() - t0)

    raise UnsupportedFileError(f"Unsupported file type {ext or '(none)'!r}. Supported: {', '.join(SUPPORTED_EXTS)}")


# ------------------------------------------------------------------ PDF
def extract_pdf_text(data: bytes, settings: Settings) -> str:
    _check_page_count(data, settings.max_pdf_pages)
    marker_available = shutil.which("marker_single") is not None
    parser = settings.pdf_parser
    if parser == "marker" and not marker_available:
        raise ParsingError("PDF_PARSER=marker but `marker_single` is not installed (pip install '.[marker]').")

    if parser == "marker" or (parser == "auto" and marker_available):
        text = _pdf_text_marker(data, settings.marker_timeout_s)
    else:
        text = _pdf_text_pypdf(data)

    if len(text.strip()) < MIN_TEXT_CHARS:
        raise ScannedPDFError(
            "No text layer found; this looks like a scanned PDF. Install the marker extra "
            "(OCR) or upload the pages as images."
        )
    return text


def _open_reader(data: bytes):
    from pypdf import PdfReader
    from pypdf.errors import PyPdfError

    try:
        reader = PdfReader(BytesIO(data))
        if reader.is_encrypted and reader.decrypt("") == 0:
            raise ParsingError("PDF is password protected.")
        return reader
    except PyPdfError as exc:
        raise ParsingError(f"Could not read PDF: {exc}") from exc


def _check_page_count(data: bytes, max_pages: int) -> None:
    n = len(_open_reader(data).pages)
    if n > max_pages:
        raise ParsingError(f"PDF has {n} pages; the limit is {max_pages}.")


def _pdf_text_pypdf(data: bytes) -> str:
    reader = _open_reader(data)
    parts = []
    for i, page in enumerate(reader.pages, 1):
        try:
            txt = page.extract_text(extraction_mode="layout")
        except TypeError:  # very old pypdf
            txt = page.extract_text()
        txt = re.sub(r"[ \t]+\n", "\n", txt or "")
        txt = re.sub(r"\n{3,}", "\n\n", txt).strip()
        parts.append(f"[page {i}]\n{txt}")
    return "\n\n".join(parts)


def _pdf_text_marker(data: bytes, timeout_s: int) -> str:
    """Layout-aware markdown via the marker CLI, inside a throw-away directory (no shell)."""
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "input.pdf"
        out_dir = Path(tmp) / "out"
        out_dir.mkdir()
        pdf_path.write_bytes(data)
        try:
            subprocess.run(
                ["marker_single", str(pdf_path), "--output_dir", str(out_dir)],
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise ParsingError(f"marker timed out after {timeout_s}s.") from exc
        except subprocess.CalledProcessError as exc:
            raise ParsingError(f"marker failed: {(exc.stderr or '')[-300:]}") from exc
        md_files = sorted(out_dir.glob("**/*.md"))
        if not md_files:
            raise ParsingError("marker produced no markdown output.")
        return md_files[0].read_text(encoding="utf-8")


# ------------------------------------------------------------------ images
def normalize_image(data: bytes) -> str:
    """Decode any supported image to upright, white-background RGB JPEG (<=2048px) as base64."""
    try:
        img = Image.open(BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ParsingError(f"Could not read image: {exc}") from exc

    img = ImageOps.exif_transpose(img)  # phone photos: honour the rotation flag
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")  # flatten onto white; a plain convert('RGB') turns transparency black
        flat = Image.new("RGB", rgba.size, (255, 255, 255))
        flat.paste(rgba, mask=rgba.getchannel("A"))
        img = flat
    else:
        img = img.convert("RGB")

    img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
    return _encode_jpeg_within_limit(img)


def _encode_jpeg_within_limit(img: Image.Image) -> str:
    """JPEG -> base64, lowering quality and then resolution until it fits the provider payload limit."""
    for _ in range(6):
        for quality in (90, 80, 70, 60):
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=quality, optimize=True)
            encoded = base64.b64encode(buf.getvalue()).decode("ascii")
            if len(encoded) <= MAX_IMAGE_B64_BYTES:
                return encoded
        img = img.resize((int(img.width * 0.8), int(img.height * 0.8)), Image.Resampling.LANCZOS)
    raise ParsingError("Image is too large to send to the model even after compression.")
