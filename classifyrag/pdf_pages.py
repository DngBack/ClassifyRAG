from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import fitz  # pymupdf
from PIL import Image

logger = logging.getLogger(__name__)

_OCR_UNAVAILABLE_LOGGED = False


@dataclass
class PageData:
    """One PDF page: raster image and optional text layer."""

    page_index: int
    image: Image.Image
    text: str
    #: How ``text`` was obtained: embedded PDF text, Tesseract OCR, or none.
    text_source: str = "empty"


def _normalize_text(raw: str) -> str:
    t = raw.replace("\r", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _text_from_text_dict(page: fitz.Page) -> str:
    """Fallback for PDFs where plain ``get_text('text')`` is empty but span data exists."""
    parts: list[str] = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                t = span.get("text") or ""
                if t:
                    parts.append(t)
    return _normalize_text(" ".join(parts))


def extract_page_text_native(page: fitz.Page) -> tuple[str, str]:
    """
    Extract text without OCR. Returns (text, source) where source is ``native`` or ``empty``.
    """
    raw = page.get_text("text") or ""
    t = _normalize_text(raw)
    if t:
        return t, "native"
    t = _text_from_text_dict(page)
    if t:
        return t, "native"
    return "", "empty"


def extract_page_text_ocr(page: fitz.Page, dpi: int = 150, language: str = "vie+eng") -> tuple[str, str]:
    """
    Run Tesseract OCR via PyMuPDF (needs ``tesseract-ocr`` and language packs on the system).
    Returns (text, ``ocr``) or (\"\", ``empty``) if OCR is unavailable or fails.
    """
    global _OCR_UNAVAILABLE_LOGGED
    try:
        tp = page.get_textpage_ocr(dpi=dpi, language=language, full=True)
        try:
            raw = page.get_text("text", textpage=tp) or ""
        finally:
            del tp
    except RuntimeError as e:
        if not _OCR_UNAVAILABLE_LOGGED:
            warnings.warn(
                "OCR skipped (Tesseract/tessdata not available). "
                "On Ubuntu: sudo apt install tesseract-ocr tesseract-ocr-vie tesseract-ocr-eng. "
                f"Details: {e}",
                stacklevel=2,
            )
            _OCR_UNAVAILABLE_LOGGED = True
            logger.warning("OCR unavailable: %s", e)
        return "", "empty"
    t = _normalize_text(raw)
    return (t, "ocr") if t else ("", "empty")


def extract_page_text(
    doc: fitz.Document,
    page_index: int,
    *,
    ocr: bool = False,
    ocr_dpi: int = 150,
    ocr_language: str = "vie+eng",
) -> tuple[str, str]:
    """
    Extract text for one page. If ``ocr`` is True and native extraction is empty,
    runs OCR (requires system Tesseract).
    Returns (text, text_source).
    """
    page = doc.load_page(page_index)
    text, src = extract_page_text_native(page)
    if text or not ocr:
        return text, src
    return extract_page_text_ocr(page, dpi=ocr_dpi, language=ocr_language)


def render_page_image(doc: fitz.Document, page_index: int, dpi: float = 144.0) -> Image.Image:
    page = doc.load_page(page_index)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    mode = "RGB" if pix.n < 4 else "RGBA"
    img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def iter_pdf_pages(
    path: str | Path,
    dpi: float = 144.0,
    *,
    ocr: bool = False,
    ocr_dpi: int = 150,
    ocr_language: str = "vie+eng",
) -> Iterator[PageData]:
    path = Path(path)
    doc = fitz.open(path)
    try:
        for i in range(doc.page_count):
            img = render_page_image(doc, i, dpi=dpi)
            text, text_source = extract_page_text(
                doc,
                i,
                ocr=ocr,
                ocr_dpi=ocr_dpi,
                ocr_language=ocr_language,
            )
            yield PageData(page_index=i, image=img, text=text, text_source=text_source)
    finally:
        doc.close()


def page_count(path: str | Path) -> int:
    doc = fitz.open(path)
    try:
        return doc.page_count
    finally:
        doc.close()
