"""
AION Module: OCR Engine
Wraps Unlimited-OCR for scanned PDF handling.
Falls back to PyMuPDF for digital PDFs.
"""

from __future__ import annotations

import re
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OCRBlock:
    """A single extracted content block."""
    block_type:  str        # "paragraph" | "heading" | "figure" | "table"
    content:     str
    page:        int
    confidence:  float
    bbox:        Optional[tuple] = None   # (x0, y0, x1, y1)
    level:       int = 0                  # heading level: 1=chapter, 2=section


@dataclass
class OCRResult:
    """Full OCR result for a document."""
    blocks:         list[OCRBlock]
    pages_total:    int
    pages_ocr_used: int          # pages where OCR was needed
    method:         str          # "pymupdf" | "unlimited_ocr" | "hybrid"
    has_tables:     bool = False
    has_figures:    bool = False
    raw_text:       str  = ""


# -------------------------------------------------------------
# Digital PDF extractor (existing PyMuPDF — keep as-is)
# -------------------------------------------------------------

def _extract_digital(pdf_path: str) -> Optional[OCRResult]:
    """
    Fast path: extract text from digital PDFs using PyMuPDF.
    Returns None if the PDF is scanned (text yield < threshold).
    """
    try:
        import fitz
        doc    = fitz.open(pdf_path)
        blocks = []
        total_words = 0

        pages = len(doc) if hasattr(doc, '__len__') else 1

        for page_num, page in enumerate(doc, 1):
            # Try structured block extraction first
            raw_blocks = page.get_text("dict")["blocks"]

            for block in raw_blocks:
                if "lines" not in block:
                    continue

                block_text = ""
                max_size   = 0
                is_bold    = False

                for line in block["lines"]:
                    for span in line["spans"]:
                        block_text += span["text"] + " "
                        if span["size"] > max_size:
                            max_size = span["size"]
                        if "bold" in span["font"].lower():
                            is_bold = True

                block_text = block_text.strip()
                if not block_text or len(block_text) < 3:
                    continue

                total_words += len(block_text.split())

                # Classify block type by font size
                block_type, level = _classify_block(
                    block_text, max_size, is_bold, page
                )

                blocks.append(OCRBlock(
                    block_type = block_type,
                    content    = block_text,
                    page       = page_num,
                    confidence = 0.95,
                    bbox       = block["bbox"],
                    level      = level,
                ))

        doc.close()

        # If very few words extracted, PDF is likely scanned
        words_per_page = total_words / max(pages, 1)

        if words_per_page < 50:
            return None   # Signal to use OCR fallback

        raw_text = "\n\n".join(
            b.content for b in blocks
            if b.block_type == "paragraph"
        )

        return OCRResult(
            blocks         = blocks,
            pages_total    = pages,
            pages_ocr_used = 0,
            method         = "pymupdf",
            raw_text       = raw_text,
        )

    except ImportError:
        return None
    except Exception as e:
        print(f"[OCR] PyMuPDF error: {e}")
        return None


def _classify_block(
    text: str,
    font_size: float,
    is_bold: bool,
    page,
) -> tuple[str, int]:
    """
    Classifies a PDF block as paragraph/heading/table/figure.
    Returns (block_type, heading_level).
    """
    text_lower = text.lower().strip()

    # Figure captions
    if re.match(r"^(figure|fig\.?|diagram|illustration)\s*\d*", text_lower):
        return "figure", 0

    # Table captions
    if re.match(r"^(table|tab\.?)\s*\d*", text_lower):
        return "table", 0

    # Get body font size from page
    try:
        spans = [
            span
            for block in page.get_text("dict")["blocks"]
            if "lines" in block
            for line in block["lines"]
            for span in line["spans"]
        ]
        sizes     = [s["size"] for s in spans if s["text"].strip()]
        body_size = max(set(sizes), key=sizes.count) if sizes else 12
    except Exception:
        body_size = 12

    # Heading detection by font size delta
    size_ratio = font_size / max(body_size, 1)

    if size_ratio >= 1.6 or (
        is_bold and re.match(
            r"^(chapter|module|unit|part|section)\s+[\divxlc]+",
            text_lower
        )
    ):
        return "heading", 1

    if size_ratio >= 1.3 or (
        is_bold and len(text.split()) < 12
        and text[0].isupper()
    ):
        return "heading", 2

    if size_ratio >= 1.1 and is_bold:
        return "heading", 3

    return "paragraph", 0


# -------------------------------------------------------------
# Unlimited-OCR extractor (scanned PDFs)
# -------------------------------------------------------------

def _extract_with_unlimited_ocr(pdf_path: str) -> Optional[OCRResult]:
    """
    Uses Unlimited-OCR for scanned or image-based PDFs.
    Extracts paragraphs, headings, figures, and candidate tables.
    """
    try:
        from unlimited_ocr import DocumentOCR

        ocr    = DocumentOCR()
        result = ocr.process(pdf_path)

        blocks      = []
        ocr_pages   = 0
        has_tables  = False
        has_figures = False

        for page_result in getattr(result, "pages", []):
            ocr_pages += 1

            # -- Paragraphs ------------------------------------
            for para in getattr(page_result, "paragraphs", []):
                text = getattr(para, "text", "").strip()
                if text and len(text.split()) > 3:
                    blocks.append(OCRBlock(
                        block_type = "paragraph",
                        content    = text,
                        page       = getattr(page_result, "page_number", ocr_pages),
                        confidence = getattr(para, "confidence", 0.85),
                    ))

            # -- Headings --------------------------------------
            for heading in getattr(page_result, "headings", []):
                text  = getattr(heading, "text", "").strip()
                level = getattr(heading, "level", 2)
                if text:
                    blocks.append(OCRBlock(
                        block_type = "heading",
                        content    = text,
                        page       = getattr(page_result, "page_number", ocr_pages),
                        confidence = getattr(heading, "confidence", 0.90),
                        level      = level,
                    ))

            # -- Figures ---------------------------------------
            for figure in getattr(page_result, "figures", []):
                caption = getattr(figure, "caption", "") or ""
                has_figures = True
                blocks.append(OCRBlock(
                    block_type = "figure",
                    content    = caption,
                    page       = getattr(page_result, "page_number", ocr_pages),
                    confidence = getattr(figure, "confidence", 0.80),
                    bbox       = getattr(figure, "bbox", None),
                ))

            # -- Candidate tables ------------------------------
            for table in getattr(page_result, "candidate_tables", []):
                raw = getattr(table, "raw_text", "") or ""
                has_tables = True
                blocks.append(OCRBlock(
                    block_type = "table",
                    content    = raw,
                    page       = getattr(page_result, "page_number", ocr_pages),
                    confidence = getattr(table, "confidence", 0.75),
                ))

        raw_text = "\n\n".join(
            b.content for b in blocks
            if b.block_type in ("paragraph", "heading") and b.content
        )

        return OCRResult(
            blocks         = blocks,
            pages_total    = ocr_pages,
            pages_ocr_used = ocr_pages,
            method         = "unlimited_ocr",
            has_tables     = has_tables,
            has_figures    = has_figures,
            raw_text       = raw_text,
        )

    except ImportError:
        print("[OCR] unlimited-ocr not installed. Run: pip install unlimited-ocr")
        return _extract_with_tesseract_fallback(pdf_path)
    except Exception as e:
        print(f"[OCR] Unlimited-OCR error: {e}")
        return _extract_with_tesseract_fallback(pdf_path)


def _extract_with_tesseract_fallback(pdf_path: str) -> Optional[OCRResult]:
    """
    Last-resort OCR using Tesseract + pdf2image.
    Works on any scanned PDF without Unlimited-OCR installed.
    """
    try:
        from pdf2image import convert_from_path
        import pytesseract

        images  = convert_from_path(pdf_path, dpi=200)
        blocks  = []

        for page_num, image in enumerate(images, 1):
            text = pytesseract.image_to_string(image, lang="eng")
            if text.strip():
                blocks.append(OCRBlock(
                    block_type = "paragraph",
                    content    = text.strip(),
                    page       = page_num,
                    confidence = 0.70,
                ))

        raw_text = "\n\n".join(b.content for b in blocks)
        return OCRResult(
            blocks         = blocks,
            pages_total    = len(images),
            pages_ocr_used = len(images),
            method         = "tesseract_fallback",
            raw_text       = raw_text,
        )

    except ImportError:
        print("[OCR] Neither unlimited-ocr nor tesseract available.")
        return None
    except Exception as e:
        print(f"[OCR] Tesseract fallback error: {e}")
        return None
