"""
AION Module: Document Parser (Master Orchestrator)
Combines OCR Engine + Docling + Table Validator into one unified result.
Drop-in replacement for content_filter.py's extract_academic_content().
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .ocr_engine     import _extract_digital, _extract_with_unlimited_ocr, OCRResult
from .docling_parser import parse_with_docling, DoclingResult
from .table_validator import validate_tables, ValidatedTable


@dataclass
class ParsedDocument:
    """Unified document parse result."""
    text:           str                    # clean academic text
    tables:         list[ValidatedTable]   # validated tables as markdown
    figures:        list[dict]             # figure captions + metadata
    structure:      list[dict]             # heading hierarchy
    method:         str                    # which pipeline was used
    ocr_used:       bool
    pages_total:    int
    word_count:     int
    confidence:     float
    warnings:       list[str] = field(default_factory=list)

    def full_text_with_tables(self) -> str:
        """Returns text with tables embedded at correct positions."""
        parts = [self.text]
        for tbl in self.tables:
            if tbl.confidence >= 0.6:
                parts.append(
                    f"\n[TABLE — Page {tbl.page} | "
                    f"Confidence: {tbl.confidence:.0%}]\n"
                    f"{tbl.markdown}\n"
                )
        return "\n\n".join(parts)


def parse_document(
    pdf_path:   str,
    use_docling: bool = True,
    use_ocr:     bool = True,
    min_confidence: float = 0.60,
) -> ParsedDocument:
    """
    Master document parser.

    Priority:
    1. Try PyMuPDF digital extraction (fast)
    2. If text yield too low -> Unlimited-OCR (scanned)
    3. Run Docling in parallel for layout + tables
    4. Cross-validate tables
    5. Merge best result
    """
    warnings    = []
    ocr_result  = None
    doc_result  = None
    ocr_used    = False

    print(f"[PARSER] Processing: {Path(pdf_path).name}")

    # -- Step 1: Try digital extraction -----------------------
    ocr_result = _extract_digital(pdf_path)

    if ocr_result is None:
        print("[PARSER] Digital text yield too low — activating OCR...")
        ocr_used   = True
        ocr_result = _extract_with_unlimited_ocr(pdf_path)
        warnings.append("ocr_used:scanned_pdf_detected")

    if ocr_result:
        print(
            f"[PARSER] OCR complete: {len(ocr_result.blocks)} blocks | "
            f"method={ocr_result.method}"
        )

    # -- Step 2: Run Docling for layout + tables ---------------
    if use_docling:
        doc_result = parse_with_docling(pdf_path)
        if doc_result:
            print(
                f"[PARSER] Docling complete: "
                f"{len(doc_result.tables)} tables | "
                f"{len(doc_result.structure)} headings"
            )
        else:
            warnings.append("docling_failed:using_ocr_only")

    # -- Step 3: Cross-validate tables ------------------------
    docling_tables = doc_result.tables if doc_result else []
    ocr_tables     = [
        {"page": b.page, "content": b.content, "confidence": b.confidence}
        for b in (ocr_result.blocks if ocr_result else [])
        if b.block_type == "table"
    ]
    validated_tables = validate_tables(docling_tables, ocr_tables)

    if validated_tables:
        print(f"[PARSER] Tables validated: {len(validated_tables)}")
        for t in validated_tables:
            status = "✓" if t.confidence >= min_confidence else "⚠"
            print(
                f"  {status} Page {t.page:3d} | "
                f"conf={t.confidence:.0%} | "
                f"source={t.source}"
                + (f" | warnings={t.warnings}" if t.warnings else "")
            )

    # -- Step 4: Figures ---------------------------------------
    figures = [
        {
            "page":    b.page,
            "caption": b.content,
            "bbox":    b.bbox,
        }
        for b in (ocr_result.blocks if ocr_result else [])
        if b.block_type == "figure"
    ]

    # -- Step 5: Build clean text ------------------------------
    # Priority: Docling layout text > OCR raw text
    if doc_result and doc_result.layout_text:
        raw_text = doc_result.layout_text
        method   = "docling+ocr" if ocr_used else "docling+pymupdf"
    elif ocr_result and ocr_result.raw_text:
        raw_text = ocr_result.raw_text
        method   = ocr_result.method
    else:
        raw_text = ""
        method   = "failed"
        warnings.append("no_text_extracted")

    # -- Step 6: Clean the text --------------------------------
    clean_text = _clean_extracted_text(raw_text)
    word_count = len(clean_text.split())

    # -- Step 7: Overall confidence ----------------------------
    confidence = _compute_confidence(
        ocr_result, doc_result, validated_tables, word_count
    )

    print(
        f"[PARSER] Done: {word_count:,} words | "
        f"method={method} | "
        f"confidence={confidence:.0%}"
    )

    return ParsedDocument(
        text         = clean_text,
        tables       = validated_tables,
        figures      = figures,
        structure    = doc_result.structure if doc_result else [],
        method       = method,
        ocr_used     = ocr_used,
        pages_total  = ocr_result.pages_total if ocr_result else 0,
        word_count   = word_count,
        confidence   = confidence,
        warnings     = warnings,
    )


def _clean_extracted_text(text: str) -> str:
    """Final cleaning pass on extracted text."""
    import re

    # Remove page numbers
    text = re.sub(r"\n\s*\d{1,4}\s*\n", "\n", text)
    # Remove repeated whitespace
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    text = re.sub(r"[ \t]{3,}", " ", text)
    # Remove markdown image tags (from Docling output)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    # Remove HTML tags if any
    text = re.sub(r"<[^>]+>", "", text)

    return text.strip()


def _compute_confidence(
    ocr_result,
    doc_result,
    tables,
    word_count: int,
) -> float:
    """Compute overall parse confidence."""
    score = 0.0

    if ocr_result:
        avg_block_conf = (
            sum(b.confidence for b in ocr_result.blocks) /
            max(len(ocr_result.blocks), 1)
        )
        score += avg_block_conf * 0.4

    if doc_result:
        score += 0.3

    if word_count > 1000:
        score += 0.2
    elif word_count > 200:
        score += 0.1

    if tables:
        avg_table_conf = sum(t.confidence for t in tables) / len(tables)
        score += avg_table_conf * 0.1

    return min(round(score, 2), 1.0)
