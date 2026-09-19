"""
AION Module: Document Parser (Master Orchestrator)
Combines OCR Engine + Docling + Table Validator into one unified result.
Drop-in replacement for content_filter.py's extract_academic_content().
"""

from __future__ import annotations

import re
import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from .ocr_engine     import _extract_digital, _extract_with_unlimited_ocr, OCRResult
from .docling_parser import parse_with_docling, DoclingResult
from .table_validator import validate_tables, ValidatedTable


def get_file_sha256(file_path: str) -> str:
    """Calculate SHA-256 hex digest of file contents for persistent caching."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def preflight_page_triage(pdf_path: str) -> dict:
    """
    Pillar 2: 3-Tier Pre-Flight Page Triaging (<500ms).
    Uses PyMuPDF to scan page fonts, drawings, and character yield:
      - Tier A: pure digital text pages (~70% of pages)
      - Tier B: complex pages with math fonts, LaTeX, or vector tables (~25%)
      - Tier C: scanned/raster bitmap pages (<5%)
    """
    tier_a = []
    tier_b = []
    tier_c = []
    math_font_signals = ("math", "cmmi", "cmsy", "symbol", "cambria")

    try:
        import fitz
        doc = fitz.open(pdf_path)
        for p_idx, page in enumerate(doc):
            page_num = p_idx + 1
            text = page.get_text() or ""
            word_count = len(text.split())
            images = page.get_images()
            drawings = page.get_drawings()

            # Inspect font list for mathematical fonts
            try:
                font_list = page.get_fonts()
                has_math_font = any(any(sig in f[3].lower() for sig in math_font_signals) for f in font_list if len(f) > 3)
            except Exception:
                has_math_font = False

            has_latex_token = any(sig in text for sig in ("$", "\\sum", "\\int", "\\frac", "\\partial", "\\alpha", "\\beta", "\\theta", "\\sigma", "\\pi"))

            if word_count < 25 and len(images) > 0:
                tier_c.append(page_num)
            elif has_math_font or has_latex_token or len(drawings) > 15:
                tier_b.append(page_num)
            else:
                tier_a.append(page_num)
        doc.close()
    except Exception as e:
        print(f"[TRIAGE] Pre-flight triage fallback: {e}", flush=True)

    return {
        "tier_a_pure_text": tier_a,
        "tier_b_complex": tier_b,
        "tier_c_scanned": tier_c,
        "total_pages": len(tier_a) + len(tier_b) + len(tier_c),
    }


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
    Master document parser with SHA-256 caching and 3-Tier Pre-Flight Triage.

    Priority:
    1. Check persistent SHA-256 extraction cache (.aion_cache/extraction/)
    2. Pre-flight page triage (Tier A/B/C)
    3. PyMuPDF digital extraction
    4. Unlimited-OCR for scanned pages
    5. Run Docling / MinerU for layout + tables
    6. Cross-validate tables and cache result
    """
    warnings    = []
    ocr_result  = None
    doc_result  = None
    ocr_used    = False
    cache_file  = None
    cache_dir   = None
    file_hash   = None

    # Pillar 1: Persistent Content-Hash Caching (SHA-256)
    try:
        file_hash = get_file_sha256(pdf_path)
        cache_dir = Path(".aion_cache/extraction") / file_hash
        cache_file = cache_dir / "cached_doc.json"
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"[PARSER] Instant SHA-256 Cache Hit ({file_hash[:8]}...) -> Loaded in 0.00s", flush=True)
            return ParsedDocument(
                text=data["text"],
                tables=[ValidatedTable(**t) for t in data["tables"]],
                figures=data["figures"],
                structure=data["structure"],
                method="sha256_cached",
                ocr_used=data["ocr_used"],
                pages_total=data["pages_total"],
                word_count=data["word_count"],
                confidence=data["confidence"],
                warnings=data.get("warnings", []),
            )
    except Exception as e:
        print(f"[PARSER] Cache check exception: {e}", flush=True)

    print(f"[PARSER] Processing: {Path(pdf_path).name}")

    # Pillar 2: Pre-Flight Page Triage
    triage = preflight_page_triage(pdf_path)
    if triage["total_pages"] > 0:
        print(
            f"[PARSER] Triage complete: {triage['total_pages']} pages | "
            f"Tier A (text): {len(triage['tier_a_pure_text'])} | "
            f"Tier B (math/tables): {len(triage['tier_b_complex'])} | "
            f"Tier C (scanned): {len(triage['tier_c_scanned'])}",
            flush=True
        )

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

    parsed_doc = ParsedDocument(
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

    if cache_dir and cache_file:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            serializable_tables = [asdict(t) for t in validated_tables]
            cache_data = {
                "text": clean_text,
                "tables": serializable_tables,
                "figures": figures,
                "structure": doc_result.structure if doc_result else [],
                "method": method,
                "ocr_used": ocr_used,
                "pages_total": ocr_result.pages_total if ocr_result else 0,
                "word_count": word_count,
                "confidence": confidence,
                "warnings": warnings,
            }
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            print(f"[PARSER] Saved parsed document to SHA-256 cache ({file_hash[:8]}...)", flush=True)
        except Exception as e:
            print(f"[PARSER] Cache write exception: {e}", flush=True)

    return parsed_doc


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
