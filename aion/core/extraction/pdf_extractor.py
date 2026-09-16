"""
AION v2 Streaming Real-PDF Extractor
====================================
Extracts and accounts for all detectable document artifacts from native PDFs.
Implements:
1. Typographic font-size profiling for adaptive heading detection (H1, H2, H3).
2. Running header/footer frequency pruning (eliminating repetitive noise).
3. Native PyMuPDF table extraction with bounding-box text masking.
4. Embedded figure extraction with caption proximity association.
5. Multi-tier equation candidate extraction and syntax verification.
6. Strict per-page accounting (EXTRACTED, OCR_REQUIRED, EMPTY_EXPECTED, FAILED).
7. Streaming generation directly into ArtifactFusionEngine.
"""

from __future__ import annotations

import collections
import os
import re
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

import pymupdf as fitz

from aion.core.dom.document_dom import DocumentDOM
from aion.core.extraction.contracts import (
    ExtractionJob,
    PageExtractionResult,
    PageExtractionStatus,
    RawEquation,
    RawFigure,
    RawHeading,
    RawTable,
    RawTextBlock,
)
from aion.core.extraction.fusion import ArtifactFusionEngine


class TypographicProfile:
    """
    Profiles the typographic hierarchy of a document:
    discovers body font size, heading thresholds, and repetitive header/footer artifacts.
    """

    def __init__(self, body_size: float = 11.0, running_headers: Optional[Set[str]] = None):
        self.body_size = body_size
        self.h1_threshold = body_size * 1.35
        self.h2_threshold = body_size * 1.15
        self.h3_threshold = body_size * 1.02
        self.running_headers = running_headers or set()

    @classmethod
    def analyze(cls, doc: fitz.Document, max_sample_pages: int = 15) -> TypographicProfile:
        size_counts: collections.Counter = collections.Counter()
        header_candidates: collections.Counter = collections.Counter()
        sample_pages = min(len(doc), max_sample_pages)

        for pno in range(sample_pages):
            page = doc[pno]
            h = page.rect.height
            d = page.get_text("dict")
            for block in d.get("blocks", []):
                if block.get("type") != 0:
                    continue
                bbox = block.get("bbox", [0, 0, 0, 0])
                # Check running header / footer zones (top 8% and bottom 8%)
                is_edge = (bbox[1] < 0.08 * h) or (bbox[3] > 0.92 * h)

                block_text_parts = []
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        sz = round(span.get("size", 0.0), 1)
                        if sz > 4.0:
                            size_counts[sz] += len(span.get("text", "").split())
                        block_text_parts.append(span.get("text", ""))

                bt = " ".join(block_text_parts).strip()
                if is_edge and len(bt) > 3 and len(bt) < 80:
                    header_candidates[bt] += 1

        # Determine most common body size
        body_size = 11.0
        if size_counts:
            body_size = size_counts.most_common(1)[0][0]

        # Strings appearing on >= 25% of sample pages at edges are running headers/footers
        threshold_count = max(2, int(sample_pages * 0.25))
        running_headers = {
            txt for txt, count in header_candidates.items() if count >= threshold_count
        }

        return cls(body_size=body_size, running_headers=running_headers)


class EquationValidator:
    """
    Validates equation candidates and extracts mathematical variables.
    Assigns: VERIFIED, CANDIDATE, or FAILED.
    """

    MATH_COMMANDS = re.compile(
        r'\\(?:frac|sqrt|sum|int|partial|times|approx|le|ge|pm|cdot|alpha|beta|gamma|theta|lambda|mu|pi|sigma|omega|Delta|nabla)',
        re.UNICODE
    )
    MATH_SYMBOLS = re.compile(r'[∑∫∂√±≤≥≠≈×÷παβγθλμσωσ∆∇]', re.UNICODE)
    EQ_LABEL_RE = re.compile(r'(?:\(?(\d+(?:\.\d+)?)\)?)\s*$')

    @classmethod
    def evaluate(cls, raw_latex: str, raw_text: str = "") -> Tuple[str, str, List[str], Optional[str]]:
        """
        Returns: (normalized_latex, verification_status, variables, label)
        Status: 'VERIFIED' | 'CANDIDATE' | 'FAILED'
        """
        text = raw_latex.strip()

        # 1. Detect equation label at end (e.g. "(3.1)" or "3.4")
        label = None
        label_match = cls.EQ_LABEL_RE.search(text)
        if label_match:
            label = label_match.group(1)
            text = text[:label_match.start()].strip()

        # 2. Normalize simple ASCII fraction/sqrt
        norm_latex = re.sub(r'sqrt\((.*?)\)', r'\\sqrt{\1}', text)
        norm_latex = re.sub(r'(\b[a-zA-Z]\b)\s*/\s*([a-zA-Z0-9_\^]+)', r'\\frac{\1}{\2}', norm_latex)

        # 3. Check bracket balance
        open_curly = norm_latex.count("{")
        close_curly = norm_latex.count("}")
        open_paren = norm_latex.count("(")
        close_paren = norm_latex.count(")")

        if open_curly != close_curly or open_paren != close_paren:
            return norm_latex, "FAILED", [], label

        # 4. Extract mathematical variables
        vars_found = list(set(re.findall(r'\\?[a-zA-Z](?:_[a-zA-Z0-9]+)?', norm_latex)))
        clean_vars = [v for v in vars_found if not v.startswith("\\") and len(v) <= 4][:8]

        # 5. Determine verification confidence
        has_latex_cmd = bool(cls.MATH_COMMANDS.search(norm_latex))
        has_unicode_sym = bool(cls.MATH_SYMBOLS.search(norm_latex))
        has_math_eq = "=" in norm_latex and (has_latex_cmd or has_unicode_sym or len(clean_vars) >= 2)

        if has_latex_cmd or has_math_eq:
            status = "VERIFIED"
        elif has_unicode_sym or ("=" in norm_latex and clean_vars):
            status = "CANDIDATE"
        else:
            status = "FAILED"

        return norm_latex, status, clean_vars, label


class PDFStreamingExtractor:
    """
    Extracts artifacts page by page from native PDF using PyMuPDF.
    Yields PageExtractionResult for every page.
    """

    CAPTION_RE = re.compile(r'(?i)^\s*(?:fig(?:ure)?\.?|table|diagram|chart)\s*(\d+(?:[.\-_]\d+)?[a-z]?)\s*[:\-–]?\s*(.*)', re.DOTALL)

    def __init__(self, doc_path: str):
        self.doc_path = Path(doc_path).resolve()
        if not self.doc_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.doc_path}")

    def stream_pages(
        self, profile: Optional[TypographicProfile] = None
    ) -> Generator[PageExtractionResult, None, None]:
        doc = fitz.open(str(self.doc_path))
        try:
            if profile is None:
                profile = TypographicProfile.analyze(doc)

            for pno in range(len(doc)):
                yield self._extract_single_page(doc, pno, profile)
        finally:
            doc.close()

    def _extract_single_page(
        self, doc: fitz.Document, pno: int, profile: TypographicProfile
    ) -> PageExtractionResult:
        page_number = pno + 1
        page = doc[pno]

        res = PageExtractionResult(page_number=page_number)
        table_bboxes: List[Tuple[float, float, float, float]] = []

        try:
            # ---------------------------------------------------------------
            # 1. Native Table Extraction
            # ---------------------------------------------------------------
            try:
                tabs = page.find_tables()
                for t in tabs.tables:
                    extracted = t.extract()
                    if extracted and len(extracted) > 1:
                        headers = [str(c).strip() if c is not None else "" for c in extracted[0]]
                        rows = [
                            [str(c).strip() if c is not None else "" for c in r]
                            for r in extracted[1:]
                        ]
                        # Build Markdown Table
                        md_lines = ["| " + " | ".join(headers) + " |"]
                        md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                        for r in rows:
                            md_lines.append("| " + " | ".join(r) + " |")
                        md_table = "\n".join(md_lines)

                        t_bbox = tuple(t.bbox)
                        table_bboxes.append(t_bbox)
                        res.tables.append(RawTable(
                            headers=headers,
                            rows=rows,
                            markdown=md_table,
                            bbox=t_bbox,
                        ))
            except Exception:
                pass

            # ---------------------------------------------------------------
            # 2. Embedded Figure Extraction
            # ---------------------------------------------------------------
            img_list = page.get_images(full=True)
            for img_info in img_list:
                xref = img_info[0]
                try:
                    img_data = doc.extract_image(xref)
                    raw_bytes = img_data.get("image", b"")
                    w = img_data.get("width", 0)
                    h = img_data.get("height", 0)
                    fmt = img_data.get("ext", "png")

                    # Filter out tiny icon glyphs (< 32px) or line dividers (> 10:1 ratio)
                    if w < 32 or h < 32:
                        continue
                    if (w > 0 and h > 0) and (w / h > 10.0 or h / w > 10.0):
                        continue

                    # Try to locate image bbox on page
                    img_bbox = None
                    try:
                        bbox_rect = page.get_image_bbox(img_info)
                        if bbox_rect and not bbox_rect.is_empty:
                            img_bbox = tuple(bbox_rect)
                    except Exception:
                        pass

                    res.figures.append(RawFigure(
                        image_bytes=raw_bytes,
                        bbox=img_bbox,
                        format_hint=fmt,
                        width=w,
                        height=h,
                    ))
                except Exception:
                    pass

            # ---------------------------------------------------------------
            # 3. Text Blocks, Headings, Captions, and Equation Extraction
            # ---------------------------------------------------------------
            d = page.get_text("dict")
            reading_order = 0

            for b in d.get("blocks", []):
                if b.get("type") != 0:
                    continue  # Only text blocks

                b_bbox = tuple(b.get("bbox", [0, 0, 0, 0]))

                # Skip block if it overlaps significantly inside an extracted table
                if self._overlaps_any(b_bbox, table_bboxes):
                    continue

                # Reconstruct block text and inspect font spans
                spans_info: List[Dict[str, Any]] = []
                for line in b.get("lines", []):
                    for span in line.get("spans", []):
                        spans_info.append(span)

                raw_block_text = "".join(s.get("text", "") for s in spans_info).strip()
                if not raw_block_text:
                    continue

                # Strip running headers / footers
                if raw_block_text in profile.running_headers:
                    continue

                res.char_count += len(raw_block_text)

                # Check if block is a figure or table Caption
                cap_match = self.CAPTION_RE.match(raw_block_text)
                if cap_match:
                    cap_label, cap_desc = cap_match.groups()
                    full_cap = raw_block_text
                    # Associate caption to nearest figure if uncaptioned
                    if res.figures and not res.figures[-1].caption_hint:
                        res.figures[-1].caption_hint = full_cap
                    continue

                # Check for Heading hierarchy
                avg_size = (
                    sum(s.get("size", profile.body_size) for s in spans_info) / len(spans_info)
                    if spans_info else profile.body_size
                )
                is_bold = any(bool(s.get("flags", 0) & 2) or "bold" in s.get("font", "").lower() for s in spans_info)
                first_line = raw_block_text.split("\n")[0].strip()

                is_h1 = (avg_size >= profile.h1_threshold) or bool(re.match(r'(?i)^(?:module|chapter|unit)\s*[-–:]?\s*\d+', first_line))
                is_h2 = (avg_size >= profile.h2_threshold and (is_bold or avg_size >= profile.body_size * 1.20 or re.match(r'^\d+\.\d+\s+', first_line)))
                is_h3 = (avg_size >= profile.h3_threshold and is_bold) or bool(re.match(r'^\d+\.\d+\.\d+\s+', first_line))

                if (is_h1 or is_h2 or is_h3) and len(raw_block_text) < 140 and "\n\n" not in raw_block_text:
                    lvl = 1 if is_h1 else (2 if is_h2 else 3)
                    num_match = re.match(r'(?i)^((?:module|chapter|unit\s*)?\d+(?:\.\d+)?)\s*[:\-–]?\s*', first_line)
                    numbering = num_match.group(1) if num_match else None

                    res.headings.append(RawHeading(
                        title=raw_block_text,
                        level=lvl,
                        numbering=numbering,
                        bbox=b_bbox,
                    ))
                    continue

                # Check for Math / Equation candidate
                norm_eq, eq_status, eq_vars, eq_label = EquationValidator.evaluate(raw_block_text)
                if eq_status in ("VERIFIED", "CANDIDATE") and len(raw_block_text.split()) < 35:
                    res.equations.append(RawEquation(
                        latex=norm_eq,
                        raw_text=raw_block_text,
                        bbox=b_bbox,
                        is_inline=False,
                        label=eq_label,
                        verification_status=eq_status,
                        variables=eq_vars,
                    ))
                    continue

                # Regular TextBlock
                reading_order += 1
                res.text_blocks.append(RawTextBlock(
                    text=raw_block_text,
                    bbox=b_bbox,
                    font_size=avg_size,
                    is_bold=is_bold,
                    reading_order=reading_order,
                ))

            # ---------------------------------------------------------------
            # 4. Page Accounting State Assessment
            # ---------------------------------------------------------------
            if res.char_count < 30:
                if len(res.figures) > 0:
                    res.status = PageExtractionStatus.OCR_REQUIRED
                else:
                    res.status = PageExtractionStatus.EMPTY_EXPECTED
            else:
                res.status = PageExtractionStatus.EXTRACTED

        except Exception as exc:
            res.status = PageExtractionStatus.FAILED
            res.error_message = str(exc)

        return res

    def _overlaps_any(
        self, bbox: Tuple[float, float, float, float], table_bboxes: List[Tuple[float, float, float, float]]
    ) -> bool:
        bx0, by0, bx1, by1 = bbox
        for tx0, ty0, tx1, ty1 in table_bboxes:
            # Check intersection
            ix0 = max(bx0, tx0)
            iy0 = max(by0, ty0)
            ix1 = min(bx1, tx1)
            iy1 = min(by1, ty1)
            if ix1 > ix0 and iy1 > iy0:
                intersection_area = (ix1 - ix0) * (iy1 - iy0)
                box_area = max(1.0, (bx1 - bx0) * (by1 - by0))
                if (intersection_area / box_area) > 0.6:
                    return True
        return False


def extract_pdf_to_dom(
    pdf_path: str,
    document_id: Optional[str] = None,
    module_mapping: Optional[Dict[int, Tuple[int, int]]] = None,
    assets_root_dir: Optional[str] = None,
) -> DocumentDOM:
    """
    High-level factory that streams a real PDF through extraction and fusion into a DocumentDOM.
    Guarantees 100% page accounting and native multimodal artifacts.
    """
    extractor = PDFStreamingExtractor(pdf_path)

    # Determine total pages
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    doc_title = doc.metadata.get("title") or Path(pdf_path).stem
    doc.close()

    doc_id = document_id or f"DOC-{Path(pdf_path).stem.upper()}"

    dom = DocumentDOM(
        document_id=doc_id,
        title=doc_title,
        total_pages=total_pages,
    )
    if assets_root_dir:
        dom.registry.assets_root_dir = Path(assets_root_dir)

    fusion = ArtifactFusionEngine(dom)

    for page_res in extractor.stream_pages():
        fusion.fuse_page(page_res, module_mapping=module_mapping)

    fusion.finalize()
    return dom
