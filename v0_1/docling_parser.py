"""
AION Module: Docling Parser
Extracts layout, tables, and document structure using Docling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DoclingTable:
    """A validated table extracted by Docling."""
    page:        int
    headers:     list[str]
    rows:        list[list[str]]
    confidence:  float
    has_merged:  bool = False
    markdown:    str  = ""

    def to_markdown(self) -> str:
        if self.markdown:
            return self.markdown
        lines = []
        if self.headers:
            lines.append("| " + " | ".join(self.headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(self.headers)) + " |")
        for row in self.rows:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        return "\n".join(lines)


@dataclass
class DoclingResult:
    """Full Docling parse result."""
    structure:   list[dict]      # heading hierarchy tree
    tables:      list[DoclingTable]
    layout_text: str             # layout-aware plain text
    method:      str = "docling"


def parse_with_docling(pdf_path: str) -> Optional[DoclingResult]:
    """
    Uses Docling to extract layout, tables, and document structure.
    Returns None if Docling is not installed or fails.
    """
    try:
        from docling.document_converter import DocumentConverter
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions

        options = PdfPipelineOptions()
        options.do_ocr              = False   # OCR handled by ocr_engine.py
        options.do_table_structure  = True
        options.table_structure_options.do_cell_matching = True

        converter = DocumentConverter()
        result    = converter.convert(pdf_path)
        doc       = result.document

        # ── Extract structure (heading hierarchy) ─────────────
        structure = []
        for item in doc.iterate_items():
            label = getattr(item, "label", "")
            text  = getattr(item, "text", "")
            if "heading" in str(label).lower() and text:
                level = _get_heading_level(str(label))
                structure.append({
                    "level": level,
                    "text":  text.strip(),
                })

        # ── Extract tables ────────────────────────────────────
        tables = []
        for table_item in getattr(doc, "tables", []):
            try:
                df          = table_item.export_to_dataframe()
                headers     = list(df.columns)
                rows        = df.values.tolist()
                page        = getattr(table_item, "page_no", 0)
                has_merged  = _detect_merged_cells(table_item)
                confidence  = 0.90 if not has_merged else 0.75

                tables.append(DoclingTable(
                    page       = page,
                    headers    = [str(h) for h in headers],
                    rows       = [[str(c) for c in row] for row in rows],
                    confidence = confidence,
                    has_merged = has_merged,
                ))
            except Exception:
                continue

        # ── Layout-aware text ─────────────────────────────────
        try:
            layout_text = doc.export_to_markdown()
        except Exception:
            layout_text = "\n\n".join(
                item.text
                for item in doc.iterate_items()
                if hasattr(item, "text") and item.text
            )

        return DoclingResult(
            structure    = structure,
            tables       = tables,
            layout_text  = layout_text,
            method       = "docling",
        )

    except ImportError:
        print("[DOCLING] Not installed. Run: pip install docling")
        return None
    except Exception as e:
        print(f"[DOCLING] Parse error: {e}")
        return None


def _get_heading_level(label: str) -> int:
    """Extract heading level from Docling label."""
    import re
    match = re.search(r"\d+", label)
    if match:
        return int(match.group())
    if "section" in label.lower():
        return 2
    if "chapter" in label.lower():
        return 1
    return 2


def _detect_merged_cells(table_item) -> bool:
    """Detect merged/spanned cells in a Docling table."""
    try:
        for cell in getattr(table_item, "table_cells", []):
            col_span = getattr(cell, "col_span", 1)
            row_span = getattr(cell, "row_span", 1)
            if col_span > 1 or row_span > 1:
                return True
    except Exception:
        pass
    return False
