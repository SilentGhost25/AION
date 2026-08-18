"""
AION Module: Table Validator
Cross-validates tables from OCR and Docling outputs.
Detects merged rows, missing tables, and scores confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from .docling_parser import DoclingTable


@dataclass
class ValidatedTable:
    """A table that has passed cross-validation."""
    page:        int
    markdown:    str
    confidence:  float
    source:      str          # "docling" | "ocr" | "merged"
    warnings:    list[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


def validate_tables(
    docling_tables: list[DoclingTable],
    ocr_tables:     list[dict],
) -> list[ValidatedTable]:
    """
    Cross-validates tables from both sources.

    Strategy:
    1. Docling tables are primary (higher structural accuracy)
    2. OCR tables fill gaps where Docling missed
    3. Merged cell tables get a repair attempt
    4. Low confidence tables get a warning flag

    Returns list of ValidatedTable sorted by page.
    """
    validated = []

    # -- Step 1: Process Docling tables (primary) --------------
    docling_pages = set()
    for tbl in docling_tables:
        warnings = []

        # Check for merged cells
        if tbl.has_merged:
            warnings.append("merged_cells_detected")
            repaired = _repair_merged_table(tbl)
            if repaired:
                tbl = repaired
                warnings.append("merged_cells_repaired")

        # Check for empty rows
        empty_rows = sum(
            1 for row in tbl.rows
            if all(not str(c).strip() for c in row)
        )
        if empty_rows > 0:
            warnings.append(f"empty_rows:{empty_rows}")

        # Check minimum size
        if len(tbl.rows) < 1 or len(tbl.headers) < 1:
            warnings.append("table_too_small")
            confidence = max(tbl.confidence - 0.2, 0.3)
        else:
            confidence = tbl.confidence

        validated.append(ValidatedTable(
            page       = tbl.page,
            markdown   = tbl.to_markdown(),
            confidence = confidence,
            source     = "docling",
            warnings   = warnings,
        ))
        docling_pages.add(tbl.page)

    # -- Step 2: Fill gaps from OCR tables ---------------------
    for ocr_tbl in ocr_tables:
        page = ocr_tbl.get("page", 0)
        text = ocr_tbl.get("content", "")

        # Only use OCR table if Docling didn't find one on this page
        if page not in docling_pages and text.strip():
            confidence = ocr_tbl.get("confidence", 0.60)
            validated.append(ValidatedTable(
                page       = page,
                markdown   = _ocr_text_to_markdown(text),
                confidence = confidence,
                source     = "ocr",
                warnings   = ["ocr_only_no_docling_confirmation"],
            ))

    # -- Step 3: Detect missing tables via gap analysis --------
    if validated:
        all_pages = sorted(v.page for v in validated)
        for i in range(len(all_pages) - 1):
            gap = all_pages[i + 1] - all_pages[i]
            if gap > 15:
                # Large gap between tables — possible missed table
                print(
                    f"[TABLE VALIDATOR] ⚠ Possible missing table between "
                    f"pages {all_pages[i]} and {all_pages[i+1]} "
                    f"(gap: {gap} pages)"
                )

    # Sort by page
    validated.sort(key=lambda t: t.page)
    return validated


def _repair_merged_table(tbl: DoclingTable) -> Optional[DoclingTable]:
    """
    Attempts to repair a table with merged cells by
    forward-filling empty cells.
    """
    try:
        repaired_rows = []
        prev_row      = [""] * len(tbl.headers)

        for row in tbl.rows:
            new_row = []
            for i, cell in enumerate(row):
                cell_str = str(cell).strip()
                if not cell_str and i < len(prev_row):
                    new_row.append(prev_row[i])    # forward-fill
                else:
                    new_row.append(cell_str)
            repaired_rows.append(new_row)
            prev_row = new_row

        return DoclingTable(
            page       = tbl.page,
            headers    = tbl.headers,
            rows       = repaired_rows,
            confidence = tbl.confidence * 0.85,
            has_merged = False,
        )
    except Exception:
        return None


def _ocr_text_to_markdown(text: str) -> str:
    """
    Converts raw OCR table text to a basic markdown table.
    Best effort — OCR tables are inherently noisy.
    """
    import re
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return text

    # Detect delimiter (tab, pipe, multiple spaces)
    delimiter = "\t" if "\t" in lines[0] else (
        "|" if "|" in lines[0] else
        r"\s{2,}"
    )

    rows = []
    for line in lines:
        if delimiter == r"\s{2,}":
            cells = re.split(delimiter, line)
        else:
            cells = line.split(delimiter)
        cells = [c.strip() for c in cells if c.strip()]
        if cells:
            rows.append(cells)

    if not rows:
        return text

    # Use first row as headers
    max_cols = max(len(r) for r in rows)
    headers  = rows[0] + [""] * (max_cols - len(rows[0]))
    md_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * max_cols) + " |",
    ]
    for row in rows[1:]:
        padded = row + [""] * (max_cols - len(row))
        md_lines.append("| " + " | ".join(padded) + " |")

    return "\n".join(md_lines)
