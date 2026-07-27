"""
AION Module: Document Cleaner
Maturity:    v0.1 — RULE-BASED NOISE STRIPPER
Upgrades to: Neural Layout & Header/Footer Classification Model
Contract:    document: Document -> CleanedDocument (see schemas.py)
"""

import re
from collections import Counter
from .schemas import Document, CleanedDocument


def clean(document: Document) -> CleanedDocument:
    original_lines = document.raw_text.splitlines()
    original_count = len(original_lines)

    # ── Pass 1: Detect repeated header/footer lines ──
    # Lines that appear on many pages are running headers/footers
    line_counts = Counter(
        ln.strip().lower()
        for ln in original_lines
        if len(ln.strip()) >= 4 and not ln.strip().isdigit()
    )
    # A line repeated more than 8 times is almost certainly a header/footer
    repeated_noise = {
        line for line, count in line_counts.items()
        if count >= 8
    }

    # ── Pass 2: Line-level filtering ──
    kept = []
    removed = 0

    for line in original_lines:
        s = line.strip()

        # Blank line (preserve as paragraph boundary)
        if not s:
            kept.append("")
            continue

        # Single char noise
        if len(s) < 2:
            removed += 1
            continue

        # Standalone page number
        if s.isdigit():
            removed += 1
            continue

        # Divider lines
        if s.startswith("---") or s.startswith("===") or s.startswith("___"):
            removed += 1
            continue

        # URLs
        if re.match(r"^\s*https?://\S+\s*$", s):
            removed += 1
            continue

        # Standalone filename references
        if re.match(r"^\s*[\w_\-]+\.(py|html|js|css|json|yaml|txt|md|cfg)\s*$", s):
            removed += 1
            continue

        # ISBN / copyright lines
        if re.search(r"^\s*(isbn|copyright\s*©?|all\s+rights\s+reserved|published\s+by)\b", s, re.I):
            removed += 1
            continue

        # Repeated header/footer
        if s.lower() in repeated_noise:
            removed += 1
            continue

        kept.append(s)

    # ── Pass 3: Rebuild with paragraph structure preserved ──
    # Group consecutive non-empty lines into paragraphs
    # separated by blank lines (which become \n\n)
    paragraphs = []
    current_para = []

    for line in kept:
        if line:
            current_para.append(line)
        else:
            if current_para:
                paragraphs.append(" ".join(current_para))
                current_para = []

    if current_para:
        paragraphs.append(" ".join(current_para))

    # Join paragraphs with double newline so learner.py
    # can split on \n\n correctly
    clean_text = "\n\n".join(p for p in paragraphs if p.strip())

    # Also fix hyphenation across line breaks
    clean_text = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", clean_text)

    return CleanedDocument(
        doc_id=document.doc_id,
        clean_text=clean_text,
        removed_line_count=removed,
        original_line_count=original_count,
    )
