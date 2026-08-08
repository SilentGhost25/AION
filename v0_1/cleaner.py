"""
AION Module: Document Cleaner & Stage 3 Semantic Cleaner
=========================================================
Removes residual PDF artifacts, OCR noise, and structural debris from chunks.
"""

import re
import unicodedata
from collections import Counter
from typing import Optional, List
from .schemas import Document, CleanedDocument


def clean(document: Document) -> CleanedDocument:
    original_lines = document.raw_text.splitlines()
    original_count = len(original_lines)

    line_counts = Counter(
        ln.strip().lower()
        for ln in original_lines
        if len(ln.strip()) >= 4 and not ln.strip().isdigit()
    )
    repeated_noise = {
        line for line, count in line_counts.items()
        if count >= 8
    }

    kept = []
    removed = 0

    for line in original_lines:
        s = line.strip()

        if not s:
            kept.append("")
            continue

        if len(s) < 2 or s.isdigit() or s.startswith("---") or s.startswith("===") or s.startswith("___"):
            removed += 1
            continue

        if re.match(r"^\s*https?://\S+\s*$", s) or re.match(r"^\s*[\w_\-]+\.(py|html|js|css|json|yaml|txt|md|cfg)\s*$", s):
            removed += 1
            continue

        if re.search(r"^\s*(isbn|copyright\s*©?|all\s+rights\s+reserved|published\s+by)\b", s, re.I) or s.lower() in repeated_noise:
            removed += 1
            continue

        kept.append(s)

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

    clean_text = "\n\n".join(p for p in paragraphs if p.strip())
    clean_text = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", clean_text)

    return CleanedDocument(
        doc_id=document.doc_id,
        clean_text=clean_text,
        removed_line_count=removed,
        original_line_count=original_count,
    )


# ── Stage 3 Extension ─────────────────────────────────────────────────────────

_PDF_ARTIFACTS = [
    re.compile(r'\b\d+\s+\d+\s+obj\b.*?endobj', re.DOTALL | re.IGNORECASE),
    re.compile(r'<<.*?>>', re.DOTALL),
    re.compile(r'/\w+\s*\[.*?\]', re.DOTALL),
    re.compile(r'stream[\s\S]*?endstream', re.IGNORECASE),
    re.compile(r'xref\s+\d+\s+\d+', re.IGNORECASE),
    re.compile(r'trailer\s*<<', re.IGNORECASE),
    re.compile(r'startxref\s+\d+', re.IGNORECASE),
]

_HEADER_FOOTER = [
    re.compile(r'^page\s+\d+\s+of\s+\d+$', re.IGNORECASE | re.MULTILINE),
    re.compile(r'^\d+\s*$', re.MULTILINE),
    re.compile(r'^www\.\S+\.(?:com|org|edu|in)\s*$', re.IGNORECASE | re.MULTILINE),
    re.compile(r'^(?:copyright|©).*$', re.IGNORECASE | re.MULTILINE),
    re.compile(r'^(?:unit|chapter|module)\s+\d+\s*$', re.IGNORECASE | re.MULTILINE),
]

_OCR_NOISE = [
    re.compile(r'[|]{2,}'),
    re.compile(r'_{4,}'),
    re.compile(r'-{4,}'),
    re.compile(r'\.{4,}'),
    re.compile(r'\s{4,}'),
]


class SemanticCleaner:
    def clean(self, text: str) -> str:
        if not text or not text.strip():
            return ""

        for pattern in _PDF_ARTIFACTS:
            text = pattern.sub(' ', text)

        for pattern in _HEADER_FOOTER:
            text = pattern.sub('', text)

        for pattern in _OCR_NOISE:
            text = pattern.sub(' ', text)

        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
        text = unicodedata.normalize('NFC', text)
        text = re.sub(r'^\s*[a-zA-Z]\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def clean_batch(self, chunks: List[str]) -> List[str]:
        return [self.clean(c) for c in chunks if c.strip()]


_cleaner = SemanticCleaner()

def semantic_clean(text: str) -> str:
    return _cleaner.clean(text)

def semantic_clean_batch(chunks: List[str]) -> List[str]:
    return _cleaner.clean_batch(chunks)
