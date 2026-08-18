"""
Document Preprocessing Pipeline — Header/TOC/Title/Section Cleaning
Per audit: header contamination makes "MODULE 5" become a concept.
Need: Header remover -> TOC remover -> Title cleaner -> Section detector -> Concept extraction
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
from collections import Counter


@dataclass
class CleanedDocument:
    clean_text: str
    sections: List[Dict[str, str]]  # [{"title": ..., "content": ...}]
    removed_headers: List[str]
    removed_toc_lines: List[str]
    confidence: float


class DocumentPreprocessor:
    """Production-grade preprocessing before concept extraction."""

    HEADER_REPEAT_THRESHOLD = 3
    TOC_DOT_PATTERN = re.compile(r".{10,}\.{3,}\s*\d+\s*$")
    TOC_NUMBERED_PATTERN = re.compile(r"^\s*\d+(\.\d+)*\s+[A-Z].*\s+\d+\s*$")
    SECTION_NUMBER_PATTERN = re.compile(r"^\s*(?:MODULE|UNIT|CHAPTER|SECTION)?\s*\d+(\.\d+)*\s*[:.\-–]?\s*", re.I)
    PURE_NUMBERING_PATTERN = re.compile(r"^\s*\d+(\.\d+)*\s*$")

    def clean(self, raw_text: str) -> CleanedDocument:
        lines = raw_text.splitlines()
        original_count = len(lines)

        # Pass 1: Remove page numbers / footers / repeated headers
        lines, removed_headers = self._remove_headers_footers(lines)

        # Pass 2: Remove TOC lines
        lines, removed_toc = self._remove_toc(lines)

        # Pass 3: Remove pure numbering lines and normalize section titles
        lines = self._normalize_section_titles(lines)

        # Rebuild text
        cleaned = "\n".join(lines)
        # Collapse excessive whitespace while preserving sections
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = re.sub(r"[ \t]{3,}", " ", cleaned)
        cleaned = cleaned.strip()

        # Pass 4: Section detection
        sections = self._detect_sections(cleaned)

        # Rebuild clean_text as content-only (exclude section titles to prevent header contamination)
        # Titles are preserved in sections metadata, but not in concept extraction text
        content_only = "\n\n".join(s["content"] for s in sections) if sections else cleaned
        # If content_only is too short, fallback to cleaned
        if len(content_only.split()) < 30:
            content_only = cleaned

        # Confidence: how much was actual content vs boilerplate
        removed_total = len(removed_headers) + len(removed_toc)
        confidence = max(0.5, 1.0 - (removed_total / max(original_count, 1)) * 0.5)
        confidence = min(0.98, confidence)

        return CleanedDocument(
            clean_text=content_only,
            sections=sections,
            removed_headers=removed_headers,
            removed_toc_lines=removed_toc,
            confidence=round(confidence, 2)
        )

    def _remove_headers_footers(self, lines: List[str]) -> Tuple[List[str], List[str]]:
        # Count normalized lines to find repeated headers/footers
        norm_counts = Counter()
        for ln in lines:
            s = ln.strip()
            if 4 <= len(s) <= 120 and not s.isdigit():
                norm_counts[s.lower()] += 1

        repeated = {txt for txt, cnt in norm_counts.items() if cnt >= self.HEADER_REPEAT_THRESHOLD}

        kept = []
        removed = []
        for ln in lines:
            s = ln.strip()
            if not s:
                kept.append("")
                continue
            low = s.lower()
            # Page number alone
            if s.isdigit() or re.match(r"^page\s+\d+(\s+of\s+\d+)?$", low):
                removed.append(s)
                continue
            if re.match(r"^[-=_]{4,}$", s):
                removed.append(s)
                continue
            if low in repeated:
                removed.append(s)
                continue
            if re.match(r"^(isbn|copyright|all rights reserved|published by|doi:)", low):
                removed.append(s)
                continue
            if re.match(r"^\s*https?://\S+\s*$", s):
                removed.append(s)
                continue
            kept.append(ln)
        return kept, removed

    def _remove_toc(self, lines: List[str]) -> Tuple[List[str], List[str]]:
        kept = []
        removed = []
        # Heuristic: if >30% of lines in first 100 look like TOC, drop the block
        # But simpler: drop individual TOC lines
        for ln in lines:
            s = ln.strip()
            if not s:
                kept.append(ln)
                continue
            if self.TOC_DOT_PATTERN.search(s):
                removed.append(s)
                continue
            if self.TOC_NUMBERED_PATTERN.match(s) and s.count(".") >= 2:
                # TOC style "1.1 Introduction ........ 12" already caught, but numbered with trailing page number
                if re.search(r"\s+\d+\s*$", s):
                    removed.append(s)
                    continue
            kept.append(ln)
        return kept, removed

    def _normalize_section_titles(self, lines: List[str]) -> List[str]:
        out = []
        for ln in lines:
            s = ln.strip()
            if not s:
                out.append("")
                continue
            # Pure numbering line like "5.1" or "5.1.2" -> remove
            if self.PURE_NUMBERING_PATTERN.match(s):
                continue
            # If line is mostly section number + title, strip numbering but keep title if title exists
            # e.g., "5.1 On-Board Diagnostics (OBD) Fundamentals" -> "On-Board Diagnostics (OBD) Fundamentals"
            # e.g., "MODULE 5 – AUTOMOTIVE DIAGNOSTICS & ON-BOARD DIAGNOSTICS" -> keep but clean
            # We remove leading MODULE/UNIT/CHAPTER prefix for concept extraction but preserve section structure
            # Only strip if line is short header-like (<80 chars) and contains numbering
            if len(s) < 100 and self.SECTION_NUMBER_PATTERN.match(s):
                # Extract title part after numbering
                title = self.SECTION_NUMBER_PATTERN.sub("", s).strip(" :.-–")
                if len(title) >= 8:
                    out.append(title)
                # If title empty (e.g., line was just "MODULE 5"), skip
                continue
            out.append(ln)
        return out

    def _detect_sections(self, text: str) -> List[Dict[str, str]]:
        # Split by detected headers (lines that look like titles)
        # Title heuristics: short, Title Case or ALL CAPS, not ending with period
        lines = text.splitlines()
        sections = []
        current_title = "Introduction"
        current_content = []

        title_pattern = re.compile(r"^[A-Z][A-Za-z0-9\-/()& ]{8,80}$")

        for ln in lines:
            s = ln.strip()
            if not s:
                current_content.append("")
                continue
            # Is this a section header?
            is_title = False
            if 10 <= len(s) <= 80 and not s.endswith(".") and title_pattern.match(s):
                # Not a sentence (few periods)
                if s.count(".") == 0 and len(s.split()) <= 8:
                    is_title = True
            # Also ALL CAPS headers
            if s.isupper() and 8 <= len(s) <= 80:
                is_title = True

            if is_title and current_content:
                # Flush previous section
                content = "\n".join(current_content).strip()
                if len(content.split()) >= 30:
                    sections.append({"title": current_title, "content": content})
                current_title = s
                current_content = []
            else:
                current_content.append(ln)

        if current_content:
            content = "\n".join(current_content).strip()
            if len(content.split()) >= 20:
                sections.append({"title": current_title, "content": content})

        if not sections:
            # Fallback: single section
            sections = [{"title": "Document", "content": text}]

        return sections
