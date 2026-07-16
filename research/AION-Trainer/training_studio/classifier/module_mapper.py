# AION-Trainer/training_studio/classifier/module_mapper.py
"""
Module Mapper — determines which book chapters correspond to which
syllabus modules, without asking the user.

Algorithm:
    1. Extract TOC from document
    2. Load syllabus structure
    3. Match chapter topics to syllabus module topics (Jaccard + sequence)
    4. Assign module per chapter with confidence

If a chapter's content signals belong to a different module than
its title suggests, flag it as an ambiguity.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("aion.studio.module_mapper")

CONFIDENCE_THRESHOLD = 0.85


@dataclass
class TOCEntry:
    title: str
    page: int = 0
    level: int = 1               # 1 = chapter, 2 = section, 3 = subsection


@dataclass
class ChapterModuleMapping:
    chapter_title: str
    chapter_index: int
    assigned_module: int
    confidence: float
    matching_topics: List[str] = field(default_factory=list)
    is_ambiguous: bool = False
    ambiguity_reason: str = ""
    alternative_module: Optional[int] = None


@dataclass
class ModuleMappingResult:
    mappings: List[ChapterModuleMapping] = field(default_factory=list)
    module_to_chapters: Dict[int, List[str]] = field(default_factory=dict)
    overall_confidence: float = 0.0
    ambiguous_chapters: List[ChapterModuleMapping] = field(default_factory=list)

    def chapters_for_module(self, module_num: int) -> List[str]:
        return self.module_to_chapters.get(module_num, [])


class ModuleMapper:
    def __init__(self, syllabus=None):
        self.syllabus = syllabus

    def map(self, toc_entries: List[TOCEntry]) -> ModuleMappingResult:
        if not self.syllabus or not toc_entries:
            return ModuleMappingResult()

        chapters = [e for e in toc_entries if e.level == 1]
        if not chapters:
            chapters = toc_entries[:20]

        mappings = []
        for i, chapter in enumerate(chapters):
            mapping = self._assign_module(chapter, i)
            mappings.append(mapping)

        # Build module → chapters map
        mod_to_chapters: Dict[int, List[str]] = {}
        for m in mappings:
            mod_to_chapters.setdefault(m.assigned_module, []).append(m.chapter_title)

        confidences = [m.confidence for m in mappings]
        overall = sum(confidences) / len(confidences) if confidences else 0.0

        ambiguous = [m for m in mappings if m.is_ambiguous]

        result = ModuleMappingResult(
            mappings=mappings,
            module_to_chapters=mod_to_chapters,
            overall_confidence=round(overall, 4),
            ambiguous_chapters=ambiguous,
        )

        logger.info(
            f"[ModuleMapper] Mapped {len(chapters)} chapters → "
            f"{len(mod_to_chapters)} modules "
            f"(confidence={overall:.2f}, ambiguous={len(ambiguous)})"
        )
        return result

    def extract_toc(self, file_path: str) -> List[TOCEntry]:
        suffix = file_path.lower().split(".")[-1]
        if suffix == "pdf":
            return self._extract_pdf_toc(file_path)
        elif suffix == "docx":
            return self._extract_docx_toc(file_path)
        return []

    def _assign_module(self, chapter: TOCEntry, index: int) -> ChapterModuleMapping:
        best_module = 0
        best_score = 0.0
        best_topics: List[str] = []
        second_module = 0
        second_score = 0.0

        chapter_words = set(self._tokenise(chapter.title))

        for mod in self.syllabus.modules:
            mod_words = set()
            for topic in mod.topics:
                mod_words.update(self._tokenise(topic))

            if not mod_words or not chapter_words:
                continue

            score = len(chapter_words & mod_words) / len(chapter_words | mod_words)
            matching = [t for t in mod.topics if any(w in self._tokenise(t) for w in chapter_words)]

            if score > best_score:
                second_score = best_score
                second_module = best_module
                best_score = score
                best_module = mod.module_number
                best_topics = matching
            elif score > second_score:
                second_score = score
                second_module = mod.module_number

        # If no topic match at all, guess based on sequential position
        if best_score == 0.0 and self.syllabus.modules:
            num_modules = len(self.syllabus.modules)
            num_chapters = max(len(self.syllabus.modules) * 2, 1)
            guessed_module = min(num_modules, (index * num_modules) // num_chapters + 1)
            return ChapterModuleMapping(
                chapter_title=chapter.title,
                chapter_index=index,
                assigned_module=guessed_module,
                confidence=0.40,
                is_ambiguous=True,
                ambiguity_reason="No topic overlap with syllabus — position-based guess",
            )

        is_ambiguous = (
            best_score < CONFIDENCE_THRESHOLD
            or (best_score - second_score) < 0.10
        )

        return ChapterModuleMapping(
            chapter_title=chapter.title,
            chapter_index=index,
            assigned_module=best_module,
            confidence=round(min(0.99, best_score * 1.4), 4),
            matching_topics=best_topics,
            is_ambiguous=is_ambiguous,
            ambiguity_reason=(
                f"Close match with Module {second_module} "
                f"(scores: {best_score:.2f} vs {second_score:.2f})"
                if is_ambiguous else ""
            ),
            alternative_module=second_module if is_ambiguous else None,
        )

    def _extract_pdf_toc(self, file_path: str) -> List[TOCEntry]:
        try:
            import fitz
            doc = fitz.open(file_path)
            toc = doc.get_toc()
            doc.close()

            if toc:
                return [TOCEntry(title=t[1], page=t[2], level=t[0]) for t in toc]

            # Fallback: scan for heading-like lines in first 10 pages
            entries = []
            doc = fitz.open(file_path)
            for page_num in range(min(10, len(doc))):
                text = doc[page_num].get_text("text")
                for line in text.splitlines():
                    line = line.strip()
                    if re.match(r"^(chapter|unit)\s+\d+", line, re.IGNORECASE) and len(line) < 80:
                        entries.append(TOCEntry(title=line, page=page_num + 1, level=1))
                    elif re.match(r"^\d+\.\d+\s+\w", line) and len(line) < 60:
                        entries.append(TOCEntry(title=line, page=page_num + 1, level=2))
            doc.close()
            return entries

        except Exception as e:
            logger.warning(f"[ModuleMapper] PDF TOC extraction failed: {e}")
            return []

    def _extract_docx_toc(self, file_path: str) -> List[TOCEntry]:
        try:
            import docx as python_docx
            doc = python_docx.Document(file_path)
            entries = []
            for para in doc.paragraphs:
                if para.style.name.startswith("Heading 1"):
                    entries.append(TOCEntry(title=para.text.strip(), level=1))
                elif para.style.name.startswith("Heading 2"):
                    entries.append(TOCEntry(title=para.text.strip(), level=2))
            return entries
        except Exception as e:
            logger.warning(f"[ModuleMapper] DOCX TOC extraction failed: {e}")
            return []

    @staticmethod
    def _tokenise(text: str) -> List[str]:
        return [
            w for w in re.sub(r"[^\w\s]", " ", text.lower()).split()
            if len(w) > 2
        ]
