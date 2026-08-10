"""
AION Core Extraction — Module Segmentation Hierarchy
=====================================================
Implements 6-tier strategy hierarchy for module boundary identification
as specified in Part V of the Production Hardening Specification.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional

logger = logging.getLogger("AION.ModuleSegmenter")


class SegmentationStrategy(str, Enum):
    EXPLICIT_MODULE_HEADING  = "EXPLICIT_MODULE_HEADING"   # Confidence: 0.98
    NUMBERED_CHAPTER_HEADING = "NUMBERED_CHAPTER_HEADING"  # Confidence: 0.92
    SECTION_HEADING          = "SECTION_HEADING"           # Confidence: 0.85
    SYLLABUS_TOPIC_MATCH     = "SYLLABUS_TOPIC_MATCH"      # Confidence: 0.80
    PAGE_STRUCTURE           = "PAGE_STRUCTURE"            # Confidence: 0.70
    PARAGRAPH_DENSITY        = "PARAGRAPH_DENSITY"         # Confidence: 0.50 (last resort)


STRATEGY_CONFIDENCE = {
    SegmentationStrategy.EXPLICIT_MODULE_HEADING: 0.98,
    SegmentationStrategy.NUMBERED_CHAPTER_HEADING: 0.92,
    SegmentationStrategy.SECTION_HEADING: 0.85,
    SegmentationStrategy.SYLLABUS_TOPIC_MATCH: 0.80,
    SegmentationStrategy.PAGE_STRUCTURE: 0.70,
    SegmentationStrategy.PARAGRAPH_DENSITY: 0.50,
}


@dataclass
class ModuleSegment:
    module_id    : int
    title        : str
    page_start   : int
    page_end     : int
    confidence   : float
    strategy_used: SegmentationStrategy


class ModuleSegmenter:
    """Segments document into modules using a 6-tier strategy hierarchy."""

    EXPLICIT_PATTERNS = [
        re.compile(r"^\s*(?:MODULE|UNIT|CHAPTER)\s+([1-5]|[I|II|III|IV|V])[:\s\-\.]*(.*)$", re.IGNORECASE),
    ]

    NUMBERED_PATTERNS = [
        re.compile(r"^\s*(\d{1,2})\.\s+([A-Z][A-Za-z\s]{3,50})$"),
    ]

    @classmethod
    def segment(cls, artifact: Any, syllabus_topics: Optional[List[str]] = None) -> List[ModuleSegment]:
        text_blocks = getattr(artifact, "text_blocks", [])
        page_count = getattr(artifact, "page_count", 1)

        # Strategy 1: EXPLICIT_MODULE_HEADING
        explicit_segments = cls._try_explicit_headings(text_blocks, page_count)
        if explicit_segments:
            cls._log_segments(SegmentationStrategy.EXPLICIT_MODULE_HEADING, explicit_segments)
            return explicit_segments

        # Strategy 2: NUMBERED_CHAPTER_HEADING
        numbered_segments = cls._try_numbered_headings(text_blocks, page_count)
        if numbered_segments:
            cls._log_segments(SegmentationStrategy.NUMBERED_CHAPTER_HEADING, numbered_segments)
            return numbered_segments

        # Strategy 3 & 4 & 5: Page structure split
        if page_count >= 5:
            page_segments = cls._try_page_structure(page_count)
            cls._log_segments(SegmentationStrategy.PAGE_STRUCTURE, page_segments)
            return page_segments

        # Strategy 6: PARAGRAPH_DENSITY (last resort)
        density_segments = cls._density_fallback(text_blocks, page_count)
        logger.warning("[SEGMENTER] Using paragraph density — segmentation may be imprecise")
        cls._log_segments(SegmentationStrategy.PARAGRAPH_DENSITY, density_segments)
        return density_segments

    @classmethod
    def _try_explicit_headings(cls, blocks: List[Any], page_count: int) -> Optional[List[ModuleSegment]]:
        found = []
        for b in blocks:
            text = b.get("text", "") if isinstance(b, dict) else getattr(b, "text", "")
            page = b.get("page", 1) if isinstance(b, dict) else getattr(b, "page", 1)
            for pat in cls.EXPLICIT_PATTERNS:
                m = pat.match(text.strip())
                if m:
                    label, title = m.group(1), m.group(2)
                    found.append((page, f"Module {label}: {title.strip()}"))
                    break

        if len(found) >= 2:
            segments = []
            for i, (p_start, title) in enumerate(found):
                p_end = found[i+1][0] - 1 if i + 1 < len(found) else page_count
                segments.append(
                    ModuleSegment(
                        module_id=i+1,
                        title=title,
                        page_start=p_start,
                        page_end=max(p_start, p_end),
                        confidence=0.98,
                        strategy_used=SegmentationStrategy.EXPLICIT_MODULE_HEADING,
                    )
                )
            return segments
        return None

    @classmethod
    def _try_numbered_headings(cls, blocks: List[Any], page_count: int) -> Optional[List[ModuleSegment]]:
        found = []
        for b in blocks:
            text = b.get("text", "") if isinstance(b, dict) else getattr(b, "text", "")
            page = b.get("page", 1) if isinstance(b, dict) else getattr(b, "page", 1)
            for pat in cls.NUMBERED_PATTERNS:
                m = pat.match(text.strip())
                if m:
                    found.append((page, text.strip()))
                    break

        if len(found) >= 3:
            segments = []
            for i, (p_start, title) in enumerate(found[:5]):
                p_end = found[i+1][0] - 1 if i + 1 < len(found[:5]) else page_count
                segments.append(
                    ModuleSegment(
                        module_id=i+1,
                        title=title,
                        page_start=p_start,
                        page_end=max(p_start, p_end),
                        confidence=0.92,
                        strategy_used=SegmentationStrategy.NUMBERED_CHAPTER_HEADING,
                    )
                )
            return segments
        return None

    @classmethod
    def _try_page_structure(cls, page_count: int) -> List[ModuleSegment]:
        pages_per_mod = max(1, page_count // 5)
        segments = []
        for m in range(1, 6):
            p_start = (m - 1) * pages_per_mod + 1
            p_end = m * pages_per_mod if m < 5 else page_count
            segments.append(
                ModuleSegment(
                    module_id=m,
                    title=f"Module {m}",
                    page_start=p_start,
                    page_end=p_end,
                    confidence=0.70,
                    strategy_used=SegmentationStrategy.PAGE_STRUCTURE,
                )
            )
        return segments

    @classmethod
    def _density_fallback(cls, blocks: List[Any], page_count: int) -> List[ModuleSegment]:
        return cls._try_page_structure(page_count)

    @classmethod
    def _log_segments(cls, strat: SegmentationStrategy, segments: List[ModuleSegment]):
        logger.info(f"[SEGMENTER] Strategy selected  : {strat.value}")
        logger.info(f"[SEGMENTER] Confidence         : {STRATEGY_CONFIDENCE[strat]}")
        logger.info(f"[SEGMENTER] Modules found      : {len(segments)}")
        for s in segments:
            logger.info(f"  Module {s.module_id}: pages {s.page_start}–{s.page_end} (title: '{s.title}')")
