"""
AION v2 Extraction Contracts & Intermediate Data Models
======================================================
Defines bounded per-page extraction results and raw element buffers.
Guarantees strict per-page accounting without monolithic flat-text blobs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from aion.core.dom.document_dom import PageState


class PageExtractionStatus(str, Enum):
    EXTRACTED = "EXTRACTED"
    PARTIAL = "PARTIAL"
    OCR_REQUIRED = "OCR_REQUIRED"
    FAILED = "FAILED"
    EMPTY_EXPECTED = "EMPTY_EXPECTED"

    def to_page_state(self) -> PageState:
        return PageState(self.value)


@dataclass
class RawTextBlock:
    text: str
    bbox: Optional[Tuple[float, float, float, float]] = None
    font_name: str = ""
    font_size: float = 0.0
    is_bold: bool = False
    reading_order: int = 0


@dataclass
class RawHeading:
    title: str
    level: int = 2                         # 1 = Chapter/Module, 2 = Section, 3 = Subsection
    numbering: Optional[str] = None        # e.g. "3.4", "Module 2"
    bbox: Optional[Tuple[float, float, float, float]] = None


@dataclass
class RawEquation:
    latex: str
    raw_text: str = ""
    bbox: Optional[Tuple[float, float, float, float]] = None
    is_inline: bool = False
    label: Optional[str] = None            # e.g. "(3.1)"
    verification_status: str = "CANDIDATE" # "VERIFIED" | "CANDIDATE" | "FAILED"
    variables: List[str] = field(default_factory=list)



@dataclass
class RawFigure:
    image_bytes: bytes = field(default=b"", repr=False)
    bbox: Optional[Tuple[float, float, float, float]] = None
    format_hint: str = "png"
    caption_hint: Optional[str] = None
    width: int = 0
    height: int = 0


@dataclass
class RawTable:
    rows: List[List[str]] = field(default_factory=list)
    headers: List[str] = field(default_factory=list)
    markdown: str = ""
    bbox: Optional[Tuple[float, float, float, float]] = None
    caption_hint: Optional[str] = None


@dataclass
class PageExtractionResult:
    """
    Complete extraction output for a single source document page.
    Every page produces one instance, guaranteeing total page accounting.
    """
    page_number: int
    status: PageExtractionStatus = PageExtractionStatus.EXTRACTED
    text_blocks: List[RawTextBlock] = field(default_factory=list)
    headings: List[RawHeading] = field(default_factory=list)
    equations: List[RawEquation] = field(default_factory=list)
    figures: List[RawFigure] = field(default_factory=list)
    tables: List[RawTable] = field(default_factory=list)
    char_count: int = 0
    ocr_applied: bool = False
    error_message: Optional[str] = None


@dataclass
class ExtractionJob:
    """
    Specifies parameters for an extraction run.
    """
    document_id: str
    source_path: str
    total_pages: int
    module_mappings: Dict[int, Tuple[int, int]] = field(default_factory=dict) # {module_idx: (start_p, end_p)}
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionBatch:
    """
    A bounded window of processed pages (e.g., 20 pages) for streaming execution.
    """
    job_id: str
    batch_index: int
    start_page: int
    end_page: int
    pages: List[PageExtractionResult] = field(default_factory=list)
