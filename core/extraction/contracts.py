"""
AION Core Extraction — Contracts & Data Structures
===================================================
Defines the canonical data models, metrics, enums, and adapter boundaries
for document extraction, evidence chunking, and quality analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Tuple


class ExtractionLevel(IntEnum):
    L1_NATIVE     = 1    # Direct PDF text extraction
    L2_STRUCTURAL = 2    # Layout-aware (Docling)
    L3_OCR        = 3    # OCR for scanned/image pages
    L4_RECOVERY   = 4    # Targeted page-level recovery


class ContentType(str, Enum):
    TEXT         = "TEXT"
    EQUATION     = "EQUATION"
    TABLE        = "TABLE"
    FIGURE       = "FIGURE"
    MIXED        = "MIXED"
    HEADER       = "HEADER"
    FOOTER       = "FOOTER"
    PDF_INTERNAL = "PDF_INTERNAL"
    BINARY       = "BINARY"
    METADATA     = "METADATA"


class ChunkStatus(str, Enum):
    VALID        = "VALID"         # enters retrieval, full weight
    RECOVERABLE  = "RECOVERABLE"   # valid after healing
    SUSPICIOUS   = "SUSPICIOUS"    # enters retrieval with penalty
    QUARANTINED  = "QUARANTINED"   # routed to healer
    INVALID      = "INVALID"       # permanently excluded
    FIGURE_ONLY  = "FIGURE_ONLY"   # image-only page — valid for VRE


class ExtractionAdapterID(str, Enum):
    PYMUPDF    = "PYMUPDF"
    DOCLING    = "DOCLING"
    OCR        = "OCR"
    PDFPLUMBER = "PDFPLUMBER"
    FALLBACK   = "FALLBACK"


class RejectionReason(str, Enum):
    BINARY_CONTAMINATION  = "BINARY_CONTAMINATION"
    UNICODE_CORRUPTION    = "UNICODE_CORRUPTION"
    LOW_ACADEMIC_SCORE    = "LOW_ACADEMIC_SCORE"
    MISSING_PROVENANCE    = "MISSING_PROVENANCE"
    EQUATION_PARSE_FAIL   = "EQUATION_PARSE_FAIL"
    EMPTY_CONTENT         = "EMPTY_CONTENT"
    INJECTION_DETECTED    = "INJECTION_DETECTED"
    IMAGE_ONLY_PAGE       = "IMAGE_ONLY_PAGE"     # → FIGURE_ONLY, not INVALID
    OCR_CONFIDENCE_LOW    = "OCR_CONFIDENCE_LOW"
    ENCODING_FAILURE      = "ENCODING_FAILURE"
    PDF_METADATA_LEAK     = "PDF_METADATA_LEAK"


@dataclass
class ExtractionMetrics:
    """
    Per-modality confidence vector.
    Provides detailed insight into extraction performance per domain.
    """
    text_confidence        : float = 1.0   # native text extraction quality
    unicode_confidence     : float = 1.0   # unicode integrity
    equation_confidence    : float = 1.0   # LaTeX/math extraction quality
    table_confidence       : float = 1.0   # table cell extraction quality
    figure_confidence      : float = 1.0   # figure detection quality
    structure_confidence   : float = 1.0   # reading order, layout
    layout_confidence      : float = 1.0   # alias for structure_confidence
    provenance_confidence  : float = 1.0   # source mapping
    unicode_integrity      : float = 1.0   # 0.0 = corrupted, 1.0 = clean
    binary_contamination   : float = 0.0   # 0.0 = clean, 1.0 = fully binary
    academic_content_score : float = 1.0   # subject-relevance estimate
    ocr_confidence         : Optional[float] = None  # None if OCR not used

    adapter_used           : ExtractionAdapterID = ExtractionAdapterID.PYMUPDF
    extraction_level       : ExtractionLevel = ExtractionLevel.L1_NATIVE
    pages_processed        : int = 0
    pages_native           : int = 0     # extracted without OCR
    pages_ocr              : int = 0     # required OCR
    pages_failed           : int = 0

    def overall_quality(self) -> float:
        """Weighted composite quality score."""
        weights = {
            "text"    : 0.25,
            "unicode" : 0.20,
            "binary"  : 0.20,   # inverted: 1.0 - contamination
            "academic": 0.20,
            "equation": 0.15,
        }
        return (
            self.text_confidence              * weights["text"]    +
            self.unicode_integrity            * weights["unicode"] +
            (1.0 - self.binary_contamination) * weights["binary"]  +
            self.academic_content_score       * weights["academic"] +
            self.equation_confidence          * weights["equation"]
        )

    def failure_summary(self) -> Dict[str, float]:
        """Returns dictionary of metrics violating quality thresholds."""
        issues = {}
        if self.binary_contamination > 0.01:
            issues["binary_contamination"] = self.binary_contamination
        if self.unicode_integrity < 0.95:
            issues["unicode_integrity"] = self.unicode_integrity
        if self.text_confidence < 0.70:
            issues["text_confidence"] = self.text_confidence
        if self.academic_content_score < 0.60:
            issues["academic_content_score"] = self.academic_content_score
        return issues


@dataclass
class TextBlock:
    text          : str
    bbox          : Optional[Tuple[float, float, float, float]] = None
    reading_order : int = 0
    adapter_id    : ExtractionAdapterID = ExtractionAdapterID.PYMUPDF
    page          : int = 1
    confidence    : float = 1.0


@dataclass
class EquationArtifact:
    eq_id        : str
    latex        : str
    page         : int = 1
    bbox         : Optional[Tuple[float, float, float, float]] = None
    confidence   : float = 1.0
    adapter_id   : ExtractionAdapterID = ExtractionAdapterID.PYMUPDF


@dataclass
class TableArtifact:
    table_id     : str
    markdown     : str
    page         : int = 1
    rows         : int = 0
    cols         : int = 0
    bbox         : Optional[Tuple[float, float, float, float]] = None
    confidence   : float = 1.0
    adapter_id   : ExtractionAdapterID = ExtractionAdapterID.PYMUPDF


@dataclass
class FigureArtifact:
    figure_id    : str
    image_bytes  : bytes = field(default=b"", repr=False)
    image_path   : Optional[str] = None
    page         : int = 1
    figure_type  : str = "GENERIC"
    caption      : Optional[str] = None
    bbox         : Optional[Tuple[float, float, float, float]] = None
    adapter_id   : ExtractionAdapterID = ExtractionAdapterID.PYMUPDF


@dataclass
class PageResult:
    page_no           : int
    classification    : str = "NATIVE_TEXT"  # "NATIVE_TEXT" | "IMAGE_ONLY" | "MIXED" | "EMPTY"
    native_char_count : int = 0
    image_count       : int = 0
    ocr_used          : bool = False


@dataclass
class ExtractionResult:
    """
    Adapter boundary output. All extraction adapters return this.
    """
    success          : bool
    adapter_id       : ExtractionAdapterID
    extraction_level : ExtractionLevel
    metrics          : ExtractionMetrics

    # Extracted content
    text_blocks      : List[TextBlock] = field(default_factory=list)
    equations        : List[EquationArtifact] = field(default_factory=list)
    tables           : List[TableArtifact] = field(default_factory=list)
    figures          : List[FigureArtifact] = field(default_factory=list)
    pages            : List[PageResult] = field(default_factory=list)

    # Failure information
    error_type       : Optional[str] = None
    error_message    : Optional[str] = None
    error_traceback  : Optional[str] = None
    recoverable      : bool = True

    def merge_with(self, other: ExtractionResult) -> ExtractionResult:
        """Merge two extraction results from different adapters."""
        return ExtractionResult(
            success          = self.success or other.success,
            adapter_id       = ExtractionAdapterID.FALLBACK,
            extraction_level = max(self.extraction_level, other.extraction_level),
            metrics          = self._merge_metrics(other.metrics),
            text_blocks      = self._deduplicate_blocks(self.text_blocks + other.text_blocks),
            equations        = self._deduplicate_eqs(self.equations + other.equations),
            tables           = self._deduplicate_tables(self.tables + other.tables),
            figures          = self._deduplicate_figs(self.figures + other.figures),
            pages            = self.pages if self.pages else other.pages,
        )

    def _merge_metrics(self, other: ExtractionMetrics) -> ExtractionMetrics:
        return ExtractionMetrics(
            text_confidence        = max(self.metrics.text_confidence, other.metrics.text_confidence),
            layout_confidence      = max(self.metrics.layout_confidence, other.metrics.layout_confidence),
            equation_confidence    = max(self.metrics.equation_confidence, other.metrics.equation_confidence),
            table_confidence       = max(self.metrics.table_confidence, other.metrics.table_confidence),
            figure_confidence      = max(self.metrics.figure_confidence, other.metrics.figure_confidence),
            unicode_integrity      = min(self.metrics.unicode_integrity, other.metrics.unicode_integrity),
            binary_contamination   = min(self.metrics.binary_contamination, other.metrics.binary_contamination),
            academic_content_score = max(self.metrics.academic_content_score, other.metrics.academic_content_score),
            adapter_used           = ExtractionAdapterID.FALLBACK,
            extraction_level       = max(self.metrics.extraction_level, other.metrics.extraction_level),
            pages_processed        = max(self.metrics.pages_processed, other.metrics.pages_processed),
            pages_native           = self.metrics.pages_native + other.metrics.pages_native,
            pages_ocr              = self.metrics.pages_ocr + other.metrics.pages_ocr,
            pages_failed           = min(self.metrics.pages_failed, other.metrics.pages_failed),
        )

    def _deduplicate_blocks(self, blocks: List[TextBlock]) -> List[TextBlock]:
        seen = set()
        unique = []
        for b in blocks:
            key = (b.page, b.text.strip())
            if key not in seen and b.text.strip():
                seen.add(key)
                unique.append(b)
        return unique

    def _deduplicate_eqs(self, eqs: List[EquationArtifact]) -> List[EquationArtifact]:
        seen = set()
        unique = []
        for e in eqs:
            key = (e.page, e.latex.strip())
            if key not in seen and e.latex.strip():
                seen.add(key)
                unique.append(e)
        return unique

    def _deduplicate_tables(self, tbls: List[TableArtifact]) -> List[TableArtifact]:
        seen = set()
        unique = []
        for t in tbls:
            key = (t.page, t.markdown.strip())
            if key not in seen and t.markdown.strip():
                seen.add(key)
                unique.append(t)
        return unique

    def _deduplicate_figs(self, figs: List[FigureArtifact]) -> List[FigureArtifact]:
        seen = set()
        unique = []
        for f in figs:
            key = (f.page, f.figure_id)
            if key not in seen:
                seen.add(key)
                unique.append(f)
        return unique


@dataclass
class EvidenceChunk:
    """
    Canonical unit of evidence passed to retrieval and vector store.
    Must have complete provenance to be VALID.
    """
    chunk_id         : str
    document_id      : str
    source_path      : str
    adapter_id       : ExtractionAdapterID
    page_start       : int
    page_end         : int

    content_type     : ContentType
    text             : str

    bbox             : Optional[Tuple[float, float, float, float]] = None
    module_id        : Optional[str] = None

    equation_ids     : List[str] = field(default_factory=list)
    table_ids        : List[str] = field(default_factory=list)
    figure_ids       : List[str] = field(default_factory=list)
    block_ids        : List[str] = field(default_factory=list)

    extraction_confidence : float = 1.0
    unicode_integrity     : float = 1.0
    binary_contamination  : float = 0.0
    academic_score        : float = 1.0

    embedding        : Optional[List[float]] = None
    embedding_model  : Optional[str] = None

    status           : ChunkStatus = ChunkStatus.VALID
    rejection_reasons: List[RejectionReason] = field(default_factory=list)
    retrieval_penalty: float = 0.0

    def is_retrieval_eligible(self) -> bool:
        return self.status in {
            ChunkStatus.VALID,
            ChunkStatus.RECOVERABLE,
            ChunkStatus.SUSPICIOUS,
            ChunkStatus.FIGURE_ONLY,
        }

    def has_math(self) -> bool:
        return len(self.equation_ids) > 0

    def provenance_complete(self) -> bool:
        return (
            self.document_id is not None and
            bool(self.source_path) and
            self.page_start >= 0 and
            self.adapter_id is not None
        )
