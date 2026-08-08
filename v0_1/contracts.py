"""
AION Pipeline Contracts
========================
Every stage consumes a typed contract and produces a typed contract.
No raw strings. No dicts. No silent type mismatches.

If a stage receives the wrong type, it raises ContractViolation immediately.
This eliminates the Document.get() class of bugs permanently.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional


# ── Pipeline health score ─────────────────────────────────────────────────────

class PipelineHealth:
    """
    Tracks pipeline health score. Every fallback reduces the score.
    Paper is rejected if score drops below MIN_EXPORT_SCORE.
    """
    MIN_EXPORT_SCORE = 40

    def __init__(self, score: int = 100):
        self._score    = score
        self._events:  list[dict] = []

    def deduct(self, amount: int, reason: str):
        self._score = max(0, self._score - amount)
        self._events.append({
            "score_after": self._score,
            "deduction":   amount,
            "reason":      reason,
            "time":        datetime.now().isoformat(),
        })
        print(f"[HEALTH] Score: {self._score} (-{amount}: {reason})")

    @property
    def score(self) -> int:
        return self._score

    @property
    def exportable(self) -> bool:
        return self._score >= self.MIN_EXPORT_SCORE

    def events(self) -> list[dict]:
        return list(self._events)


# ── Contract violation ────────────────────────────────────────────────────────

class ContractViolation(Exception):
    """Raised when a stage receives the wrong contract type."""
    pass


def require_contract(obj: Any, expected_type: type, stage: str):
    """Assert that obj is the expected contract type."""
    if not isinstance(obj, expected_type):
        raise ContractViolation(
            f"[{stage}] Expected {expected_type.__name__}, "
            f"got {type(obj).__name__}. "
            f"Check that the previous stage returned the correct contract."
        )


# ── Stage enums ───────────────────────────────────────────────────────────────

class ExamType(str, Enum):
    IA  = "IA"
    SEE = "SEE"

class Difficulty(str, Enum):
    EASY   = "Easy"
    MEDIUM = "Medium"
    HARD   = "Hard"
    MIXED  = "Mixed"

class ValidationVerdict(str, Enum):
    PASS   = "PASS"
    REPAIR = "REPAIR"
    FAIL   = "FAIL"


# ── Contracts (ordered by pipeline stage) ────────────────────────────────────

@dataclass
class RawFile:
    """Input to the pipeline. The only place a file path is used."""
    path:       str
    doc_id:     str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    subject:    str = ""
    department: str = ""
    exam_type:  ExamType = ExamType.IA
    difficulty: Difficulty = Difficulty.MIXED
    health:     PipelineHealth = field(default_factory=PipelineHealth)

    def __post_init__(self):
        if not Path(self.path).exists():
            raise ContractViolation(f"File not found: {self.path}")


@dataclass
class ExtractionResult:
    """Output of Stage 1 (Extractor). Never used after Stage 2."""
    doc_id:        str
    raw_text:      str
    word_count:    int
    confidence:    float         # 0.0 → 1.0
    pipeline_used: str           # text_direct / pymupdf / docx / ocr
    pages:         int = 0
    has_math:      bool = False
    has_images:    bool = False
    has_tables:    bool = False
    health:        PipelineHealth = field(default_factory=PipelineHealth)

    def __post_init__(self):
        if not self.raw_text or len(self.raw_text.strip()) < 50:
            raise ContractViolation(
                f"ExtractionResult for {self.doc_id} has no usable text. "
                f"Extraction failed."
            )


@dataclass
class CleanedContent:
    """Output of Stage 2 (Cleaner). Raw text with artifacts removed."""
    doc_id:            str
    clean_text:        str
    original_words:    int
    clean_words:       int
    artifacts_removed: int
    health:            PipelineHealth = field(default_factory=PipelineHealth)

    @property
    def retention_rate(self) -> float:
        return self.clean_words / max(1, self.original_words)

    def __post_init__(self):
        if self.clean_words < 50:
            raise ContractViolation(
                f"CleanedContent has only {self.clean_words} words after cleaning. "
                f"Source document may be corrupted."
            )


@dataclass
class AcademicChunk:
    """A single validated academic text chunk."""
    chunk_id:       str
    text:           str
    word_count:     int
    module_index:   int
    module_title:   str = ""
    page:           int = 0
    quality_score:  float = 0.0
    academic_score: float = 0.0
    noise_score:    float = 0.0
    topics:         list[str] = field(default_factory=list)
    entities:       list[str] = field(default_factory=list)
    has_formula:    bool = False
    has_example:    bool = False
    has_definition: bool = False
    bloom_hint:     int  = 2


@dataclass
class ChunkedContent:
    """Output of Stage 3 (Chunker + Validator). Ready for retrieval."""
    doc_id:         str
    chunks:         list[AcademicChunk]
    modules:        list[dict]
    threshold_used: float
    health:         PipelineHealth = field(default_factory=PipelineHealth)

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    @property
    def chunks_by_module(self) -> dict[int, list[AcademicChunk]]:
        result: dict[int, list[AcademicChunk]] = {}
        for c in self.chunks:
            result.setdefault(c.module_index, []).append(c)
        return result

    def __post_init__(self):
        if not self.chunks:
            raise ContractViolation(
                f"ChunkedContent for {self.doc_id} has zero chunks. "
                f"Validator rejected all content."
            )


@dataclass
class Evidence:
    """A grounded evidence unit for one question."""
    chunk_ids:      list[str]
    texts:          list[str]
    combined_text:  str
    module_index:   int
    evidence_score: float
    word_count:     int
    query:          str = ""

    def __post_init__(self):
        if len(self.texts) < 1:
            raise ContractViolation("Evidence must contain at least one chunk.")


@dataclass
class RetrievedEvidence:
    """Output of Stage 4 (Retriever + Grounding Gate)."""
    doc_id:             str
    evidence_by_module: dict[int, Evidence]
    health:             PipelineHealth = field(default_factory=PipelineHealth)

    def get(self, module_index: int) -> Optional[Evidence]:
        return self.evidence_by_module.get(module_index)

    def __post_init__(self):
        if not self.evidence_by_module:
            raise ContractViolation(
                f"RetrievedEvidence for {self.doc_id} has no evidence. "
                f"Grounding gate blocked all modules."
            )


@dataclass
class QuestionSpec:
    """
    Specification for ONE question slot.
    LLM receives this and returns ONLY the question text.
    """
    spec_id:      str
    module_index: int
    q_number:     int
    part_letter:  str
    marks:        int
    bloom_level:  int
    bloom_verb:   str
    co:           str
    is_or:        bool
    evidence:     Evidence
    exam_type:    ExamType


@dataclass
class GenerationRequest:
    """Output of Stage 5 (Template Engine). Input to Stage 6 (LLM)."""
    doc_id:      str
    specs:       list[QuestionSpec]
    exam_type:   ExamType
    subject:     str
    total_marks: int
    health:      PipelineHealth = field(default_factory=PipelineHealth)

    def __post_init__(self):
        if not self.specs:
            raise ContractViolation(
                f"GenerationRequest for {self.doc_id} has no question specs."
            )


@dataclass
class GeneratedQuestion:
    """Output of the LLM for one question spec."""
    spec_id:       str
    question_text: str
    spec:          QuestionSpec
    raw_output:    str = ""

    def __post_init__(self):
        if not self.question_text.strip():
            raise ContractViolation(
                f"GeneratedQuestion {self.spec_id} has empty text."
            )


@dataclass
class ValidatedQuestion:
    """Output of Stage 7 (Critic). Either passed or repaired."""
    question:     GeneratedQuestion
    verdict:      ValidationVerdict
    score:        float
    issues:       list[str] = field(default_factory=list)
    was_repaired: bool = False


@dataclass
class PaperDraft:
    """Assembled paper before final QA."""
    doc_id:    str
    questions: list[ValidatedQuestion]
    exam_type: ExamType
    subject:   str
    health:    PipelineHealth = field(default_factory=PipelineHealth)


@dataclass
class FinalPaper:
    """
    The final output contract. The only thing the renderer receives.
    If health.score < MIN_EXPORT_SCORE, renderer must not export.
    """
    doc_id:       str
    modules:      list[dict]
    exam_type:    str
    subject:      str
    total_marks:  int
    qa_score:     int
    health:       PipelineHealth
    generated_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )
    session_log:  list[dict] = field(default_factory=list)

    @property
    def exportable(self) -> bool:
        return self.health.exportable and self.qa_score >= 40
