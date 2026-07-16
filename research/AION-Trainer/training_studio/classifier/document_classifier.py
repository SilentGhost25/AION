# AION-Trainer/training_studio/classifier/document_classifier.py
"""
Document Classifier — determines what kind of academic document
was uploaded without asking the user.

Classification hierarchy:
    1. Structural signals (section headers, numbering patterns)
    2. Content signals (question density, definition density)
    3. Filename signals (weakest — used as tiebreaker only)

If confidence < 0.90 on the top class, the UI presents a
confirmation dialog rather than silently classifying.
"""

from __future__ import annotations

import re
import math
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("aion.studio.classifier")

CONFIDENCE_THRESHOLD = 0.90


class DocumentType:
    TEXTBOOK = "textbook"
    NOTES = "notes"
    QUESTION_BANK = "question_bank"
    PREVIOUS_PAPER = "previous_paper"
    ANSWER_KEY = "answer_key"
    SYLLABUS = "syllabus"
    IMAGES = "images"
    UNKNOWN = "unknown"


@dataclass
class ClassificationResult:
    document_type: str
    confidence: float
    all_scores: Dict[str, float] = field(default_factory=dict)
    signals_found: List[str] = field(default_factory=list)
    needs_confirmation: bool = False
    suggested_alternatives: List[str] = field(default_factory=list)

    def is_certain(self) -> bool:
        return self.confidence >= CONFIDENCE_THRESHOLD and not self.needs_confirmation


# ── Signal patterns ──────────────────────────────────────────────────────────

# Structural patterns for each document type
STRUCTURAL_SIGNALS: Dict[str, List[re.Pattern]] = {
    DocumentType.TEXTBOOK: [
        re.compile(r"^\s*(chapter|unit)\s+\d+", re.IGNORECASE | re.MULTILINE),
        re.compile(r"\b(learning\s+objectives?|summary|exercises?|review\s+questions?)\b",
                   re.IGNORECASE),
        re.compile(r"\b(definition|theorem|corollary|lemma|proof)\b", re.IGNORECASE),
        re.compile(r"^\s*\d+\.\d+(\.\d+)?\s+\w", re.MULTILINE),   # 2.3.1 Section
        re.compile(r"\bfigure\s+\d+\.\d+\b", re.IGNORECASE),
    ],
    DocumentType.NOTES: [
        re.compile(r"^\s*(module|topic|unit)\s*[-:]?\s*\d+", re.IGNORECASE | re.MULTILINE),
        re.compile(r"\b(refer|see\s+also|notes?|lecture\s+notes?)\b", re.IGNORECASE),
        re.compile(r"^\s*[-•*]\s+\w", re.MULTILINE),               # Bullet lists (common in notes)
        re.compile(r"\b(imp\.?|important|remember|note[:\s])\b", re.IGNORECASE),
    ],
    DocumentType.QUESTION_BANK: [
        re.compile(r"^\s*(q\.?\s*\d+|question\s+\d+)[\s.:]", re.IGNORECASE | re.MULTILINE),
        re.compile(r"\b(\d+\s*marks?)\b", re.IGNORECASE),
        re.compile(r"\b(unit\s*[-:]\s*\d+|module\s*[-:]\s*\d+)\s*\n",
                   re.IGNORECASE | re.MULTILINE),
        re.compile(r"\b(answer\s+any|choose\s+one|attempt)\b", re.IGNORECASE),
        re.compile(r"(a\)|b\)|c\)|d\))", re.IGNORECASE),           # MCQ options
    ],
    DocumentType.PREVIOUS_PAPER: [
        re.compile(r"\b(time\s*:\s*\d+\s*hours?|max\.?\s*marks?\s*:\s*\d+)\b", re.IGNORECASE),
        re.compile(r"\b(visvesvaraya|vtu|university\s+examination)\b", re.IGNORECASE),
        re.compile(r"\b(semester|sem\.?)\s*[-:]?\s*\d+", re.IGNORECASE),
        re.compile(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\.?\s*20\d\d",
                   re.IGNORECASE),
        re.compile(r"\b(reg\.?\s*no\.?|usn)\b", re.IGNORECASE),
    ],
    DocumentType.ANSWER_KEY: [
        re.compile(r"\b(answer\s+key|solution|model\s+answer)\b", re.IGNORECASE),
        re.compile(r"\b(ans\.?|answer\s*:)", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^\s*(q\.?\s*\d+|question\s+\d+)\s*[:)]\s*\n.*\n.*answer",
                   re.IGNORECASE | re.MULTILINE | re.DOTALL),
    ],
    DocumentType.SYLLABUS: [
        re.compile(r"\b(course\s+(objectives?|outcomes?)|co\s*\d+)\b", re.IGNORECASE),
        re.compile(r"\b(syllabus|curriculum|scheme\s+of\s+teaching)\b", re.IGNORECASE),
        re.compile(r"\b(credit\s+hours?|contact\s+hours?|l\s*:\s*t\s*:\s*p)\b", re.IGNORECASE),
        re.compile(r"module\s*\d+\s*[\n:]\s*.+hours?", re.IGNORECASE),
    ],
}

FILENAME_HINTS: Dict[str, List[str]] = {
    DocumentType.TEXTBOOK: ["textbook", "book", "reference", "text"],
    DocumentType.NOTES: ["note", "notes", "lecture", "module"],
    DocumentType.QUESTION_BANK: ["qb", "question_bank", "qbank", "bank", "questions"],
    DocumentType.PREVIOUS_PAPER: ["pyq", "previous", "paper", "exam", "question_paper"],
    DocumentType.ANSWER_KEY: ["answer", "key", "solution", "solved"],
    DocumentType.SYLLABUS: ["syllabus", "scheme", "curriculum"],
}


class DocumentClassifier:
    """
    Classifies a document by type using structural, content, and
    filename signals. Returns probability scores for all types.
    """

    def classify_text(self, text: str, filename: str = "") -> ClassificationResult:
        scores = self._compute_scores(text, filename)
        sorted_types = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_type, top_score = sorted_types[0]
        second_type, second_score = sorted_types[1] if len(sorted_types) > 1 else (DocumentType.UNKNOWN, 0.0)

        needs_confirmation = (
            top_score < CONFIDENCE_THRESHOLD
            or (top_score - second_score) < 0.15
        )

        signals_found = self._collect_signals(text, top_type)

        result = ClassificationResult(
            document_type=top_type,
            confidence=round(top_score, 4),
            all_scores={t: round(s, 4) for t, s in sorted_types},
            signals_found=signals_found,
            needs_confirmation=needs_confirmation,
            suggested_alternatives=[second_type] if needs_confirmation else [],
        )

        logger.info(
            f"[Classifier] {filename or 'document'}: "
            f"{top_type} ({top_score:.0%}) "
            f"{'⚠ needs confirmation' if needs_confirmation else '✓'}"
        )
        return result

    def classify_file(self, file_path: str) -> ClassificationResult:
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix in (".png", ".jpg", ".jpeg", ".tiff", ".gif", ".bmp"):
            return ClassificationResult(
                document_type=DocumentType.IMAGES,
                confidence=1.0,
                signals_found=["Image file extension"],
            )

        text = self._extract_text(file_path)
        return self.classify_text(text, filename=path.name)

    def _compute_scores(self, text: str, filename: str) -> Dict[str, float]:
        scores: Dict[str, float] = {}

        for doc_type, patterns in STRUCTURAL_SIGNALS.items():
            matched = 0
            for pattern in patterns:
                if pattern.search(text[:10000]):  # check first 10k chars for speed
                    matched += 1
            scores[doc_type] = matched / len(patterns)

        # Filename signals (weak — add 10% bonus if matched)
        fname_lower = filename.lower()
        for doc_type, hints in FILENAME_HINTS.items():
            if any(h in fname_lower for h in hints):
                scores[doc_type] = min(1.0, scores.get(doc_type, 0.0) + 0.10)

        # Normalise using softmax so scores sum to 1
        return self._softmax(scores)

    def _collect_signals(self, text: str, doc_type: str) -> List[str]:
        signals = []
        for pattern in STRUCTURAL_SIGNALS.get(doc_type, []):
            match = pattern.search(text[:10000])
            if match:
                signals.append(f"Pattern: '{match.group(0)[:50].strip()}'")
        return signals[:5]

    @staticmethod
    def _softmax(scores: Dict[str, float]) -> Dict[str, float]:
        if not scores:
            return {}
        vals = list(scores.values())
        # Shift for numerical stability
        max_val = max(vals)
        exps = {k: math.exp(v - max_val) for k, v in scores.items()}
        total = sum(exps.values())
        return {k: v / total for k, v in exps.items()} if total > 0 else scores

    def _extract_text(self, file_path: str) -> str:
        suffix = Path(file_path).suffix.lower()
        try:
            if suffix == ".pdf":
                import fitz
                doc = fitz.open(file_path)
                # Read first 20 pages for classification — no need for full doc
                pages = min(20, len(doc))
                text = "\n".join(doc[i].get_text("text") for i in range(pages))
                doc.close()
                return text
            elif suffix == ".docx":
                import docx as python_docx
                doc = python_docx.Document(file_path)
                return "\n".join(p.text for p in doc.paragraphs[:200])
        except Exception as e:
            logger.warning(f"[Classifier] Text extraction failed for {file_path}: {e}")
        return ""
