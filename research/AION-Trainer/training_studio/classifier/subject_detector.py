# AION-Trainer/training_studio/classifier/subject_detector.py
"""
Subject Detector — identifies which subject a document belongs to.

Strategy:
    1. Extract subject code directly (e.g., "BAI401" appears in header)
    2. Keyword matching against known subject vocabulary
    3. LLM-based detection (fallback when signals are weak)

If confidence < 0.90, the UI shows a confirmation dialog.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from document_intelligence.document_model import AcademicDocument

logger = logging.getLogger("aion.studio.subject_detector")

CONFIDENCE_THRESHOLD = 0.90

# Subject code patterns — VTU format: 2-4 letters + 3 digits
SUBJECT_CODE_PATTERN = re.compile(r"\b([A-Z]{2,4}\d{3})\b")

# Vocabulary maps: keywords that strongly imply a subject
# Extend this as new subjects are added to the system
SUBJECT_VOCABULARY: Dict[str, List[str]] = {
    "BAI401": [
        "artificial intelligence", "search algorithm", "a* search",
        "heuristic", "knowledge representation", "predicate logic",
        "machine learning", "neural network", "intelligent agents",
        "planning", "natural language processing", "game theory",
    ],
    "BAI402": [
        "machine learning", "supervised learning", "unsupervised learning",
        "decision tree", "random forest", "support vector machine", "svm",
        "gradient descent", "cross validation", "overfitting",
    ],
    "BCS401": [
        "operating system", "process scheduling", "deadlock", "semaphore",
        "virtual memory", "paging", "file system", "banker algorithm",
    ],
    "BCS402": [
        "computer network", "tcp", "udp", "ip address", "routing",
        "osi model", "ethernet", "socket", "http", "dns", "nat",
    ],
    "BCS403": [
        "database", "sql", "normalization", "transaction", "acid",
        "relational algebra", "er diagram", "indexing", "b tree",
    ],
    "BCS301": [
        "data structure", "linked list", "stack", "queue", "binary tree",
        "graph", "sorting", "searching", "hashing", "heap",
    ],
}


@dataclass
class SubjectDetectionResult:
    subject_code: str
    subject_name: str
    confidence: float
    all_scores: Dict[str, float] = field(default_factory=dict)
    signals_found: List[str] = field(default_factory=list)
    needs_confirmation: bool = False
    detected_codes_in_text: List[str] = field(default_factory=list)

    def is_certain(self) -> bool:
        return self.confidence >= CONFIDENCE_THRESHOLD and not self.needs_confirmation


SUBJECT_NAMES: Dict[str, str] = {
    "BAI401": "Artificial Intelligence",
    "BAI402": "Machine Learning",
    "BCS401": "Operating Systems",
    "BCS402": "Computer Networks",
    "BCS403": "Database Management Systems",
    "BCS301": "Data Structures and Algorithms",
}


class SubjectDetector:
    def __init__(self, known_subjects: Dict[str, str] = None, llm_client=None):
        self.vocabulary = dict(SUBJECT_VOCABULARY)
        self.names = dict(SUBJECT_NAMES)
        if known_subjects:
            self.names.update(known_subjects)
        self.llm = llm_client

    def detect_document(self, document: AcademicDocument) -> SubjectDetectionResult:
        text = document.markdown
        text_lower = text[:15000].lower()
        
        # Add TOC text to boost semantic matching
        toc_text = " ".join([entry.get("title", "") for entry in document.toc]).lower()
        text_lower += " " + toc_text

        # 1. Explicit subject code in text (highest confidence)
        code_matches = SUBJECT_CODE_PATTERN.findall(text[:3000])
        valid_codes = [c for c in code_matches if c in self.names]

        if valid_codes:
            code = valid_codes[0]
            return SubjectDetectionResult(
                subject_code=code,
                subject_name=self.names.get(code, code),
                confidence=0.99,
                signals_found=[f"Subject code '{code}' found in document"],
                needs_confirmation=False,
                detected_codes_in_text=valid_codes,
            )

        # 2. Keyword scoring
        scores: Dict[str, float] = {}
        signals: List[str] = []

        for code, keywords in self.vocabulary.items():
            matches = [kw for kw in keywords if kw in text_lower]
            if matches:
                scores[code] = len(matches) / len(keywords)
                if matches:
                    signals.extend(f"Keyword: '{kw}'" for kw in matches[:2])

        if not scores:
            return self._unknown_result()

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_code, top_score = sorted_scores[0]
        second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0

        # Normalise: max keyword score → confidence band
        confidence = min(0.98, top_score * 1.5)
        needs_confirmation = (
            confidence < CONFIDENCE_THRESHOLD
            or (top_score - second_score) < 0.10
        )

        # 3. LLM fallback for low-confidence detection
        if needs_confirmation and self.llm:
            llm_code = self._llm_detect(text[:3000])
            if llm_code and llm_code in self.names:
                top_code = llm_code
                confidence = 0.82
                signals.append(f"LLM suggested: {llm_code}")
                needs_confirmation = True  # still ask user

        alternatives = [s[0] for s in sorted_scores[1:3]]
        return SubjectDetectionResult(
            subject_code=top_code,
            subject_name=self.names.get(top_code, top_code),
            confidence=round(confidence, 4),
            all_scores={c: round(s, 4) for c, s in sorted_scores},
            signals_found=signals[:6],
            needs_confirmation=needs_confirmation,
            detected_codes_in_text=valid_codes,
        )

    def register_subject(self, code: str, name: str, keywords: List[str]):
        """Add a new subject to the detector at runtime."""
        self.names[code] = name
        self.vocabulary[code] = keywords

    def _llm_detect(self, text_sample: str) -> Optional[str]:
        prompt = (
            f"What university subject code does this document belong to?\n"
            f"Known codes: {', '.join(self.names.keys())}\n\n"
            f"Document sample:\n{text_sample[:1500]}\n\n"
            f"Reply with ONLY the subject code (e.g., BAI401) or 'UNKNOWN'."
        )
        result = self.llm.generate(prompt, temperature=0.1, max_tokens=10)
        result = result.strip().upper()
        return result if result in self.names else None

    def _unknown_result(self) -> SubjectDetectionResult:
        return SubjectDetectionResult(
            subject_code="UNKNOWN",
            subject_name="Unknown",
            confidence=0.0,
            needs_confirmation=True,
        )
