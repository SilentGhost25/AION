# AION-Trainer/acb/confidence_engine.py
"""
Confidence Engine — answers "why does this concept belong here?"

For every concept it builds:
    Evidence list
    Reasoning chain
    Confidence score
    Module assignment recommendation

Concepts below 85% confidence are flagged as NEEDS_VERIFICATION.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from acb.concept import Concept, ConceptStatus
from acb.syllabus_parser import ParsedSyllabus

logger = logging.getLogger("aion.acb.confidence")

CONFIDENCE_THRESHOLD = 0.85


@dataclass
class EvidenceItem:
    description: str
    weight: float               # contribution to confidence score
    source_type: str


@dataclass
class ConceptReasoning:
    concept_name: str
    confidence: float
    recommended_module: Optional[int]
    is_primary: bool            # explicitly named vs inferred
    evidence: List[EvidenceItem] = field(default_factory=list)
    reasoning_chain: List[str] = field(default_factory=list)
    needs_verification: bool = False
    conflicts: List[str] = field(default_factory=list)


class ConfidenceEngine:
    """
    Computes per-concept confidence scores from assembled evidence.
    """

    def __init__(self, syllabus: Optional[ParsedSyllabus] = None):
        self.syllabus = syllabus
        self._syllabus_topics: Dict[str, int] = {}
        if syllabus:
            self._build_syllabus_index()

    def _build_syllabus_index(self):
        """Normalised topic -> module_number."""
        for mod in self.syllabus.modules:
            for topic in mod.topics:
                self._syllabus_topics[self._norm(topic)] = mod.module_number

    def compute(self, concept: Concept) -> ConceptReasoning:
        evidence: List[EvidenceItem] = []
        reasoning: List[str] = []
        conflicts: List[str] = []

        # 1. Syllabus mention
        matched_module, in_syllabus = self._check_syllabus(concept)
        if in_syllabus:
            evidence.append(EvidenceItem(
                "Concept explicitly named in syllabus", 0.40, "syllabus"
            ))
            reasoning.append(f"Explicitly named in syllabus under Module {matched_module}")
        else:
            evidence.append(EvidenceItem(
                "Concept not in syllabus", -0.05, "syllabus"
            ))
            reasoning.append("Not explicitly named in syllabus (supplementary)")

        # 2. Textbook coverage
        book_sources = [s for s in concept.sources if s.source_type == "textbook"]
        if book_sources:
            book_score = min(0.25, len(book_sources) * 0.10)
            evidence.append(EvidenceItem(
                f"Appears in {len(book_sources)} textbook(s)", book_score, "textbook"
            ))
            avg_reliability = sum(s.reliability for s in book_sources) / len(book_sources)
            reasoning.append(
                f"Found in {len(book_sources)} textbook(s) with avg reliability {avg_reliability:.2f}"
            )

        # 3. Question bank frequency
        if concept.question_bank_frequency > 0:
            qb_score = min(0.15, concept.question_bank_frequency * 0.03)
            evidence.append(EvidenceItem(
                f"Appears {concept.question_bank_frequency} times in question bank",
                qb_score, "question_bank"
            ))
            reasoning.append(f"Appears {concept.question_bank_frequency} times in question bank")

        # 4. Previous paper frequency
        if concept.previous_paper_frequency > 0:
            pq_score = min(0.15, concept.previous_paper_frequency * 0.03)
            evidence.append(EvidenceItem(
                f"Asked {concept.previous_paper_frequency} times in previous papers",
                pq_score, "previous_paper"
            ))
            reasoning.append(
                f"Referenced {concept.previous_paper_frequency} times in past exam papers"
            )

        # 5. Professor notes
        if concept.professor_notes_present:
            evidence.append(EvidenceItem("Present in professor notes", 0.10, "notes"))
            reasoning.append("Mentioned in faculty-uploaded notes")

        # 6. Has definition
        if concept.definition:
            evidence.append(EvidenceItem("Has a definition", 0.05, "quality"))

        # 7. Has diagram
        if concept.requires_diagram:
            evidence.append(EvidenceItem("Diagram material available", 0.05, "quality"))

        # Compute score
        raw_score = sum(e.weight for e in evidence)
        confidence = max(0.0, min(1.0, raw_score))

        # Module conflict check
        if len(concept.module_links) > 1:
            modules = {ml.module for ml in concept.module_links}
            if len(modules) > 1:
                conflict_msg = (
                    f"Linked to multiple modules: {modules}. "
                    f"Needs manual verification of primary module."
                )
                conflicts.append(conflict_msg)

        needs_verification = (
            confidence < CONFIDENCE_THRESHOLD or bool(conflicts)
        )

        # Apply status
        if needs_verification:
            concept.status = ConceptStatus.NEEDS_VERIFICATION
        else:
            concept.status = ConceptStatus.VERIFIED

        concept.confidence = confidence

        return ConceptReasoning(
            concept_name=concept.name,
            confidence=confidence,
            recommended_module=matched_module or concept.primary_module(),
            is_primary=in_syllabus,
            evidence=evidence,
            reasoning_chain=reasoning,
            needs_verification=needs_verification,
            conflicts=conflicts,
        )

    def compute_all(self, concepts: List[Concept]) -> List[ConceptReasoning]:
        reasonings = []
        for concept in concepts:
            r = self.compute(concept)
            reasonings.append(r)

        needs_verification = sum(1 for r in reasonings if r.needs_verification)
        logger.info(
            f"[ConfidenceEngine] Processed {len(reasonings)} concepts. "
            f"{needs_verification} need verification."
        )
        return reasonings

    def _check_syllabus(self, concept: Concept):
        name_norm = self._norm(concept.name)
        matched_module = self._syllabus_topics.get(name_norm)

        if not matched_module:
            for kw in concept.keywords[:6]:
                m = self._syllabus_topics.get(kw)
                if m:
                    matched_module = m
                    break

        return matched_module, matched_module is not None

    @staticmethod
    def _norm(text: str) -> str:
        import re
        return re.sub(r"[^\w\s]", "", text.lower()).strip()
