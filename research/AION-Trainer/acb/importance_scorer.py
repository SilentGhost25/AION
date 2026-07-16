# AION-Trainer/acb/importance_scorer.py
"""
Dynamic Importance Scorer.

Importance is not "how many times mentioned". It is:

    Academic Importance = f(
        syllabus_weight,
        book_coverage,
        question_bank_frequency,
        previous_paper_frequency,
        professor_notes_weight,
        learning_outcome_importance,
        bloom_ceiling,
    )

These are weighted separately because "appeared once in a textbook"
is very different from "appeared in 16 previous papers".
"""

import logging
import re
from typing import Dict, Optional
from acb.concept import Concept
from acb.syllabus_parser import ParsedSyllabus

logger = logging.getLogger("aion.acb.importance")


WEIGHTS: Dict[str, float] = {
    "syllabus_mentioned": 0.25,
    "learning_outcome_tag": 0.10,
    "book_coverage": 0.15,
    "question_bank_frequency": 0.15,
    "previous_paper_frequency": 0.20,
    "professor_notes": 0.07,
    "bloom_ceiling": 0.08,          # higher bloom = more important
}


class ImportanceScorer:
    def __init__(
        self,
        max_qb_frequency: int = 30,
        max_pq_frequency: int = 20,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.max_qb = max_qb_frequency
        self.max_pq = max_pq_frequency
        self.weights = weights or WEIGHTS

    def compute(self, concept: Concept, syllabus: Optional[ParsedSyllabus] = None) -> float:
        scores = {}

        # 1. Syllabus mention
        is_in_syllabus = concept.syllabus_mentions > 0
        if syllabus:
            # Check direct or keyword matches
            syllabus_words = {re.sub(r"[^\w\s]", "", t.lower()).strip() for _, t in syllabus.all_topics()}
            name_norm = re.sub(r"[^\w\s]", "", concept.name.lower()).strip()
            if name_norm in syllabus_words or any(alias.lower() in syllabus_words for alias in concept.aliases):
                is_in_syllabus = True
        scores["syllabus_mentioned"] = 1.0 if is_in_syllabus else 0.0

        # 2. Learning Outcomes
        has_lo_match = False
        if syllabus:
            concept_terms = [concept.name.lower()] + [a.lower() for a in concept.aliases]
            for m in syllabus.modules:
                for outcome in m.learning_outcomes:
                    outcome_lower = outcome.lower()
                    if any(term in outcome_lower for term in concept_terms):
                        has_lo_match = True
                        break
                if has_lo_match:
                    break
        scores["learning_outcome_tag"] = 1.0 if has_lo_match else 0.0

        # 3. Book coverage (target 3 textbooks for full credit)
        book_sources = [s for s in concept.sources if s.source_type == "textbook"]
        scores["book_coverage"] = min(1.0, len(book_sources) / 3.0)

        # 4. Question Bank Frequency
        scores["question_bank_frequency"] = min(1.0, concept.question_bank_frequency / self.max_qb)

        # 5. Previous Paper Frequency
        scores["previous_paper_frequency"] = min(1.0, concept.previous_paper_frequency / self.max_pq)

        # 6. Professor Notes presence
        scores["professor_notes"] = 1.0 if concept.professor_notes_present else 0.0

        # 7. Bloom Ceiling
        bloom_scores = {
            "L6": 1.0,
            "L5": 0.9,
            "L4": 0.8,
            "L3": 0.7,
            "L2": 0.5,
            "L1": 0.3,
        }
        highest = concept.bloom_progression.highest_level()
        scores["bloom_ceiling"] = bloom_scores.get(highest, 0.3)

        # Final weighted dynamic calculation
        importance_score = sum(
            scores[key] * self.weights.get(key, 0.0)
            for key in self.weights
        )

        concept.importance = round(importance_score, 4)
        return concept.importance

    def compute_all(self, concepts: list, syllabus: Optional[ParsedSyllabus] = None):
        for concept in concepts:
            self.compute(concept, syllabus)
