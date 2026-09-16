"""
AION v2 Pedagogical Coverage Ledger
====================================
Maintains authoritative state over syllabus coverage, question allocations,
and topic assessment frequency.

Key Architectural Responsibilities:
1. Multi-KU tracking: Distinguishes PRIMARY_KU from SECONDARY_KU in assessment records.
2. Invariant enforcement: Strictly blocks candidate selection for scope-uncertain or out-of-syllabus KUs.
3. Decoupled, explainable ranking: Ranks candidates based on pedagogical demand and repetition penalties.
4. Comprehensive coverage metrics: Tracks KU coverage, concept coverage, module coverage, and assessment histograms.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from aion.core.knowledge.contracts import (
    CognitiveDemand,
    KnowledgeUnit,
    SemanticType,
)


class KUAssessmentState(str, Enum):
    UNASSESSED = "UNASSESSED"
    ALLOCATED = "ALLOCATED"
    DRAFTED = "DRAFTED"
    ASSESSED = "ASSESSED"
    EXCLUDED = "EXCLUDED"


@dataclass
class AssessmentRecord:
    """
    Record of a question's multi-KU assessment footprint.
    """
    question_id: str
    module_index: int                      # 1 to 5
    primary_ku_id: str
    secondary_ku_ids: List[str] = field(default_factory=list)
    assessed_concept: str = ""
    cognitive_demand: Optional[CognitiveDemand] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "module_index": self.module_index,
            "primary_ku_id": self.primary_ku_id,
            "secondary_ku_ids": self.secondary_ku_ids,
            "assessed_concept": self.assessed_concept,
            "cognitive_demand": self.cognitive_demand.value if self.cognitive_demand else None,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AssessmentRecord:
        cd = CognitiveDemand(data["cognitive_demand"]) if data.get("cognitive_demand") else None
        return cls(
            question_id=data["question_id"],
            module_index=int(data["module_index"]),
            primary_ku_id=data["primary_ku_id"],
            secondary_ku_ids=list(data.get("secondary_ku_ids", [])),
            assessed_concept=data.get("assessed_concept", ""),
            cognitive_demand=cd,
            timestamp=float(data.get("timestamp", 0.0)),
        )


class CoverageLedger:
    """
    Authoritative state machine governing topic coverage and candidate KU selection.
    """

    def __init__(self):
        self.kus: Dict[str, KnowledgeUnit] = {}
        self.states: Dict[str, KUAssessmentState] = {}
        self.assessment_records: List[AssessmentRecord] = []
        self._recently_assessed_concepts: List[str] = []

    def register_kus(self, kus: Dict[str, KnowledgeUnit]) -> None:
        """Register compiled KnowledgeUnits into the coverage ledger."""
        for ku_id, ku in kus.items():
            self.kus[ku_id] = ku
            if not ku.in_syllabus:
                self.states[ku_id] = KUAssessmentState.EXCLUDED
            else:
                self.states[ku_id] = KUAssessmentState.UNASSESSED

    def select_candidate_ku(
        self,
        module_index: int,
        target_bloom: Optional[CognitiveDemand] = None,
        target_semantic_type: Optional[SemanticType] = None,
    ) -> Tuple[Optional[KnowledgeUnit], Dict[str, Any]]:
        """
        Selects the best candidate KnowledgeUnit for examination question generation.
        Strictly excludes any KU that is out of syllabus or scope-uncertain.
        Ranks valid candidates using explainable scoring.
        """
        candidate_pool: List[Tuple[KnowledgeUnit, float, Dict[str, Any]]] = []

        for ku_id, ku in self.kus.items():
            # Module check
            if ku.module_index != module_index:
                continue

            # INVARIANT: Block scope-uncertain or out-of-syllabus KUs
            if ku.scope_uncertain or not ku.in_syllabus:
                continue

            # Check state
            st = self.states.get(ku_id, KUAssessmentState.UNASSESSED)
            if st in (KUAssessmentState.EXCLUDED, KUAssessmentState.ASSESSED):
                continue

            # Explainable Ranking Score
            score, breakdown = self._score_candidate(ku, target_bloom, target_semantic_type)
            candidate_pool.append((ku, score, breakdown))

        if not candidate_pool:
            return None, {"reason": "No eligible in-syllabus KUs found for module"}

        # Sort descending by composite score
        candidate_pool.sort(key=lambda x: x[1], reverse=True)
        best_ku, best_score, best_breakdown = candidate_pool[0]

        return best_ku, {
            "selected_ku_id": best_ku.ku_id,
            "concept_title": best_ku.concept_title,
            "score": round(best_score, 3),
            "breakdown": best_breakdown,
        }

    def _score_candidate(
        self,
        ku: KnowledgeUnit,
        target_bloom: Optional[CognitiveDemand],
        target_semantic_type: Optional[SemanticType],
    ) -> Tuple[float, Dict[str, Any]]:
        # 1. Base pedagogical importance
        imp_score = ku.importance.total

        # 2. Bloom demand alignment
        bloom_match = 0.0
        if target_bloom:
            if target_bloom in ku.bloom_affinities:
                bloom_match = 1.0
            else:
                bloom_match = 0.2
        else:
            bloom_match = 0.5

        # 3. Semantic archetype affinity
        type_match = 0.0
        if target_semantic_type:
            if ku.primary_semantic_type == target_semantic_type:
                type_match = 1.0
            else:
                type_match = 0.2
        else:
            type_match = 0.5

        # 4. Syllabus priority
        syl_priority = ku.importance.syllabus_priority

        # 5. Prior assessment penalty
        prior_penalty = ku.assessed_count * 0.50 + ku.secondary_assessed_count * 0.25

        # 6. Concept repetition penalty (recent concept repetition)
        repetition_penalty = 0.0
        if ku.concept_title in self._recently_assessed_concepts[-3:]:
            repetition_penalty = 0.40

        # Composite score
        total_score = (
            0.35 * imp_score +
            0.20 * bloom_match +
            0.20 * type_match +
            0.25 * syl_priority -
            prior_penalty -
            repetition_penalty
        )

        breakdown = {
            "importance": round(imp_score, 3),
            "bloom_match": round(bloom_match, 3),
            "type_match": round(type_match, 3),
            "syllabus_priority": round(syl_priority, 3),
            "prior_assessment_penalty": round(prior_penalty, 3),
            "concept_repetition_penalty": round(repetition_penalty, 3),
        }

        return total_score, breakdown

    def record_assessment(
        self,
        question_id: str,
        module_index: int,
        primary_ku_id: str,
        secondary_ku_ids: Optional[List[str]] = None,
        concept: str = "",
        cognitive_demand: Optional[CognitiveDemand] = None,
    ) -> AssessmentRecord:
        """
        Records the assessment of a question against primary and secondary KUs.
        Updates internal coverage ledger states and assessment counts.
        """
        sec_ids = secondary_ku_ids or []
        concept_title = concept

        # Update primary KU
        if primary_ku_id in self.kus:
            p_ku = self.kus[primary_ku_id]
            p_ku.assessed_count += 1
            self.states[primary_ku_id] = KUAssessmentState.ASSESSED
            if not concept_title:
                concept_title = p_ku.concept_title

        # Update secondary KUs
        for s_id in sec_ids:
            if s_id in self.kus:
                s_ku = self.kus[s_id]
                s_ku.secondary_assessed_count += 1
                if self.states.get(s_id) == KUAssessmentState.UNASSESSED:
                    self.states[s_id] = KUAssessmentState.ALLOCATED

        rec = AssessmentRecord(
            question_id=question_id,
            module_index=module_index,
            primary_ku_id=primary_ku_id,
            secondary_ku_ids=sec_ids,
            assessed_concept=concept_title,
            cognitive_demand=cognitive_demand,
        )
        self.assessment_records.append(rec)
        self._recently_assessed_concepts.append(concept_title)

        return rec

    def get_coverage_report(self) -> Dict[str, Any]:
        """
        Generates a comprehensive pedagogical coverage audit report.
        """
        total_kus = len(self.kus)
        in_syllabus_kus = [k for k in self.kus.values() if k.in_syllabus and not k.scope_uncertain]
        excluded_kus = [k for k in self.kus.values() if not k.in_syllabus]
        scope_uncertain_kus = [k for k in self.kus.values() if k.scope_uncertain]

        assessed_primary_ids = {r.primary_ku_id for r in self.assessment_records if r.primary_ku_id in self.kus}
        assessed_secondary_ids = {
            s_id for r in self.assessment_records for s_id in r.secondary_ku_ids if s_id in self.kus
        }
        all_touched_ids = assessed_primary_ids.union(assessed_secondary_ids)

        in_syl_total = len(in_syllabus_kus)
        primary_coverage_pct = (len(assessed_primary_ids) / in_syl_total * 100.0) if in_syl_total > 0 else 0.0
        total_touch_coverage_pct = (len(all_touched_ids) / in_syl_total * 100.0) if in_syl_total > 0 else 0.0

        # Per-module coverage
        module_breakdown: Dict[int, Dict[str, Any]] = {}
        for m in range(1, 6):
            m_kus = [k for k in in_syllabus_kus if k.module_index == m]
            m_assessed = [k for k in m_kus if k.ku_id in assessed_primary_ids]
            module_breakdown[m] = {
                "in_syllabus_kus": len(m_kus),
                "assessed_kus": len(m_assessed),
                "coverage_pct": (len(m_assessed) / len(m_kus) * 100.0) if m_kus else 0.0,
            }

        # Concept coverage
        assessed_concepts = {r.assessed_concept for r in self.assessment_records if r.assessed_concept}

        return {
            "total_kus": total_kus,
            "in_syllabus_kus": in_syl_total,
            "excluded_kus": len(excluded_kus),
            "scope_uncertain_kus": len(scope_uncertain_kus),
            "questions_recorded": len(self.assessment_records),
            "primary_assessed_kus": len(assessed_primary_ids),
            "total_touched_kus": len(all_touched_ids),
            "primary_coverage_pct": round(primary_coverage_pct, 1),
            "total_touch_coverage_pct": round(total_touch_coverage_pct, 1),
            "unique_concepts_assessed": len(assessed_concepts),
            "module_breakdown": module_breakdown,
        }
