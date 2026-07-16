# AION-Trainer/ese/question_ranker.py
"""
Question Ranker — deterministic scoring of candidate questions.

Each candidate is scored on multiple academic dimensions.
The highest-scoring candidate that passes minimum thresholds
is selected for language realization.

Fully deterministic — no LLM, no randomness.
The scoring logic is auditable, explainable, and testable.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

from ese.answer_blueprint import AnswerBlueprint
from ese.question_discoverer import QuestionCandidate
from server.prompt.assessment_intent import AssessmentIntent

logger = logging.getLogger("aion.ese.ranker")

BLOOM_VERBS: Dict[str, List[str]] = {
    "L1": ["define", "list", "state", "name", "identify"],
    "L2": ["explain", "describe", "discuss", "summarize"],
    "L3": ["apply", "illustrate", "trace", "demonstrate", "solve"],
    "L4": ["compare", "analyze", "contrast", "differentiate"],
    "L5": ["evaluate", "justify", "critique", "assess"],
    "L6": ["design", "develop", "propose", "construct"],
}


@dataclass
class RankingScore:
    candidate: QuestionCandidate
    bloom_alignment: float = 0.0
    component_coverage: float = 0.0
    structural_quality: float = 0.0
    novelty: float = 0.0
    vtu_style: float = 0.0
    overall: float = 0.0
    disqualified: bool = False
    disqualification_reason: str = ""

    def compute_overall(self, weights: Dict[str, float] = None):
        w = weights or {
            "bloom_alignment": 0.30,
            "component_coverage": 0.25,
            "structural_quality": 0.20,
            "novelty": 0.15,
            "vtu_style": 0.10,
        }
        self.overall = (
            self.bloom_alignment * w["bloom_alignment"] +
            self.component_coverage * w["component_coverage"] +
            self.structural_quality * w["structural_quality"] +
            self.novelty * w["novelty"] +
            self.vtu_style * w["vtu_style"]
        )

    def explain(self) -> str:
        if self.disqualified:
            return f"DISQUALIFIED: {self.disqualification_reason}"
        return (
            f"Overall: {self.overall:.3f} | "
            f"Bloom: {self.bloom_alignment:.2f} | "
            f"Coverage: {self.component_coverage:.2f} | "
            f"Structure: {self.structural_quality:.2f} | "
            f"Novelty: {self.novelty:.2f} | "
            f"VTU: {self.vtu_style:.2f}"
        )


class QuestionRanker:
    def __init__(
        self,
        min_bloom_alignment: float = 0.4,
        min_component_coverage: float = 0.2,
    ):
        self.min_bloom = min_bloom_alignment
        self.min_coverage = min_component_coverage

    def rank(
        self,
        candidates: List[QuestionCandidate],
        blueprint: AnswerBlueprint,
        intent: AssessmentIntent,
        previously_asked: List[str] = None,
    ) -> List[RankingScore]:
        scores = []
        previously_asked = previously_asked or []

        for candidate in candidates:
            score = self._score_candidate(candidate, blueprint, intent, previously_asked)
            score.compute_overall()
            scores.append(score)

        scores.sort(key=lambda s: (not s.disqualified, s.overall), reverse=True)

        for i, score in enumerate(scores[:3], 1):
            logger.info(f"[Ranker] Rank {i}: {score.explain()} | {score.candidate.text[:60]}")

        return scores

    def best(
        self,
        candidates: List[QuestionCandidate],
        blueprint: AnswerBlueprint,
        intent: AssessmentIntent,
        previously_asked: List[str] = None,
    ) -> Optional[Tuple[QuestionCandidate, RankingScore]]:
        scores = self.rank(candidates, blueprint, intent, previously_asked)
        for score in scores:
            if not score.disqualified and score.overall >= 0.3:
                return score.candidate, score
        return None

    def _score_candidate(
        self,
        candidate: QuestionCandidate,
        blueprint: AnswerBlueprint,
        intent: AssessmentIntent,
        previously_asked: List[str],
    ) -> RankingScore:
        score = RankingScore(candidate=candidate)
        text = candidate.text
        text_lower = text.lower()

        # 1. Bloom alignment
        bloom_verbs = BLOOM_VERBS.get(blueprint.bloom_level, [])
        score.bloom_alignment = (
            1.0 if any(v in text_lower for v in bloom_verbs) else 0.3
        )
        if score.bloom_alignment < self.min_bloom:
            score.disqualified = True
            score.disqualification_reason = (
                f"No {blueprint.bloom_level} verb detected. "
                f"Expected one of: {bloom_verbs}"
            )
            return score

        # 2. Component coverage
        required = [s.component for s in blueprint.required_components if s.required]
        if required:
            covered = len(set(candidate.covers_components) & set(required))
            score.component_coverage = covered / len(required)
        else:
            score.component_coverage = 0.8

        if score.component_coverage < self.min_coverage:
            score.disqualified = True
            score.disqualification_reason = (
                f"Covers only {score.component_coverage:.0%} of required components"
            )
            return score

        # 3. Structural quality
        words = text.split()
        starts_capital = text[0].isupper() if text else False
        ends_punct = text.rstrip()[-1] in ".?" if text.rstrip() else False
        length_ok = 5 <= len(words) <= 50
        verb_first = any(
            text_lower.startswith(v) for v in BLOOM_VERBS.get(blueprint.bloom_level, [])
        )
        score.structural_quality = (
            0.25 * starts_capital +
            0.25 * ends_punct +
            0.25 * length_ok +
            0.25 * verb_first
        )

        # 4. Novelty (how different is this from previously asked questions)
        score.novelty = self._compute_novelty(text, previously_asked)

        # 5. VTU style adherence
        score.vtu_style = self._compute_vtu_style(text, blueprint)

        return score

    def _compute_novelty(self, text: str, previously_asked: List[str]) -> float:
        if not previously_asked:
            return 1.0
        text_words = set(text.lower().split())
        max_overlap = 0.0
        for past in previously_asked:
            past_words = set(past.lower().split())
            overlap = len(text_words & past_words) / max(len(text_words), 1)
            max_overlap = max(max_overlap, overlap)
        return max(0.0, 1.0 - max_overlap)

    def _compute_vtu_style(self, text: str, blueprint: AnswerBlueprint) -> float:
        score = 0.0
        text_lower = text.lower()
        if re.match(r"^[A-Z]", text):
            score += 0.25
        if re.search(r"with\s+(a\s+)?(suitable\s+)?(example|diagram)", text_lower):
            score += 0.25
        elif blueprint.marks >= 10:
            pass  # expected but missing
        else:
            score += 0.25
        if len(text.split()) <= 40:
            score += 0.25
        if re.search(r"\b(neat|suitable|appropriate)\b", text_lower):
            score += 0.25
        return score
