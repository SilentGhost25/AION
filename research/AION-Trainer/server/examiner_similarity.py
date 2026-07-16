# AION-Trainer/server/examiner_similarity.py
"""
Examiner Similarity Score (ESS) — how closely the candidate model's
question-generation behavior matches real professor-authored papers.
"""

import random
import logging
import math
from collections import Counter
from typing import List, Dict, Any

from builders.examiner_style import ExaminerStyleExtractor, ExaminerStyle
from server.pyq_extractor import PYQParser
from server.candidate_generator import CandidateQuestionGenerator

logger = logging.getLogger("aion.server.examiner_similarity")

MARKS_BINS = [2, 5, 10, 15, 20]
BLOOM_LEVELS = ["L1", "L2", "L3", "L4", "L5", "L6"]


class ExaminerSimilarityScorer:
    def __init__(self, generator: CandidateQuestionGenerator, seed: int = 42):
        self.generator = generator
        self.style_extractor = ExaminerStyleExtractor()
        self.pyq_parser = PYQParser()
        self._rng = random.Random(seed)

    def compute(
        self,
        previous_paper_paths: List[str],
        knowledge_samples: List[str],
        sample_size: int = 30,
    ) -> Dict[str, float]:
        """
        Returns a dict with the overall examiner_similarity_score plus
        its three components.
        """
        real_questions = []
        for path in previous_paper_paths:
            real_questions.extend(self.pyq_parser.parse_file(path))

        if not real_questions:
            logger.warning("[ESS] No real questions extracted from previous papers; "
                            "cannot compute examiner similarity.")
            return {"examiner_similarity_score": 0.0, "verb_similarity": 0.0,
                    "bloom_similarity": 0.0, "marks_similarity": 0.0,
                    "note": "no_reference_questions"}

        real_style = self.style_extractor.extract_from_papers([{"questions": real_questions}])

        if not knowledge_samples:
            logger.warning("[ESS] No knowledge samples available to condition generation.")
            return {"examiner_similarity_score": 0.0, "verb_similarity": 0.0,
                    "bloom_similarity": 0.0, "marks_similarity": 0.0,
                    "note": "no_knowledge_samples"}

        targets = self._sample_targets(real_style, n=sample_size)
        candidate_style = ExaminerStyle()

        for bloom, marks in targets:
            knowledge = self._rng.choice(knowledge_samples)
            generated_text = self.generator.generate(knowledge, bloom, marks)
            if not generated_text:
                continue
            verb = self.style_extractor._detect_verb(generated_text)
            candidate_style.add_question(generated_text, bloom, marks, verb)

        if candidate_style.total_questions == 0:
            logger.warning("[ESS] Generator produced no usable output.")
            return {"examiner_similarity_score": 0.0, "verb_similarity": 0.0,
                    "bloom_similarity": 0.0, "marks_similarity": 0.0,
                    "note": "generator_produced_nothing"}

        verb_sim = self._distribution_similarity(
            real_style.verb_distribution, candidate_style.verb_distribution
        )
        bloom_sim = self._distribution_similarity(
            real_style.bloom_distribution, candidate_style.bloom_distribution, keys=BLOOM_LEVELS
        )
        marks_sim = self._distribution_similarity(
            real_style.marks_distribution, candidate_style.marks_distribution, keys=MARKS_BINS
        )

        overall = (verb_sim * 0.5) + (bloom_sim * 0.25) + (marks_sim * 0.25)

        logger.info(
            f"[ESS] verb={verb_sim:.3f} bloom={bloom_sim:.3f} "
            f"marks={marks_sim:.3f} overall={overall:.3f}"
        )

        return {
            "examiner_similarity_score": overall,
            "verb_similarity": verb_sim,
            "bloom_similarity": bloom_sim,
            "marks_similarity": marks_sim,
        }

    def _sample_targets(self, style: ExaminerStyle, n: int) -> List[tuple]:
        """Sample (bloom, marks) pairs weighted by their real-world frequency."""
        bloom_keys = list(style.bloom_distribution.keys()) or BLOOM_LEVELS
        bloom_weights = [style.bloom_distribution.get(k, 1) for k in bloom_keys]

        marks_keys = list(style.marks_distribution.keys()) or MARKS_BINS
        marks_weights = [style.marks_distribution.get(k, 1) for k in marks_keys]

        blooms = self._rng.choices(bloom_keys, weights=bloom_weights, k=n)
        marks = self._rng.choices(marks_keys, weights=marks_weights, k=n)
        return list(zip(blooms, marks))

    def _distribution_similarity(
        self, dist_a: Dict[Any, int], dist_b: Dict[Any, int], keys: List[Any] = None
    ) -> float:
        """
        Cosine similarity between two frequency distributions.
        """
        keys = keys or sorted(set(dist_a.keys()) | set(dist_b.keys()))
        vec_a = [dist_a.get(k, 0) for k in keys]
        vec_b = [dist_b.get(k, 0) for k in keys]

        norm_a = math.sqrt(sum(x * x for x in vec_a))
        norm_b = math.sqrt(sum(x * x for x in vec_b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        return dot / (norm_a * norm_b)
