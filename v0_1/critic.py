"""
AION Module: Self-Critic Gate & Stage 8 Extended Critic
=========================================================
Self-Critic Gate with multi-gate validation, grounding checks, and corruption detection.
"""

import re
from dataclasses import dataclass, field
from typing import Tuple, List, Optional
from .schemas import GeneratedQuestion
from .turbo import review_turbo
from .validator import _printable_ratio


def review(q: GeneratedQuestion) -> Tuple[bool, str]:
    """
    Self-Critic Gate with template-question and quality detection.
    """
    if q.ideal_answer is None:
        return review_turbo(q)

    if len(q.question_text) < 15:
        return False, "RC-06: question too short"

    if len(q.ideal_answer) < 20:
        return False, "RC-01: ideal answer missing or too thin"

    if "?" not in q.question_text:
        return False, "RC-04: no interrogative detected"

    bad_templates = [
        "Critically analyze the principle:",
        "Critically analyze the statement:",
        "What are its core academic implications?",
        "[INVALID CONCEPT SKIPPED",
        "[SKIPPED:",
    ]
    for bad in bad_templates:
        if bad in q.question_text:
            return False, "RC-07: template fallback question (LLM not connected)"

    if "http://" in q.question_text or "https://" in q.question_text:
        return False, "RC-08: question contains URLs"

    if q.ideal_answer.startswith("[SKIPPED:") or q.ideal_answer.startswith("[INVALID"):
        return False, "RC-09: ideal answer is a skip marker"

    return True, "ACCEPTED"


# -- Stage 8 Extended Critic ---------------------------------------------------

@dataclass
class CriticVerdict:
    passed:       bool
    score:        float          # 0.0 -> 1.0
    reason_code:  str            # RC-01 … RC-12
    reason:       str
    fix:          str = ""

    REASON_CODES = {
        "RC-01": "Grammar or language error",
        "RC-02": "Bloom level mismatch",
        "RC-03": "Marks mismatch",
        "RC-04": "Concept drift — out of scope",
        "RC-05": "Hallucination — unsupported concept",
        "RC-06": "Professor style mismatch",
        "RC-07": "Duplicate question",
        "RC-08": "Structural violation",
        "RC-09": "Numerical error",
        "RC-10": "Diagram missing",
        "RC-11": "Corrupted text in question",
        "RC-12": "Question not supported by retrieved evidence",
    }


class CriticExtended:
    """
    Extended critic that validates questions against their evidence.
    Runs after the LLM generates a question but before rendering.
    """

    def validate_question(
        self,
        question:        str,
        evidence_chunks: List[str],
        bloom_level:     int = 2,
        module_id:       str = "",
    ) -> CriticVerdict:
        verdict = self._check_corruption(question)
        if not verdict.passed:
            return verdict

        verdict = self._check_length(question)
        if not verdict.passed:
            return verdict

        verdict = self._check_grounding(question, evidence_chunks)
        if not verdict.passed:
            return verdict

        verdict = self._check_hallucination(question, evidence_chunks)
        if not verdict.passed:
            return verdict

        verdict = self._check_bloom_verb(question, bloom_level)
        if not verdict.passed:
            return verdict

        return CriticVerdict(
            passed=True, score=0.92,
            reason_code="PASS",
            reason="All critic checks passed.",
        )

    def _check_corruption(self, question: str) -> CriticVerdict:
        ratio = _printable_ratio(question)
        if ratio < 0.90:
            return CriticVerdict(
                passed=False, score=ratio,
                reason_code="RC-11",
                reason=f"Question contains corrupted text (printable ratio: {ratio:.0%}).",
                fix="Regenerate this question from clean chunks.",
            )

        if re.search(r'[\x00-\x1f\x7f-\x9f]', question):
            return CriticVerdict(
                passed=False, score=0.1,
                reason_code="RC-11",
                reason="Question contains control characters.",
                fix="Strip binary content and regenerate.",
            )

        if re.search(r'\$[A-Za-z0-9+/]{6,}', question):
            return CriticVerdict(
                passed=False, score=0.2,
                reason_code="RC-11",
                reason="Question contains base64/binary artifacts.",
                fix="Clean source document and regenerate.",
            )

        return CriticVerdict(passed=True, score=0.95, reason_code="PASS", reason="Corruption check passed.")

    def _check_length(self, question: str) -> CriticVerdict:
        words = question.split()
        if len(words) < 8:
            return CriticVerdict(
                passed=False, score=0.2,
                reason_code="RC-08",
                reason=f"Question is too short ({len(words)} words, min 8).",
                fix="Regenerate with a complete question sentence.",
            )
        return CriticVerdict(passed=True, score=0.9, reason_code="PASS", reason="Length check passed.")

    def _check_grounding(self, question: str, chunks: List[str]) -> CriticVerdict:
        if not chunks:
            return CriticVerdict(
                passed=False, score=0.0,
                reason_code="RC-12",
                reason="No evidence chunks provided for grounding check.",
                fix="Ensure retriever returns chunks before critic runs.",
            )

        combined  = " ".join(chunks).lower()
        q_nouns   = set(re.findall(r'\b[A-Z][a-zA-Z]{3,}\b', question))
        q_lower   = set(n.lower() for n in q_nouns)

        if not q_lower:
            return CriticVerdict(passed=True, score=0.8, reason_code="PASS", reason="No key nouns to verify.")

        found    = {n for n in q_lower if n in combined}
        coverage = len(found) / len(q_lower)

        if coverage < 0.50:
            missing = q_lower - found
            return CriticVerdict(
                passed=False, score=coverage,
                reason_code="RC-12",
                reason=f"Question grounding failed. Key terms not in evidence: {list(missing)[:300]}. Coverage: {coverage:.0%}.",
                fix="Retrieve chunks that cover the missing concepts.",
            )

        return CriticVerdict(passed=True, score=min(1.0, 0.7 + coverage * 0.3), reason_code="PASS", reason="Grounding check passed.")

    def _check_hallucination(self, question: str, chunks: List[str]) -> CriticVerdict:
        if not chunks:
            return CriticVerdict(passed=True, score=0.5, reason_code="PASS", reason="No chunks to check against.")

        combined = " ".join(chunks).lower()
        concepts = re.findall(r'\b[A-Z][A-Z][a-zA-Z]+\b', question)
        if not concepts:
            return CriticVerdict(passed=True, score=0.85, reason_code="PASS", reason="No named concepts to verify.")

        unsupported = [c for c in concepts if c.lower() not in combined and len(c) > 3]

        if len(unsupported) > 2:
            return CriticVerdict(
                passed=False, score=0.3,
                reason_code="RC-05",
                reason=f"Potential hallucination: {unsupported[:300]} not found in evidence.",
                fix="Verify these concepts exist in source material.",
            )

        return CriticVerdict(passed=True, score=0.88, reason_code="PASS", reason="Hallucination check passed.")

    def _check_bloom_verb(self, question: str, bloom_level: int) -> CriticVerdict:
        bloom_verbs = {
            1: {"define","list","state","recall","identify","name"},
            2: {"explain","describe","summarize","discuss","interpret"},
            3: {"apply","illustrate","demonstrate","solve","calculate"},
            4: {"compare","analyze","differentiate","examine","contrast"},
            5: {"evaluate","justify","assess","critique","argue"},
            6: {"design","develop","create","propose","formulate"},
        }

        first_word = question.strip().split()[0].lower().rstrip()
        expected   = bloom_verbs.get(bloom_level, set())

        if expected and first_word not in expected:
            actual_level = next((lvl for lvl, verbs in bloom_verbs.items() if first_word in verbs), None)
            if actual_level and actual_level != bloom_level:
                return CriticVerdict(
                    passed=False, score=0.6, reason_code="RC-02",
                    reason=f"Bloom mismatch: verb '{first_word}' maps to L{actual_level} but question is declared L{bloom_level}.",
                    fix=f"Replace '{first_word}' with: {list(expected)[:300]}",
                )

        return CriticVerdict(passed=True, score=0.9, reason_code="PASS", reason="Bloom verb check passed.")


_academic_critic = CriticExtended()

def review_extended(question: str, evidence_chunks: List[str], bloom_level: int = 2) -> CriticVerdict:
    return _academic_critic.validate_question(question, evidence_chunks, bloom_level)
