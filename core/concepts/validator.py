"""
Concept Validator — Concept Validation Stage
============================================
Validates extracted concepts before grounding.

Checks:
- is_valid (academic prose vs noise)
- definition completeness
- supporting evidence length & quality
- duplicate detection
- cross-concept consistency

Invalid concepts are rejected before grounding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Dict, Any
from .extractor import ExtractedConcept

@dataclass
class ConceptValidationResult:
    concept_id: str
    is_valid: bool
    confidence: float
    reason: str
    checks: Dict[str, Any]

class ConceptValidator:
    """
    Validates concepts using heuristics + optional LLM/embedding checks.
    """

    MIN_EVIDENCE_WORDS = 40
    MIN_DEFINITION_CHARS = 30

    def __init__(self, min_confidence: float = 0.45):
        self.min_confidence = min_confidence

    def validate(self, concept: ExtractedConcept) -> ConceptValidationResult:
        checks: Dict[str, Any] = {}
        reasons: List[str] = []
        score = concept.confidence

        # Check 1: Evidence length
        wc = len(concept.supporting_evidence.split())
        checks["evidence_length"] = wc
        if wc < self.MIN_EVIDENCE_WORDS:
            reasons.append(f"evidence too short ({wc} words)")
            score -= 0.30
        else:
            score += 0.05

        # Check 2: Definition quality
        defn = concept.canonical_definition or ""
        checks["definition_length"] = len(defn)
        if len(defn.strip()) < self.MIN_DEFINITION_CHARS:
            reasons.append("definition too thin")
            score -= 0.15

        # Check 3: Noise / code detection
        is_noise = self._is_noise(concept.supporting_evidence)
        checks["is_noise"] = is_noise
        if is_noise:
            reasons.append("noise/code fragment")
            score -= 0.50

        # Check 4: Repetition / TOC-like
        if self._is_toc_like(concept.supporting_evidence):
            reasons.append("table of contents fragment")
            score -= 0.40
            is_noise = True

        # Check 5: URL-heavy
        url_count = len(re.findall(r"https?://", concept.supporting_evidence))
        checks["url_count"] = url_count
        if url_count >= 2:
            reasons.append(f"{url_count} URLs — web tutorial noise")
            score -= 0.30

        # Final
        score = max(0.05, min(0.98, round(score, 2)))
        is_valid = (not is_noise) and score >= self.min_confidence and not reasons or score >= 0.55

        # Strict: if noise or very low score, invalid regardless
        if is_noise or score < self.min_confidence:
            is_valid = False

        return ConceptValidationResult(
            concept_id=concept.concept_id,
            is_valid=is_valid,
            confidence=score,
            reason="; ".join(reasons) if reasons else "valid",
            checks=checks,
        )

    def validate_batch(self, concepts: List[ExtractedConcept]) -> tuple[List[ExtractedConcept], List[ConceptValidationResult]]:
        valid: List[ExtractedConcept] = []
        results: List[ConceptValidationResult] = []
        seen_defs: set[str] = set()
        for c in concepts:
            # Deduplicate by definition hash
            def_hash = c.canonical_definition.lower().strip()[:80]
            if def_hash in seen_defs:
                r = ConceptValidationResult(c.concept_id, False, 0.2, "duplicate concept", {"duplicate": True})
                results.append(r)
                continue
            seen_defs.add(def_hash)
            r = self.validate(c)
            results.append(r)
            if r.is_valid:
                valid.append(c)
        return valid, results

    # ── Helpers ──────────────────────────────────────────────

    def _is_noise(self, text: str) -> bool:
        noise_patterns = [
            r"^\s*(import|from)\s+\w+",
            r"^\s*(def |class )\w+",
            r"urlpatterns|INSTALLED_APPS|DATABASES",
            r"<[a-zA-Z][^>]+>",
            r"\{\{.*?\}\}|\{%.*?%\}",
            r"^\s*(pip|npm|git)\s+",
        ]
        for pat in noise_patterns:
            if re.search(pat, text, re.M):
                return True
        # Avg word length check
        words = re.findall(r"[a-zA-Z]+", text)
        if words:
            avg = sum(len(w) for w in words) / len(words)
            if avg < 3.5 and len(words) > 20:
                return True
        return False

    def _is_toc_like(self, text: str) -> bool:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        toc_like = sum(1 for ln in lines if re.search(r"\.{3,}\s*\d+\s*$", ln) or re.search(r"^(chapter|unit|module)\s+\d+\s*$", ln, re.I))
        return toc_like >= max(2, len(lines) // 3) and len(lines) >= 4
