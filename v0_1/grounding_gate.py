"""
AION Grounding Gate — Stage 5
==============================
Hard barrier between the retriever and the generator.
If the evidence base is insufficient, generation does not happen.

This is the most important quality gate in the pipeline.
Every question must be traceable to retrieved evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .validator import ACADEMIC_VOCAB, ContentValidator


# ── Configuration ─────────────────────────────────────────────────────────────

MIN_CHUNKS      = 2      # minimum valid evidence chunks
MAX_CHUNKS      = 3      # hard ceiling — never send more than 3
MIN_WORDS       = 60     # minimum total context words
MAX_WORDS       = 900    # maximum total context words
MIN_ACADEMIC    = 3      # minimum academic indicator words in combined context
ENTITY_COVERAGE = 0.50   # fraction of query key nouns that must appear in evidence


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class GatingResult:
    proceed:        bool
    reason:         str                 # gate outcome code
    message:        str                 # human-readable explanation
    pruned_chunks:  list[str]           = field(default_factory=list)
    evidence_score: float               = 0.0
    word_count:     int                 = 0

    def __bool__(self) -> bool:
        return self.proceed


# ── Grounding Gate ────────────────────────────────────────────────────────────

class GroundingGate:
    """
    Pre-generation evidence verification gate.
    Called with (query, chunks, optional module_id).
    Returns GatingResult — if proceed=False, the generator is never called.
    """

    def __init__(
        self,
        min_chunks:      int   = MIN_CHUNKS,
        max_chunks:      int   = MAX_CHUNKS,
        min_words:       int   = MIN_WORDS,
        max_words:       int   = MAX_WORDS,
        min_academic:    int   = MIN_ACADEMIC,
        entity_coverage: float = ENTITY_COVERAGE,
    ):
        self.min_chunks      = min_chunks
        self.max_chunks      = max_chunks
        self.min_words       = min_words
        self.max_words       = max_words
        self.min_academic    = min_academic
        self.entity_coverage = entity_coverage
        self.validator       = ContentValidator()

    def check(
        self,
        query:       str,
        chunks:      list[str],
        module_id:   Optional[str]       = None,
        chunk_metas: Optional[list[dict]] = None,
    ) -> GatingResult:
        """
        Main gate check.
        """
        if not chunks:
            return GatingResult(
                proceed=False,
                reason="EMPTY_RETRIEVAL",
                message="No chunks retrieved. Check the retriever or expand the query.",
            )

        clean_chunks = []
        for i, chunk in enumerate(chunks):
            score = self.validator.validate_chunk(chunk)
            if score.printable_ratio >= 0.85:
                clean_chunks.append(chunk)

        if module_id and chunk_metas:
            scoped = []
            for chunk, meta in zip(clean_chunks, chunk_metas or [{}]*len(clean_chunks)):
                chunk_module = meta.get("module_id", "")
                if not chunk_module or chunk_module == module_id:
                    scoped.append(chunk)
            clean_chunks = scoped

        if len(clean_chunks) < self.min_chunks:
            return GatingResult(
                proceed=False,
                reason="BELOW_MIN_CHUNKS",
                message=(
                    f"Only {len(clean_chunks)} valid chunk(s) available. "
                    f"Minimum {self.min_chunks} required for grounded generation."
                ),
                pruned_chunks=clean_chunks,
            )

        pruned = clean_chunks[:self.max_chunks]
        combined = " ".join(pruned)
        words    = combined.split()

        if len(words) < self.min_words:
            return GatingResult(
                proceed=False,
                reason="THIN_CONTEXT",
                message=(
                    f"Combined context is only {len(words)} words. "
                    f"Minimum {self.min_words} words required."
                ),
                pruned_chunks=pruned,
                word_count=len(words),
            )

        if len(words) > self.max_words:
            combined = " ".join(words[:self.max_words])
            pruned   = [combined]

        combined_lower = combined.lower()
        academic_count = sum(1 for w in ACADEMIC_VOCAB if w in combined_lower)

        if academic_count < self.min_academic:
            return GatingResult(
                proceed=False,
                reason="LOW_ACADEMIC",
                message=(
                    f"Only {academic_count} academic indicator words found. "
                    f"Minimum {self.min_academic} required."
                ),
                pruned_chunks=pruned,
                word_count=len(words),
            )

        key_nouns = self._extract_key_nouns(query)
        if key_nouns:
            matched = sum(1 for n in key_nouns if n in combined_lower)
            coverage = matched / len(key_nouns)
            if coverage < self.entity_coverage:
                missing = [n for n in key_nouns if n not in combined_lower]
                return GatingResult(
                    proceed=False,
                    reason="ENTITY_MISMATCH",
                    message=(
                        f"Query entities not found in evidence: {missing[:3]}. "
                        f"Coverage: {coverage:.0%} (min {self.entity_coverage:.0%})."
                    ),
                    pruned_chunks=pruned,
                    word_count=len(words),
                )

        evidence_score = self._compute_evidence_score(pruned, query, academic_count)

        print(f"[GATE] PASSED — {len(pruned)} chunks, {len(words)} words, evidence_score={evidence_score:.2f}")

        return GatingResult(
            proceed        = True,
            reason         = "PASSED_GATE",
            message        = "Evidence verified. Proceeding to generation.",
            pruned_chunks  = pruned,
            evidence_score = evidence_score,
            word_count     = len(words),
        )

    def _extract_key_nouns(self, query: str) -> list[str]:
        stopwords = {
            "the","a","an","in","of","to","for","and","or","with",
            "using","given","find","show","prove","explain","describe",
            "calculate","apply","analyze","compare","illustrate","discuss",
        }
        words = re.findall(r'\b[a-zA-Z]{4,}\b', query)
        return [w.lower() for w in words if w.lower() not in stopwords]

    def _compute_evidence_score(
        self,
        chunks:         list[str],
        query:          str,
        academic_count: int,
    ) -> float:
        combined = " ".join(chunks).lower()
        q_words  = set(re.findall(r'\b[a-zA-Z]{4,}\b', query.lower()))
        e_words  = set(re.findall(r'\b[a-zA-Z]{4,}\b', combined))

        overlap  = len(q_words & e_words) / max(1, len(q_words))
        acad_n   = min(1.0, academic_count / 10)
        chunk_n  = min(1.0, len(chunks) / self.max_chunks)

        return round(0.5 * overlap + 0.3 * acad_n + 0.2 * chunk_n, 4)


_gate = GroundingGate()

def check_grounding(
    query:       str,
    chunks:      list[str],
    module_id:   Optional[str]        = None,
    chunk_metas: Optional[list[dict]] = None,
) -> GatingResult:
    return _gate.check(query, chunks, module_id, chunk_metas)
