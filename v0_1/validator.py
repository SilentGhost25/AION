"""
AION Content Validator — Stage 2
=================================
Runs immediately after extraction.
Every chunk is scored. Chunks below threshold are permanently rejected.
Corrupted content never reaches the LLM.

Algorithm:
    For each chunk:
        1. Compute printable_ratio  (binary/garbage detector)
        2. Compute unicode_validity (malformed Unicode detector)
        3. Compute word_density     (real words vs symbol soup)
        4. Compute academic_density (subject-relevant vocabulary)
        5. Compute length_score     (too short = useless, too long = unfocused)
        6. Aggregate -> quality_score ∈ [0.0, 1.0]
        7. quality_score >= MIN_QUALITY -> PASS, else REJECT

Thresholds (tuned for VTU academic material):
    MIN_PRINTABLE   = 0.85   (85% of chars must be printable ASCII/Unicode letters)
    MIN_WORD_DENSITY = 0.60  (60% of tokens must be dictionary-like words)
    MIN_ACADEMIC    = 2      (at least 2 academic indicator words)
    MIN_QUALITY     = 0.70   (overall gate)
    MIN_WORDS       = 25     (minimum meaningful chunk length)
    MAX_WORDS       = 1200   (maximum chunk length before splitting)
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional


# -- Thresholds ----------------------------------------------------------------

MIN_PRINTABLE    = 0.85
MIN_WORD_DENSITY = 0.60
MIN_ACADEMIC     = 2
MIN_QUALITY      = 0.70
MIN_WORDS        = 25
MAX_WORDS        = 1200

# Academic indicator vocabulary (domain-agnostic engineering terms)
ACADEMIC_VOCAB: set[str] = {
    # Generic academic
    "define", "explain", "describe", "analyze", "apply", "evaluate",
    "design", "implement", "derive", "prove", "state", "list", "compare",
    "illustrate", "calculate", "solve", "determine", "find", "compute",
    "theorem", "principle", "concept", "algorithm", "system", "method",
    "process", "function", "equation", "formula", "theorem", "proof",
    "module", "chapter", "unit", "example", "solution", "problem",
    "given", "note", "remark", "definition", "property", "lemma",

    # Engineering
    "circuit", "signal", "network", "frequency", "voltage", "current",
    "resistance", "capacitance", "inductance", "impedance", "power",
    "efficiency", "temperature", "pressure", "velocity", "force",
    "stress", "strain", "matrix", "vector", "integral", "derivative",
    "probability", "distribution", "entropy", "complexity", "traversal",
    "node", "edge", "graph", "tree", "stack", "queue", "array", "sort",
    "search", "heuristic", "optimization", "convergence", "stability",
    "transfer", "transform", "modulation", "bandwidth", "gain",
    "satellite", "orbit", "antenna", "transponder", "link", "propagation",
}

# Patterns that indicate corrupted content
CORRUPTION_PATTERNS: list[re.Pattern] = [
    re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]'),   # control chars
    re.compile(r'(?:[^\x00-\x7f]{3,}\s*){3,}'),                # dense non-ASCII runs
    re.compile(r'\$[A-Za-z0-9+/]{8,}={0,2}'),                  # base64 artifacts
    re.compile(r'(?:obj\s+\d+\s+\d+|stream|endstream|xref)'),  # PDF internals
    re.compile(r'\\u[0-9a-fA-F]{4}(?:\\u[0-9a-fA-F]{4}){3,}'), # escaped unicode runs
]


# -- Result Types --------------------------------------------------------------

@dataclass
class ChunkScore:
    text:             str
    quality_score:    float
    printable_ratio:  float
    word_density:     float
    academic_count:   int
    word_count:       int
    passed:           bool
    rejection_reason: str = ""
    clean_text:       str = ""

    def __str__(self) -> str:
        status = "PASS" if self.passed else f"FAIL({self.rejection_reason})"
        return (f"ChunkScore[{status} q={self.quality_score:.2f} "
                f"p={self.printable_ratio:.2f} w={self.word_count}]")


@dataclass
class ValidationReport:
    total:    int = 0
    passed:   int = 0
    rejected: int = 0
    scores:   list[ChunkScore] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passed / max(1, self.total)

    @property
    def valid_chunks(self) -> list[str]:
        return [s.clean_text or s.text for s in self.scores if s.passed]

    def summary(self) -> str:
        return (f"Validation: {self.passed}/{self.total} passed "
                f"({self.pass_rate:.0%}) — "
                f"{self.rejected} rejected")


# -- Validator -----------------------------------------------------------------

class ContentValidator:
    """
    Stage 2 of the AION pipeline.
    Scores every extracted chunk and rejects corrupted content.
    """

    def __init__(
        self,
        min_quality:     float = MIN_QUALITY,
        min_printable:   float = MIN_PRINTABLE,
        min_word_density: float = MIN_WORD_DENSITY,
        min_academic:    int   = MIN_ACADEMIC,
        min_words:       int   = MIN_WORDS,
    ):
        self.min_quality      = min_quality
        self.min_printable    = min_printable
        self.min_word_density = min_word_density
        self.min_academic     = min_academic
        self.min_words        = min_words

    # -- Public API ------------------------------------------------------------

    def validate_chunk(self, text: str) -> ChunkScore:
        """Score a single chunk. Returns ChunkScore with pass/fail decision."""

        if not text or not text.strip():
            return ChunkScore(
                text=text, quality_score=0.0,
                printable_ratio=0.0, word_density=0.0,
                academic_count=0, word_count=0,
                passed=False, rejection_reason="EMPTY",
            )

        text = text.strip()

        # -- Check 1: Corruption pattern match (fast reject) -------------------
        for pattern in CORRUPTION_PATTERNS:
            if pattern.search(text):
                return ChunkScore(
                    text=text, quality_score=0.0,
                    printable_ratio=0.0, word_density=0.0,
                    academic_count=0, word_count=len(text.split()),
                    passed=False, rejection_reason="CORRUPTION_PATTERN",
                )

        # -- Check 2: Printable ratio ------------------------------------------
        printable_ratio = self._printable_ratio(text)
        if printable_ratio < self.min_printable:
            return ChunkScore(
                text=text, quality_score=printable_ratio,
                printable_ratio=printable_ratio, word_density=0.0,
                academic_count=0, word_count=len(text.split()),
                passed=False,
                rejection_reason=f"LOW_PRINTABLE({printable_ratio:.0%})",
            )

        # -- Check 3: Word count -----------------------------------------------
        words      = text.split()
        word_count = len(words)
        if word_count < self.min_words:
            return ChunkScore(
                text=text, quality_score=0.3,
                printable_ratio=printable_ratio,
                word_density=0.0, academic_count=0,
                word_count=word_count,
                passed=False,
                rejection_reason=f"TOO_SHORT({word_count}<{self.min_words})",
            )

        # -- Check 4: Word density (real words vs symbol soup) -----------------
        word_density = self._word_density(words)
        if word_density < self.min_word_density:
            return ChunkScore(
                text=text, quality_score=word_density,
                printable_ratio=printable_ratio,
                word_density=word_density, academic_count=0,
                word_count=word_count,
                passed=False,
                rejection_reason=f"LOW_WORD_DENSITY({word_density:.0%})",
            )

        # -- Check 5: Academic vocabulary density ------------------------------
        text_lower    = text.lower()
        academic_count = sum(1 for w in ACADEMIC_VOCAB if w in text_lower)

        # -- Compute quality score ---------------------------------------------
        quality_score = self._compute_quality(
            printable_ratio, word_density, academic_count, word_count
        )

        passed = (
            quality_score      >= self.min_quality and
            academic_count     >= self.min_academic
        )

        reason = ""
        if not passed:
            if academic_count < self.min_academic:
                reason = f"LOW_ACADEMIC({academic_count}<{self.min_academic})"
            else:
                reason = f"LOW_QUALITY({quality_score:.2f}<{self.min_quality})"

        clean = self._clean(text) if passed else ""

        return ChunkScore(
            text=text,
            quality_score=quality_score,
            printable_ratio=printable_ratio,
            word_density=word_density,
            academic_count=academic_count,
            word_count=word_count,
            passed=passed,
            rejection_reason=reason,
            clean_text=clean,
        )

    def validate_batch(self, chunks: list[str]) -> ValidationReport:
        """Validate a list of chunks. Returns report with valid_chunks."""
        report = ValidationReport(total=len(chunks))
        for chunk in chunks:
            score = self.validate_chunk(chunk)
            report.scores.append(score)
            if score.passed:
                report.passed += 1
            else:
                report.rejected += 1
        print(f"[VALIDATOR] Evaluated {len(chunks)} chunks -> Passed: {report.passed}, Rejected: {report.rejected}")
        return report

    # -- Scoring Algorithms ----------------------------------------------------

    def _printable_ratio(self, text: str) -> float:
        """
        Ratio of printable characters to total characters.
        Printable = Unicode letter, digit, punctuation, space.
        """
        if not text:
            return 0.0
        printable = sum(
            1 for ch in text
            if unicodedata.category(ch)[0] in ("L", "N", "P", "Z")
        )
        return printable / len(text)

    def _word_density(self, words: list[str]) -> float:
        """
        Ratio of 'real' words (alphabetic, 2+ chars) to total tokens.
        Filters out symbol sequences, numbers alone, and single chars.
        """
        if not words:
            return 0.0
        real = sum(
            1 for w in words
            if len(w) >= 2 and re.search(r'[a-zA-Z]{2,}', w)
        )
        return real / len(words)

    def _compute_quality(
        self,
        printable:  float,
        word_den:   float,
        academic:   int,
        word_count: int,
    ) -> float:
        """
        Weighted quality score in [0, 1].

        Weights:
            printable_ratio   40%
            word_density      35%
            academic_density  25%
        """
        academic_norm = min(1.0, academic / 10)   # cap at 10 words = full score

        score = (
            0.40 * printable +
            0.35 * word_den  +
            0.25 * academic_norm
        )

        # Length bonus: chunks in [50, 400] words are ideal
        if 50 <= word_count <= 400:
            score = min(1.0, score + 0.05)

        return round(score, 4)

    def _clean(self, text: str) -> str:
        """
        Light cleaning of valid text.
        Removes obvious PDF artifacts without destroying meaning.
        """
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', text)
        # Remove PDF object markers
        text = re.sub(r'\b\d+ \d+ obj\b.*?endobj', ' ', text,
                      flags=re.DOTALL | re.IGNORECASE)
        # Normalize whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()


# -- Module-level singleton ----------------------------------------------------

_validator = ContentValidator()

def validate_chunk(text: str) -> ChunkScore:
    return _validator.validate_chunk(text)

def validate_batch(chunks: list[str]) -> ValidationReport:
    return _validator.validate_batch(chunks)

def _printable_ratio(text: str) -> float:
    return _validator._printable_ratio(text)
