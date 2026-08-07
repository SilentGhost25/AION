"""
AION Content Validator
======================
Runs before question generation.
Rejects corrupted, noisy, or non-academic chunks.
Production-safe — no laptop-specific code.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class ValidationResult:
    valid:           bool
    noise_score:     float      # 0.0 = clean, 1.0 = completely corrupt
    confidence:      float      # 0.0 to 1.0
    rejection_reason: str = ""
    clean_text:      str  = ""


# Academic vocabulary that should appear in valid content
ACADEMIC_INDICATORS = {
    "define", "explain", "describe", "analyze", "apply",
    "compare", "evaluate", "design", "implement", "derive",
    "theorem", "principle", "concept", "algorithm", "system",
    "method", "process", "function", "equation", "formula",
    "module", "chapter", "unit", "example", "solution",
    "problem", "given", "find", "calculate", "prove",
    "state", "list", "discuss", "illustrate", "derive",
}

# Patterns that indicate corrupted content
CORRUPTION_PATTERNS = [
    r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]',  # control characters
    r'[^\x00-\x7f]{5,}',                           # long non-ASCII runs
    r'\$[a-zA-Z0-9]{6,}',                          # binary artifacts
    r'[^\w\s.,;:?!\-\(\)\/\[\]]{4,}',             # symbol runs
    r'(?:[^\w])\w{1,2}(?:[^\w])\w{1,2}(?:[^\w])', # fragmented chars
]

MIN_CONFIDENCE   = 0.60
MAX_NOISE        = 0.30
MIN_WORD_COUNT   = 30
MIN_ACADEMIC     = 2     # minimum academic indicator words


class ContentValidator:

    def validate(self, text: str) -> ValidationResult:
        if not text or not text.strip():
            return ValidationResult(
                valid=False, noise_score=1.0, confidence=0.0,
                rejection_reason="Empty content"
            )

        words      = text.split()
        word_count = len(words)

        if word_count < MIN_WORD_COUNT:
            return ValidationResult(
                valid=False, noise_score=0.5, confidence=0.3,
                rejection_reason=f"Too short: {word_count} words (min {MIN_WORD_COUNT})"
            )

        # Noise score
        noise_score = self._compute_noise(text, word_count)
        if noise_score > MAX_NOISE:
            return ValidationResult(
                valid=False,
                noise_score=noise_score,
                confidence=1.0 - noise_score,
                rejection_reason=f"Noise score {noise_score:.0%} exceeds threshold {MAX_NOISE:.0%}"
            )

        # Academic density
        text_lower      = text.lower()
        academic_count  = sum(1 for w in ACADEMIC_INDICATORS if w in text_lower)
        if academic_count < MIN_ACADEMIC:
            return ValidationResult(
                valid=False,
                noise_score=noise_score,
                confidence=0.4,
                rejection_reason=f"Low academic density: {academic_count} indicators found (min {MIN_ACADEMIC})"
            )

        # Clean the text
        clean = self._clean(text)
        confidence = min(1.0, 0.5 + academic_count * 0.05 + (1.0 - noise_score) * 0.4)

        return ValidationResult(
            valid       = True,
            noise_score = noise_score,
            confidence  = confidence,
            clean_text  = clean,
        )

    def validate_chunk_list(
        self,
        chunks: List[str],
        min_valid: int = 2,
    ) -> Tuple[List[str], List[str], float]:
        """
        Validate a list of chunks.
        Returns (valid_chunks, rejected_chunks, avg_confidence)
        """
        valid    = []
        rejected = []

        for chunk in chunks:
            result = self.validate(chunk)
            if result.valid:
                valid.append(result.clean_text or chunk)
            else:
                rejected.append(chunk)
                print(f"[VALIDATOR] Rejected chunk: {result.rejection_reason}")

        if len(valid) < min_valid:
            print(
                f"[VALIDATOR] Only {len(valid)}/{len(chunks)} chunks valid. "
                f"Generation may be unreliable."
            )

        avg_conf = sum(
            self.validate(c).confidence for c in valid
        ) / max(1, len(valid))

        return valid, rejected, avg_conf

    def _compute_noise(self, text: str, word_count: int) -> float:
        """Compute noise ratio 0.0 (clean) to 1.0 (corrupt)."""
        noise_chars = 0
        total_chars = len(text)

        for pattern in CORRUPTION_PATTERNS:
            for match in re.finditer(pattern, text):
                noise_chars += len(match.group())

        # Also count non-printable unicode
        for ch in text:
            cat = unicodedata.category(ch)
            if cat in ("Cc", "Cf", "Cs", "Co", "Cn"):
                noise_chars += 1

        return min(1.0, noise_chars / max(1, total_chars))

    def _clean(self, text: str) -> str:
        """Remove corruption artifacts from text."""
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', ' ', text)
        # Remove binary artifact patterns
        text = re.sub(r'\$[a-zA-Z0-9+/]{6,}', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


_validator = ContentValidator()

def validate_content(text: str) -> ValidationResult:
    return _validator.validate(text)

def validate_chunk(text: str) -> bool:
    return _validator.validate(text).valid

def clean_chunk(text: str) -> str:
    return _validator._clean(text)

def validate_chunks(chunks: List[str]) -> Tuple[List[str], List[str], float]:
    return _validator.validate_chunk_list(chunks)
