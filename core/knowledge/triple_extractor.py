"""
Deterministic SPO triple extraction from academic text.

Rules cover the most common academic relations:
    - is_a / are
    - consists_of / is_part_of
    - includes / contains
    - requires / needs
    - enables / allows
    - used_for
    - has / have
    - refers_to / describes

No LLM. No network. Deterministic for a given input string.

This is a "good enough" extractor: noisy, but generalizes across subjects
without domain-specific tuning. Phase 3 can swap in an LLM-assisted
extractor behind the same interface if quality demands it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class RawTriple:
    """Triple before provenance is attached by the builder."""
    subject: str
    predicate: str
    object: str
    confidence: float


# -----------------------------------------------------------------------------
# Pattern definitions
# -----------------------------------------------------------------------------
# (regex, normalized_predicate, base_confidence)
#
# Patterns are tried in order; the first match wins. Order matters:
# more specific patterns (e.g. "is used for") must come before more
# general ones (e.g. "is a") because "is used for" would otherwise be
# partially matched by "is".
# -----------------------------------------------------------------------------

PATTERNS: List[Tuple[re.Pattern, str, float]] = [
    # Compound forms first (higher specificity)
    (re.compile(r"^(.{2,80}?)\s+is\s+used\s+for\s+(.{2,80}?)$", re.IGNORECASE), "used_for", 0.85),
    (re.compile(r"^(.{2,80}?)\s+is\s+used\s+to\s+(.{2,80}?)$", re.IGNORECASE), "used_for", 0.80),
    (re.compile(r"^(.{2,80}?)\s+consists\s+of\s+(.{2,80}?)$", re.IGNORECASE), "consists_of", 0.85),
    (re.compile(r"^(.{2,80}?)\s+is\s+part\s+of\s+(.{2,80}?)$", re.IGNORECASE), "part_of", 0.85),
    (re.compile(r"^(.{2,80}?)\s+refers\s+to\s+(.{2,80}?)$", re.IGNORECASE), "refers_to", 0.80),

    # Dependency / enablement
    (re.compile(r"^(.{2,80}?)\s+requires\s+(.{2,80}?)$", re.IGNORECASE), "requires", 0.80),
    (re.compile(r"^(.{2,80}?)\s+needs\s+(.{2,80}?)$", re.IGNORECASE), "requires", 0.70),
    (re.compile(r"^(.{2,80}?)\s+enables\s+(.{2,80}?)$", re.IGNORECASE), "enables", 0.80),
    (re.compile(r"^(.{2,80}?)\s+allows\s+(.{2,80}?)$", re.IGNORECASE), "enables", 0.75),

    # Composition
    (re.compile(r"^(.{2,80}?)\s+includes\s+(.{2,80}?)$", re.IGNORECASE), "includes", 0.75),
    (re.compile(r"^(.{2,80}?)\s+contains\s+(.{2,80}?)$", re.IGNORECASE), "contains", 0.75),

    # Attribute
    (re.compile(r"^(.{2,80}?)\s+has\s+(.{2,80}?)$", re.IGNORECASE), "has", 0.65),
    (re.compile(r"^(.{2,80}?)\s+have\s+(.{2,80}?)$", re.IGNORECASE), "has", 0.65),

    # Description
    (re.compile(r"^(.{2,80}?)\s+describes\s+(.{2,80}?)$", re.IGNORECASE), "refers_to", 0.70),

    # Taxonomy (most general — last)
    (re.compile(r"^(.{2,80}?)\s+is\s+(?:a|an)\s+(.{2,80}?)$", re.IGNORECASE), "is_a", 0.85),
    (re.compile(r"^(.{2,80}?)\s+are\s+(.{2,80}?)$", re.IGNORECASE), "is_a", 0.70),
]


# Sentence splitting: split on terminal punctuation followed by whitespace
# and an uppercase letter or end of string.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$")

# Articles to strip from the front of extracted subjects/objects
_LEADING_ARTICLE = re.compile(r"^(?:a|an|the)\s+", re.IGNORECASE)

# Trailing punctuation to strip
_TRAILING_PUNCT = re.compile(r"[\s.,;:!?]+$")

# Word-count bounds for valid subjects/objects
_MIN_WORDS = 1
_MAX_WORDS = 6

_CLAUSE_STARTERS = {
    "it", "this", "that", "these", "those", "there", "which", "what", "where",
    "when", "why", "how", "who", "whom", "whose", "we", "you", "they", "he",
    "she", "but", "and", "or", "because", "although", "since", "while", "if",
    "however", "therefore", "thus", "moreover", "furthermore",
}


class RuleBasedTripleExtractor:
    """
    Extracts SPO triples using ordered regex patterns.

    Parameters
    ----------
    min_confidence : float
        Triples below this confidence are discarded. Default 0.55.
    """

    def __init__(self, min_confidence: float = 0.55) -> None:
        if not (0.0 <= min_confidence <= 1.0):
            raise ValueError("min_confidence must be in [0.0, 1.0]")
        self._min_confidence = min_confidence

    def extract(self, text: str) -> List[RawTriple]:
        if not text or not text.strip():
            return []

        results: List[RawTriple] = []
        seen: set[Tuple[str, str, str]] = set()

        for sentence in self._split_sentences(text):
            triple = self._extract_from_sentence(sentence)
            if triple is None:
                continue
            key = (triple.subject.lower(), triple.predicate, triple.object.lower())
            if key in seen:
                continue
            seen.add(key)
            results.append(triple)

        return results

    # ------------------------------------------------------------- Internals

    def _split_sentences(self, text: str) -> List[str]:
        # Normalize whitespace first
        text = re.sub(r"\s+", " ", text.strip())
        parts = _SENTENCE_SPLIT.split(text)
        return [p.strip() for p in parts if p.strip()]

    def _extract_from_sentence(self, sentence: str) -> RawTriple | None:
        # Skip questions
        if sentence.rstrip().endswith("?"):
            return None

        # Skip very short or very long sentences
        words = sentence.split()
        if len(words) < 4 or len(words) > 60:
            return None

        # Try each pattern in order
        for pattern, predicate, base_conf in PATTERNS:
            m = pattern.match(sentence)
            if not m:
                continue

            subject = self._clean(m.group(1))
            obj = self._clean(m.group(2))

            if not self._is_valid(subject) or not self._is_valid(obj):
                continue

            confidence = self._score(base_conf, subject, obj)
            if confidence < self._min_confidence:
                continue

            return RawTriple(
                subject=subject,
                predicate=predicate,
                object=obj,
                confidence=confidence,
            )

        return None

    def _clean(self, phrase: str) -> str:
        phrase = phrase.strip()
        phrase = _LEADING_ARTICLE.sub("", phrase)
        phrase = _TRAILING_PUNCT.sub("", phrase)
        phrase = re.sub(r"\s+", " ", phrase)
        return phrase.strip()

    def _is_valid(self, phrase: str) -> bool:
        if not phrase:
            return False
        words = phrase.split()
        if len(words) < _MIN_WORDS or len(words) > _MAX_WORDS:
            return False
        # Reject phrases that are only punctuation or digits
        if not any(c.isalpha() for c in phrase):
            return False
        # Reject phrases starting with pronouns, conjunctions, or clause markers
        first_word = words[0].lower().rstrip(".,:;!?")
        if first_word in _CLAUSE_STARTERS:
            return False
        # Reject phrases ending with conjunctions or prepositions
        last_word = words[-1].lower().rstrip(".,:;!?")
        if last_word in _CLAUSE_STARTERS or last_word in {"in", "on", "at", "for", "with", "by", "from", "as", "to", "of", "and", "or", "but"}:
            return False
        return True

    def _score(self, base_conf: float, subject: str, obj: str) -> float:
        score = base_conf

        # Bonus: both sides look like proper nouns or technical terms
        if subject[0].isupper() and obj[0].isupper():
            score += 0.05

        # Penalty: very short subject or object
        if len(subject.split()) == 1:
            score -= 0.05
        if len(obj.split()) == 1:
            score -= 0.05

        # Penalty: subject or object too long (unlikely to be a concept)
        if len(subject.split()) > 8 or len(obj.split()) > 8:
            score -= 0.10

        return max(0.0, min(1.0, score))
