# AION-Trainer/acb/concept_discoverer.py
"""
Concept Discoverer — the stage that converts raw document text into
candidate Concept objects. This is NOT a chunker. It operates at
the level of named concepts, not arbitrary text windows.

It produces candidates — ConceptMerger decides whether each candidate
merges into an existing concept or becomes a new one.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

from acb.concept import Concept, ConceptSource, BloomProgression, ConceptScope

logger = logging.getLogger("aion.acb.discoverer")


# Patterns that reliably indicate a concept being introduced / defined
DEFINITION_PATTERNS = [
    re.compile(r"^(.{3,60}?)\s+(?:is|are)\s+(?:defined|a|an|the)\s", re.IGNORECASE),
    re.compile(r"^Definition\s*[:\-]\s*(.+)", re.IGNORECASE),
    re.compile(r"^(.{3,60}?)\s*[:]\s*(?:it|this|the)\s+(?:is|refers)", re.IGNORECASE),
]

APPLICATION_PATTERNS = re.compile(r"application[s]?\s+(?:of|include|:)", re.IGNORECASE)
ALGORITHM_PATTERNS = re.compile(r"\b(?:algorithm|procedure|pseudocode|steps?)\b", re.IGNORECASE)
DIAGRAM_PATTERNS = re.compile(r"\b(?:figure|diagram|fig\.|flowchart|graph|tree|chart)\b", re.IGNORECASE)
BLOOM_VERB_PATTERNS = {
    "L1": re.compile(r"\b(?:define|list|state|name|identify|recall)\b", re.IGNORECASE),
    "L2": re.compile(r"\b(?:explain|describe|discuss|summarize|interpret)\b", re.IGNORECASE),
    "L3": re.compile(r"\b(?:apply|implement|solve|demonstrate|illustrate)\b", re.IGNORECASE),
    "L4": re.compile(r"\b(?:analyze|compare|contrast|differentiate|trace)\b", re.IGNORECASE),
    "L5": re.compile(r"\b(?:evaluate|justify|assess|critique)\b", re.IGNORECASE),
    "L6": re.compile(r"\b(?:design|create|develop|propose|formulate)\b", re.IGNORECASE),
}


@dataclass
class ConceptCandidate:
    """
    A concept candidate as detected from one document.
    May match an existing concept (handled by ConceptMerger) or be new.
    """
    name: str
    source_id: str
    source_type: str
    source_location: str              # chapter/page reference
    definition: str = ""
    explanation: str = ""
    key_points: List[str] = field(default_factory=list)
    algorithms: List[str] = field(default_factory=list)
    applications: List[str] = field(default_factory=list)
    requires_diagram: bool = False
    bloom_signals: List[str] = field(default_factory=list)   # detected Bloom levels
    raw_excerpt: str = ""
    confidence: float = 0.7


class ConceptDiscoverer:
    """
    Discovers concept candidates from structured text blocks.

    Input:  blocks of text labelled with their position in the document
    Output: list of ConceptCandidate objects, one per detected concept

    The discoverer never makes module-assignment decisions —
    that is the ConfidenceEngine's job.
    """

    def __init__(self, min_concept_name_length: int = 3):
        self.min_length = min_concept_name_length

    def discover_from_blocks(
        self,
        blocks: List[Dict[str, Any]],
        source_id: str,
        source_type: str,
    ) -> List[ConceptCandidate]:
        """
        blocks: list of dicts with keys: text, kind, page, location
        kind in: heading | text | algorithm | equation | table | image
        """
        candidates: List[ConceptCandidate] = []
        current_heading: str = ""
        buffer: List[Dict] = []

        for block in blocks:
            if block.get("kind") == "heading":
                if current_heading and buffer:
                    cand = self._build_candidate(
                        current_heading, buffer, source_id, source_type,
                        location=block.get("location", ""),
                    )
                    if cand:
                        candidates.append(cand)
                current_heading = block.get("text", "").strip()
                buffer = []
            else:
                buffer.append(block)

        if current_heading and buffer:
            cand = self._build_candidate(
                current_heading, buffer, source_id, source_type, location="end"
            )
            if cand:
                candidates.append(cand)

        logger.info(f"[Discoverer] Found {len(candidates)} concept candidates "
                    f"from source {source_id}")
        return candidates

    def discover_from_pyq_records(
        self, records: List[Dict[str, Any]], source_id: str
    ) -> List[ConceptCandidate]:
        """
        Extract concept names from previous question paper records.
        These don't build full definitions but contribute frequency
        and question history to existing concepts.
        """
        from server.pyq_extractor import classify_bloom
        candidates = []
        for record in records:
            text = record.get("text", "")
            name = self._extract_topic_from_question(text)
            if name and len(name) >= self.min_length:
                candidates.append(ConceptCandidate(
                    name=name,
                    source_id=source_id,
                    source_type="previous_paper",
                    source_location="",
                    bloom_signals=[record.get("bloom", classify_bloom(text))],
                    raw_excerpt=text,
                    confidence=0.6,
                ))
        return candidates

    def _build_candidate(
        self,
        heading: str,
        blocks: List[Dict],
        source_id: str,
        source_type: str,
        location: str = "",
    ) -> Optional[ConceptCandidate]:
        if len(heading) < self.min_length or len(heading) > 120:
            return None

        all_text = " ".join(b.get("text", "") for b in blocks)
        candidate = ConceptCandidate(
            name=heading,
            source_id=source_id,
            source_type=source_type,
            source_location=location,
            raw_excerpt=all_text[:400],
        )

        # Definition
        for pattern in DEFINITION_PATTERNS:
            m = pattern.search(all_text[:800])
            if m:
                candidate.definition = all_text[:300].strip()
                break

        # Explanation (first full sentence if no definition found)
        if not candidate.definition:
            sentences = re.split(r'(?<=[.!?])\s+', all_text)
            for s in sentences[:3]:
                if len(s.split()) >= 6:
                    candidate.explanation = s.strip()
                    break

        # Key points from bullets
        bullets = re.findall(r"(?:^|\n)\s*[-•*]\s*(.+)", all_text)
        candidate.key_points = [b.strip() for b in bullets[:8]]

        # Applications
        if APPLICATION_PATTERNS.search(all_text):
            after = APPLICATION_PATTERNS.split(all_text, maxsplit=1)[-1]
            candidate.applications = [
                a.strip(" .") for a in re.split(r"[,;]", after[:300]) if a.strip()
            ][:6]

        # Algorithms
        if ALGORITHM_PATTERNS.search(all_text):
            candidate.algorithms.append(heading)

        # Diagrams
        candidate.requires_diagram = bool(DIAGRAM_PATTERNS.search(all_text))

        # Bloom signals
        for level, pattern in BLOOM_VERB_PATTERNS.items():
            if pattern.search(all_text):
                candidate.bloom_signals.append(level)

        return candidate

    def _extract_topic_from_question(self, text: str) -> str:
        """Heuristic: skip the verb, take the next meaningful phrase."""
        skip_words = {
            "explain", "define", "describe", "compare", "differentiate",
            "trace", "apply", "design", "solve", "list", "discuss",
            "state", "illustrate", "with", "and", "the", "a", "an",
            "of", "to", "for", "is", "using", "suitable", "example",
        }
        words = re.sub(r"[^\w\s]", " ", text.lower()).split()
        phrase = []
        for w in words[1:]:
            if w in skip_words and phrase:
                break
            if w not in skip_words and len(w) > 2:
                phrase.append(w)
            if len(phrase) >= 4:
                break
        return " ".join(phrase).title()
