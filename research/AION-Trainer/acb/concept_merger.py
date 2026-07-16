# AION-Trainer/acb/concept_merger.py
"""
Concept Merger — prevents duplicate concepts from ever existing in the
store. When a new candidate arrives, it either:
    1. Creates a new concept
    2. Merges into an existing concept (enriching it with new evidence)

Matching uses three-tier similarity:
    Tier 1: exact normalised name match
    Tier 2: alias / known-variant match
    Tier 3: token overlap similarity (Jaccard) above threshold

Tier 3 is the only one with a tunable threshold. Threshold too low =
false merges (BFS merges with DFS). Threshold too high = duplicates
(A* and A-Star stored separately).
"""

import re
import logging
from typing import Optional, List, Dict, Tuple

from acb.concept import Concept, ConceptSource, BloomProgression, ConceptStore
from acb.concept_discoverer import ConceptCandidate
from acb.source_registry import SourceRegistry

logger = logging.getLogger("aion.acb.merger")

# Known aliases for common academic concepts.
# The merger checks these before falling back to Jaccard similarity.
KNOWN_ALIASES: Dict[str, List[str]] = {
    "a* search": ["a star", "a-star", "a* algorithm", "astar", "a star search"],
    "breadth first search": ["bfs", "breadth-first search", "breadth first"],
    "depth first search": ["dfs", "depth-first search", "depth first"],
    "uniform cost search": ["ucs", "dijkstra search"],
    "artificial neural network": ["ann", "neural network", "neural net"],
    "binary search tree": ["bst", "binary search"],
    "binary tree": ["bt"],
    "avl tree": ["avl", "adelson velsky landis"],
    "machine learning": ["ml"],
    "deep learning": ["dl"],
    "natural language processing": ["nlp"],
    "convolutional neural network": ["cnn"],
    "recurrent neural network": ["rnn"],
    "long short term memory": ["lstm"],
}

# Build reverse map: alias -> canonical
ALIAS_TO_CANONICAL: Dict[str, str] = {}
for canonical, aliases in KNOWN_ALIASES.items():
    for alias in aliases:
        ALIAS_TO_CANONICAL[alias] = canonical


class ConceptMerger:
    def __init__(
        self,
        concept_store: ConceptStore,
        source_registry: SourceRegistry,
        similarity_threshold: float = 0.55,
    ):
        self.store = concept_store
        self.registry = source_registry
        self.threshold = similarity_threshold
        self._merge_log: List[Dict] = []

    def merge_candidates(self, candidates: List[ConceptCandidate]) -> Dict[str, int]:
        """
        Process all candidates. Returns stats dict.
        Each candidate either updates an existing concept or creates a new one.
        """
        stats = {"created": 0, "merged": 0, "skipped": 0}

        for cand in candidates:
            result = self._merge_one(cand)
            stats[result] += 1

        logger.info(
            f"[Merger] {stats['created']} created, {stats['merged']} merged, "
            f"{stats['skipped']} skipped"
        )
        return stats

    def _merge_one(self, candidate: ConceptCandidate) -> str:
        existing = self._find_match(candidate.name)

        if existing is None:
            self._create_from_candidate(candidate)
            return "created"

        self._enrich_concept(existing, candidate)
        return "merged"

    def _find_match(self, name: str) -> Optional[Concept]:
        """Three-tier matching."""
        # Tier 1: direct name lookup (normalised)
        existing = self.store.find_by_name(name)
        if existing:
            return existing

        # Tier 2: resolve via known aliases
        normalised = name.lower().strip()
        canonical = ALIAS_TO_CANONICAL.get(normalised)
        if canonical:
            existing = self.store.find_by_name(canonical)
            if existing:
                existing.aliases.append(name)  # record variant for future lookups
                return existing

        # Tier 3: Jaccard token similarity
        return self._find_by_jaccard(name)

    def _find_by_jaccard(self, name: str) -> Optional[Concept]:
        name_tokens = set(self._tokenise(name))
        if not name_tokens:
            return None

        best_concept, best_score = None, 0.0
        for concept in self.store.all_concepts():
            candidate_tokens = set(self._tokenise(concept.name))
            if not candidate_tokens:
                continue
            intersection = len(name_tokens & candidate_tokens)
            union = len(name_tokens | candidate_tokens)
            score = intersection / union if union > 0 else 0.0
            if score > best_score:
                best_score = score
                best_concept = concept

        if best_score >= self.threshold:
            return best_concept
        return None

    def _create_from_candidate(self, candidate: ConceptCandidate) -> Concept:
        concept = Concept(
            name=candidate.name,
            canonical_name=candidate.name,
            definition=candidate.definition,
            explanation=candidate.explanation,
            key_points=list(candidate.key_points),
            algorithms=list(candidate.algorithms),
            applications=list(candidate.applications),
            requires_diagram=candidate.requires_diagram,
            confidence=candidate.confidence,
        )

        self._set_bloom_from_signals(concept, candidate.bloom_signals)
        concept.sources.append(self._make_source(candidate))
        concept.keywords = self._extract_keywords(candidate.name, candidate.explanation)

        self.store.add(concept)
        return concept

    def _enrich_concept(self, concept: Concept, candidate: ConceptCandidate):
        """
        Merge new evidence into an existing concept.
        The higher-reliability source wins for the definition field.
        Everything else is additive.
        """
        source = self._make_source(candidate)
        new_reliability = self.registry.reliability(candidate.source_id)
        existing_reliability = max(
            (self.registry.reliability(s.source_id) for s in concept.sources), default=0.0
        )

        # Higher-reliability source wins definition
        if new_reliability > existing_reliability and candidate.definition:
            concept.definition = candidate.definition

        # Additive enrichment
        if candidate.explanation and len(candidate.explanation) > len(concept.explanation):
            concept.explanation = candidate.explanation

        for pt in candidate.key_points:
            if pt not in concept.key_points:
                concept.key_points.append(pt)

        for app in candidate.applications:
            if app not in concept.applications:
                concept.applications.append(app)

        for algo in candidate.algorithms:
            if algo not in concept.algorithms:
                concept.algorithms.append(algo)

        if candidate.requires_diagram:
            concept.requires_diagram = True

        self._set_bloom_from_signals(concept, candidate.bloom_signals)

        if source.source_id not in {s.source_id for s in concept.sources}:
            concept.sources.append(source)

        # Update frequency counters
        if candidate.source_type == "previous_paper":
            concept.previous_paper_frequency += 1
        elif candidate.source_type == "question_bank":
            concept.question_bank_frequency += 1
        elif candidate.source_type == "notes":
            concept.professor_notes_present = True

        concept.touch()

    def _make_source(self, candidate: ConceptCandidate) -> ConceptSource:
        return ConceptSource(
            source_id=candidate.source_id,
            source_type=candidate.source_type,
            location=candidate.source_location,
            excerpt=candidate.raw_excerpt[:200],
            reliability=self.registry.reliability(candidate.source_id),
        )

    def _set_bloom_from_signals(self, concept: Concept, signals: List[str]):
        for level in signals:
            if hasattr(concept.bloom_progression, level):
                setattr(concept.bloom_progression, level, True)

    @staticmethod
    def _tokenise(text: str) -> List[str]:
        return [
            w for w in re.sub(r"[^\w\s]", " ", text.lower()).split()
            if len(w) > 2
        ]

    @staticmethod
    def _extract_keywords(name: str, explanation: str) -> List[str]:
        combined = name + " " + explanation
        words = re.findall(r"\b[a-zA-Z]{4,}\b", combined.lower())
        stop = {
            "with", "this", "that", "from", "they", "their", "which",
            "when", "where", "what", "have", "been", "will", "also",
        }
        seen, keywords = set(), []
        for w in words:
            if w not in stop and w not in seen:
                seen.add(w)
                keywords.append(w)
            if len(keywords) == 12:
                break
        return keywords
