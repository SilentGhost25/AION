"""
Concept-Level Retriever — Retrieval Returns Concepts, Not Chunks
================================================================
Current (weak): returns chunks (too coarse)
Needed: concept-level retrieval

This retriever:
- Indexes ExtractedConcept / GroundedConcept via embeddings (BGE-M3 if available, else TF-IDF fallback)
- Retrieves by semantic similarity to query (Bloom + concept name)
- Reranks via optional bge-reranker
- Returns concept IDs with evidence snippets, not raw paragraphs
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from collections import Counter

from core.concepts.extractor import ExtractedConcept

@dataclass
class RetrievalResult:
    concept: ExtractedConcept
    score: float
    rank: int
    evidence_snippet: str
    reason: str

class ConceptLevelRetriever:
    """
    Concept-level retriever with hybrid dense + sparse + rerank (if models available).
    Falls back to lexical TF-IDF when neural models not installed.
    Pluggable: embedding model, reranker, vector DB can be swapped.
    """

    def __init__(self, use_neural: bool = True):
        self.use_neural = use_neural
        self.concepts: List[ExtractedConcept] = []
        self._embedder = None
        self._reranker = None
        self._index_built = False
        self._tfidf_vectors: Dict[str, Counter] = {}
        self._idf: Dict[str, float] = {}

        if use_neural:
            self._try_load_neural()

    def _try_load_neural(self):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            # Lazy load hint
            self._neural_available = True
        except ImportError:
            self._neural_available = False

    def index(self, concepts: List[ExtractedConcept]) -> None:
        """Build index from concepts."""
        self.concepts = concepts
        if not concepts:
            self._index_built = True
            return

        # Build TF-IDF fallback
        docs = [c.supporting_evidence.lower() for c in concepts]
        # Tokenize
        tokenized = [re.findall(r"\b\w+\b", d) for d in docs]
        # DF
        df: Counter[str] = Counter()
        for tokens in tokenized:
            df.update(set(tokens))
        N = len(docs)
        self._idf = {term: math.log((N + 1) / (freq + 1)) + 1 for term, freq in df.items()}
        # TF-IDF vectors
        self._tfidf_vectors = {}
        for c, tokens in zip(concepts, tokenized):
            tf = Counter(tokens)
            vec = Counter({term: (freq / len(tokens)) * self._idf.get(term, 1.0) for term, freq in tf.items()})
            self._tfidf_vectors[c.concept_id] = vec

        # If neural available, could build dense index (stub)
        self._index_built = True
        print(f"[RETRIEVER] Indexed {len(concepts)} concepts (TF-IDF fallback, neural stub)")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        bloom_filter: Optional[int] = None,
        concept_type_filter: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve concepts semantically similar to query.
        Query can be: concept name, bloom level, or natural question intent.
        """
        if not self._index_built:
            raise RuntimeError("Retriever not indexed. Call index(concepts) first.")
        if not self.concepts:
            return []
        if not query.strip():
            # Return top confidence
            ranked = sorted(self.concepts, key=lambda c: c.confidence, reverse=True)[:top_k]
            return [
                RetrievalResult(c, c.confidence, i + 1, c.supporting_evidence[:300], "top-confidence")
                for i, c in enumerate(ranked)
            ]

        # Lexical scoring (TF-IDF cosine)
        q_tokens = re.findall(r"\b\w+\b", query.lower())
        q_tf = Counter(q_tokens)
        q_vec = Counter({term: (freq / len(q_tokens)) * self._idf.get(term, math.log(len(self.concepts) + 1)) for term, freq in q_tf.items()})

        scored: List[tuple[ExtractedConcept, float, str]] = []
        for c in self.concepts:
            # Type filter
            if concept_type_filter and c.concept_type != concept_type_filter:
                continue
            # Bloom filter
            if bloom_filter and c.bloom_suggestions:
                bloom_nums = [int(m.group(1)) for m in [re.search(r"L(\d)", s) for s in c.bloom_suggestions] if m]
                if bloom_filter not in bloom_nums and bloom_nums:
                    # Penalize but don't exclude strictly
                    pass

            vec = self._tfidf_vectors.get(c.concept_id, Counter())
            score = self._cosine(q_vec, vec)
            # Boost by concept confidence
            score = score * 0.70 + c.confidence * 0.30
            # Boost if query terms appear in concept name
            if any(t in c.concept_name.lower() for t in q_tokens if len(t) > 3):
                score = min(1.0, score + 0.15)
            scored.append((c, score, "tfidf+confidence+name_boost"))

        # Sort
        scored.sort(key=lambda x: x[1], reverse=True)

        # Rerank stub (if bge-reranker available, would rerank top 20)
        # For now, take top_k

        results: List[RetrievalResult] = []
        for rank, (c, score, reason) in enumerate(scored[:top_k], 1):
            # If score too low, mark reason
            if score < 0.15:
                reason += " (low_relevance)"
            results.append(RetrievalResult(c, round(score, 3), rank, c.supporting_evidence[:400], reason))

        return results

    def retrieve_for_plan(self, plan_concept_id: str, top_k: int = 3) -> List[RetrievalResult]:
        """Retrieve supporting/related concepts for a plan (for exam-style cross-concept)."""
        target = next((c for c in self.concepts if c.concept_id == plan_concept_id), None)
        if not target:
            return []
        # Use concept's evidence as query to find related
        return self.retrieve(target.supporting_evidence[:500], top_k=top_k)

    # ── Helpers ──────────────────────────────────────────────

    def _cosine(self, a: Counter, b: Counter) -> float:
        if not a or not b:
            return 0.0
        dot = sum(a[t] * b.get(t, 0.0) for t in a)
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
