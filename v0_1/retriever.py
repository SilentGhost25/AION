"""
AION Grounded Retriever — Stage 6
===================================
Retrieves ONLY validated, semantically relevant chunks.
Enforces the hard ceiling of MAX_CHUNKS = 3.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from .validator import ACADEMIC_VOCAB

MAX_CHUNKS = 3


@dataclass
class RetrievedChunk:
    text:      str
    score:     float
    meta:      dict = field(default_factory=dict)

    def __lt__(self, other: "RetrievedChunk") -> bool:
        return self.score < other.score


class GroundedRetriever:
    """
    Stage 6: Retrieve the top-K most relevant validated chunks.
    """

    def __init__(
        self,
        max_chunks:    int = MAX_CHUNKS,
        embed_fn:      Optional[Callable[[str], list[float]]] = None,
    ):
        self.max_chunks = max_chunks
        self.embed_fn   = embed_fn

    def retrieve(
        self,
        query:       str,
        chunks:      list[str],
        chunk_metas: Optional[list[dict]] = None,
        module_id:   Optional[str] = None,
    ) -> list[RetrievedChunk]:
        """
        Score and rank chunks against the query.
        Returns top-K RetrievedChunk objects.
        Enforces strict module lock if module_id is provided.
        """
        if not chunks:
            return []

        metas = chunk_metas or [{} for _ in chunks]

        scored: list[RetrievedChunk] = []
        for chunk, meta in zip(chunks, metas):
            # Strict module lock: reject chunks explicitly belonging to another module
            if module_id:
                c_mod = str(meta.get("module_id") or meta.get("module_num") or meta.get("module") or "").strip()
                t_mod = str(module_id).strip()
                if c_mod and t_mod and c_mod != t_mod:
                    print(f"[RETRIEVER] Rejected cross-module chunk (chunk module '{c_mod}' != target '{t_mod}')", flush=True)
                    continue

            score = self._score(chunk, query, meta, module_id)
            scored.append(RetrievedChunk(text=chunk, score=score, meta=meta))

        scored.sort(key=lambda c: c.score, reverse=True)
        result = scored[:self.max_chunks]

        for i, c in enumerate(result):
            print(f"[RETRIEVER] Chunk {i+1}: score={c.score:.3f} | {c.text[:500].strip()!r}")

        return result

    def retrieve_texts(
        self,
        query:     str,
        chunks:    list[str],
        metas:     Optional[list[dict]] = None,
        module_id: Optional[str] = None,
    ) -> list[str]:
        """Convenience wrapper — returns plain text list."""
        return [c.text for c in self.retrieve(query, chunks, metas, module_id)]

    def _score(
        self,
        chunk:     str,
        query:     str,
        meta:      dict,
        module_id: Optional[str],
    ) -> float:
        if self.embed_fn:
            try:
                sim = self._cosine_sim(
                    self.embed_fn(query),
                    self.embed_fn(chunk[:512])
                )
                lex = self._lexical_score(chunk, query)
                return 0.60 * sim + 0.40 * lex
            except Exception:
                pass

        base = self._lexical_score(chunk, query)

        if module_id:
            c_mod = str(meta.get("module_id") or meta.get("module_num") or meta.get("module") or "").strip()
            t_mod = str(module_id).strip()
            if c_mod == t_mod:
                base = min(1.0, base + 0.15)
            elif c_mod and c_mod != t_mod:
                base = 0.0

        return base

    def _lexical_score(self, chunk: str, query: str) -> float:
        chunk_lower = chunk.lower()
        query_lower = query.lower()

        q_terms = set(re.findall(r'\b[a-zA-Z]{3,}\b', query_lower))
        c_terms = set(re.findall(r'\b[a-zA-Z]{3,}\b', chunk_lower))
        if q_terms:
            overlap = len(q_terms & c_terms) / len(q_terms)
        else:
            overlap = 0.0

        acad = sum(1 for w in ACADEMIC_VOCAB if w in chunk_lower)
        acad_score = min(1.0, acad / 8)

        wc = len(chunk.split())
        if 80 <= wc <= 400:
            length_score = 1.0
        elif wc < 80:
            length_score = wc / 80
        else:
            length_score = max(0.5, 400 / wc)

        return round(
            0.50 * overlap +
            0.30 * acad_score +
            0.20 * length_score,
            4
        )

    def _cosine_sim(self, a: list[float], b: list[float]) -> float:
        dot  = sum(x*y for x, y in zip(a, b))
        na   = math.sqrt(sum(x*x for x in a))
        nb   = math.sqrt(sum(x*x for x in b))
        return dot / (na * nb + 1e-9)


_retriever = GroundedRetriever()

def retrieve(
    query:     str,
    chunks:    list[str],
    metas:     Optional[list[dict]] = None,
    module_id: Optional[str] = None,
) -> list[str]:
    return _retriever.retrieve_texts(query, chunks, metas, module_id)

