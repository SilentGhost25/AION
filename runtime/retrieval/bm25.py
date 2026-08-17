# runtime/retrieval/bm25.py
"""Lightweight, pure-Python BM25 index with strict module locking.

Each module gets its own physically separate index.  A query for
module M3 can NEVER access chunks from M1, M2, M4, or M5 because
the search space itself is restricted — not filtered after retrieval.

Usage:
    index = BM25Index(module_id="M3")
    index.add_document(doc_id="M3-chunk-7", text="...")
    index.build()
    results = index.query("operating system scheduling", top_k=4)
"""

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class RetrievedChunk:
    """A single retrieved evidence chunk with its BM25 score."""

    doc_id: str
    text: str
    score: float
    module_id: str


# ── Tokeniser ─────────────────────────────────────────────────────────

_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "can", "could", "must", "and", "but", "or",
    "nor", "not", "so", "yet", "both", "either", "neither", "each",
    "every", "all", "any", "few", "more", "most", "other", "some", "such",
    "no", "only", "own", "same", "than", "too", "very", "just", "because",
    "as", "until", "while", "of", "at", "by", "for", "with", "about",
    "against", "between", "through", "during", "before", "after", "above",
    "below", "to", "from", "up", "down", "in", "out", "on", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "what", "which", "who", "whom", "this", "that",
    "these", "those", "i", "me", "my", "myself", "we", "our", "ours",
    "you", "your", "yours", "he", "him", "his", "she", "her", "hers",
    "it", "its", "they", "them", "their", "theirs",
})

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenise(text: str) -> List[str]:
    """Lowercase tokenisation with stopword removal."""
    return [
        tok for tok in _TOKEN_RE.findall(text.lower())
        if tok not in _STOPWORDS and len(tok) > 1
    ]


# ── BM25 Index ────────────────────────────────────────────────────────

# BM25 tuning parameters
_K1 = 1.5
_B = 0.75


class BM25Index:
    """BM25 index for a single module's evidence chunks."""

    def __init__(self, module_id: str):
        self.module_id = module_id
        self._docs: Dict[str, str] = {}           # doc_id -> raw text
        self._doc_tokens: Dict[str, List[str]] = {}  # doc_id -> tokens
        self._df: Dict[str, int] = {}              # term -> document frequency
        self._avg_dl: float = 0.0
        self._n_docs: int = 0
        self._built: bool = False

    def add_document(self, doc_id: str, text: str) -> None:
        """Add a document to the index (does not trigger build)."""
        self._docs[doc_id] = text
        self._doc_tokens[doc_id] = _tokenise(text)
        self._built = False

    def build(self) -> None:
        """Build the inverted index statistics (document frequencies, avg length)."""
        self._df.clear()
        total_length = 0
        self._n_docs = len(self._doc_tokens)

        for tokens in self._doc_tokens.values():
            total_length += len(tokens)
            seen = set()
            for tok in tokens:
                if tok not in seen:
                    self._df[tok] = self._df.get(tok, 0) + 1
                    seen.add(tok)

        self._avg_dl = total_length / self._n_docs if self._n_docs else 1.0
        self._built = True

    def query(self, query_text: str, top_k: int = 4) -> List[RetrievedChunk]:
        """Retrieve the top-k chunks most relevant to the query."""
        if not self._built:
            self.build()

        query_tokens = _tokenise(query_text)
        if not query_tokens:
            return []

        scores: Dict[str, float] = {}
        for doc_id, doc_tokens in self._doc_tokens.items():
            dl = len(doc_tokens)
            score = 0.0
            # Build term-frequency map for this document
            tf_map: Dict[str, int] = {}
            for tok in doc_tokens:
                tf_map[tok] = tf_map.get(tok, 0) + 1

            for qt in query_tokens:
                if qt not in self._df:
                    continue
                df = self._df[qt]
                tf = tf_map.get(qt, 0)
                if tf == 0:
                    continue
                # IDF component
                idf = math.log(
                    (self._n_docs - df + 0.5) / (df + 0.5) + 1.0
                )
                # TF component with length normalisation
                tf_norm = (tf * (_K1 + 1)) / (
                    tf + _K1 * (1 - _B + _B * dl / self._avg_dl)
                )
                score += idf * tf_norm

            if score > 0:
                scores[doc_id] = score

        # Sort by score descending, take top-k
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            RetrievedChunk(
                doc_id=doc_id,
                text=self._docs[doc_id],
                score=score,
                module_id=self.module_id,
            )
            for doc_id, score in ranked
        ]

    # ── Serialisation ─────────────────────────────────────────────────

    def save(self, directory: Path) -> None:
        """Persist the index to disk."""
        directory.mkdir(parents=True, exist_ok=True)
        data = {
            "module_id": self.module_id,
            "docs": self._docs,
        }
        with open(directory / "index.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, directory: Path) -> "BM25Index":
        """Load a previously saved index from disk."""
        path = directory / "index.json"
        if not path.exists():
            raise FileNotFoundError(f"No BM25 index at {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        idx = cls(module_id=data["module_id"])
        for doc_id, text in data["docs"].items():
            idx.add_document(doc_id, text)
        idx.build()
        return idx
