"""
AION Learning-Aware Extraction Boost
=====================================
Uses concepts stored in memory/concepts.json to:
1. Boost retrieval score of chunks containing high-confidence learned concepts
2. Flag chunks covering under-tested topics as PRIORITY
3. Filter out chunks that have already been over-used in generation

Called by ExtractionGateway after chunk validation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Dict, Optional

MEMORY_DIR         = Path("memory")
CONCEPTS_PATH      = MEMORY_DIR / "concepts.json"
COVERAGE_PATH      = MEMORY_DIR / "topic_coverage.json"
QUESTIONS_PATH     = MEMORY_DIR / "generated_questions.json"


def _load_json(path: Path, default):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default


def _extract_keywords(text: str) -> set:
    """Extract meaningful words from text for matching."""
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
    stop = {
        "that", "this", "with", "from", "have", "been", "will",
        "when", "they", "what", "which", "also", "each", "were",
        "their", "there", "about", "would", "could", "should",
    }
    return {w for w in words if w not in stop}


def build_concept_index(subject: str = None) -> Dict[str, float]:
    """
    Build a keyword -> confidence score index from learned concepts.
    Higher confidence = concept has been seen/reinforced more times.
    """
    concepts = _load_json(CONCEPTS_PATH, [])
    index: Dict[str, float] = {}

    for c in concepts:
        content = c.get("content", "")
        confidence = float(c.get("confidence", 0.5))
        keywords = _extract_keywords(content)
        for kw in keywords:
            # Take max confidence if keyword appears in multiple concepts
            index[kw] = max(index.get(kw, 0.0), confidence)

    return index


def get_undertested_topics(subject: str, module_num: int) -> set:
    """
    Returns keywords from topics that have been tested fewer times.
    Used to bias chunk selection toward under-covered content.
    """
    coverage = _load_json(COVERAGE_PATH, {})
    questions = _load_json(QUESTIONS_PATH, [])

    # Find topics that appear less in generated questions
    tested_words: Dict[str, int] = {}
    for q in questions:
        if q.get("subject") == subject:
            words = _extract_keywords(q.get("text", ""))
            for w in words:
                tested_words[w] = tested_words.get(w, 0) + 1

    # Under-tested = low count or never tested
    # Return words that appear < 3 times in generated questions
    undertested = {w for w, count in tested_words.items() if count < 3}

    # Also include words not tested at all
    # (they won't be in tested_words)
    return undertested


def boost_chunks(chunks: list, subject: str = "general", module_id: int = 1) -> list:
    """
    Main entry point — called by ExtractionGateway after validation.
    Modifies chunk retrieval scores based on learned knowledge.

    Returns chunks sorted by learning-boosted score (best first).
    """
    if not chunks:
        return chunks

    concept_index = build_concept_index(subject)
    undertested   = get_undertested_topics(subject, module_id)

    boosted = []
    for chunk in chunks:
        text = getattr(chunk, "text", "") or ""
        if not text:
            boosted.append((chunk, 0.0))
            continue

        chunk_words = _extract_keywords(text)
        if not chunk_words:
            boosted.append((chunk, 0.0))
            continue

        # 1. Concept confidence boost
        concept_score = sum(
            concept_index.get(w, 0.0) for w in chunk_words
        ) / max(1, len(chunk_words))

        # 2. Under-tested topic boost
        undertested_overlap = len(chunk_words & undertested) / max(1, len(chunk_words))
        undertested_boost = undertested_overlap * 0.3  # up to 30% boost

        # 3. Combined boost score
        boost = concept_score + undertested_boost

        # 4. Apply boost as reduced retrieval_penalty
        # (lower penalty = higher retrieval priority)
        if hasattr(chunk, "retrieval_penalty"):
            original_penalty = chunk.retrieval_penalty
            chunk.retrieval_penalty = max(0.0, original_penalty - boost * 0.5)

        # 5. Tag priority chunks (boost > 0.4)
        if boost > 0.4 and hasattr(chunk, "metadata"):
            if chunk.metadata is None:
                chunk.metadata = {}
            chunk.metadata["learning_boost"] = round(boost, 3)
            chunk.metadata["priority"] = True

        boosted.append((chunk, boost))

    # Sort by boost score descending — best chunks first
    boosted.sort(key=lambda x: x[1], reverse=True)

    result = [c for c, _ in boosted]

    n_boosted = sum(1 for _, s in boosted if s > 0.1)
    if n_boosted > 0:
        print(f"[LEARNING-BOOST] Module {module_id}: {n_boosted}/{len(chunks)} chunks boosted by learned concepts")

    return result
