"""
AION Semantic Cache Layer.

Task 3 adds async Krites-style promotion. When a query misses the static
tier but lands in a borderline similarity band, an LLM judge is asked
whether the static response is acceptable. Approval promotes the response
into the dynamic tier for future hits.

The async path is opt-in: pass `judge` and `executor` to enable it.
"""

from .semantic_cache import (
    SemanticCache,
    CachedEntry,
    CacheStats,
    EmbedFn,
)
from .krites_judge import KritesJudge, JudgeFn

__all__ = [
    "SemanticCache",
    "CachedEntry",
    "CacheStats",
    "EmbedFn",
    "KritesJudge",
    "JudgeFn",
]
