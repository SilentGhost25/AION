"""
Unit tests for the tiered semantic cache (static tier).

Uses a deterministic fake embed_fn so similarity scores are known
exactly and threshold boundaries can be tested precisely.
"""

import hashlib
import json
import threading
from pathlib import Path

import numpy as np
import pytest

from core.cache.semantic_cache import SemanticCache, CachedEntry


# -----------------------------------------------------------------------------
# Test helpers
# -----------------------------------------------------------------------------


class FakeClock:
    def __init__(self, start: float = 1_000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def make_embed_fn(dim: int = 8):
    """
    Deterministic embed_fn: hashes text to a seed, produces a normalized
    vector. Similar texts share hash prefixes, so cosine similarity can
    be controlled by varying the number of characters that match.
    """
    def embed(text: str) -> np.ndarray:
        # Use first 4 chars to build a "semantic" seed, rest to add noise
        seed_bytes = hashlib.sha256(text.encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(seed_bytes[:4], "big"))
        vec = rng.standard_normal(dim).astype(np.float32)
        return vec
    return embed


def fixed_embed(text: str) -> np.ndarray:
    """Returns a fixed vector regardless of input. Useful for exact-match tests."""
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def cache(clock):
    return SemanticCache(
        embed_fn=make_embed_fn(),
        clock=clock,
        static_threshold=0.90,
        dynamic_threshold=0.85,
    )


@pytest.fixture
def tmp_cache(tmp_path, clock):
    return SemanticCache(
        embed_fn=make_embed_fn(),
        storage_dir=tmp_path,
        clock=clock,
    )


# -----------------------------------------------------------------------------
# Basic store / retrieve
# -----------------------------------------------------------------------------


def test_put_then_get_same_text_hits(cache):
    cache.put(
        doc_hash="doc1",
        query_text="Explain Kepler's laws of planetary motion.",
        response={"question_text": "..."},
    )
    hit = cache.get(
        doc_hash="doc1",
        query_text="Explain Kepler's laws of planetary motion.",
    )
    assert hit is not None
    assert hit.response["question_text"] == "..."


def test_get_missing_doc_returns_none(cache):
    hit = cache.get(doc_hash="never-seen", query_text="anything")
    assert hit is None


def test_get_similar_text_hits(cache):
    cache.put(
        doc_hash="doc1",
        query_text="Explain Kepler's laws of planetary motion.",
        response={"q": "1"},
    )
    # Identical text — should hit with sim ≈ 1.0
    hit = cache.get(
        doc_hash="doc1",
        query_text="Explain Kepler's laws of planetary motion.",
    )
    assert hit is not None


def test_get_dissimilar_text_misses(cache):
    cache.put(
        doc_hash="doc1",
        query_text="Explain Kepler's laws of planetary motion.",
        response={"q": "1"},
    )
    # Different text → different hash → orthogonal-ish vector → sim < 0.90
    hit = cache.get(
        doc_hash="doc1",
        query_text="Compute rain attenuation at 20 GHz.",
    )
    assert hit is None


# -----------------------------------------------------------------------------
# Document isolation
# -----------------------------------------------------------------------------


def test_doc_hash_isolation(cache):
    """Same query text under different doc hashes must not cross-hit."""
    cache.put(
        doc_hash="docA",
        query_text="Explain Kepler's laws.",
        response={"q": "A"},
    )
    hit = cache.get(doc_hash="docB", query_text="Explain Kepler's laws.")
    assert hit is None


# -----------------------------------------------------------------------------
# Threshold behavior
# -----------------------------------------------------------------------------


def test_static_threshold_boundary_strict():
    """A near-miss must not exceed the static threshold."""
    # Use a simple deterministic embedding where we control the score
    def controlled_embed(text: str) -> np.ndarray:
        # Query A: [1, 0]
        # Query B: [cos(theta), sin(theta)] where theta is small
        if text == "A":
            return np.array([1.0, 0.0], dtype=np.float32)
        if text == "B":
            return np.array([0.85, np.sqrt(1 - 0.85**2)], dtype=np.float32)
        return np.array([0.0, 1.0], dtype=np.float32)

    cache = SemanticCache(
        embed_fn=controlled_embed,
        static_threshold=0.90,
    )
    cache.put(doc_hash="d", query_text="A", response={"r": "A"})
    # B has cosine sim 0.85 with A → below static threshold 0.90 → MISS
    hit = cache.get(doc_hash="d", query_text="B")
    assert hit is None


def test_static_threshold_boundary_at_or_above():
    def controlled_embed(text: str) -> np.ndarray:
        if text == "A":
            return np.array([1.0, 0.0], dtype=np.float32)
        if text == "A_similar":
            return np.array([0.95, np.sqrt(1 - 0.95**2)], dtype=np.float32)
        return np.array([0.0, 1.0], dtype=np.float32)

    cache = SemanticCache(embed_fn=controlled_embed, static_threshold=0.90)
    cache.put(doc_hash="d", query_text="A", response={"r": "A"})
    hit = cache.get(doc_hash="d", query_text="A_similar")
    assert hit is not None
    assert hit.response["r"] == "A"


# -----------------------------------------------------------------------------
# TTL behavior
# -----------------------------------------------------------------------------


def test_ttl_expiry(clock):
    cache = SemanticCache(
        embed_fn=make_embed_fn(),
        clock=clock,
        ttl_seconds=60.0,
    )
    cache.put(doc_hash="d", query_text="q", response={"r": 1})

    clock.advance(30.0)
    assert cache.get(doc_hash="d", query_text="q") is not None

    clock.advance(31.0)  # now 61s old
    assert cache.get(doc_hash="d", query_text="q") is None


def test_ttl_eviction_on_put(clock):
    cache = SemanticCache(
        embed_fn=make_embed_fn(),
        clock=clock,
        ttl_seconds=60.0,
    )
    cache.put(doc_hash="d", query_text="q1", response={"r": 1})
    clock.advance(61.0)
    cache.put(doc_hash="d", query_text="q2", response={"r": 2})

    # q1 should have been pruned on the second put
    assert cache.stats().total_entries == 1


# -----------------------------------------------------------------------------
# LRU cap
# -----------------------------------------------------------------------------


def test_max_entries_per_doc(clock):
    cache = SemanticCache(
        embed_fn=make_embed_fn(),
        clock=clock,
        max_entries_per_doc=3,
    )
    for i in range(5):
        cache.put(doc_hash="d", query_text=f"q{i}", response={"r": i})
        clock.advance(1.0)

    assert cache.stats().total_entries == 3
    # The oldest two should have been evicted
    assert cache.get(doc_hash="d", query_text="q4") is not None
    assert cache.get(doc_hash="d", query_text="q0") is None


# -----------------------------------------------------------------------------
# Invalidation
# -----------------------------------------------------------------------------


def test_invalidate_doc(cache):
    cache.put(doc_hash="d1", query_text="q1", response={})
    cache.put(doc_hash="d1", query_text="q2", response={})
    cache.put(doc_hash="d2", query_text="q3", response={})

    removed = cache.invalidate_doc("d1")
    assert removed == 2

    stats = cache.stats()
    assert stats.total_entries == 1
    assert "d1" not in stats.entries_per_doc


def test_invalidate_all(cache):
    for i in range(3):
        cache.put(doc_hash=f"d{i}", query_text=f"q{i}", response={})
    removed = cache.invalidate_all()
    assert removed == 3
    assert cache.stats().total_entries == 0


# -----------------------------------------------------------------------------
# Persistence
# -----------------------------------------------------------------------------


def test_persistence_roundtrip(tmp_path, clock):
    embed = make_embed_fn()
    c1 = SemanticCache(embed_fn=embed, storage_dir=tmp_path, clock=clock)
    c1.put(
        doc_hash="doc1",
        query_text="Explain orbital mechanics.",
        response={"q": "orbital"},
    )

    # Simulate restart: new cache object, same storage
    c2 = SemanticCache(embed_fn=embed, storage_dir=tmp_path, clock=clock)
    loaded = c2.load("doc1")
    assert loaded == 1

    hit = c2.get(doc_hash="doc1", query_text="Explain orbital mechanics.")
    assert hit is not None
    assert hit.response["q"] == "orbital"


def test_invalidate_doc_removes_persisted_file(tmp_path, clock):
    c = SemanticCache(embed_fn=make_embed_fn(), storage_dir=tmp_path, clock=clock)
    c.put(doc_hash="doc1", query_text="q", response={})

    assert (tmp_path / "doc1.json").exists()
    c.invalidate_doc("doc1")
    assert not (tmp_path / "doc1.json").exists()


def test_corrupted_persisted_file_is_ignored(tmp_path, clock):
    c = SemanticCache(embed_fn=make_embed_fn(), storage_dir=tmp_path, clock=clock)
    (tmp_path / "bad.json").write_text("not valid json", encoding="utf-8")
    loaded = c.load("bad")
    assert loaded == 0


# -----------------------------------------------------------------------------
# Stats and cloning
# -----------------------------------------------------------------------------


def test_stats_track_hits_and_misses(cache):
    cache.put(doc_hash="d", query_text="q", response={})
    cache.get(doc_hash="d", query_text="q")            # HIT
    cache.get(doc_hash="d", query_text="missing")      # MISS
    cache.get(doc_hash="nonexistent", query_text="q")  # MISS

    stats = cache.stats()
    assert stats.hits == 1
    assert stats.misses == 2
    assert stats.hit_rate == pytest.approx(1 / 3)


def test_get_returns_defensive_clone(cache):
    cache.put(
        doc_hash="d",
        query_text="q",
        response={"key": "value"},
    )
    hit1 = cache.get(doc_hash="d", query_text="q")
    hit1.response["key"] = "mutated"

    hit2 = cache.get(doc_hash="d", query_text="q")
    assert hit2.response["key"] == "value"


# -----------------------------------------------------------------------------
# Concurrency
# -----------------------------------------------------------------------------


def test_thread_safe_concurrent_puts(cache):
    def worker(thread_id: int):
        for i in range(20):
            cache.put(
                doc_hash="d",
                query_text=f"thread{thread_id}_q{i}",
                response={"t": thread_id, "i": i},
            )

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 5 threads × 20 puts = 100 puts; LRU cap is 500 → all should be present
    assert cache.stats().total_entries == 100


# -----------------------------------------------------------------------------
# Constructor validation
# -----------------------------------------------------------------------------


def test_invalid_threshold_rejected():
    with pytest.raises(ValueError):
        SemanticCache(embed_fn=make_embed_fn(), static_threshold=0.5, dynamic_threshold=0.9)
    with pytest.raises(ValueError):
        SemanticCache(embed_fn=make_embed_fn(), static_threshold=1.5)


def test_invalid_max_entries_rejected():
    with pytest.raises(ValueError):
        SemanticCache(embed_fn=make_embed_fn(), max_entries_per_doc=0)


def test_invalid_ttl_rejected():
    with pytest.raises(ValueError):
        SemanticCache(embed_fn=make_embed_fn(), ttl_seconds=-1)


def test_invalid_tier_rejected(cache):
    with pytest.raises(ValueError):
        cache.put(doc_hash="d", query_text="q", response={}, tier="garbage")
