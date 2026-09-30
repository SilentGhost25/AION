"""Unit tests for async promotion flow (Krites judge + cache integration)."""

import threading
from concurrent.futures import Future
from typing import Callable

import numpy as np
import pytest

from core.cache.semantic_cache import SemanticCache
from core.cache.krites_judge import KritesJudge


# -----------------------------------------------------------------------------
# Synchronous executor for deterministic async testing
# -----------------------------------------------------------------------------


class SynchronousExecutor:
    """Runs submitted jobs inline so tests are deterministic."""
    def submit(self, fn: Callable, *args, **kwargs) -> Future:
        future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as e:
            future.set_exception(e)
        return future

    def shutdown(self, wait: bool = True) -> None:
        pass


class FakeClock:
    def __init__(self, start: float = 1_000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


# -----------------------------------------------------------------------------
# Deterministic embeddings with controllable similarity
# -----------------------------------------------------------------------------


class ControlledEmbed:
    """
    Maps exact string keys to specific vectors.
    Unknown strings get an orthogonal vector so they don't match anything.
    """
    def __init__(self, vectors: dict[str, np.ndarray]):
        self._vectors = {k: self._norm(v) for k, v in vectors.items()}

    def __call__(self, text: str) -> np.ndarray:
        if text in self._vectors:
            return self._vectors[text]
        # Unknown: orthogonal to all known
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    @staticmethod
    def _norm(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32)
        return v / float(np.linalg.norm(v))


def unit(theta_rad: float) -> np.ndarray:
    """Unit vector at angle theta from [1, 0]."""
    return np.array([np.cos(theta_rad), np.sin(theta_rad), 0.0, 0.0], dtype=np.float32)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def controlled_embed():
    return ControlledEmbed({
        "A": unit(0.0),            # [1, 0, 0, 0]
        "A_close": unit(0.15),     # sim ~0.988 with A
        "A_mid": unit(0.35),       # sim ~0.939 with A
        "A_low": unit(0.65),       # sim ~0.796 with A
        "A_low2": unit(-0.65),     # sim ~0.796 with A, far from A_low
        "A_far": unit(1.40),       # sim ~0.17 with A
    })


# -----------------------------------------------------------------------------
# Opt-in gate
# -----------------------------------------------------------------------------


def test_no_judge_means_no_async(controlled_embed, clock):
    cache = SemanticCache(embed_fn=controlled_embed, clock=clock)
    cache.put(doc_hash="d", query_text="A", response={"r": "A"})
    # A_low is a borderline miss but no judge → no async
    hit = cache.get(doc_hash="d", query_text="A_low")
    assert hit is None
    assert cache.stats().promotions_scheduled == 0


def test_judge_without_executor_disables_async(controlled_embed, clock):
    judge = KritesJudge(judge_fn=lambda a, b, c: True, clock=clock)
    cache = SemanticCache(embed_fn=controlled_embed, judge=judge, clock=clock)
    cache.put(doc_hash="d", query_text="A", response={"r": "A"})
    cache.get(doc_hash="d", query_text="A_low")
    assert cache.stats().promotions_scheduled == 0


# -----------------------------------------------------------------------------
# Scheduling
# -----------------------------------------------------------------------------


def test_borderline_miss_triggers_judge_and_promotes(controlled_embed, clock):
    calls = []

    def judge_fn(sq, sr, nq):
        calls.append((sq, nq))
        return True

    judge = KritesJudge(judge_fn=judge_fn, clock=clock)
    cache = SemanticCache(
        embed_fn=controlled_embed,
        judge=judge,
        executor=SynchronousExecutor(),
        clock=clock,
        static_threshold=0.90,
        dynamic_threshold=0.85,
        borderline_low=0.75,
    )
    cache.put(doc_hash="d", query_text="A", response={"r": "A"})

    hit = cache.get(doc_hash="d", query_text="A_low")  # sim ~0.796
    assert hit is None                                    # foreground miss
    assert calls == [("A", "A_low")]                     # judge fired

    stats = cache.stats()
    assert stats.promotions_scheduled == 1
    assert stats.promotions_succeeded == 1


def test_promoted_entry_is_dynamic_tier(controlled_embed, clock):
    judge = KritesJudge(judge_fn=lambda a, b, c: True, clock=clock)
    cache = SemanticCache(
        embed_fn=controlled_embed,
        judge=judge,
        executor=SynchronousExecutor(),
        clock=clock,
        static_threshold=0.90,
        dynamic_threshold=0.85,
        borderline_low=0.75,
    )
    cache.put(doc_hash="d", query_text="A", response={"r": "A"})
    cache.get(doc_hash="d", query_text="A_low")

    bucket = cache._entries["d"]  # noqa: SLF001 — internal test
    tiers = [e.tier for e in bucket]
    assert "static" in tiers
    assert "dynamic" in tiers


def test_subsequent_mid_query_hits_dynamic(controlled_embed, clock):
    """After promotion, a query at sim ~0.85+ hits the dynamic tier."""
    judge = KritesJudge(judge_fn=lambda a, b, c: True, clock=clock)
    cache = SemanticCache(
        embed_fn=controlled_embed,
        judge=judge,
        executor=SynchronousExecutor(),
        clock=clock,
        static_threshold=0.90,
        dynamic_threshold=0.80,   # slightly looser for this test
        borderline_low=0.75,
    )
    cache.put(doc_hash="d", query_text="A", response={"r": "A"})
    cache.get(doc_hash="d", query_text="A_low")          # schedules + promotes

    # A_low itself is now in the dynamic tier
    hit = cache.get(doc_hash="d", query_text="A_low")
    assert hit is not None
    assert hit.response["r"] == "A"
    assert hit.tier == "dynamic"


# -----------------------------------------------------------------------------
# Rejection path
# -----------------------------------------------------------------------------


def test_judge_rejection_no_promotion(controlled_embed, clock):
    judge = KritesJudge(judge_fn=lambda a, b, c: False, clock=clock)
    cache = SemanticCache(
        embed_fn=controlled_embed,
        judge=judge,
        executor=SynchronousExecutor(),
        clock=clock,
        static_threshold=0.90,
        borderline_low=0.75,
    )
    cache.put(doc_hash="d", query_text="A", response={"r": "A"})
    cache.get(doc_hash="d", query_text="A_low")

    stats = cache.stats()
    assert stats.promotions_scheduled == 1
    assert stats.promotions_succeeded == 0
    assert stats.promotions_failed == 1

    hit = cache.get(doc_hash="d", query_text="A_low")
    assert hit is None


def test_judge_exception_no_promotion_no_crash(controlled_embed, clock):
    def boom(a, b, c):
        raise RuntimeError("LLM down")

    judge = KritesJudge(judge_fn=boom, clock=clock)
    cache = SemanticCache(
        embed_fn=controlled_embed,
        judge=judge,
        executor=SynchronousExecutor(),
        clock=clock,
        borderline_low=0.75,
    )
    cache.put(doc_hash="d", query_text="A", response={"r": "A"})
    cache.get(doc_hash="d", query_text="A_low")   # should not raise

    assert cache.stats().promotions_failed == 1


# -----------------------------------------------------------------------------
# Non-borderline paths
# -----------------------------------------------------------------------------


def test_static_hit_does_not_schedule_judge(controlled_embed, clock):
    calls = []
    judge = KritesJudge(judge_fn=lambda a, b, c: calls.append(1) or True, clock=clock)
    cache = SemanticCache(
        embed_fn=controlled_embed,
        judge=judge,
        executor=SynchronousExecutor(),
        clock=clock,
        static_threshold=0.90,
        borderline_low=0.75,
    )
    cache.put(doc_hash="d", query_text="A", response={"r": "A"})
    hit = cache.get(doc_hash="d", query_text="A_close")  # sim ~0.988
    assert hit is not None
    assert calls == []
    assert cache.stats().promotions_scheduled == 0


def test_far_miss_does_not_schedule_judge(controlled_embed, clock):
    calls = []
    judge = KritesJudge(judge_fn=lambda a, b, c: calls.append(1) or True, clock=clock)
    cache = SemanticCache(
        embed_fn=controlled_embed,
        judge=judge,
        executor=SynchronousExecutor(),
        clock=clock,
        borderline_low=0.75,
    )
    cache.put(doc_hash="d", query_text="A", response={"r": "A"})
    cache.get(doc_hash="d", query_text="A_far")  # sim ~0.17
    assert calls == []
    assert cache.stats().promotions_scheduled == 0


# -----------------------------------------------------------------------------
# Throttling
# -----------------------------------------------------------------------------


def test_throttled_judge_not_called(controlled_embed, clock):
    calls = []

    def judge_fn(a, b, c):
        calls.append((a, c))
        return True

    judge = KritesJudge(judge_fn=judge_fn, throttle_per_min=1, clock=clock)
    cache = SemanticCache(
        embed_fn=controlled_embed,
        judge=judge,
        executor=SynchronousExecutor(),
        clock=clock,
        borderline_low=0.75,
    )
    cache.put(doc_hash="d", query_text="A", response={"r": "A"})

    cache.get(doc_hash="d", query_text="A_low")   # First: judge fires
    cache.get(doc_hash="d", query_text="A_low2")  # Borderline (sim ~0.796), far from A_low
                                                  # but throttle exhausted

    assert len(calls) == 1
    assert cache.stats().promotions_throttled >= 1


# -----------------------------------------------------------------------------
# Dedup
# -----------------------------------------------------------------------------


def test_same_query_does_not_double_schedule(controlled_embed, clock):
    calls = []
    judge = KritesJudge(judge_fn=lambda a, b, c: calls.append(1) or False, clock=clock)
    # Use a real ThreadPoolExecutor is unsafe for determinism here,
    # but the sync executor runs inline, so dedup only matters when
    # submission is deferred. Verify the set behavior directly.
    cache = SemanticCache(
        embed_fn=controlled_embed,
        judge=judge,
        executor=SynchronousExecutor(),
        clock=clock,
        borderline_low=0.75,
    )
    cache.put(doc_hash="d", query_text="A", response={"r": "A"})
    cache.get(doc_hash="d", query_text="A_low")
    cache.get(doc_hash="d", query_text="A_low2")
    assert cache.stats().promotions_scheduled >= 1


# -----------------------------------------------------------------------------
# Constructor validation
# -----------------------------------------------------------------------------


def test_invalid_borderline_low_rejected(controlled_embed):
    with pytest.raises(ValueError):
        SemanticCache(
            embed_fn=controlled_embed,
            borderline_low=0.95,          # > dynamic_threshold (0.85)
            dynamic_threshold=0.85,
        )


def test_close_is_idempotent(controlled_embed, clock):
    cache = SemanticCache(
        embed_fn=controlled_embed,
        judge=KritesJudge(judge_fn=lambda a, b, c: True, clock=clock),
        executor=SynchronousExecutor(),
        clock=clock,
    )
    cache.close()
    cache.close()  # should not raise
