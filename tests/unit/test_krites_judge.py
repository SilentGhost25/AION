"""Unit tests for the Krites judge wrapper."""

import pytest

from core.cache.krites_judge import KritesJudge


class FakeClock:
    def __init__(self, start: float = 1_000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


@pytest.fixture
def clock():
    return FakeClock()


def test_judge_returns_bool_true(clock):
    judge = KritesJudge(judge_fn=lambda a, b, c: True, clock=clock)
    assert judge.judge(doc_hash="d", static_query="q1", static_response={}, new_query="q2") is True


def test_judge_returns_bool_false(clock):
    judge = KritesJudge(judge_fn=lambda a, b, c: False, clock=clock)
    assert judge.judge(doc_hash="d", static_query="q1", static_response={}, new_query="q2") is False


def test_judge_exception_treated_as_rejection(clock):
    def boom(a, b, c):
        raise RuntimeError("LLM crashed")
    judge = KritesJudge(judge_fn=boom, clock=clock)
    assert judge.judge(doc_hash="d", static_query="q1", static_response={}, new_query="q2") is False


def test_throttle_allows_under_limit(clock):
    judge = KritesJudge(judge_fn=lambda a, b, c: True, throttle_per_min=3, clock=clock)
    assert judge.should_judge("d") is True
    judge.judge(doc_hash="d", static_query="q1", static_response={}, new_query="q2")
    assert judge.should_judge("d") is True
    judge.judge(doc_hash="d", static_query="q1", static_response={}, new_query="q2")
    assert judge.should_judge("d") is True
    judge.judge(doc_hash="d", static_query="q1", static_response={}, new_query="q2")
    # Now at limit (3)
    assert judge.should_judge("d") is False


def test_throttle_window_slides(clock):
    judge = KritesJudge(judge_fn=lambda a, b, c: True, throttle_per_min=2, clock=clock)
    for _ in range(2):
        judge.judge(doc_hash="d", static_query="q1", static_response={}, new_query="q2")
    assert judge.should_judge("d") is False

    clock.advance(61.0)
    assert judge.should_judge("d") is True


def test_throttle_isolated_per_doc(clock):
    judge = KritesJudge(judge_fn=lambda a, b, c: True, throttle_per_min=1, clock=clock)
    judge.judge(doc_hash="A", static_query="q1", static_response={}, new_query="q2")
    assert judge.should_judge("A") is False
    assert judge.should_judge("B") is True  # different doc, no calls yet


def test_invalid_throttle_rejected(clock):
    with pytest.raises(ValueError):
        KritesJudge(judge_fn=lambda a, b, c: True, throttle_per_min=0, clock=clock)


def test_snapshot_counts_recent_calls(clock):
    judge = KritesJudge(judge_fn=lambda a, b, c: True, throttle_per_min=10, clock=clock)
    judge.judge(doc_hash="A", static_query="q", static_response={}, new_query="q2")
    judge.judge(doc_hash="A", static_query="q", static_response={}, new_query="q3")
    judge.judge(doc_hash="B", static_query="q", static_response={}, new_query="q2")
    snap = judge.snapshot()
    assert snap["A"] == 2
    assert snap["B"] == 1
    clock.advance(61.0)
    assert judge.snapshot() == {"A": 0, "B": 0}
