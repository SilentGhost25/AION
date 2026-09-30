"""
Unit tests for the three-state circuit breaker.

A `FakeClock` is used so cooldown behavior is deterministic.
"""

import threading
import pytest

from core.api.circuit_breaker import CircuitBreaker, CircuitState


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


@pytest.fixture
def breaker(clock):
    return CircuitBreaker(failure_threshold=3, cooldown_seconds=60.0, clock=clock)


def test_closed_state_allows_requests(breaker):
    assert breaker.state("groq") == CircuitState.CLOSED
    assert breaker.is_available("groq") is True


def test_opens_after_threshold_failures(breaker):
    for _ in range(3):
        breaker.record_failure("groq")
    assert breaker.state("groq") == CircuitState.OPEN
    assert breaker.is_available("groq") is False


def test_success_below_threshold_does_not_open(breaker):
    breaker.record_failure("groq")
    breaker.record_failure("groq")
    breaker.record_success("groq")
    assert breaker.state("groq") == CircuitState.CLOSED
    # Failure counter should reset on success
    breaker.record_failure("groq")
    breaker.record_failure("groq")
    assert breaker.state("groq") == CircuitState.CLOSED  # only 2 failures


def test_transitions_to_half_open_after_cooldown(breaker, clock):
    for _ in range(3):
        breaker.record_failure("groq")
    assert breaker.is_available("groq") is False

    clock.advance(59.0)
    assert breaker.is_available("groq") is False

    clock.advance(2.0)  # total 61s > 60s cooldown
    assert breaker.is_available("groq") is True
    assert breaker.state("groq") == CircuitState.HALF_OPEN


def test_half_open_success_closes(breaker, clock):
    for _ in range(3):
        breaker.record_failure("groq")
    clock.advance(61.0)
    breaker.is_available("groq")  # transitions to HALF_OPEN

    breaker.record_success("groq")
    assert breaker.state("groq") == CircuitState.CLOSED
    assert breaker.is_available("groq") is True


def test_half_open_failure_reopens_immediately(breaker, clock):
    for _ in range(3):
        breaker.record_failure("groq")
    clock.advance(61.0)
    breaker.is_available("groq")  # HALF_OPEN

    breaker.record_failure("groq")
    assert breaker.state("groq") == CircuitState.OPEN
    # Must wait another cooldown
    clock.advance(30.0)
    assert breaker.is_available("groq") is False


def test_per_provider_isolation(breaker):
    for _ in range(3):
        breaker.record_failure("groq")
    assert breaker.state("groq") == CircuitState.OPEN
    assert breaker.state("nvidia_nim") == CircuitState.CLOSED
    assert breaker.is_available("nvidia_nim") is True


def test_snapshot_reports_all_providers(breaker):
    breaker.record_failure("groq")
    breaker.record_failure("nvidia_nim")
    snap = breaker.snapshot()
    assert "groq" in snap
    assert "nvidia_nim" in snap
    assert snap["groq"]["failures"] == 1


def test_reset_clears_state(breaker):
    for _ in range(3):
        breaker.record_failure("groq")
    breaker.reset("groq")
    assert breaker.state("groq") == CircuitState.CLOSED
    assert breaker.is_available("groq") is True


def test_invalid_threshold_rejected():
    with pytest.raises(ValueError):
        CircuitBreaker(failure_threshold=0)
    with pytest.raises(ValueError):
        CircuitBreaker(cooldown_seconds=0)


def test_thread_safety_under_concurrent_failures():
    """Concurrent failures must not corrupt the counter."""
    breaker = CircuitBreaker(failure_threshold=100)  # high, so it stays CLOSED
    def worker():
        for _ in range(100):
            breaker.record_failure("groq")

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 10 threads × 100 failures = 1000
    assert breaker.snapshot()["groq"]["failures"] == 1000
