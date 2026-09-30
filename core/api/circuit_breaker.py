"""
Three-state circuit breaker for LLM API providers.

State machine:

    CLOSED ──(N consecutive failures)──▶ OPEN
      ▲                                   │
      │                                   │ (cooldown elapses)
      │                                   ▼
      └──(success)──────────────── HALF_OPEN
                                          │
                                          └──(failure)──▶ OPEN

Thread-safe. All state is per-provider. Time is injectable for tests.
"""

from __future__ import annotations
import threading
import time
from enum import Enum
from typing import Callable, Dict, Optional


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """
    Per-provider three-state breaker.

    Parameters
    ----------
    failure_threshold : int
        Consecutive failures required to open the circuit. Default 3.
    cooldown_seconds : float
        Time the circuit stays OPEN before allowing a HALF_OPEN probe.
        Default 60 seconds.
    clock : Callable[[], float]
        Monotonic time source. Override for deterministic tests.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        cooldown_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if cooldown_seconds <= 0:
            raise ValueError("cooldown_seconds must be > 0")

        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._clock = clock
        self._lock = threading.RLock()

        self._state: Dict[str, CircuitState] = {}
        self._failures: Dict[str, int] = {}
        self._opened_at: Dict[str, float] = {}

    # ------------------------------------------------------------------ API

    def is_available(self, provider: str) -> bool:
        """
        Return True if the provider may be attempted now.
        Transitions OPEN → HALF_OPEN lazily when the cooldown has elapsed.
        """
        with self._lock:
            state = self._state.get(provider, CircuitState.CLOSED)

            if state == CircuitState.CLOSED:
                return True

            if state == CircuitState.OPEN:
                elapsed = self._clock() - self._opened_at.get(provider, 0.0)
                if elapsed >= self.cooldown_seconds:
                    self._state[provider] = CircuitState.HALF_OPEN
                    return True
                return False

            # HALF_OPEN — allow the probe
            return True

    def record_success(self, provider: str) -> None:
        with self._lock:
            self._state[provider] = CircuitState.CLOSED
            self._failures[provider] = 0
            self._opened_at.pop(provider, None)

    def record_failure(self, provider: str) -> None:
        with self._lock:
            self._failures[provider] = self._failures.get(provider, 0) + 1

            # A failure in HALF_OPEN reopens immediately
            current = self._state.get(provider, CircuitState.CLOSED)
            if current == CircuitState.HALF_OPEN:
                self._state[provider] = CircuitState.OPEN
                self._opened_at[provider] = self._clock()
                return

            if self._failures[provider] >= self.failure_threshold:
                self._state[provider] = CircuitState.OPEN
                self._opened_at[provider] = self._clock()

    def state(self, provider: str) -> CircuitState:
        with self._lock:
            return self._state.get(provider, CircuitState.CLOSED)

    def snapshot(self) -> Dict[str, dict]:
        """Return a debug-safe view of all known providers."""
        with self._lock:
            providers = set(self._state) | set(self._failures)
            return {
                p: {
                    "state": self._state.get(p, CircuitState.CLOSED).value,
                    "failures": self._failures.get(p, 0),
                    "opened_at": self._opened_at.get(p),
                }
                for p in providers
            }

    def reset(self, provider: Optional[str] = None) -> None:
        """For tests and administrative recovery."""
        with self._lock:
            if provider is None:
                self._state.clear()
                self._failures.clear()
                self._opened_at.clear()
            else:
                self._state.pop(provider, None)
                self._failures.pop(provider, None)
                self._opened_at.pop(provider, None)
