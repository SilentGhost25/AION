"""
Rate-limit-aware provider selection.

The router decides *which* provider should serve a request. It does not
make network calls. The caller is responsible for:
    1. Requesting selection via `router.select(...)`
    2. Attempting the call
    3. Reporting the outcome via `router.record_success/failure(...)`

Time is injectable so tests can simulate sliding windows deterministically.
"""

from __future__ import annotations
import os
import threading
import time
from collections import deque
from typing import Callable, Dict, List, Optional

from .providers import APIProvider
from .circuit_breaker import CircuitBreaker


class AllProvidersUnavailable(RuntimeError):
    """Raised when no provider can serve the request."""


class APIRouter:
    """
    Selects the best provider for a request based on:
        - capability (vision required?)
        - API key availability (env var present)
        - circuit breaker state
        - RPM / RPD sliding windows
        - priority (tie-break)
        - least-recently-used (final tie-break)
    """

    def __init__(
        self,
        providers: List[APIProvider],
        circuit_breaker: Optional[CircuitBreaker] = None,
        clock: Callable[[], float] = time.monotonic,
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        if not providers:
            raise ValueError("APIRouter requires at least one provider")

        self.providers = list(providers)
        self._clock = clock
        self._env = env if env is not None else os.environ
        self.breaker = circuit_breaker or CircuitBreaker(clock=clock)
        self._lock = threading.RLock()

        # Sliding windows per provider
        self._minute_log: Dict[str, deque] = {p.name: deque(maxlen=p.rpm_limit) for p in providers}
        self._day_log: Dict[str, deque] = {
            p.name: deque(maxlen=p.rpd_limit if p.rpd_limit else 100_000)
            for p in providers
        }

    # ------------------------------------------------------------------ API

    def select(
        self,
        *,
        requires_vision: bool = False,
        exclude: Optional[List[str]] = None,
    ) -> APIProvider:
        """
        Return the best provider. Raises AllProvidersUnavailable if none qualify.
        """
        exclude_set = set(exclude or [])
        now = self._clock()

        with self._lock:
            self._prune(now)

            candidates: List[APIProvider] = []
            for p in self.providers:
                if p.name in exclude_set:
                    continue
                if not self._has_api_key(p):
                    continue
                if requires_vision and not p.supports_vision:
                    continue
                if not self.breaker.is_available(p.name):
                    continue
                if not self._within_limits(p):
                    continue
                candidates.append(p)

            if not candidates:
                raise AllProvidersUnavailable(
                    f"No provider eligible (vision={requires_vision}, "
                    f"excluded={sorted(exclude_set)})"
                )

            # Sort: priority (lower wins), then least recent usage
            candidates.sort(key=lambda p: (p.priority, self._last_request_time(p.name)))
            return candidates[0]

    def record_success(self, provider_name: str) -> None:
        with self._lock:
            self.breaker.record_success(provider_name)

    def record_failure(self, provider_name: str) -> None:
        with self._lock:
            self.breaker.record_failure(provider_name)

    def record_request(self, provider_name: str) -> None:
        """Call this before attempting a request (records the timestamp)."""
        with self._lock:
            now = self._clock()
            self._minute_log[provider_name].append(now)
            self._day_log[provider_name].append(now)

    def snapshot(self) -> Dict[str, dict]:
        with self._lock:
            now = self._clock()
            return {
                p.name: {
                    "priority": p.priority,
                    "rpm_used": self._rpm_used(p, now),
                    "rpm_limit": p.rpm_limit,
                    "rpd_used": self._rpd_used(p, now),
                    "rpd_limit": p.rpd_limit,
                    "breaker_state": self.breaker.state(p.name).value,
                    "has_key": self._has_api_key(p),
                    "supports_vision": p.supports_vision,
                }
                for p in self.providers
            }

    # ---------------------------------------------------------------- Internals

    def _has_api_key(self, provider: APIProvider) -> bool:
        # Local Ollama tolerates a missing key.
        if provider.name == "ollama_local":
            return True
        return bool(self._env.get(provider.api_key_env, "").strip())

    def _within_limits(self, provider: APIProvider) -> bool:
        return (
            self._rpm_used(provider, self._clock()) < provider.rpm_limit
            and (
                provider.rpd_limit is None
                or self._rpd_used(provider, self._clock()) < provider.rpd_limit
            )
        )

    def _rpm_used(self, provider: APIProvider, now: float) -> int:
        log = self._minute_log[provider.name]
        cutoff = now - 60.0
        return sum(1 for t in log if t >= cutoff)

    def _rpd_used(self, provider: APIProvider, now: float) -> int:
        log = self._day_log[provider.name]
        cutoff = now - 86_400.0
        return sum(1 for t in log if t >= cutoff)

    def _last_request_time(self, provider_name: str) -> float:
        log = self._minute_log.get(provider_name)
        if not log:
            return 0.0
        return log[-1] if log else 0.0

    def _prune(self, now: float) -> None:
        minute_cutoff = now - 60.0
        day_cutoff = now - 86_400.0
        for name, log in self._minute_log.items():
            while log and log[0] < minute_cutoff:
                log.popleft()
        for name, log in self._day_log.items():
            while log and log[0] < day_cutoff:
                log.popleft()
