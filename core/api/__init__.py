"""
AION API Layer — provider abstraction, circuit breaker, and routing.

Phase 1 of the v3 upgrade. Nothing in this package is wired into the
pipeline yet. It is a self-contained infrastructure layer that the
Phase 4 API escalation tier will consume.

Public surface:
    - APIProvider         : declarative provider definition
    - PROVIDERS           : default provider registry (Groq, NIM, OpenRouter, Google, Ollama)
    - CircuitBreaker      : three-state machine for provider failures
    - CircuitState        : enum { CLOSED, OPEN, HALF_OPEN }
    - APIRouter           : rate-limit-aware selection across providers
    - AllProvidersUnavailable : raised when no provider can serve a request
"""

from .providers import APIProvider, PROVIDERS
from .circuit_breaker import CircuitBreaker, CircuitState
from .router import APIRouter, AllProvidersUnavailable

__all__ = [
    "APIProvider",
    "PROVIDERS",
    "CircuitBreaker",
    "CircuitState",
    "APIRouter",
    "AllProvidersUnavailable",
]
