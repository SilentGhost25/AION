"""
Unit tests for the rate-limit-aware provider router.
"""

import pytest

from core.api.providers import APIProvider
from core.api.router import APIRouter, AllProvidersUnavailable
from core.api.circuit_breaker import CircuitBreaker


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
def providers():
    return [
        APIProvider(
            name="groq",
            base_url="https://api.groq.com/openai/v1",
            api_key_env="GROQ_API_KEY",
            rpm_limit=3,
            rpd_limit=10,
            text_models=["llama-3.3-70b"],
            vision_models=["llama-3.2-vision"],
            priority=10,
        ),
        APIProvider(
            name="nvidia_nim",
            base_url="https://integrate.api.nvidia.com/v1",
            api_key_env="NVIDIA_API_KEY",
            rpm_limit=5,
            rpd_limit=None,
            text_models=["llama-3.3-70b"],
            vision_models=["llama-3.2-vision"],
            priority=20,
        ),
        APIProvider(
            name="google_ai",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key_env="GOOGLE_AI_API_KEY",
            rpm_limit=15,
            rpd_limit=500,
            text_models=["gemini-2.5-flash"],
            vision_models=[],       # no vision
            priority=30,
        ),
    ]


@pytest.fixture
def env():
    return {
        "GROQ_API_KEY": "gk-test",
        "NVIDIA_API_KEY": "nv-test",
        "GOOGLE_AI_API_KEY": "ga-test",
    }


@pytest.fixture
def router(providers, env, clock):
    return APIRouter(
        providers=providers,
        circuit_breaker=CircuitBreaker(clock=clock),
        clock=clock,
        env=env,
    )


# ------------------------------------------------------------- Selection

def test_select_prefers_lowest_priority(router):
    p = router.select()
    assert p.name == "groq"


def test_vision_routing_excludes_non_vision(router):
    p = router.select(requires_vision=True)
    assert p.name == "groq"
    assert p.supports_vision


def test_vision_routing_falls_back_when_primary_rate_limited(router, clock):
    # Fill groq's RPM (3)
    for _ in range(3):
        router.record_request("groq")
    # Next vision request must skip groq
    p = router.select(requires_vision=True)
    assert p.name == "nvidia_nim"


# ------------------------------------------------------------- Rate limits

def test_rpm_window_slides(router, clock):
    for _ in range(3):
        router.record_request("groq")
    with pytest.raises(AllProvidersUnavailable):
        # groq RPM exhausted; only groq has vision in this fixture variant
        # (vision on nvidia_nim exists, so this actually falls back)
        router.select(requires_vision=True, exclude=["nvidia_nim"])

    # Advance past RPM window
    clock.advance(61.0)
    p = router.select()
    assert p.name == "groq"


def test_rpd_window_slides(router, clock):
    # Drain groq's RPD (10) but stay under RPM by spacing requests
    for i in range(10):
        router.record_request("groq")
        clock.advance(61.0)  # RPM window rolls over

    # groq should be blocked by RPD
    p = router.select()
    assert p.name == "nvidia_nim"

    # Advance past day window
    clock.advance(86_401.0)
    p = router.select()
    assert p.name == "groq"


# ------------------------------------------------------------- Circuit integration

def test_open_circuit_skipped(router, clock):
    # Force groq's circuit OPEN
    for _ in range(3):
        router.record_failure("groq")
    p = router.select()
    assert p.name == "nvidia_nim"


def test_circuit_recovers_after_cooldown(router, clock):
    for _ in range(3):
        router.record_failure("groq")
    assert router.select().name == "nvidia_nim"

    clock.advance(61.0)
    # HALF_OPEN — router should prefer groq again (lower priority)
    p = router.select()
    assert p.name == "groq"


# ------------------------------------------------------------- Env-based eligibility

def test_missing_api_key_excludes_provider(providers, clock):
    env = {"NVIDIA_API_KEY": "nv-test"}  # no GROQ_API_KEY
    router = APIRouter(
        providers=providers,
        circuit_breaker=CircuitBreaker(clock=clock),
        clock=clock,
        env=env,
    )
    p = router.select()
    assert p.name == "nvidia_nim"


def test_all_providers_missing_keys_raises(providers, clock):
    router = APIRouter(
        providers=providers,
        circuit_breaker=CircuitBreaker(clock=clock),
        clock=clock,
        env={},
    )
    with pytest.raises(AllProvidersUnavailable):
        router.select()


# ------------------------------------------------------------- Exclusions

def test_exclude_list(router):
    p = router.select(exclude=["groq"])
    assert p.name == "nvidia_nim"


def test_exclude_all_raises(router):
    with pytest.raises(AllProvidersUnavailable):
        router.select(exclude=["groq", "nvidia_nim", "google_ai"])


# ------------------------------------------------------------- Snapshot

def test_snapshot_shape(router):
    snap = router.snapshot()
    assert set(snap.keys()) == {"groq", "nvidia_nim", "google_ai"}
    for v in snap.values():
        assert {"priority", "rpm_used", "rpm_limit", "breaker_state", "has_key"} <= v.keys()


# ------------------------------------------------------------- Invalid config

def test_empty_providers_rejected(clock):
    with pytest.raises(ValueError):
        APIRouter(providers=[], clock=clock)
