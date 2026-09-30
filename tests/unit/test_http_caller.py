"""
Unit tests for the HTTP API caller.

Uses unittest.mock.patch to control requests.post, and a real
APIRouter with stub providers.
"""

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
import requests

from core.api.circuit_breaker import CircuitBreaker
from core.api.http_caller import (
    APIUnavailable,
    CONFIG_STATUSES,
    HTTPAPICaller,
    TRANSIENT_STATUSES,
)
from core.api.providers import APIProvider
from core.api.router import APIRouter


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


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
            rpm_limit=30,
            rpd_limit=14_400,
            text_models=["llama-3.3-70b-versatile"],
            priority=10,
        ),
        APIProvider(
            name="nvidia_nim",
            base_url="https://integrate.api.nvidia.com/v1",
            api_key_env="NVIDIA_API_KEY",
            rpm_limit=40,
            rpd_limit=None,
            text_models=["meta/llama-3.3-70b-instruct"],
            priority=20,
        ),
    ]


@pytest.fixture
def env():
    return {"GROQ_API_KEY": "gk-test", "NVIDIA_API_KEY": "nv-test"}


@pytest.fixture
def router(providers, env, clock):
    return APIRouter(
        providers=providers,
        circuit_breaker=CircuitBreaker(clock=clock),
        clock=clock,
        env=env,
    )


@pytest.fixture
def caller(router, env):
    return HTTPAPICaller(
        router=router,
        env=env,
        timeout_seconds=5.0,
        max_attempts=3,
    )


# -----------------------------------------------------------------------------
# Mock response builders
# -----------------------------------------------------------------------------


def ok_response(content: str, usage=None) -> MagicMock:
    r = MagicMock(spec=requests.Response)
    r.status_code = 200
    r.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 20},
    }
    r.text = content
    return r


def error_response(status: int, body: str = "error") -> MagicMock:
    r = MagicMock(spec=requests.Response)
    r.status_code = status
    r.text = body
    r.json.side_effect = ValueError("not json")
    return r


# -----------------------------------------------------------------------------
# Happy path
# -----------------------------------------------------------------------------


def test_call_success_returns_parsed_json(caller):
    with patch("core.api.http_caller.requests.Session.post") as mock_post:
        mock_post.return_value = ok_response('{"verdicts": [{"slot_id": "Q1"}]}')
        result = caller.call("test prompt")
    assert isinstance(result, dict)
    assert result["verdicts"][0]["slot_id"] == "Q1"


def test_call_success_returns_string_for_non_json(caller):
    with patch("core.api.http_caller.requests.Session.post") as mock_post:
        mock_post.return_value = ok_response("just some plain text")
        result = caller.call("test prompt")
    assert isinstance(result, str)
    assert result == "just some plain text"


def test_call_strips_json_code_fences(caller):
    with patch("core.api.http_caller.requests.Session.post") as mock_post:
        mock_post.return_value = ok_response('```json\n{"key": "value"}\n```')
        result = caller.call("test prompt")
    assert isinstance(result, dict)
    assert result["key"] == "value"


def test_call_records_provider_name(caller, router):
    with patch("core.api.http_caller.requests.Session.post") as mock_post:
        mock_post.return_value = ok_response('{"ok": true}')
        caller.call("test prompt")
    assert caller.last_provider_name == "groq"


def test_call_records_usage(caller):
    usage = {"prompt_tokens": 100, "completion_tokens": 50}
    with patch("core.api.http_caller.requests.Session.post") as mock_post:
        mock_post.return_value = ok_response('{"ok": true}', usage=usage)
        caller.call("test prompt")
    assert caller.last_usage == usage


# -----------------------------------------------------------------------------
# Provider fallback
# -----------------------------------------------------------------------------


def test_call_falls_back_on_500(caller):
    with patch("core.api.http_caller.requests.Session.post") as mock_post:
        mock_post.side_effect = [
            error_response(500, "internal error"),
            ok_response('{"ok": true}'),
        ]
        result = caller.call("test prompt")
    assert result == {"ok": True}
    # groq failed, nvidia_nim succeeded
    assert caller.last_provider_name == "nvidia_nim"


def test_call_falls_back_on_429(caller):
    with patch("core.api.http_caller.requests.Session.post") as mock_post:
        mock_post.side_effect = [
            error_response(429, "rate limited"),
            ok_response('{"ok": true}'),
        ]
        result = caller.call("test prompt")
    assert result == {"ok": True}
    assert caller.last_provider_name == "nvidia_nim"


def test_call_falls_back_on_timeout(caller):
    with patch("core.api.http_caller.requests.Session.post") as mock_post:
        mock_post.side_effect = [
            requests.Timeout("timed out"),
            ok_response('{"ok": true}'),
        ]
        result = caller.call("test prompt")
    assert result == {"ok": True}


def test_call_falls_back_on_connection_error(caller):
    with patch("core.api.http_caller.requests.Session.post") as mock_post:
        mock_post.side_effect = [
            requests.ConnectionError("refused"),
            ok_response('{"ok": true}'),
        ]
        result = caller.call("test prompt")
    assert result == {"ok": True}


def test_call_excludes_failed_provider(caller, router):
    """After groq fails, router.select should not pick groq again."""
    with patch("core.api.http_caller.requests.Session.post") as mock_post:
        mock_post.side_effect = [
            error_response(500),
            ok_response('{"ok": true}'),
        ]
        caller.call("test prompt")
    # Two posts happened: first to groq, second to nvidia_nim.
    # Second call's URL must not be groq's.
    first_call_url = mock_post.call_args_list[0][0][0]
    second_call_url = mock_post.call_args_list[1][0][0]
    assert "groq" in first_call_url
    assert "nvidia" in second_call_url


# -----------------------------------------------------------------------------
# All providers unavailable
# -----------------------------------------------------------------------------


def test_call_raises_when_all_fail(caller):
    with patch("core.api.http_caller.requests.Session.post") as mock_post:
        mock_post.side_effect = [
            error_response(500),
            error_response(500),
        ]
        with pytest.raises(APIUnavailable):
            caller.call("test prompt")


def test_call_raises_when_no_providers_have_keys(providers, clock):
    router = APIRouter(
        providers=providers,
        circuit_breaker=CircuitBreaker(clock=clock),
        clock=clock,
        env={},  # no keys
    )
    caller = HTTPAPICaller(router=router, env={})
    with pytest.raises(APIUnavailable):
        caller.call("test prompt")


def test_call_raises_on_empty_prompt(caller):
    with pytest.raises(ValueError):
        caller.call("")


# -----------------------------------------------------------------------------
# Config errors
# -----------------------------------------------------------------------------


def test_call_treats_404_as_config_error(caller):
    """404 (model not found) fails the provider but tries another."""
    with patch("core.api.http_caller.requests.Session.post") as mock_post:
        mock_post.side_effect = [
            error_response(404, "model not found"),
            ok_response('{"ok": true}'),
        ]
        result = caller.call("test prompt")
    assert result == {"ok": True}


def test_call_treats_401_as_config_error(caller):
    with patch("core.api.http_caller.requests.Session.post") as mock_post:
        mock_post.side_effect = [
            error_response(401, "unauthorized"),
            ok_response('{"ok": true}'),
        ]
        result = caller.call("test prompt")
    assert result == {"ok": True}


# -----------------------------------------------------------------------------
# Response parsing
# -----------------------------------------------------------------------------


def test_call_raises_when_choices_empty(caller):
    r = MagicMock(spec=requests.Response)
    r.status_code = 200
    r.json.return_value = {"choices": []}
    r.text = ""
    with patch("core.api.http_caller.requests.Session.post") as mock_post:
        mock_post.side_effect = [r, r, r]
        with pytest.raises(APIUnavailable):
            caller.call("test prompt")


def test_call_raises_when_content_missing(caller):
    r = MagicMock(spec=requests.Response)
    r.status_code = 200
    r.json.return_value = {"choices": [{"message": {}}]}
    r.text = ""
    with patch("core.api.http_caller.requests.Session.post") as mock_post:
        mock_post.side_effect = [r, r, r]
        with pytest.raises(APIUnavailable):
            caller.call("test prompt")


def test_call_raises_when_body_not_json(caller):
    r = MagicMock(spec=requests.Response)
    r.status_code = 200
    r.json.side_effect = ValueError("bad json")
    r.text = "not json"
    with patch("core.api.http_caller.requests.Session.post") as mock_post:
        mock_post.side_effect = [r, r, r]
        with pytest.raises(APIUnavailable):
            caller.call("test prompt")


# -----------------------------------------------------------------------------
# Schema pass-through
# -----------------------------------------------------------------------------


def test_call_passes_json_object_format_when_schema_given(caller):
    with patch("core.api.http_caller.requests.Session.post") as mock_post:
        mock_post.return_value = ok_response('{"ok": true}')
        caller.call("test prompt", schema={"type": "object"})
    # Inspect the body sent
    body = mock_post.call_args_list[0][1]["json"]
    assert body["response_format"] == {"type": "json_object"}


def test_call_no_response_format_when_no_schema(caller):
    with patch("core.api.http_caller.requests.Session.post") as mock_post:
        mock_post.return_value = ok_response('{"ok": true}')
        caller.call("test prompt")
    body = mock_post.call_args_list[0][1]["json"]
    assert "response_format" not in body


# -----------------------------------------------------------------------------
# Request shape
# -----------------------------------------------------------------------------


def test_call_includes_bearer_token(caller):
    with patch("core.api.http_caller.requests.Session.post") as mock_post:
        mock_post.return_value = ok_response('{"ok": true}')
        caller.call("test prompt")
    headers = mock_post.call_args_list[0][1]["headers"]
    assert headers["Authorization"] == "Bearer gk-test"


def test_call_includes_seed_and_temperature(caller):
    with patch("core.api.http_caller.requests.Session.post") as mock_post:
        mock_post.return_value = ok_response('{"ok": true}')
        caller.call("test prompt")
    body = mock_post.call_args_list[0][1]["json"]
    assert body["seed"] == 42
    assert body["temperature"] == 0.1


# -----------------------------------------------------------------------------
# Circuit breaker integration
# -----------------------------------------------------------------------------


def test_call_trips_circuit_after_threshold_failures(caller, router, clock):
    """After 3 consecutive failures on groq, the circuit opens."""
    with patch("core.api.http_caller.requests.Session.post") as mock_post:
        # First call: groq fails, nvidia succeeds
        mock_post.side_effect = [
            error_response(500),  # groq fail 1
            ok_response('{"ok": true}'),
        ]
        caller.call("p1")

        mock_post.side_effect = [
            error_response(500),  # groq fail 2
            ok_response('{"ok": true}'),
        ]
        caller.call("p2")

        mock_post.side_effect = [
            error_response(500),  # groq fail 3 → circuit opens
            ok_response('{"ok": true}'),
        ]
        caller.call("p3")

    # Now groq is OPEN
    assert router.breaker.state("groq").value == "open"

    # Next call should skip groq
    with patch("core.api.http_caller.requests.Session.post") as mock_post:
        mock_post.return_value = ok_response('{"ok": true}')
        result = caller.call("p4")
    # Only one POST because groq was skipped
    assert mock_post.call_count == 1
    assert caller.last_provider_name == "nvidia_nim"


# -----------------------------------------------------------------------------
# Constructor validation
# -----------------------------------------------------------------------------


def test_missing_router_rejected():
    with pytest.raises(ValueError):
        HTTPAPICaller(router=None)


def test_invalid_timeout_rejected(router, env):
    with pytest.raises(ValueError):
        HTTPAPICaller(router=router, env=env, timeout_seconds=0)


def test_invalid_max_attempts_rejected(router, env):
    with pytest.raises(ValueError):
        HTTPAPICaller(router=router, env=env, max_attempts=0)


def test_invalid_temperature_rejected(router, env):
    with pytest.raises(ValueError):
        HTTPAPICaller(router=router, env=env, temperature=3.0)
