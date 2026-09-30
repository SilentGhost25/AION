"""
HTTP API caller for the Evaluation Agent (and future callers).

Uses APIRouter to select a healthy provider, makes the HTTP call to
the OpenAI-compatible /chat/completions endpoint, parses the response,
and reports success/failure back to the router's circuit breaker.

The caller is transport-only. It does not parse JSON out of the
returned content — that is the caller's (Evaluation Agent's) job,
using its own tolerant parser.

Public surface:
    HTTPAPICaller(router, ...)
    .call(prompt, schema=None) -> parsed content or string
    .last_provider_name -> Optional[str]
    .last_usage -> Optional[dict]
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, List, Optional

try:
    import requests
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "core.api.http_caller requires requests. "
        "Install with: pip install requests"
    ) from e

from .providers import APIProvider
from .router import APIRouter, AllProvidersUnavailable


# -----------------------------------------------------------------------------
# Errors
# -----------------------------------------------------------------------------


class APIUnavailable(RuntimeError):
    """
    Raised when no provider could serve the request.

    Callers (Evaluation Agent, Refinement Agent) catch this and degrade.
    """
    pass


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


DEFAULT_CALL_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 4096
RETRY_DELAY_SECONDS = 0.25
OPENROUTER_REFERER = "https://github.com/Tarun-J/AION"

# HTTP statuses that indicate a transient failure we should retry
# against another provider.
TRANSIENT_STATUSES = frozenset({408, 429, 500, 502, 503, 504})

# Statuses that indicate a config problem (bad model name, bad key)
# — retrying against another provider may still work, but not against
# the same one.
CONFIG_STATUSES = frozenset({400, 401, 403, 404})


# -----------------------------------------------------------------------------
# Caller
# -----------------------------------------------------------------------------


class HTTPAPICaller:
    """
    OpenAI-compatible HTTP caller with provider rotation.

    Parameters
    ----------
    router : APIRouter
        Owns circuit breaker, rate limits, and provider selection.
    timeout_seconds : float
        Per-request timeout. Default 30s.
    max_attempts : int
        Hard cap on provider attempts within a single call. Default 5.
    temperature : float
        Sampling temperature. Default 0.1 (deterministic-ish).
    max_tokens : int
        Max completion tokens. Default 4096.
    seed : int | None
        If supported by the provider, passes as `seed`. Default 42.
    env : dict | None
        Env source for API keys. Defaults to os.environ.
    session : requests.Session | None
        Optional session for connection pooling and test injection.
    """

    def __init__(
        self,
        router: APIRouter,
        timeout_seconds: float = DEFAULT_CALL_TIMEOUT_SECONDS,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        seed: Optional[int] = 42,
        env: Optional[Dict[str, str]] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        if router is None:
            raise ValueError("router is required")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if not (0.0 <= temperature <= 2.0):
            raise ValueError("temperature must be in [0, 2]")
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")

        self._router = router
        self._timeout = timeout_seconds
        self._max_attempts = max_attempts
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._seed = seed
        self._env = env if env is not None else os.environ
        self._session = session or requests.Session()

        # Telemetry from the most recent successful call
        self._last_provider_name: Optional[str] = None
        self._last_usage: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ API

    @property
    def last_provider_name(self) -> Optional[str]:
        return self._last_provider_name

    @property
    def last_usage(self) -> Optional[Dict[str, Any]]:
        return self._last_usage

    def call(
        self,
        prompt: str,
        schema: Optional[dict] = None,
    ) -> Any:
        """
        Send `prompt` to the best available provider.

        Returns the message content. If the content is valid JSON,
        returns it as a parsed dict; otherwise returns the raw string.

        Raises APIUnavailable if all providers fail.
        """
        if not prompt or not prompt.strip():
            raise ValueError("prompt must be non-empty")

        excluded: List[str] = []
        last_error: Optional[str] = None

        for attempt in range(1, self._max_attempts + 1):
            # 1. Select provider
            try:
                provider = self._router.select(
                    requires_vision=False,
                    exclude=excluded,
                )
            except AllProvidersUnavailable as e:
                last_error = str(e)
                break

            # 2. Record request for rate-limit windows
            self._router.record_request(provider.name)

            # 3. Make the HTTP call
            try:
                response = self._post_to_provider(
                    provider=provider,
                    prompt=prompt,
                    schema=schema,
                )
            except requests.Timeout:
                self._router.record_failure(provider.name)
                excluded.append(provider.name)
                last_error = f"{provider.name}: timeout after {self._timeout}s"
                self._log_attempt(attempt, provider, "TIMEOUT")
                if attempt < self._max_attempts:
                    time.sleep(RETRY_DELAY_SECONDS)
                continue
            except requests.ConnectionError as e:
                self._router.record_failure(provider.name)
                excluded.append(provider.name)
                last_error = f"{provider.name}: connection error: {e}"
                self._log_attempt(attempt, provider, "CONN_ERR")
                if attempt < self._max_attempts:
                    time.sleep(RETRY_DELAY_SECONDS)
                continue
            except requests.RequestException as e:
                self._router.record_failure(provider.name)
                excluded.append(provider.name)
                last_error = f"{provider.name}: request error: {e}"
                self._log_attempt(attempt, provider, "REQ_ERR")
                if attempt < self._max_attempts:
                    time.sleep(RETRY_DELAY_SECONDS)
                continue

            # 4. Handle HTTP status
            if response.status_code == 200:
                try:
                    content, usage = self._parse_success_response(response)
                except requests.RequestException as e:
                    self._router.record_failure(provider.name)
                    excluded.append(provider.name)
                    last_error = f"{provider.name}: {e}"
                    self._log_attempt(attempt, provider, "BAD_RESPONSE")
                    if attempt < self._max_attempts:
                        time.sleep(RETRY_DELAY_SECONDS)
                    continue

                self._router.record_success(provider.name)
                self._last_provider_name = provider.name
                self._last_usage = usage
                self._log_attempt(attempt, provider, "OK")
                return self._maybe_parse_json(content)

            if response.status_code in TRANSIENT_STATUSES:
                self._router.record_failure(provider.name)
                excluded.append(provider.name)
                last_error = (
                    f"{provider.name}: HTTP {response.status_code} "
                    f"({response.text[:200]})"
                )
                self._log_attempt(attempt, provider, f"HTTP_{response.status_code}")
                if attempt < self._max_attempts:
                    time.sleep(RETRY_DELAY_SECONDS)
                continue

            if response.status_code in CONFIG_STATUSES:
                # Config error — this provider likely won't work regardless
                # of retries. Record and exclude, but do not sleep (it's not
                # a rate-limit issue).
                self._router.record_failure(provider.name)
                excluded.append(provider.name)
                last_error = (
                    f"{provider.name}: HTTP {response.status_code} (config) "
                    f"({response.text[:200]})"
                )
                self._log_attempt(attempt, provider, f"CONFIG_{response.status_code}")
                continue

            # Unknown status — treat as transient to be safe
            self._router.record_failure(provider.name)
            excluded.append(provider.name)
            last_error = (
                f"{provider.name}: HTTP {response.status_code} (unknown) "
                f"({response.text[:200]})"
            )
            self._log_attempt(attempt, provider, f"HTTP_{response.status_code}")
            if attempt < self._max_attempts:
                time.sleep(RETRY_DELAY_SECONDS)
            continue

        # All attempts exhausted
        raise APIUnavailable(
            f"all providers failed after {len(excluded)} attempt(s): {last_error}"
        )

    # -------------------------------------------------------------- Internals

    def _post_to_provider(
        self,
        provider: APIProvider,
        prompt: str,
        schema: Optional[dict],
    ) -> requests.Response:
        """Build the OpenAI-compatible request and POST it."""
        url = provider.base_url.rstrip("/") + "/chat/completions"
        api_key = self._env.get(provider.api_key_env, "").strip()

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if provider.name == "openrouter":
            headers["HTTP-Referer"] = OPENROUTER_REFERER

        body: Dict[str, Any] = {
            "model": provider.best_text_model(),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }

        # Optional: seed for reproducibility (OpenAI-compatible)
        if self._seed is not None:
            body["seed"] = self._seed

        # Optional: structured output. Use json_object universally for
        # compatibility; the schema itself lives in the prompt.
        if schema is not None:
            body["response_format"] = {"type": "json_object"}

        return self._session.post(
            url,
            headers=headers,
            json=body,
            timeout=self._timeout,
        )

    def _parse_success_response(
        self, response: requests.Response
    ) -> tuple:
        """
        Extract (content, usage) from a 200 OK response.

        Content is the assistant message string. Usage is the token
        usage block if present.
        """
        try:
            body = response.json()
        except (ValueError, json.JSONDecodeError) as e:
            raise requests.RequestException(
                f"provider returned non-JSON body: {e}"
            )

        choices = body.get("choices") or []
        if not choices:
            raise requests.RequestException(
                f"provider returned no choices: {body}"
            )

        message = choices[0].get("message") or {}
        content = message.get("content")
        if content is None:
            raise requests.RequestException(
                f"provider returned no content: {choices[0]}"
            )

        usage = body.get("usage") if isinstance(body.get("usage"), dict) else None
        return str(content), usage

    def _maybe_parse_json(self, content: str) -> Any:
        """
        Try to parse the content as JSON. Return the dict if valid,
        else return the raw string.

        Handles markdown code fences around JSON.
        """
        import re
        text = content.strip()
        # Strip ```json ... ``` fences
        fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
        if fence:
            text = fence.group(1).strip()
        # Extract first { ... } block if surrounded by prose
        if not text.startswith("{") and not text.startswith("["):
            m = re.search(r"[\{\[].*[\}\]]", text, re.DOTALL)
            if m:
                text = m.group(0)
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return content

    def _log_attempt(self, attempt: int, provider: APIProvider, status: str) -> None:
        print(
            f"[HTTP-API] attempt={attempt} provider={provider.name} status={status}",
            flush=True,
        )
