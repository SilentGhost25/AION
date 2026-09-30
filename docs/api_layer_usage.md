# API Layer Usage (Phase 1, Task 1)

The `core/api/` package provides provider abstraction, circuit breaking,
and rate-limit-aware routing. It does not make network calls.

## When to use it

Phase 4 (API escalation) will use it like this:

```python
from core.api import APIRouter, PROVIDERS, AllProvidersUnavailable

router = APIRouter(providers=PROVIDERS)

def escalated_generate(prompt, requires_vision=False):
    excluded = []
    while True:
        try:
            provider = router.select(
                requires_vision=requires_vision,
                exclude=excluded,
            )
        except AllProvidersUnavailable:
            raise  # caller decides: UNRESOLVED or fall back to local

        router.record_request(provider.name)
        try:
            result = call_provider(provider, prompt)
            router.record_success(provider.name)
            return result
        except TransientProviderError:
            router.record_failure(provider.name)
            excluded.append(provider.name)
        except FatalProviderError:
            # Malformed request — do not retry, do not penalize circuit
            raise
```

## What is not yet wired

- No pipeline code calls `APIRouter` in Phase 1.
- No `api_key` is read at import time. The router checks env vars on every `select()` call.
- No HTTP client is included. The caller supplies the transport.

## Testing

```
pytest tests/unit/test_circuit_breaker.py -v
pytest tests/unit/test_router.py -v
```

Both suites are fast (< 1 second) and use a `FakeClock` for determinism.
