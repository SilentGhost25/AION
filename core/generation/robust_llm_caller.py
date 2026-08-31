# core/generation/robust_llm_caller.py

import time, json, logging, re
from dataclasses import dataclass
from typing import Optional, Dict, Any

LOG = logging.getLogger("aion.llm")


@dataclass
class LLMRequest:
    model         : str
    prompt        : str
    schema        : Optional[dict] = None
    temperature   : float = 0.55
    seed          : int = 42
    timeout_sec   : int = 45


@dataclass
class LLMResponse:
    success     : bool
    text        : Optional[str]
    parsed      : Optional[dict]
    elapsed_sec : float
    error       : Optional[str] = None
    timed_out   : bool = False


class RobustLLMCaller:
    """
    ONE inference request per call.
    Does NOT own retry logic — SlotOrchestrator owns all retries.
    """

    def __init__(self, backend: str = "ollama", host: str = "http://localhost:11434"):
        self.backend = backend
        self.host    = host

    def call(self, request: LLMRequest) -> LLMResponse:
        start = time.monotonic()
        try:
            if self.backend == "ollama":
                return self._call_ollama(request, start)
            raise ValueError(f"Unknown backend: {self.backend}")
        except TimeoutError:
            elapsed = time.monotonic() - start
            LOG.warning(f"[LLM] Timeout after {elapsed:.1f}s for {request.model}")
            return LLMResponse(
                success=False, text=None, parsed=None,
                elapsed_sec=elapsed, timed_out=True,
                error="LLM_TIMEOUT"
            )
        except Exception as e:
            elapsed = time.monotonic() - start
            LOG.error(f"[LLM] Error: {e}")
            return LLMResponse(
                success=False, text=None, parsed=None,
                elapsed_sec=elapsed, error=str(e)
            )

    def _call_ollama(self, request: LLMRequest, start: float) -> LLMResponse:
        import requests as req
        payload: Dict[str, Any] = {
            "model"  : request.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "stream"  : False,
            "options" : {
                "temperature": request.temperature,
                "seed"       : request.seed,
                "num_ctx"    : 8192,
            }
        }
        if request.schema:
            payload["format"] = request.schema

        try:
            response = req.post(
                f"{self.host}/api/chat",
                json=payload,
                timeout=request.timeout_sec,
            )
        except req.exceptions.Timeout:
            raise TimeoutError(f"Request to Ollama timed out after {request.timeout_sec}s")

        elapsed = time.monotonic() - start

        if not response.ok:
            return LLMResponse(
                success=False, text=None, parsed=None,
                elapsed_sec=elapsed, error=f"HTTP {response.status_code}"
            )

        resp_data = response.json()
        raw_text = resp_data.get("message", {}).get("content", "") or resp_data.get("response", "")

        # Extract JSON from response
        try:
            match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            parsed = json.loads(match.group()) if match else json.loads(raw_text)
            return LLMResponse(
                success=True, text=raw_text, parsed=parsed, elapsed_sec=elapsed
            )
        except json.JSONDecodeError as e:
            return LLMResponse(
                success=False, text=raw_text, parsed=None,
                elapsed_sec=elapsed, error=f"JSON_PARSE_ERROR: {e}"
            )
