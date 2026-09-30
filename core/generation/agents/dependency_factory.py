"""
Dependency factory for the v3 agent pipeline.

Constructs the five agents with their real callers:
    - PlanningAgent    : no external deps (KG built from artifact)
    - WritingAgent     : LLM caller (required), VLM caller (optional)
    - EvaluationAgent  : API caller (optional; degrades if absent)
    - RefinementAgent  : LLM caller (shared), SymPy verifier
    - CheckingAgent    : no external deps

All construction failures are reported via FactoryResult so the
pipeline bridge can fall back to the legacy path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from core.knowledge import KnowledgeGraphBuilder
from core.api import APIRouter, PROVIDERS
from .checking_agent import CheckingAgent
from .evaluation_agent import EvaluationAgent
from .planning_agent import PlanningAgent
from .refinement_agent import RefinementAgent
from .writing_agent import WritingAgent


# -----------------------------------------------------------------------------
# Factory result
# -----------------------------------------------------------------------------


@dataclass
class FactoryResult:
    success: bool
    planning: Optional[PlanningAgent] = None
    writing: Optional[WritingAgent] = None
    evaluation: Optional[EvaluationAgent] = None
    refinement: Optional[RefinementAgent] = None
    checking: Optional[CheckingAgent] = None
    failure_code: Optional[str] = None
    failure_detail: Optional[str] = None
    notes: list = field(default_factory=list)


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------


class DependencyFactory:
    """
    Build the agent graph from the current environment.

    Parameters are callables so the factory is testable without touching
    real LLMs or APIs.
    """

    def __init__(
        self,
        llm_caller_factory: Callable[[], Any],
        vlm_caller_factory: Optional[Callable[[], Any]] = None,
        api_caller_factory: Optional[Callable[[], Any]] = None,
        sympy_verifier_factory: Optional[Callable[[], Any]] = None,
        knowledge_graph_builder_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self._make_llm = llm_caller_factory
        self._make_vlm = vlm_caller_factory
        self._make_api = api_caller_factory
        self._make_sympy = sympy_verifier_factory
        self._make_kg_builder = knowledge_graph_builder_factory or KnowledgeGraphBuilder

    def build(self) -> FactoryResult:
        notes: list = []

        # --- LLM caller (required for Writing) ---
        try:
            llm_caller = self._make_llm()
        except Exception as e:
            return FactoryResult(
                success=False,
                failure_code="LLM_CALLER_UNAVAILABLE",
                failure_detail=f"failed to construct LLM caller: {e}",
            )
        if llm_caller is None:
            return FactoryResult(
                success=False,
                failure_code="LLM_CALLER_UNAVAILABLE",
                failure_detail="LLM caller factory returned None",
            )

        # --- VLM caller (optional) ---
        vlm_caller = None
        if self._make_vlm is not None:
            try:
                vlm_caller = self._make_vlm()
                if vlm_caller is None:
                    notes.append("vlm_caller factory returned None; visual slots will use text LLM")
            except Exception as e:
                notes.append(f"vlm_caller unavailable: {e}")

        # --- API caller (optional) ---
        api_caller = None
        if self._make_api is not None:
            try:
                api_caller = self._make_api()
                if api_caller is None:
                    notes.append("api_caller factory returned None; evaluation will degrade")
            except Exception as e:
                notes.append(f"api_caller unavailable: {e}")
        else:
            notes.append("no api_caller_factory provided; evaluation will degrade")

        # --- SymPy verifier (optional) ---
        sympy_verifier = None
        if self._make_sympy is not None:
            try:
                sympy_verifier = self._make_sympy()
                if sympy_verifier is None:
                    notes.append("sympy_verifier factory returned None; numerical repairs unverified")
            except Exception as e:
                notes.append(f"sympy_verifier unavailable: {e}")
        else:
            notes.append("no sympy_verifier_factory provided; numerical repairs unverified")

        # --- Build agents ---
        planning = PlanningAgent()
        writing = WritingAgent(llm_caller=llm_caller, vlm_caller=vlm_caller)
        evaluation = EvaluationAgent(api_caller=api_caller) if api_caller else None
        refinement = RefinementAgent(
            llm_caller=llm_caller if (evaluation is not None) else None,
            sympy_verifier=sympy_verifier,
        )
        checking = CheckingAgent()

        return FactoryResult(
            success=True,
            planning=planning,
            writing=writing,
            evaluation=evaluation,
            refinement=refinement,
            checking=checking,
            notes=notes,
        )


# -----------------------------------------------------------------------------
# Real caller constructors (used by the pipeline bridge, not by tests)
# -----------------------------------------------------------------------------


class RobustLLMCallerAdapter:
    """Adapter bridging RobustLLMCaller to the v3 LLMCaller protocol."""

    def __init__(self, caller: Optional[Any] = None) -> None:
        from core.generation.robust_llm_caller import RobustLLMCaller
        self._caller = caller or RobustLLMCaller()

    def call(
        self,
        prompt: str,
        schema: Optional[dict] = None,
        image_path: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Any:
        from core.generation.robust_llm_caller import LLMRequest
        from core.config.production_model import get_production_model
        model = os.getenv("AION_MODEL") or get_production_model()
        backend = (os.getenv("AION_BACKEND") or getattr(self._caller, "backend", "") or "").lower().strip()
        if backend == "vllm" and (not model or model == "qwen2.5:14b" or ":" in model):
            model = os.getenv("AION_VLLM_MODEL") or "Qwen/Qwen2.5-14B-Instruct-AWQ"
        timeout = int(os.getenv("AION_LLM_TIMEOUT", "180"))
        req = LLMRequest(
            model=model,
            prompt=prompt,
            schema=schema,
            seed=seed or 42,
            timeout_sec=timeout,
        )
        res = self._caller.call(req)
        if not res.success:
            if res.text:
                return res.text
            raise RuntimeError(res.error or "LLM call failed")
        return res.parsed if res.parsed is not None else res.text


def default_llm_caller_factory():
    """Construct the real RobustLLMCaller adapter from environment."""
    return RobustLLMCallerAdapter()


def default_vlm_caller_factory():
    """
    Construct the VLM caller if AION_VLM_URL is set.

    Returns None if:
        - AION_VLM_URL is unset (no VLM endpoint configured)
        - The configured URL is unreachable at construction time

    When None, the Writing Agent routes visual slots to the text LLM.
    The visual directive is still in the prompt, but the model cannot
    see the figure — it will produce a question that references the
    figure by caption, not by content.
    """
    from core.api import VLMCaller

    vlm_url = os.getenv("AION_VLM_URL", "").strip()
    if not vlm_url:
        return None

    vlm_model = os.getenv(
        "AION_VLM_MODEL",
        "Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
    ).strip()

    try:
        return VLMCaller(
            base_url=vlm_url,
            model=vlm_model,
            api_key=os.getenv("AION_VLM_API_KEY", "").strip() or None,
            timeout_seconds=float(os.getenv("AION_VLM_TIMEOUT", "60")),
            max_attempts=int(os.getenv("AION_VLM_MAX_ATTEMPTS", "3")),
            seed=int(os.getenv("AION_SEED", "42")),
        )
    except Exception as e:
        print(f"[DEP-FACTORY] VLM caller construction failed: {e}", flush=True)
        return None


def default_api_caller_factory():
    """
    Construct the API caller that wraps the router with a real HTTP client.
    Returns None if no provider keys are present.
    """
    from core.api.http_caller import HTTPAPICaller

    router = APIRouter(providers=PROVIDERS)
    return HTTPAPICaller(router=router)


def default_sympy_verifier_factory():
    """Wrap the existing numerical engine as a SymPy verifier."""
    try:
        from core.numerical_engine import NumericalVerifier
        return NumericalVerifier()
    except ImportError:
        return None
