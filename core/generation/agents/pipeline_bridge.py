"""
Pipeline bridge — the single entry point for v3.

Callers use `run_v3_pipeline()` if they want to opt into v3 unconditionally.
Callers use `run_pipeline_v3_or_legacy()` if they want flag-driven dispatch.

The v3 pipeline:
    1. Builds the KG from the artifact
    2. Constructs the agent graph via DependencyFactory
    3. Runs AgentOrchestrator
    4. Converts the OrchestratorResult to legacy shape
    5. Returns `(paper_parts, full_paper)`

On any unexpected failure, returns a structured failure that callers
can detect via an empty `paper_parts` + a populated `orchestrator_meta`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

from .agent_orchestrator import AgentOrchestrator
from .base import AgentContext
from .dependency_factory import (
    DependencyFactory,
    default_api_caller_factory,
    default_llm_caller_factory,
    default_sympy_verifier_factory,
    default_vlm_caller_factory,
    FactoryResult,
)
from .legacy_adapter import to_legacy_shape
from .orchestrator_contracts import OrchestratorResult


# -----------------------------------------------------------------------------
# Feature flag
# -----------------------------------------------------------------------------


V3_FLAG_ENV = "AION_ENABLE_V3_AGENTS"


def v3_enabled() -> bool:
    val = os.getenv(V3_FLAG_ENV, "false").strip().lower()
    return val in ("true", "1", "yes", "on")


# -----------------------------------------------------------------------------
# Public entry points
# -----------------------------------------------------------------------------


def run_v3_pipeline(
    *,
    paper_spec: Any,
    artifact: Any,
    marks_split: List[List[int]],
    knowledge_graph: Any = None,
    request: Optional[dict] = None,
    dependency_factory: Optional[DependencyFactory] = None,
) -> Tuple[List[dict], List[dict], dict]:
    """
    Run the v3 pipeline. Returns (paper_parts, full_paper, orchestrator_meta).

    The third element is a diagnostic dict — callers that don't care can
    ignore it. Callers integrating with the API can use it to populate
    the QA report and telemetry.
    """
    if dependency_factory is None:
        dependency_factory = DependencyFactory(
            llm_caller_factory=default_llm_caller_factory,
            vlm_caller_factory=default_vlm_caller_factory,
            api_caller_factory=default_api_caller_factory,
            sympy_verifier_factory=default_sympy_verifier_factory,
        )

    factory = dependency_factory.build()
    if not factory.success:
        return [], [], {
            "success": False,
            "failure_code": factory.failure_code,
            "failure_detail": factory.failure_detail,
            "notes": factory.notes,
        }

    # Build KnowledgeGraph if not provided
    if knowledge_graph is None and artifact is not None:
        knowledge_graph = _build_kg_for_artifact(artifact, paper_spec)

    # Construct the orchestrator
    orchestrator = AgentOrchestrator(
        planning_agent=factory.planning,
        writing_agent=factory.writing,
        evaluation_agent=factory.evaluation,
        refinement_agent=factory.refinement,
        checking_agent=factory.checking,
    )

    # Build the context
    context = AgentContext(
        request=request or {},
        paper_spec=paper_spec,
        artifact=artifact,
        knowledge_graph=knowledge_graph,
        marks_split=list(marks_split or []),
    )

    result: OrchestratorResult = orchestrator.run(context)

    # Convert to legacy shape
    paper_parts, full_paper = to_legacy_shape(result, context)

    meta = {
        "success": result.success,
        "paper_status": (
            getattr(result.qa_report, "status", None) if result.qa_report else None
        ),
        "failure_code": result.failure_code,
        "failure_detail": result.failure_detail,
        "telemetry": dict(result.telemetry or {}),
        "factory_notes": list(factory.notes),
        "qa_report": (
            result.qa_report.to_dict()
            if result.qa_report and hasattr(result.qa_report, "to_dict")
            else None
        ),
    }

    return paper_parts, full_paper, meta


def run_pipeline_v3_or_legacy(
    *,
    legacy_runner: Callable[..., Tuple[List[dict], List[dict]]],
    v3_kwargs: dict,
    legacy_kwargs: dict,
) -> Tuple[List[dict], List[dict]]:
    """
    Flag-driven dispatch.

    When AION_ENABLE_V3_AGENTS=true:
        Runs the v3 pipeline. On v3 failure, logs and returns empty
        paper_parts + empty full_paper (fail-closed — the caller's
        export gate will block).

    When flag is false:
        Calls legacy_runner(**legacy_kwargs) — identical to pre-v3 behavior.
    """
    if not v3_enabled():
        return legacy_runner(**legacy_kwargs)

    paper_parts, full_paper, meta = run_v3_pipeline(**v3_kwargs)

    if not meta["success"]:
        print(
            f"[V3-BRIDGE] v3 failed: {meta.get('failure_code')} — "
            f"{meta.get('failure_detail')}",
            flush=True,
        )
    else:
        status = meta.get("paper_status")
        print(
            f"[V3-BRIDGE] v3 completed: status={status} "
            f"questions={len(full_paper)}",
            flush=True,
        )

    # Attach meta to paper_parts so downstream can inspect it
    if paper_parts:
        paper_parts[0]["_orchestrator_meta"] = meta

    return paper_parts, full_paper


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _build_kg_for_artifact(artifact: Any, paper_spec: Any) -> Any:
    """
    Build a KnowledgeGraph by partitioning the artifact into modules.

    Uses per-module page ranges derived from the artifact's page count
    and the paper spec's module count.
    """
    try:
        from core.knowledge import KnowledgeGraphBuilder
    except ImportError:
        return None

    max_page = max(
        (getattr(b, "page", 0) for b in getattr(artifact, "text_blocks", []) or []),
        default=0,
    )
    if max_page <= 0 or paper_spec is None:
        return None

    module_count = getattr(paper_spec, "module_count", 0)
    if module_count <= 0:
        return None

    pages_per_module = max(1, max_page // module_count)
    ranges = {}
    for m in range(1, module_count + 1):
        start = (m - 1) * pages_per_module + 1
        end = max_page if m == module_count else m * pages_per_module
        ranges[f"module_{m}"] = (start, end)

    try:
        builder = KnowledgeGraphBuilder()
        return builder.build_all(artifact=artifact, module_page_ranges=ranges)
    except Exception as e:
        print(f"[V3-BRIDGE] KG build failed: {e}", flush=True)
        return None
