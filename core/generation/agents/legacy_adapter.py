"""
Convert v3 OrchestratorResult into the legacy run_pipeline return shape.

The legacy pipeline returns `(paper_parts, full_paper)`:
    paper_parts : List[dict]  — one entry per module, with questions
    full_paper  : List[dict]  — flat list of questions across all modules

Each question is a dict with (at minimum) these keys:
    slot_id, question_text, solution, marking_scheme,
    marks, bloom, co, module_id,
    image_path, figure_caption, table_data, diagram_request

This adapter is the ONLY place that knows about the legacy shape.
If the shape changes, this file changes. Agents do not.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple

from .generated_question import GeneratedQuestion
from .orchestrator_contracts import OrchestratorResult


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def to_legacy_shape(
    result: OrchestratorResult,
    context: Any,
) -> Tuple[List[dict], List[dict]]:
    """
    Return `(paper_parts, full_paper)` matching the legacy pipeline.

    paper_parts : one entry per module, with module metadata + questions
    full_paper  : flat list of all question dicts (used by DOCX export)
    """
    if not result.success:
        # Legacy shape on failure: empty outputs.
        return [], []

    questions = list(result.questions or [])
    paper_spec = getattr(context, "paper_spec", None)
    artifact = getattr(context, "artifact", None)

    # Build flat list of question dicts
    full_paper: List[dict] = [_to_question_dict(q) for q in questions]

    # Group by module
    by_module: Dict[str, List[dict]] = defaultdict(list)
    for q_dict in full_paper:
        by_module[q_dict["module_id"]].append(q_dict)

    # Build paper_parts
    paper_parts: List[dict] = []
    if paper_spec is not None and hasattr(paper_spec, "module_count"):
        for module_idx in range(1, paper_spec.module_count + 1):
            module_id = f"module_{module_idx}"
            module_questions = by_module.get(module_id, [])
            paper_parts.append({
                "module_id": module_id,
                "module_index": module_idx,
                "module_title": _module_title(artifact, module_idx),
                "questions": module_questions,
                "question_count": len(module_questions),
            })
    else:
        # No spec: emit one part per module_id we saw.
        for module_id, module_questions in sorted(by_module.items()):
            paper_parts.append({
                "module_id": module_id,
                "module_index": _module_idx_from_id(module_id),
                "module_title": "",
                "questions": module_questions,
                "question_count": len(module_questions),
            })

    # Attach QA report to both for downstream consumers
    qa = _qa_dict(result.qa_report)
    for part in paper_parts:
        part["qa_report"] = qa

    return paper_parts, full_paper


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _to_question_dict(q: GeneratedQuestion) -> dict:
    """Convert one GeneratedQuestion to the legacy question dict."""
    return {
        "slot_id": q.slot_id,
        "module_id": q.module_id,
        "global_q_idx": q.global_q_idx,
        "question_text": q.question_text,
        "solution": q.solution,
        "marking_scheme": list(q.marking_scheme or []),
        "marks": q.marks,
        "partition": list(q.partition or []),
        "bloom": q.bloom,
        "co": q.co,
        "topic": q.topic,
        "image_path": q.image_path,
        "figure_caption": q.figure_caption,
        "references_image": q.references_image,
        "diagram_request": q.diagram_request,
        "generation_source": q.generation_source,
        "attempts": q.attempts,
        "telemetry": dict(q.telemetry or {}),
    }


def _module_title(artifact: Any, module_idx: int) -> str:
    """Extract a module title from the artifact, if available."""
    if artifact is None:
        return f"Module {module_idx}"
    titles = getattr(artifact, "module_titles", None)
    if isinstance(titles, list) and 1 <= module_idx <= len(titles):
        return titles[module_idx - 1]
    return f"Module {module_idx}"


def _module_idx_from_id(module_id: str) -> int:
    try:
        return int(module_id.split("_")[1])
    except (IndexError, ValueError):
        return 0


def _qa_dict(qa_report: Any) -> dict:
    if qa_report is None:
        return {}
    if hasattr(qa_report, "to_dict"):
        return qa_report.to_dict()
    return {}
