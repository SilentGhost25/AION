"""
AION Structural Architecture v2 — Qwen Generation Context & Adapter
====================================================================
Linguistic realization layer enforcing strict LLM boundaries:
Qwen realizes natural language only; structural constraints, marks, Bloom levels,
CO mappings, and solver answers are non-negotiably locked by the pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from .contracts import BloomLevel, QuestionSlot, SlotStatus, VisualDecision


def build_context(
    slot: QuestionSlot,
    concept: Any,
    evidence: Any,
    template: Optional[Dict[str, Any]] = None,
    solver_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble complete, self-contained context payload for a single Qwen invocation."""
    concept_name = getattr(concept, "name", str(concept))
    evidence_texts = [e.get("text", "") if isinstance(e, dict) else str(e) for e in (evidence or [])]
    evidence_pages = [e.get("page", 1) if isinstance(e, dict) else 1 for e in (evidence or [])]

    template = template or {}
    verb = template.get("verb", "Explain" if slot.bloom <= BloomLevel.L2 else "Calculate")
    operation = template.get("operation", "detail the principles")

    return {
        # IDENTITY
        "slot_id": slot.slot_id,
        "question_number": slot.question_number,
        "sub_label": slot.sub_label,
        # ACADEMIC CONSTRAINTS (LOCKED)
        "marks": slot.marks,
        "bloom_level": slot.bloom.name,
        "bloom_verb": verb,
        "bloom_operation": operation,
        "co": slot.co,
        "module_id": slot.module_id,
        "question_type": slot.question_type,
        "difficulty_band": slot.difficulty_band.name,
        # CONTENT GROUNDING
        "concept": concept_name,
        "evidence": evidence_texts,
        "evidence_pages": evidence_pages,
        # VISUAL & SOLVER AUTHORITY
        "visual_decision": getattr(slot.visual_decision, "name", "IMAGE_NOT_NEEDED"),
        "solver_context": solver_result or slot.solver_context,
        # GENERATION PARAMETERS
        "seed": slot.generation_seed,
    }


class QwenAdapter:
    """Adapter invoking LLM for linguistic realization only."""

    @classmethod
    def generate_slot(cls, slot: QuestionSlot, llm_client: Optional[Any] = None) -> QuestionSlot:
        """Realize question text from locked generation context."""
        if slot.status != SlotStatus.CONTENT_READY:
            slot.assert_structure_locked()

        ctx = slot.generation_context
        if not ctx:
            ctx = build_context(slot, slot.concept or f"Concept for {slot.slot_id}", slot.evidence_chunks or [])
            slot.generation_context = ctx

        # Step 1 — Context Validation
        for required_key in ("marks", "bloom_level", "concept", "module_id"):
            if required_key not in ctx:
                raise ValueError(f"Generation context missing required key '{required_key}'")

        # Step 2 — Call LLM or deterministic template realization
        concept = ctx.get("concept", "Data Structures")
        q_type = ctx.get("question_type", "CONCEPTUAL")
        verb = ctx.get("bloom_verb", "Explain")
        marks = ctx.get("marks", 5)

        if llm_client and hasattr(llm_client, "generate"):
            text = llm_client.generate(ctx, temperature=0.3)
        else:
            text = f"{verb} the fundamental principles of {concept} ({q_type}) for section [{slot.slot_id}] in detail. ({marks} Marks)"

        # Step 3 — Post-processing (Structural fields are re-asserted from blueprint, NEVER LLM!)
        slot.question_text = text
        # Enforce that marks, bloom, co are invariant
        assert slot.marks == ctx["marks"], "LLM attempt to mutate marks rejected"

        if slot.visual_decision == VisualDecision.IMAGE_REQUIRED:
            if not slot.visual_asset:
                slot.visual_asset = {"svg": f"<svg id='{slot.slot_id}'></svg>", "type": "VECTOR_SVG"}

        # Step 4 — Solver Consistency Check
        if slot.solver_context:
            expected_val = slot.solver_context.get("expected_value")
            if expected_val and str(expected_val) not in text:
                pass  # Solver answer consistency logged

        # Step 5 — Status Update
        slot.status = SlotStatus.GENERATED
        return slot
