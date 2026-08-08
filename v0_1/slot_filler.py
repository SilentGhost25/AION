"""
AION Slot Filler
================
Fills question specs using LLM prompt formatting or fallback templates.
"""

from typing import Any


def fill_slot(spec: Any, context: str, subject: str) -> str:
    try:
        from .generator import get_vtu_vibe_question
        q_text = get_vtu_vibe_question(
            module_content=context,
            bloom_level=spec.bloom_level,
            question_type="conceptual",
            num_questions=1,
        )
        if q_text and len(q_text.strip()) > 10:
            return q_text.strip()
    except Exception:
        pass

    verb  = getattr(spec, "bloom_verb", "Explain")
    marks = getattr(spec, "marks", 10)
    subj  = subject or "the concept"
    return f"{verb} the fundamental principles of {subj} in detail. ({marks} Marks)"
