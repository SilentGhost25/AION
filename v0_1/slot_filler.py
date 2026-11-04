"""
AION Slot Filler
================
Fills question specs using LLM prompt formatting or fallback templates.
"""

from typing import Any


def fill_slot(spec: Any, context: str, subject: str) -> str:
    try:
        from .generator import get_vtu_vibe_question
        texts = getattr(getattr(spec, "evidence", None), "texts", None)
        if not texts and context:
            texts = [context]

        part_idx = (ord(str(getattr(spec, "part_letter", "a")).lower()[0]) - ord('a')) if getattr(spec, "part_letter", None) else 0
        selected_chunk = texts[part_idx % len(texts)] if texts else (context or "")

        bloom_val = getattr(spec, "bloom_level", 2)
        if isinstance(bloom_val, str):
            bloom_val = int(bloom_val.replace("L", "")) if "L" in bloom_val else int(bloom_val)
        marks = getattr(spec, "marks", 10)

        q_text = get_vtu_vibe_question(
            chunk=selected_chunk,
            marks=marks,
            bloom=bloom_val,
        )
        if q_text and len(q_text.strip()) > 10 and "given concept" not in q_text:
            q_clean = q_text.strip()
            if f"{marks} Marks" not in q_clean:
                q_clean = f"{q_clean} ({marks} Marks)"
            return q_clean
    except Exception:
        pass

    verb = getattr(spec, "bloom_verb", "Explain")
    marks = getattr(spec, "marks", 10)
    part_letter = str(getattr(spec, "part_letter", "a")).lower()

    # Extract distinct topic/concept within the same module
    evidence_query = getattr(getattr(spec, "evidence", None), "query", "")
    topic = evidence_query or subject or "the system"

    # Distinct pedagogical angle per subpart to prevent duplicate/near-identical subquestions
    aspects = {
        "a": "the architectural structure and fundamental operating principles",
        "b": "the operational characteristics, performance trade-offs, and design constraints",
        "c": "the implementation mechanisms and real-world deployment challenges",
        "d": "the comparative analysis and technical evaluation parameters",
    }
    aspect = aspects.get(part_letter, "the key functional concepts and operational details")
    return f"{verb} {aspect} of {topic} in detail. ({marks} Marks)"
