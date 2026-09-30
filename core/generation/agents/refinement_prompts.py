"""
Prompt construction for the LLM repair tier.

The prompt gives the LLM:
    - The original question
    - Its solution and marking scheme
    - The failure code and the evaluator's suggested fix
    - The topic + evidence context (bounded)
    - A strict output schema matching GeneratedQuestion fields

The LLM is instructed to change only what's necessary, preserving
any content that was correct.
"""

from __future__ import annotations

from typing import Optional


MAX_ORIGINAL_QUESTION = 1000
MAX_ORIGINAL_SOLUTION = 500
MAX_EVIDENCE_SUMMARY = 300


OUTPUT_SCHEMA = """{
  "question_text": "string — the corrected question, self-contained",
  "solution": "string — the corrected model answer",
  "marking_scheme": [
    {"criterion": "string", "marks": integer}
  ],
  "diagram_request": null | {"diagram_type": "string", "description": "string"},
  "references_image": boolean,
  "change_summary": "string — brief note on what was changed"
}"""


# -----------------------------------------------------------------------------
# Reason-code-specific guidance
# -----------------------------------------------------------------------------


REASON_GUIDANCE = {
    "FACTUAL_ERROR": (
        "The original question contains a factual error. Correct only the "
        "incorrect factual claim. Preserve everything else."
    ),
    "UNGROUNDED": (
        "The original question asks about material not present in the "
        "evidence. Rephrase to ask about what the evidence actually supports."
    ),
    "CIRCULAR": (
        "The original question defines a term and then asks the student to "
        "recall that definition. Change the question so it asks for "
        "application, analysis, or derivation instead of recall."
    ),
    "OFF_TOPIC": (
        "The original question drifted from the assigned module topic. "
        "Rewrite to align with the target topic without losing difficulty."
    ),
    "VERB_TASK_MISMATCH": (
        "The question's first verb does not match the task it asks. "
        "Choose a Bloom verb appropriate to the task, and adjust the "
        "task to match the verb if needed."
    ),
    "BLOOM_MISMATCH": (
        "The question's first verb does not match the required Bloom level. "
        "Start the question with an approved verb from the allowed list."
    ),
    "MULTI_PART_STRUCTURE_MISMATCH": (
        "The question must be structured as distinct sub-parts matching "
        "the partition. Add explicit (i)/(ii) or (a)/(b) labels."
    ),
}


# -----------------------------------------------------------------------------
# Prompt builder
# -----------------------------------------------------------------------------


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def build_repair_prompt(
    question,
    reason_code: str,
    detail: str,
    suggested_fix: Optional[str],
    *,
    strict_json: bool = False,
) -> str:
    """
    Compose the repair prompt for a single question.

    strict_json: if True, adds a stronger directive to return only JSON.
    """
    guidance = REASON_GUIDANCE.get(
        reason_code,
        "Fix the flagged issue without changing other content.",
    )

    marks_display = (
        " + ".join(str(m) for m in question.partition)
        if question.partition else str(question.marks)
    )

    sections = [
        "[REPAIR TASK]",
        f"Reason code: {reason_code}",
        f"Evaluator's detail: {detail}",
    ]

    if suggested_fix:
        sections.append(f"Suggested fix: {suggested_fix}")

    sections.extend([
        "",
        "[GUIDANCE]",
        guidance,
        "Preserve all content that is not directly related to the failure.",
        "Do not invent new facts, do not change the difficulty level.",
        "",
        "[SLOT CONTRACT]",
        f"Marks: {question.marks} ({marks_display})",
        f"Bloom: {question.bloom}",
        f"Course Outcome: {question.co}",
        f"Module: {question.module_id}",
        f"Topic: {question.topic}",
        "",
        "[ORIGINAL QUESTION]",
        _truncate(question.question_text, MAX_ORIGINAL_QUESTION),
        "",
        "[ORIGINAL SOLUTION]",
        _truncate(question.solution, MAX_ORIGINAL_SOLUTION),
        "",
        "[OUTPUT SCHEMA]",
        OUTPUT_SCHEMA,
    ])

    if strict_json:
        sections.append(
            "\n[IMPORTANT]\n"
            "Return ONLY the JSON object. Do not include any commentary."
        )

    return "\n".join(sections)
