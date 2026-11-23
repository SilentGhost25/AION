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

from typing import Optional, Tuple
import re


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
    affected_subparts: Optional[Tuple[str, ...]] = None,
    strict_json: bool = False,
) -> str:
    """
    Compose the repair prompt for a question.
    When a single subpart is affected, generates an isolated subpart repair prompt
    with preserved sibling subparts marked strictly read-only.
    When whole question or no subparts exist, generates full question repair prompt.

    strict_json: if True, adds a stronger directive to return only JSON.
    """
    guidance = REASON_GUIDANCE.get(
        reason_code,
        "Fix the flagged issue without changing other content.",
    )

    # Check if this can be isolated to a single subpart
    target_label = None
    if getattr(question, "sub_parts", None):
        if affected_subparts and len(affected_subparts) == 1:
            target_label = affected_subparts[0].lower()
        elif not affected_subparts:
            m = re.findall(r"SubPart\s*\(([a-z])\)", detail, re.IGNORECASE)
            if len(m) == 1:
                target_label = m[0].lower()

    if target_label and getattr(question, "sub_parts", None):
        target_sp = next((sp for sp in question.sub_parts if sp.label.lower() == target_label), None)
        if target_sp is not None:
            siblings = [
                f"({sp.label}) [{sp.marks} marks] {sp.text}"
                for sp in question.sub_parts
                if sp.label.lower() != target_label
            ]
            siblings_display = "\n\n".join(siblings) if siblings else "None"

            sections = [
                "[SUBPART REPAIR TASK]",
                f"Target subpart: ({target_label})",
                f"Assigned marks: {target_sp.marks}",
                f"Topic: {target_sp.topic or question.topic}",
                f"Reason code: {reason_code}",
                f"Evaluator's detail: {detail}",
            ]

            if suggested_fix:
                sections.append(f"Suggested fix: {suggested_fix}")

            sections.extend([
                "",
                "[REPAIR INSTRUCTIONS]",
                guidance,
                f"1. Repair ONLY subpart ({target_label}).",
                f"2. The criteria in 'marking_scheme' MUST sum exactly to {target_sp.marks} marks.",
                f"3. Return self-contained question text for subpart ({target_label}) WITHOUT any leading label marker like '({target_label})'.",
                "4. Do NOT duplicate, alter, re-phrase, or cross-reference the preserved sibling subparts shown below.",
                "",
                "[PRESERVED SIBLING SUBPARTS (READ-ONLY - DO NOT MODIFY OR DUPLICATE)]",
                siblings_display,
                "",
                f"[CURRENT SUBPART ({target_label})]",
                f"Text: {_truncate(target_sp.text, MAX_ORIGINAL_QUESTION)}",
                f"Solution: {_truncate(target_sp.solution, MAX_ORIGINAL_SOLUTION)}",
                f"Marks: {target_sp.marks}",
                "",
                "[OUTPUT SCHEMA]",
                """{
  "label": "%s",
  "text": "string — corrected self-contained question text for subpart (%s)",
  "solution": "string — corrected model answer for subpart (%s)",
  "marking_scheme": [
    {"criterion": "string", "marks": integer}
  ],
  "change_summary": "string — brief note on what was changed"
}""" % (target_label, target_label, target_label),
            ])

            if strict_json:
                sections.append(
                    "\n[IMPORTANT]\n"
                    "Return ONLY the JSON object. Do not include any commentary."
                )

            return "\n".join(sections)

    # Full question repair path
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
