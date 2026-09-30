"""
Prompt construction for the API evaluation call.

The prompt is a single JSON-friendly text that lists all candidate
questions and asks for structured verdicts. Output schema is embedded.
"""

from __future__ import annotations

from typing import List


# -----------------------------------------------------------------------------
# Caps
# -----------------------------------------------------------------------------


MAX_QUESTION_CHARS = 800
MAX_SOLUTION_CHARS = 400
MAX_EVIDENCE_SUMMARY_CHARS = 300
MAX_QUESTIONS_PER_BATCH = 20


# -----------------------------------------------------------------------------
# Output schema description
# -----------------------------------------------------------------------------


OUTPUT_SCHEMA = """{
  "verdicts": [
    {
      "slot_id": "string — the slot ID exactly as provided",
      "verdict": "pass" | "fail",
      "reason_codes": ["FACTUAL_ERROR" | "UNGROUNDED" | "CIRCULAR" | "OFF_TOPIC" | "VERB_TASK_MISMATCH"],
      "detail": "short explanation, max 200 chars",
      "suggested_fix": "what to change, or null if none"
    }
  ]
}"""


# -----------------------------------------------------------------------------
# Instructions
# -----------------------------------------------------------------------------


INSTRUCTIONS = """You are an exam question auditor. Evaluate each question below against four criteria:

1. FACTUAL_COHERENCE: Are the claims in the question factually correct for the stated topic? Flag FACTUAL_ERROR if the question asserts incorrect facts or implies a wrong physical/mathematical relationship.

2. GROUNDING: Is the question answerable from the provided evidence? Flag UNGROUNDED if the question asks about material not present in the evidence, or requires facts not supplied.

3. CIRCULARITY: Does the question define a term and then ask the student to recall that definition? Flag CIRCULAR. Also flag if the question gives the answer within its own text.

4. TOPIC_ALIGNMENT: Does the question match the assigned module topic and Course Outcome? Flag OFF_TOPIC if the question drifts to another topic.

5. VERB_TASK_ALIGNMENT: Does the question's first verb match the task it asks (e.g., "Classify" needs categories, "Solve" needs numerical givens)? Flag VERB_TASK_MISMATCH if the verb and task are semantically incompatible.

Rules:
- Verdict "pass" requires NO critical issues.
- Verdict "fail" requires at least one reason_code.
- If unsure, prefer "pass" — the Refinement Agent will handle marginal cases.
- Return ONLY valid JSON matching the schema below."""


# -----------------------------------------------------------------------------
# Prompt builder
# -----------------------------------------------------------------------------


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _question_block(question, evidence_summary: str) -> str:
    return (
        f"---\n"
        f"slot_id: {question.slot_id}\n"
        f"module: {question.module_id}\n"
        f"topic: {question.topic}\n"
        f"co: {question.co}\n"
        f"bloom: {question.bloom}\n"
        f"marks: {question.marks}\n"
        f"question: {_truncate(question.question_text, MAX_QUESTION_CHARS)}\n"
        f"solution: {_truncate(question.solution, MAX_SOLUTION_CHARS)}\n"
        f"evidence_summary: {_truncate(evidence_summary, MAX_EVIDENCE_SUMMARY_CHARS)}"
    )


def build_evidence_summary(question) -> str:
    """
    Build a compact summary of the evidence attached to a question.
    Uses the first evidence block's text (already truncated upstream).
    """
    # Writing Agent carries evidence via the plan; but GeneratedQuestion
    # itself doesn't hold evidence blocks. If absent, use empty string.
    # Downstream callers can pass the evidence via `question.evidence_summary`
    # if the Writing Agent adds it in a later iteration.
    return getattr(question, "evidence_summary", "")


def build_batch_prompt(questions: List[object]) -> str:
    """Compose the API prompt for a batch of questions."""
    blocks = [_question_block(q, build_evidence_summary(q)) for q in questions]
    body = "\n".join(blocks)
    return (
        f"{INSTRUCTIONS}\n\n"
        f"[OUTPUT SCHEMA]\n{OUTPUT_SCHEMA}\n\n"
        f"[QUESTIONS TO EVALUATE]\n{body}\n"
    )
