"""
AION Stable Prompt Builder
==========================
One prompt builder. Frozen for deadline.
No templates, no genome, no planner.
Just clean, effective prompts.
"""

from typing import Tuple, List
from v0_1.grounding_validator import GroundingValidator

_validator = GroundingValidator()


def build_question_prompt(
    chunks:        List[str],
    exam_type:     str,
    marks:         int,
    bloom_level:   str,
    subject:       str,
    chapter:       str,
    module:        int,
) -> Tuple[str, bool]:
    """
    Build a validated, grounded prompt.

    Returns:
        (prompt, is_valid)
        If is_valid is False, do not call the LLM.
    """

    # Validate retrieval first
    validation = _validator.validate_retrieval(
        chunks    = chunks,
        query     = f"{subject} {chapter}",
        module_id = f"module_{module}",
    )

    if not validation.valid:
        print(f"[PROMPT] Retrieval rejected: {validation.reason}")
        return "", False

    prompt = _validator.build_grounded_prompt(
        chunks      = validation.supporting_chunks,
        exam_type   = exam_type,
        marks       = marks,
        bloom_level = bloom_level,
        subject     = subject,
        chapter     = chapter,
    )

    return prompt, True


def validate_generated_question(
    question: str,
    chunks:   List[str],
) -> Tuple[bool, str]:
    """
    Validate a generated question against source chunks.

    Returns:
        (is_valid, reason)
    """
    result = _validator.validate_question(question, chunks)
    return result.valid, result.reason
