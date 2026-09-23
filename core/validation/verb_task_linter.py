# core/validation/verb_task_linter.py

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from core.validation.common import CheckResult, RetryAction

LOG = logging.getLogger("aion.validation.verb_task_linter")


class VerbTaskCoherenceLinter:
    """
    Behavioral Verb-Task Coherence Linter (Phase C.5).
    Enforces structural, cognitive constraints on verb-object relationships
    using NLP/regex patterns rather than narrow string blacklists.
    """

    @classmethod
    def lint_instruction(
        cls,
        instruction: str,
        question_text: str = "",
        math_required: bool = False,
        visual_required: bool = False,
    ) -> CheckResult:
        """
        Validates that the cognitive verb matches the structural entity requested.
        """
        # Feature flag check
        enabled = os.getenv("AION_ENABLE_VERB_TASK_LINTER", "false").lower() in ("true", "1", "yes")
        if not enabled:
            return CheckResult.pass_()

        text = instruction.strip()
        full_text = (instruction + " " + question_text).strip()
        if not text:
            return CheckResult.fail("EMPTY_INSTRUCTION", "Instruction text is empty.", action=RetryAction.REGENERATE)

        first_word = text.split()[0].lower().rstrip(".,;:")

        # 1. Numerical / Calculation Verbs: Solve, Calculate, Compute, Determine
        if first_word in {"solve", "calculate", "compute", "evaluate", "derive"}:
            # Must possess numerical values, formulas, or explicit math artifacts
            has_numbers = bool(re.search(r'\b\d+(?:\.\d+)?\b', full_text))
            has_formula = bool(re.search(r'[\=\+\-\*\/\^\\]|\[MATH:', full_text))
            has_calc_terms = bool(re.search(r'\b(?:given|assuming|where|rate|values?|parameters?|efficiency|latency|bandwidth)\b', full_text, re.IGNORECASE))
            if not (has_numbers or has_formula or has_calc_terms or math_required):
                return CheckResult.fail(
                    "UNGROUNDED_CALCULATION_TASK",
                    f"Instruction begins with numerical verb '{first_word.capitalize()}' but question provides no numerical givens, formulas, or math context.",
                    action=RetryAction.REGENERATE
                )

        # 2. Classification Verbs: Classify, Categorize
        if first_word in {"classify", "categorize"}:
            has_category_objects = bool(re.search(r'\b(?:types?|categories|models?|classes|levels?|approaches|architectures?|taxonom(?:y|ies))\b', full_text, re.IGNORECASE))
            has_procedure = bool(re.search(r'\b(?:steps?|procedure|algorithm|workflow|sequence|phases?)\b', full_text, re.IGNORECASE))
            if has_procedure and not has_category_objects:
                return CheckResult.fail(
                    "INCOHERENT_CLASSIFY_TASK",
                    f"Instruction begins with '{first_word.capitalize()}' but targets sequential steps/procedure rather than taxonomies or categories.",
                    action=RetryAction.REGENERATE
                )

        # 3. Comparative Verbs: Compare, Distinguish, Differentiate, Contrast
        if first_word in {"compare", "distinguish", "differentiate", "contrast"}:
            has_plural_or_and = bool(re.search(r'\b(?:and|between|versus|vs\.?)\b', full_text, re.IGNORECASE))
            if not has_plural_or_and:
                return CheckResult.fail(
                    "INSUFFICIENT_COMPARISON_ENTITIES",
                    f"Instruction begins with comparative verb '{first_word.capitalize()}' but does not specify two or more distinct entities to compare.",
                    action=RetryAction.REGENERATE
                )

        # 4. Visual Verbs: Draw, Sketch, Illustrate
        if first_word in {"draw", "sketch"}:
            # If instruction demands drawing but visual policy is not declared
            if not visual_required and not re.search(r'\b(?:diagram|circuit|block|schematic|architecture|graph|chart|flowchart)\b', full_text, re.IGNORECASE):
                return CheckResult.fail(
                    "UNSUPPORTED_DRAW_TASK",
                    f"Instruction begins with '{first_word.capitalize()}' without declaring a valid diagram object or visual requirement.",
                    action=RetryAction.REGENERATE
                )

        return CheckResult.pass_()
