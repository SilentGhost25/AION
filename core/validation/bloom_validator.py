# core/validation/bloom_validator.py

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.contracts.question_slot import QuestionSlot

BLOOM_VERB_LEVEL_MAP = {
    "L1": {"define","list","identify","name","state","recall","recognize","enumerate"},
    "L2": {"explain","describe","summarize","illustrate","interpret","classify","discuss","outline"},
    "L3": {"calculate","apply","demonstrate","determine","solve","implement","use","compute","derive","show","find"},
    "L4": {"analyze","analyse","compare","differentiate","examine","investigate","categorize","contrast","distinguish"},
    "L5": {"evaluate","critique","justify","assess","validate","judge","defend","appraise"},
    "L6": {"design","develop","propose","formulate","create","generate","construct","build","invent"},
}

BLOOM_TO_OP = {
    "L1":"RECALL","L2":"EXPLAIN","L3":"APPLY",
    "L4":"ANALYZE","L5":"EVALUATE","L6":"CREATE",
}

# Verbs whose presence suggests a higher Bloom level than requested
EVALUATION_MARKERS = {
    "evaluate","evaluate","judge","recommend","justify superiority",
    "assess effectiveness","defend","argue whether","critique",
    "assess which is better","which is better","which is superior",
}

EVALUATION_PHRASES = [
    r"\badvantages\s+(?:and\s+disadvantages)?\b",
    r"\bwhich\s+(?:is|are)\s+(?:better|superior|more\s+effective)\b",
    r"\bjudge\b",
    r"\brecommend\b",
    r"\bdefend\b",
    r"\bjustify\s+(?:why|which)\b",
    r"\bcritique\b",
]


@dataclass
class BloomCheckResult:
    passed  : bool
    code    : str
    detail  : str
    action  : str = "REGENERATE_WITH_BLOOM_HINT"

    def to_check_result(self) -> "Any":
        from core.validation.common import CheckResult, RetryAction
        if self.passed:
            return CheckResult.pass_()
        return CheckResult.fail(
            self.code, self.detail,
            action=getattr(RetryAction, self.action, RetryAction.REGENERATE)
        )


def check_bloom_two_layer(
    instruction : str,
    slot        : "QuestionSlot",
) -> BloomCheckResult:
    """
    Layer 1 — Lexical: question must begin with expected Bloom verb.
    Layer 2 — Semantic: instruction must not embed higher-level cognitive ops.
    """

    # -- LAYER 1: Lexical ------------------------------------------------------

    if not instruction or not instruction.strip():
        return BloomCheckResult(False, "INSTRUCTION_EMPTY", "Instruction is empty")

    first_word = instruction.strip().split()[0].rstrip(".,;:").lower()
    expected   = slot.bloom_verb.lower() if hasattr(slot, "bloom_verb") else "explain"

    # Accept British/American spelling variants
    EQUIVALENTS = {
        "analyse": {"analyze", "analyse"},
        "analyze": {"analyze", "analyse"},
    }
    accepted = EQUIVALENTS.get(expected, {expected})

    if first_word not in accepted:
        # Diagnose: is it a different bloom level or not a verb at all?
        detected_level = None
        for level, verbs in BLOOM_VERB_LEVEL_MAP.items():
            if first_word in verbs:
                detected_level = level
                break

        return BloomCheckResult(
            passed = False,
            code   = "BLOOM_VERB_WRONG",
            detail = (
                f"Expected '{expected}', got '{first_word}'. "
                f"Detected level: {detected_level or 'not a bloom verb'}."
            ),
            action = "REGENERATE_WITH_BLOOM_HINT"
        )

    # -- LAYER 2: Semantic -----------------------------------------------------

    if slot.bloom_level in {"L1", "L2", "L3"}:
        # Only enforce semantic check for L4 and below
        return BloomCheckResult(True, "PASS", "")

    instruction_lower = instruction.lower()

    if slot.bloom_level == "L4":
        # L4 must be analytic — reject evaluation-heavy phrasing
        for pattern in EVALUATION_PHRASES:
            m = re.search(pattern, instruction_lower)
            if m:
                return BloomCheckResult(
                    passed = False,
                    code   = "BLOOM_SEMANTIC_MISMATCH",
                    detail = (
                        f"L4 ANALYZE question contains evaluation language: "
                        f"'{m.group()}'. "
                        f"Use comparison/decomposition/examination instead."
                    ),
                    action = "REGENERATE_WITH_BLOOM_HINT"
                )

    return BloomCheckResult(True, "PASS", "")
