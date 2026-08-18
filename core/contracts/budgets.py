# core/contracts/budgets.py

from dataclasses import dataclass

BLOOM_DEMAND_TYPE = {
    "L1": "recall",
    "L2": "explain",
    "L3": "apply",
    "L4": "analyze",
    "L5": "evaluate",
    "L6": "create",
}

MARKS_OUTPUT_TABLE = [
    (1,  3,  1),    # 1–3 marks  -> 1 outputs
    (4,  5,  2),    # 4–5 marks  -> 2 outputs
    (6,  8,  3),    # 6–8 marks  -> 3 outputs
    (9,  20, 4),    # 9+ marks   -> 4 outputs
]


def _marks_to_outputs(marks: int) -> int:
    for lo, hi, outputs in MARKS_OUTPUT_TABLE:
        if lo <= marks <= hi:
            return outputs
    return max(1, marks // 3)


def _bloom_scope(bloom: str, marks: int) -> str:
    """Rule-based scope description — covers all marks×bloom combinations."""
    demand = BLOOM_DEMAND_TYPE.get(bloom, "explain")
    outputs = _marks_to_outputs(marks)
    return (
        f"{marks}-mark {demand} answer "
        f"requiring {outputs} distinct outputs"
    )


@dataclass(frozen=True)
class AnswerBudget:
    """
    H6 — Rule-based, covers all marks×bloom combinations.
    Does NOT own required_outputs — TaskSignature owns that.
    Describes ANSWER characteristics only.
    """
    marks            : int
    scope_words      : str
    cognitive_demand : str

    @classmethod
    def from_marks_and_bloom(cls, marks: int, bloom: str) -> "AnswerBudget":
        return cls(
            marks            = marks,
            scope_words      = _bloom_scope(bloom, marks),
            cognitive_demand = BLOOM_DEMAND_TYPE.get(bloom, "explain"),
        )


@dataclass(frozen=True)
class QuestionBudget:
    """Describes question TEXT characteristics — not the answer."""
    max_question_words : int
    min_question_words : int

    @classmethod
    def from_bloom(cls, bloom: str, marks: int) -> "QuestionBudget":
        BASE = {
            "L1": 25,
            "L2": 35,
            "L3": 40,
            "L4": 50,
            "L5": 55,
            "L6": 65
        }
        mx = BASE.get(bloom, 40) + marks * 3
        return cls(max_question_words=mx, min_question_words=8)
