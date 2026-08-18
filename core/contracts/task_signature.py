# core/contracts/task_signature.py

from dataclasses import dataclass
from typing import Tuple
from core.contracts.budgets import _marks_to_outputs

BLOOM_OPERATION_MAP = {
    "L1": "REMEMBER",
    "L2": "UNDERSTAND",
    "L3": "APPLY",
    "L4": "ANALYZE",
    "L5": "EVALUATE",
    "L6": "CREATE",
}

_ALL_OPS = ("REMEMBER", "UNDERSTAND", "APPLY", "CALCULATE", "COMPARE",
            "ANALYZE", "EVALUATE", "JUSTIFY", "DESIGN", "CREATE")

ALLOWED_SECONDARY_OPERATIONS = {
    "REMEMBER":   _ALL_OPS,
    "UNDERSTAND": _ALL_OPS,
    "APPLY":      _ALL_OPS,
    "ANALYZE":    _ALL_OPS,
    "EVALUATE":   _ALL_OPS,
    "CREATE":     _ALL_OPS,
}


@dataclass(frozen=True)
class TaskSignature:
    """
    Cognitive constraints for a question slot.
    Defines operations allowed in generation and checked by linter.
    """
    primary_operation            : str
    allowed_secondary_operations  : Tuple[str, ...]
    requires_comparison          : bool
    requires_calculation         : bool
    requires_justification       : bool
    
    # DEPRECATED — never used for generation or validation in v5
    required_outputs             : int = 1

    @classmethod
    def from_bloom_marks_type(cls, bloom_level: str, marks: int, question_type: str) -> "TaskSignature":
        primary = BLOOM_OPERATION_MAP.get(bloom_level, "UNDERSTAND")
        allowed_sec = ALLOWED_SECONDARY_OPERATIONS.get(primary, ())
        
        requires_comparison = (bloom_level == "L4")
        requires_calculation = (bloom_level == "L3" or question_type.upper() == "NUMERICAL")
        requires_justification = (bloom_level in ("L5", "L6"))
        
        # Pull legacy outputs value
        outputs = _marks_to_outputs(marks)

        return cls(
            primary_operation=primary,
            allowed_secondary_operations=allowed_sec,
            requires_comparison=requires_comparison,
            requires_calculation=requires_calculation,
            requires_justification=requires_justification,
            required_outputs=outputs,
        )
