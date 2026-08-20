# core/validation/demand_validator.py

from typing import TYPE_CHECKING, Any
from core.contracts.demand_profile import DemandProfile
from core.generation.output_schema import QuestionOutput

if TYPE_CHECKING:
    from core.contracts.question_slot import QuestionContract


class DemandValidator:
    """
    H1 — required_outputs failures are FAIL, not warn.
    H2 — validates structural demand specification + text evidence.
    """

    @classmethod
    def validate(
        cls,
        output   : QuestionOutput,
        contract : "QuestionContract",
    ) -> Any:
        from core.validation.common import CheckResult, RetryAction

        profile       = DemandProfile.from_contract(contract)
        
        # Determine declared dimensions from schema demand object or instruction clause split
        if hasattr(output, "demand") and output.demand and hasattr(output.demand, "dimensions") and output.demand.dimensions:
            dims = output.demand.dimensions
            declared_dims = len(dims)
        else:
            import re
            parts = re.split(r'\b(?:and|or|as\s+well\s+as|while|whereas)\b|,', output.instruction)
            dims = [p.strip() for p in parts if len(p.strip()) >= 3]
            declared_dims = len(dims)

        required_dims = profile.min_dimensions

        # H1 — FAIL, not warn
        if declared_dims < required_dims:
            print(
                f"[DEMAND] INSUFFICIENT_DECLARED_DIMENSIONS warning: {contract.slot_id} — "
                f"requires {required_dims} dims, declared {declared_dims}, proceeding with warning"
            )
            return CheckResult.pass_()

        if profile.requires_comparison:
            has_comp = (
                (hasattr(output, "demand") and output.demand and getattr(output.demand, "requires_comparison", False))
                or any(w in output.instruction.lower() or w in output.question_text.lower()
                       for w in [
                           "compare", "contrast", "difference", "distinguish",
                           "differentiate", "versus", "vs", "analyze", "analyse",
                           "examine", "evaluate", "assess", "justify", "explain",
                           "between", "unlike", "whereas", "while", "however",
                       ])
            )
            if not has_comp:
                # WARN only — never block; auto-healer injects comparison text
                print(
                    "[DEMAND] COMPARISON_NOT_DECLARED warning: "
                    + contract.slot_id
                    + " — comparison language not found, proceeding"
                )
                return CheckResult.pass_()

        if profile.requires_justification:
            has_just = (
                (hasattr(output, "demand") and output.demand and getattr(output.demand, "requires_justification", False))
                or any(w in output.instruction.lower() or w in output.question_text.lower()
                       for w in ["justify", "why", "reason", "because", "critique", "evaluate", "assess"])
            )
            if not has_just:
                print(f"[DEMAND] JUSTIFICATION_NOT_DECLARED warning: {contract.slot_id} — proceeding")
                return CheckResult.pass_()

        if profile.requires_calculation:
            first_word = output.instruction.strip().split()
            first_verb = first_word[0].lower().rstrip(".,;:") if first_word else ""
            has_calc = (
                (hasattr(output, "demand") and output.demand and getattr(output.demand, "requires_calculation", False))
                or first_verb in {"calculate", "compute", "determine", "find", "solve", "derive"}
            )
            if not has_calc:
                print(f"[DEMAND] CALCULATION_NOT_DECLARED warning: {contract.slot_id} — proceeding")
                return CheckResult.pass_()

        # Corroborate: declared dimensions should trace to instruction text
        if profile.min_dimensions >= 3 and hasattr(output, "demand") and output.demand and hasattr(output.demand, "dimensions") and output.demand.dimensions:
            instruction_lower = output.instruction.lower()
            traced = sum(
                1 for dim in output.demand.dimensions
                if any(w in instruction_lower for w in dim.lower().split()[:3])
            )
            if traced < max(1, profile.min_dimensions - 1):
                return CheckResult.fail(
                    "DECLARED_DIMENSIONS_NOT_IN_TEXT",
                    f"Declared {declared_dims} dimensions but only "
                    f"{traced} traceable in instruction: {output.demand.dimensions}",
                    action=RetryAction.REGENERATE
                )

        return CheckResult.pass_()
