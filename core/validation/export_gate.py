# core/validation/export_gate.py

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Any
from core.contracts.question import GeneratedQuestion
from core.validation.common import CheckResult
from core.validation.math_validator import validate_math_consistency, validate_math_block_with_render
from core.validation.linter import check_multi_slot_contamination, check_unicode_integrity

LOG = logging.getLogger("aion.export_gate")


@dataclass
class ExportDecision:
    status: str
    failures: List[str] = field(default_factory=list)
    message: str = ""

    @property
    def passed(self) -> bool:
        return self.status == "PASS"


class ExportGate:
    """
    Export Integrity Gate.
    Final authority. No paper reaches the frontend unless this passes.
    Legacy QA runs for diagnostics only — it cannot override ExportGate.
    """

    @classmethod
    def evaluate(
        cls,
        paper: Any = None,
        plan: Any = None,
        generated_slots: Optional[List[GeneratedQuestion]] = None,
        profile: Any = None,
    ) -> ExportDecision:
        questions = generated_slots or []
        if not questions and paper:
            # Attempt to extract questions from paper object if provided
            if hasattr(paper, "or_pairs"):
                for pair in paper.or_pairs:
                    questions.extend(getattr(pair, "alternatives_a", []))
                    questions.extend(getattr(pair, "alternatives_b", []))

        check_res = cls.validate(questions)
        if not check_res.passed:
            return ExportDecision(
                status="BLOCKED",
                failures=[f"{check_res.code}: {check_res.message}"],
                message=f"Paper blocked by ExportGate: {check_res.message}"
            )
        return ExportDecision(status="PASS", failures=[], message="Export Gate PASS")

    @classmethod
    def validate(cls, questions: List[GeneratedQuestion]) -> CheckResult:
        if not questions:
            return CheckResult.fail("EMPTY_PAPER", "Paper contains no questions.")

        slot_ids = set()
        for q in questions:
            # 1. Text checks
            if not q.question_text or not q.question_text.strip():
                return CheckResult.fail("EMPTY_QUESTION_TEXT", f"Slot {q.slot_id} has empty question text.")

            # 2. Duplicate checks
            if q.slot_id in slot_ids:
                return CheckResult.fail("DUPLICATE_QUESTION_SLOT", f"Slot {q.slot_id} appears multiple times in the paper.")
            slot_ids.add(q.slot_id)

            # 3. Unicode check
            uni_check = check_unicode_integrity(q.question_text)
            if not uni_check.passed:
                return CheckResult.fail("UNICODE_CORRUPTION", f"Slot {q.slot_id} has unicode corruption: {uni_check.message}")

            # 4. Multi-slot contamination
            multi_check = check_multi_slot_contamination(q.question_text)
            if not multi_check.passed:
                return CheckResult.fail("MULTI_SLOT_CONTAMINATION", f"Slot {q.slot_id} has multi-slot contamination: {multi_check.message}")

            # 5. Math checks
            if hasattr(q, "output") and q.output is not None:
                math_check = validate_math_consistency(q.output)
                if not math_check.passed:
                    return CheckResult.fail("MATH_INTEGRITY_FAILURE", f"Slot {q.slot_id} has math inconsistency: {math_check.message}")

                for block in q.output.math_blocks:
                    render_check = validate_math_block_with_render(block)
                    if not render_check.passed:
                        return CheckResult.fail("MATH_RENDER_FAILURE", f"Slot {q.slot_id} has math render failure for block {block.block_id}: {render_check.message}")

            # 6. Provenance binding
            prov = getattr(q, "provenance", None)
            if prov is None:
                return CheckResult.fail(
                    "MISSING_PROVENANCE",
                    f"Slot {q.slot_id} has no QuestionProvenance record. "
                    f"This indicates the question was not built from a QuestionSlot."
                )

            if prov.slot_id != q.slot_id:
                return CheckResult.fail(
                    "PROVENANCE_SLOT_MISMATCH",
                    f"Slot {q.slot_id}: provenance.slot_id={prov.slot_id!r} "
                    f"does not match question slot_id={q.slot_id!r}."
                )

            if prov.module_id != q.module_id:
                return CheckResult.fail(
                    "PROVENANCE_MODULE_MISMATCH",
                    f"Slot {q.slot_id}: provenance.module_id={prov.module_id} "
                    f"does not match question module_id={q.module_id}. "
                    f"Cross-module contamination detected."
                )

            if prov.co != q.co:
                return CheckResult.fail(
                    "PROVENANCE_CO_MISMATCH",
                    f"Slot {q.slot_id}: provenance.co={prov.co!r} "
                    f"does not match question co={q.co!r}. "
                    f"CO drift detected."
                )

            if prov.bloom_level != q.bloom_level:
                return CheckResult.fail(
                    "PROVENANCE_BLOOM_MISMATCH",
                    f"Slot {q.slot_id}: provenance.bloom_level={prov.bloom_level!r} "
                    f"does not match question bloom_level={q.bloom_level!r}."
                )

            if prov.marks != q.marks:
                return CheckResult.fail(
                    "PROVENANCE_MARKS_MISMATCH",
                    f"Slot {q.slot_id}: provenance.marks={prov.marks} "
                    f"does not match question marks={q.marks}."
                )

        return CheckResult.pass_()
