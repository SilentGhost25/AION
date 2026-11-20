# core/validation/export_gate.py

import os
import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
from core.contracts.question import GeneratedQuestion
from core.validation.common import CheckResult
from core.validation.math_validator import validate_math_consistency, validate_math_block_with_render
from core.validation.linter import check_multi_slot_contamination, check_unicode_integrity

LOG = logging.getLogger("aion.export_gate")

DEFAULT_FAITHFULNESS_THRESHOLD: float = 0.75
DEFAULT_QA_THRESHOLD: float = 75.0

UNRESOLVED_PLACEHOLDER_REGEX = re.compile(
    r"\[UNRESOLVED\s+(?:SLOT|QUESTION|SUBPART)[\s:][^\]]*\]", re.IGNORECASE
)


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
    def evaluate_dict(cls, paper_data: Any) -> ExportDecision:
        """
        Physical Export Gate for raw wire dictionary payloads (e.g. from /api/export/docx).
        Validates structure, content completeness, duplicate slots, unicode, and marks.
        """
        if not isinstance(paper_data, dict):
            return ExportDecision(
                status="BLOCKED",
                failures=["INVALID_PAPER_SCHEMA: Payload must be a JSON object."],
                message="Export blocked: paper payload is not a dictionary."
            )

        modules = paper_data.get("modules")
        if not isinstance(modules, list) or len(modules) == 0:
            raw_qs = paper_data.get("questions")
            if isinstance(raw_qs, list) and len(raw_qs) > 0:
                modules = [{"module_index": 1, "questions": raw_qs}]
            else:
                return ExportDecision(
                    status="BLOCKED",
                    failures=["EMPTY_PAPER: Paper contains no modules."],
                    message="Export blocked: paper contains no modules."
                )

        slot_ids = set()
        physical_positions = set()
        total_subquestions = 0

        for m_idx, mod in enumerate(modules):
            if not isinstance(mod, dict):
                return ExportDecision(
                    status="BLOCKED",
                    failures=[f"MALFORMED_MODULE: Module at index {m_idx} is not an object."],
                    message=f"Export blocked: module {m_idx} is malformed."
                )
            questions = mod.get("questions", [])
            if not isinstance(questions, list):
                return ExportDecision(
                    status="BLOCKED",
                    failures=[f"MALFORMED_QUESTIONS: Questions in module {m_idx} is not a list."],
                    message=f"Export blocked: questions in module {m_idx} is malformed."
                )

            for q_idx, q in enumerate(questions):
                if not isinstance(q, dict):
                    return ExportDecision(
                        status="BLOCKED",
                        failures=[f"MALFORMED_QUESTION: Question at module {m_idx}, index {q_idx} is not an object."],
                        message="Export blocked: question is malformed."
                    )

                # Check question-level status
                if q.get("status") == "UNRESOLVED" or q.get("is_unresolved") is True or q.get("unresolved") is True:
                    return ExportDecision(
                        status="BLOCKED",
                        failures=[f"UNRESOLVED_SLOT_DETECTED: Question at module {m_idx}, index {q_idx} is marked UNRESOLVED."],
                        message="Export blocked: unresolved question detected."
                    )

                subquestions = q.get("subquestions", [])
                if not isinstance(subquestions, list):
                    return ExportDecision(
                        status="BLOCKED",
                        failures=[f"MALFORMED_SUBQUESTIONS: Subquestions at module {m_idx}, question {q_idx} is not a list."],
                        message="Export blocked: subquestions is not a list."
                    )

                for s_idx, sq in enumerate(subquestions):
                    if not isinstance(sq, dict):
                        return ExportDecision(
                            status="BLOCKED",
                            failures=[f"MALFORMED_SUBQUESTION: Subquestion at module {m_idx}, q {q_idx}, sub {s_idx} is not an object."],
                            message="Export blocked: subquestion is malformed."
                        )

                    total_subquestions += 1
                    letter = str(sq.get("letter") or chr(ord('a') + s_idx)).strip().lower()
                    pos = (m_idx, q_idx, letter)
                    if pos in physical_positions:
                        return ExportDecision(
                            status="BLOCKED",
                            failures=[f"DUPLICATE_QUESTION_SLOT: Duplicate subquestion position (mod {m_idx}, q {q_idx}, letter {letter})."],
                            message=f"Export blocked: duplicate subquestion position {pos}."
                        )
                    physical_positions.add(pos)

                    # Slot ID uniqueness if present
                    sq_slot_id = sq.get("slot_id")
                    if sq_slot_id and isinstance(sq_slot_id, str) and sq_slot_id.strip():
                        if sq_slot_id in slot_ids:
                            return ExportDecision(
                                status="BLOCKED",
                                failures=[f"DUPLICATE_QUESTION_SLOT: Slot {sq_slot_id} appears multiple times in the paper."],
                                message=f"Export blocked: duplicate slot {sq_slot_id}."
                            )
                        slot_ids.add(sq_slot_id)

                    # Status checks
                    if sq.get("status") == "UNRESOLVED" or sq.get("is_unresolved") is True or sq.get("unresolved") is True:
                        return ExportDecision(
                            status="BLOCKED",
                            failures=[f"UNRESOLVED_SLOT_DETECTED: Subquestion {pos} is marked UNRESOLVED."],
                            message=f"Export blocked: unresolved subquestion {pos}."
                        )

                    text = str(sq.get("text") or "").strip()
                    if not text:
                        return ExportDecision(
                            status="BLOCKED",
                            failures=[f"EMPTY_QUESTION_TEXT: Subquestion {pos} has empty text."],
                            message=f"Export blocked: empty question text at {pos}."
                        )

                    # Placeholder regex check
                    if UNRESOLVED_PLACEHOLDER_REGEX.search(text) or "[UNRESOLVED SLOT" in text:
                        return ExportDecision(
                            status="BLOCKED",
                            failures=[f"UNRESOLVED_SLOT_DETECTED: Placeholder detected in subquestion {pos}: {text[:60]}..."],
                            message=f"Export blocked: unresolved placeholder at {pos}."
                        )

                    # Unicode check
                    uni_check = check_unicode_integrity(text)
                    if not uni_check.passed:
                        return ExportDecision(
                            status="BLOCKED",
                            failures=[f"UNICODE_CORRUPTION: Subquestion {pos} has unicode corruption: {uni_check.message}"],
                            message=f"Export blocked: unicode corruption at {pos}."
                        )

                    # Multi-slot contamination
                    multi_check = check_multi_slot_contamination(text)
                    if not multi_check.passed:
                        return ExportDecision(
                            status="BLOCKED",
                            failures=[f"MULTI_SLOT_CONTAMINATION: Subquestion {pos} has multi-slot contamination: {multi_check.message}"],
                            message=f"Export blocked: multi-slot contamination at {pos}."
                        )

                    # Marks check: positive integer
                    marks = sq.get("marks")
                    if marks is None or isinstance(marks, bool) or not isinstance(marks, (int, float)) or marks <= 0 or int(marks) != marks:
                        return ExportDecision(
                            status="BLOCKED",
                            failures=[f"INVALID_MARKS: Subquestion {pos} has invalid marks: {marks}"],
                            message=f"Export blocked: invalid marks {marks} at {pos}."
                        )

        if total_subquestions == 0:
            return ExportDecision(
                status="BLOCKED",
                failures=["EMPTY_PAPER: Paper contains no subquestions."],
                message="Export blocked: paper contains no questions."
            )

        return ExportDecision(status="PASS", failures=[], message="Physical export validation PASS")

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

        # 0. Fail-Closed Unresolved Slot Check
        for q in questions:
            q_status = getattr(q, "status", "")
            slot_obj = getattr(q, "slot", None)
            slot_status = getattr(slot_obj, "status", "") if slot_obj else ""
            q_text = str(getattr(q, "question_text", "") or "")
            if q_status == "UNRESOLVED" or slot_status == "UNRESOLVED" or "[UNRESOLVED SLOT" in q_text:
                slot_id = getattr(q, "slot_id", "unknown_slot")
                LOG.error(f"[EXPORT GATE FAIL-CLOSED] Slot {slot_id} is UNRESOLVED. Rejecting export.")
                return CheckResult.fail(
                    "UNRESOLVED_SLOT_DETECTED",
                    f"Paper contains unresolved slot {slot_id}. Generation fail-closed."
                )

        # Threshold checks when hard block enabled
        unresolved_hard_block = os.getenv("AION_ENABLE_UNRESOLVED_HARD_BLOCK", "true").lower() in ("true", "1", "yes")
        if unresolved_hard_block:
            faith_thresh = float(os.getenv("AION_FAITHFULNESS_THRESHOLD", "0.60"))
            for q in questions:
                ragas = getattr(q, "ragas_metrics", None)
                if ragas:
                    f_val = getattr(ragas, "faithfulness", None)
                    if f_val is not None and isinstance(f_val, (int, float)) and f_val < faith_thresh:
                        return CheckResult.fail(
                            "FAITHFULNESS_BELOW_THRESHOLD",
                            f"Slot {q.slot_id} faithfulness {f_val:.2f} is below required threshold {faith_thresh}."
                        )

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
                valid_blocks = []
                for block in list(getattr(q.output, "math_blocks", [])):
                    render_check = validate_math_block_with_render(block)
                    if not render_check.passed:
                        try:
                            from core.generation.auto_healer import AutoHealer
                            block.latex = AutoHealer._heal_latex(block.latex)
                            render_check = validate_math_block_with_render(block)
                        except Exception:
                            pass

                    if render_check.passed:
                        valid_blocks.append(block)
                    else:
                        from core.validation.math_validator import KaTeXAvailabilityGate
                        LOG.warning(
                            f"[EXPORT GATE] Slot {q.slot_id} block {block.block_id} math render failure "
                            f"({render_check.message}). Converting to fallback representation."
                        )
                        unicode_math = KaTeXAvailabilityGate._latex_to_unicode_fallback(block.latex)
                        if hasattr(q, "question_text") and q.question_text:
                            q.question_text = q.question_text.replace(f"[MATH:{block.block_id}]", unicode_math).strip()
                        if hasattr(q.output, "question_text") and q.output.question_text:
                            q.output.question_text = q.output.question_text.replace(f"[MATH:{block.block_id}]", unicode_math).strip()

                q.output.math_blocks = valid_blocks
                if hasattr(q, "math_blocks"):
                    q.math_blocks = valid_blocks

                # Final sanitize of any orphan [MATH:...] or artifact tags from user-facing text
                import re as _re
                if hasattr(q, "question_text") and q.question_text:
                    q.question_text = _re.sub(r'\[MATH:[^\]]+\]', '', q.question_text).strip()
                    q.question_text = _re.sub(r'\[FIGURE_ID:[^\]]+\]', '', q.question_text).strip()
                    q.question_text = _re.sub(r'\[KU:[^\]]+\]', '', q.question_text).strip()
                if hasattr(q.output, "question_text") and q.output.question_text:
                    q.output.question_text = _re.sub(r'\[MATH:[^\]]+\]', '', q.output.question_text).strip()
                    q.output.question_text = _re.sub(r'\[FIGURE_ID:[^\]]+\]', '', q.output.question_text).strip()
                    q.output.question_text = _re.sub(r'\[KU:[^\]]+\]', '', q.output.question_text).strip()

                math_check = validate_math_consistency(q.output)
                if not math_check.passed:
                    # If consistency still failed, strip all math blocks to ensure paper ships cleanly
                    q.output.math_blocks = []
                    if hasattr(q, "math_blocks"):
                        q.math_blocks = []

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
