"""
AION Core Validators — Question Validation Pipeline
====================================================
Implements 6-stage validation pipeline for generated questions
as specified in Part X of the Production Hardening Specification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("AION.QuestionValidationPipeline")


@dataclass
class ValidationStageResult:
    stage_name : str
    passed     : bool
    reason     : str = "OK"
    action     : str = "PROCEED"


@dataclass
class QuestionValidationResult:
    status         : str            # "APPROVED" | "REJECTED"
    reason         : str = "OK"
    action         : str = "PROCEED"
    stage_results  : List[ValidationStageResult] = field(default_factory=list)


class QuestionValidationPipeline:
    """6-stage pipeline validating generated questions before approval."""

    MAX_REGENERATION_ATTEMPTS : int = 3

    @classmethod
    def validate(cls, generated_question: Any, intent: Any) -> QuestionValidationResult:
        stage_results = []

        # STAGE 1 — EVIDENCE VALIDATION
        st1 = cls._validate_evidence(generated_question, intent)
        stage_results.append(st1)
        if not st1.passed:
            return QuestionValidationResult(status="REJECTED", reason=st1.reason, action="REJECT", stage_results=stage_results)

        # STAGE 2 — BLOOM GRAMMAR VALIDATION
        st2 = cls._validate_bloom_grammar(generated_question, intent)
        stage_results.append(st2)
        if not st2.passed:
            return QuestionValidationResult(status="REJECTED", reason=st2.reason, action="REGENERATE_SLOT", stage_results=stage_results)

        # STAGE 3 — SEMANTIC VALIDATION
        st3 = cls._validate_semantic_grounding(generated_question, intent)
        stage_results.append(st3)
        if not st3.passed:
            return QuestionValidationResult(status="REJECTED", reason=st3.reason, action="REGENERATE_SLOT", stage_results=stage_results)

        # STAGE 4 — EQUATION VALIDATION
        st4 = cls._validate_equation_integrity(generated_question, intent)
        stage_results.append(st4)
        if not st4.passed:
            return QuestionValidationResult(status="REJECTED", reason=st4.reason, action="REGENERATE_SLOT", stage_results=stage_results)

        # STAGE 5 — COMPLETENESS VALIDATION
        st5 = cls._validate_completeness(generated_question, intent)
        stage_results.append(st5)
        if not st5.passed:
            return QuestionValidationResult(status="REJECTED", reason=st5.reason, action="REGENERATE_SLOT", stage_results=stage_results)

        # STAGE 6 — STRUCTURE VALIDATION
        st6 = cls._validate_structure(generated_question, intent)
        stage_results.append(st6)
        if not st6.passed:
            return QuestionValidationResult(status="REJECTED", reason=st6.reason, action="REJECT", stage_results=stage_results)

        return QuestionValidationResult(status="APPROVED", reason="ALL_STAGES_PASSED", action="PROCEED", stage_results=stage_results)

    @classmethod
    def validate_with_regeneration(
        cls,
        intent: Any,
        generator_fn: Callable[[Any, int], Any]
    ) -> Any:
        """Runs validation with up to 3 regeneration attempts on failure."""
        for attempt in range(cls.MAX_REGENERATION_ATTEMPTS):
            gen = generator_fn(intent, attempt)
            res = cls.validate(gen, intent)
            if res.status == "APPROVED":
                logger.info(f"[VALIDATOR] Question approved on attempt {attempt + 1}")
                return gen
            logger.warning(f"[VALIDATOR] Attempt {attempt + 1} rejected: {res.reason} ({res.action})")

        raise RuntimeError(f"Slot regeneration exhausted for slot after {cls.MAX_REGENERATION_ATTEMPTS} attempts")

    @classmethod
    def _validate_evidence(cls, gen: Any, intent: Any) -> ValidationStageResult:
        ev_chunks = getattr(gen, "evidence_refs", []) or getattr(intent, "evidence_chunks", [])
        for chunk in ev_chunks:
            status = getattr(chunk, "status", "VALID")
            if str(status) in ("QUARANTINED", "INVALID"):
                return ValidationStageResult(stage_name="EVIDENCE", passed=False, reason="QUARANTINED_EVIDENCE")
        return ValidationStageResult(stage_name="EVIDENCE", passed=True)

    @classmethod
    def _validate_bloom_grammar(cls, gen: Any, intent: Any) -> ValidationStageResult:
        q_text = getattr(gen, "question_text", "") or getattr(gen, "text", "")
        bloom = getattr(intent, "bloom", "")
        if "Create between" in q_text or "Apply why" in q_text:
            return ValidationStageResult(stage_name="BLOOM_GRAMMAR", passed=False, reason="FORBIDDEN_PHRASE")
        return ValidationStageResult(stage_name="BLOOM_GRAMMAR", passed=True)

    @classmethod
    def _validate_semantic_grounding(cls, gen: Any, intent: Any) -> ValidationStageResult:
        q_text = getattr(gen, "question_text", "") or getattr(gen, "text", "")
        if len(q_text.strip()) < 10:
            return ValidationStageResult(stage_name="SEMANTIC_GROUNDING", passed=False, reason="POOR_GROUNDING")
        return ValidationStageResult(stage_name="SEMANTIC_GROUNDING", passed=True)

    @classmethod
    def _validate_equation_integrity(cls, gen: Any, intent: Any) -> ValidationStageResult:
        q_text = getattr(gen, "question_text", "") or getattr(gen, "text", "")
        if "\ufffd" in q_text:
            return ValidationStageResult(stage_name="EQUATION_INTEGRITY", passed=False, reason="UNICODE_CORRUPTION")
        return ValidationStageResult(stage_name="EQUATION_INTEGRITY", passed=True)

    @classmethod
    def _validate_completeness(cls, gen: Any, intent: Any) -> ValidationStageResult:
        q_text = getattr(gen, "question_text", "") or getattr(gen, "text", "")
        if q_text.endswith("...") or q_text.endswith("as shown in Figure"):
            return ValidationStageResult(stage_name="COMPLETENESS", passed=False, reason="TRUNCATED_QUESTION")
        return ValidationStageResult(stage_name="COMPLETENESS", passed=True)

    @classmethod
    def _validate_structure(cls, gen: Any, intent: Any) -> ValidationStageResult:
        g_marks = getattr(gen, "marks", None)
        i_marks = getattr(intent, "marks", None)
        if g_marks is not None and i_marks is not None and g_marks != i_marks:
            return ValidationStageResult(stage_name="STRUCTURE", passed=False, reason="MARKS_MISMATCH")
        return ValidationStageResult(stage_name="STRUCTURE", passed=True)
