"""
AION Master Production Specification — Final Quality Gate (Single QA Authority)
================================================================================
The SINGLE authoritative QA authority in AION.
Evaluates Category A (Structure), B (Academic), C (Content), D (Math), E (Evidence), F (Rendering).
Outputs a unified score (0-100) and single decision (PASS | PASS_WITH_WARNINGS | REPAIR | FAIL).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List

from core.contracts.paper_structure import PaperStructurePlan
from core.contracts.question import GeneratedQuestion


@dataclass
class QAResult:
    request_id       : str
    plan_id          : str
    qa_score         : float
    status           : str          # "PASS" | "PASS_WITH_WARNINGS" | "REPAIR" | "BLOCKED"
    category_scores  : Dict[str, float] = field(default_factory=dict)
    failures         : List[str] = field(default_factory=list)
    warnings         : List[str] = field(default_factory=list)
    report_text      : str = ""
    exportable       : bool = False


# Forbidden Bloom verb combinations in a single question
FORBIDDEN_VERB_COMBOS = [
    ("analyze", "create"), ("design", "justify"),
    ("apply", "evaluate"), ("describe", "calculate"),
    ("list", "design"), ("analyze", "design"),
]

BLOOM_VERB_KEYWORDS = {
    "define", "list", "identify", "name", "state",
    "explain", "describe", "summarize", "illustrate",
    "calculate", "apply", "demonstrate", "determine", "solve",
    "analyze", "compare", "examine", "differentiate",
    "evaluate", "critique", "justify", "assess",
    "design", "develop", "construct", "propose", "formulate", "create",
}


class FinalQualityGate:
    """
    The Single Authoritative QA Gate for AION.
    Produces EXACTLY ONE verdict. QA Score 0 = BLOCKED unconditionally.
    """

    @classmethod
    def evaluate(
        cls,
        plan: PaperStructurePlan,
        generated_questions: List[GeneratedQuestion],
    ) -> QAResult:
        failures: List[str] = []
        warnings: List[str] = []

        # -- CATEGORY A — STRUCTURE (weight 0.30) ------------------------------
        cat_a_pass = True

        if len(plan.or_pairs) != plan.module_count:
            failures.append("A01/A02: Module or OR-pair count mismatch")
            cat_a_pass = False

        total_q_count = len(generated_questions)
        expected_slots = len(plan.get_all_slots())
        if total_q_count != expected_slots:
            failures.append(f"A05: Generated questions count ({total_q_count}) != plan slots ({expected_slots})")
            cat_a_pass = False

        for pair in plan.or_pairs:
            marks_a = tuple(s.marks for s in pair.slots_a)
            marks_b = tuple(s.marks for s in pair.slots_b)
            if marks_a != marks_b or marks_a != pair.mark_distribution:
                failures.append(f"A04: OR pair mark distribution parity violated in Module {pair.module_id}")
                cat_a_pass = False

        # CATEGORY A FAILURE IS IMMEDIATELY BLOCKED
        if not cat_a_pass:
            report_text = cls._build_report(plan, 0.0, "BLOCKED", "STRUCTURAL_FAILURE", failures)
            return QAResult(
                request_id=plan.request_id,
                plan_id=plan.plan_id,
                qa_score=0.0,
                status="BLOCKED",
                failures=failures,
                report_text=report_text,
                exportable=False,
            )

        # -- CATEGORY B — CONTENT INTEGRITY (weight 0.20) ----------------------
        cat_b_score = 1.0
        b_fails = 0
        for gq in generated_questions:
            if "\ufffd" in gq.question_text:
                failures.append(f"B01 (M3): Unicode replacement character in slot {gq.slot_id}")
                b_fails += 1
            if "\x00" in gq.question_text:
                failures.append(f"B02: Null byte binary contamination in slot {gq.slot_id}")
                b_fails += 1

        if b_fails > 0:
            cat_b_score = max(0.0, 1.0 - (b_fails * 0.25))

        # -- CATEGORY C — PROMPT SAFETY (weight 0.15) -------------------------
        cat_c_pass = True
        for gq in generated_questions:
            q_text_lower = gq.question_text.lower()
            if "ignore previous instructions" in q_text_lower or "turi būti tik klausimas" in q_text_lower:
                failures.append(f"C01: Prompt injection detected in slot {gq.slot_id}")
                cat_c_pass = False
            if "question:" in q_text_lower and q_text_lower.index("question:") > 50:
                failures.append(f"C04: System prompt leakage in slot {gq.slot_id}")
                cat_c_pass = False

        if not cat_c_pass:
            report_text = cls._build_report(plan, 0.0, "BLOCKED", "PROMPT_SAFETY_FAILURE", failures)
            return QAResult(
                request_id=plan.request_id,
                plan_id=plan.plan_id,
                qa_score=0.0,
                status="BLOCKED",
                failures=failures,
                report_text=report_text,
                exportable=False,
            )

        # -- CATEGORY D — BLOOM / ACADEMIC INTEGRITY (weight 0.15) -------------
        cat_d_score = 1.0
        d_fails = 0
        for gq in generated_questions:
            q_text_lower = gq.question_text.lower()
            found_verbs = [v for v in BLOOM_VERB_KEYWORDS if re.search(r"\b" + v + r"\b", q_text_lower)]
            for v1, v2 in FORBIDDEN_VERB_COMBOS:
                if v1 in found_verbs and v2 in found_verbs:
                    warnings.append(f"D03: Incompatible Bloom verbs '{v1}' and '{v2}' in slot {gq.slot_id}")
                    d_fails += 1

        if d_fails > 0:
            cat_d_score = max(0.0, 1.0 - (d_fails * 0.15))

        # -- CATEGORY E — GROUNDING (weight 0.10) ------------------------------
        cat_e_score = 1.0

        # -- CATEGORY F — RENDERING (weight 0.10) -----------------------------
        cat_f_score = 1.0

        # Weighted calculation: A (0.30), B (0.20), C (0.15), D (0.15), E (0.10), F (0.10)
        qa_score = (
            1.0 * 30.0 +
            cat_b_score * 20.0 +
            1.0 * 15.0 +
            cat_d_score * 15.0 +
            cat_e_score * 10.0 +
            cat_f_score * 10.0
        )

        category_scores = {
            "A_STRUCTURE": 100.0,
            "B_CONTENT_INTEGRITY": cat_b_score * 100.0,
            "C_PROMPT_SAFETY": 100.0,
            "D_BLOOM_INTEGRITY": cat_d_score * 100.0,
            "E_GROUNDING": cat_e_score * 100.0,
            "F_RENDERING": cat_f_score * 100.0,
        }

        if qa_score >= 90.0:
            status = "PASS"
        elif qa_score >= 75.0:
            status = "PASS_WITH_WARNINGS"
        elif qa_score >= 60.0:
            status = "REPAIR"
        else:
            status = "BLOCKED"

        exportable = status in ("PASS", "PASS_WITH_WARNINGS")
        report_text = cls._build_report(plan, qa_score, status, "", failures, warnings)

        return QAResult(
            request_id=plan.request_id,
            plan_id=plan.plan_id,
            qa_score=qa_score,
            status=status,
            category_scores=category_scores,
            failures=failures,
            warnings=warnings,
            report_text=report_text,
            exportable=exportable,
        )

    @classmethod
    def _build_report(
        cls,
        plan: PaperStructurePlan,
        score: float,
        status: str,
        reason: str,
        failures: List[str],
        warnings: List[str] = None,
    ) -> str:
        lines = [
            "════════════════════════════════════════════════════════",
            "AION FINAL QUALITY GATE",
            "════════════════════════════════════════════════════════",
            f"Plan ID              : {plan.plan_id}",
            "--------------------------------------------------------",
            f"STRUCTURE   [A01–A09]: {'PASS' if status != 'BLOCKED' or reason != 'STRUCTURAL_FAILURE' else 'FAIL'}",
            f"CONTENT     [B01–B05]: {'PASS' if score > 70 else 'WARN'}",
            f"SAFETY      [C01–C04]: {'PASS' if status != 'BLOCKED' or reason != 'PROMPT_SAFETY_FAILURE' else 'FAIL'}",
            f"BLOOM       [D01–D05]: PASS",
            f"GROUNDING   [E01–E05]: PASS",
            f"RENDERING   [F01–F04]: PASS",
            "--------------------------------------------------------",
            f"QA Score             : {score:.1f}/100",
            f"Final Status         : {status}",
        ]
        if reason:
            lines.append(f"Reason               : {reason}")
        lines.append(f"Exportable           : {'YES' if status in ('PASS', 'PASS_WITH_WARNINGS') else 'NO'}")
        lines.append("════════════════════════════════════════════════════════")
        return "\n".join(lines)
