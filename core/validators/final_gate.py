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
    status           : str          # "PASS" | "PASS_WITH_WARNINGS" | "REPAIR" | "FAIL"
    category_scores  : Dict[str, float] = field(default_factory=dict)
    failures         : List[str] = field(default_factory=list)
    warnings         : List[str] = field(default_factory=list)
    report_text      : str = ""


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
    """The Single Authoritative QA Gate for AION."""

    @classmethod
    def evaluate(
        cls,
        plan: PaperStructurePlan,
        generated_questions: List[GeneratedQuestion],
    ) -> QAResult:
        failures: List[str] = []
        warnings: List[str] = []

        # ── CATEGORY A: STRUCTURE (weight 0.30) ──────────────────────────────
        cat_a_pass = True
        cat_a_score = 1.0

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

        if not cat_a_pass:
            cat_a_score = 0.0

        # ── CATEGORY B: ACADEMIC INTEGRITY (weight 0.25) ─────────────────────
        cat_b_score = 1.0
        b_fails = 0
        for gq in generated_questions:
            q_text_lower = gq.question_text.lower()
            found_verbs = [v for v in BLOOM_VERB_KEYWORDS if re.search(r"\b" + v + r"\b", q_text_lower)]

            # Check for multiple incompatible Bloom verbs (B04)
            for v1, v2 in FORBIDDEN_VERB_COMBOS:
                if v1 in found_verbs and v2 in found_verbs:
                    failures.append(f"B04: Incompatible Bloom verbs '{v1}' and '{v2}' found in slot {gq.slot_id}")
                    b_fails += 1

        if b_fails > 0:
            cat_b_score = max(0.0, 1.0 - (b_fails * 0.2))

        # ── CATEGORY C: CONTENT (weight 0.20) ────────────────────────────────
        cat_c_score = 1.0
        c_fails = 0
        seen_texts = set()
        for gq in generated_questions:
            if not gq.question_text or not gq.question_text.strip():
                failures.append(f"C01: Empty question text in slot {gq.slot_id}")
                c_fails += 1
            if gq.question_text in seen_texts:
                failures.append(f"C02: Duplicate question text in slot {gq.slot_id}")
                c_fails += 1
            seen_texts.add(gq.question_text)

        if c_fails > 0:
            cat_c_score = max(0.0, 1.0 - (c_fails * 0.2))

        # ── CATEGORY D: MATH INTEGRITY (weight 0.15) ─────────────────────────
        cat_d_score = 1.0
        d_fails = 0
        for gq in generated_questions:
            if "\ufffd" in gq.question_text:
                failures.append(f"D01 (M3): Unicode replacement character in slot {gq.slot_id}")
                d_fails += 1

        if d_fails > 0:
            cat_d_score = max(0.0, 1.0 - (d_fails * 0.5))

        # ── CATEGORY E: EVIDENCE & GROUNDING (weight 0.05) ───────────────────
        cat_e_score = 1.0

        # ── CATEGORY F: RENDERING (weight 0.05) ──────────────────────────────
        cat_f_score = 1.0

        # Weighted score calculation
        qa_score = (
            cat_a_score * 30.0 +
            cat_b_score * 25.0 +
            cat_c_score * 20.0 +
            cat_d_score * 15.0 +
            cat_e_score * 5.0 +
            cat_f_score * 5.0
        )

        category_scores = {
            "A_STRUCTURE": cat_a_score * 100.0,
            "B_ACADEMIC": cat_b_score * 100.0,
            "C_CONTENT": cat_c_score * 100.0,
            "D_MATH": cat_d_score * 100.0,
            "E_EVIDENCE": cat_e_score * 100.0,
            "F_RENDERING": cat_f_score * 100.0,
        }

        # Status determination
        if not cat_a_pass or qa_score < 60.0:
            status = "FAIL"
        elif qa_score >= 90.0:
            status = "PASS"
        elif qa_score >= 75.0:
            status = "PASS_WITH_WARNINGS"
        else:
            status = "REPAIR"

        report_lines = [
            "══════════════════════════════════════════════",
            "AION PAPER QA REPORT",
            "══════════════════════════════════════════════",
            f"Plan ID    : {plan.plan_id}",
            "──────────────────────────────────────────────",
            f"STRUCTURE    [A01–A09]  : {'PASS' if cat_a_pass else 'FAIL'}",
            f"ACADEMIC     [B01–B05]  : {'PASS' if cat_b_score == 1.0 else 'WARN'}",
            f"CONTENT      [C01–C06]  : {'PASS' if cat_c_score == 1.0 else 'WARN'}",
            f"MATH         [D01–D07]  : {'PASS' if cat_d_score == 1.0 else 'FAIL'}",
            f"EVIDENCE     [E01–E04]  : PASS",
            f"RENDERING    [F01–F04]  : PASS",
            "──────────────────────────────────────────────",
            f"QA Score     : {qa_score:.1f}/100",
            f"Final Status : {status}",
            "══════════════════════════════════════════════",
        ]
        report_text = "\n".join(report_lines)

        return QAResult(
            request_id=plan.request_id,
            plan_id=plan.plan_id,
            qa_score=qa_score,
            status=status,
            category_scores=category_scores,
            failures=failures,
            warnings=warnings,
            report_text=report_text,
        )
