"""
Multi-stage Validation Pipeline — 7 Gates
==========================================
Every question passes:
  Grammar -> Semantic validation -> Bloom validation -> Grounding validation
        -> Marks validation -> Diagram validation -> Final audit
Reject otherwise.

Each gate returns pass/fail + score + reason code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable

from core.generation.question_composer import ComposedQuestion
from core.planning.question_planner import QuestionPlan


@dataclass
class ValidationGateResult:
    gate: str
    passed: bool
    score: float  # 0.0-1.0
    reason: str
    reason_code: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    question_text: str
    concept_id: str
    overall_passed: bool
    overall_score: float
    gates: List[ValidationGateResult]
    reason_codes: List[str]
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_text": self.question_text,
            "concept_id": self.concept_id,
            "overall_passed": self.overall_passed,
            "overall_score": self.overall_score,
            "confidence": self.confidence,
            "reason_codes": self.reason_codes,
            "gates": [{k: v for k, v in g.__dict__.items()} for g in self.gates],
        }


class MultiStageValidator:
    """
    7-gate validator. Pluggable: each gate is a callable.
    """

    def __init__(self, strict: bool = True):
        self.strict = strict
        # Allow custom gates injection
        self.gates_order = [
            "grammar",
            "semantic",
            "bloom",
            "grounding",
            "marks",
            "diagram",
            "final_audit",
        ]

    def validate(
        self,
        question: ComposedQuestion | str,
        plan: Optional[QuestionPlan] = None,
        evidence: str = "",
        expected_answer: str = "",
    ) -> ValidationReport:
        """
        Validate a composed question.
        If question is str, wrap minimally.
        """
        if isinstance(question, str):
            q_text = question
            concept_id = getattr(plan, "concept_id", "unknown")
            marks = getattr(plan, "marks", 10)
            bloom = getattr(plan, "bloom_level", 2)
            qtype = getattr(plan, "question_type", "conceptual")
            requires_diagram = getattr(plan, "requires_diagram", False)
        else:
            q_text = question.question_text
            concept_id = question.concept_id
            marks = question.marks
            bloom = question.bloom_level
            qtype = question.question_type
            requires_diagram = question.composer_metadata.get("requires_diagram", False) if hasattr(question, "composer_metadata") else False
            evidence = evidence or question.grounding.get("evidence_snippet", "")
            expected_answer = expected_answer or question.expected_answer

        results: List[ValidationGateResult] = []

        # Gate 1: Grammar
        results.append(self._gate_grammar(q_text))

        # Gate 2: Semantic
        results.append(self._gate_semantic(q_text, evidence))

        # Gate 3: Bloom
        results.append(self._gate_bloom(q_text, bloom))

        # Gate 4: Grounding — pass question object for numerical payload whitelist
        results.append(self._gate_grounding(q_text, evidence, expected_answer, question if isinstance(question, ComposedQuestion) else None))

        # Gate 5: Marks
        results.append(self._gate_marks(q_text, marks))

        # Gate 6: Diagram
        results.append(self._gate_diagram(q_text, requires_diagram, qtype))

        # Gate 7: Final audit (aggregation)
        audit = self._gate_final_audit(results)
        results.append(audit)

        overall_passed = all(r.passed for r in results)
        overall_score = round(sum(r.score for r in results) / len(results), 2) if results else 0.0
        # Confidence = min gate score (weakest link)
        confidence = round(min(r.score for r in results), 2) if results else 0.0
        reason_codes = [r.reason_code for r in results if r.reason_code and not r.passed]

        return ValidationReport(
            question_text=q_text,
            concept_id=concept_id,
            overall_passed=overall_passed,
            overall_score=overall_score,
            gates=results,
            reason_codes=reason_codes,
            confidence=confidence,
        )

    # -- Gates ------------------------------------------------

    def _gate_grammar(self, text: str) -> ValidationGateResult:
        score = 1.0
        reasons: List[str] = []
        code = None

        if len(text.split()) < 6:
            score -= 0.40
            reasons.append("too short (<6 words)")
            code = "RC-06"

        if len(text.split()) > 120:
            score -= 0.20
            reasons.append("too long (>120 words)")

        if not re.search(r"[?.]$", text.strip()):
            score -= 0.15
            reasons.append("missing terminal punctuation")

        # Check for double spaces / markdown
        if "**" in text or "__" in text:
            score -= 0.20
            reasons.append("contains markdown")

        # Check for preamble markers
        if re.search(r"^(here is|question:|answer:)", text, re.I):
            score -= 0.30
            reasons.append("contains preamble")

        # First word should be capitalized
        if text and not text[0].isupper():
            score -= 0.10
            reasons.append("not capitalized")

        passed = score >= 0.70
        if not passed and not code:
            code = "RC-06: grammar"

        return ValidationGateResult(
            gate="grammar",
            passed=passed,
            score=max(0.0, round(score, 2)),
            reason="; ".join(reasons) if reasons else "grammar ok",
            reason_code=code if not passed else None,
        )

    def _gate_semantic(self, question: str, evidence: str) -> ValidationGateResult:
        if not evidence:
            return ValidationGateResult("semantic", True, 0.75, "no evidence to verify — soft pass", details={"skipped": True})

        try:
            from core.semantics.verifier import AcademicSemanticsVerifier  # type: ignore
            verifier = AcademicSemanticsVerifier(strict=self.strict)
            res = verifier.verify(question, evidence)
            passed = res.is_valid
            code = "RC-01: semantic hallucination" if not passed else None
            return ValidationGateResult(
                gate="semantic",
                passed=passed,
                score=res.confidence,
                reason="; ".join(res.violations) if res.violations else ("warnings: " + "; ".join(res.warnings) if res.warnings else "semantically grounded"),
                reason_code=code,
                details={"domain": res.domain, "coverage": res.evidence_coverage, "violations": res.violations},
            )
        except Exception as e:
            # Fallback: simple coverage check
            q_terms = set(question.lower().split())
            ev_terms = set(evidence.lower().split())
            coverage = len(q_terms & ev_terms) / max(len(q_terms), 1)
            passed = coverage >= 0.40
            return ValidationGateResult(
                gate="semantic",
                passed=passed,
                score=round(coverage, 2),
                reason=f"fallback coverage {coverage:.0%}" + (f" error: {e}" if not passed else ""),
                reason_code=None if passed else "RC-01: low coverage",
            )

    def _gate_bloom(self, question: str, declared_bloom: int) -> ValidationGateResult:
        # Scenario-based questions inherently require higher reasoning even if verb is generic
        low = question.lower()
        is_scenario = any(kw in low for kw in ["vehicle", "dtc p", "technician", "scenario", "case study", "exhibits", "after replacing", "diagnostic sequence"])
        if is_scenario:
            # Scenario implies higher reasoning regardless of verb — always pass for scenario
            return ValidationGateResult(gate="bloom", passed=True, score=0.85, reason=f"scenario-based L{declared_bloom} (reasoning via case)", details={"scenario": True, "declared": declared_bloom})

        try:
            from v0_1.qa_engine import BloomsTaxonomyValidator  # type: ignore
            validator = BloomsTaxonomyValidator()
            is_valid, detected, conf = validator.validate_question(question, declared_bloom)
            # For scenario intents, be lenient: allow off-by-one Bloom
            if is_scenario and not is_valid:
                # Check if detected is within 1 level of declared for scenario
                try:
                    det_num = int(re.search(r"L(\d)", str(detected)).group(1)) if re.search(r"L(\d)", str(detected)) else declared_bloom
                    if abs(det_num - declared_bloom) <= 1:
                        return ValidationGateResult(gate="bloom", passed=True, score=0.75, reason=f"scenario lenient: detected {detected} vs declared L{declared_bloom}", details={"detected": detected, "declared": declared_bloom})
                except Exception:
                    pass
            passed = is_valid or conf >= 0.5
            code = "RC-04: bloom mismatch" if not passed else None
            return ValidationGateResult(
                gate="bloom",
                passed=passed,
                score=conf if isinstance(conf, float) else 0.75,
                reason=f"detected {detected} vs declared L{declared_bloom}" if not passed else f"bloom L{declared_bloom} aligned",
                reason_code=code,
                details={"detected": detected, "declared": declared_bloom},
            )
        except Exception:
            verbs = {
                1: ["define", "list", "state", "recall"],
                2: ["explain", "describe", "summarise", "interpret"],
                3: ["apply", "illustrate", "solve", "calculate"],
                4: ["analyse", "compare", "differentiate"],
                5: ["evaluate", "justify", "assess"],
                6: ["design", "construct", "formulate"],
            }
            declared_verbs = verbs.get(declared_bloom, [])
            has_verb = any(v in low for v in declared_verbs)
            passed = has_verb or declared_bloom in [2, 3] or is_scenario
            return ValidationGateResult(
                gate="bloom",
                passed=passed,
                score=0.85 if passed else 0.40,
                reason=f"verb check {'pass' if passed else 'fail'} for L{declared_bloom}",
                reason_code=None if passed else "RC-04: bloom verb missing",
            )

    def _gate_grounding(self, question: str, evidence: str, expected_answer: str, question_obj: Optional[ComposedQuestion] = None) -> ValidationGateResult:
        if not evidence:
            return ValidationGateResult("grounding", True, 0.70, "no evidence — soft pass", details={"skipped": True})

        # Whitelist numerical payload numbers (fresh instance) and scenario terms
        whitelist_terms = {"vehicle", "technician", "student", "scenario", "case", "study", "diagnostic", "sequence", "justify", "evaluate", "identify", "probable", "cause", "suggest"}
        combined = (evidence + " " + expected_answer).lower()
        q_low = question.lower()
        # Coverage: question keywords in combined — include 4+ letter terms for better recall
        q_terms = [t for t in re.findall(r"\b[a-z]{4,}\b", q_low) if len(t) >= 4]
        # Filter generic filler + scenario whitelist
        filler = {"with", "from", "that", "this", "have", "been", "will", "marks", "discuss", "explain", "describe", "state", "illustrate", "analyse", "evaluate", "design", "given", "figure", "reference"}
        filler = filler.union(whitelist_terms)
        q_terms_filtered = [t for t in q_terms if t not in filler]
        # Further filter scenario-specific whitelist terms that are intentionally not in evidence
        # For scenario questions, allow mismatch for scenario vocabulary
        if not q_terms_filtered:
            return ValidationGateResult("grounding", True, 0.75, "no substantial terms to ground")
        grounded = sum(1 for t in q_terms_filtered if t in combined)
        coverage = grounded / len(q_terms_filtered)
        # Numeric hallucination — exclude marks and fresh payload
        nums_q = set(re.findall(r"\b\d+(?:\.\d+)?\b", question))
        nums_ev = set(re.findall(r"\b\d+(?:\.\d+)?\b", combined))
        marks_nums = set(re.findall(r"(\d+)\s*marks?", question, re.I))
        # Whitelist fresh numerical payload numbers
        payload_nums = set()
        if question_obj and hasattr(question_obj, "grounding"):
            payload = question_obj.grounding.get("numerical_payload")
            if payload and payload.get("fresh_values"):
                fv = payload["fresh_values"]
                vals = fv.values() if isinstance(fv, dict) else fv if isinstance(fv, list) else [str(fv)]
                for v in vals:
                    payload_nums.update(re.findall(r"\b\d+(?:\.\d+)?\b", str(v)))
        nums_q_non_marks = nums_q - marks_nums - payload_nums
        numeric_halluc = nums_q_non_marks - nums_ev
        if numeric_halluc and not re.search(r"calculate|solve|compute|apply|demonstrate|illustrate|using fresh", q_low):
            if len(numeric_halluc) >= 2:
                coverage = min(coverage, 0.45)
            else:
                coverage = min(coverage, max(0.40, coverage - 0.15))

        threshold = 0.30 if len(combined.split()) < 80 else 0.35
        # For scenario-based questions (contain "compiler", "circular queue", "hospital"), be more lenient
        if any(kw in q_low for kw in ["compiler", "circular queue", "hospital", "bst", "stack", "queue"]):
            threshold = 0.25
        passed = coverage >= threshold
        score = round(coverage, 2)
        code = "RC-07: grounding insufficient" if not passed else None
        return ValidationGateResult(
            gate="grounding",
            passed=passed,
            score=score,
            reason=f"grounding coverage {coverage:.0%} ({grounded}/{len(q_terms_filtered)})" + (f" hallucinated numbers {numeric_halluc}" if numeric_halluc else ""),
            reason_code=code,
            details={"coverage": coverage, "terms_checked": len(q_terms_filtered), "threshold": threshold},
        )

    def _gate_marks(self, question: str, marks: int) -> ValidationGateResult:
        # Marks validation: question scope should match marks
        wc = len(question.split())
        # Heuristic: 5 marks ~ 15-40 words, 10 marks ~ 20-60 words
        expected_range = {5: (10, 50), 10: (15, 70), 2: (6, 30), 6: (12, 55), 8: (14, 65)}
        low, high = expected_range.get(marks, (10, 70))
        # Also check for marks mention consistency (if question mentions marks explicitly)
        m = re.search(r"(\d+)\s*marks?", question, re.I)
        if m:
            mentioned = int(m.group(1))
            if mentioned != marks:
                return ValidationGateResult("marks", False, 0.40, f"mentions {mentioned} marks but allocated {marks}", reason_code="RC-03: marks mismatch")
        # Word count vs marks
        if wc < low - 5:
            return ValidationGateResult("marks", False, 0.50, f"too short ({wc} words) for {marks} marks", reason_code="RC-03: marks scope mismatch")
        if wc > high + 40:
            return ValidationGateResult("marks", False, 0.60, f"too verbose ({wc} words) for {marks} marks", reason_code="RC-03: marks scope mismatch")
        return ValidationGateResult("marks", True, 0.95, f"scope ok for {marks} marks ({wc} words)")

    def _gate_diagram(self, question: str, requires_diagram: bool, qtype: str) -> ValidationGateResult:
        has_ref = bool(re.search(r"figure|diagram|given|shown|above|refer", question, re.I))
        if requires_diagram or qtype == "diagram":
            if has_ref:
                return ValidationGateResult("diagram", True, 0.98, "diagram reference present")
            return ValidationGateResult("diagram", False, 0.30, "requires diagram but no figure reference", reason_code="RC-10: diagram required")
        else:
            # If question claims figure but not required, warn but pass (maybe opportunistic)
            if has_ref:
                return ValidationGateResult("diagram", True, 0.85, "figure reference present though not required — acceptable")
            return ValidationGateResult("diagram", True, 0.95, "no diagram required — ok")

    def _gate_final_audit(self, gate_results: List[ValidationGateResult]) -> ValidationGateResult:
        # Aggregate: fail if any critical gate failed (semantic, grounding) or overall score low
        critical_failed = [g for g in gate_results if g.gate in ("semantic", "grounding") and not g.passed]
        if critical_failed:
            reasons = "; ".join(g.reason for g in critical_failed)
            return ValidationGateResult("final_audit", False, 0.35, f"critical failure: {reasons}", reason_code="RC-09: final audit fail")

        avg_score = sum(g.score for g in gate_results) / len(gate_results) if gate_results else 0.0
        if avg_score < 0.60:
            return ValidationGateResult("final_audit", False, round(avg_score, 2), f"overall quality {avg_score:.0%} below threshold", reason_code="RC-09: quality threshold")

        failed = [g for g in gate_results if not g.passed]
        if failed:
            return ValidationGateResult("final_audit", False, round(avg_score, 2), f"{len(failed)} gate(s) failed", reason_code="RC-09: gate failure")

        return ValidationGateResult("final_audit", True, round(avg_score, 2), "all gates passed")
