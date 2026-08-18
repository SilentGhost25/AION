"""
Self-Critic — Reasoning-level review before Auditor
Validates: reasoning operations, expected answer alignment, examiner style, Bloom via operations (not verb)
Sits after Composer, before Auditor: Composer -> Self-Critic -> Auditor
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Optional
from core.knowledge.knowledge_unit import KnowledgeUnit
from core.reasoning.reasoning_engine import ReasoningIntent

@dataclass
class CriticResult:
    passed: bool
    score: float
    reason: str
    reason_code: Optional[str]
    details: Dict

class SelfCritic:
    """Checks if question actually exercises intended reasoning."""

    def critique(self, question: str, ku: KnowledgeUnit, intent: ReasoningIntent, expected_answer: str) -> CriticResult:
        low_q = question.lower()
        low_exp = expected_answer.lower()
        # 1. Reasoning alignment: does question require the intended operations?
        ops = intent.reasoning_operations
        # Simple heuristic: check if question contains operation cues
        op_cues = {
            "identify": ["identify", "determine", "recognize"],
            "evaluate": ["evaluate", "justify", "assess", "critique"],
            "sequence": ["sequence", "order", "steps", "next"],
            "justify": ["justify", "explain why", "reason"],
            "diagnose": ["diagnose", "probable cause", "interpret"],
            "compare": ["compare", "differentiate", "contrast"],
            "calculate": ["calculate", "compute", "determine"],
            "predict_mistake": ["what if", "predict", "mistake", "confuse", "incorrectly assumes", "misconception"],
            "explain_correct": ["explain correct", "clarify", "correct interpretation"],
        }
        # Not strict fail, but warn if operation not hinted
        missing_ops = []
        for op in ops:
            cues = op_cues.get(op, [op])
            if not any(c in low_q for c in cues):
                missing_ops.append(op)

        # 2. Expected answer alignment: question should be answerable from expected answer (not evidence copy)
        # Check if key terms from expected answer appear in question's implied answer scope
        exp_terms = set(re.findall(r"\b[a-z]{5,}\b", low_exp))
        q_terms = set(re.findall(r"\b[a-z]{5,}\b", low_q))
        # Question should not directly quote expected answer but should be related
        # Instead check that question's scenario terms are in KU evidence/procedure
        # For now, pass if not disjoint
        overlap = len(exp_terms & q_terms) / max(len(exp_terms), 1)
        # Overlap should be moderate: too high (near 1.0) suggests question copies answer; too low suggests misalignment
        # Ideal 0.3-0.6 for scenario questions
        # For recall, higher overlap okay

        # 3. Examiner style: check for scenario vs generic "explain"
        is_scenario = intent.intent_type in ("scenario", "misconception", "procedure")
        is_generic = question.lower().startswith(("explain", "describe", "discuss")) and len(question.split()) < 20
        style_score = 0.9 if is_scenario else 0.6 if is_generic else 0.8

        # 4. Bloom via reasoning: Bloom should be determined by operations, not verb
        op_bloom = {"identify": 1, "recall": 1, "explain": 2, "interpret": 2, "apply": 3, "calculate": 3, "diagnose": 4, "compare": 4, "sequence": 4, "evaluate": 5, "justify": 5, "predict_mistake": 5, "relate": 4, "predict": 4, "explain_correct": 2, "relate": 4}
        implied_bloom = max([op_bloom.get(op, 2) for op in ops], default=2) if ops else 2
        bloom_match = abs(implied_bloom - intent.bloom_target) <= 1

        # Aggregate
        critic_score = 0.9
        reasons = []
        code = None
        if missing_ops and len(missing_ops) == len(ops):
            critic_score -= 0.25
            reasons.append(f"no reasoning cue for operations {missing_ops}")
        if style_score < 0.7 and is_scenario:
            critic_score -= 0.15
            reasons.append("scenario intent but question is generic explain")
        if not bloom_match:
            critic_score -= 0.2
            reasons.append(f"bloom via operations {implied_bloom} != target {intent.bloom_target}")
            code = "RC-04: bloom via reasoning mismatch"
        if overlap > 0.75:
            critic_score -= 0.1
            reasons.append("question copies expected answer verbatim")

        passed = critic_score >= 0.65 and bloom_match
        if not passed and not code:
            code = "RC-05: self-critic reasoning"

        return CriticResult(
            passed=passed, score=round(critic_score, 2),
            reason="; ".join(reasons) if reasons else "reasoning aligned",
            reason_code=code if not passed else None,
            details={"missing_ops": missing_ops, "overlap": round(overlap,2), "style_score": style_score, "implied_bloom": implied_bloom}
        )
