"""
AION Module: Self-Critic Gate
Maturity:    v0.1 — REASON-CODE BASED REVIEWER STUB
Upgrades to: Self-Critic Ensemble (Grammar, Bloom Taxonomy, Academic Validity, Discriminator Models)
Contract:    q: GeneratedQuestion -> tuple[bool, str] (accepted, reason_code)
"""

from typing import Tuple
from .schemas import GeneratedQuestion


def review(q: GeneratedQuestion) -> Tuple[bool, str]:
    """
    Self-Critic Gate with template-question and quality detection.
    """

    # RC-06: Language & Length Quality
    if len(q.question_text) < 15:
        return False, "RC-06: question too short"

    # RC-01: Concept & Answer Validity
    if not q.ideal_answer or len(q.ideal_answer) < 20:
        return False, "RC-01: ideal answer missing or too thin"

    # RC-04: Question Interrogative Check
    if "?" not in q.question_text:
        return False, "RC-04: no interrogative detected"

    # ── Reject template questions ──
    bad_templates = [
        "Critically analyze the principle:",
        "Critically analyze the statement:",
        "What are its core academic implications?",
        "[INVALID CONCEPT SKIPPED",
        "[SKIPPED:",
    ]
    for bad in bad_templates:
        if bad in q.question_text:
            return False, "RC-07: template fallback question (LLM not connected)"

    # ── Reject if question contains URLs ──
    if "http://" in q.question_text or "https://" in q.question_text:
        return False, "RC-08: question contains URLs"

    # ── Reject if ideal answer is just a template ──
    if q.ideal_answer.startswith("[SKIPPED:") or q.ideal_answer.startswith("[INVALID"):
        return False, "RC-09: ideal answer is a skip marker"

    return True, "ACCEPTED"
