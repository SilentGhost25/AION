"""
Academic Semantics Engine — Semantic Constraints & Verification
===============================================================
Per AION Development Context:

Mechanical Engineering should never use Binary Tree unless explicitly present.
Automotive should never generate Diesel ignition from SI engine chapter.
Satellite Communication should never mention Radar unless grounded.

Need semantic verifier before question approval.

Approach:
- Domain lexicons per department
- Concept grounding check: every domain-specific term must appear in evidence
- Cross-domain contamination detection
- Bloom-appropriate verb check orthogonal (validation pipeline handles Bloom, this handles domain)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Dict, Set, Optional


@dataclass
class SemanticsResult:
    is_valid: bool
    confidence: float
    violations: List[str]
    warnings: List[str]
    domain: str
    evidence_coverage: float


class AcademicSemanticsVerifier:
    """
    Verifies question semantics against evidence and domain constraints.
    """

    # Domain lexicons: terms that are exclusive to a domain
    # If evidence does NOT contain domain-exclusive term, question must NOT invent it
    DOMAIN_LEXICONS: Dict[str, Set[str]] = {
        "mechanical": {
            "lathe", "milling", "forging", "casting", "welding", "thermodynamics",
            "entropy", "enthalpy", "rankine", "otto cycle", "diesel", "si engine",
            "ci engine", "brake", "clutch", "gear", "cam", "follower",
        },
        "cse": {
            "binary tree", "bst", "avl", "hash table", "quick sort", "merge sort",
            "dijkstra", "tcp", "ip", "osi", "compiler", "parser", "deadlock",
        },
        "electronics": {
            "diode", "transistor", "mosfet", "op-amp", "rectifier", "modulation",
            "antenna", "satellite", "tdma", "fdma", "cdma", "qpsk", "bpsk",
        },
        "civil": {
            "beam", "column", "slab", "foundation", "surveying", "concrete",
            "cement", "aggregate", "levelling", "contour", "benchmark",
        },
        "electrical": {
            "transformer", "induction", "synchronous", "relay", "circuit breaker",
            "power factor", "phasor", "impedance",
        },
        "automotive": {
            "si engine", "ci engine", "diesel", "petrol", "ignition", "carburetor",
            "fuel injection", "exhaust", "chassis", "suspension",
        },
        "satcom": {
            "satellite", "orbit", "transponder", "uplink", "downlink", "tdma",
            "fdma", "beacon", "look angle", "azimuth", "elevation", "geostationary",
        },
    }

    # Forbidden cross-domain hallucinations (examples from brief)
    # If evidence is about SI engine, question must not mention Diesel ignition
    ANTI_HALLUCINATION_RULES = [
        {
            "evidence_must_contain": ["si engine", "spark ignition"],
            "forbidden_in_question": ["diesel", "ci engine", "compression ignition"],
            "reason": "SI engine chapter must not hallucinate Diesel/CI concepts",
        },
        {
            "evidence_must_contain": ["satellite communication", "satcom", "transponder", "tdma", "fdma", "geostationary", "satellite"],
            "forbidden_in_question": ["radar", "sonar"],
            "reason": "Satellite Communication must not hallucinate Radar unless grounded",
        },
        {
            "evidence_must_contain": ["quick sort", "sorting"],
            "forbidden_in_question": ["binary tree", "graph traversal", "dfs", "bfs"],
            "reason": "Sorting chapter must not hallucinate graph structures",
        },
        {
            "evidence_must_contain": ["mechanical", "thermodynamics"],
            "forbidden_in_question": ["binary tree", "hash table", "tcp"],
            "reason": "Mechanical Engineering must not use CS terms unless present",
        },
    ]

    def __init__(self, strict: bool = True):
        self.strict = strict

    def verify(
        self,
        question_text: str,
        evidence: str,
        concept_name: str = "",
        domain_hint: Optional[str] = None,
    ) -> SemanticsResult:
        q_low = question_text.lower()
        ev_low = evidence.lower()
        violations: List[str] = []
        warnings: List[str] = []

        # 1. Anti-hallucination rules
        for rule in self.ANTI_HALLUCINATION_RULES:
            # Check if evidence matches trigger (any must_contain present -> rule active)
            trigger_present = any(term.lower() in ev_low for term in rule["evidence_must_contain"])
            # Also trigger if concept_name indicates domain
            if concept_name:
                trigger_present = trigger_present or any(term.lower() in concept_name.lower() for term in rule["evidence_must_contain"])
            if not trigger_present:
                continue
            # If rule active, check forbidden terms in question
            for forbidden in rule["forbidden_in_question"]:
                if forbidden.lower() in q_low and forbidden.lower() not in ev_low:
                    violations.append(
                        f"Hallucination: '{forbidden}' in question not grounded in evidence — {rule['reason']}"
                    )

        # 2. Domain lexicon contamination
        # Detect domain of evidence
        evidence_domain = self._detect_domain(ev_low)
        question_domains = self._detect_domains_in_text(q_low)
        for q_domain in question_domains:
            if q_domain != evidence_domain and evidence_domain != "unknown" and q_domain != "unknown":
                # Check if lexicon terms from q_domain appear in question but NOT in evidence
                lex = self.DOMAIN_LEXICONS.get(q_domain, set())
                for term in lex:
                    if term.lower() in q_low and term.lower() not in ev_low:
                        # Only violation if evidence domain lexicon does NOT alsocontain term
                        ev_lex = self.DOMAIN_LEXICONS.get(evidence_domain, set())
                        if term.lower() not in ev_lex:
                            warnings.append(
                                f"Cross-domain term '{term}' ({q_domain}) not grounded in {evidence_domain} evidence"
                            )
                            if self.strict:
                                violations.append(f"Cross-domain hallucination: '{term}'")

        # 3. Evidence coverage for domain terms
        # If question mentions a technical term, it should be in evidence (or be generic)
        q_terms = re.findall(r"\b[a-z]{4,}\b", q_low)
        ev_terms = set(re.findall(r"\b[a-z]{4,}\b", ev_low))
        hallucinated = []
        for term in set(q_terms):
            if term in ev_terms:
                continue
            # Check if term is domain-specific
            is_domain_specific = any(term in lex for lex in self.DOMAIN_LEXICONS.values())
            if is_domain_specific:
                hallucinated.append(term)
        # Only flag if many hallucinated terms
        coverage = 1.0 - min(1.0, len(hallucinated) / max(len(set(q_terms)), 1))
        if coverage < 0.6 and hallucinated:
            # Not necessarily violation, but low confidence
            warnings.append(f"Low evidence coverage: hallucinated domain terms {hallucinated[:5]}")
            if coverage < 0.4:
                violations.append(f"Low grounding — only {coverage:.0%} of question terms in evidence")

        # 4. Numeric/formula grounding
        nums_in_q = set(re.findall(r"\b\d+(?:\.\d+)?\b", question_text))
        nums_in_ev = set(re.findall(r"\b\d+(?:\.\d+)?\b", evidence))
        # Allow fresh numerical payload numbers even if not in evidence — handled elsewhere
        # Here we just warn if numbers appear but none in evidence and not numerical type
        if nums_in_q and not nums_in_ev:
            # Check if question is supposed to be numerical — if not, this is hallucination
            warnings.append(f"Numbers {nums_in_q} in question but none in evidence (possible hallucination)")

        is_valid = len(violations) == 0
        confidence = 0.95 if is_valid and not warnings else 0.75 if is_valid else 0.35
        # Adjust by coverage
        confidence = round(coverage * 0.5 + confidence * 0.5, 2) if hallucinated else confidence

        return SemanticsResult(
            is_valid=is_valid,
            confidence=confidence,
            violations=violations,
            warnings=warnings,
            domain=evidence_domain,
            evidence_coverage=round(coverage, 2),
        )

    # ── Domain Detection ─────────────────────────────────────

    def _detect_domain(self, text: str) -> str:
        scores: Dict[str, int] = {}
        for domain, lex in self.DOMAIN_LEXICONS.items():
            scores[domain] = sum(1 for term in lex if term.lower() in text)
        if not any(scores.values()):
            return "unknown"
        return max(scores, key=scores.get)

    def _detect_domains_in_text(self, text: str) -> Set[str]:
        hits: Set[str] = set()
        for domain, lex in self.DOMAIN_LEXICONS.items():
            if any(term.lower() in text for term in lex):
                hits.add(domain)
        return hits if hits else {"unknown"}
