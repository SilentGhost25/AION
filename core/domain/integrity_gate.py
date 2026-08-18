"""
Domain Integrity Gate — Safety layer before composition
Per audit: Don't rely on lexicons alone. Add gate immediately before composition:
Question -> Extract technical entities -> Compare against Knowledge Unit + Retrieved Concepts -> Any unseen domain entity? -> Reject -> Repair

For Data Structures example:
Knowledge Unit: Binary Tree, BST, Traversal, AVL
Generated question contains ECU -> ECU ∉ KU, ∉ Retrieved Evidence -> Reject

This catches the problem regardless of whether ECU is in manually maintained automotive lexicon.
"""

import re
from dataclasses import dataclass
from typing import List, Set, Dict

@dataclass
class IntegrityResult:
    passed: bool
    score: float
    violations: List[str]
    unseen_entities: List[str]

class DomainIntegrityGate:
    """Verifies every technical entity in question is grounded."""

    # Technical entity extraction: only domain-specific acronyms (all caps) and known phrases, not generic verbs
    # Separate patterns: acronyms are case-sensitive (all caps), phrases are case-insensitive
    ACRONYM_PATTERN = re.compile(r"\b[A-Z]{2,5}\b")
    PHRASE_PATTERN = re.compile(r"\b(?:binary tree|bst|avl|o2 sensor|maf sensor|ecu|dtc|dlc|can|misfire|catalyst|crankshaft|forging|lathe|satellite|antenna|beam|column|equ|esi)\b", re.I)

    STOP_ENTITIES = {"The", "This", "That", "With", "From", "Module", "Chapter", "Section", "Figure", "Table", "Marks", "Marks."}

    SCENARIO_WHITELIST = {"vehicle", "technician", "student", "case", "scenario", "study", "explain", "describe", "analyse", "illustrate", "binary", "tree"}

    def check(self, question: str, knowledge_unit_concepts: Set[str], retrieved_evidence: str, subject_profile=None) -> IntegrityResult:
        # Extract entities from question
        q_entities = self._extract_entities(question)
        # Build grounded vocabulary: KU concepts + evidence terms + subject permitted vocabulary
        grounded_vocab = set()
        for c in knowledge_unit_concepts:
            grounded_vocab.update(re.findall(r"\b\w+\b", c.lower()))
            # Add concept as phrase
            grounded_vocab.add(c.lower())
        # Add evidence terms (4+ letters)
        evidence_terms = set(re.findall(r"\b[a-z]{4,}\b", retrieved_evidence.lower())) if retrieved_evidence else set()
        grounded_vocab.update(evidence_terms)
        # Add subject permitted vocabulary if available
        if subject_profile:
            grounded_vocab.update(t.lower() for t in subject_profile.permitted_vocabulary)
            # Also whitelist scenario terms
            grounded_vocab.update(self.SCENARIO_WHITELIST)

        violations = []
        unseen = []
        for ent in q_entities:
            ent_low = ent.lower()
            # Skip stop words and whitelisted
            if ent in self.STOP_ENTITIES or ent_low in self.SCENARIO_WHITELIST or len(ent) < 3:
                continue
            # Check if entity or its words are grounded
            # For multi-word entities, check if any word is grounded
            ent_words = re.findall(r"\b\w+\b", ent_low)
            # If entity is acronym like ECU, check directly
            if ent.isupper() and len(ent) <= 5:
                # Acronym must be in grounded vocab
                if ent_low not in grounded_vocab and ent_low not in evidence_terms:
                    unseen.append(ent)
                    violations.append(f"Unseen entity '{ent}' not in Knowledge Unit nor evidence")
            else:
                # For phrase entities, check if core technical term is grounded
                # e.g., "Binary Tree" should be in KU concepts
                if ent_low not in grounded_vocab:
                    # Check if any word is technical and unseen
                    technical_words = [w for w in ent_words if len(w) >= 4 and w not in {"with", "from", "that", "this", "case", "study"}]
                    if technical_words and not any(w in grounded_vocab for w in technical_words):
                        # Only flag if word looks technical (capitalized or domain-specific)
                        if ent[0].isupper() or ent_low in ["ecu", "obd", "dtc", "maf", "avl", "bst"]:
                            unseen.append(ent)
                            violations.append(f"Unseen technical entity '{ent}'")

        # Also check for any word in question that is a subject-specific forbidden term not in evidence
        if subject_profile and subject_profile.forbidden_cross_terms:
            q_low = question.lower()
            for forbidden in subject_profile.forbidden_cross_terms:
                if re.search(r"\b" + re.escape(forbidden.lower()) + r"\b", q_low):
                    # Check if forbidden term is actually in grounded vocab (maybe document is cross-domain intentionally?)
                    if forbidden.lower() not in grounded_vocab and forbidden.lower() not in evidence_terms:
                        if forbidden not in unseen:
                            unseen.append(forbidden)
                            violations.append(f"Cross-domain forbidden term '{forbidden}' not grounded")

        passed = len(violations) == 0
        score = 1.0 if passed else max(0.2, 1.0 - len(violations) * 0.25)
        return IntegrityResult(passed=passed, score=round(score, 2), violations=violations, unseen_entities=unseen)

    def _extract_entities(self, question: str) -> List[str]:
        # Extract only domain-specific technical entities, not generic verbs like Support
        candidates = []
        # Acronyms: only all-caps like ECU, DTC, DLC, BST, AVL
        for m in self.ACRONYM_PATTERN.finditer(question):
            candidates.append(m.group())
        # Domain phrases: case-insensitive
        for m in self.PHRASE_PATTERN.finditer(question):
            candidates.append(m.group())
        return list(set(candidates))
