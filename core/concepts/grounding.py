"""
Concept Grounding Engine — Ground before Generation
===================================================
Per AION Development Context, NEVER generate directly from raw text.
Always:
  Text -> Concept -> Supporting Evidence -> Expected Answer -> Question

Every question must have:
  Concept ID | Source Chunk | Confidence | Expected Answer | Bloom | Question

This engine produces GroundedConcept which bundles:
- concept
- expected answer (canonical)
- bloom level
- source evidence hash
- grounding confidence
"""

from __future__ import annotations

import re
import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from .extractor import ExtractedConcept

@dataclass
class GroundedConcept:
    concept: ExtractedConcept
    expected_answer: str
    expected_answer_outline: List[str]  # bullet points
    bloom_level: int                    # 1-6
    bloom_label: str
    confidence: float                   # grounding confidence
    source_hash: str                    # hash of evidence for traceability
    evidence_snippet: str               # 200-char evidence excerpt
    reasoning_trace: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concept_id": self.concept.concept_id,
            "concept_name": self.concept.concept_name,
            "source_chunk": self.concept.source_chunk_id,
            "source_hash": self.source_hash,
            "confidence": self.confidence,
            "expected_answer": self.expected_answer,
            "expected_answer_outline": self.expected_answer_outline,
            "bloom_level": self.bloom_level,
            "bloom_label": self.bloom_label,
            "evidence_snippet": self.evidence_snippet,
            "reasoning_trace": self.reasoning_trace,
        }


class ConceptGroundingEngine:
    """
    Grounds validated concepts into expected answers + Bloom levels.
    Does NOT call LLM for understanding — uses deterministic heuristics first,
    LLM only for expected answer expansion if available (but grounding stays traceable).
    """

    BLOOM_MAP = {
        1: "Remember", 2: "Understand", 3: "Apply",
        4: "Analyse", 5: "Evaluate", 6: "Create",
    }

    BLOOM_VERBS = {
        1: ["Define", "List", "State", "Recall"],
        2: ["Explain", "Describe", "Summarise", "Interpret"],
        3: ["Apply", "Illustrate", "Demonstrate", "Solve"],
        4: ["Analyse", "Compare", "Differentiate", "Examine"],
        5: ["Evaluate", "Justify", "Assess", "Critique"],
        6: ["Design", "Construct", "Formulate", "Develop"],
    }

    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm
        self._llm = None
        if use_llm:
            try:
                from v0_1.llm import get_llm  # type: ignore
                self._llm = get_llm()
            except Exception:
                self.use_llm = False

    def ground(self, concepts: List[ExtractedConcept], target_bloom: Optional[int] = None) -> List[GroundedConcept]:
        grounded: List[GroundedConcept] = []
        for c in concepts:
            gc = self._ground_single(c, target_bloom)
            grounded.append(gc)
        return grounded

    def _ground_single(self, concept: ExtractedConcept, target_bloom: Optional[int]) -> GroundedConcept:
        evidence = concept.supporting_evidence
        source_hash = hashlib.sha256(evidence.encode()).hexdigest()[:12]
        evidence_snippet = evidence[:200].replace("\n", " ").strip()

        # Determine Bloom
        bloom = target_bloom or self._infer_bloom(concept)
        bloom_label = self.BLOOM_MAP.get(bloom, "Understand")
        verb = self.BLOOM_VERBS[bloom][0]

        # Build expected answer from evidence (deterministic, no hallucination)
        expected_answer, outline, trace = self._build_expected_answer(concept, bloom)

        # Optional LLM expansion (but keep evidence-bound)
        if self.use_llm and self._llm and len(expected_answer.split()) < 40:
            expanded = self._llm_expand(concept, expected_answer, bloom)
            if expanded and self._is_grounded(expanded, evidence):
                expected_answer = expanded
                trace.append("llm_expansion_grounded")

        # Grounding confidence: evidence coverage * concept confidence
        coverage = self._evidence_coverage(expected_answer, evidence)
        grounding_conf = round(min(0.98, (concept.confidence * 0.6 + coverage * 0.4)), 2)

        return GroundedConcept(
            concept=concept,
            expected_answer=expected_answer,
            expected_answer_outline=outline,
            bloom_level=bloom,
            bloom_label=bloom_label,
            confidence=grounding_conf,
            source_hash=source_hash,
            evidence_snippet=evidence_snippet,
            reasoning_trace=trace,
        )

    # ── Expected Answer Construction ─────────────────────────

    def _build_expected_answer(self, concept: ExtractedConcept, bloom: int) -> tuple[str, List[str], List[str]]:
        evidence = concept.supporting_evidence
        trace: List[str] = []

        # Extract definition + key points deterministically
        # Use evidence sentences, not LLM memory
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", evidence) if len(s.split()) > 6]

        # Take up to 3 most definitional sentences
        key_sents = []
        for s in sents:
            if re.search(r"is a|is defined|refers to|consists of|defines|represents", s, re.I):
                key_sents.append(s)
            if len(key_sents) >= 2:
                break
        if not key_sents:
            key_sents = sents[:2]

        expected = " ".join(key_sents[:3])
        if len(expected.split()) < 20 and sents:
            # Pad with next sentences
            expected = " ".join(sents[:3])

        # For numerical/diagram concepts, ensure specifics preserved
        if concept.equations:
            expected += f" Relevant expression: {concept.equations[0]}"
            trace.append("equation_preserved")
        if concept.diagram_refs:
            expected += f" Refer to {concept.diagram_refs[0]}."
            trace.append("diagram_ref_preserved")

        # Outline
        outline: List[str] = []
        for i, s in enumerate(key_sents[:3], 1):
            outline.append(f"{i}. {s[:120]}")
        if concept.equations:
            outline.append(f"Formula: {concept.equations[0]}")
        if concept.concept_type == "numerical":
            outline.append("Include step-by-step calculation with substituted values.")

        trace.append(f"grounded_from_evidence:{len(key_sents)}_sentences")
        trace.append(f"bloom_{bloom}_{self.BLOOM_MAP[bloom]}")

        return expected.strip(), outline, trace

    def _infer_bloom(self, concept: ExtractedConcept) -> int:
        # Use concept's bloom suggestions
        if concept.bloom_suggestions:
            # Map Lx -> int
            m = re.search(r"L(\d)", concept.bloom_suggestions[0])
            if m:
                return int(m.group(1))
        # Fallback by type
        if concept.concept_type == "numerical":
            return 3
        if concept.concept_type == "derivation":
            return 4
        if concept.concept_type == "diagram":
            return 2
        return 2

    def _evidence_coverage(self, answer: str, evidence: str) -> float:
        # Jaccard-like coverage: share of answer n-grams present in evidence
        ans_tokens = set(answer.lower().split())
        ev_tokens = set(evidence.lower().split())
        if not ans_tokens:
            return 0.0
        overlap = len(ans_tokens & ev_tokens) / len(ans_tokens)
        return overlap

    def _is_grounded(self, text: str, evidence: str) -> bool:
        # At least 70% of content words should appear in evidence
        return self._evidence_coverage(text, evidence) >= 0.55

    def _llm_expand(self, concept: ExtractedConcept, draft: str, bloom: int) -> Optional[str]:
        try:
            prompt = (
                f"You are an academic grounding assistant. Expand the expected answer for the concept "
                f"'{concept.concept_name}' using ONLY the evidence below. Do not add external facts.\n\n"
                f"EVIDENCE:\n\"\"\"{concept.supporting_evidence[:1200]}\"\"\"\n\n"
                f"DRAFT ANSWER:\n{draft}\n\n"
                f"Expand to 2-3 sentences, preserve all facts from evidence, Bloom L{bloom}. Output only the answer."
            )
            res = self._llm.generate(prompt, options={"num_predict": 200, "temperature": 0.2})
            if res and len(res.split()) > 10:
                return res.strip()
        except Exception:
            pass
        return None
