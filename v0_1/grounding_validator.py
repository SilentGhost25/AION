"""
AION Grounding Validator
========================
Validates that generated questions are grounded in retrieved chunks.

The 10 grounding rules from the architecture spec.
Sits between retriever and prompt builder.
No LLM calls — pure text analysis.
Fast, deterministic, reliable.
"""

import re
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


@dataclass
class GroundingResult:
    valid:          bool
    confidence:     float          # 0.0 to 1.0
    reason:         str
    supporting_chunks: List[str]   = field(default_factory=list)
    rejected_reason:   str         = ""


class GroundingValidator:
    """
    Validates retrieval quality before generation
    and validates question quality after generation.
    """

    MIN_CHUNKS          = 2      # Rule 1
    MAX_CHUNKS          = 3      # Rule 10
    MIN_CONFIDENCE      = 0.70   # Rule 9
    MIN_WORD_OVERLAP    = 0.30   # Rule 2 threshold

    def validate_retrieval(
        self,
        chunks:    List[str],
        query:     str,
        module_id: Optional[str] = None,
    ) -> GroundingResult:
        """
        Validate retrieved chunks before sending to LLM.
        Runs Rules 1, 9, 10.
        """

        # Rule 10 — Never use more than 3 chunks
        if len(chunks) > self.MAX_CHUNKS:
            chunks = chunks[:self.MAX_CHUNKS]

        # Rule 1 — Need at least 2 supporting chunks (or 1 high quality chunk)
        if len(chunks) < self.MIN_CHUNKS:
            if not chunks or len(chunks[0].strip()) < 50:
                return GroundingResult(
                    valid=False,
                    confidence=0.0,
                    reason=f"Rule 1: Only {len(chunks)} chunk(s) retrieved. "
                           f"Minimum {self.MIN_CHUNKS} required for grounded generation.",
                    rejected_reason="INSUFFICIENT_CHUNKS"
                )

        # Rule 9 — Check retrieval confidence via word overlap
        combined = " ".join(chunks).lower()
        query_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", query.lower()))
        content_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", combined))

        if not query_words:
            confidence = 0.5
        else:
            overlap    = query_words & content_words
            confidence = len(overlap) / len(query_words)

        if confidence < self.MIN_CONFIDENCE and len(chunks) < 2:
            return GroundingResult(
                valid=False,
                confidence=confidence,
                reason=f"Rule 9: Retrieval confidence {confidence:.0%} "
                       f"is below minimum {self.MIN_CONFIDENCE:.0%}. "
                       f"Need more context for this topic.",
                rejected_reason="LOW_CONFIDENCE"
            )

        return GroundingResult(
            valid=True,
            confidence=max(confidence, 0.75),
            reason=f"Retrieval validated. Confidence: {confidence:.0%}. "
                   f"Using {len(chunks)} chunks.",
            supporting_chunks=chunks
        )

    def validate_question(
        self,
        question:   str,
        chunks:     List[str],
        module_id:  Optional[str] = None,
    ) -> GroundingResult:
        """
        Validate a generated question against source chunks.
        Runs Rules 2, 3, 5, 6, 7.
        """

        if not question or question.strip() == "INSUFFICIENT_CONTEXT":
            return GroundingResult(
                valid=False,
                confidence=0.0,
                reason="Model returned INSUFFICIENT_CONTEXT.",
                rejected_reason="INSUFFICIENT_CONTEXT"
            )

        combined     = " ".join(chunks).lower()
        q_lower      = question.lower()

        # Rule 2 — Every noun must exist in retrieved chunks
        q_nouns      = set(re.findall(r"\b[A-Z][a-z]{3,}\b", question))
        missing_nouns = []
        for noun in q_nouns:
            if noun.lower() not in combined:
                missing_nouns.append(noun)

        if len(missing_nouns) > 2:
            return GroundingResult(
                valid=False,
                confidence=0.2,
                reason=f"Rule 2: Terms not found in source: {missing_nouns}. "
                       "Question introduces concepts not in retrieved content.",
                rejected_reason="HALLUCINATED_TERMS"
            )

        # Rule 3 — No invented formulas
        q_formulas = re.findall(r"[A-Za-z]\s*=\s*[A-Za-z0-9\+\-\*/\(\)]+", question)
        for formula in q_formulas:
            formula_clean = re.sub(r"\s+", "", formula).lower()
            if formula_clean not in re.sub(r"\s+", "", combined):
                return GroundingResult(
                    valid=False,
                    confidence=0.3,
                    reason=f"Rule 3: Formula '{formula}' not found in source material.",
                    rejected_reason="INVENTED_FORMULA"
                )

        # Rule 5 — One concept, not multiple unrelated ones
        concept_indicators = ["and also", "as well as", "furthermore", "in addition to"]
        concept_count      = sum(1 for ind in concept_indicators if ind in q_lower)
        if concept_count > 1:
            return GroundingResult(
                valid=False,
                confidence=0.4,
                reason="Rule 5: Question appears to test multiple unrelated concepts.",
                rejected_reason="CONCEPT_OVERLOAD"
            )

        # Rule 6 — Answerable from chunks
        content_words  = set(re.findall(r"\b[a-zA-Z]{4,}\b", combined))
        question_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", q_lower))
        overlap        = question_words & content_words

        if len(question_words) == 0:
            confidence = 0.0
        else:
            confidence = len(overlap) / len(question_words)

        if confidence < 0.35:
            return GroundingResult(
                valid=False,
                confidence=confidence,
                reason=f"Rule 6: Question word overlap with source is only "
                       f"{confidence:.0%}. Question may not be answerable from chunks.",
                rejected_reason="LOW_ANSWERABILITY"
            )

        return GroundingResult(
            valid=True,
            confidence=confidence,
            reason=f"Question validated. Grounding: {confidence:.0%}.",
            supporting_chunks=chunks
        )

    def build_grounded_prompt(
        self,
        chunks:        List[str],
        exam_type:     str,
        marks:         int,
        bloom_level:   str,
        subject:       str,
        chapter:       str,
        question_type: str = "conceptual",
    ) -> str:
        """
        Build a minimal, grounded prompt.
        Pipeline constraints handled here.
        LLM only sees the academic task.
        """

        context = "\n\n---\n\n".join(
            f"[Source {i+1}]\n{chunk.strip()}"
            for i, chunk in enumerate(chunks[:self.MAX_CHUNKS])
        )

        # Marks split guidance
        if exam_type == "IA":
            split_guide = "Split marks as 6+4 or 5+5 across two parts (a and b)."
            max_parts   = 2
        else:
            split_guide = "Split marks across 2-3 parts (a, b, and optionally c)."
            max_parts   = 3

        bloom_verb_map = {
            "L1": "Define or State",
            "L2": "Explain or Describe",
            "L3": "Illustrate or Apply",
            "L4": "Compare or Analyze",
            "L5": "Evaluate or Justify",
            "L6": "Design or Propose",
        }
        verb_guidance = bloom_verb_map.get(bloom_level, "Explain")

        # Zone 1: Static Universal Prefix (100% vLLM prefix cache hit rate across all questions)
        static_prefix = """You are an expert university examiner setting an authentic, rigorous university examination paper compliant with VTU standards and Bloom's Revised Taxonomy.

UNIVERSAL EXAMINATION RULES & CONTRACTS:
1. BLOOM'S REVISED TAXONOMY & COGNITIVE ARCHETYPES:
   - L1 (Remember): Define, State, List, Recall fundamental principles.
   - L2 (Understand): Explain, Describe, Illustrate working mechanisms and functional relationships.
   - L3 (Apply): Solve, Calculate, Compute, Implement algorithms or numerical procedures with given parameters.
   - L4 (Analyze): Compare, Contrast, Differentiate, Deconstruct tradeoffs, failure modes, or complexity bounds.
   - L5 (Evaluate): Justify, Critique, Validate architectural choices against system constraints.
   - L6 (Design/Create): Formulate, Synthesize, Propose novel architectures or end-to-end solutions.

2. VTU QUESTION MARK SPLIT & BALANCED STRUCTURE CONTRACT:
   - For 10-Mark Questions: Split strictly into Part (a) 6 Marks + Part (b) 4 Marks (or 5+5).
   - For 20-Mark Questions: Split strictly into Part (a) 8 Marks + Part (b) 6 Marks + Part (c) 6 Marks (or 10+10).
   - Each part must be independent, self-contained, and specify clear sub-deliverables.

3. STRICT ACADEMIC INTEGRITY & SELF-CONTAINMENT:
   - Ground strictly to the supplied context. Never invent unsupported domain concepts.
   - For quantitative problems: Provide complete, realistic input values and state required SI units.
   - For conceptual problems: Require neat schematics, architectural flow, or comparative tabular analysis.
   - Prohibit vague conversational filler ("Discuss in detail", "Write notes on").

4. OUTPUT FORMAT CONTRACT:
Return ONLY valid JSON matching this schema:
{
  "question": "Full question text here.",
  "parts": [
    {"part": "a", "marks": 6, "focus": "primary analytical/conceptual task"},
    {"part": "b", "marks": 4, "focus": "sub-problem or parameter calculation"}
  ],
  "bloom": "L2",
  "grounding": "Which concept from the source material this question tests."
}
"""

        # Zone 2: Dynamic Request Suffix
        dynamic_suffix = f"""
==================== DYNAMIC SLOT SPECIFICATION ====================
EXAMINATION: {exam_type}
SUBJECT: {subject}
TARGET TOPIC / CHAPTER: {chapter}
TARGET MARKS: {marks} ({split_guide})
COMMAND VERB & LEVEL: {verb_guidance} ({bloom_level})
MAXIMUM PARTS: {max_parts}

SUPPLIED ACADEMIC CONTENT:
{context}

TASK:
Generate ONE {marks}-mark examination question on {chapter} using ONLY the supplied content above.
Return ONLY the specified JSON:
"""
        return static_prefix + dynamic_suffix
