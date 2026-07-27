"""
AION Module: Question Generator
Maturity:    v0.1 — RAG² (REVERSE ASSESSMENT GENERATION) ENGINE
Upgrades to: Fine-Tuned Fine-Grained Academic Examiner LLM / MoE
Contract:    concept: Concept -> GeneratedQuestion (see schemas.py)
             MUST enforce ideal_answer generation BEFORE question_text generation.
"""

import os
import re
from .schemas import Concept, GeneratedQuestion
from .content_validator import validate_chunk, clean_chunk

from .llm import get_llm

def _run_prompt(prompt: str, max_tokens: int = 250) -> str:
    try:
        res = get_llm().generate(prompt)
        if res:
            return res
    except Exception:
        pass
    return ""


def generate(concept: Concept, mode: str = "balanced") -> GeneratedQuestion:
    """
    RAG² enforced: validate chunk quality, generate the IDEAL ANSWER first,
    then reverse-generate a question that elicits that answer.

    Modes:
    - "turbo": 5 Marks, 150 words max (Maximum Speed Mode)
    - "balanced": 10 Marks, 250-300 words (Speed + Depth Mode)
    - "deep": 20 Marks, 400-500 words (Examiner-Level Model Answer)
    """
    mode = mode.lower()
    if mode not in {"turbo", "balanced", "deep"}:
        mode = "balanced"

    # Set parameters by mode
    if mode == "turbo":
        marks = 5
        max_tokens = 150
        answer_instructions = """Generate a concise VTU-style answer.
Marks: 5
Word limit: 150 words maximum.
Prioritize brevity and fast response. Avoid redundancy.
No explanation outside the answer. No meta commentary.
Direct exam-ready answer only."""
    elif mode == "deep":
        marks = 20
        max_tokens = 500
        answer_instructions = """Generate a comprehensive VTU-style model answer.
Marks: 20
Include:
- Clear definition
- Explanation
- Advantages and disadvantages
- Comparison (if applicable)
- Conclusion
Word limit: 400–500 words.
Strict academic tone. No meta commentary."""
    else:  # balanced
        marks = 10
        max_tokens = 300
        answer_instructions = """Generate a VTU-style descriptive answer.
Marks: 10
Word limit: 250–300 words.
Include definition, key points, and brief conclusion.
No extra commentary.
Exam-ready structured format."""

    # ── Step 0: Validate & Clean Concept Prose ──
    cleaned_content = clean_chunk(concept.content)
    quality = validate_chunk(cleaned_content)

    if not quality.is_valid:
        # Return flagged question marked skipped
        return GeneratedQuestion(
            concept_id=concept.concept_id,
            ideal_answer="[SKIPPED: Non-academic code or noise fragment]",
            question_text=f"[INVALID CONCEPT SKIPPED: {quality.reason}]",
            marks=0,
            bloom_level=0,
        )

    # Sanitize content snippet from code variable names / file paths
    clean_snippet = re.sub(r"https?://\S+", "", cleaned_content)
    clean_snippet = re.sub(r"\b[\w_\-]+\.(py|html|js|css|json|yaml)\b", "", clean_snippet)
    clean_snippet = clean_snippet.strip().rstrip(".")

    # ── Step 1: Ideal Answer (Target Generation) ──
    answer_prompt = f"""You are an academic exam expert for VTU engineering.
{answer_instructions}

Based ONLY on this academic concept:
{clean_snippet}

Rules:
- Focus ONLY on academic concepts, definitions, and theory
- Do NOT generate answers about code syntax, file paths, or variable names

Ideal Answer:"""

    ideal_answer = _run_prompt(answer_prompt, max_tokens=max_tokens)
    if not ideal_answer:
        # LLM offline — construct a structured answer from the concept itself
        key_points = []
        sentences = re.split(r"(?<=[.?!])\s+", clean_snippet)
        for s in sentences[:5]:
            s = s.strip()
            if len(s) > 20:
                key_points.append(s)
        if key_points:
            ideal_answer = "Key points:\n" + "\n".join(f"- {kp}" for kp in key_points)
        else:
            ideal_answer = f"The concept relates to: {clean_snippet[:300]}"

    # ── Step 2: Reverse-generate the Question ──
    question_prompt = f"""You are a university examiner.
Write ONE descriptive exam question using VTU command verbs (Explain / Define / Derive / Compare / Analyze).
Target Marks: {marks}

Ideal Answer:
{ideal_answer}

Rules:
- NO questions about code syntax, file names, or variable names
- Focus on academic concept understanding

Exam Question:"""

    question_text = _run_prompt(question_prompt, max_tokens=100)
    if not question_text:
        snippet = clean_snippet[:80]
        if "defined as" in clean_snippet.lower() or "is a" in clean_snippet.lower():
            question_text = f"Define and explain the concept of '{snippet}'. What are its key characteristics and applications?"
        elif "algorithm" in clean_snippet.lower() or "method" in clean_snippet.lower():
            question_text = f"Explain the working of '{snippet}' with suitable examples. What are its advantages and limitations?"
        elif "theorem" in clean_snippet.lower() or "law" in clean_snippet.lower():
            question_text = f"State and explain '{snippet}'. Derive the key result and discuss its significance."
        else:
            question_text = f"Explain in detail: '{snippet}'. Discuss its significance in the context of the subject."

    # Ensure question mark
    if not question_text.strip().endswith("?"):
        question_text = question_text.strip() + "?"

    return GeneratedQuestion(
        concept_id=concept.concept_id,
        ideal_answer=ideal_answer,
        question_text=question_text,
        marks=marks,
        bloom_level=concept.bloom_dna or 2,
    )
