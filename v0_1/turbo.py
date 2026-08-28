from core.generation.marks_partitioner import get_user_split
"""
AION Module: Turbo Mode Direct Question Generator
Maximum Speed Mode: Bypasses answer-first generation & LLM self-critic.
Uses strict question-only prompt, stop sequences, num_predict cap, and instant regex validation.
"""

import re
from typing import Tuple
from .schemas import Concept, GeneratedQuestion
from .content_validator import validate_chunk, clean_chunk
from .llm import get_llm

TURBO_QUESTION_PROMPT = """You are a VTU exam paper setter.

SOURCE MATERIAL:
\"\"\"{chunk}\"\"\"

TASK: Generate ONE descriptive exam question worth {marks} marks.

STRICT CONSTRAINTS:
- Output ONLY the raw question text. 1-2 sentences maximum.
- Start immediately with a VTU command verb (Explain, Compare, Derive, Analyse, Illustrate, Describe, Define, Discuss).
- DO NOT include an Ideal Answer.
- DO NOT include a Marking Scheme.
- DO NOT include any Notes, hints, or preamble.
- DO NOT write "Here is a question" or "Question:".

Start your response directly with the verb:"""

TURBO_STOP_SEQUENCES = [
    "\n\n",           # Stop at double newline (end of question)
    "Ideal Answer", 
    "Marking Scheme",
    "Note:",
    "Answer:",
    "Explanation:",
    "Here is",
    "Question:",      # Stop if it tries to prepend "Question:"
]


def clean_turbo_output(raw_text: str) -> str:
    """Strips any preamble, notes, or markdown artifacts from turbo generation."""
    t = raw_text.strip()
    # Remove preambles like "Here is a question:" or "Sure, here is..."
    t = re.sub(r"^(here('s| is)|sure|certainly|below is)[^\n]*[:\n]+", "", t, flags=re.I)
    # Remove trailing notes
    t = re.sub(r"\n\s*(\*\*)?Note:?.*$", "", t, flags=re.S|re.I)
    # Remove "Question:" prefix if the model added it
    t = re.sub(r"^\**Question\**\s*:?\s*", "", t, flags=re.I)
    # Remove bold markdown wrappers
    t = t.strip().strip("*").strip()

    # Ensure question mark at end
    if t and not t.endswith("?"):
        t += "?"
    return t


def generate_turbo(concept: Concept, marks: int = 5) -> GeneratedQuestion:
    """
    Direct ultra-fast question generation for Turbo Mode.
    Generates ONLY the question text directly from the concept chunk.
    """
    cleaned_content = clean_chunk(concept.content)
    quality = validate_chunk(cleaned_content)

    if not quality.is_valid:
        return GeneratedQuestion(
            concept_id=concept.concept_id,
            ideal_answer="[SKIPPED: Non-academic code or noise fragment]",
            question_text=f"[INVALID CONCEPT SKIPPED: {quality.reason}]",
            marks=0,
            bloom_level=0,
        )

    chunk = cleaned_content[:1200]
    prompt = TURBO_QUESTION_PROMPT.format(chunk=chunk, marks=marks)

    try:
        raw_response = get_llm().generate(
            prompt=prompt,
            options={
                "num_predict": 120,          # Hard limit: ~60 words max
                "temperature": 0.6,
                "stop": TURBO_STOP_SEQUENCES # Critical stop sequences
            }
        )
    except Exception as e:
        raw_response = ""

    clean_q = clean_turbo_output(raw_response)

    # Fallback if generation failed
    if not clean_q:
        snippet = chunk[:80]
        if "defined as" in chunk.lower() or "is a" in chunk.lower():
            clean_q = f"Define and explain the concept of '{snippet}'. What are its key characteristics and applications?"
        else:
            clean_q = f"Explain in detail: '{snippet}'. Discuss its significance in the context of the subject."

    return GeneratedQuestion(
        concept_id=concept.concept_id,
        ideal_answer=None,  # Bypassed in Turbo Mode
        question_text=clean_q,
        marks=marks,
        bloom_level=concept.bloom_dna or 2,
    )


def review_turbo(q: GeneratedQuestion) -> Tuple[bool, str]:
    """
    Instant (~0ms) regex validator for Turbo Mode questions.
    """
    if q.marks == 0 or q.question_text.startswith("[INVALID"):
        return False, "RC-00: Invalid concept chunk"

    if len(q.question_text) < 15:
        return False, "RC-06: Question too short"

    # Fast VTU verb check
    if re.match(r"^(explain|compare|derive|analyse|illustrate|describe|define|discuss)", q.question_text, re.I):
        return True, "ACCEPTED (TURBO)"
    
    # If question starts with allowed phrase or ends with ?, also accept
    if "?" in q.question_text and len(q.question_text.split()) >= 5:
        return True, "ACCEPTED (TURBO)"

    return False, "RC-04: Failed VTU verb check"