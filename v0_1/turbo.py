def _visual_constraint(equations=None, images=None):
    eqs = equations or []
    imgs = images or []
    if not eqs and not imgs:
        return ''
    eqb = '\n'.join(f'  eq_{i+1}: {e}' for i,e in enumerate(eqs[:8])) or '  (none)'
    imb = '\n'.join(f'  {p}' for p in imgs[:8]) or '  (none)'
    return (
        '\n=== SOURCE VISUALS & EQUATIONS (MANDATORY) ===\n'
        'Use ONLY these extracted equations/figures. Put latex in math_blocks as [MATH:eq_k].\n'
        'If an image path is given, set image_path and say Refer to the figure.\n'
        'Do NOT invent other subjects.\nEQUATIONS:\n' + eqb +
        '\nIMAGES:\n' + imb + '\n==============================================\n'
    )

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
    # Bind extracted figures into the LLM prompt (mandatory figure use)
    try:
        import aion_patch as _ap
        _figs = list(getattr(_ap, "_ACTIVE_DIAGRAMS", None) or [])
    except Exception:
        _figs = []
    _visual_add = ""
    if _figs:
        _paths = []
        for _d in _figs[:8]:
            if isinstance(_d, dict):
                _ip = _d.get("image_path") or _d.get("path") or ""
                _cap = _d.get("caption") or _d.get("label") or ""
            else:
                _ip = getattr(_d, "image_path", "") or ""
                _cap = getattr(_d, "caption", "") or ""
            if _ip:
                _paths.append(f"- {_cap}: {_ip}")
        if _paths:
            _visual_add = (
                "\n=== SOURCE FIGURES (MANDATORY) ===\n"
                "You MUST use at least one of these extracted figures.\n"
                "In question_text say \"With reference to the given figure...\".\n"
                "In the JSON set BOTH image_path and associated_image to the exact file path.\n"
                + "\n".join(_paths) +
                "\nDo NOT invent other subjects or figures.\n"
                "===================================\n"
            )
    if "_visual_constraint" in globals() and _figs:
        try:
            _visual_add = _visual_constraint(_figs) + "\n" + _visual_add
        except Exception:
            pass

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
    prompt = _visual_add + TURBO_QUESTION_PROMPT.format(chunk=chunk, marks=marks)

    try:
        raw_response = get_llm().generate(
            prompt=prompt,
            options={
                "num_predict": 200,          # Hard limit: ~60 words max
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