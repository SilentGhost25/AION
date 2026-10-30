"""
Prompt templates for the Writing Agent.

Pure data + string formatting. No LLM calls, no state.

Design notes:
    - Bloom verb lists per level. The prompt instructs the model to
      start with one of these. Post-generation enforcement is the
      Refinement Agent's job.
    - The visual directive is a separate section that only appears
      when `plan.visual_required=True`.
    - Evidence and KG neighborhood are truncated before insertion so
      the total prompt stays bounded.
"""

# UNTESTED — run quickstart.py on server first
from __future__ import annotations

from typing import Any, List, Optional


# -----------------------------------------------------------------------------
# Bloom verb lists
# -----------------------------------------------------------------------------


BLOOM_VERBS = {
    "L1": ["define", "list", "state", "identify", "recall", "name"],
    "L2": ["explain", "describe", "summarize", "classify",
           "interpret", "illustrate", "outline"],
    "L3": ["apply", "calculate", "solve", "compute", "demonstrate",
           "implement", "show"],
    "L4": ["analyze", "analyse", "compare", "contrast", "differentiate",
           "examine", "investigate"],
    "L5": ["evaluate", "assess", "justify", "critique", "judge", "defend"],
    "L6": ["design", "create", "formulate", "develop", "construct",
           "propose", "synthesize"],
}


# -----------------------------------------------------------------------------
# Truncation caps (chars)
# -----------------------------------------------------------------------------


MAX_EVIDENCE_BLOCK_CHARS = 500
MAX_TOTAL_EVIDENCE_CHARS = 2000
MAX_NEIGHBORHOOD_TRIPLES = 5
MAX_TOPIC_CHARS = 120


# -----------------------------------------------------------------------------
# Output schema (embedded in prompt as a guide to the model)
# -----------------------------------------------------------------------------


OUTPUT_SCHEMA_DESCRIPTION = """{
  "question_text": "string — the full question text",
  "solution": "string — a complete, self-contained model answer",
  "marking_scheme": [
    {"criterion": "string — what earns marks", "marks": integer}
  ],
  "diagram_request": {
    "diagram_type": "string — e.g. schematic, graph, block_diagram",
    "description": "string — short description of the required figure"
  },
  "references_image": boolean,
  "bloom_verb_used": "string — the first verb of the question_text"
}

CRITICAL: marking_scheme rules
  - Provide 2 to 4 entries.
  - Each entry has {"criterion": "...", "marks": N}.
  - The FINAL entry's marks MUST equal TOTAL_MARKS - sum(all previous entries).
  - The sum of all entries MUST equal TOTAL_MARKS exactly.
  - Do NOT invent a different total. TOTAL_MARKS is given in the slot contract.
  - Double-check your arithmetic before returning."""


# -----------------------------------------------------------------------------
# Section builders
# -----------------------------------------------------------------------------


def build_slot_contract(plan) -> str:
    marks_display = " + ".join(str(m) for m in plan.partition) if plan.partition else str(plan.total_marks)
    return (
        "[SLOT CONTRACT]\n"
        f"Marks: {plan.total_marks} ({marks_display})\n"
        f"Bloom level: {plan.bloom}\n"
        f"Course Outcome: {plan.co}\n"
        f"Module: {plan.module_id} (Module {plan.module_idx})\n"
        f"Question number: Q{plan.global_q_idx}"
    )


def build_topic_section(plan) -> str:
    topic = (plan.topic or "").strip()[:MAX_TOPIC_CHARS] or "(unspecified)"
    return f"[TOPIC]\n{topic}"


def build_kg_section(neighborhood: List[Any]) -> str:
    """
    Format KG triples for the prompt. Accepts Triple-like objects
    (with .subject/.predicate/.object) or (s, p, o) tuples.
    """
    if not neighborhood:
        return ""
    lines = ["[KNOWLEDGE GRAPH CONTEXT]"]
    for t in neighborhood[:MAX_NEIGHBORHOOD_TRIPLES]:
        if hasattr(t, "subject"):
            lines.append(f"  {t.subject} --{t.predicate}--> {t.object}")
        elif isinstance(t, (tuple, list)) and len(t) >= 3:
            lines.append(f"  {t[0]} --{t[1]}--> {t[2]}")
    return "\n".join(lines)


def build_evidence_section(evidence_blocks: List[Any]) -> str:
    if not evidence_blocks:
        return ""
    lines = ["[EVIDENCE FROM SOURCE MATERIAL]"]
    total = 0
    for block in evidence_blocks:
        text = getattr(block, "text", "")
        if not text:
            continue
        text = text[:MAX_EVIDENCE_BLOCK_CHARS]
        if total + len(text) > MAX_TOTAL_EVIDENCE_CHARS:
            break
        lines.append(f"  • {text}")
        total += len(text)
    return "\n".join(lines)


def build_visual_section(plan) -> str:
    if not plan.visual_required or plan.figure is None:
        return ""
    fig = plan.figure
    caption = getattr(fig, "caption", "") or "(no caption)"
    page = getattr(fig, "page", "?")
    return (
        "[VISUAL DIRECTIVE — FIGURE PROVIDED]\n"
        f"An authoritative figure is provided with this question.\n"
        f"Caption: {caption}\n"
        f"Page: {page}\n"
        "Requirements:\n"
        "  1. The question text MUST explicitly ask the student to refer to,\n"
        "     interpret, or sketch the given figure.\n"
        "  2. Acceptable openings include: 'With the aid of the given diagram...',\n"
        "     'Refer to the given figure and...', 'Using the provided schematic...'\n"
        "  3. The question must be answerable ONLY with the figure present.\n"
        "  4. Include `diagram_request` and `references_image: true` in the output."
    )


def build_difficulty_feedback(estimate: Any, target_band: str) -> str:
    """
    Construct prompt guidance when IRT difficulty falls outside target band.
    """
    diff = getattr(estimate, "difficulty", 0.0)
    if diff < -0.5 and target_band in ("medium", "hard"):
        return (
            "The previous question was measured as TOO EASY for the target difficulty level. "
            "Increase analytical depth: require multi-step reasoning, use higher-order cognitive operations, "
            "and ask for detailed derivation, comparative evaluation, or system design rather than simple recall."
        )
    if diff > 0.8 and target_band in ("easy", "medium"):
        return (
            "The previous question was measured as TOO HARD for the target difficulty level. "
            "Reduce to direct, single-step application. State necessary formulas or assumptions "
            "explicitly, and ask for a well-defined outcome rather than an open-ended derivation."
        )
    return ""


def build_task_section(plan, difficulty_feedback: Optional[str] = None) -> str:
    verbs = BLOOM_VERBS.get(plan.bloom, BLOOM_VERBS["L2"])
    verb_list = ", ".join(verbs)
    minutes = max(2, int(plan.total_marks * 1.5))

    lines = [
        "[TASK]",
        f"Generate ONE {plan.bloom}-level question worth {plan.total_marks} marks.",
        f"Start with an approved {plan.bloom} verb: {verb_list}.",
        f"Target completion time: ~{minutes} minutes.",
        "Provide a complete solution and a marking scheme.",
    ]

    if plan.is_multi_part:
        parts = ", ".join(f"part {i+1}: {m} marks" for i, m in enumerate(plan.partition))
        lines.append(f"Structure the question as sub-parts with these marks: {parts}.")
        lines.append(
            f"Write each part as a COMPLETE self-contained question with its own context. "
            f"Do NOT write a shared preamble. Do NOT restate the problem statement. "
            f"Mark each part with (i)/(ii) or (a)/(b)."
        )

    if difficulty_feedback:
        lines.append("")
        lines.append("[DIFFICULTY ADJUSTMENT]")
        lines.append(difficulty_feedback)

    return "\n".join(lines)


def build_constraints_section(total_marks: int = 10) -> str:
    return (
        "[CONSTRAINTS]\n"
        "  - The question must be self-contained; no references to 'the notes',\n"
        "    'the textbook', 'chapter N', or meta-text.\n"
        "  - FORBIDDEN PHRASES: Do NOT include preamble phrases such as 'Based on the provided evidence',\n"
        "    'Based on the evidence provided', 'Apply the following', or 'Refer to the text'.\n"
        "  - Open directly with the target Bloom action verb or scenario context.\n"
        "  - Keep each sub-question under 40 words. State the scenario in ONE sentence, then pose the task.\n"
        "  - If this question has a multi-part partition, emit each part with an explicit marker: (i)/(ii) or (a)/(b). Never merge them into one instruction.\n"
        "  - Do not reveal the answer in the question text.\n"
        "  - Do not use the words 'and' or 'or' to combine unrelated tasks.\n"
        "  - If the topic is mathematical or numerical, provide concrete givens.\n"
        f"  - The marking_scheme MUST sum to exactly {total_marks}.\n"
        f"    Make the final entry absorb any remainder so the sum equals {total_marks}.\n"
        "  - Return ONLY valid JSON. No commentary, no code fences, no preamble."
    )


# -----------------------------------------------------------------------------
# Full prompt
# -----------------------------------------------------------------------------


def build_prompt(
    plan,
    *,
    strict_json: bool = False,
    difficulty_feedback: Optional[str] = None,
) -> str:
    """
    Compose the full Writing Agent prompt.

    strict_json: if True, adds a shorter, more directive framing used on
    retry after a JSON parse failure.
    difficulty_feedback: guidance from IRT calibrator if previous attempt
    missed the difficulty target band.
    """
    sections = [
        build_slot_contract(plan),
        build_topic_section(plan),
    ]

    kg = build_kg_section(plan.concept_neighborhood)
    if kg:
        sections.append(kg)

    ev = build_evidence_section(plan.evidence_blocks)
    if ev:
        sections.append(ev)

    vis = build_visual_section(plan)
    if vis:
        sections.append(vis)

    sections.append(build_task_section(plan, difficulty_feedback=difficulty_feedback))
    sections.append(build_constraints_section(total_marks=plan.total_marks))
    sections.append(f"[OUTPUT JSON SCHEMA]\n{OUTPUT_SCHEMA_DESCRIPTION}")

    if strict_json:
        sections.append(
            "[IMPORTANT]\n"
            "Your previous response could not be parsed as JSON.\n"
            "Return ONLY the JSON object. Do not include any other text."
        )

    return "\n\n".join(s for s in sections if s)
