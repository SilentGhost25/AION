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
# Figure reference phrases
# -----------------------------------------------------------------------------

IMAGE_REFERENCE_PHRASES = (
    "given figure", "given diagram", "given schematic",
    "provided figure", "provided diagram", "provided schematic",
    "shown in the figure", "shown in the diagram",
    "shown in figure", "shown in diagram",
    "refer to the figure", "refer to the diagram",
    "refer to figure", "refer to diagram",
    "with the aid of the given", "with the aid of the provided",
    "using the provided", "using the given",
    "figure below", "diagram below",
    "accompanying figure", "accompanying diagram",
    "illustrated in the figure", "illustrated in the diagram",
    "with reference to the figure", "with reference to the diagram",
)


def get_figure_field(figure: Any, key: str, default: Any = None) -> Any:
    """Safely retrieve a field from a figure represented as either an object or a dictionary."""
    if figure is None:
        return default
    if isinstance(figure, dict):
        return figure.get(key, default)
    return getattr(figure, key, default)


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


SUBPART_OUTPUT_SCHEMA_DESCRIPTION = """{
  "text": "string — the full, self-contained question text for this subpart",
  "solution": "string — a complete, self-contained model answer for this subpart",
  "marking_scheme": [
    {"criterion": "string — what earns marks", "marks": 0}
  ],
  "diagram_request": {
    "diagram_type": "string — e.g. schematic, graph, block_diagram",
    "description": "string — short description of the required figure"
  },
  "references_image": false,
  "bloom_verb_used": "string — the first verb of the text"
}

FIELD REQUIREMENTS:
  - MANDATORY keys: "text", "solution", "marking_scheme". These MUST be present in every response.
  - OPTIONAL/CONDITIONAL keys:
      * "diagram_request": include if a figure is provided or a diagram is requested.
      * "references_image": boolean, include (true) if referencing a provided figure.
      * "bloom_verb_used": string, include the first verb of the text.

CRITICAL: marking_scheme rules
  - Provide 1 to 4 entries.
  - Each entry has {"criterion": "...", "marks": N}.
  - The sum of all entries MUST equal the subpart's marks exactly.
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
    if getattr(plan, "is_multi_part", False) and getattr(plan, "topic_a", None) and getattr(plan, "topic_b", None):
        if plan.topic_a != plan.topic_b:
            part_a_marks = plan.partition[0] if plan.partition else 6
            part_b_marks = plan.partition[1] if len(plan.partition) > 1 else 4
            top_a = plan.topic_a.strip()[:MAX_TOPIC_CHARS]
            top_b = plan.topic_b.strip()[:MAX_TOPIC_CHARS]
            return (
                f"[TOPIC FOR (a) — {part_a_marks} MARKS]\n{top_a}\n\n"
                f"[TOPIC FOR (b) — {part_b_marks} MARKS]\n{top_b}\n\n"
                f"[CRITICAL SIBLING INDEPENDENCE RULE]\n"
                f"These are TWO COMPLETELY INDEPENDENT TOPICS from the same module. "
                f"Generate sub-question (a) strictly about '{top_a}' and sub-question (b) strictly about '{top_b}'. "
                f"They MUST be independently answerable. Sub-question (b) must NOT chain, extend, or refer to (a)."
            )
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
    caption = get_figure_field(fig, "caption", "") or "(no caption)"
    page = get_figure_field(fig, "page", "?")
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
    if plan.bloom not in BLOOM_VERBS:
        raise ValueError(f"Unknown bloom level '{plan.bloom}'. Must be one of {list(BLOOM_VERBS.keys())}")
    verbs = BLOOM_VERBS[plan.bloom]
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
            "[MULTI-PART SIBLING INDEPENDENCE — MANDATORY VTU CONTRACT]\n"
            "This question consists of TWO INDEPENDENT sub-parts:\n"
            f"  (a) {plan.partition[0]} marks\n"
            f"  (b) {plan.partition[1]} marks\n"
            "RULES:\n"
            "1. (a) and (b) MUST be independently answerable. A student can answer (b) without having answered (a).\n"
            "2. (a) and (b) MUST test DIFFERENT concepts from the module.\n"
            "3. (b) must NOT say 'with reference to (a)', 'extend the above', 'continuing from (a)', or similar chains.\n"
            "4. (b) must NOT ask to 'compare to your answer in (a)' or use pronouns (it, this, that) referring to (a).\n"
            "5. Think of them as two separate questions that happen to be grouped under one main question.\n"
            "GOOD EXAMPLE:\n"
            "  (a) [6 marks] Explain the challenges of deploying ML in production.\n"
            "  (b) [4 marks] Apply the Find-S algorithm to the given training data.\n"
            "BAD EXAMPLE (FORBIDDEN):\n"
            "  (a) [6 marks] Explain the challenges of deploying ML.\n"
            "  (b) [4 marks] With reference to (a), how do those challenges affect model accuracy?"
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


# -----------------------------------------------------------------------------
# SubPart prompt builders (Stage G1-C)
# -----------------------------------------------------------------------------


def build_subpart_slot_contract(plan, sp_plan) -> str:
    return (
        "[SUBPART CONTRACT]\n"
        f"Subpart: ({sp_plan.label})\n"
        f"Marks: {sp_plan.marks}\n"
        f"Bloom level: {sp_plan.bloom}\n"
        f"Course Outcome: {sp_plan.co}\n"
        f"Module: {plan.module_id} (Module {plan.module_idx})\n"
        f"Question number: Q{plan.global_q_idx}({sp_plan.label})"
    )


def build_subpart_topic_section(sp_plan) -> str:
    topic = (sp_plan.topic or "").strip()[:MAX_TOPIC_CHARS] or "(unspecified)"
    return f"[TOPIC]\n{topic}"


def build_subpart_visual_section(sp_plan) -> str:
    if not sp_plan.visual_required or sp_plan.figure is None:
        return ""
    fig = sp_plan.figure
    caption = get_figure_field(fig, "caption", "") or "(no caption)"
    page = get_figure_field(fig, "page", "?")
    return (
        "[VISUAL DIRECTIVE — FIGURE PROVIDED]\n"
        f"An authoritative figure is provided with this sub-question.\n"
        f"Caption: {caption}\n"
        f"Page: {page}\n"
        "Requirements:\n"
        "  1. The sub-question text MUST explicitly ask the student to refer to,\n"
        "     interpret, or sketch the given figure.\n"
        "  2. Acceptable openings include: 'With the aid of the given diagram...',\n"
        "     'Refer to the given figure and...', 'Using the provided schematic...'\n"
        "  3. The sub-question must be answerable ONLY with the figure present.\n"
        "  4. Include `diagram_request` and `references_image: true` in the output."
    )


def build_subpart_task_section(sp_plan, difficulty_feedback: Optional[str] = None) -> str:
    if sp_plan.bloom not in BLOOM_VERBS:
        raise ValueError(f"Unknown bloom level '{sp_plan.bloom}'. Must be one of {list(BLOOM_VERBS.keys())}")
    verbs = BLOOM_VERBS[sp_plan.bloom]
    verb_list = ", ".join(verbs)
    minutes = max(2, int(sp_plan.marks * 1.5))

    lines = [
        "[TASK]",
        f"Generate ONE {sp_plan.bloom}-level sub-question ({sp_plan.label}) worth {sp_plan.marks} marks.",
        f"Start with an approved {sp_plan.bloom} verb: {verb_list}.",
        f"Target completion time: ~{minutes} minutes.",
        f"Provide a complete solution and a marking scheme totaling exactly {sp_plan.marks} marks.",
        "INDEPENDENCE: This sub-question must be completely self-contained and independently answerable. Do not reference other sub-parts.",
    ]

    if difficulty_feedback:
        lines.append("")
        lines.append("[DIFFICULTY ADJUSTMENT]")
        lines.append(difficulty_feedback)

    return "\n".join(lines)


def build_subpart_constraints_section(marks: int) -> str:
    return (
        "[CONSTRAINTS]\n"
        "  - The sub-question must be self-contained; no references to 'the notes',\n"
        "    'the textbook', 'chapter N', or other subparts.\n"
        "  - FORBIDDEN PHRASES: Do NOT include preamble phrases such as 'Based on the provided evidence',\n"
        "    'Based on the evidence provided', 'Apply the following', or 'Refer to the text'.\n"
        "  - Open directly with the target Bloom action verb or scenario context.\n"
        "  - SIBLING INDEPENDENCE: Do NOT reference other sub-parts (e.g. 'with reference to part (a)', 'using your answer from part (a)') and do NOT start with referring pronouns ('It', 'This', 'These', 'They').\n"
        "  - NO FORMATTING ARTIFACTS: Do NOT include bullet markers (•, ▪, →, ◆, ■) or unparsed placeholders ({TOPIC}, [TOPIC]).\n"
        "  - COMPLETE CLAUSES: Do NOT end the question text mid-clause or with trailing punctuation like ',' or ';'.\n"
        "  - Keep under 60 words. State the scenario or context concisely, then pose the task.\n"
        "  - Do not reveal the answer in the question text.\n"
        "  - If the topic is mathematical or numerical, provide concrete givens.\n"
        f"  - The marking_scheme MUST sum to exactly {marks}.\n"
        "  - Return ONLY valid JSON. Mandatory keys: \"text\", \"solution\", \"marking_scheme\". "
        "Optional/conditional keys: \"diagram_request\", \"references_image\", \"bloom_verb_used\". "
        "No commentary, no code fences, no preamble."
    )


def build_subpart_prompt(
    plan,
    sp_plan,
    *,
    strict_json: bool = False,
    difficulty_feedback: Optional[str] = None,
    validation_feedback: Optional[str] = None,
) -> str:
    """
    Compose the Writing Agent prompt for an individual SubPartPlan.
    """
    sections = [
        build_subpart_slot_contract(plan, sp_plan),
        build_subpart_topic_section(sp_plan),
    ]

    kg_neighborhood = getattr(sp_plan, "concept_neighborhood", None) or getattr(plan, "concept_neighborhood", None)
    kg = build_kg_section(kg_neighborhood)
    if kg:
        sections.append(kg)

    # Merge subpart and parent evidence blocks, preserving provenance and deduplicating
    merged_blocks = []
    seen_keys = set()
    sp_blocks = getattr(sp_plan, "evidence_blocks", None) or []
    parent_blocks = getattr(plan, "evidence_blocks", None) or []
    for b in list(sp_blocks) + list(parent_blocks):
        b_id = getattr(b, "id", None) if not isinstance(b, dict) else b.get("id")
        b_text = getattr(b, "text", "") if not isinstance(b, dict) else b.get("text", "")
        dedup_key = b_id or (b_text[:80] if b_text else None)
        if dedup_key and dedup_key in seen_keys:
            continue
        if dedup_key:
            seen_keys.add(dedup_key)
        merged_blocks.append(b)

    ev = build_evidence_section(merged_blocks)
    if ev:
        sections.append(ev)

    vis = build_subpart_visual_section(sp_plan)
    if vis:
        sections.append(vis)

    sections.append(build_subpart_task_section(sp_plan, difficulty_feedback=difficulty_feedback))
    sections.append(build_subpart_constraints_section(marks=sp_plan.marks))
    sections.append(f"[OUTPUT JSON SCHEMA]\n{SUBPART_OUTPUT_SCHEMA_DESCRIPTION}")

    if validation_feedback and str(validation_feedback).strip():
        sections.append(
            "[CORRECTIVE GUIDANCE FOR RETRY]\n"
            f"Your previous attempt could not be accepted for the following reason:\n"
            f"{str(validation_feedback).strip()}\n"
            "Please correct this issue in your response while maintaining all other constraints."
        )

    if strict_json:
        sections.append(
            "[IMPORTANT]\n"
            "Your previous response could not be parsed as JSON.\n"
            "Return ONLY the JSON object. Do not include any other text."
        )

    return "\n\n".join(s for s in sections if s)

