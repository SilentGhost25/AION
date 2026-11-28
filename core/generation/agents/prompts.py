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

from typing import Any, Dict, List, Optional

from core.validation.bloom_validator import BLOOM_VERB_LEVEL_MAP


# -----------------------------------------------------------------------------
# Canonical Bloom verb lists (derived directly from core.validation.bloom_validator)
# -----------------------------------------------------------------------------


BLOOM_VERBS: Dict[str, List[str]] = {
    lvl: sorted(list(verbs))
    for lvl, verbs in sorted(BLOOM_VERB_LEVEL_MAP.items())
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


def get_active_bloom_verbs(level: str, rule_snapshot: Any = None) -> List[str]:
    """
    Resolve active Bloom verbs dynamically from rule_snapshot or canonical BLOOM_VERB_LEVEL_MAP.
    Fails closed if level is unknown or empty (never falls back to stale defaults).
    """
    lvl = (level or "").strip().upper()
    if rule_snapshot is not None and hasattr(rule_snapshot, "get_verbs"):
        return list(rule_snapshot.get_verbs(lvl))
    if lvl not in BLOOM_VERB_LEVEL_MAP or not BLOOM_VERB_LEVEL_MAP[lvl]:
        raise ValueError(f"Unknown bloom level '{level}' in BLOOM_VERB_LEVEL_MAP")
    return sorted(str(v).strip().lower() for v in BLOOM_VERB_LEVEL_MAP[lvl] if str(v).strip())


def build_subpart_json_schema(sp_plan, rule_snapshot: Any = None) -> Dict[str, Any]:
    """
    Build a strict, dynamic JSON Schema for SubPart generation compatible with
    vLLM guided_json / json_schema and Ollama format.
    """
    bloom = getattr(sp_plan, "bloom", "L2")
    marks = int(getattr(sp_plan, "marks", 5))
    visual_req = bool(getattr(sp_plan, "visual_required", False))
    active_verbs = get_active_bloom_verbs(bloom, rule_snapshot=rule_snapshot)

    properties: Dict[str, Any] = {
        "text": {"type": "string", "minLength": 10},
        "solution": {"type": "string", "minLength": 10},
        "marking_scheme": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["criterion", "marks"],
                "additionalProperties": False,
                "properties": {
                    "criterion": {"type": "string", "minLength": 2},
                    "marks": {"type": "integer", "minimum": 1, "maximum": max(1, marks)},
                },
            },
        },
        "bloom_verb_used": {"type": "string", "enum": active_verbs},
        "references_image": {"type": "boolean"},
        "diagram_request": {
            "type": "object",
            "required": ["diagram_type", "description"],
            "additionalProperties": False,
            "properties": {
                "diagram_type": {"type": "string", "minLength": 2},
                "description": {"type": "string", "minLength": 5},
            },
        },
    }

    required = ["text", "solution", "marking_scheme", "bloom_verb_used"]
    if visual_req:
        properties["references_image"] = {"type": "boolean", "enum": [True]}
        required.extend(["references_image", "diagram_request"])

    return {
        "type": "object",
        "required": required,
        "additionalProperties": False,
        "properties": properties,
    }


def _sanitize_evidence_source_metadata(text: str) -> str:
    """
    Remove non-semantic source-label metadata (e.g. 'Figure 5.1:', 'Chapter 3:',
    '[blk_m1_1]') from model-facing evidence while preserving equations, numbers,
    definitions, conditions, and technical terminology.
    """
    import re
    cleaned = text
    # Strip bracketed source/page metadata tags like [Page 4], [Source: ...], [blk_1]
    cleaned = re.sub(r"\[(?:Page|Source|Module|Block|blk_)[^\]]*\]", "", cleaned, flags=re.IGNORECASE)
    # Strip leading numbered figure/chapter/section labels at start of sentence/line
    cleaned = re.sub(r"(?:^|\n)\s*(?:Fig(?:ure)?|Table|Chapter|Section|Module|Unit)\s+\d+(?:\.\d+)*\s*[:\-]\s*", " ", cleaned, flags=re.IGNORECASE)
    return re.sub(r"[ \t]+", " ", cleaned).strip()


def compute_token_aware_evidence_budget(
    *,
    context_tokens: int = 8192,
    system_tokens: int = 300,
    task_tokens: int = 350,
    schema_tokens: int = 350,
    output_reserve_tokens: int = 1024,
    safety_margin_tokens: int = 256,
    image_tokens: int = 0,
    chars_per_token: float = 3.5,
) -> int:
    """
    Compute available evidence character budget from token-level context formula:
    B_evidence = B_context - B_system - B_task - B_schema - B_output_reserve - B_safety_margin - B_image
    """
    avail_tokens = max(
        256,
        int(context_tokens)
        - int(system_tokens)
        - int(task_tokens)
        - int(schema_tokens)
        - int(output_reserve_tokens)
        - int(safety_margin_tokens)
        - int(image_tokens),
    )
    return int(avail_tokens * chars_per_token)


def build_evidence_section(
    evidence_blocks: List[Any],
    *,
    topic: Optional[str] = None,
    char_budget: Optional[int] = None,
    context_tokens: Optional[int] = None,
    image_tokens: int = 0,
) -> str:
    """
    Query-focused evidence selection within the model context budget.
    Preserves equations, numeric values, definitions, and conditions while
    stripping non-semantic source metadata tags.
    """
    import re
    if not evidence_blocks:
        return ""

    if char_budget is None:
        import os
        env_budget = os.getenv("AION_EVIDENCE_CHAR_BUDGET")
        if env_budget:
            char_budget = int(env_budget)
        elif context_tokens is not None:
            char_budget = compute_token_aware_evidence_budget(
                context_tokens=context_tokens,
                image_tokens=image_tokens,
            )
        else:
            char_budget = MAX_TOTAL_EVIDENCE_CHARS

    topic_tokens = {t for t in re.findall(r"[a-z0-9]+", (topic or "").lower()) if len(t) > 2}

    scored_blocks = []
    for idx, block in enumerate(evidence_blocks):
        raw_text = getattr(block, "text", "") if not isinstance(block, dict) else block.get("text", "")
        if not raw_text or not str(raw_text).strip():
            continue
        sanitized = _sanitize_evidence_source_metadata(str(raw_text))
        if not sanitized:
            continue
        if topic_tokens:
            blk_tokens = set(re.findall(r"[a-z0-9]+", sanitized.lower()))
            overlap = len(topic_tokens & blk_tokens)
        else:
            overlap = 0
        # Prioritize high-value technical content: equations, numeric givens, definitions
        high_val_bonus = 0
        if re.search(r"[=<>±∑∫]|(?:\b\d+(?:\.\d+)?\s*(?:ms|s|hz|khz|mhz|gb|mb|kb|v|a|w|ohm|Ω|%)\b)", sanitized, re.IGNORECASE):
            high_val_bonus += 3
        if re.search(r"\b(?:defined as|is a|refers to|theorem|lemma|equation|formula|condition|where)\b", sanitized, re.IGNORECASE):
            high_val_bonus += 2
        # Preserve original relative order on ties
        scored_blocks.append((-(overlap + high_val_bonus), idx, sanitized))

    if topic_tokens or any(b[0] < 0 for b in scored_blocks):
        scored_blocks.sort(key=lambda x: (x[0], x[1]))

    lines = ["[EVIDENCE FROM SOURCE MATERIAL]"]
    total = 0
    for _, _, text in scored_blocks:
        # Preserve full relevant blocks up to remaining budget (not arbitrary tiny truncation)
        remaining = char_budget - total
        if remaining <= 0:
            break
        chunk = text if len(text) <= remaining else text[:remaining].rsplit(" ", 1)[0]
        if not chunk:
            continue
        lines.append(f"  • {chunk}")
        total += len(chunk)
    return "\n".join(lines) if len(lines) > 1 else ""


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
        "  - FORBIDDEN PHRASES & SOURCE TAGS: Do NOT include preamble phrases such as 'Based on the provided evidence',\n"
        "    'Based on the evidence provided', 'Apply the following', or 'Refer to the text'.\n"
        "    NEVER include textbook/source tags such as 'Figure 1.2', 'Fig. 5.1', 'Table 2.1', 'Chapter 3', 'Section 2.4', 'Unit 1', or 'Module 2'.\n"
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


def build_critical_rules_header(sp_plan, rule_snapshot: Any = None) -> str:
    """Top-of-prompt critical rules banner (Rule-First & Rule-Last sandwich)."""
    verbs = get_active_bloom_verbs(sp_plan.bloom, rule_snapshot=rule_snapshot)
    verb_list = ", ".join(verbs)
    vis_rule = "MUST explicitly reference the provided diagram; " if getattr(sp_plan, "visual_required", False) else ""
    return (
        "[CRITICAL RULES — MUST FOLLOW]\n"
        f"  1. Start with ONE approved {sp_plan.bloom} verb: {verb_list}.\n"
        "  2. SINGULAR DIRECTIVE ONLY: Ask ONE task on ONE single topic; never combine two directives.\n"
        f"  3. {vis_rule}NO textbook tags ('Figure 5.1', 'Chapter 3', 'Section 2.4', 'Module 1').\n"
        f"  4. marking_scheme integer marks MUST sum to EXACTLY {sp_plan.marks}."
    )


def build_dynamic_exemplar_section(sp_plan, rule_snapshot: Any = None) -> str:
    """
    Generate compliant and forbidden examples dynamically from the active rule snapshot,
    exact marks, and visual requirement (never hardcoded stale examples).
    """
    verbs = get_active_bloom_verbs(sp_plan.bloom, rule_snapshot=rule_snapshot)
    v1 = verbs[0]
    marks = int(getattr(sp_plan, "marks", 5))
    m1 = max(1, marks // 2)
    m2 = marks - m1
    scheme_str = f'[{{"criterion": "Core concept", "marks": {m1}}}, {{"criterion": "Technical detail", "marks": {m2}}}]' if m2 > 0 else f'[{{"criterion": "Complete answer", "marks": {marks}}}]'
    topic = (getattr(sp_plan, "topic", "") or "the assigned topic").strip()

    if getattr(sp_plan, "visual_required", False):
        good_text = f"With reference to the given diagram, {v1} the primary mechanism of {topic}."
        bad_text = f"Using Figure 2.1, {v1} {topic} and also list its advantages."
    else:
        good_text = f"{v1.capitalize()} the core mechanism of {topic} given concrete system parameters."
        bad_text = f"Define {topic} and {v1} an unrelated topic from Chapter 3."

    return (
        "[ACTIVE RULE EXEMPLAR]\n"
        f"  COMPLIANT: {{\"text\": \"{good_text}\", \"bloom_verb_used\": \"{v1}\", \"marking_scheme\": {scheme_str}}}\n"
        f"  FORBIDDEN (DO NOT COPY): \"{bad_text}\" (multi-directive / leaked source tag)"
    )


def build_subpart_task_section(
    sp_plan,
    difficulty_feedback: Optional[str] = None,
    rule_snapshot: Any = None,
) -> str:
    verbs = get_active_bloom_verbs(sp_plan.bloom, rule_snapshot=rule_snapshot)
    verb_list = ", ".join(verbs)
    minutes = max(2, int(sp_plan.marks * 1.5))

    lines = [
        "[TASK]",
        f"Generate ONE {sp_plan.bloom}-level sub-question ({sp_plan.label}) worth {sp_plan.marks} marks aligned with {sp_plan.co}.",
        f"Start with ONE approved {sp_plan.bloom} verb: {verb_list}.",
        "SINGULAR DIRECTIVE: Use ONE and only ONE directive verb (e.g. only define, OR only describe, OR only explain, OR only implement). NEVER combine multiple directives (e.g. 'define X and explain Y').",
        "SINGLE TOPIC: Ask strictly about the ONE assigned topic below; do NOT bundle multiple unrelated topics.",
        f"Target completion time: ~{minutes} minutes.",
        f"Provide a complete solution and a marking scheme totaling exactly {sp_plan.marks} marks.",
        "INDEPENDENCE: This sub-question must be completely self-contained and independently answerable. Do not reference other sub-parts.",
    ]

    lvl_upper = (sp_plan.bloom or "").strip().upper()
    if lvl_upper == "L4":
        lines.append("COGNITIVE DEPTH (L4 - Analyze): Examine relationships, causes, components, or system/mathematical behavior (e.g., diagnose why behavior differs under two conditions), not mere recall.")
    elif lvl_upper == "L5":
        lines.append("COGNITIVE DEPTH (L5 - Evaluate): Apply explicit criteria, compare alternatives, and justify a judgment.")
    elif lvl_upper == "L6":
        lines.append("COGNITIVE DEPTH (L6 - Create): Design, construct, or develop a solution satisfying explicit constraints.")

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
        "  - FORBIDDEN PHRASES & SOURCE TAGS: Do NOT include preamble phrases such as 'Based on the provided evidence',\n"
        "    'Based on the evidence provided', 'Apply the following', or 'Refer to the text'.\n"
        "    NEVER include textbook/source tags such as 'Figure 1.2', 'Fig. 5.1', 'Table 2.1', 'Chapter 3', 'Section 2.4', 'Unit 1', or 'Module 2'.\n"
        "  - Open directly with the target Bloom action verb or scenario context.\n"
        "  - SINGULAR DIRECTIVE & SINGLE TOPIC: Ask ONLY ONE task using ONE Bloom action verb on ONE single topic. Do NOT combine multiple directives (such as 'define X, explain Y, and implement Z') or stack multiple questions.\n"
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


def build_subpart_contract_guidance_section(sp_plan) -> str:
    parts = []
    cf = getattr(sp_plan, "concept_footprint", None)
    if cf:
        parts.append(
            f"Target Concept: {cf.canonical_concept}\n"
            f"Expected Operation: {cf.operation}\n"
            f"Response Type: {cf.response_type}"
        )
    er = getattr(sp_plan, "expected_response", None)
    if er:
        reqs = ", ".join(er.required_elements)
        ops = ", ".join(er.expected_operations)
        parts.append(
            f"Solution Required Elements: {reqs}\n"
            f"Solution Expected Operations: {ops}"
        )
    vt = getattr(sp_plan, "visual_task", None)
    if vt:
        feats = ", ".join(vt.required_features)
        parts.append(
            f"Visual Operation: {vt.task_operation}\n"
            f"Required Visual Features: {feats}"
        )
    if parts:
        return "[ACADEMIC CONTRACT GUIDANCE]\n" + "\n".join(parts)
    return ""


def build_subpart_prompt(
    plan,
    sp_plan,
    *,
    strict_json: bool = False,
    difficulty_feedback: Optional[str] = None,
    validation_feedback: Optional[str] = None,
    rule_snapshot: Any = None,
) -> str:
    """
    Compose the Writing Agent prompt for an individual SubPartPlan.
    Uses Rule-First & Rule-Last sandwich and query-focused evidence selection.
    """
    sections = [
        build_subpart_slot_contract(plan, sp_plan),
        build_critical_rules_header(sp_plan, rule_snapshot=rule_snapshot),
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

    ev = build_evidence_section(merged_blocks, topic=getattr(sp_plan, "topic", None))
    if ev:
        sections.append(ev)

    vis = build_subpart_visual_section(sp_plan)
    if vis:
        sections.append(vis)

    contract_guidance = build_subpart_contract_guidance_section(sp_plan)
    if contract_guidance:
        sections.append(contract_guidance)

    sections.append(build_subpart_task_section(sp_plan, difficulty_feedback=difficulty_feedback, rule_snapshot=rule_snapshot))
    sections.append(build_dynamic_exemplar_section(sp_plan, rule_snapshot=rule_snapshot))
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


def build_archetype_subpart_prompt(
    plan,
    sp_plan,
    spec: Any,
    *,
    strict_json: bool = False,
    difficulty_feedback: Optional[str] = None,
    validation_feedback: Optional[str] = None,
    rule_snapshot: Any = None,
) -> str:
    """
    Compose a task-driven Writing Agent prompt for an individual SubPartPlan
    governed by a SubPartSpecification contract.
    """
    sections = [
        build_subpart_slot_contract(plan, sp_plan),
        build_critical_rules_header(sp_plan, rule_snapshot=rule_snapshot),
    ]

    # Archetype & Task Specification section
    archetype_val = getattr(spec.primary_archetype, "value", str(spec.primary_archetype))
    sec_archetypes = [getattr(sa, "value", str(sa)) for sa in (getattr(spec, "secondary_archetypes", ()) or ())]
    capabilities = [getattr(rc, "value", str(rc)) for rc in (getattr(spec, "required_capabilities", ()) or ())]

    task_spec_lines = [
        "[ASSESSMENT TASK CONTRACT]",
        f"Primary Archetype: {archetype_val}",
    ]
    if sec_archetypes:
        task_spec_lines.append(f"Secondary Archetypes: {', '.join(sec_archetypes)}")
    if capabilities:
        task_spec_lines.append(f"Required Capabilities: {', '.join(capabilities)}")
    sections.append("\n".join(task_spec_lines))

    sections.append(build_subpart_topic_section(sp_plan))

    # Input Problem Data Contract
    input_artifacts = getattr(spec, "input_artifacts", ()) or ()
    if input_artifacts:
        input_lines = ["[INPUT PROBLEM DATA CONTRACT]"]
        for art in input_artifacts:
            input_lines.append(f"- Artifact ID: {art.artifact_id} ({art.artifact_type}, modality: {getattr(art.modality, 'value', art.modality)})")
            if getattr(art, "parameters", None):
                for p in art.parameters:
                    p_state = getattr(p.state, "value", str(p.state))
                    val_str = f" = {p.value!r}" if p.value is not None else ""
                    unit_str = f" {p.unit}" if p.unit else ""
                    input_lines.append(f"  * Parameter: {p.name} [{p_state}]{val_str}{unit_str}")
                    if p_state == "GIVEN":
                        input_lines.append(f"    MANDATORY: You MUST explicitly include this given value in the question text or table.")
                    elif p_state == "STUDENT_SELECTED":
                        input_lines.append(f"    CONSTRAINTS: Student must select under: {p.selection_constraints}")
        sections.append("\n".join(input_lines))

    # Expected Student Response Contract
    exp_resp = getattr(spec, "expected_response", None)
    if exp_resp:
        resp_lines = [
            "[EXPECTED STUDENT RESPONSE CONTRACT]",
            f"Response Type: {exp_resp.response_type}",
            f"Estimated Time: ~{exp_resp.estimated_time_minutes} minutes",
        ]
        if exp_resp.expected_milestones:
            resp_lines.append("Expected Solution Milestones:")
            for m in exp_resp.expected_milestones:
                resp_lines.append(f"  * {m}")
        if exp_resp.requested_artifacts:
            resp_lines.append(f"Requested Student Artifacts to Draw/Synthesize: {', '.join(exp_resp.requested_artifacts)}")
        sections.append("\n".join(resp_lines))

    # Marking Rubric Contract
    rubric = getattr(spec, "marking_rubric", None)
    if rubric and getattr(rubric, "criteria", None):
        rubric_lines = ["[MARKING RUBRIC ALLOCATION]"]
        for c in rubric.criteria:
            rubric_lines.append(f"  * {c.criterion}: {c.marks} marks")
        rubric_lines.append(f"Total Subpart Marks: {rubric.total_marks}")
        sections.append("\n".join(rubric_lines))

    # Knowledge graph & evidence
    kg_neighborhood = getattr(sp_plan, "concept_neighborhood", None) or getattr(plan, "concept_neighborhood", None)
    kg = build_kg_section(kg_neighborhood)
    if kg:
        sections.append(kg)

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

    ev = build_evidence_section(merged_blocks, topic=getattr(sp_plan, "topic", None))
    if ev:
        sections.append(ev)

    vis = build_subpart_visual_section(sp_plan)
    if vis:
        sections.append(vis)

    contract_guidance = build_subpart_contract_guidance_section(sp_plan)
    if contract_guidance:
        sections.append(contract_guidance)

    sections.append(build_subpart_task_section(sp_plan, difficulty_feedback=difficulty_feedback, rule_snapshot=rule_snapshot))
    sections.append(build_dynamic_exemplar_section(sp_plan, rule_snapshot=rule_snapshot))
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

