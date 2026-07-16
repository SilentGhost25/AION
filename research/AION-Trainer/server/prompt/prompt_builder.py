# AION-Trainer/server/prompt/prompt_builder.py
"""
Prompt Builder — converts an AssessmentIntent into the best possible
input string for the model.

Design decisions:
    1. System persona is set first (examiner, not assistant).
    2. Academic context is structured with markdown headers so the
       model can parse it even at small sizes (T5-base reads sections).
    3. VTU rules are explicit, numbered, and SHORT — models follow
       numbered lists better than flowing prose instructions.
    4. Reference questions give the model concrete style targets.
    5. Word count is stated as an explicit hard constraint.
    6. The final line is a clear, unambiguous generation trigger.

Every method is independently testable — the class never touches the
model directly.
"""

from typing import Dict, List
from server.prompt.assessment_intent import AssessmentIntent

# Maps Bloom level to (cognitive description, suitable action verbs, example framing)
BLOOM_METADATA: Dict[str, Dict] = {
    "L1": {
        "description": "Remember — recall facts and basic concepts",
        "verbs": ["Define", "List", "State", "Name", "Identify", "Recall", "Label"],
        "framing": "Ask the student to recall or recognise a specific term, fact, or concept.",
    },
    "L2": {
        "description": "Understand — explain ideas or concepts",
        "verbs": ["Explain", "Describe", "Discuss", "Summarize", "Interpret", "Illustrate"],
        "framing": "Ask the student to explain the concept in their own words with context.",
    },
    "L3": {
        "description": "Apply — use information in new situations",
        "verbs": ["Apply", "Solve", "Demonstrate", "Illustrate", "Implement", "Calculate"],
        "framing": (
            "Ask the student to apply the concept to a concrete example or "
            "problem instance; ask for a worked trace or calculation."
        ),
    },
    "L4": {
        "description": "Analyze — draw connections among ideas",
        "verbs": ["Analyze", "Compare", "Contrast", "Differentiate", "Trace", "Distinguish"],
        "framing": (
            "Ask the student to break the concept into parts or compare two "
            "related concepts; ask for criteria or structured reasoning."
        ),
    },
    "L5": {
        "description": "Evaluate — justify a decision or course of action",
        "verbs": ["Evaluate", "Justify", "Assess", "Critique", "Judge", "Defend"],
        "framing": (
            "Ask the student to make a judgment about trade-offs, "
            "advantages/limitations, or suitability for a context."
        ),
    },
    "L6": {
        "description": "Create — produce a new or original work",
        "verbs": ["Design", "Develop", "Propose", "Construct", "Formulate", "Create"],
        "framing": (
            "Ask the student to design or develop something original — "
            "an algorithm, a system, a solution — for a given scenario."
        ),
    },
}

QUESTION_TYPE_GUIDANCE: Dict[str, str] = {
    "definition": (
        "Ask for a precise definition, optionally with a short example or properties."
    ),
    "explanation": (
        "Ask the student to explain the concept with context, "
        "working, and/or applications."
    ),
    "algorithm": (
        "Ask for the algorithm steps and a worked trace on a concrete input. "
        "Specify the input or ask the student to choose a suitable one."
    ),
    "comparison": (
        "Ask the student to compare two concepts — "
        "similarities, differences, and when to prefer each."
    ),
    "numerical": (
        "Provide or ask the student to assume concrete numerical values; "
        "ask for a step-by-step solution."
    ),
    "diagram_based": (
        "Ask the student to draw and label the diagram "
        "and explain each labeled component."
    ),
    "short_note": (
        "Ask for a concise short note covering definition, "
        "key properties, and at least one application."
    ),
    "case_study": (
        "Present or ask the student to describe a real-world scenario "
        "and apply the concept to it."
    ),
    "derivation": (
        "Ask the student to derive or prove the result "
        "from first principles, showing each step."
    ),
}

MARKS_GUIDANCE: Dict[int, str] = {
    2:  "Very short — one sentence definition or a two-point list.",
    5:  "Short — definition plus two or three key points or a brief example.",
    10: "Medium — full explanation with example, diagram if applicable, applications.",
    15: "Long — detailed explanation, worked algorithm trace, comparison, or derivation.",
    20: "Very long — comprehensive question covering multiple sub-parts of the topic.",
}


class PromptBuilder:
    """
    Builds optimised prompts for every generation scenario AION encounters.
    All methods are pure functions (no state, no side effects) so they
    are trivially unit-testable.
    """

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def build_question_generation_prompt(self, intent: AssessmentIntent) -> str:
        """
        Main entry point.
        Returns the complete prompt string to feed to the model.
        """
        sections = [
            self._persona_section(intent),
            self._academic_context_section(intent),
            self._assessment_requirements_section(intent),
            self._vtu_rules_section(intent),
            self._reference_questions_section(intent),
            self._previously_asked_section(intent),
            self._generation_trigger(),
        ]
        return "\n\n".join(s for s in sections if s.strip())

    def build_answer_outline_prompt(self, question: str, intent: AssessmentIntent) -> str:
        """
        Prompt that asks the model to produce an expected-answer outline
        for a given question. Used by QuestionGenerator for the answer
        outline field.
        """
        bloom_meta = BLOOM_METADATA.get(intent.bloom_level, BLOOM_METADATA["L2"])
        context = self._build_knowledge_block(intent)

        return (
            f"You are an experienced VTU examiner writing an answer key.\n\n"
            f"## QUESTION\n{question}\n\n"
            f"## SOURCE KNOWLEDGE\n{context}\n\n"
            f"## REQUIREMENTS\n"
            f"- Marks: {intent.marks}\n"
            f"- Bloom Level: {intent.bloom_level} ({bloom_meta['description']})\n"
            f"- Write bullet-point key points the answer must contain\n"
            f"- Include diagram labels if the question asks for a diagram\n"
            f"- Do not write full sentences — write concise expected points only\n\n"
            f"## EXPECTED ANSWER OUTLINE:"
        )

    def build_bloom_classification_prompt(self, question_text: str) -> str:
        """
        Prompt that asks the model to classify the Bloom level of a question.
        """
        return (
            "Classify the Bloom's Taxonomy level of the following exam question.\n"
            "Choose exactly ONE level from: L1, L2, L3, L4, L5, L6.\n"
            "Respond with only the level code (e.g., 'L3').\n\n"
            f"## QUESTION\n{question_text}\n\n"
            "## CLASSIFICATION:"
        )

    def build_diagram_requirement_prompt(self, topic: str, question: str) -> str:
        """
        Prompt that asks the model to decide if a question requires a diagram.
        """
        return (
            f"Topic: {topic}\n"
            f"Exam Question: {question}\n\n"
            "Does answering this question require drawing a diagram, flowchart, "
            "tree, or graph?\n"
            "Answer with exactly: 'Yes' or 'No'.\n\n"
            "## ANSWER:"
        )

    def build_reflection_prompt(
        self,
        concept_id: str,
        topic: str,
        context_text: str,
        target_marks: int,
    ) -> str:
        """
        Self-RAG reflection prompt — used by AcademicSelfRAG to decide
        if retrieved context is sufficient for a given question target.
        """
        return (
            f"A {target_marks}-mark VTU examination question must be set on: "
            f"{topic} (concept: {concept_id}).\n\n"
            f"## AVAILABLE CONTEXT\n{context_text[:1200]}\n\n"
            f"Is this context sufficient to write a complete, academically "
            f"correct answer for a {target_marks}-mark question on this topic?\n\n"
            "Consider:\n"
            "- Does the context include a clear definition?\n"
            "- Are key properties or algorithm steps present?\n"
            "- Is there at least one example or application?\n"
            "- For marks >= 10, is there enough depth for a full answer?\n\n"
            'Respond in JSON: {"sufficient": true/false, '
            '"missing_aspects": ["..."], "confidence": 0.0-1.0}'
        )

    def build_intent_planning_prompt(
        self,
        topic: str,
        concept_summary: str,
        related_topics: List[str],
        previously_asked: List[str],
        bloom_level: str,
        marks: int,
        verb_pool: List[str],
    ) -> str:
        """
        Examiner Reasoning Engine prompt — plans WHAT to ask before
        any question text is written.
        """
        bloom_meta = BLOOM_METADATA.get(bloom_level, BLOOM_METADATA["L2"])

        return (
            f"You are an experienced VTU university examiner planning an "
            f"examination question at Bloom level {bloom_level} "
            f"({bloom_meta['description']}) worth {marks} marks.\n\n"
            f"## TOPIC\n{topic}\n\n"
            f"## CONCEPT SUMMARY\n{concept_summary[:600]}\n\n"
            f"## RELATED TOPICS AVAILABLE FOR COMPARISON\n"
            f"{', '.join(related_topics) if related_topics else 'None'}\n\n"
            f"## RECENTLY ASKED (DO NOT REPEAT THESE)\n"
            f"{chr(10).join('- ' + q for q in previously_asked[-5:]) if previously_asked else 'None'}\n\n"
            f"## YOUR TASK\n"
            f"Choose the single most effective action verb from this list: "
            f"{', '.join(verb_pool)}\n"
            f"Decide whether the question should require a diagram.\n"
            f"Decide whether it should be a comparison with one of the related topics.\n\n"
            f"Respond in JSON:\n"
            f'{{"action_verb": "...", "rationale": "...", '
            f'"requires_diagram": true/false, "compare_with": "topic or null"}}'
        )

    # ------------------------------------------------------------------ #
    #  Private section builders                                           #
    # ------------------------------------------------------------------ #

    def _persona_section(self, intent: AssessmentIntent) -> str:
        subject_info = ""
        if intent.subject_name or intent.subject_code:
            subject_info = f" for {intent.subject_name or ''} ({intent.subject_code or ''})"
            if intent.semester:
                subject_info += f", Semester {intent.semester}"
            if intent.module:
                subject_info += f", Module {intent.module}"

        return (
            f"You are an experienced {intent.university} university examiner "
            f"setting a question paper{subject_info}.\n"
            f"Generate exactly ONE examination question. "
            f"Do not generate the answer."
        )

    def _academic_context_section(self, intent: AssessmentIntent) -> str:
        lines = ["## ACADEMIC CONTEXT"]
        lines.append(f"Topic: {intent.topic}")

        if intent.subtopic:
            lines.append(f"Subtopic: {intent.subtopic}")

        if intent.definition:
            lines.append(f"Definition: {intent.definition}")

        if intent.explanation:
            trunc = intent.explanation[:400] + ("..." if len(intent.explanation) > 400 else "")
            lines.append(f"Explanation: {trunc}")

        if intent.key_points:
            lines.append("Key Points:")
            for pt in intent.key_points[:6]:
                lines.append(f"  - {pt}")

        if intent.algorithms:
            lines.append(f"Algorithms: {', '.join(intent.algorithms[:4])}")

        if intent.applications:
            lines.append(f"Applications: {', '.join(intent.applications[:4])}")

        if intent.formulas:
            lines.append(f"Formula(s): {'; '.join(intent.formulas[:3])}")

        if intent.requires_diagram and intent.diagram_description:
            lines.append(f"Diagram: {intent.diagram_description}")

        if intent.compare_with:
            lines.append(f"Compare With: {intent.compare_with}")

        return "\n".join(lines)

    def _assessment_requirements_section(self, intent: AssessmentIntent) -> str:
        bloom_meta = BLOOM_METADATA.get(intent.bloom_level, BLOOM_METADATA["L2"])
        qtype_guidance = QUESTION_TYPE_GUIDANCE.get(intent.question_type, "")
        marks_guidance = self._get_marks_guidance(intent.marks)

        lines = [
            "## ASSESSMENT REQUIREMENTS",
            f"Bloom Level  : {intent.bloom_level} — {bloom_meta['description']}",
            f"Action Verb  : {intent.action_verb}",
            f"Marks        : {intent.marks}",
            f"Question Type: {intent.question_type.replace('_', ' ').title()}",
            f"Difficulty   : {intent.difficulty.title()}",
        ]

        if intent.co_tag:
            lines.append(f"Course Outcome: {intent.co_tag}")

        if intent.requires_diagram:
            lines.append("Diagram      : Required — ask student to draw and explain")

        lines.append(f"\nCognitive goal: {bloom_meta['framing']}")

        if qtype_guidance:
            lines.append(f"Question style: {qtype_guidance}")

        lines.append(f"Length target: {marks_guidance}")

        return "\n".join(lines)

    def _vtu_rules_section(self, intent: AssessmentIntent) -> str:
        rules = [
            f'1. Begin the question with the action verb: "{intent.action_verb}"',
            "2. Write in formal academic English — no informal phrasing",
            "3. Do not include the answer, hints, or answer structure in the question",
            "4. End the question with a period",
            f"5. Length: between {intent.min_words} and {intent.max_words} words",
        ]

        if intent.marks >= 10:
            rules.append(
                '6. For questions worth 10+ marks, include "with a suitable example" '
                "or ask for a diagram or worked trace"
            )

        if intent.requires_diagram:
            rules.append(
                "7. Explicitly ask the student to draw the diagram "
                "and label all important components"
            )

        if intent.compare_with:
            rules.append(
                f'8. The comparison must name both "{intent.topic}" '
                f'and "{intent.compare_with}" explicitly in the question'
            )

        if intent.question_type == "algorithm":
            rules.append(
                "9. For algorithm questions, ask for the algorithm steps "
                "AND a trace on a concrete example"
            )

        return "## VTU EXAMINER RULES\n" + "\n".join(rules)

    def _reference_questions_section(self, intent: AssessmentIntent) -> str:
        if not intent.reference_questions:
            return ""
        lines = ["## REFERENCE VTU QUESTIONS (match this style — do not copy)"]
        for q in intent.reference_questions[:4]:
            lines.append(f'- "{q}"')
        return "\n".join(lines)

    def _previously_asked_section(self, intent: AssessmentIntent) -> str:
        if not intent.previously_asked:
            return ""
        lines = ["## ALREADY ASKED (do not repeat, must be clearly different)"]
        for q in intent.previously_asked[-5:]:
            lines.append(f'- "{q}"')
        return "\n".join(lines)

    def _generation_trigger(self) -> str:
        return "## GENERATE ONE QUESTION:"

    def _build_knowledge_block(self, intent: AssessmentIntent) -> str:
        """Compact knowledge summary used in answer-outline and reflection prompts."""
        parts = []
        if intent.definition:
            parts.append(intent.definition)
        if intent.key_points:
            parts.append("Key points: " + "; ".join(intent.key_points[:5]))
        if intent.algorithms:
            parts.append("Algorithms: " + ", ".join(intent.algorithms[:3]))
        if intent.applications:
            parts.append("Applications: " + ", ".join(intent.applications[:3]))
        return "\n".join(parts)

    def _get_marks_guidance(self, marks: int) -> str:
        """Return the guidance string for the closest known marks value."""
        closest = min(MARKS_GUIDANCE.keys(), key=lambda k: abs(k - marks))
        return MARKS_GUIDANCE[closest]
