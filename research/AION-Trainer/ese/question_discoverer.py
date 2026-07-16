# AION-Trainer/ese/question_discoverer.py
"""
Question Discoverer — Step 4 of the ESE.

Given an AnswerBlueprint and AssessmentIntent, generates multiple
candidate question TEXTS (not blueprints — those already exist).

This is the first neural stage. The question is: given an answer,
what are all the reasonable exam questions whose ideal response IS
this answer?

Architecture:
    Template-based candidates (deterministic, fast)
    +
    Neural candidates (LLM-generated, when available)
    =
    Candidate pool (passed to QuestionRanker)

The neural path is optional — if no LLM is configured the system
falls back to the template candidates, which still produce good VTU
questions for standard bloom levels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from ese.answer_blueprint import AnswerBlueprint, AnswerComponent
from server.prompt.assessment_intent import AssessmentIntent
from server.prompt.prompt_builder import PromptBuilder

logger = logging.getLogger("aion.ese.discoverer")


@dataclass
class QuestionCandidate:
    """
    One candidate question discovered from an AnswerBlueprint.
    Carries initial confidence before ranking.
    """
    text: str
    source: str                     # "template" | "neural" | "pyq_adapted"
    initial_confidence: float = 0.7
    covers_components: List[str] = field(default_factory=list)
    bloom_alignment: float = 1.0
    structural_completeness: float = 1.0


DISCOVERY_TEMPLATES: Dict[str, Dict[str, str]] = {
    "definition": {
        "L1": "Define {topic}.",
        "L2": "Explain the concept of {topic} with a suitable example.",
    },
    "explanation": {
        "L2": "Explain {topic} with a neat diagram and a suitable example.",
        "L3": "Describe the working of {topic} with a suitable example, illustrating each step.",
        "L5": "Evaluate the advantages and limitations of {topic} in real-world scenarios.",
    },
    "algorithm": {
        "L2": "Explain the {topic} algorithm with a suitable example.",
        "L3": "Illustrate the {topic} algorithm with a worked example. Show all steps and the final result.",
        "L3_trace": "Trace the working of {topic} for the given input. Show all intermediate states.",
        "L4": "Analyze the time and space complexity of {topic} and compare with {compare_with}.",
    },
    "comparison": {
        "L4": "Compare and contrast {topic} and {compare_with} with respect to time complexity, space complexity, and applicability.",
        "L4_alt": "Differentiate between {topic} and {compare_with}. When would you prefer one over the other?",
    },
    "diagram_based": {
        "L2": "Draw and explain the {topic} with a neat diagram. Label all important components.",
        "L3": "Illustrate {topic} with a neat diagram and explain the significance of each component.",
    },
    "case_study": {
        "L5": "Evaluate the suitability of {topic} for the following scenario: {scenario}. Justify your answer.",
        "L6": "Design a solution using {topic} for the given problem statement. Justify your design choices.",
    },
    "short_note": {
        "L2": "Write a short note on {topic}.",
        "L2_detailed": "Write a short note on {topic}, covering its definition, working, and applications.",
    },
}


class QuestionDiscoverer:
    """
    Discovers candidate question texts from an AnswerBlueprint.
    """

    def __init__(self, llm_client=None):
        self.llm = llm_client
        self._prompt_builder = PromptBuilder()

    def discover(
        self,
        blueprint: AnswerBlueprint,
        intent: AssessmentIntent,
        max_candidates: int = 5,
    ) -> List[QuestionCandidate]:
        candidates: List[QuestionCandidate] = []

        # Always generate template candidates first — no latency, no cost
        template_candidates = self._discover_from_templates(blueprint, intent)
        candidates.extend(template_candidates)

        # Neural candidates if LLM available
        if self.llm and len(candidates) < max_candidates:
            neural_candidates = self._discover_neural(blueprint, intent, max_candidates - len(candidates))
            candidates.extend(neural_candidates)

        # Deduplicate by similarity
        candidates = self._deduplicate(candidates)

        logger.info(
            f"[Discoverer] {len(candidates)} candidates for '{blueprint.topic}' "
            f"({blueprint.bloom_level}, {blueprint.marks}M)"
        )
        return candidates[:max_candidates]

    def _discover_from_templates(
        self, blueprint: AnswerBlueprint, intent: AssessmentIntent
    ) -> List[QuestionCandidate]:
        candidates = []
        qtype = blueprint.question_type
        bloom = blueprint.bloom_level
        topic = blueprint.topic
        compare_with = intent.compare_with or "a related concept"

        type_templates = DISCOVERY_TEMPLATES.get(qtype, DISCOVERY_TEMPLATES["explanation"])

        for key, template in type_templates.items():
            if bloom in key or key == bloom:
                text = template.format(
                    topic=topic,
                    compare_with=compare_with,
                    scenario="[describe appropriate scenario]",
                )
                # Add diagram instruction if needed and not already present
                if blueprint.diagram_required and "diagram" not in text.lower():
                    text = text.rstrip(".") + ", with a neat diagram."
                # Add marks hint for 10+ mark questions
                if blueprint.marks >= 10 and "example" not in text.lower():
                    text = text.rstrip(".") + " Provide a suitable example."

                covers = self._detect_covered_components(text, blueprint)
                candidates.append(QuestionCandidate(
                    text=text,
                    source="template",
                    initial_confidence=0.75,
                    covers_components=covers,
                    bloom_alignment=1.0,
                ))

        # Always add a generic fallback
        verb = intent.action_verb
        generic = f"{verb} {topic} with a suitable example."
        if blueprint.diagram_required:
            generic = f"{verb} {topic} with a neat diagram and a suitable example."
        candidates.append(QuestionCandidate(
            text=generic,
            source="template",
            initial_confidence=0.65,
            covers_components=self._detect_covered_components(generic, blueprint),
        ))

        return candidates

    def _discover_neural(
        self,
        blueprint: AnswerBlueprint,
        intent: AssessmentIntent,
        n: int,
    ) -> List[QuestionCandidate]:
        """Generate n question candidates using the LLM."""
        candidates = []

        # Multi-candidate prompt
        prompt = self._build_multi_candidate_prompt(blueprint, intent, n)
        raw = self.llm.generate(prompt, temperature=0.7, max_tokens=300)

        # Parse numbered list from LLM output
        parsed = self._parse_numbered_list(raw, n)
        for text in parsed:
            if len(text.split()) >= 5:
                covers = self._detect_covered_components(text, blueprint)
                candidates.append(QuestionCandidate(
                    text=text,
                    source="neural",
                    initial_confidence=0.80,
                    covers_components=covers,
                ))

        return candidates

    def _build_multi_candidate_prompt(
        self, blueprint: AnswerBlueprint, intent: AssessmentIntent, n: int
    ) -> str:
        components = blueprint.components_summary()
        return (
            f"You are a VTU university examiner.\n\n"
            f"## ANSWER BLUEPRINT (what a student must demonstrate)\n"
            f"Topic: {blueprint.topic}\n"
            f"Bloom Level: {blueprint.bloom_level}\n"
            f"Marks: {blueprint.marks}\n"
            f"Required Answer Components:\n"
            + "\n".join(f"  - {c}" for c in components) + "\n\n"
            f"## TASK\n"
            f"Generate {n} different exam questions whose ideal answer IS this blueprint.\n"
            f"Each question must begin with the verb '{intent.action_verb}'.\n"
            f"Questions must be in VTU examination style.\n"
            f"{'Questions must ask for a diagram.' if blueprint.diagram_required else ''}\n"
            f"Do not generate answers — only questions.\n"
            f"Format: numbered list (1. ... 2. ... etc.)\n\n"
            f"## CANDIDATE QUESTIONS:"
        )

    def _detect_covered_components(
        self, text: str, blueprint: AnswerBlueprint
    ) -> List[str]:
        text_lower = text.lower()
        covered = []
        checks = {
            AnswerComponent.DEFINITION: ["define", "definition", "what is"],
            AnswerComponent.ALGORITHM: ["algorithm", "steps", "procedure"],
            AnswerComponent.DIAGRAM: ["diagram", "draw", "illustrate", "figure"],
            AnswerComponent.EXAMPLE: ["example", "instance", "trace"],
            AnswerComponent.COMPARISON: ["compare", "contrast", "differentiate", "difference"],
            AnswerComponent.COMPLEXITY: ["complexity", "time", "space", "big o"],
            AnswerComponent.APPLICATIONS: ["application", "use case", "used for"],
            AnswerComponent.ADVANTAGES: ["advantage", "benefit", "limitation"],
        }
        for component, keywords in checks.items():
            if any(kw in text_lower for kw in keywords):
                covered.append(component)
        return covered

    def _parse_numbered_list(self, text: str, max_items: int) -> List[str]:
        import re
        pattern = re.compile(r"^\s*\d+\s*[.):]\s*(.+)", re.MULTILINE)
        matches = pattern.findall(text)
        return [m.strip() for m in matches[:max_items] if m.strip()]

    def _deduplicate(self, candidates: List[QuestionCandidate]) -> List[QuestionCandidate]:
        seen: List[str] = []
        unique = []
        for c in candidates:
            words = set(c.text.lower().split())
            is_duplicate = False
            for prev in seen:
                prev_words = set(prev.lower().split())
                overlap = len(words & prev_words) / max(len(words), 1)
                if overlap > 0.75:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique.append(c)
                seen.append(c.text)
        return unique
