# aion_core/generation/question_generator.py
"""
QuestionGenerator — updated to use AssessmentIntent and PromptBuilder
when a full intent is available, falling back to the minimal signature
for backward compatibility with code that hasn't been updated yet.
No structural changes — existing callers all still work.
"""

from typing import Optional
from aion_core.engine_base import AionEngine, EngineRole
from aion_core.schemas import QuestionIntent, GeneratedQuestion
from aion_core.knowledge.answer_graph import AnswerNode
from aion_core.planning.question_discovery import QuestionCandidate
from aion_core.llm_client import LLMClient


class QuestionGenerator(AionEngine):
    role = EngineRole.QUESTION_GENERATOR
    name = "QuestionGenerator"
    version = "1.1.0"

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client
        # Import lazily — aion_core does not depend on server-side modules
        self._prompt_builder = None
        self._try_load_prompt_builder()

    def _try_load_prompt_builder(self):
        try:
            from server.prompt.prompt_builder import PromptBuilder
            from server.prompt.assessment_intent import AssessmentIntent
            from server.prompt.vtu_reference_library import VTUReferenceLibrary
            self._prompt_builder = PromptBuilder()
            self._reference_library = VTUReferenceLibrary()
            self._AssessmentIntent = AssessmentIntent
        except ImportError:
            pass  # server package not installed; fall back to simple prompts

    def health_check(self) -> bool:
        return True

    def generate(
        self,
        node: AnswerNode,
        intent: QuestionIntent,
        candidate: QuestionCandidate,
    ) -> GeneratedQuestion:
        text = candidate.text

        if self.llm:
            text = self._generate_with_best_prompt(node, intent, candidate)

        outline = node.expected_answer
        if self.llm:
            outline = self._generate_answer_outline(text, node, intent)

        return GeneratedQuestion(
            text=text,
            expected_answer_outline=outline,
            intent=intent,
        )

    def _generate_with_best_prompt(
        self,
        node: AnswerNode,
        intent: QuestionIntent,
        candidate: QuestionCandidate,
    ) -> str:
        """
        Uses structured PromptBuilder if available (server environment),
        otherwise falls back to the simple polish prompt used previously.
        """
        if self._prompt_builder is not None:
            assessment_intent = self._build_assessment_intent(node, intent)
            prompt = self._prompt_builder.build_question_generation_prompt(assessment_intent)
            result = self.llm.generate(prompt, temperature=0.3, max_tokens=100)
            if result.strip():
                return self._post_process(result.strip())

        # Simple fallback (no PromptBuilder available)
        polish_prompt = (
            f"Rewrite this exam question in formal, grammatically correct "
            f"academic English, keeping the meaning identical. "
            f"Begin with the verb '{intent.action_verb}'. "
            f"Do not add new requirements. "
            f"Maximum 35 words. "
            f'Question: "{candidate.text}"'
        )
        polished = self.llm.generate(polish_prompt, temperature=0.2)
        return polished.strip().strip('"') if polished.strip() else candidate.text

    def _generate_answer_outline(
        self, question: str, node: AnswerNode, intent: QuestionIntent
    ) -> str:
        if self._prompt_builder is not None:
            assessment_intent = self._build_assessment_intent(node, intent)
            prompt = self._prompt_builder.build_answer_outline_prompt(question, assessment_intent)
        else:
            prompt = (
                f"Write a concise expected-answer outline (bullet points) "
                f"for this {intent.marks}-mark question:\n\"{question}\"\n\n"
                f"Ground your answer strictly in this material:\n"
                f"{node.expected_answer[:1000]}"
            )
        result = self.llm.generate(prompt, temperature=0.2, max_tokens=300)
        return result or node.expected_answer

    def _build_assessment_intent(self, node: AnswerNode, intent: QuestionIntent):
        gene = node.gene
        ai = self._AssessmentIntent(
            topic=gene.topic,
            definition=gene.definition,
            explanation=gene.explanation[:400],
            key_points=gene.key_points[:6],
            algorithms=gene.algorithms[:4],
            applications=gene.applications[:4],
            formulas=getattr(gene, "formulas", [])[:3],
            diagram_description=gene.diagram_description,
            bloom_level=intent.bloom_level,
            action_verb=intent.action_verb,
            marks=intent.marks,
            requires_diagram=intent.requires_diagram,
            compare_with=intent.compare_with,
            subject_code=getattr(gene, "subject_code", ""),
            module=getattr(gene, "module", 0),
            previously_asked=node.asked_questions[-5:],
        )
        if self._reference_library:
            ai.reference_questions = self._reference_library.get_references(
                topic=gene.topic,
                bloom_level=intent.bloom_level,
                question_type=self._intent_to_qtype(intent),
            )
        return ai

    def _intent_to_qtype(self, intent: QuestionIntent) -> str:
        if intent.compare_with:
            return "comparison"
        if intent.requires_diagram:
            return "diagram_based"
        if intent.action_verb.lower() in ("trace", "apply", "implement", "solve"):
            return "algorithm"
        return "explanation"

    def _post_process(self, text: str) -> str:
        if text and not text[0].isupper():
            text = text[0].upper() + text[1:]
        if text and text[-1] not in ".?":
            text += "."
        return text
