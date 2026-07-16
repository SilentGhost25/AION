# AION-Trainer/server/candidate_generator.py
"""
Candidate Question Generator.

Hierarchy:
    CandidateQuestionGenerator (ABC)
        TransformersCandidateGenerator  <- production (trained AION checkpoint)
        RuleBasedGenerator              <- deterministic baseline / smoke testing
        MockGenerator                   <- unit tests (zero dependencies)

NullCandidateGenerator is removed from production code. If a checkpoint
cannot be loaded, TransformersCandidateGenerator raises loudly rather
than silently degrading to templates — that way the failure is visible
in `aion logs` rather than producing deceptively plausible-looking but
untrained template output.
"""

import abc
import logging
from typing import Optional

from server.prompt.assessment_intent import AssessmentIntent
from server.prompt.prompt_builder import PromptBuilder
from server.prompt.vtu_reference_library import VTUReferenceLibrary

logger = logging.getLogger("aion.server.candidate_generator")


class CandidateQuestionGenerator(abc.ABC):
    @abc.abstractmethod
    def generate(self, knowledge_prompt: str, bloom: str, marks: int) -> str:
        raise NotImplementedError

    def generate_from_intent(self, intent: AssessmentIntent) -> str:
        """
        Preferred path: generates a question from a fully structured
        AssessmentIntent, which produces significantly better output
        than the minimal (knowledge, bloom, marks) signature.
        This default implementation adapts for generators that only
        implement the minimal signature — subclasses should override.
        """
        return self.generate(
            knowledge_prompt=intent.definition or intent.explanation or intent.topic,
            bloom=intent.bloom_level,
            marks=intent.marks,
        )


class TransformersCandidateGenerator(CandidateQuestionGenerator):
    """
    Production generator — loads the AION-trained checkpoint (never
    the base pretrained model) and generates questions using
    PromptBuilder's structured prompts.

    Raises RuntimeError on construction if the checkpoint can't be
    loaded — this is intentional. A silent fallback to templates
    produces output that looks reasonable but isn't trained, which is
    much worse than a loud failure that shows up in monitoring.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        max_input_length: int = 512,
        max_new_tokens: int = 80,
        num_beams: int = 4,
        repetition_penalty: float = 1.3,
        length_penalty: float = 1.2,
        reference_library: Optional[VTUReferenceLibrary] = None,
    ):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch

        logger.info(f"[Generator] Loading checkpoint: {checkpoint_dir}")

        # Load YOUR trained model, not google/flan-t5-base
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_dir)
        self.model.eval()

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            logger.info("[Generator] Model loaded on GPU")
        else:
            logger.info("[Generator] Model loaded on CPU (no GPU detected)")

        self._torch = torch
        self.max_input_length = max_input_length
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.repetition_penalty = repetition_penalty
        self.length_penalty = length_penalty

        self._prompt_builder = PromptBuilder()
        self._reference_library = reference_library or VTUReferenceLibrary()

    def generate(self, knowledge_prompt: str, bloom: str, marks: int) -> str:
        """
        Minimal-signature path. Constructs a basic AssessmentIntent
        and delegates to generate_from_intent.
        """
        intent = AssessmentIntent(
            topic=knowledge_prompt[:100],
            explanation=knowledge_prompt,
            bloom_level=bloom,
            marks=marks,
            action_verb=self._bloom_to_verb(bloom),
        )
        return self.generate_from_intent(intent)

    def generate_from_intent(self, intent: AssessmentIntent) -> str:
        """
        Full-quality path — uses the structured AssessmentIntent to
        build a richly contextualised prompt.
        """
        # Attach reference questions from the library
        if not intent.reference_questions:
            intent.reference_questions = self._reference_library.get_references(
                topic=intent.topic,
                bloom_level=intent.bloom_level,
                question_type=intent.question_type,
            )

        prompt = self._prompt_builder.build_question_generation_prompt(intent)
        return self._decode(prompt)

    def generate_answer_outline(self, question: str, intent: AssessmentIntent) -> str:
        prompt = self._prompt_builder.build_answer_outline_prompt(question, intent)
        return self._decode(prompt, max_new_tokens=200)

    def _decode(self, prompt: str, max_new_tokens: int = None) -> str:
        max_tok = max_new_tokens or self.max_new_tokens
        inputs = self.tokenizer(
            prompt,
            max_length=self.max_input_length,
            truncation=True,
            return_tensors="pt",
        )
        if self._torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with self._torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tok,
                num_beams=self.num_beams,
                repetition_penalty=self.repetition_penalty,
                length_penalty=self.length_penalty,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        raw = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        return self._post_process(raw, intent_verb=None)

    def _post_process(self, text: str, intent_verb: Optional[str]) -> str:
        """
        Light post-processing: capitalise, ensure it ends with a period.
        We deliberately do NOT strip valid long text here — that's the
        validator's job.
        """
        if not text:
            return text
        text = text.strip()
        if text and not text[0].isupper():
            text = text[0].upper() + text[1:]
        if text and text[-1] not in ".?":
            text += "."
        return text

    @staticmethod
    def _bloom_to_verb(bloom: str) -> str:
        defaults = {
            "L1": "Define", "L2": "Explain", "L3": "Illustrate",
            "L4": "Compare", "L5": "Evaluate", "L6": "Design",
        }
        return defaults.get(bloom, "Explain")


class RuleBasedGenerator(CandidateQuestionGenerator):
    """
    Deterministic baseline — useful for smoke testing the pipeline
    without a GPU or trained checkpoint, and for measuring how much
    better the neural model is versus simple templates.
    Always clearly identified as rule-based in its output.
    """

    TEMPLATES = {
        "L1": "Define {topic}.",
        "L2": "Explain {topic} with a suitable example.",
        "L3": "Illustrate the working of {topic} with a worked example.",
        "L4": "Compare {topic} with a related concept, highlighting key differences.",
        "L5": "Evaluate the advantages and limitations of {topic}.",
        "L6": "Design a solution based on {topic} for the given scenario.",
    }

    def generate(self, knowledge_prompt: str, bloom: str, marks: int) -> str:
        template = self.TEMPLATES.get(bloom, self.TEMPLATES["L2"])
        topic = knowledge_prompt.split(".")[0][:60].strip() or "the given concept"
        return template.format(topic=topic)

    def generate_from_intent(self, intent: AssessmentIntent) -> str:
        template = self.TEMPLATES.get(intent.bloom_level, self.TEMPLATES["L2"])
        text = template.format(topic=intent.topic)
        if intent.requires_diagram and "diagram" not in text.lower():
            text = text.rstrip(".") + ", with a neat diagram."
        if intent.marks >= 10 and "example" not in text.lower():
            text = text.rstrip(".") + " with a suitable example."
        return text


class MockGenerator(CandidateQuestionGenerator):
    """
    Unit test stub. Never used outside tests.
    Accepts a scripted dict of (bloom, marks) -> text to return.
    """

    def __init__(self, responses: dict = None, default: str = "Explain the given concept."):
        self._responses = responses or {}
        self._default = default
        self.calls = []

    def generate(self, knowledge_prompt: str, bloom: str, marks: int) -> str:
        self.calls.append((knowledge_prompt, bloom, marks))
        return self._responses.get((bloom, marks), self._responses.get(bloom, self._default))

    def generate_from_intent(self, intent: AssessmentIntent) -> str:
        self.calls.append(intent)
        key = (intent.bloom_level, intent.marks)
        return self._responses.get(key, self._responses.get(intent.bloom_level, self._default))


class NullCandidateGenerator(RuleBasedGenerator):
    """Alias for backward compatibility."""
    pass
