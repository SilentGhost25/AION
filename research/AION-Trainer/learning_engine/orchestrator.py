# learning_engine/orchestrator.py
"""
Learning Orchestrator — coordinates all stages of the Academic Learning Loop.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from acb.acb_pipeline import ACBPipeline
from ese.answer_blueprint import AnswerBlueprintBuilder
from server.prompt.assessment_intent import AssessmentIntent

from learning_engine.stages import ConceptStage
from learning_engine.memory.concept_memory import ConceptMemory
from learning_engine.memory.relationship_memory import RelationshipMemory
from learning_engine.memory.examiner_memory import ExaminerMemory
from learning_engine.memory.mistake_memory import MistakeMemory
from learning_engine.memory.confidence_memory import ConfidenceMemory
from learning_engine.memory.question_memory import QuestionMemory
from learning_engine.memory.answer_memory import AnswerMemory
from learning_engine.stages.concept_learner import ConceptLearner
from learning_engine.stages.relationship_builder import RelationshipBuilder
from learning_engine.stages.answer_generator import LearningAnswerGenerator
from learning_engine.stages.question_generator import LearningQuestionGenerator
from learning_engine.stages.self_evaluator import SelfEvaluator
from learning_engine.progress.academic_iq import AcademicIQCalculator, AcademicIQDetails
from learning_engine.progress.epoch_report import EpochReport
from learning_engine.progress.progress_tracker import ProgressTracker

logger = logging.getLogger("aion.learning.orchestrator")


class LearningOrchestrator:
    def __init__(
        self,
        subject_code: str,
        academic_root: str,
        db_dir: Optional[str] = None,
        department: str = "AIML",
        semester: int = 4,
        llm_client=None,
    ):
        self.subject_code = subject_code
        self.academic_root = Path(academic_root)
        self.department = department
        self.semester = semester
        self.llm = llm_client

        # Build paths similar to ACBPipeline
        self.subject_dir = self.academic_root / department / f"semester_{semester}" / subject_code
        if not self.subject_dir.exists():
            self.subject_dir = self.academic_root / subject_code

        self.db_dir = Path(db_dir) if db_dir else self.subject_dir
        self.db_dir.mkdir(parents=True, exist_ok=True)

        # Set up memory directory
        self.mem_dir = self.db_dir / "learning"
        self.mem_dir.mkdir(parents=True, exist_ok=True)

        # Instantiate memory sub-systems
        self.concept_memory = ConceptMemory(str(self.mem_dir / "concept_memory.json"))
        self.relationship_memory = RelationshipMemory(str(self.mem_dir / "relationship_memory.json"))
        self.examiner_memory = ExaminerMemory(str(self.mem_dir / "examiner_memory.json"))
        self.mistake_memory = MistakeMemory(str(self.mem_dir / "mistake_memory.json"))
        self.confidence_memory = ConfidenceMemory(str(self.mem_dir / "confidence_memory.json"))
        self.question_memory = QuestionMemory(str(self.mem_dir / "question_memory.json"))
        self.answer_memory = AnswerMemory(str(self.mem_dir / "answer_memory.json"))
        
        self.progress_tracker = ProgressTracker(str(self.mem_dir / "progress_history.json"))

        # Load existing progress if any
        self.load_memories()

        # Instantiate helper pipeline
        self.acb_pipeline = ACBPipeline(
            subject_code=subject_code,
            academic_root=str(academic_root),
            db_dir=str(db_dir) if db_dir else None,
            department=department,
            semester=semester,
        )

        # Initialize stages
        self.concept_learner = ConceptLearner(
            self.concept_memory, self.relationship_memory, llm_client=llm_client
        )
        self.relationship_builder = RelationshipBuilder(
            self.concept_memory, self.relationship_memory
        )
        self.answer_generator = LearningAnswerGenerator(
            self.concept_memory, self.answer_memory, llm_client=llm_client
        )
        self.question_generator = LearningQuestionGenerator(
            self.concept_memory, self.question_memory, self.mistake_memory
        )
        self.self_evaluator = SelfEvaluator(
            self.concept_memory, self.question_memory, self.answer_memory, self.mistake_memory
        )
        self.iq_calculator = AcademicIQCalculator(
            self.concept_memory,
            self.relationship_memory,
            self.question_memory,
            self.answer_memory,
            self.examiner_memory,
        )

        # Flag indicating if memory bootstrap from store has completed
        self._bootstrapped = False

    def load_memories(self):
        self.concept_memory.load()
        self.relationship_memory.load()
        self.examiner_memory.load()
        self.mistake_memory.load()
        self.confidence_memory.load()
        self.question_memory.load()
        self.answer_memory.load()
        self.progress_tracker.load()

    def save_memories(self):
        self.concept_memory.save()
        self.relationship_memory.save()
        self.examiner_memory.save()
        self.mistake_memory.save()
        self.confidence_memory.save()
        self.question_memory.save()
        self.answer_memory.save()
        self.progress_tracker.save()

    def bootstrap(self):
        """Initialise ConceptMemory from the pipeline's ConceptStore."""
        if not self._bootstrapped:
            self.concept_memory.initialise_from_store(
                self.acb_pipeline.concept_store, self.subject_code
            )
            self._bootstrapped = True
            logger.info("[LearningOrchestrator] Bootstrapped ConceptMemory from ConceptStore")

    def run_epoch(self, epoch: int) -> EpochReport:
        logger.info(f"[LearningOrchestrator] Running learning epoch {epoch}...")
        self.bootstrap()

        all_concepts = self.acb_pipeline.concept_store.concepts_for_subject(self.subject_code)
        if not all_concepts:
            logger.warning("[LearningOrchestrator] No concepts found in ConceptStore. Empty epoch.")
            return EpochReport(epoch, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        # ── Step 1 & 2: Read, Understand & Link ────────────────────────────────
        for concept in all_concepts:
            entry = self.concept_memory.get(concept.concept_id)
            if entry:
                # If confidence is low or flagged weak, study/revisit
                needs_study = entry.needs_revisit() or entry.times_studied == 0
                if needs_study:
                    self.concept_learner.study_concept(concept, epoch)
                    self.concept_learner.build_relationships(concept, all_concepts)

        # ── Step 3: Analyze graph links topologically ─────────────────────────
        self.relationship_builder.analyze_graph(self.subject_code)

        # Populate examiner patterns from existing previous papers / questions
        for concept in all_concepts:
            if concept.previous_paper_frequency > 0:
                self.examiner_memory.observe(
                    verb="Explain",
                    marks=10,
                    bloom="L2",
                    concept_type=concept.scope,
                )

        # ── Step 4: Generate answers for explainable concepts ────────────────
        for concept in all_concepts:
            self.answer_generator.generate_for_concept(concept, epoch)

        # ── Step 5: Generate questions for questionable concepts ─────────────
        blueprint_builder = AnswerBlueprintBuilder()
        for concept in all_concepts:
            entry = self.concept_memory.get(concept.concept_id)
            if entry and entry.current_stage.can_generate_questions():
                # Make intent & answer blueprint
                intent = AssessmentIntent(
                    topic=concept.name,
                    bloom_level="L2",
                    action_verb="Explain",
                    marks=10,
                    question_type="explanation",
                    difficulty="medium",
                    subject_code=self.subject_code,
                )
                ab = blueprint_builder.build(concept, intent)

                self.question_generator.generate_for_concept(
                    concept, ab, intent, epoch
                )

        # ── Step 6: Self-Evaluate & report ────────────────────────────────────
        eval_result = self.self_evaluator.evaluate(self.subject_code, epoch)

        # Compile Report
        num_concepts = len(all_concepts)
        total_rels = self.relationship_memory.total_count()
        learned_rels = self.relationship_memory.learned_count()
        rel_strength = (learned_rels / total_rels * 100) if total_rels > 0 else 0.0

        report = EpochReport(
            epoch=epoch,
            concept_understanding=round(eval_result.avg_understanding * 100, 2),
            relationship_strength=round(rel_strength, 2),
            question_quality=round(eval_result.question_acceptance_rate * 100, 2),
            answer_quality=round(eval_result.answer_quality * 100, 2),
            examiner_similarity=round(self.question_memory.acceptance_rate() * 100, 2),
            grammar=round(self.question_memory.average_grammar_score() * 100, 2),
            coverage=round(self.answer_memory.coverage([c.concept_id for c in all_concepts]) * 100, 2),
            weak_concepts_count=len(eval_result.weak_concepts),
            strong_concepts_count=len(eval_result.strong_concepts),
            weak_concepts=eval_result.weak_concepts,
            strong_concepts=eval_result.strong_concepts,
        )

        # Record snapshots in confidence memory for history curves
        for entry in self.concept_memory._entries.values():
            if entry.subject_code == self.subject_code:
                conf = entry.confidence
                self.confidence_memory.record_snapshot(
                    entry.concept_id,
                    epoch,
                    conf.understand,
                    conf.explain,
                    conf.compare,
                    conf.generate_question,
                    conf.generate_answer,
                    conf.predict_exam_use,
                )

        # Save to progress tracker & file system
        self.progress_tracker.record_epoch(report)
        self.save_memories()

        logger.info(f"[LearningOrchestrator] Epoch {epoch} complete. Understanding: {report.concept_understanding}%")
        return report

    def calculate_iq(self) -> AcademicIQDetails:
        self.bootstrap()
        return self.iq_calculator.calculate(self.subject_code)
