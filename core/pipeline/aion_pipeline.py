"""
AION Universal Academic Pipeline — Senior ML Engineer Design Brief Implementation
=================================================================================
Per AION Development Context (READ FIRST):

Upload
  ↓ Extract
  ↓ Understand
  ↓ Build Concept Graph
  ↓ Ground
  ↓ Reason
  ↓ Plan Question
  ↓ Compose Question
  ↓ Audit
  ↓ Output

NOT: Upload → Prompt LLM → Return Output
LLM is only one component, not the architecture.

This file implements the Universal Academic Pipeline:

Document → Cleaning → OCR → Image Extraction → Concept Extraction
        → Concept Validation → Grounding → Question Planning
        → Question Composition → Question Audit

Properties:
- Stateless APIs (each stage is stateless function)
- Pluggable models / OCR / vision / retriever / LLM
- Every question traceable: Concept ID | Source chunk | Confidence | Expected answer | Bloom | Question
- Confidence-aware recovery (never silently hallucinate)
- Real-world testing only (no dummy data)
"""

from __future__ import annotations

import time
import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

# Stage imports
try:
    from core.extraction.layered_extractor import extract_layered, LayeredExtractionResult  # type: ignore
except ImportError:
    extract_layered = None  # type: ignore

from core.concepts.extractor import ConceptExtractor, ExtractedConcept
from core.concepts.validator import ConceptValidator
from core.concepts.grounding import ConceptGroundingEngine, GroundedConcept
from core.retrieval.concept_retriever import ConceptLevelRetriever
from core.planning.question_planner import QuestionPlanner, PlannerConfig, QuestionPlan
from core.generation.question_composer import QuestionComposer, ComposedQuestion
from core.validation.pipeline import MultiStageValidator, ValidationReport
from core.confidence.recovery import ConfidenceRecoveryEngine


@dataclass
class PipelineMetrics:
    extraction_ms: float = 0.0
    concept_ms: float = 0.0
    grounding_ms: float = 0.0
    planning_ms: float = 0.0
    composition_ms: float = 0.0
    audit_ms: float = 0.0
    total_ms: float = 0.0
    extraction_confidence: float = 0.0
    concepts_extracted: int = 0
    concepts_validated: int = 0
    questions_planned: int = 0
    questions_composed: int = 0
    questions_passed: int = 0
    questions_failed: int = 0
    grounding_avg: float = 0.0
    hallucination_rate: float = 0.0

@dataclass
class AionPipelineResult:
    source: str
    clean_text_path: Optional[Path]
    concepts: List[ExtractedConcept]
    grounded: List[GroundedConcept]
    plans: List[QuestionPlan]
    questions: List[ComposedQuestion]
    validations: List[ValidationReport]
    accepted: List[ComposedQuestion]
    rejected: List[tuple[ComposedQuestion, ValidationReport]]
    metrics: PipelineMetrics
    recovery_note: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    grounding_report: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "clean_text": str(self.clean_text_path) if self.clean_text_path else None,
            "concepts_extracted": len(self.concepts),
            "grounded": len(self.grounded),
            "questions_accepted": len(self.accepted),
            "questions_rejected": len(self.rejected),
            "metrics": asdict(self.metrics),
            "recovery_note": self.recovery_note,
        }


class AionUniversalPipeline:
    """
    Universal Academic Pipeline — runs across all VTU engineering subjects
    without department-specific prompts.
    """

    def __init__(
        self,
        use_llm: bool = True,
        allow_external: bool = False,
        exam_type: str = "SEE",
        difficulty: str = "mixed",
    ):
        self.use_llm = use_llm
        self.exam_type = exam_type
        self.difficulty = difficulty

        # Pluggable components
        self.concept_extractor = ConceptExtractor(use_neural=True)
        self.concept_validator = ConceptValidator(min_confidence=0.45)
        self.grounding_engine = ConceptGroundingEngine(use_llm=use_llm)
        self.retriever = ConceptLevelRetriever(use_neural=True)
        self.planner = QuestionPlanner(PlannerConfig(exam_type=exam_type, difficulty=difficulty))
        self.composer = QuestionComposer(use_llm=use_llm)
        self.validator = MultiStageValidator(strict=True)
        self.recovery_engine = ConfidenceRecoveryEngine(allow_external=allow_external)

    def run(
        self,
        source_path: str | Path,
        output_dir: str | Path = "extracted_output",
        num_questions: int = 4,
        target_bloom: Optional[int] = None,
    ) -> AionPipelineResult:
        """
        Run full pipeline: Upload → Extract → Understand → Concept Graph → Ground → Reason → Plan → Compose → Audit → Output
        Stateless: each call independent.
        """
        t0 = time.time()
        source = str(source_path)
        warnings: List[str] = []
        metrics = PipelineMetrics()

        # ── Stage 1: Extract (Layered — 6 layers) ──────────────
        t = time.time()
        print(f"\n{'='*60}\n[PIPELINE] Stage 1: Layered Extraction — {Path(source).name}\n{'='*60}")
        if extract_layered is None:
            raise ImportError("Layered extractor not available — check core/extraction/layered_extractor.py")
        layered = extract_layered(source_path, output_dir=output_dir)
        metrics.extraction_ms = round((time.time() - t) * 1000, 1)
        metrics.extraction_confidence = layered.overall_confidence
        clean_text = layered.clean_text
        clean_path = layered.output_path
        print(f"[EXTRACT] {layered.word_count:,} words | conf={layered.overall_confidence:.0%} | method={layered.merged_method} | {metrics.extraction_ms}ms")
        if layered.warnings:
            warnings.extend(layered.warnings)

        # Confidence recovery
        recovery = self.recovery_engine.recover(layered)
        recovery_note = self.recovery_engine.mark_output(recovery)
        if recovery.recovery_path != ["no_recovery_needed"]:
            print(f"[RECOVERY] Path: {' → '.join(recovery.recovery_path)} | final_conf={recovery.final_confidence:.0%}")
            warnings.extend(recovery.warnings)
            clean_text = recovery.clean_text
            metrics.extraction_confidence = recovery.final_confidence

        # ── Stage 2: Understand (Concept Extraction) ───────────
        t = time.time()
        print(f"\n[PIPELINE] Stage 2: Concept Extraction (concept-level, not paragraph)")
        # Generate source_id from path
        source_id = hashlib.sha256(source.encode()).hexdigest()[:8]
        concepts = self.concept_extractor.extract(clean_text, source_id=source_id)
        metrics.concept_ms = round((time.time() - t) * 1000, 1)
        metrics.concepts_extracted = len(concepts)
        print(f"[CONCEPTS] Extracted {len(concepts)} concepts | {metrics.concept_ms}ms")
        for c in concepts[:3]:
            print(f"  - [{c.concept_id}] {c.concept_name[:50]} | conf={c.confidence:.0%} | type={c.concept_type}")

        # ── Stage 3: Concept Validation ────────────────────────
        t = time.time()
        print(f"[PIPELINE] Stage 3: Concept Validation")
        valid_concepts, validation_results = self.concept_validator.validate_batch(concepts)
        metrics.concepts_validated = len(valid_concepts)
        rejected = len(concepts) - len(valid_concepts)
        print(f"[VALIDATE] {len(valid_concepts)} valid, {rejected} rejected")
        if rejected:
            for r in validation_results:
                if not r.is_valid:
                    print(f"  ✗ {r.concept_id}: {r.reason}")

        # Build concept graph (index for retrieval)
        # Understand → Build Concept Graph
        print(f"[PIPELINE] Stage 3.5: Build Concept Graph + Index (retrieval)")
        self.retriever.index(valid_concepts)

        # ── Stage 4: Ground ────────────────────────────────────
        t = time.time()
        print(f"[PIPELINE] Stage 4: Grounding (Text → Concept → Evidence → Expected Answer → Question)")
        grounded = self.grounding_engine.ground(valid_concepts, target_bloom=target_bloom)
        metrics.grounding_ms = round((time.time() - t) * 1000, 1)
        if grounded:
            metrics.grounding_avg = round(sum(g.confidence for g in grounded) / len(grounded), 2)
        print(f"[GROUND] {len(grounded)} grounded | avg_conf={metrics.grounding_avg:.0%} | {metrics.grounding_ms}ms")
        for g in grounded[:2]:
            print(f"  ▶ [{g.concept.concept_id}] Bloom L{g.bloom_level} | expected: {g.expected_answer[:90]}...")

        # ── Stage 5: Reason (Retrieve context for planning) ────
        print(f"[PIPELINE] Stage 5: Reason (retrieve related concepts for planning)")
        # Retrieval is woven into planner — each plan can pull related concepts via retriever
        # Here we just demonstrate one retrieval
        if grounded:
            sample_q = grounded[0].concept.concept_name
            retrieved = self.retriever.retrieve(sample_q, top_k=3)
            print(f"[RETRIEVE] Sample '{sample_q}' → {len(retrieved)} related concepts")

        # ── Stage 6: Plan Question ─────────────────────────────
        t = time.time()
        print(f"[PIPELINE] Stage 6: Question Planning (Planner decides, Composer will write)")
        # Limit to num_questions * factor for candidates, then select top
        # Planner already handles distribution; we request more than needed and let audit filter
        self.planner.config.num_questions = num_questions * 2  # generate 2x, audit will keep best
        plans = self.planner.plan(grounded)
        # Take top num_questions*2 for composition, will filter to num_questions accepted later
        plans = plans[: num_questions * 2]
        metrics.planning_ms = round((time.time() - t) * 1000, 1)
        metrics.questions_planned = len(plans)
        print(f"[PLAN] {len(plans)} plans | {metrics.planning_ms}ms")
        for p in plans[:3]:
            print(f"  ◇ [{p.plan_id}] {p.concept_name[:40]} | L{p.bloom_level} {p.action_verb} | {p.marks}m | {p.question_type} | conf={p.confidence:.0%}")

        # ── Stage 7: Compose Question ──────────────────────────
        t = time.time()
        print(f"[PIPELINE] Stage 7: Question Composition (Composer writes English)")
        questions = self.composer.compose_batch(plans)
        metrics.composition_ms = round((time.time() - t) * 1000, 1)
        metrics.questions_composed = len(questions)
        print(f"[COMPOSE] {len(questions)} composed | {metrics.composition_ms}ms")
        for q in questions[:2]:
            print(f"  ✎ [{q.concept_id}] {q.question_text}")

        # ── Stage 8: Audit (Multi-stage Validation) ────────────
        t = time.time()
        print(f"[PIPELINE] Stage 8: Audit (7 gates: Grammar → Semantic → Bloom → Grounding → Marks → Diagram → Final)")
        validations: List[ValidationReport] = []
        accepted: List[ComposedQuestion] = []
        rejected_list: List[tuple[ComposedQuestion, ValidationReport]] = []

        for q, plan in zip(questions, plans):
            report = self.validator.validate(
                q,
                plan=plan,
                evidence=q.grounding.get("evidence_snippet", ""),
                expected_answer=q.expected_answer,
            )
            validations.append(report)
            if report.overall_passed:
                accepted.append(q)
                print(f"  ✓ PASS [{q.concept_id}] score={report.overall_score:.0%} gates=✓")
            else:
                rejected_list.append((q, report))
                print(f"  ✗ REJECT [{q.concept_id}] score={report.overall_score:.0%} codes={report.reason_codes} | {report.gates[-1].reason}")

        # If too few accepted, try to accept top rejected by score (graceful degradation)
        if len(accepted) < num_questions and rejected_list:
            rejected_list.sort(key=lambda x: x[1].overall_score, reverse=True)
            need = num_questions - len(accepted)
            # Only promote if score >= 0.55 and not critical semantic/grounding fail
            for q, rep in rejected_list[:need]:
                if rep.overall_score >= 0.55 and not any(c in ("RC-01: semantic hallucination", "RC-07: grounding insufficient") for c in rep.reason_codes):
                    print(f"  ~ PROMOTED low-confidence [{q.concept_id}] score={rep.overall_score:.0%}")
                    accepted.append(q)
            # Remove promoted from rejected
            rejected_list = [x for x in rejected_list if x[0] not in accepted]

        # Trim to requested num_questions
        accepted = accepted[:num_questions]

        metrics.audit_ms = round((time.time() - t) * 1000, 1)
        metrics.questions_passed = len(accepted)
        metrics.questions_failed = len(rejected_list)
        metrics.hallucination_rate = 0.0 if not validations else round(
            sum(1 for v in validations if any("hallucination" in c.lower() or "RC-01" in c for c in v.reason_codes)) / len(validations), 3
        )
        metrics.total_ms = round((time.time() - t0) * 1000, 1)

        print(f"[AUDIT] {len(accepted)} accepted, {len(rejected_list)} rejected | {metrics.audit_ms}ms")
        print(f"{'='*60}\n[PIPELINE] Done in {metrics.total_ms}ms | Extraction {metrics.extraction_confidence:.0%} | "
              f"Grounding {metrics.grounding_avg:.0%} | Hallucination {metrics.hallucination_rate:.0%}\n{'='*60}")

        # Grounding report
        grounding_report = {
            "extraction_method": layered.merged_method,
            "extraction_confidence": metrics.extraction_confidence,
            "grounding_avg": metrics.grounding_avg,
            "hallucination_rate": metrics.hallucination_rate,
            "every_question_has": ["Concept ID", "Source chunk", "Confidence", "Expected answer", "Bloom level", "Question"],
        }

        if metrics.total_ms > 60000:
            warnings.append(f"Performance target missed: {metrics.total_ms}ms > 60s for {layered.word_count} words")

        return AionPipelineResult(
            source=source,
            clean_text_path=clean_path,
            concepts=concepts,
            grounded=grounded,
            plans=plans,
            questions=questions,
            validations=validations,
            accepted=accepted,
            rejected=rejected_list,
            metrics=metrics,
            recovery_note=recovery_note,
            warnings=warnings,
            grounding_report=grounding_report,
        )

    def run_legacy_adapter(self, file_path: str, **kwargs) -> tuple[List[dict], List[dict]]:
        """
        Adapter for v0_1.main.run_pipeline compatibility.
        Returns (accepted, rejected) as dicts.
        """
        result = self.run(file_path, **kwargs)
        accepted_dicts = []
        for q in result.accepted:
            accepted_dicts.append({
                "question_text": q.question_text,
                "concept_id": q.concept_id,
                "source_hash": q.source_hash,
                "marks": q.marks,
                "bloom_level": q.bloom_level,
                "expected_answer": q.expected_answer,
                "grounding": q.grounding,
                "confidence": q.confidence,
            })
        rejected_dicts = []
        for q, rep in result.rejected:
            rejected_dicts.append({
                "question_text": q.question_text,
                "concept_id": q.concept_id,
                "reason_codes": rep.reason_codes,
                "overall_score": rep.overall_score,
            })
        return accepted_dicts, rejected_dicts
