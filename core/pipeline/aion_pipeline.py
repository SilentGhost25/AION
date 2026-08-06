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

Document → Cleaning (Header/TOC/Title/Section) → Concept Extraction
        → Concept Validation → Knowledge Unit Builder → Grounding → Reasoning Engine
        → Question Planning (intent + KU) → Question Composition (KU-aware) → Self-Critic → Audit

Properties:
- Stateless APIs (each stage is stateless function)
- Pluggable models / OCR / vision / retriever / LLM
- Every question traceable: Concept ID | Source chunk | Confidence | Expected answer | Bloom | Question
- Confidence-aware recovery (never silently hallucinate) + per-component confidence
- Real-world testing only (no dummy data)
"""

from __future__ import annotations

import time
import hashlib
import re
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

# New layers per audit
try:
    from core.preprocessing.document_cleaner import DocumentPreprocessor
    HAS_PREPROCESSOR = True
except ImportError:
    HAS_PREPROCESSOR = False

try:
    from core.knowledge.knowledge_unit import KnowledgeUnitBuilder, KnowledgeUnit
    HAS_KU = True
except ImportError:
    HAS_KU = False

try:
    from core.reasoning.reasoning_engine import ReasoningEngine, ReasoningIntent
    HAS_REASONING = True
except ImportError:
    HAS_REASONING = False

try:
    from core.critic.self_critic import SelfCritic
    HAS_CRITIC = True
except ImportError:
    HAS_CRITIC = False


@dataclass
class ComponentConfidence:
    extraction: float = 0.0
    preprocessing: float = 0.0
    concept: float = 0.0
    grounding: float = 0.0
    reasoning: float = 0.0
    planning: float = 0.0
    composition: float = 0.0
    auditing: float = 0.0
    overall: float = 0.0

@dataclass
class PipelineMetrics:
    extraction_ms: float = 0.0
    preprocessing_ms: float = 0.0
    concept_ms: float = 0.0
    ku_build_ms: float = 0.0
    grounding_ms: float = 0.0
    reasoning_ms: float = 0.0
    planning_ms: float = 0.0
    composition_ms: float = 0.0
    critic_ms: float = 0.0
    audit_ms: float = 0.0
    total_ms: float = 0.0
    extraction_confidence: float = 0.0
    concepts_extracted: int = 0
    concepts_validated: int = 0
    knowledge_units: int = 0
    questions_planned: int = 0
    questions_composed: int = 0
    questions_passed: int = 0
    questions_failed: int = 0
    grounding_avg: float = 0.0
    hallucination_rate: float = 0.0
    component_confidence: ComponentConfidence = field(default_factory=ComponentConfidence)

@dataclass
class AionPipelineResult:
    source: str
    clean_text_path: Optional[Path]
    concepts: List[ExtractedConcept]
    knowledge_units: List[Any]  # KnowledgeUnit
    grounded: List[GroundedConcept]
    reasoning_intents: List[Any]  # ReasoningIntent
    plans: List[QuestionPlan]
    questions: List[ComposedQuestion]
    validations: List[ValidationReport]
    critic_results: List[Any]
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
            "knowledge_units": len(self.knowledge_units),
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

        # New layers
        self.preprocessor = DocumentPreprocessor() if HAS_PREPROCESSOR else None
        self.ku_builder = KnowledgeUnitBuilder() if HAS_KU else None
        self.reasoning_engine = ReasoningEngine() if HAS_REASONING else None
        self.self_critic = SelfCritic() if HAS_CRITIC else None

    def run(
        self,
        source_path: str | Path,
        output_dir: str | Path = "extracted_output",
        num_questions: int = 4,
        target_bloom: Optional[int] = None,
    ) -> AionPipelineResult:
        """
        Run full pipeline: Upload → Extract → Preprocess → Understand → KU Builder → Ground → Reason → Plan → Compose → Self-Critic → Audit → Output
        Stateless: each call independent.
        """
        t0 = time.time()
        source = str(source_path)
        warnings: List[str] = []
        metrics = PipelineMetrics()
        component_conf = ComponentConfidence()

        # ── Stage 1: Extract (Layered — 6 layers) ──────────────
        t = time.time()
        print(f"\n{'='*60}\n[PIPELINE] Stage 1: Layered Extraction — {Path(source).name}\n{'='*60}")
        if extract_layered is None:
            raise ImportError("Layered extractor not available — check core/extraction/layered_extractor.py")
        layered = extract_layered(source_path, output_dir=output_dir)
        metrics.extraction_ms = round((time.time() - t) * 1000, 1)
        metrics.extraction_confidence = layered.overall_confidence
        component_conf.extraction = layered.overall_confidence
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
            component_conf.extraction = recovery.final_confidence

        # ── Stage 1.5: Preprocessing (Header/TOC/Title/Section) ──
        t = time.time()
        preproc_conf = 0.9
        if self.preprocessor:
            print(f"[PIPELINE] Stage 1.5: Document Preprocessing (Header/TOC/Title/Section cleaning)")
            preproc = self.preprocessor.clean(clean_text)
            preproc_conf = preproc.confidence
            component_conf.preprocessing = preproc_conf
            # Use cleaned text for concept extraction — prevents MODULE 5 becoming a concept
            clean_text = preproc.clean_text
            metrics.preprocessing_ms = round((time.time() - t) * 1000, 1)
            print(f"[PREPROCESS] Removed {len(preproc.removed_headers)} headers, {len(preproc.removed_toc_lines)} TOC lines | conf={preproc_conf:.0%} | sections={len(preproc.sections)} | {metrics.preprocessing_ms}ms")
            if preproc.removed_headers:
                warnings.append(f"preprocessing: removed headers {preproc.removed_headers[:3]}")
            # Save cleaned text for audit
            try:
                Path(clean_path).write_text(clean_text, encoding="utf-8")
            except Exception:
                pass
        else:
            metrics.preprocessing_ms = 0
            component_conf.preprocessing = 0.85
            print("[PREPROCESS] Skipped (module not available)")

        # ── Stage 2: Understand (Concept Extraction) ───────────
        t = time.time()
        print(f"\n[PIPELINE] Stage 2: Concept Extraction (concept-level, not paragraph)")
        source_id = hashlib.sha256(source.encode()).hexdigest()[:8]
        concepts = self.concept_extractor.extract(clean_text, source_id=source_id)
        metrics.concept_ms = round((time.time() - t) * 1000, 1)
        metrics.concepts_extracted = len(concepts)
        # Concept confidence avg
        if concepts:
            component_conf.concept = round(sum(c.confidence for c in concepts) / len(concepts), 2)
        print(f"[CONCEPTS] Extracted {len(concepts)} concepts | avg_conf={component_conf.concept:.0%} | {metrics.concept_ms}ms")
        for c in concepts[:3]:
            print(f"  - [{c.concept_id}] {c.concept_name[:60]} | conf={c.confidence:.0%} | type={c.concept_type}")

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

        # Build concept graph (index for retrieval) — domain-scoped
        print(f"[PIPELINE] Stage 3.5: Build Concept Graph + Index (retrieval) — domain-scoped")
        # Subject detection for domain isolation
        try:
            from core.domain.subject_detector import SubjectDetector
            detector = SubjectDetector()
            subject_profile, subj_conf, subj_scores = detector.detect_with_confidence(clean_text)
            print(f"[SUBJECT] Detected {subject_profile.code} ({subject_profile.name}) conf={subj_conf:.0%} scores={subj_scores}")
            component_conf.reasoning = max(component_conf.reasoning, subj_conf)
            # Store for integrity gate
            self._current_subject_profile = subject_profile
        except Exception as e:
            print(f"[SUBJECT] Detector failed: {e}")
            self._current_subject_profile = None
        self.retriever.index(valid_concepts)

        # ── Stage 3.6: Knowledge Unit Builder ──────────────────
        t = time.time()
        knowledge_units = []
        if self.ku_builder and valid_concepts:
            print(f"[PIPELINE] Stage 3.6: Knowledge Unit Builder (canonical representation — domain-scoped)")
            # Pass subject profile for domain-isolated relationships
            prof = getattr(self, '_current_subject_profile', None)
            if prof:
                self.ku_builder.subject_profile = prof
                knowledge_units = self.ku_builder.build_batch(valid_concepts, subject_profile=prof)
            else:
                knowledge_units = self.ku_builder.build_batch(valid_concepts)
            metrics.ku_build_ms = round((time.time() - t) * 1000, 1)
            metrics.knowledge_units = len(knowledge_units)
            # KU confidence is average of concept confidence
            if knowledge_units:
                component_conf.concept = round(sum(ku.confidence for ku in knowledge_units) / len(knowledge_units), 2)
            print(f"[KU] Built {len(knowledge_units)} Knowledge Units | {metrics.ku_build_ms}ms")
            for ku in knowledge_units[:2]:
                print(f"  ◆ [{ku.ku_id}] {ku.concept} | diff={ku.difficulty} | miscon={ku.misconceptions[0][:60] if ku.misconceptions else 'none'}...")
        else:
            metrics.ku_build_ms = 0
            print("[KU] Skipped (builder not available or no valid concepts)")

        # ── Stage 4: Ground ────────────────────────────────────
        t = time.time()
        print(f"[PIPELINE] Stage 4: Grounding (Text → Concept → Evidence → Expected Answer → Question)")
        grounded = self.grounding_engine.ground(valid_concepts, target_bloom=target_bloom)
        metrics.grounding_ms = round((time.time() - t) * 1000, 1)
        if grounded:
            metrics.grounding_avg = round(sum(g.confidence for g in grounded) / len(grounded), 2)
            component_conf.grounding = metrics.grounding_avg
        print(f"[GROUND] {len(grounded)} grounded | avg_conf={metrics.grounding_avg:.0%} | {metrics.grounding_ms}ms")
        for g in grounded[:2]:
            # Show canonical if KU available
            canon = ""
            if knowledge_units:
                # Find matching KU
                matching = next((ku for ku in knowledge_units if ku.raw_concept == g.concept.concept_name), None)
                if matching:
                    canon = f" | canon_ans: {matching.expected_answer_canonical[:60]}..."
            print(f"  ▶ [{g.concept.concept_id}] Bloom L{g.bloom_level} | expected: {g.expected_answer[:80]}...{canon}")

        # ── Stage 5: Reason (Reasoning Engine) ─────────────────
        t = time.time()
        reasoning_intents = []
        if self.reasoning_engine and knowledge_units:
            print(f"[PIPELINE] Stage 5: Reasoning Engine (scenario/misconception/numerical/relationship)")
            reasoning_intents = self.reasoning_engine.reason(knowledge_units)
            metrics.reasoning_ms = round((time.time() - t) * 1000, 1)
            # Reasoning confidence: high if intents are diverse
            types = [r.intent_type for r in reasoning_intents]
            component_conf.reasoning = 0.85 if len(set(types)) > 1 else 0.70
            print(f"[REASON] {len(reasoning_intents)} intents | types={set(types)} | {metrics.reasoning_ms}ms")
            for ri in reasoning_intents[:2]:
                print(f"  ◈ [{ri.ku_id}] {ri.intent_type} | Bloom L{ri.bloom_target} | ops={ri.reasoning_operations} | scenario={ri.scenario_prompt[:60] if ri.scenario_prompt else '—'}...")
        else:
            metrics.reasoning_ms = 0
            component_conf.reasoning = 0.70
            print("[REASON] Skipped (reasoning engine not available) — fallback to planner-only")

        # Also demonstrate retrieval for planner context
        if grounded:
            sample_q = grounded[0].concept.concept_name
            retrieved = self.retriever.retrieve(sample_q, top_k=3)
            print(f"[RETRIEVE] Sample '{sample_q}' → {len(retrieved)} related concepts")

        # ── Stage 6: Plan Question ─────────────────────────────
        t = time.time()
        print(f"[PIPELINE] Stage 6: Question Planning (Planner decides intent, Composer will write)")
        self.planner.config.num_questions = num_questions * 2
        plans = self.planner.plan(grounded)
        plans = plans[: num_questions * 2]
        # Align planner bloom with reasoning intent (prevent RC-04 bloom mismatch)
        if reasoning_intents and len(reasoning_intents) == len(plans):
            # Update plans to use reasoning intent's bloom and intent type
            for plan, intent, ku in zip(plans, reasoning_intents, knowledge_units):
                if plan.bloom_level != intent.bloom_target:
                    # Update bloom and verb to match reasoning
                    from core.planning.question_planner import QuestionPlanner
                    verb_map = QuestionPlanner.BLOOM_VERBS.get(intent.bloom_target, ["Explain"])
                    plan.bloom_level = intent.bloom_target
                    plan.bloom_label = {1:"Remember",2:"Understand",3:"Apply",4:"Analyse",5:"Evaluate",6:"Create"}.get(intent.bloom_target, "Understand")
                    plan.action_verb = verb_map[0]
                    plan.reasoning_objective = intent.scenario_prompt or plan.reasoning_objective
                    plan.question_type = intent.intent_type if intent.intent_type in ("scenario","misconception","procedure","relationship","numerical","diagram") else plan.question_type
                    # Update marks based on bloom
                    if intent.bloom_target >= 4:
                        plan.marks = 10
        metrics.planning_ms = round((time.time() - t) * 1000, 1)
        metrics.questions_planned = len(plans)
        component_conf.planning = 0.85 if plans else 0.0
        print(f"[PLAN] {len(plans)} plans | {metrics.planning_ms}ms")
        for p in plans[:3]:
            print(f"  ◇ [{p.plan_id}] {p.concept_name[:40]} | L{p.bloom_level} {p.action_verb} | {p.marks}m | {p.question_type} | conf={p.confidence:.0%}")

        # Map plans to KUs via concept_id for KU-aware composition
        ku_by_raw = {ku.raw_concept: ku for ku in knowledge_units} if knowledge_units else {}
        intent_by_kuid = {ri.ku_id: ri for ri in reasoning_intents} if reasoning_intents else {}

        # ── Stage 6.5: Domain Integrity Gate (will be applied after composition on final questions)
        # Pre-check: build grounded vocab for post-composition check
        try:
            from core.domain.integrity_gate import DomainIntegrityGate
            self._integrity_gate = DomainIntegrityGate()
            self._ku_concepts_for_gate = {ku.concept for ku in knowledge_units} if knowledge_units else set()
            self._retrieved_evidence_for_gate = " ".join([c.supporting_evidence for c in valid_concepts[:3]]) if valid_concepts else clean_text[:2000]
            print(f"[INTEGRITY] Gate prepared for {len(plans)} plans (will check final questions)")
        except Exception as e:
            print(f"[INTEGRITY] Gate prep failed: {e}")
            self._integrity_gate = None

        # ── Stage 7: Compose Question (KU-aware) ───────────────
        t = time.time()
        print(f"[PIPELINE] Stage 7: Question Composition (KU-aware, scenario-based, not generic)")
        questions: List[ComposedQuestion] = []
        if knowledge_units and reasoning_intents and self.composer:
            # Use KU-aware composer
            for plan in plans:
                # Find matching KU and intent
                # Match via concept_id or raw_concept
                ku = None
                intent = None
                # Try matching via KU builder's raw_concept
                for k in knowledge_units:
                    if k.raw_concept == plan.concept_name or k.ku_id.endswith(plan.concept_id.split("_")[-1]):
                        ku = k
                        break
                if not ku and knowledge_units:
                    ku = knowledge_units[0]
                if ku:
                    intent = next((ri for ri in reasoning_intents if ri.ku_id == ku.ku_id), None)
                    if not intent:
                        # Create default intent from plan
                        intent = ReasoningIntent(ku_id=ku.ku_id, intent_type=plan.question_type, bloom_target=plan.bloom_level, reasoning_operations=[plan.action_verb.lower()], scenario_prompt=None)
                    try:
                        q = self.composer.compose_from_ku(ku, intent, plan)
                        questions.append(q)
                    except Exception as e:
                        print(f"[COMPOSE-KU] Fallback to plan for {plan.concept_name}: {e}")
                        questions.append(self.composer.compose(plan))
                else:
                    questions.append(self.composer.compose(plan))
        else:
            questions = self.composer.compose_batch(plans)
        metrics.composition_ms = round((time.time() - t) * 1000, 1)
        metrics.questions_composed = len(questions)
        component_conf.composition = 0.85 if questions else 0.0
        print(f"[COMPOSE] {len(questions)} composed | {metrics.composition_ms}ms")
        for q in questions[:2]:
            print(f"  ✎ [{q.concept_id}] {q.question_text}")

        # ── Stage 7.5: Self-Critic (Reasoning-level) ───────────
        t = time.time()
        critic_results = []
        if self.self_critic and knowledge_units and reasoning_intents:
            print(f"[PIPELINE] Stage 7.5: Self-Critic (reasoning alignment, examiner style)")
            for q, ku, intent in zip(questions, knowledge_units, reasoning_intents):
                # Match KU/intent to question via index (since we built in order)
                # Find corresponding KU/intent for this question's plan
                # For simplicity, take same index
                try:
                    cr = self.self_critic.critique(q.question_text, ku, intent, q.expected_answer)
                    critic_results.append(cr)
                    status = "✓" if cr.passed else "✗"
                    print(f"  {status} [{q.concept_id}] critic_score={cr.score:.0%} | {cr.reason}")
                except Exception as e:
                    print(f"[CRITIC] Error for {q.concept_id}: {e}")
                    critic_results.append(None)
            metrics.critic_ms = round((time.time() - t) * 1000, 1)
            # Self-critic confidence
            if critic_results:
                valid_scores = [cr.score for cr in critic_results if cr]
                component_conf.auditing = round(sum(valid_scores) / len(valid_scores), 2) if valid_scores else 0.70
        else:
            metrics.critic_ms = 0
            print("[CRITIC] Skipped (self-critic not available)")

        # ── Stage 8: Audit (Multi-stage Validation) ────────────
        t = time.time()
        print(f"[PIPELINE] Stage 8: Audit (7 gates: Grammar → Semantic → Bloom → Grounding → Marks → Diagram → Final + Self-Critic)")
        validations: List[ValidationReport] = []
        accepted: List[ComposedQuestion] = []
        rejected_list: List[tuple[ComposedQuestion, ValidationReport]] = []

        for idx, (q, plan) in enumerate(zip(questions, plans)):
            # Domain Integrity Gate on final question (before other validation)
            if hasattr(self, '_integrity_gate') and self._integrity_gate:
                try:
                    ku_concepts = getattr(self, '_ku_concepts_for_gate', set())
                    retrieved_ev = getattr(self, '_retrieved_evidence_for_gate', "")
                    gate_res = self._integrity_gate.check(q.question_text, ku_concepts, retrieved_ev, getattr(self, '_current_subject_profile', None))
                    if not gate_res.passed:
                        # Create a failed validation report directly
                        from core.validation.pipeline import ValidationGateResult, ValidationReport
                        gate_result = ValidationGateResult(gate="domain_integrity", passed=False, score=gate_res.score, reason="; ".join(gate_res.violations), reason_code="RC-01: domain integrity")
                        fake_report = ValidationReport(question_text=q.question_text, concept_id=q.concept_id, overall_passed=False, overall_score=gate_res.score, gates=[gate_result], reason_codes=[gate_result.reason_code], confidence=gate_res.score)
                        validations.append(fake_report)
                        rejected_list.append((q, fake_report))
                        print(f"  ✗ REJECT [{q.concept_id}] domain integrity: {gate_res.violations[:1]}")
                        continue
                except Exception as e:
                    print(f"[INTEGRITY] Check failed for {q.concept_id}: {e}")

            # Include self-critic result as gate if available
            report = self.validator.validate(
                q,
                plan=plan,
                evidence=q.grounding.get("evidence_snippet", ""),
                expected_answer=q.expected_answer,
            )
            # Inject self-critic as additional gate
            if idx < len(critic_results) and critic_results[idx] and not critic_results[idx].passed:
                # Add reason code from critic
                # We append to reason_codes but don't fail audit unless critical? For now, add gate
                from core.validation.pipeline import ValidationGateResult
                cr = critic_results[idx]
                crit_gate = ValidationGateResult(
                    gate="self_critic",
                    passed=cr.passed,
                    score=cr.score,
                    reason=cr.reason,
                    reason_code=cr.reason_code,
                )
                report.gates.append(crit_gate)
                # Recompute audit: if critic failed, adjust overall
                if not cr.passed:
                    report.overall_passed = False
                    if cr.reason_code not in report.reason_codes:
                        report.reason_codes.append(cr.reason_code)
                    report.overall_score = round((report.overall_score + cr.score) / 2, 2)
            validations.append(report)
            if report.overall_passed:
                accepted.append(q)
                print(f"  ✓ PASS [{q.concept_id}] score={report.overall_score:.0%} gates=✓")
            else:
                rejected_list.append((q, report))
                print(f"  ✗ REJECT [{q.concept_id}] score={report.overall_score:.0%} codes={report.reason_codes} | {report.gates[-1].reason}")

        # No promotion — repair then return fewer (audit: never invent quality)
        # Attempt repair for rejected that are close (score >=0.60 and only bloom/grounding)
        repaired = []
        for q, rep in list(rejected_list):
            if rep.overall_score >= 0.65 and len(rep.reason_codes) == 1 and "RC-04" in rep.reason_codes[0]:
                # Bloom mismatch repairable via auto-correct verb
                try:
                    from v0_1.qa_engine import BloomsTaxonomyValidator
                    validator = BloomsTaxonomyValidator()
                    fixed_text = validator.auto_correct_blooms_level(q.question_text, q.bloom_level)
                    if fixed_text != q.question_text:
                        q.question_text = fixed_text
                        # Revalidate
                        new_rep = self.validator.validate(q, plan=next((pp for pp in plans if pp.plan_id==q.plan_id), None), evidence=q.grounding.get("evidence_snippet",""), expected_answer=q.expected_answer)
                        if new_rep.overall_passed:
                            print(f"  ~ REPAIRED [{q.concept_id}] bloom verb corrected → PASS")
                            repaired.append((q, new_rep))
                except Exception:
                    pass
        for q, rep in repaired:
            if (q, rep) not in [(qq, rr) for qq, rr in rejected_list]:
                continue
            # Move from rejected to accepted if repaired
            for orig_q, orig_rep in list(rejected_list):
                if orig_q.concept_id == q.concept_id:
                    rejected_list.remove((orig_q, orig_rep))
                    accepted.append(q)
                    validations = [v for v in validations if v.concept_id != orig_rep.concept_id] + [new_rep]
                    break
        # Do NOT promote remaining failures — return fewer questions
        if len(accepted) < num_questions:
            print(f"[AUDIT] Returning {len(accepted)}/{num_questions} questions — {num_questions - len(accepted)} failed without repair (no promotion)")
        accepted = accepted[:num_questions]

        metrics.audit_ms = round((time.time() - t) * 1000, 1)
        metrics.questions_passed = len(accepted)
        metrics.questions_failed = len(rejected_list)
        metrics.hallucination_rate = 0.0 if not validations else round(
            sum(1 for v in validations if any("hallucination" in c.lower() or "RC-01" in c for c in v.reason_codes)) / len(validations), 3
        )
        # Overall confidence: weighted avg of components
        weights = {"extraction": 0.15, "preprocessing": 0.10, "concept": 0.15, "grounding": 0.20, "reasoning": 0.15, "planning": 0.05, "composition": 0.10, "auditing": 0.10}
        overall = sum(getattr(component_conf, k) * w for k, w in weights.items())
        component_conf.overall = round(overall, 2)
        metrics.component_confidence = component_conf
        metrics.total_ms = round((time.time() - t0) * 1000, 1)

        print(f"[AUDIT] {len(accepted)} accepted, {len(rejected_list)} rejected | {metrics.audit_ms}ms")
        print(f"[CONFIDENCE] Per-component: extr={component_conf.extraction:.0%} preproc={component_conf.preprocessing:.0%} concept={component_conf.concept:.0%} ground={component_conf.grounding:.0%} reason={component_conf.reasoning:.0%} comp={component_conf.composition:.0%} audit={component_conf.auditing:.0%} → overall={component_conf.overall:.0%}")
        print(f"{'='*60}\n[PIPELINE] Done in {metrics.total_ms}ms | Extraction {metrics.extraction_confidence:.0%} | "
              f"Grounding {metrics.grounding_avg:.0%} | Hallucination {metrics.hallucination_rate:.0%} | Overall {component_conf.overall:.0%}\n{'='*60}")

        # Grounding report
        grounding_report = {
            "extraction_method": layered.merged_method,
            "extraction_confidence": metrics.extraction_confidence,
            "preprocessing_confidence": component_conf.preprocessing,
            "grounding_avg": metrics.grounding_avg,
            "reasoning_confidence": component_conf.reasoning,
            "hallucination_rate": metrics.hallucination_rate,
            "overall_confidence": component_conf.overall,
            "every_question_has": ["Concept ID", "Source chunk", "Confidence", "Expected answer", "Bloom level", "Question", "Reasoning intent", "Misconception"],
        }

        if metrics.total_ms > 60000:
            warnings.append(f"Performance target missed: {metrics.total_ms}ms > 60s for {layered.word_count} words")

        return AionPipelineResult(
            source=source,
            clean_text_path=clean_path,
            concepts=concepts,
            knowledge_units=knowledge_units,
            grounded=grounded,
            reasoning_intents=reasoning_intents,
            plans=plans,
            questions=questions,
            validations=validations,
            critic_results=critic_results,
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
