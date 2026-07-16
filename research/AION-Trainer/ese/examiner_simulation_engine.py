# AION-Trainer/ese/examiner_simulation_engine.py
"""
Examiner Simulation Engine (ESE) Orchestrator.
Coordinates the end-to-end 6-step workflow:
1. Paper Blueprinting
2. Examiner Planning (Intent mapping + Answer blueprinted notes)
3. Question Discovery (Candidate drafting)
4. Question Ranking (Deterministic criteria sorting)
5. Language Realization (Neural smoothing)
6. Hard Rules Validation (Grammar + VTU standards)
7. Chief Examiner verification (Paper-level gate)
"""

from __future__ import annotations

import uuid
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any

from ese.exam_blueprint import ExamBlueprint, ExamBlueprintBuilder, QuestionSlot
from ese.question_planner import QuestionPlanner, PlannerOutput
from ese.question_discoverer import QuestionDiscoverer, QuestionCandidate
from ese.question_ranker import QuestionRanker, RankingScore
from ese.language_realizer import LanguageRealizer
from ese.grammar_validator import GrammarValidator
from ese.vtu_validator import VTUValidator
from ese.chief_examiner import ChiefExaminer, ChiefExaminerReport
from ese.question_metadata import QuestionMetadata, CandidateRecord

logger = logging.getLogger("aion.ese.engine")


class ExaminerSimulationEngine:
    def __init__(
        self,
        concept_store,
        reference_library=None,
        llm_client=None,
    ):
        self.concept_store = concept_store
        self.reference_library = reference_library
        self.llm = llm_client

        # Initialize subcomponents
        self.blueprint_builder = ExamBlueprintBuilder()
        self.planner = QuestionPlanner(concept_store, reference_library)
        self.discoverer = QuestionDiscoverer(llm_client)
        self.ranker = QuestionRanker()
        self.realizer = LanguageRealizer(llm_client)
        self.grammar_validator = GrammarValidator()
        self.vtu_validator = VTUValidator()
        self.chief_examiner = ChiefExaminer()

    def generate_paper(
        self,
        subject_code: str,
        subject_name: str,
        semester: int,
        previously_asked: List[str] = None,
        include_optional: bool = True,
        max_repair_loops: int = 2,
    ) -> Tuple[ExamBlueprint, ChiefExaminerReport, Dict[str, QuestionMetadata]]:
        logger.info(f"[ESE] Starting question paper generation for {subject_code}")
        previously_asked = previously_asked or []

        # Step 1: Paper-Level Blueprinting
        blueprint = self.blueprint_builder.build(
            subject_code=subject_code,
            subject_name=subject_name,
            semester=semester,
            concept_store=self.concept_store,
            include_optional=include_optional,
        )

        metadata_store: Dict[str, QuestionMetadata] = {}

        # Generation loop
        self._populate_blueprint_slots(blueprint, metadata_store, previously_asked)

        # Step 7: Chief Examiner Quality Gate
        report = self.chief_examiner.evaluate_paper(blueprint)

        # Autonomous Repair Loop
        repair_loop = 0
        while not report.passed and report.slots_to_regenerate and repair_loop < max_repair_loops:
            repair_loop += 1
            logger.info(
                f"[ESE] Chief Examiner flagged slots for regeneration: {report.slots_to_regenerate}. "
                f"Attempting repair loop {repair_loop}/{max_repair_loops}."
            )
            
            # Record currently generated texts to avoid repeating
            current_texts = [s.question_text for s in blueprint.slots if s.question_text]
            combined_history = previously_asked + current_texts

            for slot_id in report.slots_to_regenerate:
                slot = next((s for s in blueprint.slots if s.slot_id == slot_id), None)
                if slot:
                    # Clear current choice
                    slot.filled = False
                    slot.question_text = ""
                    
                    # Rerun slot population with updated history
                    self._populate_single_slot(slot, metadata_store, combined_history)
            
            # Re-evaluate paper
            report = self.chief_examiner.evaluate_paper(blueprint)

        return blueprint, report, metadata_store

    def _populate_blueprint_slots(
        self,
        blueprint: ExamBlueprint,
        metadata_store: Dict[str, QuestionMetadata],
        previously_asked: List[str],
    ):
        # Step 2: Examiner Planning
        planned_outputs = self.planner.plan_all(blueprint, previously_asked)
        plan_map = {out.slot.slot_id: out for out in planned_outputs}

        for slot in blueprint.slots:
            plan_out = plan_map.get(slot.slot_id)
            if not plan_out:
                # Fallback if planning failed
                logger.warning(f"[ESE] No plan found for slot {slot.slot_id}")
                continue

            self._populate_from_plan(slot, plan_out, metadata_store, previously_asked)

    def _populate_single_slot(
        self,
        slot: QuestionSlot,
        metadata_store: Dict[str, QuestionMetadata],
        previously_asked: List[str],
    ):
        plan_out = self.planner.plan_slot(slot, previously_asked)
        if plan_out:
            self._populate_from_plan(slot, plan_out, metadata_store, previously_asked)

    def _populate_from_plan(
        self,
        slot: QuestionSlot,
        plan_out: PlannerOutput,
        metadata_store: Dict[str, QuestionMetadata],
        previously_asked: List[str],
    ):
        # Step 3: Question Discovery
        candidates = self.discoverer.discover(plan_out.blueprint, plan_out.intent)

        # Step 4: Question Ranking
        best_candidate, rank_score = None, None
        ranked_scores = self.ranker.rank(
            candidates, plan_out.blueprint, plan_out.intent, previously_asked
        )

        for rs in ranked_scores:
            if not rs.disqualified:
                best_candidate = rs.candidate
                rank_score = rs
                break

        if not best_candidate and candidates:
            # Fallback to first candidate if all are disqualified
            best_candidate = candidates[0]
            rank_score = next((rs for rs in ranked_scores if rs.candidate == best_candidate), None)

        selected_text = best_candidate.text if best_candidate else f"Explain the concept of {slot.concept_name}."

        # Step 5: Language Realization
        realized_text = self.realizer.realize(selected_text, slot.bloom_level, slot.marks)

        # Step 6: Hard Rules Validation
        grammar_issues = self.grammar_validator.validate(realized_text)
        vtu_issues = self.vtu_validator.validate(
            realized_text, slot.bloom_level, slot.marks, plan_out.blueprint.diagram_required
        )

        # Build QuestionMetadata audit log
        meta_id = str(uuid.uuid4())[:8]
        cand_records = [
            CandidateRecord(
                text=c.text,
                source=c.source,
                scores={
                    "bloom_alignment": rs.bloom_alignment,
                    "component_coverage": rs.component_coverage,
                    "structural_quality": rs.structural_quality,
                    "novelty": rs.novelty,
                    "vtu_style": rs.vtu_style,
                    "overall": rs.overall
                } if (rs := next((r for r in ranked_scores if r.candidate == c), None)) else {},
                disqualified=rs.disqualified if rs else False,
                disqualification_reason=rs.disqualification_reason if rs else ""
            )
            for c in candidates
        ]

        from dataclasses import asdict
        metadata = QuestionMetadata(
            metadata_id=meta_id,
            slot_id=slot.slot_id,
            concept_id=slot.concept_id,
            concept_name=slot.concept_name,
            bloom_level=slot.bloom_level,
            marks=slot.marks,
            question_type=slot.question_type,
            planner_intent=asdict(plan_out.intent),
            candidates=cand_records,
            selected_text=selected_text,
            realized_text=realized_text,
            grammar_issues=[issue.__dict__ for issue in grammar_issues],
            vtu_issues=[issue.__dict__ for issue in vtu_issues],
            status="flagged" if (any(i.severity == "error" for i in grammar_issues) or any(i.severity == "error" for i in vtu_issues)) else "draft",
        )

        metadata_store[slot.slot_id] = metadata

        # Update slot references
        slot.question_text = realized_text
        slot.filled = True
        slot.question_metadata_id = meta_id
