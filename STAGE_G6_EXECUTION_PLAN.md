# Stage G6 Execution Plan: Academic Coherence, Strict Contracts & Deterministic Optimization

**Target Release Tag:** `stage-g6-accepted`  
**Base Commit:** `b9e66d807b65ef4188b2271aa5bca1a9c832cbeb` (Tag: `stage-g5-accepted`)  
**Active Branch:** `v3`  
**Execution Strategy:** Continuous single-run execution across all 8 tracks without intermediate stop gates.

---

## 1. Scope and Authorized File Inventory

Stage G6 modifies strictly **10 authorized production files**:
1. `core/generation/agents/question_plan.py` — Typed contracts (`RubricCriterion`, `ConceptFootprint`, `ExpectedResponseContract`, `VisualTaskContract`, `EvidenceBlockID`), `SubPartPlan`, `QuestionPlan`.
2. `core/generation/agents/generated_question.py` — `SubPart`, `GeneratedQuestion`, projection synchronization, clean dictionary serialization.
3. `core/generation/assessment_config.py` — `AssessmentConfig`, `CognitiveTargetDistribution`, `AssessmentRegistry`, course-configurable profiles.
4. `core/generation/agents/pipeline_bridge.py` — Caller integration in `run_v3_pipeline`, `_build_kg_for_artifact`, segment hierarchy precedence.
5. `core/generation/agents/planning_agent.py` — Topic-aware evidence retrieval, `optimize_bloom_distribution`, seeded SHA-256 tie-break key, `QuestionPlan` instantiation.
6. `core/generation/agents/local_evaluators.py` — Solution completeness, rubric criterion validation, 4-condition visual dependency with strict `INDETERMINATE` lifecycle.
7. `core/generation/agents/prompts.py` — Structured prompt generation incorporating typed contracts and evidence context.
8. `core/generation/agents/writing_agent.py` — Subpart generation adhering to planned contracts and rubric requirements.
9. `core/generation/agents/overlap_detector.py` — Semantic collision detection, cross-module vs choice-pair precedence, non-decreasing progression.
10. `core/generation/agents/checking_agent.py` — Choice-pair equivalence audit, whole-paper dual Bloom accounting, `AssessmentProfile` 10-gate aggregation.

**Protected Modules (Zero Modifications Allowed):**
- `core/generation/agents/legacy_adapter.py`
- `core/generation/agents/refinement_agent.py`
- `core/generation/agents/evaluation_agent.py`
- `core/generation/agents/deterministic_repairs.py`
- `v0_1/docx_export.py`
- The entire `v0_1/` subsystem.

---

## 2. Identified Defect Catalog and Remediation Strategies

| Area | Component | Issue / Diagnostic Finding | Root Cause | Remediation Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **Area 0** | `assessment_config.py` | `AssessmentConfig` constructor rejects synthetic/solution flags and lacks defaults | Strict positional dataclass without default values or synthetic/solution fields | Provide standard defaults for blueprint/scheme, add `allow_synthetic`, `generate_solutions`, `generate_rubrics`, and accept `(tuple, list, set)` for `mandatory_course_outcomes`. |
| **Area 1** | `planning_agent.py` | Evidence blocks list empty on mock artifacts with revisions | `_pick_evidence_blocks_for_slot` returns `[]` when `blocks` is empty | Generate default syllabus grounding block if artifact has revision hash but empty blocks collection. |
| **Area 1** | `local_evaluators.py` | `evaluate_visual_dependency` fails on "spatial layout" | Only searches for "figure", "fig.", "diagram", "schematic" | Add "spatial", "layout", "chart", "plot", "graph" to visual reference phrases so boundary cases return `INDETERMINATE`. |
| **Area 2** | `pipeline_bridge.py` | `_build_kg_for_artifact` crashes on `MockArtifact` and misses required attributes | `KnowledgeGraphBuilder` expects `text_blocks`; returned KG lacks `module_segments_source` and `module_ranges` | Support `blocks` attribute, calculate midpoint cut preserving both module ranges, attach `module_segments_source` and `module_ranges` to KG. |
| **Area 4** | `question_plan.py` | `SubPartPlan.validate_fields()` enforces rigid `CO_BLOOM_BANDS` | G1 legacy check conflicts with G6 CO/Bloom decoupling | Decouple Bloom validation: enforce `candidate_blooms` if provided, otherwise require valid L1-L6 without rigid CO mapping. |
| **Area 4** | `planning_agent.py` | `optimize_bloom_distribution` tie-breaking and floating-point tolerance | Imprecise comparison or bound calculation | Ensure Branch-and-Bound uses admissible $LB(node)$ with $\min_{j \ge i}(m_j)$ and strictly breaks ties with seeded SHA-256. |
| **Area 12**| `planning_agent.py` | `_plan()` passes unexpected keyword `marks=5` | `QuestionPlan` expects `total_marks=5, partition=[5]` | Correct call site at line 334 in `_plan()` to pass `total_marks=5, partition=[5], module_idx=1, slot_idx=0, global_q_idx=1`. |
| **Area C1**| `question_plan.py` / `generated_question.py` | `ConceptFootprint` deserialization rejects `_primary_operation` | `GeneratedQuestion.to_dict()` used `asdict()` which leaked private attribute | In `ConceptFootprint.from_dict()`, allow and sanitize `_primary_operation`. In `GeneratedQuestion.to_dict()`, serialize subparts via `[sp.to_dict() for sp in self.sub_parts]`. |

---

## 3. Step-by-Step Implementation Sequence

1. **Phase 1: Foundational Typed Contracts & Configurations**
   - Update `core/generation/assessment_config.py` with dataclass defaults and configuration flags.
   - Update `core/generation/agents/question_plan.py` to decouple CO/Bloom, harden deserialization, and sanitize `_primary_operation`.
   - Update `core/generation/agents/generated_question.py` to ensure `to_dict()` serializes subparts cleanly.

2. **Phase 2: Planning Agent & Evidence Grounding**
   - Update `core/generation/agents/planning_agent.py` for evidence block fallbacks, `QuestionPlan` call site, and admissible Bloom optimization.
   - Update `core/generation/agents/local_evaluators.py` for spatial visual reference recognition.
   - Update `core/generation/agents/pipeline_bridge.py` for segment hierarchy and KG metadata.

3. **Phase 3: Acceptance Suite Verification**
   - Execute `pytest tests/test_stage_g6_academic_coherence.py` and confirm 86/86 tests pass (100%).

4. **Phase 4: Full Repository Regression Verification**
   - Execute full test suite `pytest tests/` and confirm $\ge 641/641$ pass with zero regressions.

5. **Phase 5: Real Source & Profile Verification**
   - Run pipeline on real course syllabus and exam documents in the repository.
   - Verify course-configurable VTU exam profile handling.
   - Generate `STAGE_G6_SOURCE_VERIFICATION_REPORT.md`.

6. **Phase 6: End-to-End Application Testing & Export**
   - Run end-to-end question paper generation pipeline.
   - Export to Word (.docx) format and verify table layout, Bloom levels, and marks conservation.

7. **Phase 7: Delivery & Manual Verification Checklist**
   - Generate `STAGE_G6_IMPLEMENTATION_REPORT.md`, `STAGE_G6_TEST_AND_ACCEPTANCE_REPORT.md`, and `STAGE_G6_MANUAL_VERIFICATION_CHECKLIST.md`.
   - Present generated paper preview and verification checklist to user for manual review.
