# AION Stage G6 Test and Acceptance Report

**Test Suite Execution**: Complete Repository Regression & Stage G6 Acceptance Suite  
**Date of Run**: November 2026  
**Status**: **100% PASSED (727 / 727 Tests)**  
**Duration**: ~14.0s  
**Baseline Git Tag**: `stage-g5-accepted`  
**Active Branch**: `v3`  

---

## 1. Acceptance Test Suite Results Summary

| Test Suite / Area | Tests Executed | Passed | Failed | Pass Rate | Key Verification Focus |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `tests/test_stage_g6_academic_coherence.py` | 86 | 86 | 0 | **100%** | Full Stage G6 Academic Coherence Gates (Tracks G6-00 to G6-14, G6-C1) |
| `tests/test_generated_question_contracts.py` | 33 | 33 | 0 | **100%** | Exact 25-field shape, strict `SubPart` typing, non-eager projections, `from_dict`, direct constructors |
| `tests/test_stage_g2c_assessment_blueprint.py` | 16 | 16 | 0 | **100%** | 50M IA & 100M SEE Blueprint Synthesis, MarkingRubric, ExpectedResponse contracts |
| `tests/test_stage_g4_export_alignment.py` | 23 | 23 | 0 | **100%** | Subpart Bloom/CO table column rendering, multi-visual physical validation |
| `tests/test_stage_g1d_integration.py` | 13 | 13 | 0 | **100%** | 5-agent pipeline execution (Planning -> Writing -> Evaluation -> Refinement -> Checking) |
| Full Repository Regression Suite (`tests/`) | **727** | **727** | **0** | **100%** | Entire test suite across all stages (G1 to G6) |

---

## 2. Detailed Breakdown of Stage G6 Verification Tracks

### Track G6-00: Contract Foundations, Backward Compatibility & Direct Constructor Validation
- Verified clean imports without circular dependencies across all agent modules.
- Non-applicable gates do not produce false `INDETERMINATE` states.
- Rejection of vacuous passes when zero applicable gates run.
- Positional dataclass constructors preserved for all historical caller patterns.
- Direct constructor validation in `question_plan.py`: `ExpectedResponseContract`, `ConceptFootprint`, `VisualTaskContract`, `RubricCriterion`, and `SubPartPlan` constructors strictly reject invalid types, bools-as-ints, empty strings, and unknown kwargs identically to `from_dict`.

### Track G6-01: Evidence Grounding & Revision Invariants
- Verified topic-aware evidence retrieval with semantic filtering.
- Rejection of cross-module evidence leakage.
- Detected document revision hash changes, preventing stale evidence restoration during refinement.
- Fail-closed behavior on missing or ungrounded evidence blocks.
- In `planning_agent.py`, real production `DocumentArtifact` instances fail closed (`INSUFFICIENT_EVIDENCE`) when blocks are empty; synthetic fallback generation is strictly isolated to test `MockArtifact` / `is_synthetic` fixtures.

### Track G6-04 & G6-14: Branch-and-Bound Cognitive Optimization & Dual Accounting
- Exact **Branch-and-Bound** tree search allocating integer marks across Bloom levels (L1–L6) minimizing $L_1$ deviation from syllabus targets.
- Uses admissible lower bounding $bound = \max\left(\sum_{b: acc[b] > T[b]} (acc[b] - T[b]),\; \sum_b \max(0, T[b] - (acc[b] + r))\right)$.
- Deterministic, seed-dependent SHA-256 tie-break across globally optimal candidates: `SHA-256(f"assessment_id={id}|slots={canonical_slots}|bloom={','.join(p)}")`.
- Formally verified against an exhaustive brute-force oracle across all integer partitions.
- Evaluated both *printed paper distribution* and *all possible student attempt paths*.
- Attempt path enumeration verifying that worst-case choice options satisfy academic distribution bounds without cognitive collapse.

### Track G6-05 & G6-06: Sibling Progression & Cross-Module Deduplication
- Enforced non-decreasing Bloom order for sibling subparts (`(a) L2 -> (b) L3`).
- Permitted justified pedagogical inversions with logged telemetry (e.g. simulation L4 followed by reflection L2).
- Class-leading deduplication detecting concept collisions and topic repetition across choices and module boundaries.

### Track G6-08 & G6-09: Solution Completeness & Marking Rubric Conservation
- Strict criteria arithmetic conservation: $\sum \text{criterion marks} == \text{subpart marks}$.
- Model solutions provide complete, step-by-step derivations for computational questions.
- Rejection of invalid types (`str`, `float`, `bool`) in marking scheme criteria.

### Track G6-10 & G6-11: Authentic Visual Grounding & Dependency Gating
- Verified multi-condition visual gate requiring genuine figure dependency.
- Enforced cryptographic SHA-256 asset verification.
- Rejection of non-visual questions containing hallucinated spatial geometry references.

### Track G6-12 & G6-13: Choice Pair Parity & Blueprint Architecture
- Both alternatives in an `AlternativeGroup` maintain parity in:
  - Subpart count parity
  - Equivalent total marks (e.g. exactly 10M or 20M)
  - Cognitive Bloom level parity (no asymmetric easy/hard alternative trap)
  - Archetype symmetry

---

## 3. Whole-Paper End-to-End Pipeline & DOCX Export Verification

- **Script**: `scratch/test_e2e_pipeline_and_export.py`
- **Course**: `BCS303: Operating Systems & Systems Programming` (5 modules, 5 questions, 10 subparts)
- **Generated Artifact**: `STAGE_G6_SAMPLE_OUTPUT.docx` (39,297 bytes)
- **Pipeline Execution Summary**:
  - `PlanningAgent`: Generated 5 plans across 5 modules, all status `PLANNED`, partition `[5, 5]`, total marks 10 each (50M total).
  - `WritingAgent`: Generated 5 complete questions, 0 failures. Subparts strictly comply with Bloom opening verbs, semantic cognitive demand, and criteria-based marking schemes.
  - `CheckingAgent`: Full-paper gate returned `status=OK`, `resolved_slots=5`, `unresolved_slots=0`, `blocked_reasons=none`.
- **Verified Layout Elements**:
  1. **Institutional Header**: VTU / Institution Header, Department, Exam Type (IA / SEE).
  2. **Metadata Grid (Table 0)**: Course Title: *Operating Systems & Systems Programming*, Course Code: *BCS303*, Semester: *VI Semester*, Max Marks: *50*, Duration: *90 Minutes*.
  3. **Question Layout Table (Table 1)**:
     - Columns: `['Q.No', 'Part', 'Question Description', 'Marks', 'CO', 'RBT']`.
     - Q1: `(a)` 5M, CO1, L1; `(b)` 5M, CO1, L1 $\rightarrow 10\text{M}$
     - Q2: `(a)` 5M, CO2, L2; `(b)` 5M, CO2, L2 $\rightarrow 10\text{M}$
     - Q3: `(a)` 5M, CO3, L3; `(b)` 5M, CO3, L3 $\rightarrow 10\text{M}$
     - Q4: `(a)` 5M, CO4, L4; `(b)` 5M, CO4, L4 $\rightarrow 10\text{M}$
     - Q5: `(a)` 5M, CO5, L5; `(b)` 5M, CO5, L5 $\rightarrow 10\text{M}$
     - Whole-paper marks conservation: $\sum_{q=1}^5 \text{marks}(Q_q) = 50\text{ Marks}$ strictly conserved.

---

## 4. Protected Modules Audit

Command:
```bash
git diff --exit-code stage-g5-accepted -- core/generation/agents/legacy_adapter.py core/generation/agents/refinement_agent.py core/generation/agents/evaluation_agent.py core/generation/agents/deterministic_repairs.py v0_1/docx_export.py v0_1/
```
Result: **Exit Code 0 (Clean, 0 differences). All protected modules remain 100% untouched.**
