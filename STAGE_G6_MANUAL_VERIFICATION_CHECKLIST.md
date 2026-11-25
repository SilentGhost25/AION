# Stage G6 Manual Verification Checklist & Acceptance Guide

**Review Target**: AION Stage G6 Academic Coherence & Autonomous Generation System  
**Baseline Commit**: `b9e66d807b65ef4188b2271aa5bca1a9c832cbeb` (`stage-g5-accepted`)  
**Active Branch**: `v3`  
**Generated DOCX Output**: `STAGE_G6_SAMPLE_OUTPUT.docx`  
**Verification Status**: **PENDING REVIEWER SIGN-OFF**  

---

## Instructions for the Reviewer

Before approving Stage G6 and authorizing a git commit or tag, please review the five verification gates below and mark them upon completion.

---

### [Gate 1] Verify Protected Modules Untouched
Execute in PowerShell:
```powershell
git diff --exit-code stage-g5-accepted -- core/generation/agents/legacy_adapter.py core/generation/agents/refinement_agent.py core/generation/agents/evaluation_agent.py core/generation/agents/deterministic_repairs.py v0_1/docx_export.py v0_1/
```
- [ ] **Pending Sign-off**: Verify that command exits with code `0` and emits zero output. All protected legacy subsystems remain 100% frozen.

---

### [Gate 2] Verify Production Change Footprint
Execute in PowerShell:
```powershell
git status -s
```
- [ ] **Pending Sign-off**: Verify that only the 10 authorized production files under `core/generation/` are modified:
  1. `core/generation/assessment_config.py`
  2. `core/generation/agents/question_plan.py`
  3. `core/generation/agents/generated_question.py`
  4. `core/generation/agents/pipeline_bridge.py`
  5. `core/generation/agents/planning_agent.py`
  6. `core/generation/agents/local_evaluators.py`
  7. `core/generation/agents/prompts.py`
  8. `core/generation/agents/writing_agent.py`
  9. `core/generation/agents/overlap_detector.py`
  10. `core/generation/agents/checking_agent.py`

---

### [Gate 3] Verify Complete Regression Test Suite
Execute in PowerShell:
```powershell
python -m pytest tests/ -q
```
- [ ] **Pending Sign-off**: Verify that all 727 tests pass (100% pass rate, 0 failures, 0 skipped, 0 xfail) in ~14 seconds.

---

### [Gate 4] Inspect Exported Sample Question Paper (`STAGE_G6_SAMPLE_OUTPUT.docx`)
Open `STAGE_G6_SAMPLE_OUTPUT.docx` in Microsoft Word or any DOCX viewer:
- [ ] **Header Information**:
  - University/Institution title centered.
  - Course Title: *Operating Systems & Systems Programming*.
  - Course Code: *BCS303*.
  - Semester: *VI Semester*.
  - Maximum Marks: *50*.
  - Duration: *90 Minutes*.
- [ ] **Question Layout Table (Table 1)**:
  - Column headers: `Q.No | Part | Question Description | Marks | CO | RBT`.
  - All 5 questions (Q1 through Q5) present across all 5 syllabus modules.
  - Each question has two cleanly partitioned subquestions: `(a)` and `(b)`.
  - Subpart marks: each subpart is exactly `5` marks ($5 + 5 = 10\text{M}$ per question).
  - Total marks: $\sum_{q=1}^5 \text{marks}(Q_q) = 50\text{ Marks}$ strictly conserved.
- [ ] **Pedagogical Quality & Progression**:
  - Q1: `(a)` 5M, CO1, L1 (`State ...`); `(b)` 5M, CO1, L1 (`State ...`).
  - Q2: `(a)` 5M, CO2, L2 (`Describe ...`); `(b)` 5M, CO2, L2 (`Describe ...`).
  - Q3: `(a)` 5M, CO3, L3 (`Apply ...`); `(b)` 5M, CO3, L3 (`Apply ...`).
  - Q4: `(a)` 5M, CO4, L4 (`Analyze ...`); `(b)` 5M, CO4, L4 (`Analyze ...`).
  - Q5: `(a)` 5M, CO5, L5 (`Evaluate ...`); `(b)` 5M, CO5, L5 (`Evaluate ...`).
  - Clean formatting without leaked JSON brackets, Markdown artifacts, or internal prompt instructions.

---

### [Gate 5] Final Authorization Directive
Once you are satisfied with the manual review of the generated paper:
1. To stage and commit:
   ```powershell
   git add core/generation/
   git commit -m "feat(g6): Stage G6 Academic Coherence, Course-Configurable VTU Profiles & Evaluation Engine"
   ```
2. To create the official acceptance tag:
   ```powershell
   git tag stage-g6-accepted
   ```
*(Note: As per our instructions, no git commit or tag has been created by the assistant. The working tree remains in a clean, staged-ready state for your manual inspection.)*
