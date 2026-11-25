# Stage G6.1 Goal & Implementation Plan: Visual Asset Correctness & Answer-Leakage Gate

## Goal
Introduce a mandatory, two-mechanism **Visual Asset Correctness & Answer-Leakage Gate** as a sub-gate inside AION's existing Stage G6 visual validation architecture. Every visual asset must be independently verified for:
1. **Source Authenticity**: Originates from the registered syllabus/study document, revision, module, and page, with verified SHA-256 digest.
2. **Correct Image Identification**: Contains all `VisualTaskContract.required_features` and aligns with the intended question, topic, and `ExpectedResponseContract` (not merely caption/filename overlap).
3. **Answer-Leakage Prevention**: Contains no final numerical answer, worked calculation, intermediate solution steps, answer annotations, highlighted MCQ options, or solved diagrams, while permitting legitimate input givens, unsolved topologies, and raw graph data.
4. **Fail-Closed Escalation**: `PASS` (accept), `FAIL` (reject), `INDETERMINATE` (human review; never promoted to `PASS` on valid hash alone).
5. **Double-Audit Protection**: Audited **before generation** and **after DOCX export** against the approved asset manifest (without modifying frozen `v0_1/` modules).

---

## Implementation Plan

### Step 1: `VisualAssetAudit` Contract (`core/generation/agents/question_plan.py`)
- Add `VisualAssetAudit` dataclass with fields:
  - `figure_id: str`
  - `source_document_id: str`
  - `source_revision: str`
  - `image_sha256: str`
  - `module_id: str`
  - `page_number: int`
  - `required_features_verified: bool`
  - `question_alignment_verified: bool`
  - `detected_text: List[str]`
  - `detected_values: List[str]`
  - `detected_equations: List[str]`
  - `detected_annotations: List[str]`
  - `answer_leakage: str` (`"NONE" | "POSSIBLE" | "CONFIRMED"`)
  - `leakage_evidence: List[str]`
  - `decision: str` (`"PASS" | "FAIL" | "INDETERMINATE"`)
- Enforce strict type validation in `__init__` and `from_dict`.

### Step 2: Two-Mechanism Visual Verification & Post-Export DOCX Audit (`core/generation/agents/local_evaluators.py`)
- **Mechanism A (Deterministic OCR & Content Checks)**:
  - Extract text, numeric values, mathematical expressions, labels/units, and annotations (via RapidOCR if available, injected OCR/content payload, or image inspection).
  - Verify required features from `VisualTaskContract` and `ExpectedResponseContract`.
  - Detect answer-leakage patterns (final result, worked steps, `Ans:`/`Solution:`, highlighted MCQ markers, solved constructions) while distinguishing allowed input givens.
  - Detect unreadable/corrupted labels (`INDETERMINATE`).
- **Mechanism B (Independent Visual Reasoning)**:
  - Invoke an independent `vlm_verifier` callable (distinct from generation `VLMCaller`) that reports depiction, required features, topic/module match, completed solution, leakage, and concern regions.
  - Reconcile Mechanism A + Mechanism B: conflicts or uncertainty yield `INDETERMINATE`; mismatches or confirmed leakage yield `FAIL`.
- **Post-Export DOCX Manifest Verification (`verify_exported_docx_visuals`)**:
  - Inspect exported `.docx` embedded image blobs (`inline_shapes` / `related_parts`), compute SHA-256 digests, and verify exact match with approved `VisualAssetAudit` manifest entries.

### Step 3: Mandatory Sub-Gate Integration (`core/generation/agents/checking_agent.py` & `writing_agent.py`)
- Wire `audit_visual_asset` into `CheckingAgent`'s `visual_dependency_gate` when visual assets/audits are present:
  - `FAIL` -> `visual_dependency_gate = "FAIL"` (blocks paper).
  - `INDETERMINATE` -> `visual_dependency_gate = "INDETERMINATE"` (never promoted to `PASS`).
- Enforce pre-generation visual audit check in `WritingAgent` when `visual_audit` or `vlm_verifier` is provided on `SubPartPlan` / context.

### Step 4: Critical Visual Acceptance Suite (`tests/test_stage_g6_visual_audit_gate.py`)
- Implement all 12 critical scenarios + multi-distractor plausible-incorrect-image tests + pre-generation & post-DOCX-export double-audit tests.
- Run full test suite (`pytest tests/ -q`) and verify zero diff on all protected frozen modules.
