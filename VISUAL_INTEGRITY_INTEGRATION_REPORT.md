# AION Visual Integrity Integration Report

**Branch:** `v4`  
**Date:** 2026-11-25  
**Scope:** Stage G6.1 — Visual Integrity & Answer-Leakage Hardening

---

## 1. Architecture Overview

Visual integrity enforcement operates as a **two-stage gate**:

```
Stage A (PlanningAgent._resolve_visual())
  → Pre-selection screening before image is assigned
  → Checks: file existence, decodability, SHA-256, module provenance, topic alignment
  → Derives: required_features, prohibited_features from topic/concept
  → Decision: PASS / INDETERMINATE (image not assigned if INDETERMINATE)

Stage B (CheckingAgent.run())
  → Post-generation audit with the actual generated question
  → Checks: answer leakage, prohibited features, OCR text vs expected answer
  → Decision: PASS blocks / FAIL blocks / INDETERMINATE → warning
  → Mandatory for ANY subpart with visual_required, visual_task, figure_id, or image_path
```

---

## 2. VisualTaskContract Extension

`VisualTaskContract` now carries `prohibited_features: Tuple[str, ...]` alongside `required_features`.

The prohibited list is derived **dynamically** from topic/concept at planning time:

| Topic Pattern | Required Features | Prohibited Features |
|---------------|------------------|---------------------|
| Circuit/Resistance | `circuit_topology`, `resistor_values` | `final_equivalent_resistance`, `worked_calculation` |
| Graph/Tree | `node_labels`, `edge_weights` | `shortest_path_highlighted`, `traversal_result` |
| Memory/Cache | `address_layout`, `block_boundaries` | `cache_hit_result`, `final_access_pattern` |
| Generic | _(empty)_ | `answer_highlighted`, `final_result_annotated` |

---

## 3. Answer-Leakage Detection

`audit_visual_asset()` performs semantic-role-aware numeric comparison:

- **Input-role patterns** (e.g., `R1 = 15Ω`, `V_in = 5V`) → **not flagged** as leakage
- **Output-role patterns** (e.g., `R_eq = 15Ω`, `V_out = 5V`) → **flagged** as potential leakage

SymPy equivalence is used to detect mathematically equivalent expressions (e.g., `1/2` vs `0.5`).

**OCR Backend Resolution Order:**
1. `rapidocr-onnxruntime` (preferred — CPU-only, no Tesseract dependency)
2. `pytesseract` (fallback if rapidocr unavailable)
3. `INDETERMINATE` if neither is available (never silently passes)

---

## 4. DOCX Export Verification

`verify_exported_docx_visuals(docx_buffer, approved_manifest)`:

1. Finds the question table by header signature (not hardcoded index)
2. Walks rows, keying by `{Q#}_{subpart_label}` (e.g., `Q1_a`)
3. Extracts embedded image SHA-256 from XML blip relationships per row
4. Verifies: SHA-256 match, caption presence, no unapproved images in visual rows
5. Returns `(True, [])` on full pass, or `(False, [error_list])` on any failure

**Export is blocked (HTTP 422)** if any verification step fails.

---

## 5. API Export Gate

`/api/export/docx` handler (authorized narrow scope):

```python
# Server-side manifest (never trusts client-supplied 'approved' flags)
manifest_provider = app.config.get("TRUSTED_VISUAL_MANIFEST")
if manifest_provider is not None:
    manifest = list(manifest_provider(body) if callable(manifest_provider) else manifest_provider)
else:
    manifest = []  # built by auditing each visual subpart server-side

ok_manifest, errors = verify_exported_docx_visuals(docx_buffer, manifest)
if not ok_manifest:
    return jsonify({"error": "Export blocked: ...", "details": errors}), 422
```

---

## 6. Test Coverage

| Test | Scenario |
|------|----------|
| `test_14` (compliance suite) | 422 without manifest; 200 with approved manifest |
| `test_api_export_docx_route_enforces_visual_manifest` (G6 suite) | Same gate from visual audit gate suite |
| `test_stage_g6_visual_audit_gate.py` (all) | Stage A/B audit logic, OCR, leakage detection |
