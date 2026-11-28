# AION Live Model Accuracy & Dynamic Rule Compliance — Implementation Report

**Branch:** `v4`  
**Tag baseline:** `fixed-and-beta-ready`  
**Date:** 2026-11-25  
**Status:** Implementation Complete — Pending Live Benchmark

---

## 1. Objective

Make the live Qwen model generate questions at the same accuracy level as the deterministic mock, while ensuring all academic constraints (Bloom verbs, marks, CO, partition) are enforced dynamically from the single active source of truth (`BLOOM_VERB_LEVEL_MAP`, `AssessmentRegistry`) with no stale fallbacks.

---

## 2. Authorized Files Modified

| File | Scope of Change |
|------|----------------|
| `core/generation/assessment_config.py` | Added `RuleSnapshot` dataclass + `capture_rule_snapshot()` function |
| `core/generation/agents/question_plan.py` | Extended `VisualTaskContract` with `prohibited_features` |
| `core/generation/agents/prompts.py` | Dynamic schema builder, evidence section, exemplar section, rule-first prompt header |
| `core/generation/agents/dependency_factory.py` | `AION_LLM_TEMPERATURE` env var forwarding via `LLMRequest` |
| `core/generation/agents/writing_agent.py` | Schema forwarding, `RuleSnapshot`-aware Bloom validation, surgical retry categorization |
| `core/generation/agents/local_evaluators.py` | OCR provider, answer-leakage detection, `verify_exported_docx_visuals` rewrite |
| `core/generation/agents/checking_agent.py` | Mandatory Stage B visual audit gate |
| `core/generation/agents/planning_agent.py` | Slot-based visual eligibility, `_derive_visual_contract()`, caption normalization |
| `aion_api.py` | `/api/export/docx` handler: manifest verification before `send_file()` |
| `tests/test_live_model_rule_compliance.py` | 14 new offline acceptance tests |
| `tests/test_stage_g6_visual_audit_gate.py` | Existing visual audit tests (no modifications) |

---

## 3. Protected Files — Confirmed Unmodified

The following files were inspected read-only and **not modified**:

- `v0_1/` (entire subsystem, including `llm_server.py`)
- `core/generation/agents/legacy_adapter.py`
- `core/generation/agents/refinement_agent.py`
- `core/generation/agents/evaluation_agent.py`
- `core/generation/agents/deterministic_repairs.py`
- `core/validation/bloom_validator.py`
- `requirements.txt`, `pyproject.toml`

---

## 4. Schema Forwarding Chain — Verified

The active generation chain is:
```
WritingAgent._generate_structured_subparts()
  → build_subpart_json_schema(sp_plan, rule_snapshot)   # dynamic enum from active BLOOM_VERB_LEVEL_MAP
  → RobustLLMCallerAdapter.call(prompt, schema=schema)   # passes schema + temperature
    → LLMRequest(schema=schema, temperature=temperature)
      → RobustLLMCaller.call(req)                        # payload["guided_json"] = req.schema
        → vLLM endpoint                                   # structured JSON decoding enforced
```

`v0_1/llm_server.py` was inspected read-only. It is a legacy path **not** used in the production chain. No modification was needed or made.

---

## 5. Temperature Control

- `RobustLLMCallerAdapter` now reads `AION_LLM_TEMPERATURE` env var (default `0.1`, clamped `[0.0, 1.0]`).
- Passed explicitly on every `LLMRequest`, preventing stale caller-level defaults.
- Low temperature (0.1) biases the model toward the constrained JSON schema verbs and structure.

---

## 6. Rule Snapshot — Single Source of Truth

`capture_rule_snapshot(assessment_identity=None, config=None)` builds an immutable `RuleSnapshot` from:
- Active `BLOOM_VERB_LEVEL_MAP` (all levels, sorted, cleaned)
- Active `AssessmentConfig` (marks, partitions, max_subparts, cognitive target)

The snapshot fingerprint changes immediately when either source changes — no restart required.

**Fail-closed guarantees:**
- Empty BLOOM_VERB_LEVEL_MAP → `ValueError`
- Empty verb set for any level → `ValueError`
- Unknown `assessment_identity` → graceful `None` config (no crash), falls back to verb-only snapshot

---

## 7. Surgical Retry System

Each validation failure is categorized and surfaced in the retry prompt:

| Category | Trigger |
|----------|---------|
| `CALLER_TIMEOUT` | LLM call timeout |
| `CALLER_ERROR` | Unexpected caller exception |
| `INVALID_JSON` | Non-parseable response |
| `MARKING_SCHEME_TYPE` | Non-list, non-int marks, bool marks |
| `MARKING_SCHEME_SUM` | Criteria sum ≠ subpart marks |
| `RAW_ARTIFACT` | Template fragments, preamble leakage |
| `SIBLING_CHAINING` | References prior subpart |
| `MULTIPLE_DIRECTIVES` | Compound task (two verbs) |
| `SOURCE_TAG_LEAK` | Figure N.N / Chapter N references |
| `COGNITIVE_DEMAND_MISMATCH` | Semantic depth below/above Bloom level |
| `BLOOM_VERB_MISMATCH` | Opening verb not in active approved set |
| `MULTIPLE_TOPICS` | Bundles sibling module topics |

Retry prompt format:
```
[CATEGORY] {error_detail} Keep marks=X, CO=Y, Bloom=Z, topic='T' unchanged.
```

Immutable plan fields (marks, CO, Bloom, topic, module) are **never altered by retry**.

---

## 8. Test Results

| Suite | Before | After |
|-------|--------|-------|
| `tests/` total | 748 passed / 7 failed | 755 passed / 0 failed |
| Protected diff | PASS | PASS |

---

## 9. Known Live Benchmark Gap

The 90% first-attempt validity target is an **offline design objective**, not a confirmed live measurement.  
See `LIVE_MODEL_BENCHMARK_PLAN.md` for the benchmark procedure to close this gap.
