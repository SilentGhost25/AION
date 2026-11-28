# AION Dynamic Rule Compliance Report

**Branch:** `v4`  
**Date:** 2026-11-25

---

## 1. Problem Statement

The live Qwen model was generating questions that ignored the active Bloom verb constraints, used stale verb lists, and treated academic rules as suggestions rather than hard constraints. When rules changed in `BLOOM_VERB_LEVEL_MAP` or `AssessmentConfig`, the model continued to use old defaults.

---

## 2. Root Cause

| Component | Old Behaviour | Problem |
|-----------|--------------|---------|
| `prompts.py` | Static `BLOOM_VERBS` dict in prompt | Stale — not refreshed on rule changes |
| `writing_agent.py` | No schema forwarding to LLM | Model guessed structure freely |
| `dependency_factory.py` | Hardcoded temperature in caller | No runtime control |
| `checking_agent.py` | Bloom validation optional | Visual subparts could bypass audit |

---

## 3. Solution: RuleSnapshot

`RuleSnapshot` is an immutable value object capturing the state of all active rules at generation time:

```python
@dataclass(frozen=True)
class RuleSnapshot:
    bloom_verb_map: Dict[str, Tuple[str, ...]]   # active verbs per Bloom level
    fingerprint: str                              # SHA-256(16) of verb map + config
    assessment_identity: Optional[tuple]
    total_marks_per_question: Optional[int]
    permitted_partitions: Tuple[tuple, ...]
    max_subparts: Optional[int]
    cognitive_target: Optional[Dict]
```

**Key properties:**
- Created fresh on every generation call via `capture_rule_snapshot()`
- Fingerprint changes immediately when BLOOM_VERB_LEVEL_MAP or AssessmentConfig changes
- Fails closed: empty verb sets raise `ValueError` before any LLM call

---

## 4. Dynamic JSON Schema

`build_subpart_json_schema(sp_plan, rule_snapshot)` produces a strict JSON Schema that:

1. Enumerates `bloom_verb_used` only from the **active** verb set for the subpart's Bloom level
2. Sets `marks` bounds to `[1, sp_plan.marks]` with `maximum = sp_plan.marks`
3. Adds `image_description` and `visual_features_identified` as required fields when `visual_required=True`
4. Uses `additionalProperties: false` to prevent extra fields

This schema is forwarded to the vLLM `guided_json` parameter, forcing the model to produce responses that conform to current rules at the token level.

---

## 5. Rule Mutation Tests

| Test | Scenario | Outcome |
|------|----------|---------|
| `test_01_02_03` | Mutate L2 verbs → new schema has new enum | PASS |
| `test_04` | Change AssessmentConfig → new fingerprint | PASS |
| `test_05_06` | Mutate verbs → exemplar updated; empty set → ValueError | PASS |
| `test_07` | Schema forwarded to RobustLLMCaller | PASS |

---

## 6. Hard-Coded Rule Override Guarantee

Rules are **not stored in the model context** — they are compiled fresh on each generation request from:
- `core/validation/bloom_validator.py` → `BLOOM_VERB_LEVEL_MAP`
- `core/generation/assessment_config.py` → `AssessmentRegistry`

If a rule changes (e.g., a new verb added to L4, or max marks per question updated), the next generation request automatically uses the new rule. There is no cache, fallback default, or stale copy.

---

## 7. Fail-Closed Behaviour

| Condition | Outcome |
|-----------|---------|
| Unknown Bloom level in schema builder | `ValueError` before LLM call |
| Empty verb set for a level | `ValueError` before LLM call |
| Unregistered assessment_identity | Snapshot built without config (verb-only); generation continues |
| LLM returns verb not in active set | Rejected, retry with `[BLOOM_VERB_MISMATCH]` feedback |
| LLM returns marks sum ≠ subpart marks | Rejected, retry with `[MARKING_SCHEME_SUM]` feedback |
