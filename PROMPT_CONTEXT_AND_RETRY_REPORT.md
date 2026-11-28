# AION Prompt Context & Retry Report

**Branch:** `v4`  
**Date:** 2026-11-25

---

## 1. Evidence Section (Context Retention)

### Problem
The old evidence section used a fixed 1200-character cap, which truncated equations and definitions — the most important content for academic question generation. The cap was arbitrary and not adjustable at runtime.

### Solution: Query-Focused Evidence Selection

`build_evidence_section(blocks, topic, char_budget)` now:

1. **Scores blocks** by topic-token overlap — blocks containing words from the topic string score higher
2. **Prioritises equations** — blocks containing `=`, `Σ`, `∫`, numeric values, or defined terms are always retained
3. **Respects `AION_EVIDENCE_CHAR_BUDGET`** env var (default: no arbitrary cap; uses all available relevant blocks)
4. **Strips source metadata** before including blocks (via `_sanitize_evidence_source_metadata`)

### Source Metadata Sanitization

The following patterns are stripped from evidence text before including in the prompt:
- `[blk_m1_2]` block ID tags
- `Figure N.N:` / `Fig. N.N:` prefixes
- `Chapter N:` / `Section N.N:` labels

**Rule:** Only non-semantic metadata is stripped. Equation text, definitions, and factual content are preserved exactly.

---

## 2. Rule-First Prompt Header

`build_critical_rules_header(sp_plan, rule_snapshot)` prepends to every subpart prompt:

```
[CRITICAL CONSTRAINTS — READ FIRST]
Marks: X (FIXED — do not change)
Bloom Level: LN (FIXED — allowed verbs: verb1, verb2, ...)
Course Outcome: CON (FIXED)
Topic: T (FIXED)
...
```

The header uses the **active** verb list from the current `RuleSnapshot`, not a static default.

---

## 3. Dynamic Exemplar Section

`build_dynamic_exemplar_section(sp_plan, rule_snapshot)` injects:

- **Compliant example:** A correctly structured JSON using an active verb, correct marks, correct CO
- **Forbidden example:** Explicitly shows what NOT to do (wrong verb, wrong marks, multiple directives)

Both examples are derived from the active rule state at generation time.

---

## 4. Surgical Retry Feedback

When an attempt fails validation, the retry prompt includes:

```
[{CATEGORY}] {specific_error_detail} Keep marks=X, CO=Y, Bloom=Z, topic='T' unchanged.
```

**Key properties:**
- The category name is explicit and machine-readable (`MULTIPLE_DIRECTIVES`, `BLOOM_VERB_MISMATCH`, etc.)
- Immutable fields are restated in every retry prompt — the model cannot "forget" them
- The retry prompt is always the same structure regardless of category — the model builds a consistent expectation

---

## 5. Context Window Management

The prompts are structured as a "Rule-Evidence-Task-Schema sandwich":

```
[CRITICAL CONSTRAINTS]   ← always first
[EVIDENCE]               ← topic-filtered, metadata-stripped
[TASK]                   ← Bloom verb, marks, CO
[JSON SCHEMA]            ← dynamic enum from active rules
[EXAMPLES]               ← compliant + forbidden
[VALIDATION FEEDBACK]    ← surgical retry message (attempt > 1 only)
```

This ordering ensures the model's attention is anchored on constraints before seeing evidence or examples.

---

## 6. Test Coverage

| Test | Scenario |
|------|----------|
| `test_11_12` | Evidence retains equations; source tags are stripped |
| `test_13` | Surgical retry carries `[MULTIPLE_DIRECTIVES]` category; immutable fields unchanged |
| `test_07` | Schema forwarded end-to-end to RobustLLMCaller |
