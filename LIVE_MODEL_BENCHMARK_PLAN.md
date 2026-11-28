# AION Live Model Benchmark Plan

**Branch:** `v4`  
**Date:** 2026-11-25  
**Target:** >= 90% first-attempt schema & Bloom validity on live Qwen/vLLM

---

## 1. Purpose

Offline unit tests (755 passed) verify schema forwarding, rule snapshots, surgical retry formatting, and visual/export gates with mocks. They do **not** measure live Qwen model compliance. This plan defines the live benchmark required before declaring production readiness.

---

## 2. Benchmark Environment

- **Server:** Live vLLM server (`v0_1/llm_server.py` or configured endpoint)
- **Temperature:** `AION_LLM_TEMPERATURE=0.1` (default)
- **Evidence Budget:** `AION_EVIDENCE_CHAR_BUDGET` unset (topic-filtered full retention)
- **OCR Backend:** `rapidocr-onnxruntime` (CPU)

---

## 3. Test Matrix

| Run | Subject / Domain | Exam Type | Questions | Focus |
|-----|------------------|-----------|-----------|-------|
| B1 | Operating Systems | CIE_IAT | 10 (mixed 10M, 6+4M) | Bloom L1–L4, text-only |
| B2 | Basic Electrical Engineering | CIE_IAT | 10 (with circuits/figures) | VLM + Stage A/B visual audit |
| B3 | Data Structures & Algorithms | SEE | 10 (8+6+6M) | Multi-part independence, CO3–CO5 |
| B4 | Rule Mutation Smoke | CIE_IAT | 4 | Mutate `BLOOM_VERB_LEVEL_MAP` mid-process; verify new verbs used |

---

## 4. Metrics to Record

1. **First-attempt validity rate:** % of subparts passing all validators on attempt 1 (target: >= 90%)
2. **Post-retry validity rate:** % of subparts passing within `max_attempts` (target: >= 98%)
3. **Failure category distribution:** Counts by `INVALID_JSON`, `BLOOM_VERB_MISMATCH`, `MULTIPLE_DIRECTIVES`, `MARKING_SCHEME_SUM`, `COGNITIVE_DEMAND_MISMATCH`, etc.
4. **Visual audit outcomes:** Counts of `PASS`, `FAIL`, `INDETERMINATE` in Stage A and Stage B
5. **DOCX export verification:** 100% of exported papers pass `verify_exported_docx_visuals()`

---

## 5. Exit Criteria

- First-attempt validity >= 90% across B1–B3
- Zero `BLOOM_VERB_MISMATCH` after `guided_json` enum enforcement
- Zero answer-leakage false negatives in B2
- B4 confirms immediate rule-mutation compliance without server restart
