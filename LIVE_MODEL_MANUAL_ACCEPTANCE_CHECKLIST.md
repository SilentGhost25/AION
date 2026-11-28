# AION Live Model & Visual Integrity — Manual Acceptance Checklist

**Branch:** `v4`  
**Date:** 2026-11-25

---

## 1. Automated Pre-Flight (Completed)

- [x] Full pytest suite passes (`755 passed, 0 failed`)
- [x] Protected files untouched (`git diff --exit-code fixed-and-beta-ready -- ...` exit code 0)
- [x] Dynamic `RuleSnapshot` & `build_subpart_json_schema()` verified
- [x] `AION_LLM_TEMPERATURE` forwarding verified
- [x] Stage A / Stage B visual audit & `/api/export/docx` manifest gate verified offline

---

## 2. Live Server Manual Checks (Pending User Execution)

- [ ] Start vLLM server and verify `guided_json` / `response_format` is accepted without errors
- [ ] Run `scratch/test_e2e_dbms_visual_server.py` against live server
- [ ] Generate a 10-question IA paper and inspect:
  - [ ] Opening Bloom verbs match planned `L1`–`L5` levels
  - [ ] Marking scheme sums match exact subpart marks
  - [ ] Equations and numerical values from source PDF are preserved in questions
  - [ ] No `[blk_...]`, `Figure X.Y:`, or `Chapter N:` tags leak into question text
- [ ] Generate a paper with visual subparts (`(a)` and `(b)`) and inspect:
  - [ ] Exported DOCX embeds the exact approved image in the matching row
  - [ ] Attempting `/api/export/docx` with an unapproved or swapped image returns HTTP `422`
- [ ] Temporarily mutate `BLOOM_VERB_LEVEL_MAP` in a test script and verify the next generation uses the new verb set immediately
- [ ] Confirm live benchmark metrics meet `LIVE_MODEL_BENCHMARK_PLAN.md` exit criteria before any commit/tag
