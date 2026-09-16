# AION v1.1.0 Production Baseline & Session Handoff (2026-09-13)

---

## 1. Release Baseline: `v1.1.0`

* **Tag**: `v1.1.0`
* **Branch**: `mod_eval`
* **Summary**: Hardened, production-verified pipeline on top of golden commit `6e3631a` (`v1.0.0`). Zero unverified architectural rewrites.
* **Rollback Guarantee**: `v1.0.0` (`6e3631a`) remains preserved in git history if a clean rollback is ever required.

---

## 2. Confirmed & Tested in Production (`v0_1/`, `core/`, `aion_api.py`)

1. **Module-Level Concurrency Restructuring**: ThreadPool-isolated `SlotOrchestrator` execution with dynamic port allocation (`start_server.ps1` / `start_server.sh`).
2. **Cross-Module Deduplication Registry**: `shared_generated_texts` thread-safe registry across per-module orchestrators + whole-paper Jaccard audit pass ($\ge 0.55$) in `v0_1/main.py`.
3. **Bloom Verb & Operation Synchronization**: Unified `BLOOM_VERB_LEVEL_MAP` across critic, linter, and difficulty manager (`b353a42f`, `f6676e91`).
4. **Fail-Closed Subject Archetype Registry**: Whitelist preventing cross-domain keyword contamination (e.g. DBMS vs Circuits).
5. **RapidOCR Word-Count Gating & Extraction Caching**: Word-count threshold (`OCR_WORD_COUNT_THRESHOLD=100`) and persistent disk caching (`.aion_cache/`), eliminating 90-second OCR freezes.
6. **Sparse Module-Index Fix in `v0_1/segmenter.py`**: Correct module alignment in segmentation without off-by-one or missing-module bleed.
7. **Topic Neighborhood Chunking (`Fix a/b`)**: `c.depth != 'EXTERNAL'` exclusion and 3-part related chunk distribution in `v0_1/chunk_image_mapper.py`.
8. **SSE Payload Normalization & Loop-Scope Fix**: Clean server-sent event streaming in `aion_api.py`.
9. **Multi-File Upload Synthesis with Loud Validation**: Explicit error reporting instead of silent file-dropping.
10. **Bloom-Verb Retry Oscillation Fix**: Auto-healer repair loop wired to prevent infinite retry bounce on verb mismatch.
11. **Marks-Partition & Bloom-Metadata Sync**: Automatic healing sync between mark splits and Bloom taxonomy tags.
12. **Phrasing Diversity via Archetype Rotation**: Question-type-aware rotation preventing repetitive opening clauses.
13. **Shadowed Duplicate Score Removal**: Removed duplicate `calculate_retrieval_score` implementation in chunk mapper.
14. **DOCX Export Module-Grouping & Coverage Table**: Fixed table layout and module ordering in output exports.
15. **Module-Header Scaffolding Leak Cleanup**: Prevented raw slide headers and metadata from leaking into question stems.
16. **Dynamic Model Resolution in SlotOrchestrator**: Replaced hardcoded `model="qwen2.5:7b"` in `core/generation/orchestrator.py` with canonical `get_production_model()`.
17. **Extractor Kwargs Absorption (Crash Prevention)**: Added `**kwargs` to `v0_1/extractor.py:extract` signature so calls with `extract_images=False` in multi-file upload and single-file gateway fallback execute safely without `TypeError`.

---

## 3. Explicitly Deferred to Future Sessions

* **Agentic Self-Repair / Critic Loops**: Deferred. Any future repair mechanism should start lightweight (e.g., deterministic diagnosis mapping `CheckResult` codes to targeted prompts, not multi-agent LLM critics).
* **Observability / ELK Stack**: Deferred. If structured telemetry is added later, it should be a simple, non-blocking NDJSON logger on the existing retry loop, decoupled from external daemons.
* **Parallel `aion/` Greenfield Prototype**: Stays **inert**. Not wired into production; requires incremental, isolated validation against real PDFs before any adoption is considered.
* **Semantic Bloom-Depth Checking**: Verification of cognitive depth beyond superficial action-verb matching.
* **Terminology / OCR-Garbling Cleanup**: Scrubbing scan artifacts ("Oriented Consolidated Review", "Practise Figure") in raw PDF text.

---

## 4. Prioritized Latent Audit Items (Tracked for Next Session)

* **High Priority — #2. Hardcoded Linux Path in Asset Serving**: `aion_api.py:1797-1800` hardcodes `allowed_roots` to `_Path("/home/AIML1/AIQ/AION/workspace")`. On Windows or any alternate server, frontend figure/diagram requests (`/api/asset?path=...`) abort with HTTP 404. Also line 8: `sys.path.insert(0, '/home/AIML1/AIQ/AION')`. Remediation: replace with `Path(__file__).parent.resolve()`.
* **Medium Priority — #4. Hardcoded Ollama Default URLs**: `v0_1/llm.py` (lines 36, 100, 267, 419) and `aion_api.py` (line 451, 464) default to `http://127.0.0.1:11434` without fallback to `os.environ.get("OLLAMA_HOST")` or `os.environ.get("OLLAMA_BASE_URL")`.
* **Low Priority — #5. Hardcoded Linux `/tmp/` in V2 Pipeline Demo**: `core/pipeline/aion_pipeline_v2.py:40` uses `memory_path="/tmp/variation_v2_test.json"`, causing an unexpected `C:\tmp` directory to be created on Windows.
* **Benchmark Only — #6. Hardcoded 120s Timeout in A/B Grounding Benchmark**: `scratch/compare_qwen_grounding.py:438` sets `timeout_sec=120`, triggering runner deadlines when executing 14B under large context prompts.
