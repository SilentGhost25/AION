# AION System Context & Patch State

> **Notice for AI Assistants & LLMs**: This file is the single source of truth for the working architecture, repository directory layout, and current patch state of AION. **This document must be updated after every patch, feature addition, or architectural change.**

---

## 1. System Overview at a Glance

**AION (Academic Intelligence Oriented Network)** is a production-grade, autonomous exam paper and evaluation scheme generation engine designed for university engineering curricula (primarily VTU - Visvesvaraya Technological University OBE standards).

### Primary Capabilities
- **Multimodal Document Understanding**: Extracts structured text, formulas (KaTeX/LaTeX), tables, and technical diagrams from course syllabus PDFs and reference textbooks.
- **Pedagogy & Blueprint Planning**: Partitions questions into strict marks divisions (VTU Internal Assessment: 6+4=10 marks per question; Semester End Exam: 8+6+6=20 marks per question, 5 modules, total 100 marks).
- **Taxonomy Enforcement**: Enforces Revised Bloom's Taxonomy (L1 Remember to L5 Evaluate) with strict cross-field validation between marks, question types, cognitive level, and opening action verbs.
- **RAG² Answer-First Synthesis**: Generates full solutions and marking schemes alongside question stems to ensure mathematical and logical solvability.
- **Auto-Healing & Validation Gates**: Multi-stage linting (`core/validation/linter.py`, `bloom_validator.py`, `math_validator.py`, `export_gate.py`) and non-destructive deterministic auto-healing (`auto_healer.py`).
- **Real-Time Evaluation**: Real-time deterministic RAG evaluation metrics (`core/evaluation/`) calculating groundedness, faithfulness, hallucination detection, and relevance with SSE streaming.
- **Multi-Format Export**: Production DOCX generation (`v0_1/docx_export.py`) with VTU styling, embedded figures, formula rendering, and JSON metadata.

---

## 2. Directory Layout & Key Modules

| Path | Purpose & Responsibilities |
|---|---|
| `v0_1/` | **Active Production Pipeline Engine**. Primary execution core for extraction, segmentation, chunking, slot allocation, generation, and export. |
| `v0_1/main.py` | Core pipeline entrypoint and orchestrator; manages module partitioning, LLM prompts, slot filling, cross-module deduplication, and paper assembly. |
| `v0_1/chunk_image_mapper.py`| Semantic chunking, figure-to-text association, topical neighborhood clustering, and `ContentRole` firewalling (`CORE`, `SUPPORTING`, `EXTERNAL`). |
| `v0_1/difficulty_policy.py` | Mapping of marks and question types to Course Outcomes (CO) and Bloom's levels (L1–L5); implements L3 numerical invariant. |
| `v0_1/content_filter.py` | Pre-generation text cleaning, noise removal, OCR artifact filtering, and academic density validation (`_is_dense_academic`). |
| `v0_1/extractor.py` | Multi-engine document text and image extraction (Docling, PyMuPDF, RapidOCR) with caching in `.aion_cache/`. |
| `v0_1/docx_export.py` | Formal university exam paper exporter (.docx) with question layout, marks tables, RBT/CO headers, and embedded figures. |
| `aion_api.py` | **FastAPI Server** exposing REST API and Server-Sent Events (SSE) for live generation streaming, asset serving (`/api/asset`), job queue management, and metrics. |
| `core/` | **Modular Production Framework**: Refactored evaluation, generation, validation, and extraction subsystems. |
| `core/evaluation/` | Real-time deterministic RAG evaluation suite (`deterministic.py`, `contracts.py`, `aggregation.py`) providing RAGAS-inspired metrics without external network calls. |
| `core/generation/` | `SlotOrchestrator` for threaded slot-level generation, retry policies, and `auto_healer.py` for deterministic repair. |
| `core/validation/` | Gatekeeper linting: `bloom_validator.py` (canonical Bloom verb sets), `linter.py` (opening verb & syntax checks), `math_validator.py` (KaTeX/equation sanity), `export_gate.py`. |
| `aion/` | **Greenfield V2 Architecture Baseline**: Multimodal DOM (`aion/core/dom`), Knowledge Compiler (`aion/core/knowledge`), Evidence Planner (`aion/core/planning`), and fusion extraction (`aion/core/extraction`). |
| `frontend/` | Next.js / React application with Tailwind CSS and Radix UI for interactive paper configuration, live generation streaming, and paper preview. |
| `configs/` | System, university blueprint, and model configuration YAMLs (`aion_config.yaml`). |
| `workspace/` | Local runtime artifacts, figure crops (`workspace/artifacts/figures/`), generated exports (`exports/`), and temporary files. |
| `tests/` | Comprehensive test suite covering regression tests (`test_golden_pipeline_regressions.py`), DOM (`test_multimodal_dom.py`), RAG metrics (`test_realtime_rag_metrics.py`), and evidence planner (`test_evidence_planner.py`). |
| Root Launch Scripts | `start_server.ps1` / `start_server.sh` (dynamic multi-instance server launcher), `start_aion.py`, `aiq` CLI runner. |

---

## 3. Current Release & Patch State

- **Active Branch**: `v2`
- **Head Commit**: `72d5cd15` (`fix(pipeline): make _print_exam_paper dynamically handle 2 questions per module`)
- **Remote Tracking**: Synchronized with `origin/v2`
- **Production Predecessor**: `v1.1.0` (Branch `mod_eval`, commit `4b9fed00`)
- **Golden Rollback Baseline**: `v1.0.0` (Tag `v1.0.0`, commit `6e3631a`)

---

## 4. Changelog: Updates in the Current Patch (`v2`)

The current patch (branch `v2`, commits `4978eb52` through `72d5cd15`) introduces major architectural stabilization, strict pedagogical invariants, and real-time evaluation capabilities:

### 1. Equal Module Question Allocation (`v0_1/main.py`)
- **Problem**: Multi-module papers previously had uneven slot allocations or generated 4 partitions per module in multi-module exams, violating the standard VTU 2-question-per-module layout.
- **Fix**: Standardized multi-module and SEE/IA exam generation to allocate exactly **2 main questions per module** (Set 1 → Q1 & Q2; Set 2 → Q3 & Q4; ... Set 5 → Q9 & Q10). Single-module pool mode preserves 4 partitions for generation variety.
- **Display Formatter**: Updated `_print_exam_paper` to dynamically handle 2 questions per module with interleaved `[OR]` dividers without raising `IndexError`.

### 2. Numerical L3 Bloom Invariant (`v0_1/difficulty_policy.py`, `core/validation/`)
- **Problem (Q3 Defect)**: Calculation and derivation questions using verbs like *"Calculate"* were being classified as L4 (Analyze) or L5 (Evaluate) when assigned high marks (8M/10M), causing taxonomy validation failures.
- **Fix**: Enforced a strict cross-field invariant: all `NUMERICAL` tasks and procedural calculation verbs (`calculate`, `solve`, `determine`, `compute`, `find`, `derive`) are capped at **Bloom L3 (Apply)** across all marks tiers.
- **Linter & Validator**: `core/validation/linter.py` and `bloom_validator.py` reject questions where slot Bloom verbs do not strictly belong to `BLOOM_VERB_LEVEL_MAP[L{bloom}]`.

### 3. ContentRole Firewall (`v0_1/chunk_image_mapper.py`)
- **Problem**: Textbook revision tables, summary glossaries, and end-of-chapter question banks were contaminating primary question generation evidence.
- **Fix**: Implemented regex filtering in `classify_chunk_depth` that classifies revision patterns (`"important terms for revision"`, `"chapter summary"`, `"question bank"`, `"exercises for module"`) as `EXTERNAL`, preventing them from serving as primary generation chunks.

### 4. Bloom Verb & Starting Keyword Linter (`core/validation/linter.py`)
- **Enhancement**: Hardened `check_bloom_verb_at_start` to enforce:
  1. Slot Bloom verb must strictly belong to `BLOOM_VERB_LEVEL_MAP` for the declared Bloom level.
  2. The question instruction must start with the exact Bloom verb assigned (case-insensitive, supporting UK/US spellings).
  3. Opening word must belong to the declared level.
  4. AutoHealer prefixes the exact Bloom keyword if missing.

### 5. Configurable Course Outcome (CO) Policy (`v0_1/difficulty_policy.py`)
- Reverted default Course Outcome assignment to `marks-based` mode (`<=4M` → CO1, `6M` → CO2, `8M+` → CO3) to match standard OBE curriculum practices.
- Institutional syllabus mode (`module-based`: Module $M \to \text{CO}M$) and `hybrid` mode remain available via the `AION_CO_MODE` environment variable.

### 6. Real-Time Deterministic RAG Evaluation Framework (`core/evaluation/`)
- Implemented `RealtimeRAGMetrics` contract (`evaluation_framework='aion_ragas_inspired'`).
- Added `DeterministicRAGEvaluator` observer operating without external network/LLM dependencies:
  - **Harmonic Mean**: Mathematical harmonic mean with zero-division safety.
  - **Hallucination Detection**: Independent detection decoupled from faithfulness (identifies phantom figures, ungrounded equations, or unsupported entities).
  - **Question Relevance**: Evaluates alignment with Bloom target, marks, archetype, and module topic.
  - **Metric Provenance**: Tracks `trace_id`, `generation_id`, `slot_id`, `model`, `provider`, and git `commit`.
- Integrated thread-safe metric listener registry and SSE streaming (`ragas_metric`, `question_ready`) into `aion_api.py`.

### 7. V2 Architecture Baseline Scaffolding (`aion/core/`)
- Introduced clean, typed greenfield architecture modules:
  - `aion/core/dom`: Multimodal Document Object Model (`document_dom.py`, `evidence.py`, `artifacts.py`, `relationships.py`).
  - `aion/core/knowledge`: Semantic knowledge compiler (`compiler.py`), ledger (`ledger.py`), and topic topology (`topology.py`).
  - `aion/core/planning`: Evidence planner (`evidence_planner.py`), grounding verifier (`grounding_verifier.py`), and serializer (`serializer.py`).
  - `aion/core/extraction`: Extraction benchmark and multimodal fusion (`fusion.py`, `pdf_extractor.py`).

### 8. Python 3.12 / 3.14 Crash Resolution (`v0_1/content_filter.py`)
- Fixed character class range `[=∑∫√≤≥±->←]` in `_is_dense_academic` where `±->←` caused an invalid range `re.PatternError` on Python 3.14 and deprecation warnings on 3.12. Modified to literal `[=∑∫√≤≥±←→>-]`.

### 9. Import & Execution Safety
- Fixed erroneous import of `Orchestrator` in emergency salvage loops and regression tests by pointing to `SlotOrchestrator` in `core/generation/orchestrator.py`.
- Added `**kwargs` to `v0_1/extractor.py:extract` to safely absorb parameters (such as `extract_images=False`) in multi-file upload workflows.
- Resolved `UnboundLocalError` for `math_required` in `_generate_main_question`.

### 10. Subject Threading & Archetype Hardening (`aion_api.py`, `v0_1/main.py`)
- Completely threaded the `subject` parameter through generation calls.
- Added explicit `iot` (Internet of Things) subject archetype to the archetype registry to prevent cross-domain contamination.
- Fixed KaTeX formula validation on Windows environments and raw extracted text pass-through.

### 11. Generation Scaffolding, Length & Preamble Defense (Steps 0–3b)
- **Prompt Scaffolding Removal (`core/generation/orchestrator.py`)**: Deleted `Min Clauses: {min_dims}` prompt constraint and dead `sec_verb` resolution code. Replaced compound multi-clause few-shot examples with pure single-task archetypes (target length: 15–30 words) with no clause-joining `and` conjunctions. Added negative few-shot example demonstrating prohibited compound multi-task and mechanism-leak patterns.
- **Embedded Preamble Defense (`core/generation/auto_healer.py`)**: Added `_has_embedded_action_verb` check inside `_fix_bloom_verb_start`. When the model produces a leading preamble with an embedded Bloom verb (e.g. *"In satellite communications, explain..."*), AutoHealer refuses to prepend a second verb, avoiding double-verb corruption and cleanly escalating to an LLM retry.
- **Canned Dimension Filler Removal (`core/generation/auto_healer.py`)**: Deleted `_fix_add_dimensions` to prevent mechanical padding of canned strings (*", explaining the underlying principles"*) onto questions.
- **Normalizer Given Preservation (`core/generation/orchestrator.py`)**: Deleted the blind `_pos > 15` Bloom verb chop in the pre-validation normalizer, preserving essential numerical problem parameters and problem givens.
- **Standing Test Suite**: Added 4 standing tests in `tests/test_golden_pipeline_regressions.py` covering all defenses (37/37 passing).

### 12. vLLM High-Throughput Inference Backend & Universal Domain Adaptability
- **vLLM Engine Integration (`core/generation/robust_llm_caller.py`, `v0_1/llm.py`)**:
  - Added native OpenAI-compatible `/v1/chat/completions` endpoint support for vLLM serving on NVIDIA L40 (48GB VRAM), resolving the multi-minute sequential generation bottleneck.
  - Added `max_tokens` (default: 4096) to `LLMRequest` preserving full backward compatibility across all keyword construction sites.
  - Structured output parity: Passes `response_format={"type": "json_schema", ...}` when schema is provided with automatic fallback to `{"type": "json_object"}` on older server runtimes.
  - Token truncation guard: Automatically flags unclosed responses reaching $\ge 0.9 \times \text{max\_tokens}$ with `TRUNCATED_AT_MAX_TOKENS` rather than generic parse errors.
  - Pre-flight model assertion: `check_health()` and `assert_model_ready()` query `/v1/models` and assert that `AION_MODEL` is actively served, failing loudly with operational restart guidance if mismatched.
  - Gated `self_heal_ollama` in `aion_patch.py` to prevent terminating processes when operating in vLLM mode.
- **Universal Academic Domain Defense (Phase 1 P0 Fixes)**:
  - Replaced fail-closed `[ENGINEERING & QUANTITATIVE EXAMINATION DIRECTIVE]` in `aion_patch.py` with domain-neutral `[ACADEMIC DOMAIN DIRECTIVE]`.
  - Removed anti-theoretical directive (*"Avoid descriptive, theoretical, or purely textual tasks"*) from VTU examination prompt header in `aion_patch.py`.
  - Refined `_code_signals` in `core/generation/orchestrator.py` with strict regexes (`\bclass\s+[A-Z]\w*\s*[:\(]`) to prevent natural prose (e.g. "class action suit", "social class") from incorrectly tripping programming mode in Law and Humanities.
  - Decoupled CS Data Structures syllabus distance penalties in `core/validation/teacher_suitability_gate.py` for non-CS subjects.
  - Injected universal wrapper support in `aion_patch.py` to seamlessly handle both `str` prompts and `LLMRequest` dataclass objects.
  - Test validation: Created `tests/test_vllm_integration.py` (9/9 passing; combined full suite: 46/46 passing).

---

## 5. End-to-End Pipeline Execution Lifecycle

```text
[Course Material Upload: Syllabus / Notes / Textbook PDF]
                           │
                           ▼
  1. Document Ingestion & Extraction (v0_1/extractor.py)
     • RapidOCR (word count > 100 gating) + PyMuPDF / Docling
     • Disk-backed extraction cache (.aion_cache/)
                           │
                           ▼
  2. Content Filtering & Firewalling (v0_1/content_filter.py, chunk_image_mapper.py)
     • Academic density filter (_is_dense_academic)
     • ContentRole firewall (marks revision tables/question banks as EXTERNAL)
     • Semantic chunking with figure bounding box mapping
                           │
                           ▼
  3. Blueprint & Pedagogical Partitioning (v0_1/main.py, difficulty_policy.py)
     • VTU marks split: IA (6+4=10M) or SEE (8+6+6=20M across 5 modules)
     • 2 Questions per module (Set 1: Q1/Q2, Set 2: Q3/Q4, etc.)
     • Bloom level assignment (L1-L5) & L3 Numerical Invariant
                           │
                           ▼
  4. Generation & RAG² Synthesis (v0_1/main.py, core/generation/)
     • LLM prompt construction with grounding excerpts & assigned Bloom verb
     • Solution & marking scheme generated concurrently with question stem
     • Thread-safe cross-module deduplication registry (shared_generated_texts)
                           │
                           ▼
  5. Validation & Auto-Healing (core/validation/, core/generation/auto_healer.py)
     • Linter: Bloom verb opening check, KaTeX math syntax, question length
     • AutoHealer: Non-destructive repair of missing keywords or formatting
     • Whole-paper Jaccard similarity audit (threshold < 0.55)
                           │
                           ▼
  6. Real-Time RAGAS-Inspired Evaluation (core/evaluation/)
     • Groundedness, Faithfulness, Hallucination, Relevance scored per slot
     • Emitted via SSE events (ragas_metric) to frontend
                           │
                           ▼
  7. Final Assembly & Export (v0_1/docx_export.py, aion_api.py)
     • Production VTU .docx generation with embedded diagrams and formula tables
     • JSON paper blueprint and solution archive returned to client
```

---

## 6. Critical Invariants for Developers & LLMs

When modifying or generating code in this repository, **strictly uphold these rules**:

1. **VTU Marks & Question Allocations**:
   - **Internal Assessment (IA)**: 6 + 4 = 10 marks per question pair.
   - **Semester End Exam (SEE)**: 8 + 6 + 6 = 20 marks per question pair across 5 modules (total 100 marks).
   - In multi-module exams, allocate exactly **2 main questions per module** (Set $M$ contains questions $2M-1$ and $2M$ separated by an `[OR]` alternative).
2. **Bloom Taxonomy & Numerical Invariant**:
   - Every question instruction MUST start with an approved Bloom verb belonging to `BLOOM_VERB_LEVEL_MAP[L{bloom}]`.
   - Numerical calculations (`calculate`, `solve`, `determine`, `compute`, `find`, `derive`) **MUST NOT exceed Bloom L3**.
3. **Grounding & ContentRole**:
   - Do NOT use chunks classified as `EXTERNAL` (revision summaries, glossaries, question banks) as primary question stems.
4. **No Direct Hardcoded Paths**:
   - Use `Path(__file__).parent.resolve()` or configured workspace paths. Never hardcode `/home/...` or `C:\...` file paths.
5. **Deduplication**:
   - Register all generated stems into `shared_generated_texts` across threads. Whole-paper Jaccard similarity between any two subquestions must remain $< 0.55$.
6. **Inference Backend & vLLM Protocol**:
   - `AION_BACKEND` defaults to `"ollama"` if unset, preserving full backward compatibility with local developer setups.
   - Setting `AION_BACKEND=vllm` shifts all generation to OpenAI-compatible `/v1/chat/completions` served on `AION_LLM_HOST` (default `http://localhost:8000`).
   - `self_heal_ollama` is strictly gated to `AION_BACKEND == "ollama"` and is a safe no-op under vLLM.
   - Sampling temperature for exam generation is standardized to `0.1` across both `RobustLLMCaller` and `v0_1/llm.py` to prevent formatting and Bloom verb drift.
   - Random seed defaults to `42` across all inference calls and can be overridden via `AION_SEED` environment variable for reproducible testing or deliberate variance.


---

## 7. Useful Commands

```powershell
# Run FastAPI server locally
python -m uvicorn aion_api:app --host 0.0.0.0 --port 8000 --reload

# Run standalone v0.1 prototype pipeline
python -m v0_1.main

# Run full regression test suite
pytest tests/test_golden_pipeline_regressions.py -v

# Run real-time evaluation metrics test
pytest tests/test_realtime_rag_metrics.py -v

# Check git status ignoring untracked files
git status -uno
```

---

## 8. Context Maintenance Protocol

Whenever changes are made to this repository:
1. Check off or update the **Changelog** section with new commits and their specific functional impacts.
2. Update **Current Release & Patch State** with the latest commit hash and active branch.
3. If new modules or architecture directories are added, update the **Directory Layout** table.
4. Ensure any changes to pedagogical policies or marks allocation are reflected in **Critical Invariants**.
