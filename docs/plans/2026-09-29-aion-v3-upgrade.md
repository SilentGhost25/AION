# AION v3 Pipeline Upgrade Implementation Plan

> **For Claude / Agent:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Upgrade the AION exam generation engine from v2 (single-agent, flat-RAG, local-only caller) to v3: a multi-agent generation loop (Planning, Writing, Evaluation, Refinement, Checking) grounded in Knowledge Graph concept weighting (KAQG), IRT-calibrated difficulty (SMART), tiered semantic caching (Krites), multi-provider circuit-breaker routing, and speculative RAG pipelining (RAGCache).

**Architecture:** Maintain strict backwards compatibility with existing Tier-0 deterministic contracts (`PaperSpec`, `QuestionSlot`, `GeneratedQuestion`, `run_pipeline`). Introduce modular, clean sub-packages under `core/api/`, `core/cache/`, `core/knowledge/`, `core/evaluation/`, and `core/generation/agents/`. Wrap existing slot-level generation cleanly so that multi-agent execution is toggled seamlessly via `AION_ENABLE_V3_AGENTS=true`, allowing fail-closed and deterministic fallbacks.

**Tech Stack:** Python 3.10+, NetworkX (Knowledge Graph & PageRank), FAISS & NumPy (Tiered Vector Cache), SciPy (IRT model curve fitting), Requests / HTTPX (Circuit-Breaker API Router), Pytest (TDD).

---

## Component Architecture & Codebase Mapping

```
Existing Pipeline Entry: v0_1/main.py (run_pipeline)
                  │
                  ▼
┌────────────────────────────────────────────────────────┐
│ Phase 1: Core Infra & API Gate                         │
│ 1. core/api/circuit_breaker.py & core/api/router.py    │
│ 2. core/cache/semantic_cache.py & krites_judge.py      │
└────────────────────────────────────────────────────────┘
                  │
                  ▼
┌────────────────────────────────────────────────────────┐
│ Phase 2: Knowledge Graph & Difficulty Calibration      │
│ 3. core/knowledge/kg_builder.py (KAQG PageRank)        │
│ 4. core/evaluation/irt_calibrator.py (SMART 1PL IRT)   │
└────────────────────────────────────────────────────────┘
                  │
                  ▼
┌────────────────────────────────────────────────────────┐
│ Phase 3: Multi-Agent Generation Loop (EduAgentQG)      │
│ 5. core/generation/agents/planning_agent.py            │
│ 6. core/generation/agents/writing_agent.py             │
│ 7. core/generation/agents/evaluation_agent.py          │
│ 8. core/generation/agents/refinement_agent.py          │
│ 9. core/generation/agents/checking_agent.py            │
│ 10. core/generation/agents/agent_orchestrator.py       │
└────────────────────────────────────────────────────────┘
                  │
                  ▼
┌────────────────────────────────────────────────────────┐
│ Phase 4: Speculative RAG & Pipeline Wiring             │
│ 11. core/generation/speculative_pipeline.py            │
│ 12. v0_1/main.py wiring + regression suite             │
└────────────────────────────────────────────────────────┘
```

---

## Phase 1: Circuit-Breaker API Router & Tiered Semantic Cache

### Task 1: Three-State Circuit Breaker
**Files:**
- Create: `core/api/__init__.py`
- Create: `core/api/circuit_breaker.py`
- Test: `tests/unit/test_circuit_breaker.py`

- **Behavior:**
  - Implements states: `CLOSED` (normal operation), `OPEN` (tripped, rejects calls), `HALF_OPEN` (testing recovery).
  - Trips to `OPEN` after `failure_threshold` (default 3) consecutive failures within sliding window.
  - Remains `OPEN` for `cooldown_sec` (default 60s), then transitions to `HALF_OPEN` on next call.
  - Single success in `HALF_OPEN` resets to `CLOSED`; failure immediately trips back to `OPEN`.
  - Thread-safe using `threading.Lock`.

### Task 2: Multi-Provider LLM Router with RPM/RPD Tracking
**Files:**
- Create: `core/api/router.py`
- Test: `tests/unit/test_api_router.py`

- **Behavior:**
  - Providers configured: Groq (30 RPM, 14.4k RPD), NVIDIA NIM (40 RPM), OpenRouter (20 RPM, 50 RPD), Google AI (15 RPM, 500 RPD), Local Ollama/vLLM (unlimited fallback).
  - Selects provider based on: (1) Circuit state != OPEN, (2) Active RPM/RPD within quota, (3) Priority rank + least-recently-used tiebreak.
  - Records request timestamps for rolling 60-second RPM window and daily RPD reset.
  - Exposes unified method `dispatch(prompt: str, schema: Optional[dict], max_tokens: int) -> LLMResponse`.

### Task 3: Tiered Semantic Cache with FAISS & TTL
**Files:**
- Create: `core/cache/__init__.py`
- Create: `core/cache/semantic_cache.py`
- Test: `tests/unit/test_semantic_cache.py`

- **Behavior:**
  - Two tiers: Static Cache (curated / log-mined, cosine $\ge 0.90$) and Dynamic Cache (runtime promoted, cosine $\ge 0.85$).
  - Fast index search via `faiss.IndexFlatIP` on normalized 384-d embeddings (using local lightweight model or fallback hash projection).
  - Cache entry schema: `prompt_hash`, `embedding`, `response_json`, `created_at`, `ttl_sec` (default 24h), `source_doc_hash`.
  - Invalidation hook: invalidate entries if `source_doc_hash` changes.

### Task 4: Asynchronous LLM Judge for Dynamic Cache Promotion (Krites)
**Files:**
- Create: `core/cache/krites_judge.py`
- Test: `tests/unit/test_krites_judge.py`

- **Behavior:**
  - When similarity is borderline ($0.75 \le \text{sim} < 0.85$), triggers an asynchronous promotion evaluation task.
  - Prompts evaluator/judge LLM: "Is response B semantically equivalent and accurate for query A given syllabus context?"
  - If verdict is `APPROVED`, promotes entry to Dynamic FAISS cache with active TTL.
  - Zero latency impact on main generation path (fire-and-forget background task via `ThreadPoolExecutor`).

---

## Phase 2: Knowledge Graph Concept Layer & IRT Difficulty Calibration

### Task 5: Knowledge Graph Builder with Multi-Graph Isolation (KAQG)
**Files:**
- Create: `core/knowledge/kg_builder.py`
- Test: `tests/unit/test_kg_builder.py`

- **Behavior:**
  - Ingests `DocumentArtifact` text blocks and extracts Subject-Predicate-Object (SPO) triples using lightweight regex/pattern extraction + LLM structured fallback.
  - Multi-graph isolation: Builds an isolated `networkx.DiGraph` per module to prevent inter-module cross-contamination.
  - Calculates PageRank on each module graph (`nx.pagerank(G)`) to assign centrality/importance weights to core syllabus concepts.
  - Method `get_concept_neighborhood(module_id: str, concept: str, max_depth: int = 2) -> List[dict]` returns connected nodes, relations, and PageRank weights for prompt injection.
  - Caches graph by `source_pdf_sha256` to avoid recomputing across runs.

### Task 6: IRT-Based Difficulty Calibrator (SMART)
**Files:**
- Create: `core/evaluation/irt_calibrator.py`
- Test: `tests/unit/test_irt_calibrator.py`

- **Behavior:**
  - Constructs simulated student personas with abilities $\theta \in \{-2.0, -1.0, 0.0, +1.0, +2.0\}$.
  - Simulates probabilistic answer accuracy per ability level using question cognitive demand (Bloom level, math requirements, mark weight).
  - Fits a 1-Parameter Logistic (1PL / Rasch) IRT model: $P(\theta) = \frac{1}{1 + e^{-a(\theta - b)}}$ using `scipy.optimize.curve_fit`.
  - Outputs estimated item difficulty parameter $b \in [-3.0, +3.0]$ and standard error.
  - Flags questions where target difficulty does not match assigned marks (e.g., a 2-mark question with $b > 2.0$ or a 10-mark question with $b < -1.0$).

---

## Phase 3: Multi-Agent Generation Layer (EduAgentQG)

### Task 7: Planning Agent
**Files:**
- Create: `core/generation/agents/__init__.py`
- Create: `core/generation/agents/planning_agent.py`
- Test: `tests/unit/test_planning_agent.py`

- **Behavior:**
  - Accepts `PaperSpec`, `module_index`, `marks_partition`, and `KnowledgeGraph`.
  - Allocates `QuestionSlot` instances with locked Bloom targets, CO mappings, and KG concept targets.
  - Determines `visual_required` gate based on chunk proximity and visual RAG policy (prevents over-binding).
  - Emits `QuestionPlan` dataclass for the Writing Agent.

### Task 8: Writing Agent
**Files:**
- Create: `core/generation/agents/writing_agent.py`
- Test: `tests/unit/test_writing_agent.py`

- **Behavior:**
  - Takes `QuestionPlan`, evidence text chunks, and KG neighborhood context.
  - Invokes LLM Router with strict prompt contracts (Bloom action verb at start, standalone question, KaTeX math blocks, figure references).
  - Parses JSON response into `GeneratedQuestion`.

### Task 9: Evaluation Agent (Batched & Multi-Metric)
**Files:**
- Create: `core/generation/agents/evaluation_agent.py`
- Test: `tests/unit/test_evaluation_agent.py`

- **Behavior:**
  - Evaluates candidate question across 5 dimensions:
    1. Bloom verb compliance (`BLOOM_VERB_LEVEL_MAP`).
    2. Factual grounding against evidence chunks & KG triples.
    3. Structural completeness & self-contained phrasing.
    4. Sibling deduplication (Jaccard similarity $< 0.40$).
    5. IRT difficulty alignment ($b$ parameter compatibility).
  - Emits `EvaluationReport(passed: bool, defect_codes: List[str], severity: str)`.

### Task 10: Refinement Agent (AutoHealer + API Repair)
**Files:**
- Create: `core/generation/agents/refinement_agent.py`
- Test: `tests/unit/test_refinement_agent.py`

- **Behavior:**
  - Inspects `EvaluationReport.defect_codes`.
  - Layer 1 (Zero-cost): Applies local `AutoHealer` (regex repairs, verb substitution, KaTeX formatting).
  - Layer 2 (Targeted API repair): If programmatic healing fails, prompts router with targeted differential instruction (e.g., "Rewrite instruction to begin with Bloom verb 'Determine' without altering technical premises").
  - Layer 3 (SymPy validation): Validates numerical calculations and formulas.

### Task 11: Checking Agent & Fail-Closed Gate
**Files:**
- Create: `core/generation/agents/checking_agent.py`
- Test: `tests/unit/test_checking_agent.py`

- **Behavior:**
  - Enforces fail-closed validation: slot contract compliance, marks sum matching, visual asset integrity (file existence and non-empty path).
  - Produces final `VerifiedQuestion` or raises `UnresolvedSlotException`.

### Task 12: Multi-Agent Orchestrator Assembly
**Files:**
- Create: `core/generation/agents/agent_orchestrator.py`
- Test: `tests/unit/test_agent_orchestrator.py`

- **Behavior:**
  - Coordinates the 5-agent pipeline per module:
    `PlanningAgent -> WritingAgent -> EvaluationAgent -> RefinementAgent (if needed) -> CheckingAgent`.
  - Drop-in interface replacement for existing `SlotOrchestrator` methods.
  - Feature flag `AION_ENABLE_V3_AGENTS` allows instant rollback to legacy orchestrator if needed.

---

## Phase 4: Speculative Pipelining & End-to-End System Integration

### Task 13: Speculative RAG Pipeline Orchestrator (RAGCache)
**Files:**
- Create: `core/generation/speculative_pipeline.py`
- Test: `tests/unit/test_speculative_pipeline.py`

- **Behavior:**
  - Decouples document ingestion from generation: As soon as Module 1 artifact is extracted/chunked, dispatches Module 1 generation workers immediately.
  - While Module 1 generation executes, extraction and OCR continue concurrently for Modules 2–5.
  - Reduces total Time-To-First-Token (TTFT) and paper completion time by 30–40%.

### Task 14: Wiring into `v0_1/main.py`
**Files:**
- Modify: `v0_1/main.py`
- Test: `tests/integration/test_v3_pipeline_integration.py`

- **Behavior:**
  - Injects `KnowledgeGraphBuilder` after document artifact ingestion/caching.
  - Connects `AgentOrchestrator` and `SpeculativeRAGPipeline` when `AION_ENABLE_V3_AGENTS=true`.
  - Integrates `TieredSemanticCache` into question generation query path.
  - Emits telemetry in final QA report: KG nodes/edges count, IRT difficulty distributions, cache hit/miss ratio, circuit breaker states.

---

## Verification & Acceptance Criteria
1. **Circuit Breaker:** Under simulated provider timeouts, trips to OPEN after 3 failures and redirects requests to secondary providers without dropping slot generation.
2. **Knowledge Graph:** Concept triples extracted and PageRank computed per module; concepts present in prompt context.
3. **IRT Calibration:** Produces calibrated difficulty $b$ scores for all 24 slots in a SEE paper; warns on mismatch.
4. **Semantic Cache:** Second run of an identical syllabus achieves $\ge 60\%$ cache hits with $\ge 4\times$ latency reduction.
5. **Backwards Compatibility:** All existing tests in `tests/` pass with zero regressions when `AION_ENABLE_V3_AGENTS=false`.
