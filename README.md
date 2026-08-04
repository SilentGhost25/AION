# AION: Academic Intelligence Oriented Network

AION is a modular, high-fidelity **Research Operating System for Academic AI**. It combines a **production-grade engineering skeleton** (Unlimited-OCR, Docling, GLiNER, bge-m3, Qwen3-8B, vLLM, RAFT → GRPO → DPO training) with a **novel cognitive reasoning brain** (Academic Knowledge Genome, Academic Thought Graph traversal, RAG² Answer-First Generation, Examiner Personality Engine, and Reason Code guardrails).

This repository merges the native academic intelligence of the **AION Core** with the production-grade engineering, governance, workflow routing, and auditing practices of **DSATM**.

---

## Unified System Philosophy

1. **Strict Contract Enforcement**: All internal modules communicate strictly via the **Academic Object Model (AOM)** defined in `core/sdk/aom.py`. No generic JSON parsing or untyped structures are permitted.
2. **Academic Knowledge Genomes**: Knowledge concepts are represented as Genomes containing DNA strands (`definition_dna`, `relationship_dna`, `bloom_dna`, `source_dna`, `evolution_history`) supporting Genome Algebra (`MERGE`, `DIFF`, `CROSS`, `DECAY`).
3. **Academic Thought Graph (ATG) Traversal (Stage 3.5)**: Sits between retrieval and generation to decide *what* to ask about using traversal policies (`Depth-First`, `Breadth-First`, `Bloom-Directed`, `Examiner-Directed`, `Socratic`).
4. **RAG² Answer-First Generation**: Generates the *Ideal Answer* and *Marking Scheme* BEFORE reverse-generating the *Exam Question*.
5. **7-Signal GRPO Training**: Reinforcement learning fine-tuning optimizes for groundedness, originality, examiner style match, Bloom alignment, concept simulation consistency, and ATG traversal compliance.
6. **Reason-Code Guardrails (`RC-01` to `RC-07`)**: Causal failure routing directs rejections back to the specific stage that failed instead of blind retries.
7. **Traceability & Audit Logs (`AICallLog`)**: Every external inference or processing call is saved with latency, confidence metrics, prompt versions, and validation details.

---

## The Integrated 6-Stage + Stage 3.5 Pipeline

```text
Document Upload (PDF / Textbook / Syllabus)
  │
  ▼
Stage 1: Document Understanding (Unlimited-OCR + Docling + Table Transformer + Diagram Intelligence)
  │
  ▼
Stage 2: Knowledge Extraction & Genome Construction (GLiNER-relex -> Postgres Genome DNA Strands)
  │
  ▼
Stage 3: Hybrid Retrieval & Indexing (bge-m3 + BM25 + bge-reranker-v2-m3 + Qdrant)
  │
  ▼
Stage 3.5: Academic Thought Graph (ATG) Traversal (Depth-First / Socratic / Bloom-Directed -> Intent)
  │
  ▼
Stage 4: RAG² Answer-First Generation (Qwen3-8B + LoRA Examiner Personality -> Ideal Answer -> Question)
  │
  ▼
Stage 5: Self-Critic Ensemble & Guardrails (LettuceDetect Faithfulness + MinHash Originality + Reason Codes RC-01..07)
  │
  ▼
Stage 6: Faculty Review & Continual Learning (Streamlit Dashboard -> DPO Preference Pairs & Genome KEG Updates)
```

---

## Directory Overview

* `v0_1/` - Working standalone evolution-ready prototype pipeline ([v0_1/main.py](file:///c:/Users/Tarun%20J/OneDrive/Desktop/AION/v0_1/main.py)).
* `configs/` - Centralized parameters for universities, departments, generation tasks, default models (`aion_config.yaml`), and runtime environments.
* `core/` - The kernel, plug-in managers, registry engines, event bus, and the central SDK (`core/sdk/aom.py`).
* `database/` - Production PostgreSQL DDL (`database/schema.sql`) and migrations (`database/migrations/001_integrated_schema.sql`).
* `engines/` - 17 decoupled mini-project directories corresponding to each stage of the reconstruction, generation, and validation pipelines.
* `neural/` - Custom neural layers, graph networks, encoders, decoders, and continual training modules.
* `knowledge/` - Persistent domain graphs, prerequisite trees, Bloom's taxonomy maps, and examiner profile memory (`memory/concepts.json`).
* `datasets/` - Raw, synthetic, validation, and continual learning datasets segmented for curriculum training.
* `api/` - HTTP REST & WebSocket endpoint route handlers.
* `frontend/` - Skeletons for the ingestion, review, analytics, and admin panels.
* `plugins/` - Extendable behaviors for university question patterns, generators, custom tokenizers, and validators.
* `research/` - Prototype notebooks, benchmark suites, and publication paper sources (including `AION-Trainer`).
* `learning_factory/` - The ALF orchestrator (`aion_learning_manager.py`) and dataset compilers.

---

## Quick Start & Verification

### Run the v0.1 Prototype Pipeline
```bash
python -m v0_1.main
```

### Check SDK Schema Types
```bash
python -c "import core.sdk.aom as aom; print(aom.ReasonCode.RC_01_CONCEPT_AMBIGUOUS); print(aom.TraversalPolicy.SOCRATIC)"
```
