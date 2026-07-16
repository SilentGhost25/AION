# AION: Academic Intelligence Oriented Network

AION is a modular, high-fidelity **Research Operating System for Academic AI**. It provides a unified pipeline designed to parse raw books, previous papers, and notes into structured knowledge genomes, construct reasoning graphs, simulate expert examiners, and generate verified, benchmark-compliant academic evaluations (e.g., examination papers, rubrics, and educational content).

This repository merges the native academic intelligence of the **AION Core** with the production-grade engineering, governance, workflow routing, and auditing practices of **DSATM**.

---

## Unified System Philosophy

1. **Strict Contract Enforcement**: All internal modules communicate strictly via the **Academic Object Model (AOM)** defined in `core/sdk/aom.py`. No generic JSON parsing or untyped structures are permitted.
2. **Comprehensive Versioning Matrix**: Beyond model checkpoints and training datasets, AION actively versions generation prompts, template structures, parser configurations, and engine binaries.
3. **Traceability & Audit Logs (`AICallLog`)**: Every external inference or processing call is saved with latency, confidence metrics, prompt versions, and validation details to form a clean audit trail.
4. **Human-in-the-Loop Safeguards**: Model adjustments and incremental learning datasets are gated by a human review process (`ApprovalStatus`) before ingestion into the continual training replay buffer.
5. **Fail-Safe Generation Loops**: Mismatches in validator results trigger immediate context expansion and alternative model reasoning paths prior to manual fallback triggers.

---

## Directory Overview

* `configs/` - Centralized parameters for universities, departments, generation tasks, default models, and runtime environments.
* `academic/` - The physical filesystem hierarchy mapping academic schemes, semesters, and subjects containing textbook sources, notes, question banks, and generated papers.
* `core/` - The kernel, plug-in managers, registry engines, events bus, and the central SDK containing the AOM definitions.
* `engines/` - 17 decoupled mini-project directories corresponding to each stage of the reconstruction, generation, and validation pipelines.
* `neural/` - Custom neural layers, graph networks, encoders, decoders, and continual training modules.
* `knowledge/` - Persistent domain graphs, prerequisite trees, Bloom's taxonomy maps, and examiner profile memory.
* `datasets/` - Raw, synthetic, validation, and continual learning datasets segmented for curriculum training.
* `database/` - Postgres, Vector, and Graph storage connector configurations and migration scripts.
* `api/` - HTTP REST & WebSocket endpoint route handlers.
* `frontend/` - Skeletons for the ingestion, review, analytics, and admin panels.
* `plugins/` - Extendable behaviors for university question patterns, generators, custom tokenizers, and validators.
* `research/` - Prototype notebooks, benchmark suites, and publication paper sources (including `AION-Trainer`).
* `learning_factory/` - The ALF orchestrator (`aion_learning_manager.py`) and dataset compilers.

---

## The Unified Ingestion & Generation Pipeline

```text
Upload Book / Syllabus
  │
  ▼
Parser Engine (Document Intelligence extraction)
  │
  ▼
Academic Reconstruction (Construct course directories & subject structures)
  │
  ▼
Academic Genome & Answer Graph Construction (Genes mapping & prerequisite maps)
  │
  ▼
Knowledge Validator (Contradiction & gap validation check)
  │
  ▼
Dataset Versioning (Version metadata & training splits increment)
  │
  ▼
Exam Blueprint Locking (Freeze total marks, modules covered, Bloom levels)
  │
  ▼
Question Planner & Examiner Reasoning (Draft section and question outlines)
  │
  ▼
Question Discovery (Determine implied examinable questions from genome nodes)
  │
  ▼
Academic Self-RAG (Verify candidates against textbooks & notes)
  │
  ▼
Context-Expanding Fail-safe Loops (Trigger re-retrieval / re-reasoning on failure)
  │
  ▼
Validation & Discriminator Gates (Check grammar, Bloom levels, and university styles)
  │
  ▼
Question Ranking & Formatting (Select top candidates and build layout drafts)
  │
  ▼
Paper Compiler (Generate LaTeX, PDF, or DOCX publications)
  │
  ▼
Review & Approval Gates (Human reviewer audits edits & approves Learning Episodes)
  │
  ▼
Replay Buffer (Insert approved training episodes into Learning Factory)
  │
  ▼
Scheduled Retraining & Model weights Promotion (Weights benchmarking & update)
```

---

## Development Setup

To initialize the project and dependencies:
```bash
pip install -r requirements.txt
```

To run a syntax and lint check:
```bash
ruff check .
```

To execute test suites:
```bash
pytest
```
