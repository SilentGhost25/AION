# AION Unified System Architecture Blueprint (Merged)

This document freezes the final combined system architecture of AION (Academic Intelligence Oriented Network) integrated with production-grade governance, versioning, auditing, and state machine workflows.

---

## 1. Unified Information Flow Topology

AION combines native neural academic intelligence engines with a strict contract-first software execution layer.

```
                    ┌─────────────────────────┐
                    │      Upload Client      │
                    └────────────┬────────────┘
                                 │ Ingestion Upload API
                    ┌────────────▼────────────┐
                    │      API Gateway        │
                    │   (Pydantic AOM Gate)   │
                    └────────────┬────────────┘
                                 │ Strict AOM Contracts
      ┌──────────────────────────┼──────────────────────────┐
      │ Core Engine Registry     │                          │
      │  ┌───────────────────────▼───────────────────────┐  │
      │  │                 Event Bus / Queue             │  │
      │  └───────────────────────┬───────────────────────┘  │
      │  ┌───────────────────────▼───────────────────────┐  │
      │  │        Audited Workflow Engine (States)       │  │
      │  └───────────────────────┬───────────────────────┘  │
      └──────────────────────────┼──────────────────────────┘
                                 │
      ┌──────────────────────────┼──────────────────────────┐
      │ Generation Pipeline      │                          │
      │  ┌───────────────────────▼───────────────────────┐  │
      │  │  1. Ingestion & Reconstruction                │  │
      │  └───────────────────────┬───────────────────────┘  │
      │  ┌───────────────────────▼───────────────────────┐  │
      │  │  2. Academic Genome & Answer Graph Builder    │  │
      │  └───────────────────────┬───────────────────────┘  │
      │  ┌───────────────────────▼───────────────────────┐  │
      │  │  3. Question Discovery & Blueprint Engine     │  │
      │  └───────────────────────┬───────────────────────┘  │
      │  ┌───────────────────────▼───────────────────────┐  │
      │  │  4. Context Expansion & AI Generator          │  │
      │  └───────────────────────┬───────────────────────┘  │
      │  ┌───────────────────────▼───────────────────────┐  │
      │  │  5. Verification & Discriminator Gates        │  │
      │  └───────────────────────┬───────────────────────┘  │
      │  ┌───────────────────────▼───────────────────────┐  │
      │  │  6. Review, Edit & Feedback Capture           │  │
      │  └───────────────────────┬───────────────────────┘  │
      └──────────────────────────┼──────────────────────────┘
                                 │
      ┌──────────────────────────▼──────────────────────────┐
      │ Continuous Learning                                 │
      │  ┌───────────────────────────────────────────────┐  │
      │  │  7. Human Approval Gate                       │  │
      │  ├───────────────────────────────────────────────┤  │
      │  │  8. Replay Buffer & Training (ALF)            │  │
      │  └───────────────────────────────────────────────┘  │
      └─────────────────────────────────────────────────────┘
```

---

## 2. Engineering Governance Features

### 2.1 Contract-First Schema Enforcements
Every data transaction across engines, databases, APIs, and client runtimes must adhere to the Pydantic schemas in `core/sdk/aom.py`. Untyped dictionaries, unstructured database rows, and raw strings are disallowed.

### 2.2 Global Versioning Policy
Every asset is versioned explicitly:
- **Models**: Version of the base weights / fine-tuned checkpoints.
- **Datasets**: Versioned incrementally via `DatasetVersion` metadata (e.g., `v1`, `v2`).
- **Prompts**: Versioned templates sourced from the active database.
- **AI Configurations**: Hyperparameters, decoding configurations, and routing options.
- **Engines**: The software module versions (e.g., `validation_engine v1.1.2`).
Each generated paper stores these exact versions in its metadata fields for full historical auditability.

### 2.3 Comprehensive AI Call Auditing (`AICallLog`)
Every call made by an academic engine to any LLM, parser, or translation service is logged synchronously with:
- Target engine name and software version.
- Active prompt version ID.
- Exact input payload and returned model outputs.
- Latency (ms), model confidence scores, and validation check outcomes.
These audit logs provide a clean telemetry trail and double as high-quality episodic training inputs for future iterations of the AION Learning Factory.

### 2.4 Human Approval Gates for Learning Episodes
Before any user correction, teacher edit, or validation failure feedback is pushed into the active **Replay Buffer** for continual training, it must pass a human approval checklist. A designated administrator must approve the `LearningObject` in the ingestion queue (`approval_status` == `APPROVED`). This mitigates the risk of catastrophic forgetting or low-quality feedback loop amplification.

### 2.5 Strict Blueprint Locking
Before generation starts, the `PaperBlueprint` is frozen (`is_locked = True`). The generation and validation engines can modify candidate question formulations and answer wording, but are restricted from mutating:
- Maximum marks allocation.
- Selected Syllabus modules.
- Targets for Bloom levels.

---

## 3. Modular Intelligence Subsystems

### 3.1 The Question Discovery Engine
Rather than creating questions directly from text segments (RAG chunks), AION reverses the paradigm. The Question Discovery stage:
1. Translates reconstructed concepts into `KnowledgeObject` nodes and `KnowledgeGene` lists.
2. Forms an **Answer Graph** representing the optimal academic target answers.
3. Computes the **Examiner Intention** and matches it against potential Student Learning Objectives.
4. Identifies curriculum gaps and lists the most examinable questions implied by the topic nodes.
5. Ranks discovered questions prior to actual prompt token assembly.

### 3.2 Context-Expanding Failure Recovery Loops
If a generated paper or question fails any semantic, validation, or compliance gates (e.g., Bloom level mismatch, grammar score < 0.90, or syllabus deviation):
1. **Retrieve Again**: Re-query the Academic Genome and prerequisites maps using a broader semantic search window.
2. **Expand Context**: Inject neighboring nodes, related topic definitions, and historical question templates into the active prompt workspace.
3. **Reason Again**: Call the Examiner Reasoning Engine to re-evaluate the question design approach.
4. **Generate Again**: Produce a candidate question.
5. **Fallback Generator**: If validation still fails after 3 retries, swap to an emergency rule-based compiler or a highly-stable, smaller deterministic template builder.
6. **Manual Entry**: Trigger a fallback warning notifying the human reviewer for direct instruction only if all recovery routines fail.

---

## 4. The Complete Ingestion & Execution Pipeline

```text
Upload Book / Syllabus
  ↓
Document Intelligence (Text, OCR, tables, equations, structures)
  ↓
Academic Reconstruction (Construct semesters, course directories, subject files)
  ↓
Academic Genome & Answer Graph Construction (Genes mapping & prerequisite maps)
  ↓
Knowledge Validator (Identify contradictions or gaps in syllabus)
  ↓
Dataset Versioning (Increment training splits and version metadata)
  ↓
Exam Blueprint Locking (Freeze total marks, modules covered, Bloom distributions)
  ↓
Question Planner & Examiner Reasoning (Design outline for each question section)
  ↓
Question Discovery (Determine all implied question objectives from genome nodes)
  ↓
Academic Self-RAG (Validate candidates against textbook truth references)
  ↓
Context-Expanding Fail-safe Loops (Expand context & re-reason on validation failure)
  ↓
Validation & Discriminator Gates (Grammar, Bloom taxonomy, VTU style checks)
  ↓
Question Ranking & Formatting (Select highest quality candidates and compile layouts)
  ↓
Paper Compiler (Generate PDF, Word, LaTeX outputs)
  ↓
Review Loop (Reviewer approves or directly edits generated questions)
  ↓
Human Ingestion Review (Audit edits and save as approved Learning Episodes)
  ↓
Replay Buffer (Insert approved training episodes)
  ↓
Scheduled Retraining & Model Weights Promotion (Update active foundation weights)
```

---

## 5. Paper Workflow State Machine

The life-cycle of a question paper follows a strict state transition:
```
  [ Draft ] ──(Trigger Generation)──> [ Generating ]
                                            │
                                            │ (Generation & Validation Pass)
                                            ▼
  [ Approved ] <──(Human Review Pass)── [ Review ]
        │
        │ (Publish & Export)
        ▼
  [ Archived ]
```
Each state change is validated by the workflow gate controller.
