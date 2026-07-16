# AION Architecture Specification (v1.0)

This document is the official, frozen architectural specification for **AION (Academic Intelligence Oriented Network)**. It defines the core philosophy, fundamental laws, database entities, internal pipelines, and validation protocols governing the system.

---

## 1. System Vision & Philosophy

AION is not a question paper generator, chatbot, or a wrapper around a traditional vector retrieval system (RAG).

> **AION is a self-evolving academic foundation model that continuously acquires, verifies, organizes, reasons over, and assesses academic knowledge while learning examiner behavior to generate institution-specific examination material with minimal human intervention.**

Rather than next-token token sequences or chunk-to-question mappings, AION optimizes for **Academic DNA representation** (concept mapping, Bloom taxonomic levels, and examiner intent).

---

## 2. The Four Fundamental Laws of AION

```
 ┌──────────────────────────────────────────────────────────┐
 │ LAW 1: Knowledge is never memorized; it is understood.   │
 └────────────────────────────┬─────────────────────────────┘
                              │
 ┌────────────────────────────▼─────────────────────────────┐
 │ LAW 2: Questions are never generated; they are discovered│
 └────────────────────────────┬─────────────────────────────┘
                              │
 ┌────────────────────────────▼─────────────────────────────┐
 │ LAW 3: Learning never stops.                             │
 └────────────────────────────┬─────────────────────────────┘
                              │
 ┌────────────────────────────▼─────────────────────────────┐
 │ LAW 4: Nothing is trusted immediately.                   │
 └──────────────────────────────────────────────────────────┘
```

- **Law 1: Understanding**: Text is broken down into semantic `KnowledgeObject` structures, mapping concepts, prerequisite structures, and relational graphs.
- **Law 2: Discovery**: Questions are discovered by analyzing syllabus assessment possibilities and ranking candidate objectives before generating prompts.
- **Law 3: Lifelong Adaptation**: Every review correction, teacher edit, model answer, and student score compiles into a `LearningObject` episode.
- **Law 4: Zero-Trust Ingestion**: Concepts must be cross-verified, confidence-scored, and compared against reference textbooks before insertion into the genome.

---

## 3. Core Subsystem Architectures

### 3.1 The Academic Cognitive Cycle (ACC)
AION replaces standard prompt-retrieval structures with the **ACC**, which runs continuously:
```
  Observe (Ingest files)
     ↓
  Understand (Construct Genome)
     ↓
  Verify (Zero-trust checks)
     ↓
  Reason (Simulate Examiner Intent)
     ↓
  Predict (Select testing objectives)
     ↓
  Generate (Create questions & rubrics)
     ↓
  Evaluate (Shadow Professor filter)
     ↓
  Learn (Capture Episode)
     ↓
  Repeat (Curriculum update)
```

### 3.2 The Academic Curiosity Network (ACN)
ACN makes the learning pipeline proactive rather than passive. ACN scans the active **Academic Genome** to identify:
- **Low-density concepts**: High-frequency curriculum headings with minimal verification data.
- **Topic updates**: Novel subject variants that appear in exam templates but are missing from internal textbook repositories.
When a gap is identified, ACN issues a **Learning Objective** to query external indexes, retrieve material, verify context, and trigger incremental training.

### 3.3 The Shadow Professor Ensemble
An invisible gatekeeper ensemble that validates every generated question prior to publication. It scores candidates against a locked quality threshold:

| Evaluator Metric | Evaluation Objective | Method |
| :--- | :--- | :--- |
| **Purity** | Verify candidate tests only concepts inside target blueprint modules. | Exact keyword/tag matching |
| **Taxonomy** | Confirm matching Bloom verb maps to the requested blueprint level. | Classifier prediction |
| **Plausibility** | Grade expected answers against textbook genomes. | Semantic Self-RAG score |
| **Styling** | Match university-specific guidelines (e.g. choice requirements). | Rule-engine verification |
| **Originality** | Compute similarity against historical exam papers database. | MinHash / Vector matching |

---

## 4. Academic Object Model (AOM) Schemas

All systems exchange data strictly using the schemas defined in [core/sdk/aom.py](file:///d:/AION/core/sdk/aom.py).

### 4.1 KnowledgeGene (Academic DNA Unit)
The primary unit of memory in AION representing a concept. It contains:
- `gene_id` & `knowledge_id`: Identifiers.
- `concept_name`: Concept title.
- `canonical_definition`: Approved reference definition.
- `alternative_definitions`: Equivalent explanations.
- `confidence_score`: Ingestion trust score (0.0 to 1.0).
- `prerequisites`: Prerequisite concept keys.
- `relationships`: Typed edges to other concepts.
- `bloom_progression`: Sequence of expected cognitive stages.
- `difficulty` & `exam_frequency`: Difficulty rating and occurrence tally.
- `recent_trends` & `professor_notes`: Custom trends and teaching notes.
- `typical_mistakes`: Student misconceptions and grading pitfalls.

### 4.2 PaperObject & State Transitions
Exam papers transition through a strict state machine:
- **DRAFT**: Blueprint locking and question planner initialize.
- **GENERATING**: In-progress generation and fail-safe recovery loops.
- **REVIEW**: Quality inspection by human faculty.
- **APPROVED**: Released for production export.
- **ARCHIVED**: Stored in telemetry databases for learning loop ingestion.

---

## 5. Ingestion & In-Loop Training Protocol

### 5.1 Fail-Safe Generation Recovery Loop
If a candidate question fails validation:
```
  Validation Failure
          │
          ▼
   Retrieve Again (Widen genome query)
          │
          ▼
   Expand Context (Inject prerequisite genes into prompt workspace)
          │
          ▼
   Reason & Generate Again
          │
  ┌───────┴───────┐
  │ Pass?         ├─(Yes)─> [Publish]
  └───────┬───────┘
          │ (No - Retry Limit Exceeded)
          ▼
   Fallback Generator (Deterministic template-based builder)
          │
          ▼
   Manual Alert (Human intervention fallback flag)
```

### 5.2 The Learning Ingestion Loop (ALF)
```
  Human Edit / Approved Paper
               │
               ▼
      Learning Episode
               │
               ▼
     [ Human Review Gate ] ──(Approved)──> [ Replay Buffer ]
                                                  │
                                                  ▼
                                         [ Scheduled Retraining ]
```
Nothing is written to the replay buffer or models without passing the human-in-the-loop audit gate.
