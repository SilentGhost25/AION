# AION Integrated Architecture Blueprint (Final Synthesis)

This document freezes the integrated architecture of **AION (Academic Intelligence Oriented Network)**, unifying **Design B's production engineering skeleton** with **Design A's cognitive reasoning brain**.

---

## 1. Integrated System Architecture Topology

AION combines a production-grade ingestion, retrieval, and training stack with an explicit academic reasoning layer:

```text
DOCUMENT INPUT (PDF / Textbook / Syllabus)
     │
     ▼
┌──────────────────────────────────────────────────────────────┐
│ STAGE 1 — DOCUMENT UNDERSTANDING                            │
│ baidu/Unlimited-OCR + docling-models + table-transformer      │
│ → Hierarchical JSON per textbook                             │
│ ★ Diagram Intelligence Layer: Figure extraction to queryable │
│   academic objects (nodes, relationships, learning obj)     │
└──────────────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────────────┐
│ STAGE 2 — KNOWLEDGE EXTRACTION                               │
│ Hierarchical Chunking + GLiNER-relex → Concepts + Relations   │
│ → Postgres Academic Knowledge Genome (DNA Strands)           │
│ ★ Genome Algebra: MERGE, DIFF, CROSS, DECAY operations      │
└──────────────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────────────┐
│ STAGE 3 — HYBRID RETRIEVAL & INDEXING                        │
│ Dense: BAAI/bge-m3 | Sparse: BM25 | Rerank: bge-reranker-v2-m3│
│ Vector: Qdrant | Multimodal Figures: Qwen3-VL-2B             │
└──────────────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────────────┐
│ STAGE 3.5 — ACADEMIC THOUGHT GRAPH (ATG) TRAVERSAL            │
│ Inputs: Retrieved Chunks + Concept Genomes                   │
│ Traversal Policies: Depth-First | Breadth-First | Socratic |   │
│                    Bloom-Directed | Examiner-Directed        │
│ Output: Stage 3.5 ThoughtGraphIntent Specification            │
└──────────────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────────────┐
│ STAGE 4 — RAG² ANSWER-FIRST GENERATION                       │
│ Base LLM: Qwen3-8B-Instruct + Multi-LoRA (vLLM serving)      │
│ Sequence: 1. Ideal Answer -> 2. Marking Scheme ->            │
│           3. Question Constraints -> 4. Exam Question        │
│ ★ Examiner Personality Engine (EPE): Consistency Fingerprint │
└──────────────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────────────┐
│ STAGE 5 — SELF-CRITIC ENSEMBLE & GUARDRAILS                  │
│ Gates: Faithfulness (LettuceDetect) | Originality (MinHash)  │
│ Reason Codes: RC-01 (Concept) | RC-02 (Rel) | RC-03 (EPE)   │
│              RC-04 (Bloom)   | RC-05 (Ped) | RC-06 (Grammar)│
│              RC-07 (Abstain Retrieval Path)                  │
│ Causal Failure Routing: Failures route back to failing stage │
└──────────────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────────────┐
│ STAGE 6 — HUMAN REVIEW & CONTINUAL TRAINING (ALF)            │
│ Interface: Streamlit Faculty Dashboard (Accept/Reject/Edit)  │
│ Dual Impact:                                                 │
│  1. Updates Genome Confidence DNA (Knowledge Evolution Graph)│
│  2. Logs Preference Pairs into RAFT -> GRPO -> DPO Buffer    │
└──────────────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────────────┐
│ METACOGNITIVE MONITOR (Wrapping System & Knowledge Health)   │
│ Tracks: Knowledge Completeness | Critic Rejection Rate |     │
│         Reason Code Histogram | Embedding Output Drift       │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. Integrated Model Stack

| Pipeline Stage | Model / Component | Purpose in Integrated AION |
|---|---|---|
| **OCR & Parsing** | `baidu/Unlimited-OCR` | High-fidelity textbook OCR extraction |
| **Document Structure** | `docling-project/docling-models` | Hierarchical JSON tree & figure extraction |
| **Table Verification** | `microsoft/table-transformer` | Structural header & row validation |
| **Text Embedding** | `BAAI/bge-m3` | Dense & sparse hybrid vector embedding |
| **Reranking** | `BAAI/bge-reranker-v2-m3` | Cross-encoder retrieval reranking |
| **Multimodal Figures** | `Qwen3-VL-2B` | Image/diagram semantic embedding & OCR |
| **Concept Extraction** | `knowledgator/gliner-relex` | Zero-shot concept & relation extraction -> Genome DNA |
| **Generation Engine** | `Qwen3-8B-Instruct` + LoRA | RAG² Answer-First Generation & Examiner Personality |
| **Serving & Inference** | `vLLM` (AWQ/FP8, Multi-LoRA) | Production high-throughput multi-personality serving |
| **Faithfulness Gate** | `LettuceDetect` / `HHEM` | Token-level grounding verification (RC-01) |
| **Originality Gate** | `MinHash-LSH` | N-gram & embedding novelty check (RC-05) |
| **Observability** | `MLflow` + Streamlit Trace Viewer | System & metacognitive monitoring telemetry |

---

## 3. Integrated 3-Stage Training Pipeline

### Stage 1: RAFT-Formatted SFT (Retrieval-Augmented Fine-Tuning)
- **Data Composition**: $P \approx 0.75$ golden context inclusion, $0.25$ distractor-only context.
- **Thought Traversal**: Chain-of-thought traces explicitly follow **Academic Thought Graph** node traversals (`Concept -> Prerequisite -> Relationship -> Expected Answer`).

### Stage 2: GRPO (Group Relative Policy Optimization)
- Candidates per prompt ($G \ge 4$).
- **Integrated 7-Signal Reward Function**:
  $$\text{Reward} = w_1 \cdot R_{\text{faithfulness}} + w_2 \cdot R_{\text{originality}} + w_3 \cdot R_{\text{judge}} + w_4 \cdot R_{\text{fingerprint}} + w_5 \cdot R_{\text{bloom}} + w_6 \cdot R_{\text{simulation}} + w_7 \cdot R_{\text{atg}} - w_8 \cdot P_{\text{format}}$$
  - $w_1 = 0.25$ (LettuceDetect Grounding)
  - $w_2 = 0.15$ (MinHash Novelty)
  - $w_3 = 0.15$ (RULER LLM Judge)
  - $w_4 = 0.15$ (EPE Consistency Fingerprint Match)
  - $w_5 = 0.10$ (Bloom Taxonomy Target Match)
  - $w_6 = 0.10$ (Concept Simulation Consistency)
  - $w_7 = 0.10$ (ATG Traversal Validity)
  - $w_8 = 0.20$ (Format Violation Penalty)

### Stage 3: DPO (Direct Preference Optimization)
- Preference pairs constructed from real faculty accept/reject/edit decisions in Stage 6.

---

## 4. The 8 Validation Checkpoints

1. **Checkpoint 1 (Wk 2-4)**: Document understanding quality (Unlimited-OCR & Docling accuracy $\ge 95\%$).
2. **Checkpoint 2 (Wk 4-6)**: Concept extraction sanity (GLiNER precision $\ge 90\%$).
3. **Checkpoint 3 (Wk 7)**: Baseline RAG² generation quality (zero-shot Qwen3-8B).
4. **Checkpoint 4 (Wk 10)**: Fine-tuned RAFT vs. baseline blind review ($\ge 80\%$ faculty preference).
5. **Checkpoint 5 (Ongoing)**: Anti-memorization & originality audit ($< 5\%$ source overlap).
6. **Checkpoint 6 (Wk 5-6)**: Genome integrity & Genome Algebra verification (MERGE/CROSS ops).
7. **Checkpoint 7 (Wk 8-9)**: ATG traversal validity (Socratic traversal yields cross-concept questions).
8. **Checkpoint 8 (Wk 11)**: Cognitive reward correlation ($\text{Reward}_i$ correlates with faculty acceptance).
