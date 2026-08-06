# AION Optimization Backlog — Deferred (Do NOT Implement Now)

> Per AION Development Context (READ FIRST), these ideas are intentionally deferred
> and documented for future implementation, NOT added in the current milestone.
> Current milestone is **stability, correctness, grounding, and production readiness**,
> not research experimentation.

---

## 1. Cache-Augmented Generation (CAG)

**Intent:** Cache validated academic reasoning and reduce repeated LLM inference.

**Deferred Work:**
- Cache layer between grounding and generation: validated `GroundedConcept → QuestionPlan → ComposedQuestion` tuples stored in Redis/Qdrant with hash of `concept_id + bloom + marks`.
- Cache-hit path: skip LLM call, return cached question with updated grounding hash verification.
- Cache invalidation on concept confidence decay or syllabus update.
- Metrics: cache hit rate, staleness, grounding drift.

**Why deferred:** Needs stable grounding first; premature caching would freeze hallucinations.

**Pluggable location:** `core/generation/question_composer.py` → add `CAGCache` wrapper before `self._llm.generate()`.

---

## 2. Context-Augmented Generation

**Intent:** Dynamically enrich prompts with validated contextual information beyond standard retrieval.

**Deferred Work:**
- Context builder that fuses: concept prerequisites graph + Bloom trajectory + examiner personality fingerprint + historical paper stats.
- Prompt enrichment: inject prerequisite concept definitions, common student mistakes, professor style signature.
- Requires Academic Knowledge Genome algebra to be stable.

**Pluggable location:** `core/concepts/grounding.py` → enrich `expected_answer` with prerequisite chain; `core/generation/question_composer.py` → extend prompt builder.

---

## 3. Advanced Academic Knowledge Genome Algebra

**Intent:** Formal genome operations: MERGE, DIFF, CROSS, DECAY.

**Deferred Operations:**
- **MERGE**: Combine two concept genomes (e.g., Module 3 + Module 4) into synthesis concept. Needs conflict resolution policy.
- **DIFF**: Compute genome difference between textbook editions; highlight curriculum drift.
- **CROSS**: Cross-domain genome product (e.g., CSE × ECE for "Embedded Systems" — verify via joint grounding).
- **DECAY**: Time-decay confidence for outdated concepts; re-validation trigger.

**Current state:** `core/concepts/extractor.py` produces `ExtractedConcept` with `prerequisites` and `relationships` as stubs. Genome storage in `memory/concepts.json` / Postgres pending.

**Deferred because:** Requires Postgres+Qdrant schema migration + training loops; stability first.

**Pluggable location:** New `core/genome/algebra.py` + `neural/graph_networks/genome_algebra.py`.

---

## 4. Metacognitive Monitoring Layer

**Intent:** Wrapping System & Knowledge Health monitor.

**Metrics to track (spec_v1.0 Stage 6 — Metacognitive Monitor):**
- Knowledge Completeness ratio
- Critic Rejection Rate (reason code histogram)
- Embedding Output Drift
- Confidence Calibration Error (expected vs actual validation)

**Deferred Work:**
- `neural/training/monitoring.py` exporter to MLflow + Streamlit Trace Viewer
- Alert on drift > threshold → trigger re-ingestion or human review.

**Pluggable location:** `core/pipeline/aion_pipeline.py` → emit `PipelineMetrics` to monitor; new `core/monitoring/metacognitive.py`.

---

## 5. Examiner Personality Modeling and Consistency Fingerprints

**Intent:** Examiner Personality Engine (EPE) with consistency fingerprint per professor/university.

**Deferred Work:**
- Fingerprint vector per examiner: verb distribution, numerical frequency, diagram frequency, difficulty curve, signature phrases.
- Training: `research/AION-Trainer` needs fine-tuning on VTU examiner datasets.
- Generation constraint: question must match fingerprint cosine similarity ≥ 0.85.

**Current stub:** `configs/generation/examiner.yaml` exists; `core/planning/question_planner.py` has `difficulty` but not full personality.

**Pluggable location:** `engines/examiner_reasoning_engine/` + `core/generation/question_composer.py` style conditioning.

---

## 6. Academic Thought Graph Traversal Policies

**Intent:** Stage 3.5 — ATG traversal between retrieval and generation to decide *what* to ask about.

**Policies (from architecture.md):**
- Depth-First (prerequisite chains)
- Breadth-First (concept connections)
- Bloom-Directed
- Examiner-Directed
- Socratic (cross-concept synthesis)

**Deferred Work:**
- Graph construction via `core/sdk/aom.py::ThoughtGraphIntent` + `engines/question_discovery_engine/`
- Traversal engine that outputs `ThoughtGraphIntent` spec for planner.

**Current stub:** `core/retrieval/concept_retriever.py` orders by confidence; ATG would replace naive ordering with policy-driven traversal.

**Pluggable location:** New `core/thought_graph/traversal.py` inserted between `grounding` and `planning` in `core/pipeline/aion_pipeline.py`.

---

## 7. Full Continual Learning and Self-Revision Pipeline

**Intent:** ALF orchestrator: faculty accept/reject → preference pairs → RAFT → GRPO → DPO replay buffer.

**Deferred Work:**
- `learning_factory/aion_learning_manager.py` wiring
- `aion-embeddings/core/learning/*` hot reload
- `aion_web/training/*` backend registry → training jobs

**Why deferred:** Requires faculty dashboard (Streamlit) + human-in-the-loop audit gate + model registry stability.

**Pluggable location:** `learning_factory/` + `api/review/` + `core/pipeline/aion_pipeline.py` post-audit export.

---

## Ordering for Future Milestones

1. **v0.2 (Next):** CAG + Context-Augmented (quick wins, no retraining)
2. **v0.3:** ATG Traversal + Examiner Fingerprint (needs curated datasets)
3. **v0.4:** Genome Algebra MERGE/DIFF (needs DB migration)
4. **v1.0:** Metacognitive + Continual Learning (needs faculty feedback loop)

---

## Guardrails for Future Work

- Any backlog item must NOT break the Universal Academic Pipeline contract: `Upload → Clean Text → Concepts → Grounding → Plan → Compose → Audit → Output`.
- Every item must keep `PRODUCTION_MODEL = "qwen2.5:7b"` invariant (no silent model swap).
- Every question must remain traceable: `Concept ID | Source chunk | Confidence | Expected answer | Bloom | Question`.
- External references only when `ConfidenceRecoveryEngine.allow_external=True` and flagged in output.
