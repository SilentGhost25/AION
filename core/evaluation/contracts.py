"""
AION RAG Evaluation Contracts
=============================
Typed contracts for real-time RAG-inspired question evaluation metrics,
provenance tracking, and aggregate paper summaries.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class MetricProvenance:
    """Complete provenance and lineage metadata for every metric evaluation."""
    trace_id: str = ""
    generation_id: str = ""
    slot_id: str = ""
    spec_id: str = ""
    module_idx: int = 1
    ku_id: str = ""
    bundle_id: str = ""
    model_name: str = ""
    provider: str = ""
    attempt_index: int = 1
    pipeline_version: str = "v1.1.0"
    git_commit: str = ""

    def __post_init__(self):
        if not self.git_commit:
            self.git_commit = os.getenv("AION_COMMIT_HASH", "mod_eval")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RealtimeRAGMetrics:
    """
    RAGAS-inspired, deterministic real-time evaluation metrics for a single generated question.
    Observes generation without deciding acceptance or rejecting questions.
    """
    # Core RAG Triad Metrics (0.0 to 1.0)
    faithfulness: float = 1.0           # Evidence grounding accuracy, absence of hallucinated tokens/figures
    context_recall: float = 1.0         # Proportion of required evidence components represented
    question_relevance: float = 1.0     # Alignment with Bloom level, marks demand profile, and archetype
    harmonic_rag_score: float = 1.0     # Correct harmonic mean of [faithfulness, context_recall, question_relevance]

    # Independent Quality & Solvability Dimensions
    hallucination_rate: float = 0.0     # Independently computed: hallucinated_claims / max(1, total_claims)
    equation_fidelity: float = 1.0      # Syntax validity of LaTeX equations, parameter preservation
    archetype_adherence: float = 1.0    # Structural adherence to slot question type
    question_validity: float = 1.0      # Syntax, completeness, absence of meta-tokens or answer leaks

    # Operational Performance Telemetry
    latency_ms: float = 0.0             # Inference wall-clock time
    context_tokens: int = 0             # Approximate prompt context tokens
    generation_tokens: int = 0          # Approximate tokens generated
    tokens_per_sec: float = 0.0         # Generation throughput

    # Status & Audit Findings
    status: str = "PASS"                # "PASS", "WARNING", "FAIL"
    hallucination_findings: List[str] = field(default_factory=list)
    audit_notes: List[str] = field(default_factory=list)

    # Framework & Lineage
    evaluation_framework: str = "aion_ragas_inspired"
    provenance: MetricProvenance = field(default_factory=MetricProvenance)

    @property
    def answer_relevance_proxy(self) -> float:
        """Alias for compatibility with external RAGAS nomenclature."""
        return self.question_relevance

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "faithfulness": round(self.faithfulness, 4),
            "context_recall": round(self.context_recall, 4),
            "question_relevance": round(self.question_relevance, 4),
            "answer_relevance_proxy": round(self.question_relevance, 4),
            "harmonic_rag_score": round(self.harmonic_rag_score, 4),
            "hallucination_rate": round(self.hallucination_rate, 4),
            "equation_fidelity": round(self.equation_fidelity, 4),
            "archetype_adherence": round(self.archetype_adherence, 4),
            "question_validity": round(self.question_validity, 4),
            "latency_ms": round(self.latency_ms, 2),
            "context_tokens": self.context_tokens,
            "generation_tokens": self.generation_tokens,
            "tokens_per_sec": round(self.tokens_per_sec, 2),
            "status": self.status,
            "hallucination_findings": list(self.hallucination_findings),
            "audit_notes": list(self.audit_notes),
            "evaluation_framework": self.evaluation_framework,
            "provenance": self.provenance.to_dict() if isinstance(self.provenance, MetricProvenance) else self.provenance,
        }
        return d


@dataclass
class PaperRAGSummary:
    """Aggregated RAG quality and operational performance summary across a full exam paper."""
    mean_faithfulness: float = 1.0
    mean_context_recall: float = 1.0
    mean_question_relevance: float = 1.0
    mean_rag_score: float = 1.0
    mean_hallucination_rate: float = 0.0
    mean_equation_fidelity: float = 1.0

    mean_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    max_latency_ms: float = 0.0

    total_context_tokens: int = 0
    total_generation_tokens: int = 0
    avg_tokens_per_sec: float = 0.0
    questions_evaluated: int = 0

    evaluation_framework: str = "aion_ragas_inspired"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_faithfulness": round(self.mean_faithfulness, 4),
            "mean_context_recall": round(self.mean_context_recall, 4),
            "mean_question_relevance": round(self.mean_question_relevance, 4),
            "mean_rag_score": round(self.mean_rag_score, 4),
            "mean_hallucination_rate": round(self.mean_hallucination_rate, 4),
            "mean_equation_fidelity": round(self.mean_equation_fidelity, 4),
            "mean_latency_ms": round(self.mean_latency_ms, 2),
            "p50_latency_ms": round(self.p50_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "p99_latency_ms": round(self.p99_latency_ms, 2),
            "max_latency_ms": round(self.max_latency_ms, 2),
            "total_context_tokens": self.total_context_tokens,
            "total_generation_tokens": self.total_generation_tokens,
            "avg_tokens_per_sec": round(self.avg_tokens_per_sec, 2),
            "questions_evaluated": self.questions_evaluated,
            "evaluation_framework": self.evaluation_framework,
        }
