# aion_web/training/backends/demo_backend.py
"""
Demo Backend — simulates the full pipeline with realistic timing and
structured fake output. Suitable for:
    - UI demonstrations
    - Frontend development
    - Client presentations

Rules:
    - NEVER reads or processes real files (ignores file_paths entirely)
    - NEVER connects to any server or model
    - Produces deterministic fake data that looks realistic
    - Simulates realistic stage timing so the UI feels authentic
"""

from __future__ import annotations

import time
import uuid
import threading
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional

from aion_web.training.backends.base import (
    TrainingBackend, TrainingMode, BackendError,
    JobHandle, ProgressEvent, AnalysisOutput, TrainingOutput,
)


# ── Fake data constants ───────────────────────────────────────────────────────

FAKE_CONCEPTS = {
    1: ["Introduction to AI", "Intelligent Agents", "PEAS Framework", "Problem Formulation"],
    2: ["BFS", "DFS", "Uniform Cost Search", "A* Search", "Hill Climbing", "Beam Search"],
    3: ["Propositional Logic", "First Order Logic", "Unification", "Resolution"],
    4: ["Decision Trees", "Naive Bayes", "SVM", "k-NN", "Neural Networks"],
    5: ["CNN", "RNN", "LSTM", "Backpropagation", "Transfer Learning"],
}

FAKE_MODULES = [
    {"module_number": m, "title": t, "concept_count": len(FAKE_CONCEPTS[m]),
     "confidence": c, "sample_concepts": FAKE_CONCEPTS[m][:3]}
    for m, t, c in [
        (1, "Introduction to Artificial Intelligence", 0.99),
        (2, "Search Algorithms", 0.97),
        (3, "Knowledge Representation", 0.95),
        (4, "Machine Learning Fundamentals", 0.99),
        (5, "Deep Learning", 0.98),
    ]
]

FAKE_BENCHMARK_SCORES = {
    "grammar": 0.96,
    "academic_quality": 0.93,
    "bloom_accuracy": 0.91,
    "vtu_similarity": 0.94,
    "question_diversity": 0.89,
    "module_accuracy": 0.97,
    "expected_answer_quality": 0.92,
    "diagram_prediction": 0.88,
    "examiner_similarity_score": 0.91,
}

ANALYSIS_STAGES = [
    ("Reading documents...", 0.05),
    ("Document classification...", 0.15),
    ("Subject detection...", 0.25),
    ("TOC extraction...", 0.35),
    ("Module mapping...", 0.50),
    ("Concept estimation...", 0.65),
    ("Ambiguity detection...", 0.80),
    ("Building course preview...", 0.90),
    ("Analysis complete.", 1.00),
]

TRAINING_STAGES = [
    ("Academic Reconstruction", 0.08),
    ("Concept Detection", 0.18),
    ("Module Assignment", 0.28),
    ("Knowledge Merge", 0.38),
    ("Building Answer Graph", 0.50),
    ("Generating Training Samples", 0.62),
    ("Neural Training — Epoch 1/5", 0.70),
    ("Neural Training — Epoch 2/5", 0.74),
    ("Neural Training — Epoch 3/5", 0.78),
    ("Neural Training — Epoch 4/5", 0.82),
    ("Neural Training — Epoch 5/5", 0.86),
    ("Benchmarking candidate model...", 0.92),
    ("Registering candidate...", 0.97),
    ("Training complete.", 1.00),
]


class DemoBackend(TrainingBackend):
    """
    Simulates the full AION pipeline with fake data and realistic timing.
    """

    mode = TrainingMode.DEMO

    def __init__(self, stage_delay: float = 0.4):
        """
        stage_delay: seconds between simulated stages.
        Set to 0.0 in tests to make them instant.
        """
        self.stage_delay = stage_delay
        self._sessions: Dict[str, AnalysisOutput] = {}
        self._jobs: Dict[str, Dict] = {}
        self._active_job_id: Optional[str] = None
        self._lock = threading.Lock()

    # ── Core operations ───────────────────────────────────────────────

    def analyse(
        self,
        file_paths: List[str],
        subject_code: str = "",
    ) -> AnalysisOutput:
        # Simulate analysis delay (proportional to file count)
        file_count = max(len(file_paths), 1)
        for stage_msg, _ in ANALYSIS_STAGES:
            time.sleep(self.stage_delay * 0.5)

        session_id = f"DEMO-{uuid.uuid4().hex[:8].upper()}"

        output = AnalysisOutput(
            session_id=session_id,
            subject_code=subject_code or "BAI401",
            subject_name="Artificial Intelligence",
            department="AIML",
            semester=4,
            books=sum(1 for f in file_paths if "book" in f.lower() or "text" in f.lower()) or 1,
            notes=sum(1 for f in file_paths if "note" in f.lower() or "module" in f.lower()) or 2,
            question_banks=sum(1 for f in file_paths if "qb" in f.lower() or "bank" in f.lower()) or 1,
            previous_papers=sum(1 for f in file_paths if "pyq" in f.lower() or "paper" in f.lower()) or 0,
            module_summaries=FAKE_MODULES,
            ambiguities=[],
            train_enabled=True,
            mode=TrainingMode.DEMO,
        )
        self._sessions[session_id] = output
        return output

    def train(self, session_id: str, subject_code: str) -> JobHandle:
        with self._lock:
            job_id = f"DEMO-JOB-{uuid.uuid4().hex[:8].upper()}"
            self._jobs[job_id] = {
                "session_id": session_id,
                "subject_code": subject_code,
                "status": "running",
                "stages_done": 0,
                "started_at": datetime.utcnow().isoformat(),
            }
            self._active_job_id = job_id

        return JobHandle(
            job_id=job_id,
            mode=self.mode,
            subject_code=subject_code,
            started_at=self._jobs[job_id]["started_at"],
        )

    def get_progress(self, job_id: str) -> Generator[ProgressEvent, None, None]:
        if job_id not in self._jobs:
            yield ProgressEvent(
                message="Job not found.", fraction=0.0,
                is_terminal=True, is_error=True,
            )
            return

        for stage_msg, fraction in TRAINING_STAGES:
            time.sleep(self.stage_delay)

            # Inject realistic training metrics at the neural training stage
            metrics = {}
            if "Epoch" in stage_msg:
                epoch_num = int(stage_msg.split("Epoch")[1].split("/")[0].strip())
                metrics = {
                    "loss": round(1.8 - epoch_num * 0.28, 3),
                    "grammar": round(0.88 + epoch_num * 0.015, 3),
                    "bloom_accuracy": round(0.82 + epoch_num * 0.02, 3),
                    "vtu_similarity": round(0.85 + epoch_num * 0.018, 3),
                }

            is_terminal = fraction == 1.0
            if is_terminal:
                with self._lock:
                    self._jobs[job_id]["status"] = "completed"
                    self._active_job_id = None

            yield ProgressEvent(
                message=f"[Demo] {stage_msg}",
                fraction=fraction,
                stage=stage_msg,
                metrics=metrics,
                is_terminal=is_terminal,
                is_error=False,
            )

            if is_terminal:
                return

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"] = "cancelled"
                if self._active_job_id == job_id:
                    self._active_job_id = None
                return True
        return False

    def resolve_ambiguity(
        self,
        session_id: str,
        ambiguity_id: str,
        action: str,
        value: Any,
    ) -> Dict[str, Any]:
        # Demo has no real ambiguities; always succeeds
        return {
            "resolved": True,
            "ambiguity_id": ambiguity_id,
            "remaining_unresolved": 0,
            "train_enabled": True,
        }

    def confirm_course(self, session_id: str) -> bool:
        return True

    def health_check(self) -> Dict[str, Any]:
        return {
            "healthy": True,
            "details": "Demo mode — no external connections required.",
            "mode": self.mode.value,
        }

    @property
    def is_busy(self) -> bool:
        return self._active_job_id is not None
