# AION-Trainer/training_studio/analyser/analysis_result.py
"""
Analysis Result — the complete structured output of the Analyse stage.
This is what the UI reads to render the Course Preview and the
Ambiguity Review screen.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any


class AmbiguitySeverity:
    ERROR = "error"             # Must be resolved before training
    WARNING = "warning"         # Should be reviewed but won't block
    INFO = "info"               # Informational only


@dataclass
class FileAnalysis:
    """Analysis result for one uploaded file."""
    file_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    filename: str = ""
    file_path: str = ""
    file_size_bytes: int = 0

    # Classification
    document_type: str = ""
    type_confidence: float = 0.0
    type_needs_confirmation: bool = False
    type_alternatives: List[str] = field(default_factory=list)
    type_signals: List[str] = field(default_factory=list)

    # Subject
    subject_code: str = ""
    subject_name: str = ""
    subject_confidence: float = 0.0
    subject_needs_confirmation: bool = False

    # Modules
    module_mappings: List[Dict[str, Any]] = field(default_factory=list)
    ambiguous_chapters: List[Dict[str, Any]] = field(default_factory=list)

    # Concepts found (lightweight preview — full extraction during Train)
    estimated_concept_count: int = 0
    sample_concepts: List[str] = field(default_factory=list)

    # Status
    status: str = "pending"        # pending | analysing | complete | error
    error_message: str = ""
    analysis_time_seconds: float = 0.0

    @property
    def is_ready(self) -> bool:
        return (
            self.status == "complete"
            and not self.type_needs_confirmation
            and not self.subject_needs_confirmation
        )


@dataclass
class Ambiguity:
    """One ambiguity that needs human attention."""
    ambiguity_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    severity: str = AmbiguitySeverity.WARNING
    file_id: str = ""
    filename: str = ""
    title: str = ""
    description: str = ""
    options: List[Dict[str, Any]] = field(default_factory=list)
    """
    options: list of choices, e.g.:
        [
            {"label": "Yes, move to Module 2", "action": "assign_module", "value": 2},
            {"label": "Keep as Module 1", "action": "assign_module", "value": 1},
            {"label": "Skip this chapter", "action": "skip"},
        ]
    """
    resolved: bool = False
    resolution: Optional[Dict[str, Any]] = None


@dataclass
class ModulePreview:
    """One module's preview data for the Course Preview tree."""
    module_number: int
    title: str
    concept_count: int
    confidence: float
    sample_concepts: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)         # filenames contributing
    bloom_coverage: Dict[str, bool] = field(default_factory=dict)
    has_diagrams: bool = False


@dataclass
class SessionAnalysisResult:
    """Complete analysis result for a Training Studio session."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Detected academic context
    subject_code: str = ""
    subject_name: str = ""
    department: str = ""
    semester: int = 0

    # Per-file results
    file_analyses: List[FileAnalysis] = field(default_factory=list)

    # Aggregated ambiguities
    ambiguities: List[Ambiguity] = field(default_factory=list)

    # Module previews
    module_previews: List[ModulePreview] = field(default_factory=list)

    # Counts
    total_files: int = 0
    books_detected: int = 0
    notes_detected: int = 0
    qb_detected: int = 0
    pyq_detected: int = 0

    # Readiness
    analysis_complete: bool = False
    has_unresolved_errors: bool = False
    train_enabled: bool = False

    def unresolved_ambiguities(self) -> List[Ambiguity]:
        return [a for a in self.ambiguities if not a.resolved]

    def unresolved_errors(self) -> List[Ambiguity]:
        return [
            a for a in self.ambiguities
            if not a.resolved and a.severity == AmbiguitySeverity.ERROR
        ]

    def compute_readiness(self):
        """Re-evaluate train_enabled after each ambiguity resolution."""
        self.has_unresolved_errors = len(self.unresolved_errors()) > 0
        all_files_ready = all(
            fa.status == "complete" for fa in self.file_analyses
        )
        self.train_enabled = (
            self.analysis_complete
            and all_files_ready
            and not self.has_unresolved_errors
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)
