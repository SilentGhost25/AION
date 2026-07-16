# AION-Trainer/acb/source_registry.py
"""
Source Registry — every document that enters AION gets a quality
profile. The concept merger uses source reliability scores to decide
which definition to prefer when two sources disagree.
"""

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


class SourceType:
    TEXTBOOK = "textbook"
    NOTES = "notes"
    QUESTION_BANK = "question_bank"
    PREVIOUS_PAPER = "previous_paper"
    ANSWER_KEY = "answer_key"
    SYLLABUS = "syllabus"
    IMAGES = "images"


# Default reliability floor by source type —
# adjusted upward by quality signals, never above 1.0
DEFAULT_RELIABILITY: Dict[str, float] = {
    SourceType.TEXTBOOK: 0.90,
    SourceType.NOTES: 0.75,
    SourceType.QUESTION_BANK: 0.80,
    SourceType.PREVIOUS_PAPER: 0.85,
    SourceType.ANSWER_KEY: 0.88,
    SourceType.SYLLABUS: 1.00,        # Syllabus is always authoritative
    SourceType.IMAGES: 0.70,
}


@dataclass
class SourceQualityProfile:
    """Quality assessment for one uploaded document."""
    source_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    file_path: str = ""
    file_name: str = ""
    source_type: str = SourceType.TEXTBOOK
    subject_code: str = ""

    # Raw quality signals
    academic_depth: float = 0.8       # 0–1: shallow overview vs comprehensive treatment
    coverage: float = 0.8             # 0–1: fraction of module topics mentioned
    grammar_quality: float = 0.9      # 0–1: OCR / writing quality
    diagram_quality: float = 0.8      # 0–1: presence and quality of figures
    professor_rating: float = 0.0     # 0–1: explicit faculty rating (0 = not yet rated)

    # Computed
    reliability: float = 0.8          # weighted composite — filled by _compute_reliability
    total_pages: int = 0
    concepts_extracted: int = 0

    # Provenance
    uploaded_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    hash: str = ""

    def compute_reliability(self) -> float:
        base = DEFAULT_RELIABILITY.get(self.source_type, 0.75)
        quality_bonus = (
            self.academic_depth * 0.25 +
            self.coverage * 0.20 +
            self.grammar_quality * 0.20 +
            self.diagram_quality * 0.10
        )
        professor_boost = self.professor_rating * 0.25 if self.professor_rating > 0 else 0.0
        self.reliability = min(1.0, base * 0.5 + quality_bonus * 0.5 + professor_boost)
        return self.reliability

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SourceQualityProfile":
        return cls(**data)


class SourceRegistry:
    def __init__(self, registry_path: str = None):
        self._sources: Dict[str, SourceQualityProfile] = {}
        self.registry_path = Path(registry_path) if registry_path else None

    def register(self, profile: SourceQualityProfile) -> SourceQualityProfile:
        profile.compute_reliability()
        self._sources[profile.source_id] = profile
        return profile

    def create_and_register(
        self, file_path: str, source_type: str, subject_code: str
    ) -> SourceQualityProfile:
        import hashlib
        file_bytes = Path(file_path).read_bytes() if Path(file_path).exists() else b""
        profile = SourceQualityProfile(
            file_path=file_path,
            file_name=Path(file_path).name,
            source_type=source_type,
            subject_code=subject_code,
            hash=hashlib.sha256(file_bytes).hexdigest()[:16],
        )
        return self.register(profile)

    def get(self, source_id: str) -> Optional[SourceQualityProfile]:
        return self._sources.get(source_id)

    def reliability(self, source_id: str) -> float:
        src = self._sources.get(source_id)
        return src.reliability if src else 0.5

    def all_sources(self) -> List[SourceQualityProfile]:
        return list(self._sources.values())

    def save(self, path: str = None):
        target = Path(path or self.registry_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = {sid: s.to_dict() for sid, s in self._sources.items()}
        target.write_text(json.dumps(data, indent=2, default=str))

    def load(self, path: str = None):
        target = Path(path or self.registry_path)
        if not target.exists():
            return
        data = json.loads(target.read_text())
        for sid, s_dict in data.items():
            self._sources[sid] = SourceQualityProfile.from_dict(s_dict)
