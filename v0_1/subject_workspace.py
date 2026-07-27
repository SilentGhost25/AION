"""
AION Subject Workspace Manager.
Manages per-subject uploads, file classification, and aggregated extraction.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from .material_classifier import ClassificationResult, classify_material

WORKSPACE_ROOT = Path("workspace")

TYPE_FOLDERS = {
    "textbook":      "textbooks",
    "notes":         "notes",
    "question_bank": "question_banks",
    "slides":        "slides",
    "unknown":       "other",
}


@dataclass
class FileRecord:
    file_id: str
    original_name: str
    stored_path: str
    material_type: str
    classification_confidence: float
    classification_signals: List[str]
    extracted_path: Optional[str]
    word_count: int
    pages_kept: int
    pages_total: int
    filter_method: str
    uploaded_at: str
    extracted_at: Optional[str]
    status: str


class SubjectWorkspace:
    def __init__(self, subject_id: str, subject_name: str = ""):
        self.subject_id = subject_id
        self.subject_name = subject_name
        self.root = WORKSPACE_ROOT / subject_id
        self._init_dirs()
        self._meta_path = self.root / "workspace.json"
        self._meta = self._load_meta()
        if subject_name:
            self._meta["subject_name"] = subject_name
            self._save_meta()

    def _init_dirs(self):
        self.root.mkdir(parents=True, exist_ok=True)
        for folder in TYPE_FOLDERS.values():
            (self.root / "uploads" / folder).mkdir(parents=True, exist_ok=True)
        (self.root / "extracted").mkdir(exist_ok=True)

    def _load_meta(self) -> dict:
        if self._meta_path.exists():
            return json.loads(self._meta_path.read_text(encoding="utf-8"))
        return {
            "subject_id": self.subject_id,
            "subject_name": self.subject_name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "files": {}
        }

    def _save_meta(self):
        self._meta_path.write_text(
            json.dumps(self._meta, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    def add_file(
        self,
        src_path: str,
        override_type: Optional[str] = None
    ) -> FileRecord:
        """
        Copy file into workspace, classify it, return FileRecord.
        """
        src = Path(src_path)
        result: ClassificationResult = classify_material(src_path)

        mat_type = override_type if override_type else result.material_type
        folder = TYPE_FOLDERS.get(mat_type, "other")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_id = f"{ts}_{src.stem[:30]}"
        dest = self.root / "uploads" / folder / f"{file_id}{src.suffix}"
        shutil.copy2(src_path, dest)

        record = FileRecord(
            file_id=file_id,
            original_name=src.name,
            stored_path=str(dest),
            material_type=mat_type,
            classification_confidence=result.confidence,
            classification_signals=result.signals,
            extracted_path=None,
            word_count=0,
            pages_kept=0,
            pages_total=0,
            filter_method="pending",
            uploaded_at=datetime.now().isoformat(timespec="seconds"),
            extracted_at=None,
            status="uploaded"
        )
        self._meta["files"][file_id] = asdict(record)
        self._save_meta()
        return record

    def update_extraction(self, file_id: str, extracted_path: str, report: dict):
        if file_id not in self._meta["files"]:
            return
        rec = self._meta["files"][file_id]
        rec["extracted_path"] = extracted_path
        rec["word_count"] = report.get("kept_word_count", 0)
        rec["pages_kept"] = len(report.get("kept_pages", []))
        rec["pages_total"] = report.get("total_pages", 0)
        rec["filter_method"] = report.get("method", "")
        rec["extracted_at"] = datetime.now().isoformat(timespec="seconds")
        rec["status"] = "extracted"
        self._save_meta()

    def mark_failed(self, file_id: str, reason: str):
        if file_id in self._meta["files"]:
            self._meta["files"][file_id]["status"] = f"failed: {reason}"
            self._save_meta()

    def get_files(self, material_type: Optional[str] = None) -> List[FileRecord]:
        records = []
        for rec_dict in self._meta["files"].values():
            if material_type and rec_dict["material_type"] != material_type:
                continue
            records.append(FileRecord(**rec_dict))
        return sorted(records, key=lambda r: r.uploaded_at, reverse=True)

    def get_all_extracted_text(self, material_type: Optional[str] = None) -> str:
        """
        Concatenate extracted text across all files of the given type.
        """
        parts = []
        for rec in self.get_files(material_type):
            if rec.status == "extracted" and rec.extracted_path:
                ep = Path(rec.extracted_path)
                if ep.exists():
                    header = (
                        f"\n\n{'='*60}\n"
                        f"SOURCE: {rec.original_name} [{rec.material_type.upper()}]\n"
                        f"{'='*60}\n"
                    )
                    parts.append(header + ep.read_text(encoding="utf-8", errors="ignore"))
        return "\n".join(parts)

    def get_subject_name(self) -> str:
        return self._meta.get("subject_name", self.subject_id)
