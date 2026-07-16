# AION-Trainer/server/manifest.py
import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

class FileEntry:
    def __init__(self, path: str, file_hash: str):
        self.path = path
        self.hash = file_hash

    def to_dict(self) -> Dict[str, str]:
        return {"path": self.path, "hash": self.hash}

class SubjectManifest:
    def __init__(
        self,
        subject: str,
        version: str = "0.0",
        last_trained: Optional[str] = None,
        dataset_version: Optional[str] = None,
        books: Optional[List[FileEntry]] = None,
        notes: Optional[List[FileEntry]] = None,
        previous_papers: Optional[List[FileEntry]] = None,
    ):
        self.subject = subject
        self.version = version
        self.last_trained = last_trained
        self.dataset_version = dataset_version
        self.books = books or []
        self.notes = notes or []
        self.previous_papers = previous_papers or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "version": self.version,
            "last_trained": self.last_trained,
            "dataset_version": self.dataset_version,
            "books": [b.to_dict() for b in self.books],
            "notes": [n.to_dict() for n in self.notes],
            "previous_papers": [p.to_dict() for p in self.previous_papers],
        }

class ManifestDiff:
    def __init__(self, added: List[FileEntry], modified: List[FileEntry], removed: List[FileEntry]):
        self.added = added
        self.modified = modified
        self.removed = removed

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.modified or self.removed)

    @property
    def total_changed(self) -> int:
        return len(self.added) + len(self.modified) + len(self.removed)

class ManifestManager:
    def __init__(self, academic_root: str):
        self.academic_root = Path(academic_root)

    def find_subject_path(self, subject: str) -> Optional[Path]:
        for root, dirs, files in os.walk(self.academic_root):
            for d in dirs:
                if d == subject:
                    return Path(root) / d
        return None

    def _hash_file(self, filepath: Path) -> str:
        h = hashlib.md5()
        try:
            with open(filepath, "rb") as f:
                while chunk := f.read(8192):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            return ""

    def scan(self, subject: str) -> SubjectManifest:
        subject_path = self.find_subject_path(subject)
        if not subject_path:
            return SubjectManifest(subject=subject)

        books = []
        notes = []
        papers = []

        supported_exts = (".pdf", ".docx", ".pptx", ".png", ".jpg")

        def scan_category_dir(category_dir: Path, target_list: List[FileEntry]):
            if category_dir.exists():
                for root, _, files in os.walk(category_dir):
                    for file in files:
                        p = Path(root) / file
                        if p.suffix.lower() in supported_exts:
                            rel_path = os.path.relpath(p, self.academic_root).replace("\\", "/")
                            file_hash = self._hash_file(p)
                            target_list.append(FileEntry(path=rel_path, file_hash=file_hash))

        scan_category_dir(subject_path / "textbooks", books)
        scan_category_dir(subject_path / "notes", notes)
        scan_category_dir(subject_path / "previous_papers", papers)

        manifest_json_path = subject_path / "manifest.json"
        version = "0.0"
        last_trained = None
        dataset_version = None
        if manifest_json_path.exists():
            try:
                with open(manifest_json_path, encoding="utf-8") as f:
                    data = json.load(f)
                    version = data.get("version", "0.0")
                    last_trained = data.get("last_trained")
                    dataset_version = data.get("dataset_version")
            except Exception:
                pass

        return SubjectManifest(
            subject=subject,
            version=version,
            last_trained=last_trained,
            dataset_version=dataset_version,
            books=books,
            notes=notes,
            previous_papers=papers,
        )

    def diff(self, old: Optional[SubjectManifest], new: SubjectManifest) -> ManifestDiff:
        if not old:
            return ManifestDiff(
                added=new.books + new.notes + new.previous_papers,
                modified=[],
                removed=[],
            )

        def build_map(manifest: SubjectManifest) -> Dict[str, str]:
            entries = manifest.books + manifest.notes + manifest.previous_papers
            return {e.path: e.hash for e in entries}

        old_map = build_map(old)
        new_map = build_map(new)

        all_new_entries = new.books + new.notes + new.previous_papers
        all_old_entries = old.books + old.notes + old.previous_papers

        added = [e for e in all_new_entries if e.path not in old_map]
        modified = [e for e in all_new_entries if e.path in old_map and old_map[e.path] != e.hash]
        removed = [e for e in all_old_entries if e.path not in new_map]

        return ManifestDiff(added=added, modified=modified, removed=removed)

    def save(self, manifest: SubjectManifest) -> str:
        subject_path = self.find_subject_path(manifest.subject)
        if not subject_path:
            subject_path = self.academic_root / "AIML" / "semester_4" / manifest.subject
            for sub in ["textbooks", "notes", "question_bank", "previous_papers", "answer_keys", "syllabus", "images"]:
                (subject_path / sub).mkdir(parents=True, exist_ok=True)

        current_version = float(manifest.version)
        new_version = f"{round(current_version + 0.1, 1)}"
        manifest.version = new_version

        manifest_json_path = subject_path / "manifest.json"
        with open(manifest_json_path, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2)

        return new_version

    def load_manifest(self, subject: str) -> Optional[SubjectManifest]:
        subject_path = self.find_subject_path(subject)
        if not subject_path:
            return None
        manifest_json_path = subject_path / "manifest.json"
        if not manifest_json_path.exists():
            return None
        try:
            with open(manifest_json_path, encoding="utf-8") as f:
                data = json.load(f)
                return SubjectManifest(
                    subject=data["subject"],
                    version=data.get("version", "0.0"),
                    last_trained=data.get("last_trained"),
                    dataset_version=data.get("dataset_version"),
                    books=[FileEntry(b["path"], b["hash"]) for b in data.get("books", [])],
                    notes=[FileEntry(n["path"], n["hash"]) for n in data.get("notes", [])],
                    previous_papers=[FileEntry(p["path"], p["hash"]) for p in data.get("previous_papers", [])],
                )
        except Exception:
            return None

    def get_or_create(self, subject: str) -> Tuple[Optional[SubjectManifest], SubjectManifest, ManifestDiff]:
        old = self.load_manifest(subject)
        new = self.scan(subject)
        diff = self.diff(old, new)
        return old, new, diff
