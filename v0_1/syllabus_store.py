"""
AION Syllabus Store & Appendix Relevance Matcher.
Stores subject syllabus topics and matches appendices against syllabus relevance.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

WORKSPACE_ROOT = Path("workspace")


class SyllabusStore:
    def __init__(self, subject_id: str):
        self.subject_id = subject_id
        self.path = WORKSPACE_ROOT / subject_id / "syllabus.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {
            "subject_name": "",
            "modules": [],
            "raw_syllabus": ""
        }

    def _save(self):
        self.path.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    def set_subject_name(self, name: str):
        self._data["subject_name"] = name
        self._save()

    def add_raw_syllabus(self, raw: str):
        """Parse a pasted syllabus text into modules."""
        self._data["raw_syllabus"] = raw
        self._data["modules"] = self._parse_syllabus(raw)
        self._save()

    def get_all_topics(self) -> list[str]:
        topics = []
        for mod in self._data["modules"]:
            topics.extend(mod.get("topics", []))
            topics.append(mod.get("title", ""))
        return [t for t in topics if t.strip()]

    def is_appendix_relevant(self, appendix_title: str) -> bool:
        """Check if an appendix title matches any syllabus topic."""
        topics = self.get_all_topics()
        if not topics:
            return False  # default: drop appendix if no syllabus
        title_lower = appendix_title.lower()
        for topic in topics:
            topic_words = set(re.findall(r"[a-z]{3,}", topic.lower()))
            title_words = set(re.findall(r"[a-z]{3,}", title_lower))
            if topic_words & title_words:
                return True
        return False

    def get_modules(self) -> list[dict]:
        return self._data.get("modules", [])

    def get_subject_name(self) -> str:
        return self._data.get("subject_name", "")

    def _parse_syllabus(self, raw: str) -> list[dict]:
        modules = []
        current = None

        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue

            m = re.match(
                r"^(module|unit|chapter)\s*[-:]?\s*(\d+)[:\s]+(.+)$",
                line, flags=re.I
            )
            if m:
                if current:
                    modules.append(current)
                current = {
                    "number": int(m.group(2)),
                    "title": m.group(3).strip(),
                    "topics": []
                }
                continue

            if current:
                cleaned = re.sub(r"^[-•*\d+\.\)]+\s*", "", line)
                if len(cleaned) > 4:
                    current["topics"].append(cleaned)

        if current:
            modules.append(current)

        return modules
