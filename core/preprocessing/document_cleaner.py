"""AION Document Cleaner"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class CleanedDocument:
    raw_text: str = ""
    cleaned_text: str = ""
    removed_line_count: int = 0
    original_line_count: int = 0
    doc_id: Optional[str] = None
    subject: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __init__(self, raw_text: str = "", cleaned_text: str = "",
                 removed_line_count: int = 0, original_line_count: int = 0,
                 doc_id: Optional[str] = None, subject: Optional[str] = None,
                 *args, **kwargs):
        self.raw_text = raw_text
        self.cleaned_text = cleaned_text or raw_text
        self.removed_line_count = removed_line_count
        self.original_line_count = original_line_count
        self.doc_id = doc_id
        self.subject = subject
        self.metadata = kwargs.get("metadata", {})
        for k, v in kwargs.items():
            setattr(self, k, v)


class DocumentCleaner:
    def clean(self, raw_text: str, subject: Optional[str] = None, **kwargs) -> CleanedDocument:
        lines = raw_text.splitlines()
        orig_count = len(lines)
        cleaned_lines = [l.strip() for l in lines if l.strip()]
        cleaned_text = "\n".join(cleaned_lines)
        removed_count = orig_count - len(cleaned_lines)
        return CleanedDocument(
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            removed_line_count=removed_count,
            original_line_count=orig_count,
            subject=subject,
            **kwargs
        )
