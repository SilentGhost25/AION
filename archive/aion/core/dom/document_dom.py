"""
AION v2 Hierarchical Document Object Model (DOM)
=================================================
The canonical academic syllabus representation. Replaces all flat text representations.
Provides rigorous page accounting, module/section scoping, and direct query APIs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from aion.core.dom.artifacts import (
    ArtifactType,
    BaseArtifact,
    create_artifact_from_dict,
)
from aion.core.dom.artifact_registry import ArtifactRegistry
from aion.core.dom.relationships import (
    RelationshipGraph,
    RelationshipType,
)


class PageState(str, Enum):
    EXTRACTED = "EXTRACTED"           # Native text and artifacts extracted with high confidence
    PARTIAL = "PARTIAL"               # Partially extracted; minor warnings
    OCR_REQUIRED = "OCR_REQUIRED"     # Scanned or image-only page requiring OCR
    FAILED = "FAILED"                 # Parsing or extraction error encountered
    EMPTY_EXPECTED = "EMPTY_EXPECTED" # Intentional blank page / divider sheet


@dataclass
class PageAuditRecord:
    """
    Guarantees strict page accounting. Every page in the source document
    must have a recorded audit record to prevent silent page loss.
    """
    page_number: int
    state: PageState
    char_count: int = 0
    artifact_count: int = 0
    artifact_ids: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    ocr_applied: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "state": self.state.value,
            "char_count": self.char_count,
            "artifact_count": self.artifact_count,
            "artifact_ids": self.artifact_ids,
            "error_message": self.error_message,
            "ocr_applied": self.ocr_applied,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PageAuditRecord:
        d = dict(data)
        d["state"] = PageState(d["state"])
        return cls(**d)


@dataclass
class SectionNode:
    """
    A structural section or subsection within a module.
    """
    section_id: str                      # e.g., "SEC-M3-004"
    module_index: int                    # 1 to 5
    title: str                           # e.g., "3.4 Congestion Control in TCP"
    level: int = 2                       # 1 = Module, 2 = Section, 3 = Subsection
    numbering: Optional[str] = None      # e.g., "3.4", "Unit 2.1"
    start_page: int = 1
    end_page: int = 1
    parent_section_id: Optional[str] = None
    child_section_ids: List[str] = field(default_factory=list)
    artifact_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SectionNode:
        return cls(**data)


@dataclass
class ModuleNode:
    """
    A top-level academic module (Modules 1 to 5 in VTU / engineering exams).
    """
    module_index: int                    # 1 to 5
    title: str                           # e.g., "Module 3: Transport Layer"
    start_page: int = 1
    end_page: int = 1
    section_ids: List[str] = field(default_factory=list)
    artifact_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModuleNode:
        return cls(**data)


class DocumentDOM:
    """
    The canonical Document Object Model.
    Integrates the hierarchy of Modules, Sections, Artifacts, Relationships,
    and Page Accounting.
    """

    def __init__(
        self,
        document_id: str,
        title: str = "",
        total_pages: int = 0,
        registry: Optional[ArtifactRegistry] = None,
        relationships: Optional[RelationshipGraph] = None,
    ):
        self.document_id = document_id
        self.title = title
        self.total_pages = total_pages
        self.modules: Dict[int, ModuleNode] = {}
        self.sections: Dict[str, SectionNode] = {}
        self.page_audit: Dict[int, PageAuditRecord] = {}
        self.registry = registry or ArtifactRegistry()
        self.relationships = relationships or RelationshipGraph()

    # -----------------------------------------------------------------------
    # Tree Mutation
    # -----------------------------------------------------------------------

    def add_module(self, module: ModuleNode) -> None:
        self.modules[module.module_index] = module

    def add_section(self, section: SectionNode) -> None:
        self.sections[section.section_id] = section
        # Also ensure enclosing module tracks this section
        if section.module_index in self.modules:
            mod = self.modules[section.module_index]
            if section.section_id not in mod.section_ids:
                mod.section_ids.append(section.section_id)

    def attach_artifact(
        self,
        artifact: BaseArtifact,
        module_index: Optional[int] = None,
        section_id: Optional[str] = None,
    ) -> None:
        """Register an artifact and bind it to its parent section and module."""
        m_idx = module_index if module_index is not None else artifact.module_index
        s_id = section_id if section_id is not None else artifact.section_id

        artifact.module_index = m_idx
        artifact.section_id = s_id

        # Register artifact in the vault
        self.registry.register(artifact)

        # Bind to Section
        if s_id and s_id in self.sections:
            sec = self.sections[s_id]
            if artifact.artifact_id not in sec.artifact_ids:
                sec.artifact_ids.append(artifact.artifact_id)

        # Bind to Module
        if m_idx in self.modules:
            mod = self.modules[m_idx]
            if artifact.artifact_id not in mod.artifact_ids:
                mod.artifact_ids.append(artifact.artifact_id)

        # Update page audit
        p = artifact.page_number
        if p in self.page_audit:
            rec = self.page_audit[p]
            rec.artifact_count += 1
            if artifact.artifact_id not in rec.artifact_ids:
                rec.artifact_ids.append(artifact.artifact_id)

    def record_page_audit(self, record: PageAuditRecord) -> None:
        """Record the extraction state of a page."""
        self.page_audit[record.page_number] = record

    # -----------------------------------------------------------------------
    # Query APIs
    # -----------------------------------------------------------------------

    def get_module(self, module_index: int) -> Optional[ModuleNode]:
        return self.modules.get(module_index)

    def get_section(self, section_id: str) -> Optional[SectionNode]:
        return self.sections.get(section_id)

    def get_artifacts_for_section(
        self, section_id: str, artifact_type: Optional[ArtifactType] = None
    ) -> List[BaseArtifact]:
        sec = self.get_section(section_id)
        if not sec:
            return []
        artifacts = []
        for aid in sec.artifact_ids:
            a = self.registry.get(aid)
            if a and (artifact_type is None or a.artifact_type == artifact_type):
                artifacts.append(a)
        return artifacts

    def get_artifacts_for_module(
        self, module_index: int, artifact_type: Optional[ArtifactType] = None
    ) -> List[BaseArtifact]:
        mod = self.get_module(module_index)
        if not mod:
            return []
        artifacts = []
        for aid in mod.artifact_ids:
            a = self.registry.get(aid)
            if a and (artifact_type is None or a.artifact_type == artifact_type):
                artifacts.append(a)
        return artifacts

    def get_page_audit_summary(self) -> Dict[str, Any]:
        """
        Calculates page accounting integrity across the entire document.
        Detects any unrecorded/dropped pages.
        """
        counts: Dict[str, int] = {st.value: 0 for st in PageState}
        missing_pages = []

        for p in range(1, self.total_pages + 1):
            if p not in self.page_audit:
                missing_pages.append(p)
            else:
                st = self.page_audit[p].state.value
                counts[st] = counts.get(st, 0) + 1

        return {
            "total_pages": self.total_pages,
            "recorded_pages": len(self.page_audit),
            "missing_pages": missing_pages,
            "state_breakdown": counts,
            "is_complete": len(missing_pages) == 0,
        }

    # -----------------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "title": self.title,
            "total_pages": self.total_pages,
            "modules": {str(k): v.to_dict() for k, v in self.modules.items()},
            "sections": {k: v.to_dict() for k, v in self.sections.items()},
            "page_audit": {str(k): v.to_dict() for k, v in self.page_audit.items()},
            "registry": self.registry.to_dict(),
            "relationships": self.relationships.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DocumentDOM:
        reg = ArtifactRegistry.from_dict(data.get("registry", {}))
        graph = RelationshipGraph.from_dict(data.get("relationships", {}))

        dom = cls(
            document_id=data["document_id"],
            title=data.get("title", ""),
            total_pages=data.get("total_pages", 0),
            registry=reg,
            relationships=graph,
        )

        for m_str, m_data in data.get("modules", {}).items():
            dom.modules[int(m_str)] = ModuleNode.from_dict(m_data)

        for s_str, s_data in data.get("sections", {}).items():
            dom.sections[s_str] = SectionNode.from_dict(s_data)

        for p_str, p_data in data.get("page_audit", {}).items():
            dom.page_audit[int(p_str)] = PageAuditRecord.from_dict(p_data)

        return dom
