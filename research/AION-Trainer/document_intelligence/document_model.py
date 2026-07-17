from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pathlib import Path

@dataclass
class Section:
    title: str
    content: str
    layout_type: str = ""
    page_range: Optional[tuple[int, int]] = None

@dataclass
class AcademicDocument:
    """
    A structured representation of an academic document (PDF, textbook, etc.)
    """
    source_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    markdown: str = ""
    json_structure: Dict[str, Any] = field(default_factory=dict)
    
    # Semantic blocks
    sections: List[Section] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    figures: List[Dict[str, Any]] = field(default_factory=list)
    formulas: List[Dict[str, Any]] = field(default_factory=list)
    
    # Document structure
    toc: List[Dict[str, Any]] = field(default_factory=list)
    reading_order: List[str] = field(default_factory=list)
    page_map: Dict[int, Any] = field(default_factory=dict)

    def get_layout_blocks_by_type(self, block_type: str) -> List[Section]:
        """Retrieve specific layout blocks like 'Algorithm', 'Warning', 'Example'"""
        return [sec for sec in self.sections if sec.layout_type == block_type]
