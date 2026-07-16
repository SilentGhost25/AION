# AION-Trainer/acb/concept.py
"""
Concept — the atomic truth unit of AION's knowledge representation.

The Module is a VIEW over concepts, not the storage boundary.
Syllabus changes update Module metadata only; the concept itself
and all its relationships are untouched.

ConceptStore is an in-memory + JSON-on-disk store. It is intentionally
NOT a database — concepts are loaded once per pipeline run, mutated
in memory, then flushed. This keeps the ACB dependency-free (no
Postgres, no Mongo, no SQLite schema migrations needed as the concept
schema evolves during research).
"""

import json
import uuid
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set


class ConceptScope:
    CORE = "core"                     # explicitly named in syllabus
    SUPPLEMENTARY = "supplementary"   # in textbooks but not in syllabus
    PREREQUISITE = "prerequisite"     # needed to understand core concepts
    ADVANCED = "advanced"             # beyond syllabus, reference only


class ConceptStatus:
    VERIFIED = "verified"
    NEEDS_VERIFICATION = "needs_verification"
    CONFLICTED = "conflicted"         # two sources disagree on module/definition
    STUB = "stub"                     # mentioned but not yet fully populated


@dataclass
class ModuleLink:
    """A concept can belong to multiple modules. One is primary."""
    subject_code: str
    module: int
    is_primary: bool = True
    section: str = ""                 # section within the module, if known
    syllabus_text: str = ""           # exact phrase from syllabus that triggered this link


@dataclass
class ConceptSource:
    """One piece of evidence that a concept exists."""
    source_id: str                    # from SourceRegistry
    source_type: str                  # textbook | notes | question_bank | previous_paper | syllabus
    location: str                     # chapter/page/section reference
    excerpt: str = ""                 # the raw text that triggered extraction
    reliability: float = 0.8          # inherited from SourceRegistry at extraction time


@dataclass
class BloomProgression:
    """Which Bloom levels are supported by available material."""
    L1: bool = False
    L2: bool = False
    L3: bool = False
    L4: bool = False
    L5: bool = False
    L6: bool = False

    def coverage_fraction(self) -> float:
        supported = sum([self.L1, self.L2, self.L3, self.L4, self.L5, self.L6])
        return supported / 6.0

    def highest_level(self) -> str:
        for lvl in ["L6", "L5", "L4", "L3", "L2", "L1"]:
            if getattr(self, lvl):
                return lvl
        return "L1"


@dataclass
class Concept:
    """
    The single source of truth for one academic concept.

    Every field except concept_id and name has a sensible default so
    concepts can be constructed incrementally as new sources are
    processed — the ConceptMerger fills fields in as evidence arrives.
    """
    # Identity
    concept_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    aliases: List[str] = field(default_factory=list)     # "A-Star", "A Star", "A*"
    canonical_name: str = ""                              # chosen after deduplication

    # Academic location
    module_links: List[ModuleLink] = field(default_factory=list)
    scope: str = ConceptScope.CORE
    status: str = ConceptStatus.STUB

    # Knowledge content
    definition: str = ""
    explanation: str = ""
    key_points: List[str] = field(default_factory=list)
    algorithms: List[str] = field(default_factory=list)
    applications: List[str] = field(default_factory=list)
    formulas: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    common_misconceptions: List[str] = field(default_factory=list)

    # Diagrams
    requires_diagram: bool = False
    diagram_description: str = ""
    diagram_file: str = ""

    # Relationships (other concept_ids)
    prerequisites: List[str] = field(default_factory=list)
    related_concepts: List[str] = field(default_factory=list)
    subsumed_by: Optional[str] = None    # parent concept
    subsumes: List[str] = field(default_factory=list)      # child concepts

    # Evidence
    sources: List[ConceptSource] = field(default_factory=list)
    syllabus_mentions: int = 0
    question_bank_frequency: int = 0
    previous_paper_frequency: int = 0
    professor_notes_present: bool = False

    # Computed scores (filled by engines, not by extraction)
    confidence: float = 0.0
    importance: float = 0.0
    bloom_progression: BloomProgression = field(default_factory=BloomProgression)

    # Typical assessment
    typical_marks: List[int] = field(default_factory=list)
    typical_question_types: List[str] = field(default_factory=list)
    expected_answer_outline: str = ""
    past_questions: List[str] = field(default_factory=list)

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    keywords: List[str] = field(default_factory=list)

    # ---- Convenience accessors -----------------------------------

    def primary_module(self) -> Optional[int]:
        for link in self.module_links:
            if link.is_primary:
                return link.module
        return self.module_links[0].module if self.module_links else None

    def primary_subject(self) -> Optional[str]:
        for link in self.module_links:
            if link.is_primary:
                return link.subject_code
        return self.module_links[0].subject_code if self.module_links else None

    def add_module_link(self, subject_code: str, module: int, is_primary: bool = False,
                         section: str = "", syllabus_text: str = ""):
        """Add or update a module link. Never duplicates."""
        for existing in self.module_links:
            if existing.subject_code == subject_code and existing.module == module:
                if is_primary:
                    existing.is_primary = True
                return
        self.module_links.append(ModuleLink(
            subject_code=subject_code, module=module, is_primary=is_primary,
            section=section, syllabus_text=syllabus_text,
        ))

    def touch(self):
        self.updated_at = datetime.utcnow().isoformat()

    def content_hash(self) -> str:
        content = f"{self.name}|{self.definition}|{self.explanation}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Concept":
        if "module_links" in data:
            data["module_links"] = [
                ModuleLink(**m) if isinstance(m, dict) else m
                for m in data["module_links"]
            ]
        if "sources" in data:
            data["sources"] = [
                ConceptSource(**s) if isinstance(s, dict) else s
                for s in data["sources"]
            ]
        if "bloom_progression" in data:
            bp = data["bloom_progression"]
            data["bloom_progression"] = (
                BloomProgression(**bp) if isinstance(bp, dict) else bp
            )
        return cls(**data)


class ConceptStore:
    """
    In-memory concept store with JSON persistence.

    Lookup by concept_id (primary), by name, and by alias —
    all O(1) after the initial index build.
    """

    def __init__(self, store_path: str = None):
        self._concepts: Dict[str, Concept] = {}        # concept_id -> Concept
        self._name_index: Dict[str, str] = {}           # normalised name -> concept_id
        self.store_path = Path(store_path) if store_path else None

    # ---- CRUD -------------------------------------------------------

    def add(self, concept: Concept):
        self._concepts[concept.concept_id] = concept
        self._index_concept(concept)

    def get(self, concept_id: str) -> Optional[Concept]:
        return self._concepts.get(concept_id)

    def find_by_name(self, name: str) -> Optional[Concept]:
        cid = self._name_index.get(self._normalise(name))
        return self._concepts.get(cid) if cid else None

    def find_or_create(self, name: str) -> tuple:
        """Returns (concept, was_created)."""
        existing = self.find_by_name(name)
        if existing:
            return existing, False
        c = Concept(name=name, canonical_name=name)
        self.add(c)
        return c, True

    def all_concepts(self) -> List[Concept]:
        return list(self._concepts.values())

    def concepts_for_subject(self, subject_code: str) -> List[Concept]:
        return [
            c for c in self._concepts.values()
            if any(ml.subject_code == subject_code for ml in c.module_links)
        ]

    def concepts_for_module(self, subject_code: str, module: int) -> List[Concept]:
        return [
            c for c in self._concepts.values()
            if any(ml.subject_code == subject_code and ml.module == module
                   for ml in c.module_links)
        ]

    def size(self) -> int:
        return len(self._concepts)

    # ---- Persistence ------------------------------------------------

    def save(self, path: str = None):
        target = Path(path or self.store_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = {cid: c.to_dict() for cid, c in self._concepts.items()}
        target.write_text(json.dumps(data, indent=2, default=str))

    def load(self, path: str = None):
        target = Path(path or self.store_path)
        if not target.exists():
            return
        data = json.loads(target.read_text())
        for cid, c_dict in data.items():
            concept = Concept.from_dict(c_dict)
            self._concepts[cid] = concept
            self._index_concept(concept)

    # ---- Internal ---------------------------------------------------

    def _normalise(self, name: str) -> str:
        return name.lower().strip().replace("-", " ").replace("*", "star")

    def _index_concept(self, concept: Concept):
        self._name_index[self._normalise(concept.name)] = concept.concept_id
        for alias in concept.aliases:
            self._name_index[self._normalise(alias)] = concept.concept_id
        if concept.canonical_name:
            self._name_index[self._normalise(concept.canonical_name)] = concept.concept_id
