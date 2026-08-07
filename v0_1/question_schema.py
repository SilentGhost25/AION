"""
AION Unified Question Schema
============================
Single source of truth for question data structure.
Used by generator, formatter, validator, and frontend.
"""

import uuid
from dataclasses import dataclass, field
from typing import Optional, List, Dict


@dataclass
class SubQuestion:
    letter:  str
    text:    str
    marks:   int
    co:      str   = "CO1"
    bloom:   int   = 2
    image:   Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "letter": self.letter,
            "text":   self.text,
            "marks":  self.marks,
            "co":     self.co,
            "bloom":  self.bloom,
            "image":  self.image,
        }


@dataclass
class MainQuestion:
    mq_index:     int
    total_marks:  int
    bloom_level:  int
    bloom_name:   str
    sub_questions: List[SubQuestion] = field(default_factory=list)
    is_or:        bool = False

    @property
    def actual_marks(self) -> int:
        return sum(sq.marks for sq in self.sub_questions)

    @property
    def marks_valid(self) -> bool:
        return self.actual_marks == self.total_marks

    def to_dict(self) -> dict:
        return {
            "mqIndex":      self.mq_index,
            "totalMarks":   self.total_marks,
            "bloomLevel":   self.bloom_level,
            "bloomName":    self.bloom_name,
            "isOr":         self.is_or,
            "subQuestions": [sq.to_dict() for sq in self.sub_questions],
        }


@dataclass
class Module:
    module_index: int
    module_title: str
    questions:    List[MainQuestion] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "moduleIndex": self.module_index,
            "moduleTitle": self.module_title,
            "questions":   [q.to_dict() for q in self.questions],
        }


@dataclass
class GeneratedPaper:
    subject:     str
    exam_type:   str
    mode:        str
    modules:     List[Module] = field(default_factory=list)
    paper_id:    str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    @property
    def total_marks(self) -> int:
        """
        Correct marks total: only count one side of each OR pair.
        """
        total = 0
        for mod in self.modules:
            qs    = mod.questions
            seen  = set()
            pairs = {}

            for q in qs:
                pair = (q.mq_index - 1) // 2
                if pair not in pairs:
                    pairs[pair] = []
                pairs[pair].append(q)

            for pair_qs in pairs.values():
                total += max(q.actual_marks for q in pair_qs)

        return total

    @property
    def co_coverage(self) -> dict:
        """Dynamic CO coverage from actual questions."""
        counts = {}
        total  = 0
        for mod in self.modules:
            for q in mod.questions:
                if q.is_or:
                    continue  # count only non-OR side
                for sq in q.sub_questions:
                    counts[sq.co] = counts.get(sq.co, 0) + sq.marks
                    total += sq.marks

        if total == 0:
            return {f"CO{i}": 20 for i in range(1, 6)}

        return {
            co: round(marks / total * 100)
            for co, marks in sorted(counts.items())
        }

    @property
    def bloom_distribution(self) -> dict:
        """Bloom level distribution across all questions."""
        counts = {}
        total  = 0
        bloom_names = {
            1: "Remember", 2: "Understand", 3: "Apply",
            4: "Analyze",  5: "Evaluate",   6: "Create"
        }
        for mod in self.modules:
            for q in mod.questions:
                name = bloom_names.get(q.bloom_level, f"L{q.bloom_level}")
                counts[name] = counts.get(name, 0) + q.actual_marks
                total += q.actual_marks

        if total == 0:
            return {}

        return {
            name: round(marks / total * 100)
            for name, marks in counts.items()
        }

    def to_dict(self) -> dict:
        co   = self.co_coverage
        bloom = self.bloom_distribution
        return {
            "id":          self.paper_id,
            "subject":     self.subject,
            "examType":    self.exam_type,
            "mode":        self.mode,
            "modules":     [m.to_dict() for m in self.modules],
            "totalMarks":  self.total_marks,
            "generatedAt": __import__("datetime").datetime.now().isoformat(),
            "coCoverage": {
                "co1": co.get("CO1", 0),
                "co2": co.get("CO2", 0),
                "co3": co.get("CO3", 0),
                "co4": co.get("CO4", 0),
                "co5": co.get("CO5", 0),
            },
            "bloomDistribution": bloom,
            "syllabusCoverage": {
                f"s{m.module_index}": round(100 / max(1, len(self.modules)))
                for m in self.modules
            },
            "qaReport": {},
        }
