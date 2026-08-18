"""
AION Paper Template Builder
===========================
Builds question paper templates for IA and SEE exam modes.
"""

from core.contracts.question_slot import QuestionSlot
from dataclasses import dataclass, field
from typing import List

BLOOM_LEVELS = {
    1: ("Remember", ["define", "list", "state", "recall"]),
    2: ("Understand", ["explain", "describe", "discuss", "summarize"]),
    3: ("Apply", ["calculate", "solve", "apply", "illustrate"]),
    4: ("Analyze", ["analyze", "compare", "contrast", "differentiate"]),
    5: ("Evaluate", ["evaluate", "assess", "justify", "critique"]),
    6: ("Create", ["design", "formulate", "construct", "propose"]),
}


@dataclass
class SubSlot:
    slot_id:     str
    letter:      str
    marks:       int
    bloom_level: int
    verb:        str
    co:          str = "CO1"


@dataclass
class QuestionSlot:
    q_number:     int
    module_index: int
    is_or:        bool
    sub_slots:    List[SubSlot] = field(default_factory=list)


@dataclass
class ExamTemplate:
    exam_type:         str
    attemptable_marks: int
    question_slots:    List[QuestionSlot] = field(default_factory=list)


class PaperTemplateBuilder:
    def build(self, exam_type: str = "IA", n_modules: int = 1, subject: str = "") -> ExamTemplate:
        slots = []
        is_ia = str(exam_type).upper() in ("IA", "IAT1", "IAT2", "IAT3", "MID")

        if is_ia:
            attemptable = 50
            for m in range(1, max(2, n_modules + 1)):
                for q_num in (1, 2):
                    is_or = (q_num == 2)
                    subs = [
                        SubSlot(f"Q{q_num}a_M{m}", "a", 10, 2, "Explain", f"CO{m}"),
                        SubSlot(f"Q{q_num}b_M{m}", "b", 10, 3, "Calculate", f"CO{m}"),
                    ]
                    slots.append(QuestionSlot(q_number=q_num, module_index=m, is_or=is_or, sub_slots=subs))
        else:
            attemptable = 100
            for m in range(1, max(2, n_modules + 1)):
                for q_num in (m * 2 - 1, m * 2):
                    is_or = (q_num % 2 == 0)
                    subs = [
                        SubSlot(f"Q{q_num}a_M{m}", "a", 7, 2, "Explain", f"CO{m}"),
                        SubSlot(f"Q{q_num}b_M{m}", "b", 6, 3, "Derive", f"CO{m}"),
                        SubSlot(f"Q{q_num}c_M{m}", "c", 7, 4, "Analyze", f"CO{m}"),
                    ]
                    slots.append(QuestionSlot(q_number=q_num, module_index=m, is_or=is_or, sub_slots=subs))

        return ExamTemplate(
            exam_type=exam_type,
            attemptable_marks=attemptable,
            question_slots=slots,
        )
