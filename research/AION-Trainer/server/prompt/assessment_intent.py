# AION-Trainer/server/prompt/assessment_intent.py
"""
AssessmentIntent — the structured plan that sits between Knowledge
Extraction and Prompt Construction. This is the bridge Issue 3
described: instead of 'Knowledge -> ???', we now have
'Knowledge + Intent -> Question'.

Nothing in this file is new architecture. It simply makes explicit
what was previously implicit (and therefore inconsistent) across the
codebase.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AssessmentIntent:
    """
    Everything an examiner decides before writing a question.
    The prompt builder converts this into natural language instructions
    for the model.
    """

    # What to assess
    topic: str = ""
    subtopic: str = ""
    definition: str = ""
    explanation: str = ""
    key_points: List[str] = field(default_factory=list)
    algorithms: List[str] = field(default_factory=list)
    applications: List[str] = field(default_factory=list)
    formulas: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    diagram_description: str = ""

    # How to assess it
    bloom_level: str = "L2"
    action_verb: str = "Explain"
    marks: int = 10
    question_type: str = "explanation"      # definition | explanation | algorithm
                                             # comparison | numerical | diagram_based
    difficulty: str = "medium"
    requires_diagram: bool = False
    compare_with: Optional[str] = None      # for comparison questions

    # Academic context
    subject_code: str = ""
    subject_name: str = ""
    module: int = 0
    semester: int = 0
    university: str = "VTU"
    co_tag: str = ""

    # Style guidance
    reference_questions: List[str] = field(default_factory=list)
    previously_asked: List[str] = field(default_factory=list)

    # Constraints
    max_words: int = 35
    min_words: int = 8
