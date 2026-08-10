"""
AION Core Planner Package
=========================
Exports both the production PaperStructurePlanner & QuestionPlanner
and the legacy Planner & QuestionSpec.
"""

from .paper_planner import PaperPlannerError, PaperStructurePlanner
from .question_planner import QuestionPlanner, QuestionPlannerError
from .legacy_planner import Planner, QuestionSpec

__all__ = [
    "PaperStructurePlanner",
    "PaperPlannerError",
    "QuestionPlanner",
    "QuestionPlannerError",
    "Planner",
    "QuestionSpec",
]
