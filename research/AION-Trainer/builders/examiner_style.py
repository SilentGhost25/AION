import re
from typing import List, Dict, Any

class ExaminerStyle:
    def __init__(self):
        self.verb_distribution = {}
        self.bloom_distribution = {}
        self.marks_distribution = {}
        self.total_questions = 0

    def add_question(self, text: str, bloom: str, marks: int, verb: str):
        self.total_questions += 1
        self.verb_distribution[verb] = self.verb_distribution.get(verb, 0) + 1
        self.bloom_distribution[bloom] = self.bloom_distribution.get(bloom, 0) + 1
        self.marks_distribution[marks] = self.marks_distribution.get(marks, 0) + 1

class ExaminerStyleExtractor:
    def _detect_verb(self, text: str) -> str:
        # Simple verb detector: extract first word if it looks like a verb
        words = re.findall(r"\b[a-zA-Z]+\b", text)
        if not words:
            return "explain"
        first = words[0].lower()
        return first

    def extract_from_papers(self, papers: List[Dict[str, Any]]) -> ExaminerStyle:
        style = ExaminerStyle()
        for paper in papers:
            for q in paper.get("questions", []):
                text = q.get("text", "")
                bloom = q.get("bloom", "L2")
                marks = q.get("marks", 5)
                verb = self._detect_verb(text)
                style.add_question(text, bloom, marks, verb)
        return style