"""
AION Legacy Question Planner
============================
Decides what questions to generate from a GenerationContext.
No LLM calls. Pure rule-based planning.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

try:
    from core.generation_context import GenerationContext
except ImportError:
    GenerationContext = Any


BLOOM_VERBS = {
    "L1": ["Define", "State", "List", "Identify", "Name"],
    "L2": ["Explain", "Describe", "Summarize", "Discuss", "Outline"],
    "L3": ["Illustrate", "Apply", "Demonstrate", "Solve", "Construct"],
    "L4": ["Compare", "Analyze", "Differentiate", "Examine", "Contrast"],
    "L5": ["Evaluate", "Justify", "Assess", "Critique", "Argue"],
    "L6": ["Design", "Develop", "Propose", "Create", "Formulate"],
}

EXAM_CONFIG = {
    "IA": {
        "marks":      10,
        "max_parts":  2,
        "splits":     [(6, 4), (5, 5)],
    },
    "SEE": {
        "marks":      20,
        "max_parts":  3,
        "splits":     [(8, 6, 6), (10, 6, 4), (8, 8, 4), (10, 10)],
    },
}

BLOOM_DISTRIBUTIONS = {
    "Easy":   ["L1", "L1", "L2", "L2", "L3"],
    "Medium": ["L2", "L2", "L3", "L3", "L4"],
    "Hard":   ["L3", "L4", "L4", "L5", "L6"],
    "Mixed":  ["L1", "L2", "L3", "L4", "L3"],
}


@dataclass
class QuestionSpec:
    spec_id:       str
    module_id:     str
    module_number: int
    module_title:  str

    exam_type:     str
    marks:         int
    bloom_level:   str
    command_verb:  str
    marks_split:   List[int]

    chunks:        List[Dict[str, Any]] = field(default_factory=list)
    is_numerical:  bool                 = False
    numerical_template: Optional[Dict[str, Any]] = None

    subject:       str = ""
    chapter:       str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec_id":      self.spec_id,
            "module_id":    self.module_id,
            "module_number": self.module_number,
            "module_title": self.module_title,
            "exam_type":    self.exam_type,
            "marks":        self.marks,
            "bloom_level":  self.bloom_level,
            "command_verb": self.command_verb,
            "marks_split":  self.marks_split,
            "is_numerical": self.is_numerical,
            "subject":      self.subject,
            "chapter":      self.chapter,
        }


class Planner:
    def __init__(self):
        pass

    def plan(self, ctx: Any) -> List[QuestionSpec]:
        specs        = []
        exam_cfg     = EXAM_CONFIG.get(ctx.exam_type, EXAM_CONFIG["IA"])
        bloom_dist   = BLOOM_DISTRIBUTIONS.get(ctx.difficulty, BLOOM_DISTRIBUTIONS["Mixed"])
        spec_counter = 0

        selected = [
            m for m in ctx.modules
            if m["id"] in ctx.selected_modules
        ]

        if not selected:
            selected = ctx.modules

        for mod_idx, module in enumerate(selected):
            module_id    = module["id"]
            module_num   = module.get("number", mod_idx + 1)
            module_title = module.get("title", f"Module {module_num}")

            mod_chunks = ctx.get_chunks_for_module(module_id)
            if not mod_chunks:
                print(f"[PLAN] No chunks for {module_id} — skipping")
                continue

            bloom_level = bloom_dist[mod_idx % len(bloom_dist)]
            splits    = exam_cfg["splits"]
            split_idx = mod_idx % len(splits)
            marks_split = list(splits[split_idx])

            verbs       = BLOOM_VERBS.get(bloom_level, BLOOM_VERBS["L2"])
            verb        = verbs[mod_idx % len(verbs)]

            best_chunks = sorted(
                mod_chunks,
                key     = lambda c: c.get("word_count", 0),
                reverse = True
            )[:3]

            spec_id = f"spec_{spec_counter:03d}"
            spec    = QuestionSpec(
                spec_id        = spec_id,
                module_id      = module_id,
                module_number  = module_num,
                module_title   = module_title,
                exam_type      = ctx.exam_type,
                marks          = exam_cfg["marks"],
                bloom_level    = bloom_level,
                command_verb   = verb,
                marks_split    = marks_split,
                chunks         = best_chunks,
                subject        = module.get("subject", ""),
                chapter        = module_title,
            )

            specs.append(spec)
            spec_counter += 1

        return specs
