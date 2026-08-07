"""
AION Question Planner
=====================
Decides what questions to generate from a GenerationContext.
No LLM calls. Pure rule-based planning.

Output: a list of QuestionSpec objects, one per question to generate.
The generator consumes these specs one at a time.
"""

from dataclasses import dataclass, field
from typing import Optional

from core.generation_context import GenerationContext


# ── Bloom taxonomy ────────────────────────────────────────────────────────────

BLOOM_VERBS = {
    "L1": ["Define", "State", "List", "Identify", "Name"],
    "L2": ["Explain", "Describe", "Summarize", "Discuss", "Outline"],
    "L3": ["Illustrate", "Apply", "Demonstrate", "Solve", "Construct"],
    "L4": ["Compare", "Analyze", "Differentiate", "Examine", "Contrast"],
    "L5": ["Evaluate", "Justify", "Assess", "Critique", "Argue"],
    "L6": ["Design", "Develop", "Propose", "Create", "Formulate"],
}

# VTU IA exam: 10 marks, max 2 sub-questions
# VTU SEE exam: 20 marks, max 3 sub-questions
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

# Bloom distribution by difficulty
BLOOM_DISTRIBUTIONS = {
    "Easy":   ["L1", "L1", "L2", "L2", "L3"],
    "Medium": ["L2", "L2", "L3", "L3", "L4"],
    "Hard":   ["L3", "L4", "L4", "L5", "L6"],
    "Mixed":  ["L1", "L2", "L3", "L4", "L3"],
}


@dataclass
class QuestionSpec:
    """
    Specification for a single question to be generated.
    The generator receives this and produces the actual question.
    """
    spec_id:       str
    module_id:     str
    module_number: int
    module_title:  str

    exam_type:     str
    marks:         int
    bloom_level:   str
    command_verb:  str
    marks_split:   list[int]

    chunks:        list[dict]    = field(default_factory=list)
    is_numerical:  bool          = False
    numerical_template: Optional[dict] = None

    subject:       str = ""
    chapter:       str = ""

    def to_dict(self) -> dict:
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
    """
    Plans question generation from a GenerationContext.
    Decides: which modules, how many questions,
    what Bloom level, what marks split.
    No LLM calls.
    """

    def __init__(self):
        pass

    def plan(self, ctx: GenerationContext) -> list[QuestionSpec]:
        """
        Generate a list of QuestionSpecs from the context.
        One spec per question to generate.
        """
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

            # Get chunks for this module
            mod_chunks = ctx.get_chunks_for_module(module_id)
            if not mod_chunks:
                print(f"[PLAN] No chunks for {module_id} — skipping")
                continue

            # Determine Bloom level for this module
            bloom_level = bloom_dist[mod_idx % len(bloom_dist)]

            # Select marks split
            splits    = exam_cfg["splits"]
            split_idx = mod_idx % len(splits)
            marks_split = list(splits[split_idx])

            # Select command verb
            verbs       = BLOOM_VERBS.get(bloom_level, BLOOM_VERBS["L2"])
            verb        = verbs[mod_idx % len(verbs)]

            # Select best chunks (top 3 by word count, no more)
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
            print(
                f"[PLAN] {module_id} | {bloom_level} | "
                f"{verb} | {marks_split} | {len(best_chunks)} chunks"
            )

        print(f"[PLAN] Generated {len(specs)} question specs")
        return specs
