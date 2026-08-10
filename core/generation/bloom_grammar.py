"""
AION Generation — Bloom Verb Grammar & Template Alignment
==========================================================
Enforces INV-4: Action verbs are mapped to grammatical classes to ensure
proper syntactic alignment with question templates BEFORE Qwen realization.
Rejects invalid phrases like 'Create between...', 'Apply why...', 'Evaluate between...'.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class BloomLevel(str, Enum):
    L1 = "L1"  # Remember
    L2 = "L2"  # Understand
    L3 = "L3"  # Apply
    L4 = "L4"  # Analyze
    L5 = "L5"  # Evaluate
    L6 = "L6"  # Create


class BloomVerbGrammar(str, Enum):
    TRANSITIVE_FACTUAL  = "TRANSITIVE_FACTUAL"   # "Define X", "List Y"
    TRANSITIVE_PROCESS  = "TRANSITIVE_PROCESS"   # "Explain how", "Describe X"
    ANALYTICAL          = "ANALYTICAL"            # "Compare X and Y"
    EVALUATIVE          = "EVALUATIVE"            # "Justify why", "Evaluate the impact"
    CONSTRUCTIVE        = "CONSTRUCTIVE"          # "Design X", "Develop Y"
    COMPUTATIONAL       = "COMPUTATIONAL"         # "Calculate X", "Determine Y"


BLOOM_VERB_TABLE: dict[str, Tuple[BloomLevel, BloomVerbGrammar]] = {
    # L1 Remember
    "define": (BloomLevel.L1, BloomVerbGrammar.TRANSITIVE_FACTUAL),
    "list": (BloomLevel.L1, BloomVerbGrammar.TRANSITIVE_FACTUAL),
    "identify": (BloomLevel.L1, BloomVerbGrammar.TRANSITIVE_FACTUAL),
    "name": (BloomLevel.L1, BloomVerbGrammar.TRANSITIVE_FACTUAL),
    "state": (BloomLevel.L1, BloomVerbGrammar.TRANSITIVE_FACTUAL),

    # L2 Understand
    "explain": (BloomLevel.L2, BloomVerbGrammar.TRANSITIVE_PROCESS),
    "describe": (BloomLevel.L2, BloomVerbGrammar.TRANSITIVE_FACTUAL),
    "summarize": (BloomLevel.L2, BloomVerbGrammar.TRANSITIVE_FACTUAL),
    "illustrate": (BloomLevel.L2, BloomVerbGrammar.TRANSITIVE_FACTUAL),

    # L3 Apply
    "calculate": (BloomLevel.L3, BloomVerbGrammar.COMPUTATIONAL),
    "apply": (BloomLevel.L3, BloomVerbGrammar.TRANSITIVE_PROCESS),
    "demonstrate": (BloomLevel.L3, BloomVerbGrammar.TRANSITIVE_PROCESS),
    "determine": (BloomLevel.L3, BloomVerbGrammar.COMPUTATIONAL),
    "solve": (BloomLevel.L3, BloomVerbGrammar.COMPUTATIONAL),

    # L4 Analyze
    "analyze": (BloomLevel.L4, BloomVerbGrammar.TRANSITIVE_FACTUAL),
    "compare": (BloomLevel.L4, BloomVerbGrammar.ANALYTICAL),
    "examine": (BloomLevel.L4, BloomVerbGrammar.TRANSITIVE_FACTUAL),
    "differentiate": (BloomLevel.L4, BloomVerbGrammar.ANALYTICAL),

    # L5 Evaluate
    "evaluate": (BloomLevel.L5, BloomVerbGrammar.EVALUATIVE),
    "critique": (BloomLevel.L5, BloomVerbGrammar.EVALUATIVE),
    "justify": (BloomLevel.L5, BloomVerbGrammar.EVALUATIVE),

    # L6 Create
    "design": (BloomLevel.L6, BloomVerbGrammar.CONSTRUCTIVE),
    "develop": (BloomLevel.L6, BloomVerbGrammar.CONSTRUCTIVE),
    "propose": (BloomLevel.L6, BloomVerbGrammar.CONSTRUCTIVE),
    "create": (BloomLevel.L6, BloomVerbGrammar.CONSTRUCTIVE),
}


# Forbidden verb + preposition/structure combinations
FORBIDDEN_COMBINATIONS: List[Tuple[str, str]] = [
    ("create", "between"),
    ("evaluate", "between"),
    ("apply", "why"),
    ("apply", "between"),
    ("list", "why"),
    ("state", "between"),
    ("design", "between"),
    ("calculate", "why"),
    ("solve", "why"),
]


@dataclass
class BloomGrammarReport:
    valid  : bool
    verb   : str
    issue  : Optional[str] = None
    grammar: Optional[BloomVerbGrammar] = None


class BloomGrammarValidator:
    """Validates grammatical structure of Bloom action verbs in questions."""

    @classmethod
    def validate_verb_phrase(cls, verb: str, question_text: str) -> BloomGrammarReport:
        v_lower = verb.lower().strip()
        q_lower = question_text.lower().strip()

        # Check forbidden combinations
        for forb_verb, forb_prep in FORBIDDEN_COMBINATIONS:
            if forb_verb in q_lower:
                # Regex match forb_verb ... forb_prep
                pattern = r"\b" + forb_verb + r"\b.*?\b" + forb_prep + r"\b"
                if re.search(pattern, q_lower):
                    return BloomGrammarReport(
                        valid=False,
                        verb=verb,
                        issue=f"Forbidden combination: '{forb_verb}' used with '{forb_prep}'",
                    )

        entry = BLOOM_VERB_TABLE.get(v_lower)
        grammar_class = entry[1] if entry else None

        return BloomGrammarReport(
            valid=True,
            verb=verb,
            grammar=grammar_class,
        )
