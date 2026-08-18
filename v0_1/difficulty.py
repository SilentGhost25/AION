"""
AION Module: Difficulty System
Controls question complexity per sub-question.
Default: mixed (easy + medium + hard)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

DifficultyLevel = Literal["easy", "medium", "hard", "mixed"]


@dataclass
class DifficultyConfig:
    level:       DifficultyLevel = "mixed"
    easy_ratio:  float           = 0.25   # 25% easy
    medium_ratio: float          = 0.50   # 50% medium
    hard_ratio:  float           = 0.25   # 25% hard


# -- Difficulty -> Bloom level mapping -------------------------

DIFFICULTY_BLOOM_MAP = {
    "easy":   [1, 2],        # Remember, Understand
    "medium": [2, 3, 4],     # Understand, Apply, Analyse
    "hard":   [4, 5, 6],     # Analyse, Evaluate, Create
}

DIFFICULTY_MARKS_HINTS = {
    "easy":   "This is a short-answer question requiring basic recall or explanation.",
    "medium": "This requires understanding and application of concepts.",
    "hard":   "This requires critical analysis, evaluation, or design thinking.",
}

DIFFICULTY_VERB_POOLS = {
    "easy": [
        "Define", "List", "State", "Name",
        "Identify", "Recall", "Describe",
    ],
    "medium": [
        "Explain", "Illustrate", "Apply",
        "Classify", "Summarise", "Solve",
        "Demonstrate", "Interpret",
    ],
    "hard": [
        "Analyse", "Compare", "Evaluate",
        "Design", "Justify", "Differentiate",
        "Critique", "Formulate", "Construct",
    ],
}


class DifficultyManager:
    """
    Assigns difficulty levels to sub-questions.
    Tracks used verbs to prevent repetition.
    """

    def __init__(self, config: DifficultyConfig = None):
        self.config = config or DifficultyConfig()
        self._used  = set()

    def assign_difficulty(
        self,
        sub_index:  int,
        total_subs: int,
        marks:      int,
    ) -> DifficultyLevel:
        """
        Assign difficulty to a sub-question.
        For mixed: distribute easy/medium/hard across sub-questions.
        """
        level = self.config.level

        if level != "mixed":
            return level

        # Mixed distribution
        ratio = sub_index / max(total_subs - 1, 1)

        if ratio <= self.config.easy_ratio:
            return "easy"
        elif ratio <= self.config.easy_ratio + self.config.medium_ratio:
            return "medium"
        else:
            return "hard"

    def get_bloom_for_difficulty(
        self,
        difficulty: DifficultyLevel,
    ) -> int:
        """Get appropriate Bloom level for difficulty."""
        pool = DIFFICULTY_BLOOM_MAP.get(difficulty, [2, 3])
        return random.choice(pool)

    def get_verb(
        self,
        difficulty: DifficultyLevel,
        bloom:      int,
    ) -> str:
        """Get a fresh verb not recently used."""
        pool = DIFFICULTY_VERB_POOLS.get(difficulty, ["Explain"])
        fresh = [v for v in pool if v not in self._used]
        verb  = random.choice(fresh) if fresh else random.choice(pool)
        self._used.add(verb)
        # Reset tracker to avoid blocking all verbs
        if len(self._used) > 8:
            self._used.clear()
        return verb

    def get_hint(self, difficulty: DifficultyLevel) -> str:
        return DIFFICULTY_MARKS_HINTS.get(difficulty, "")

    @staticmethod
    def from_string(level: str) -> "DifficultyManager":
        level = level.lower().strip()
        if level not in ("easy", "medium", "hard", "mixed"):
            level = "mixed"

        configs = {
            "easy":   DifficultyConfig("easy",   1.0, 0.0, 0.0),
            "medium": DifficultyConfig("medium", 0.0, 1.0, 0.0),
            "hard":   DifficultyConfig("hard",   0.0, 0.0, 1.0),
            "mixed":  DifficultyConfig("mixed",  0.25, 0.50, 0.25),
        }
        return DifficultyManager(configs[level])
