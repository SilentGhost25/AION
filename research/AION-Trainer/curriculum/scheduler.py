"""
Curriculum Scheduler

The trainer learns easy concepts first, then harder ones:

Stage 1: Definitions & Basic Recall (L1, L2)
Stage 2: Explanation & Application (L2, L3)
Stage 3: Analysis & Comparison (L4, L5)
Stage 4: Design & Evaluation (L5, L6)

Exactly how humans learn.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("aion.curriculum")


class CurriculumScheduler:
    """
    Manages curriculum learning stages.
    """

    DEFAULT_STAGES = [
        {
            "stage": 1,
            "name": "Definitions & Basic Recall",
            "difficulty": "easy",
            "bloom_levels": ["L1", "L2"],
            "sample_types": ["knowledge_to_question", "definition_to_concept", "concept_to_definition"],
            "epochs": 2,
        },
        {
            "stage": 2,
            "name": "Explanation & Application",
            "difficulty": "medium",
            "bloom_levels": ["L2", "L3"],
            "sample_types": ["knowledge_to_question", "knowledge_to_answer", "algorithm_to_question"],
            "epochs": 3,
        },
        {
            "stage": 3,
            "name": "Analysis & Comparison",
            "difficulty": "hard",
            "bloom_levels": ["L4", "L5"],
            "sample_types": ["comparison_generation", "question_improvement"],
            "epochs": 3,
        },
        {
            "stage": 4,
            "name": "Design & Evaluation",
            "difficulty": "advanced",
            "bloom_levels": ["L5", "L6"],
            "sample_types": ["knowledge_to_question", "multi_concept_question"],
            "epochs": 2,
        },
    ]

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.stages = self.config.get("stages", self.DEFAULT_STAGES)
        self.current_stage = 0

    def get_schedule(self) -> List[Dict[str, Any]]:
        """Get the full curriculum schedule."""
        schedule = []
        for i, stage in enumerate(self.stages):
            schedule.append({
                "stage": stage.get("stage", i + 1),
                "name": stage["name"],
                "difficulty": stage["difficulty"],
                "bloom_levels": stage["bloom_levels"],
                "sample_types": stage["sample_types"],
                "epochs": stage["epochs"],
                "num_samples": 0,  # Will be filled when dataset is loaded
            })
        return schedule

    def get_stage(self, stage_idx: int) -> Optional[Dict[str, Any]]:
        """Get a specific curriculum stage."""
        if 0 <= stage_idx < len(self.stages):
            return self.stages[stage_idx]
        return None

    def filter_samples_for_stage(
        self,
        samples: List[Dict[str, Any]],
        stage_idx: int,
    ) -> List[Dict[str, Any]]:
        """Filter training samples for a specific curriculum stage."""
        stage = self.get_stage(stage_idx)
        if not stage:
            return samples

        bloom_levels = set(stage["bloom_levels"])
        sample_types = set(stage["sample_types"])

        filtered = []
        for sample in samples:
            bloom_match = sample.get("bloom", "L2") in bloom_levels
            type_match = sample.get("sample_type", "") in sample_types
            if bloom_match or type_match:
                filtered.append(sample)

        logger.info(
            f"Stage {stage.get('stage', stage_idx + 1)} ({stage['name']}): "
            f"{len(filtered)} / {len(samples)} samples"
        )
        return filtered

    def advance_stage(self) -> bool:
        """Advance to the next curriculum stage. Returns False if complete."""
        self.current_stage += 1
        if self.current_stage >= len(self.stages):
            logger.info("Curriculum complete!")
            return False

        stage = self.stages[self.current_stage]
        logger.info(
            f"Advancing to Stage {stage.get('stage', self.current_stage + 1)}: {stage['name']}"
        )
        return True

    @property
    def total_epochs(self) -> int:
        """Total epochs across all stages."""
        return sum(stage.get("epochs", 1) for stage in self.stages)

    @property
    def is_complete(self) -> bool:
        """Check if curriculum is complete."""
        return self.current_stage >= len(self.stages)
