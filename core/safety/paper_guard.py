"""
AION Paper Safety Guard
=======================
Strictly validates and repairs paper structure, marks, Bloom levels, and COs.
Ensures zero-mark or missing-mark questions default to active user split or 10M total.
"""
from typing import Any, List, Tuple, Optional, Dict
import logging

logger = logging.getLogger("AION.PAPER_GUARD")


class PaperSafetyGuard:
    VALID_BLOOM = {"L1", "L2", "L3", "L4", "L5", "L6"}
    VALID_CO = {"CO1", "CO2", "CO3", "CO4", "CO5"}

    @classmethod
    def validate(cls, paper: Any) -> Tuple[bool, List[str]]:
        if paper is None:
            return False, ["Paper is None"]
        modules = cls._extract_modules(paper)
        if not modules:
            return False, ["No modules found in paper"]
        total_q = sum(len(m.get("questions", [])) for m in modules if isinstance(m, dict))
        return (total_q > 0, [] if total_q > 0 else ["Zero total questions"])

    @classmethod
    def repair(cls, paper: Any, target_split: Optional[List[int]] = None) -> Any:
        """
        Guarantees sub-question marks, COs, and Bloom levels are
        strictly assigned even for fallback/zero-mark papers.
        """
        if paper is None:
            return paper

        from core.generation.marks_partitioner import get_user_split
        active_split = target_split or get_user_split()

        modules = cls._extract_modules(paper)
        for mod_idx, mod in enumerate(modules, 1):
            if not isinstance(mod, dict):
                continue

            default_co = "CO1" if mod_idx in (1, 2) else ("CO2" if mod_idx in (3, 4) else "CO3")

            for q_idx, q in enumerate(mod.get("questions", []), 1):
                if not isinstance(q, dict):
                    continue

                subs = q.get("sub_questions", q.get("subQuestions", []))
                raw_m = q.get("marks", 0) or q.get("total_marks", 0)

                # If marks <= 0, resolve target_total from active_split or default 10
                if not isinstance(raw_m, int) or raw_m <= 0:
                    target_total = sum(active_split) if active_split else 10
                else:
                    target_total = raw_m

                # Determine active split to apply
                if active_split and sum(active_split) == target_total:
                    split_to_apply = list(active_split)
                elif len(subs) == 2:
                    split_to_apply = [5, 5] if target_total == 10 else [10, 10]
                elif len(subs) == 3:
                    split_to_apply = [4, 3, 3] if target_total == 10 else [8, 6, 6]
                else:
                    split_to_apply = [target_total]

                if subs:
                    for idx, sq in enumerate(subs):
                        if isinstance(sq, dict):
                            # Assign exact marks per sub-question
                            sq["marks"] = split_to_apply[idx] if idx < len(split_to_apply) else 0

                            # Validate/assign Bloom level
                            bl = str(sq.get("bloom_level") or sq.get("bloom") or "").upper()
                            if bl not in cls.VALID_BLOOM:
                                sq["bloom_level"] = "L2" if q_idx in (1, 2) else "L3"
                                sq["bloom"] = sq["bloom_level"]
                            else:
                                sq["bloom_level"] = bl
                                sq["bloom"] = bl

                            # Validate/assign Course Outcome
                            co_str = str(sq.get("co") or "").upper()
                            if co_str not in cls.VALID_CO:
                                sq["co"] = default_co
                            else:
                                sq["co"] = co_str

                    q["marks"] = sum(split_to_apply)
                    q["total_marks"] = sum(split_to_apply)
                else:
                    q["marks"] = target_total
                    q["total_marks"] = target_total

        if isinstance(paper, tuple):
            return (modules,) + paper[1:]
        elif isinstance(paper, dict) and "modules" in paper:
            paper["modules"] = modules
        return paper

    @classmethod
    def _extract_modules(cls, paper: Any) -> list:
        if isinstance(paper, tuple):
            return paper[0] if paper and isinstance(paper[0], list) else []
        if isinstance(paper, list):
            return paper
        if isinstance(paper, dict):
            return paper.get("modules", paper.get("sections", []))
        return getattr(paper, "modules", [])


def validate_paper(paper: Any) -> Tuple[bool, List[str]]:
    return PaperSafetyGuard.validate(paper)
