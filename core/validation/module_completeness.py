# core/validation/module_completeness.py

import logging
from typing import Set, List

LOG = logging.getLogger("aion.completeness")


class ModuleIngestionFailure(RuntimeError):
    def __init__(self, missing: Set[int], expected: Set[int], message: str):
        super().__init__(message)
        self.missing = missing
        self.expected = expected


class SlotGenerationIncomplete(RuntimeError):
    def __init__(self, missing_ids: List[str], message: str):
        super().__init__(message)
        self.missing_ids = missing_ids


class DuplicateSlotError(RuntimeError):
    def __init__(self, duplicate_ids: List[str], message: str):
        super().__init__(message)
        self.duplicate_ids = duplicate_ids


class ModuleCompletenessGuard:
    """
    Ensures every requested module has been extracted, planned, and generated.
    A missing module is a HARD FAIL — never silently omit.
    """

    @staticmethod
    def assert_extraction_complete(
        expected_modules: Set[int],
        ingested_modules: Set[int],
    ) -> None:
        missing = expected_modules - ingested_modules
        if missing:
            raise ModuleIngestionFailure(
                missing  = missing,
                expected = expected_modules,
                message  = (
                    f"Modules {sorted(missing)} were not successfully ingested. "
                    f"Pipeline BLOCKED — cannot generate partial paper."
                )
            )
        LOG.info(f"[COMPLETENESS] All {len(expected_modules)} modules ingested.")

    @staticmethod
    def assert_generation_complete(
        expected_slot_ids : Set[str],
        generated_slot_ids: Set[str],
    ) -> None:
        missing   = expected_slot_ids - generated_slot_ids
        duplicate = [sid for sid in generated_slot_ids
                     if list(generated_slot_ids).count(sid) > 1]

        if missing:
            raise SlotGenerationIncomplete(
                missing_ids = sorted(missing),
                message     = (
                    f"{len(missing)} slots were never generated: {sorted(missing)}. "
                    f"Paper BLOCKED — never export an incomplete paper."
                )
            )

        if duplicate:
            raise DuplicateSlotError(
                duplicate_ids = duplicate,
                message       = f"Duplicate slot IDs: {duplicate}"
            )

        LOG.info(
            f"[COMPLETENESS] All {len(expected_slot_ids)} slots generated."
        )
