# core/validation/common.py

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class RetryAction(str, Enum):
    REGENERATE = "REGENERATE"
    REGENERATE_WITH_BLOOM_HINT = "REGENERATE_WITH_BLOOM_HINT"
    REBUILD_EVIDENCE = "REBUILD_EVIDENCE"
    CRITICAL = "CRITICAL"


class GenerationFailureCode(str, Enum):
    META_LANGUAGE = "META_LANGUAGE"
    ANSWER_LEAK = "ANSWER_LEAK"
    BLOOM_MISMATCH = "BLOOM_MISMATCH"
    DEMAND_FAILURE = "DEMAND_FAILURE"
    MATH_FAILURE = "MATH_FAILURE"
    UNICODE_FAILURE = "UNICODE_FAILURE"
    EVIDENCE_FAILURE = "EVIDENCE_FAILURE"
    OR_DUPLICATE = "OR_DUPLICATE"
    MULTI_SLOT_CONTAMINATION = "MULTI_SLOT_CONTAMINATION"
    CONTRACT_FAILURE = "CONTRACT_FAILURE"
    ANSWERABILITY_FAILURE = "ANSWERABILITY_FAILURE"
    GENERATION_TIMEOUT = "GENERATION_TIMEOUT"
    SCHEMA_FAILURE = "SCHEMA_FAILURE"


@dataclass
class GenerationFailure:
    slot_id: str
    code: GenerationFailureCode
    category: str
    message: str
    retryable: bool
    attempt: int


@dataclass
class CheckResult:
    passed: bool
    code: Optional[str] = None
    message: Optional[str] = None
    action: Optional[RetryAction] = None
    is_critical: bool = False

    @classmethod
    def fail(cls, code: str, message: str, action: Optional[RetryAction] = None) -> "CheckResult":
        return cls(passed=False, code=code, message=message, action=action)

    @classmethod
    def critical(cls, code: str, message: str) -> "CheckResult":
        return cls(passed=False, code=code, message=message, is_critical=True, action=RetryAction.CRITICAL)

    @classmethod
    def pass_(cls) -> "CheckResult":
        return cls(passed=True)
