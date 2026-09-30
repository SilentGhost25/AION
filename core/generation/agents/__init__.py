"""
AION v3 Multi-Agent Generation Layer.
"""

from .base import Agent, AgentContext, AgentResult
from .question_plan import QuestionPlan
from .generated_question import (
    GeneratedQuestion,
    SlotFailure,
    WritingAgentOutput,
)
from .planning_agent import PlanningAgent
from .writing_agent import (
    WritingAgent,
    GenerationFailed,
    LLMCaller,
    VLMCaller,
)
from .evaluation_contracts import (
    AuditReport,
    SlotVerdict,
    BLOOM_MISMATCH,
    MARKING_SCHEME_MISMATCH,
    MISSING_FIGURE_REFERENCE,
    QUESTION_TOO_SHORT,
    QUESTION_TOO_LONG,
    SOLUTION_MISSING,
    SOLUTION_TOO_SHORT,
    MULTI_PART_STRUCTURE_MISMATCH,
    FACTUAL_ERROR,
    UNGROUNDED,
    CIRCULAR,
    OFF_TOPIC,
    VERB_TASK_MISMATCH,
    UNVERIFIED,
)
from .evaluation_agent import (
    EvaluationAgent,
    APICaller,
    APIUnavailable,
)
from .refinement_contracts import (
    RefinementOutcome,
    RefinementReport,
    TIER_DETERMINISTIC,
    TIER_LLM,
    TIER_NONE,
)
from .refinement_agent import (
    RefinementAgent,
    SympyVerifier,
)
from .checking_contracts import (
    CheckingOutcome,
    QAReport,
    PaperBlockedError,
    STATUS_PASS,
    STATUS_DEGRADED,
    STATUS_BLOCKED,
    BLOCK_REASON_UNRESOLVED_SLOTS,
    BLOCK_REASON_WRITING_FAILURES,
    BLOCK_REASON_INSUFFICIENT_QUESTIONS,
    BLOCK_REASON_INVALID_STATE,
    DEGRADED_REASON_UNVERIFIED_EVALUATION,
)
from .checking_agent import CheckingAgent
from .orchestrator_contracts import OrchestratorResult
from .agent_orchestrator import AgentOrchestrator

__all__ = [
    # Core
    "Agent",
    "AgentContext",
    "AgentResult",
    "QuestionPlan",
    "GeneratedQuestion",
    "SlotFailure",
    "WritingAgentOutput",
    # Agents
    "PlanningAgent",
    "WritingAgent",
    "EvaluationAgent",
    "RefinementAgent",
    "CheckingAgent",
    # Orchestration
    "AgentOrchestrator",
    "OrchestratorResult",
    # Errors and protocols
    "GenerationFailed",
    "LLMCaller",
    "VLMCaller",
    "APICaller",
    "APIUnavailable",
    "SympyVerifier",
    "PaperBlockedError",
    # Evaluation
    "AuditReport",
    "SlotVerdict",
    # Refinement
    "RefinementOutcome",
    "RefinementReport",
    "TIER_DETERMINISTIC",
    "TIER_LLM",
    "TIER_NONE",
    # Checking
    "CheckingOutcome",
    "QAReport",
    "STATUS_PASS",
    "STATUS_DEGRADED",
    "STATUS_BLOCKED",
    "BLOCK_REASON_UNRESOLVED_SLOTS",
    "BLOCK_REASON_WRITING_FAILURES",
    "BLOCK_REASON_INSUFFICIENT_QUESTIONS",
    "BLOCK_REASON_INVALID_STATE",
    "DEGRADED_REASON_UNVERIFIED_EVALUATION",
    # Reason codes
    "BLOOM_MISMATCH",
    "MARKING_SCHEME_MISMATCH",
    "MISSING_FIGURE_REFERENCE",
    "QUESTION_TOO_SHORT",
    "QUESTION_TOO_LONG",
    "SOLUTION_MISSING",
    "SOLUTION_TOO_SHORT",
    "MULTI_PART_STRUCTURE_MISMATCH",
    "FACTUAL_ERROR",
    "UNGROUNDED",
    "CIRCULAR",
    "OFF_TOPIC",
    "VERB_TASK_MISMATCH",
    "UNVERIFIED",
]
