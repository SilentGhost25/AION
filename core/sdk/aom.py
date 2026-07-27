from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class PaperStatus(str, Enum):
    DRAFT = "draft"
    GENERATING = "generating"
    REVIEW = "review"
    APPROVED = "approved"
    ARCHIVED = "archived"

class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"

class UniversityObject(BaseModel):
    id: str = Field(..., description="Unique identifier for the university")
    name: str = Field(..., description="Full name of the university")
    code: str = Field(..., description="University code (e.g., VTU)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata dictionary for arbitrary configurations")

class DepartmentObject(BaseModel):
    id: str = Field(..., description="Unique identifier for the department")
    name: str = Field(..., description="Name of the department")
    code: str = Field(..., description="Department code (e.g., AIML)")
    semesters: List[int] = Field(default_factory=list, description="List of semesters associated with this department")

class SemesterObject(BaseModel):
    semester_number: int = Field(..., description="Semester index (e.g., 4)")
    subjects: List[str] = Field(default_factory=list, description="Subject codes in this semester")

class SubjectObject(BaseModel):
    code: str = Field(..., description="Subject code (e.g., BAI401)")
    name: str = Field(..., description="Subject title")
    credits: int = Field(..., description="Credit value of this subject")
    syllabus_modules: List[str] = Field(default_factory=list, description="List of module IDs or names")

class ModuleObject(BaseModel):
    module_number: int = Field(..., description="Module number (usually 1-5)")
    title: str = Field(..., description="Module title")
    topics: List[str] = Field(default_factory=list, description="List of topics covered in the module")
    key_concepts: List[str] = Field(default_factory=list, description="Main concepts/terms for indexing")

class KnowledgeObject(BaseModel):
    id: str = Field(..., description="Unique concept identifier")
    concept_name: str = Field(..., description="Name of the academic concept")
    definition: str = Field(..., description="Definition/description of the concept")
    bloom_level: str = Field(..., description="Bloom's taxonomy level (L1-L6)")
    prerequisites: List[str] = Field(default_factory=list, description="Concept IDs required as prerequisites")
    equations: List[str] = Field(default_factory=list, description="Relevant math or code expressions")
    diagram_requirements: List[str] = Field(default_factory=list, description="Requirements for visual illustrations")

class KnowledgeGene(BaseModel):
    gene_id: str = Field(..., description="Unique gene identifier")
    knowledge_id: str = Field(..., description="Linked knowledge concept ID")
    sequence_order: int = Field(..., description="Logical order in the academic genome")
    gene_type: str = Field(..., description="Gene type (e.g., definition, algorithm, derivation, example)")
    raw_content: str = Field(..., description="Raw text representation of the gene")
    vector_embedding: Optional[List[float]] = Field(default=None, description="Semantic embedding of this gene")
    
    # Academic DNA Properties
    concept_name: str = Field(..., description="Name of the academic concept")
    canonical_definition: str = Field(..., description="Approved standard academic definition")
    alternative_definitions: List[str] = Field(default_factory=list, description="Alternative explanations or variations")
    confidence_score: float = Field(default=1.0, description="Verification confidence score of this gene")
    prerequisites: List[str] = Field(default_factory=list, description="IDs of concepts that must be learned first")
    relationships: Dict[str, str] = Field(default_factory=dict, description="Linked concepts and their semantic relations (concept -> relation_type)")
    applications: List[str] = Field(default_factory=list, description="Real-world applications of this concept")
    expected_answer: Optional[str] = Field(default=None, description="Canonical target student answer")
    bloom_progression: List[str] = Field(default_factory=list, description="Expected Bloom levels sequence (e.g., L1, L2, L3)")
    difficulty: str = Field(default="medium", description="Estimated difficulty level (easy, medium, hard, advanced)")
    exam_frequency: int = Field(default=0, description="Historical exam occurrence count")
    recent_trends: List[str] = Field(default_factory=list, description="Recent curriculum updates or trends related to the concept")
    professor_notes: List[str] = Field(default_factory=list, description="Special notes or emphasis from faculty")
    typical_mistakes: List[str] = Field(default_factory=list, description="Common student misconceptions or validation check errors")
    question_styles: List[str] = Field(default_factory=list, description="Applicable question prefixes (e.g., Define, Explain)")
    image_references: List[str] = Field(default_factory=list, description="IDs of associated visual diagrams")
    algorithms: List[str] = Field(default_factory=list, description="Associated code algorithms or pseudocode")
    formulae: List[str] = Field(default_factory=list, description="Mathematical mathematical expressions")


class AcademicGenome(BaseModel):
    subject_code: str = Field(..., description="Subject code this genome belongs to")
    genes: List[KnowledgeGene] = Field(default_factory=list, description="Sequence of academic genes forming the genome")
    concept_map: Dict[str, List[str]] = Field(default_factory=dict, description="Map of concept keys to their prerequisite keys")

class DiagramObject(BaseModel):
    id: str = Field(..., description="Unique diagram identifier")
    diagram_type: str = Field(..., description="Type of diagram (flowchart, block_diagram, circuit)")
    source_code: str = Field(..., description="Renderable source code (mermaid, tikz, plantuml)")
    image_path: Optional[str] = Field(default=None, description="Path to pre-rendered image asset")

class ImageObject(BaseModel):
    id: str = Field(..., description="Unique image identifier")
    filepath: str = Field(..., description="Path to the image asset file")
    caption: str = Field(..., description="Caption text explaining the image")
    ocr_text: Optional[str] = Field(default=None, description="Extracted OCR text if applicable")
    analyzed_features: Dict[str, Any] = Field(default_factory=dict, description="Metadata or tags analyzed by OCR/ML engines")

class AnswerObject(BaseModel):
    question_id: str = Field(..., description="ID of the associated question")
    markdown_text: str = Field(..., description="Complete generated ideal answer text in markdown format")
    key_points: List[str] = Field(default_factory=list, description="Points that must be included to score full marks")
    grading_rubric: Dict[int, str] = Field(default_factory=dict, description="Criteria matched to specific marks weights")
    diagram_references: List[str] = Field(default_factory=list, description="Diagram IDs expected in the answer")

class QuestionObject(BaseModel):
    id: str = Field(..., description="Unique question identifier")
    text: str = Field(..., description="Question prompt text")
    bloom_level: str = Field(..., description="Expected Bloom's level (L1-L6)")
    marks: int = Field(..., description="Marks assigned to this question")
    question_type: str = Field(..., description="Question type (conceptual, numerical, derivation)")
    module_number: int = Field(..., description="Syllabus module number this question is derived from")
    knowledge_ids: List[str] = Field(default_factory=list, description="List of knowledge objects tested by this question")
    expected_answer: Optional[AnswerObject] = Field(default=None, description="Expected model answer object")

class PaperBlueprint(BaseModel):
    subject_code: str = Field(..., description="Subject code for the blueprint")
    scheme_year: int = Field(..., description="Scheme year (e.g., 2022, 2026)")
    total_marks: int = Field(..., description="Total marks of the question paper")
    modules_covered: List[int] = Field(default_factory=list, description="Module numbers to cover (typically 1-5)")
    question_rules: List[Dict[str, Any]] = Field(default_factory=dict, description="Structural rules for question distribution")
    is_locked: bool = Field(default=False, description="When true, blueprint attributes are frozen and cannot be mutated")

class PaperObject(BaseModel):
    id: str = Field(..., description="Unique paper identifier")
    blueprint_id: str = Field(..., description="Associated blueprint ID")
    questions: List[QuestionObject] = Field(default_factory=list, description="List of questions in the paper")
    generated_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of generation")
    version: str = Field(..., description="Semantic version of this paper generation run")
    status: PaperStatus = Field(default=PaperStatus.DRAFT, description="Workflow status state machine position")
    prompt_version: str = Field(..., description="Version of the generator prompt template used")
    engine_version: str = Field(..., description="Version of the generation engine used")

class ReviewObject(BaseModel):
    id: str = Field(..., description="Unique review identifier")
    paper_id: str = Field(..., description="Paper ID being reviewed")
    reviewer_id: str = Field(..., description="Reviewer system/human ID")
    is_approved: bool = Field(..., description="Approval status")
    comments: List[str] = Field(default_factory=list, description="Critique/comments from review")
    corrections: List[Dict[str, Any]] = Field(default_factory=dict, description="Corrective actions proposed during review")

class FeedbackObject(BaseModel):
    id: str = Field(..., description="Unique feedback identifier")
    source_type: str = Field(..., description="Feedback source (human, validator, examiner)")
    target_type: str = Field(..., description="Type of target object (question, answer, paper)")
    target_id: str = Field(..., description="ID of the target object being feedback-rated")
    rating: int = Field(..., description="Review score rating, e.g., 1-5 stars")
    suggested_edits: Optional[str] = Field(default=None, description="Suggested modifications/corrections")
    feedback_reason: str = Field(..., description="Explanation/reason for review score")

class LearningObject(BaseModel):
    episode_id: str = Field(..., description="Unique learning episode identifier")
    input_data: Dict[str, Any] = Field(..., description="Input provided to the model during the episode")
    generated_output: Dict[str, Any] = Field(..., description="Model generated output")
    human_corrections: Optional[Dict[str, Any]] = Field(default=None, description="Corrections made by a human supervisor")
    is_preferred: bool = Field(..., description="If this episode represents preferred behaviour (for alignment training)")
    version: str = Field(..., description="AOM version of the episode")
    approval_status: ApprovalStatus = Field(default=ApprovalStatus.PENDING, description="Human approval gate status before training ingestion")

class AICallLog(BaseModel):
    log_id: str = Field(..., description="Unique AI call log identifier")
    call_timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of the API call")
    engine_name: str = Field(..., description="Name of the calling engine")
    engine_version: str = Field(..., description="Version of the calling engine")
    prompt_version: str = Field(..., description="Version of the prompt template used")
    input_payload: Dict[str, Any] = Field(..., description="Complete input data/payload passed to the model")
    output_payload: Dict[str, Any] = Field(..., description="Raw output payload received from the model")
    latency_ms: float = Field(..., description="API roundtrip latency in milliseconds")
    confidence_score: float = Field(..., description="Model self-reported or logprob-derived confidence score")
    validation_passed: bool = Field(..., description="True if the output passed subsequent validation gates")
    validation_details: Dict[str, Any] = Field(default_factory=dict, description="Detailed verification metrics and error reports")

# ══════════════════════════════════════════════
# INTEGRATED AION ARCHITECTURE EXTENSIONS
# ══════════════════════════════════════════════

class ReasonCode(str, Enum):
    RC_01_CONCEPT_AMBIGUOUS = "RC-01: concept ambiguous"
    RC_02_RELATIONSHIP_MISSING = "RC-02: relationship missing"
    RC_03_EXAMINER_MISMATCH = "RC-03: examiner profile mismatch"
    RC_04_DIFFICULTY_INCONSISTENT = "RC-04: difficulty/bloom inconsistent"
    RC_05_PEDAGOGICALLY_INVALID = "RC-05: pedagogically invalid"
    RC_06_LANGUAGE_QUALITY_FAIL = "RC-06: language/grammar quality failure"
    RC_07_RETRIEVAL_INSUFFICIENT = "RC-07: retrieval insufficient (abstain path)"

class TraversalPolicy(str, Enum):
    DEPTH_FIRST = "depth_first"         # prerequisite chains
    BREADTH_FIRST = "breadth_first"     # concept connections
    BLOOM_DIRECTED = "bloom_directed"   # target taxonomy level
    EXAMINER_DIRECTED = "examiner_directed" # professor style matching
    SOCRATIC = "socratic"               # cross-domain relational reasoning

class ThoughtGraphIntent(BaseModel):
    """
    Stage 3.5 — Academic Thought Graph Traversal Intent.
    Produced between retrieval and generation to specify WHAT the question must accomplish.
    """
    intent_id: str = Field(..., description="Unique intent identifier")
    primary_concept_id: str = Field(..., description="Root concept ID traversed")
    traversed_genome_ids: List[str] = Field(default_factory=list, description="Genome node IDs along traversal path")
    traversal_policy: TraversalPolicy = Field(default=TraversalPolicy.BLOOM_DIRECTED, description="Traversal strategy applied")
    target_bloom_level: str = Field(default="L2_understand", description="Target Bloom level (L1-L6)")
    target_marks: int = Field(default=5, description="Target marks allocation (e.g. 2, 5, 10)")
    required_distractors: List[str] = Field(default_factory=list, description="Concepts to use as distractors/contrasts")
    pedagogical_objective: str = Field(..., description="Structured rationale of what this question tests")

class GRPORewardSignals(BaseModel):
    """
    Stage 4/5 — Integrated 7-Signal GRPO Reward Function.
    Calculated during GRPO training step across candidate groups.
    """
    faithfulness_score: float = Field(..., description="LettuceDetect/HHEM grounding score (0.0 - 1.0)")
    originality_score: float = Field(..., description="MinHash-LSH novelty score vs source/qbank (0.0 - 1.0)")
    llm_judge_score: float = Field(..., description="RULER/Evaluator LLM score (0.0 - 1.0)")
    examiner_fingerprint_match: float = Field(..., description="EPE Consistency Fingerprint cosine similarity (0.0 - 1.0)")
    bloom_alignment_score: float = Field(..., description="Bloom taxonomy target match (0.0 - 1.0)")
    simulation_consistency: float = Field(..., description="Concept Simulation Engine match vs ideal answer (0.0 - 1.0)")
    atg_traversal_validity: float = Field(..., description="Thought graph traversal compliance (0.0 - 1.0)")
    format_violation_penalty: float = Field(default=0.0, description="Penalty for formatting/verb violations")

class GRPORewardWeights(BaseModel):
    w1_faithfulness: float = Field(default=0.25)
    w2_originality: float = Field(default=0.15)
    w3_judge: float = Field(default=0.15)
    w4_fingerprint: float = Field(default=0.15)
    w5_bloom: float = Field(default=0.10)
    w6_simulation: float = Field(default=0.10)
    w7_atg: float = Field(default=0.10)
    w8_format_penalty: float = Field(default=0.20)

class FaithfulnessGateResult(BaseModel):
    is_faithful: bool = Field(..., description="True if no ungrounded claims detected")
    confidence_score: float = Field(..., description="Grounding confidence")
    hallucinated_tokens: List[Dict[str, Any]] = Field(default_factory=list, description="Positions and scores of hallucinated tokens")

class OriginalityGateResult(BaseModel):
    is_original: bool = Field(..., description="True if below similarity threshold vs qbank/source")
    minhash_similarity_source: float = Field(..., description="MinHash similarity vs source document")
    minhash_similarity_qbank: float = Field(..., description="MinHash similarity vs historical question bank")

class MetacognitiveMetrics(BaseModel):
    """
    Stage 6 — Metacognitive Monitor Observability Metrics.
    Aggregated to track system reasoning quality and knowledge health.
    """
    knowledge_completeness_ratio: float = Field(..., description="Ratio of verified concept coverage")
    critic_rejection_rate: float = Field(..., description="Self-Critic Ensemble rejection rate")
    reason_code_counts: Dict[str, int] = Field(default_factory=dict, description="Histogram of Reason Code failures")
    output_drift_score: float = Field(..., description="Embedding drift of generated questions over time")
    confidence_calibration_error: float = Field(..., description="Expected vs actual validation error gap")

