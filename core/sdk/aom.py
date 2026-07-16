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
