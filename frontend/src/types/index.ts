// ─────────────────────────────────────────────────────────────────────────────
//  SHARED TYPE DEFINITIONS
//  This is the single source of truth for all data shapes.
//  Both the frontend and the Python backend team should agree on these.
//  To add a new entity: add its types here, then add api/ + mock/ + hook files.
// ─────────────────────────────────────────────────────────────────────────────

// ── Auth ─────────────────────────────────────────────────────────────────────

export interface Teacher {
  id: string;
  name: string;
  email: string;
  designation: string;
  employeeId: string;
  department: string;
}

export interface TeacherSubject {
  id: string;
  code: string;
  name: string;
  department: string;
  semester: string;
  addedOn: string; // ISO date string
}

// ── Materials ────────────────────────────────────────────────────────────────

export type MaterialType = "notes" | "textbook";
export type ProcessingStatus = "processing" | "processed" | "failed";

export interface Material {
  id: string;
  name: string;
  type: MaterialType;
  module: number | "all";
  subject: string;
  subjectCode: string;
  size: string;
  uploadedOn: string;
  status: ProcessingStatus;
  pages: number;
  images: number;
  chunks: number;
}

export interface MaterialFilters {
  subjectCode?: string;
  module?: number | "all";
}

export interface UploadMaterialPayload {
  file: File;
  subjectCode: string;
  module: number | "all";
  type: MaterialType;
}

// ── Knowledge Base ────────────────────────────────────────────────────────────

export type ImageType = "diagram" | "figure" | "chart" | "table" | "equation";

export interface KnowledgeChunk {
  id: string;
  subjectCode: string;
  module: number;
  source: string;
  sourceType: MaterialType;
  page: number;
  text: string;
  tokens: number;
}

export interface KnowledgeImage {
  id: string;
  subjectCode: string;
  module: number;
  source: string;
  page: number;
  caption: string;
  type: ImageType;
  color: string; // tailwind bg class for placeholder
}

export interface KnowledgeFilters {
  subjectCode: string;
  module?: number | "all";
  search?: string;
}

export interface KnowledgeStats {
  totalChunks: number;
  totalImages: number;
  totalTokens: number;
  byModule: Array<{ module: number; chunks: number; images: number }>;
}

// ── Syllabus ──────────────────────────────────────────────────────────────────

export interface SyllabusModule {
  number: number;
  title: string;
  topics: string[];
  co: string;        // "CO1" | "CO2" | "CO3"
  bloomLevels: string[]; // ["L1","L2"]
}

export interface Syllabus {
  subjectCode: string;
  modules: SyllabusModule[];
  updatedAt?: string;
}

// ── Paper Generation ──────────────────────────────────────────────────────────

export interface GenerateRequest {
  subjectCode: string;
  examType: "IAT-1" | "IAT-2" | "SEE";
  department: string;
  semester: string;
  batch: string;
  maxMarks: number;
  duration: string;
  dateOfExam?: string;
  teachingDept?: string;
  prompt?: string;        // optional freetext override
  modules?: number[];     // which modules to include
}

export interface GenerationProgress {
  step: number;
  message: string;
  pct: number; // 0-100
}

// ── Paper / Question ──────────────────────────────────────────────────────────

export type PaperStatus = "draft" | "reviewed" | "finalized" | "submitted";
export type BloomLevel = "L1" | "L2" | "L3" | "L4" | "L5" | "L6";

export interface PaperQuestion {
  id: string;
  module: number;
  co: string;
  bloom: BloomLevel;
  part: string;        // "A" | "B" | "C"
  questionNumber: string; // "1", "2a", "2b"
  text: string;
  marks: number;
  hasOr: boolean;
  orText?: string;
  diagramRequired: boolean;
  diagramUrl?: string;  // pre-signed S3 URL when real backend is connected
  diagramCaption?: string;
}

export interface Paper {
  id: string;
  subjectCode: string;
  subjectName: string;
  subjectCode2?: string; // alias field used in some pages
  examType: string;
  department: string;
  semester: string;
  batch: string;
  duration: string;
  maxMarks: number;
  dateOfIat?: string;
  teachingDept?: string;
  status: PaperStatus;
  questions: PaperQuestion[];
  createdAt: string;
  finalizedAt?: string;
}

// ── Settings ──────────────────────────────────────────────────────────────────

export type LearnType = "framing" | "marks" | "diagrams" | "full-paper";
export type TrainingStatus = "processing" | "trained" | "failed";
export type ExamType = "IAT-1" | "IAT-2" | "SEE";

export interface TemplateBlock {
  id: string;
  label: string;
  instruction: string;
  questions: number;
  attemptAll: boolean;
  attemptCount: number;
  marksPerQuestion: number;
  marksMin: number;
  marksMax: number;
  hasOrPattern: boolean;
}

export interface ExamTemplate {
  examType: ExamType;
  totalMarks: number;
  duration: string;
  blocks: TemplateBlock[];
  referenceFileName: string;
  randomizationEnabled: boolean;
}

export interface TrainingFile {
  id: string;
  name: string;
  size: string;
  uploadedOn: string;
  learnType: LearnType;
  examType: string;
  status: TrainingStatus;
  questionsLearned: number;
}

export interface InstitutionSettings {
  name: string;
  affiliations: string;
  city: string;
  university: string;
  showRbtLegend: boolean;
  showCoTable: boolean;
}
