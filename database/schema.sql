-- ─────────────────────────────────────────────
-- AION Integrated Database Schema
-- Merging Design B Production Infrastructure with Design A Cognitive Architecture
-- Every table has created_at / updated_at
-- Every foreign key is explicit
-- Every status column is an ENUM
-- ─────────────────────────────────────────────

-- ══════════════════════════════════════════════
-- 1. ENUMS
-- ══════════════════════════════════════════════

CREATE TYPE doc_status AS ENUM (
    'uploaded',
    'ocr_pending',
    'ocr_done',
    'structure_pending',
    'structure_done',
    'extraction_pending',
    'extraction_done',
    'indexed',
    'failed'
);

CREATE TYPE concept_source AS ENUM (
    'gliner_auto',
    'teacher_manual',
    'llm_inferred',
    'merged'
);

CREATE TYPE relation_type AS ENUM (
    'prerequisite_of',
    'part_of',
    'contrasts_with',
    'applied_in',
    'derived_from',
    'extends',
    'equivalent_to'
);

CREATE TYPE question_status AS ENUM (
    'candidate',
    'guardrail_pass',
    'guardrail_fail',
    'faculty_accepted',
    'faculty_rejected',
    'faculty_edited',
    'exported'
);

CREATE TYPE bloom_level AS ENUM (
    'L1_remember',
    'L2_understand',
    'L3_apply',
    'L4_analyze',
    'L5_evaluate',
    'L6_create'
);

CREATE TYPE conflict_class AS ENUM (
    'terminology',
    'scope',
    'factual',
    'evolution'
);

CREATE TYPE generation_mode AS ENUM (
    'best_of_n',
    'grpo_sampled',
    'retry_corrective'
);

CREATE TYPE reason_code AS ENUM (
    'RC_01_CONCEPT_AMBIGUOUS',
    'RC_02_RELATIONSHIP_MISSING',
    'RC_03_EXAMINER_MISMATCH',
    'RC_04_DIFFICULTY_INCONSISTENT',
    'RC_05_PEDAGOGICALLY_INVALID',
    'RC_06_LANGUAGE_QUALITY_FAIL',
    'RC_07_RETRIEVAL_INSUFFICIENT'
);

CREATE TYPE traversal_policy AS ENUM (
    'DEPTH_FIRST',
    'BREADTH_FIRST',
    'BLOOM_DIRECTED',
    'EXAMINER_DIRECTED',
    'SOCRATIC'
);

-- ══════════════════════════════════════════════
-- 2. ORGANIZATIONAL & DOCUMENT TABLES
-- ══════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS institutions (
    institution_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name                TEXT NOT NULL,
    domain              TEXT,                    -- e.g. "vtu", "anna_university"
    created_at          TIMESTAMPTZ DEFAULT now(),
    updated_at          TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS departments (
    department_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    institution_id      UUID NOT NULL REFERENCES institutions(institution_id) ON DELETE CASCADE,
    name                TEXT NOT NULL,           -- e.g. "AIML", "Civil"
    created_at          TIMESTAMPTZ DEFAULT now(),
    updated_at          TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS subjects (
    subject_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    department_id       UUID NOT NULL REFERENCES departments(department_id) ON DELETE CASCADE,
    name                TEXT NOT NULL,           -- e.g. "Machine Learning"
    code                TEXT,                    -- e.g. "21AI51"
    semester            INTEGER,
    syllabus_version    TEXT,
    created_at          TIMESTAMPTZ DEFAULT now(),
    updated_at          TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS users (
    user_id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    institution_id      UUID REFERENCES institutions(institution_id),
    email               TEXT UNIQUE NOT NULL,
    role                TEXT NOT NULL,           -- 'faculty','admin','student'
    name                TEXT,
    created_at          TIMESTAMPTZ DEFAULT now(),
    updated_at          TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS documents (
    doc_id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    subject_id          UUID REFERENCES subjects(subject_id) ON DELETE SET NULL,
    institution_id      UUID REFERENCES institutions(institution_id) ON DELETE SET NULL,
    original_filename   TEXT NOT NULL,
    minio_bucket        TEXT NOT NULL,
    minio_key           TEXT NOT NULL,
    file_type           TEXT NOT NULL,           -- 'pdf', 'docx', 'txt'
    file_size_bytes     BIGINT,
    doc_status          doc_status DEFAULT 'uploaded',
    page_count          INTEGER,
    language            TEXT DEFAULT 'en',
    uploaded_by         UUID REFERENCES users(user_id) ON DELETE SET NULL,
    created_at          TIMESTAMPTZ DEFAULT now(),
    updated_at          TIMESTAMPTZ DEFAULT now()
);

-- ══════════════════════════════════════════════
-- 3. STAGE 1 & 2 — DOCUMENT UNDERSTANDING & GENOMES
-- ══════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS document_pages (
    page_id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id              UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    page_number         INTEGER NOT NULL,
    raw_text            TEXT,
    cleaned_text        TEXT,
    layout_json         JSONB,                   -- Docling hierarchical layout tree
    has_table           BOOLEAN DEFAULT FALSE,
    has_formula         BOOLEAN DEFAULT FALSE,
    has_figure          BOOLEAN DEFAULT FALSE,
    ocr_confidence      FLOAT,
    created_at          TIMESTAMPTZ DEFAULT now(),
    UNIQUE (doc_id, page_number)
);

CREATE TABLE IF NOT EXISTS document_structure (
    structure_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id              UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    chapter_number      INTEGER,
    chapter_title       TEXT,
    section_number      TEXT,                    -- e.g. "3.2.1"
    section_title       TEXT,
    block_type          TEXT NOT NULL,           -- 'paragraph','table','formula','figure','list'
    block_index         INTEGER NOT NULL,
    content_text        TEXT,
    content_json        JSONB,
    minio_key_figure    TEXT,
    page_start          INTEGER,
    page_end            INTEGER,
    parsed_by           TEXT NOT NULL,           -- 'docling','unlimited_ocr','table_transformer'
    parse_confidence    FLOAT,
    created_at          TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS document_tables (
    table_id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    structure_id        UUID NOT NULL REFERENCES document_structure(structure_id) ON DELETE CASCADE,
    doc_id              UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    raw_html            TEXT,
    structured_json     JSONB,
    transformer_verified BOOLEAN DEFAULT FALSE,
    transformer_confidence FLOAT,
    created_at          TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS document_formulas (
    formula_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    structure_id        UUID NOT NULL REFERENCES document_structure(structure_id) ON DELETE CASCADE,
    doc_id              UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    latex_source        TEXT,
    rendered_minio_key  TEXT,
    formula_context     TEXT,
    created_at          TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS concepts (
    concept_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    subject_id          UUID REFERENCES subjects(subject_id) ON DELETE CASCADE,
    name                TEXT NOT NULL,
    definition          TEXT,
    definition_source   TEXT,
    formula_latex       TEXT,
    diagram_description TEXT,
    applications        TEXT[],
    bloom_levels        bloom_level[],
    difficulty_estimate FLOAT DEFAULT 0.5,
    confidence          FLOAT DEFAULT 0.5,
    source_type         concept_source DEFAULT 'gliner_auto',
    source_doc_ids      UUID[],
    version             INTEGER DEFAULT 1,
    is_frozen           BOOLEAN DEFAULT FALSE,
    created_at          TIMESTAMPTZ DEFAULT now(),
    updated_at          TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS concept_chunks (
    mapping_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    concept_id          UUID NOT NULL REFERENCES concepts(concept_id) ON DELETE CASCADE,
    structure_id        UUID NOT NULL REFERENCES document_structure(structure_id) ON DELETE CASCADE,
    doc_id              UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    relevance_score     FLOAT,
    is_definition_source BOOLEAN DEFAULT FALSE,
    created_at          TIMESTAMPTZ DEFAULT now(),
    UNIQUE (concept_id, structure_id)
);

CREATE TABLE IF NOT EXISTS concept_relations (
    relation_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_concept_id   UUID NOT NULL REFERENCES concepts(concept_id) ON DELETE CASCADE,
    target_concept_id   UUID NOT NULL REFERENCES concepts(concept_id) ON DELETE CASCADE,
    relation_type       relation_type NOT NULL,
    confidence          FLOAT DEFAULT 0.5,
    extracted_by        TEXT NOT NULL,           -- 'gliner_relex','teacher','llm'
    source_doc_id       UUID REFERENCES documents(doc_id) ON DELETE SET NULL,
    source_chunk_id     UUID REFERENCES document_structure(structure_id) ON DELETE SET NULL,
    created_at          TIMESTAMPTZ DEFAULT now(),
    UNIQUE (source_concept_id, target_concept_id, relation_type)
);

CREATE TABLE IF NOT EXISTS concept_conflicts (
    conflict_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    concept_id          UUID NOT NULL REFERENCES concepts(concept_id) ON DELETE CASCADE,
    conflict_class      conflict_class NOT NULL,
    source_a_doc_id     UUID REFERENCES documents(doc_id),
    source_a_text       TEXT,
    source_b_doc_id     UUID REFERENCES documents(doc_id),
    source_b_text       TEXT,
    aion_resolution     TEXT,
    teacher_resolved    BOOLEAN DEFAULT FALSE,
    teacher_decision    TEXT,
    resolved_at         TIMESTAMPTZ,
    created_at          TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS concept_history (
    history_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    concept_id          UUID NOT NULL REFERENCES concepts(concept_id) ON DELETE CASCADE,
    version             INTEGER NOT NULL,
    changed_field       TEXT NOT NULL,
    old_value           TEXT,
    new_value           TEXT,
    confidence_before   FLOAT,
    confidence_after    FLOAT,
    change_trigger      TEXT NOT NULL,           -- 'new_document','teacher_edit','conflict_resolved'
    triggered_by_doc_id UUID REFERENCES documents(doc_id),
    created_at          TIMESTAMPTZ DEFAULT now()
);

-- ══════════════════════════════════════════════
-- 4. EXAMINER PROFILES & QUESTION GENERATION
-- ══════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS examiner_profiles (
    examiner_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    institution_id      UUID REFERENCES institutions(institution_id) ON DELETE CASCADE,
    name                TEXT NOT NULL,
    avg_question_length_tokens FLOAT,
    preferred_bloom_levels     bloom_level[],
    diagram_frequency   FLOAT DEFAULT 0.0,
    numerical_ratio     FLOAT DEFAULT 0.0,
    two_part_frequency  FLOAT DEFAULT 0.0,
    preferred_mark_values INTEGER[],
    profile_confidence  FLOAT DEFAULT 0.1,
    papers_seen         INTEGER DEFAULT 0,
    fingerprint_vector  FLOAT[],
    created_at          TIMESTAMPTZ DEFAULT now(),
    updated_at          TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS examiner_question_papers (
    paper_id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    examiner_id         UUID REFERENCES examiner_profiles(examiner_id) ON DELETE CASCADE,
    subject_id          UUID REFERENCES subjects(subject_id) ON DELETE CASCADE,
    doc_id              UUID REFERENCES documents(doc_id) ON DELETE SET NULL,
    year                INTEGER,
    semester            TEXT,
    created_at          TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS generation_requests (
    request_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    subject_id          UUID REFERENCES subjects(subject_id) ON DELETE CASCADE,
    examiner_id         UUID REFERENCES examiner_profiles(examiner_id) ON DELETE SET NULL,
    requested_by        UUID REFERENCES users(user_id) ON DELETE SET NULL,
    mark_value          INTEGER NOT NULL,
    bloom_target        bloom_level,
    traversal_policy    traversal_policy DEFAULT 'BLOOM_DIRECTED',
    topic_hint          TEXT,
    concept_ids         UUID[],
    num_questions       INTEGER DEFAULT 1,
    generation_mode     generation_mode DEFAULT 'best_of_n',
    model_checkpoint    TEXT NOT NULL,
    lora_adapter_id     TEXT,
    status              TEXT DEFAULT 'pending',
    completed_at        TIMESTAMPTZ,
    created_at          TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS retrieval_results (
    retrieval_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id          UUID NOT NULL REFERENCES generation_requests(request_id) ON DELETE CASCADE,
    query_text          TEXT NOT NULL,
    retrieved_chunks    JSONB NOT NULL,
    graph_neighbors     JSONB,
    total_chunks_retrieved INTEGER,
    total_chunks_after_rerank INTEGER,
    created_at          TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS questions (
    question_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id          UUID NOT NULL REFERENCES generation_requests(request_id) ON DELETE CASCADE,
    retrieval_id        UUID NOT NULL REFERENCES retrieval_results(retrieval_id) ON DELETE CASCADE,
    concept_id          UUID REFERENCES concepts(concept_id) ON DELETE SET NULL,
    subject_id          UUID REFERENCES subjects(subject_id) ON DELETE CASCADE,
    ideal_answer        TEXT NOT NULL,
    marking_scheme      JSONB NOT NULL,
    question_constraints JSONB,
    question_text       TEXT NOT NULL,
    mark_value          INTEGER NOT NULL,
    bloom_level         bloom_level,
    is_numerical        BOOLEAN DEFAULT FALSE,
    diagram_required    BOOLEAN DEFAULT FALSE,
    model_checkpoint    TEXT NOT NULL,
    lora_adapter_id     TEXT,
    candidate_rank      INTEGER,
    question_status     question_status DEFAULT 'candidate',
    version             INTEGER DEFAULT 1,
    parent_question_id  UUID REFERENCES questions(question_id) ON DELETE SET NULL,
    created_at          TIMESTAMPTZ DEFAULT now(),
    updated_at          TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS question_history (
    history_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    question_id         UUID NOT NULL REFERENCES questions(question_id) ON DELETE CASCADE,
    version             INTEGER NOT NULL,
    old_question_text   TEXT,
    new_question_text   TEXT,
    change_trigger      TEXT NOT NULL,
    changed_by          UUID REFERENCES users(user_id) ON DELETE SET NULL,
    created_at          TIMESTAMPTZ DEFAULT now()
);

-- ══════════════════════════════════════════════
-- 5. GUARDRAILS, REVIEWS & TRAINING
-- ══════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS guardrail_results (
    guardrail_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    question_id         UUID NOT NULL REFERENCES questions(question_id) ON DELETE CASCADE,
    faithfulness_model  TEXT NOT NULL,
    faithfulness_score  FLOAT NOT NULL,
    faithfulness_pass   BOOLEAN NOT NULL,
    hallucinated_tokens JSONB,
    ngram_overlap_source    FLOAT,
    ngram_overlap_qbank     FLOAT,
    embedding_sim_source    FLOAT,
    embedding_sim_qbank     FLOAT,
    originality_pass        BOOLEAN NOT NULL,
    format_violations   TEXT[],
    format_pass         BOOLEAN NOT NULL,
    overall_pass        BOOLEAN NOT NULL,
    retry_count         INTEGER DEFAULT 0,
    failure_reason_code reason_code,
    created_at          TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS faculty_reviews (
    review_id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    question_id         UUID NOT NULL REFERENCES questions(question_id) ON DELETE CASCADE,
    reviewer_id         UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    decision            TEXT NOT NULL,           -- 'accept','reject','edit'
    edited_question     TEXT,
    edited_ideal_answer TEXT,
    rejection_reason    TEXT,
    rating_accuracy     INTEGER,
    rating_clarity      INTEGER,
    rating_difficulty   INTEGER,
    rating_exam_feel    INTEGER,
    dpo_chosen          TEXT,
    dpo_rejected        TEXT,
    dpo_recorded        BOOLEAN DEFAULT FALSE,
    created_at          TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS training_runs (
    run_id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    mlflow_run_id       TEXT UNIQUE,
    run_type            TEXT NOT NULL,           -- 'raft_sft','grpo','dpo'
    base_model          TEXT NOT NULL,
    lora_adapter_out    TEXT NOT NULL,
    dataset_version     TEXT,
    hyperparams         JSONB,
    final_loss          FLOAT,
    final_reward        FLOAT,
    kl_divergence       FLOAT,
    faithfulness_avg    FLOAT,
    originality_avg     FLOAT,
    status              TEXT DEFAULT 'running',
    started_at          TIMESTAMPTZ DEFAULT now(),
    completed_at        TIMESTAMPTZ,
    created_at          TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS raft_training_examples (
    example_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id              UUID REFERENCES training_runs(run_id) ON DELETE CASCADE,
    concept_id          UUID REFERENCES concepts(concept_id) ON DELETE CASCADE,
    query               TEXT NOT NULL,
    golden_chunk_ids    UUID[],
    distractor_chunk_ids UUID[],
    is_golden_present   BOOLEAN NOT NULL,
    chain_of_thought    TEXT,
    ideal_answer        TEXT NOT NULL,
    marking_scheme      JSONB,
    question_text       TEXT NOT NULL,
    created_at          TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS grpo_samples (
    sample_id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id              UUID REFERENCES training_runs(run_id) ON DELETE CASCADE,
    request_id          UUID REFERENCES generation_requests(request_id) ON DELETE CASCADE,
    candidates          JSONB NOT NULL,
    group_size          INTEGER NOT NULL,
    mean_reward         FLOAT,
    std_reward          FLOAT,
    kl_penalty          FLOAT,
    created_at          TIMESTAMPTZ DEFAULT now()
);

-- ══════════════════════════════════════════════
-- 6. INDEXES
-- ══════════════════════════════════════════════

CREATE INDEX IF NOT EXISTS idx_documents_subject        ON documents(subject_id);
CREATE INDEX IF NOT EXISTS idx_documents_status         ON documents(doc_status);
CREATE INDEX IF NOT EXISTS idx_pages_doc               ON document_pages(doc_id);
CREATE INDEX IF NOT EXISTS idx_structure_doc           ON document_structure(doc_id);
CREATE INDEX IF NOT EXISTS idx_structure_type          ON document_structure(block_type);
CREATE INDEX IF NOT EXISTS idx_concept_subject         ON concepts(subject_id);
CREATE INDEX IF NOT EXISTS idx_concept_confidence      ON concepts(confidence);
CREATE INDEX IF NOT EXISTS idx_concept_frozen          ON concepts(is_frozen);
CREATE INDEX IF NOT EXISTS idx_relations_source        ON concept_relations(source_concept_id);
CREATE INDEX IF NOT EXISTS idx_relations_target        ON concept_relations(target_concept_id);
CREATE INDEX IF NOT EXISTS idx_chunk_concept           ON concept_chunks(concept_id);
CREATE INDEX IF NOT EXISTS idx_questions_status        ON questions(question_status);
CREATE INDEX IF NOT EXISTS idx_questions_concept       ON questions(concept_id);
CREATE INDEX IF NOT EXISTS idx_guardrail_question      ON guardrail_results(question_id);
CREATE INDEX IF NOT EXISTS idx_faculty_review_question ON faculty_reviews(question_id);
