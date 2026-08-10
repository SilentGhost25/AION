"""
AION Unified Pipeline
=====================
THE single execution path for all generation.
No legacy paths. No alternative entry points.
Every request goes through this function.

Contract flow:
    RawFile
    → ExtractionResult  (S1)
    → CleanedContent    (S2)
    → ChunkedContent    (S3)
    → RetrievedEvidence (S4)
    → GenerationRequest (S5)
    → [LLM]             (S6)
    → ValidatedQuestion (S7)
    → FinalPaper        (S8)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Any

from .contracts import (
    RawFile, ExtractionResult, CleanedContent, ChunkedContent,
    RetrievedEvidence, GenerationRequest, GeneratedQuestion,
    ValidatedQuestion, PaperDraft, FinalPaper, PipelineHealth,
    AcademicChunk, Evidence, QuestionSpec,
    ExamType, Difficulty, ValidationVerdict, ContractViolation,
    require_contract,
)
from .execution_auditor import ExecutionAuditor
from core.config.production_model import get_production_model


def run_unified(
    file_path:     str,
    exam_type:     str = "IA",
    difficulty:    str = "Mixed",
    subject:       str = "",
    department:    str = "",
    max_questions: int = 10,
    mode:          str = "standard",
) -> FinalPaper:
    """
    Single entry point for all question generation.
    Returns FinalPaper contract. Raises on unrecoverable failure.
    """
    t0     = time.time()
    model  = get_production_model()
    health = PipelineHealth()

    print(f"\n{'='*60}")
    print(f"[UNIFIED] {Path(file_path).name}")
    print(f"[UNIFIED] Model={model} Exam={exam_type}")
    print(f"{'='*60}")

    try:
        raw = RawFile(
            path       = file_path,
            subject    = subject,
            department = department,
            exam_type  = ExamType.IA if exam_type.upper() in ("IA", "IAT1", "IAT2", "IAT3", "MID") else ExamType.SEE,
            difficulty = Difficulty.MIXED,
            health     = health,
        )
    except ContractViolation as e:
        raise RuntimeError(f"Invalid input: {e}") from e

    auditor = ExecutionAuditor(raw.doc_id)

    # ── S1: Extract ───────────────────────────────────────────────────────────
    t1  = time.time()
    ext = _extract(raw)
    auditor.audit_extraction(ext, ext.pipeline_used if ext else "none",
                             elapsed_ms=(time.time()-t1)*1000)
    if ext is None:
        raise RuntimeError("S1_EXTRACTION failed: no text extracted")

    # ── S2: Clean (MUST run before S3) ───────────────────────────────────────
    t2    = time.time()
    clean = _clean(ext)
    auditor.audit_cleaning(clean, ran_before_validator=True,
                           elapsed_ms=(time.time()-t2)*1000)
    if clean is None:
        health.deduct(30, "S2_CLEANING failed — using raw text")
        clean = _emergency_clean(ext)

    # ── S3: Chunk + Validate ──────────────────────────────────────────────────
    t3      = time.time()
    chunked = _chunk_and_validate(clean, health)
    auditor.audit_chunking(chunked, elapsed_ms=(time.time()-t3)*1000)

    if chunked is None:
        raise RuntimeError(
            "S3_CHUNKING: Zero valid academic chunks. "
            "Document appears corrupted or non-academic."
        )

    # ── S4: Retrieve + Ground ─────────────────────────────────────────────────
    t4       = time.time()
    evidence = _retrieve(chunked, subject, health)
    auditor.audit_retrieval(evidence, elapsed_ms=(time.time()-t4)*1000)

    if evidence is None:
        raise RuntimeError(
            "S4_RETRIEVAL: No grounded evidence available. "
            "Cannot generate questions without verified academic content."
        )

    # ── Pre-generation safety check ───────────────────────────────────────────
    safe, reason = auditor.is_safe_to_generate()
    if not safe:
        raise RuntimeError(f"Generation blocked by auditor: {reason}")

    # ── S5: Build template ────────────────────────────────────────────────────
    gen_request = _build_generation_request(
        raw, evidence, max_questions, health
    )

    # ── S6: LLM fills question text ───────────────────────────────────────────
    t6 = time.time()
    generated, n_fallbacks = _generate(gen_request, model)
    auditor.audit_generation(
        n_questions_requested = len(gen_request.specs),
        n_questions_produced  = len(generated),
        n_fallbacks           = n_fallbacks,
        elapsed_ms            = (time.time()-t6)*1000,
    )

    # ── S7: Critic ────────────────────────────────────────────────────────────
    t7        = time.time()
    validated = _critic(generated, chunked)
    n_passed  = sum(1 for v in validated if v.verdict == ValidationVerdict.PASS)
    n_repaired= sum(1 for v in validated if v.was_repaired)
    n_failed  = sum(1 for v in validated if v.verdict == ValidationVerdict.FAIL)
    auditor.audit_critic(len(validated), n_passed, n_repaired, n_failed,
                         elapsed_ms=(time.time()-t7)*1000)

    # ── S8: Assemble final paper ──────────────────────────────────────────────
    paper = _assemble(raw, validated, health)
    auditor.audit_final_paper(paper)

    safe_export, export_reason = auditor.is_safe_to_export()
    if not safe_export:
        health.deduct(20, export_reason)
        print(f"[UNIFIED] WARNING: {export_reason}")

    paper.session_log = auditor.report.to_dict().get("stages", [])

    elapsed = round(time.time() - t0, 1)
    print(f"[UNIFIED] Done in {elapsed}s | "
          f"trust={auditor.report.trust_score} | "
          f"health={health.score}")

    return paper


# ── Stage implementations ──────────────────────────────────────────────────────

def _extract(raw: RawFile) -> Optional[ExtractionResult]:
    require_contract(raw, RawFile, "S1_EXTRACT")
    from .extractor_gateway import extract_document
    try:
        res = extract_document(raw.path, health=raw.health)
        # Ensure doc_id matches raw doc_id
        res.doc_id = raw.doc_id
        return res
    except Exception as e:
        print(f"[S1] Extraction failed: {e}")
        return None


def _clean(ext: ExtractionResult) -> Optional[CleanedContent]:
    require_contract(ext, ExtractionResult, "S2_CLEAN")
    try:
        from .pdf_artifact_filter import filter_pdf_artifacts
        from .cleaner             import semantic_clean
        cleaned = filter_pdf_artifacts(ext.raw_text)
        cleaned = semantic_clean(cleaned)
        return CleanedContent(
            doc_id=ext.doc_id,
            clean_text=cleaned,
            original_words=ext.word_count,
            clean_words=len(cleaned.split()),
            artifacts_removed=ext.word_count - len(cleaned.split()),
            health=ext.health,
        )
    except Exception as e:
        print(f"[S2] Cleaning failed: {e}")
    return None


def _emergency_clean(ext: ExtractionResult) -> CleanedContent:
    import re
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', ext.raw_text)
    text = re.sub(r'\s+', ' ', text).strip()
    return CleanedContent(
        doc_id=ext.doc_id, clean_text=text,
        original_words=ext.word_count, clean_words=len(text.split()),
        artifacts_removed=0, health=ext.health,
    )


def _chunk_and_validate(
    clean: CleanedContent, health: PipelineHealth
) -> Optional[ChunkedContent]:
    require_contract(clean, CleanedContent, "S3_CHUNK")
    try:
        from .shp.content_healer import ContentHealer
        from .shp.error_knowledge import ErrorKnowledgeBase
        healed = ContentHealer(ErrorKnowledgeBase()).heal(
            clean.clean_text, chunk_size=300, overlap=30
        )
        if not healed.chunks:
            return None

        chunks = []
        for i, text in enumerate(healed.chunks):
            chunks.append(AcademicChunk(
                chunk_id     = f"chunk_{i:04d}",
                text         = text,
                word_count   = len(text.split()),
                module_index = 1,
            ))

        return ChunkedContent(
            doc_id         = clean.doc_id,
            chunks         = chunks,
            modules        = [{"module_index": 1, "title": "Module 1"}],
            threshold_used = healed.thresholds_used[0] if healed.thresholds_used else 0.7,
            health         = health,
        )
    except ContractViolation:
        return None
    except Exception as e:
        print(f"[S3] Chunking failed: {e}")
        return None


def _retrieve(
    chunked: ChunkedContent, subject: str, health: PipelineHealth
) -> Optional[RetrievedEvidence]:
    require_contract(chunked, ChunkedContent, "S4_RETRIEVE")
    try:
        from .retriever      import GroundedRetriever
        from .grounding_gate import check_grounding

        ret       = GroundedRetriever(max_chunks=3)
        all_texts = [c.text for c in chunked.chunks]
        query     = f"{subject} concepts and principles" if subject else "academic concepts"

        top  = ret.retrieve_texts(query, all_texts)
        gate = check_grounding(query, top)

        if not gate.proceed:
            health.deduct(15, f"Grounding gate failed: {gate.reason}")
            if top:
                top = top[:2]
            else:
                return None

        pruned = gate.pruned_chunks if gate.proceed else top

        evidence = Evidence(
            chunk_ids      = [f"chunk_{i}" for i in range(len(pruned))],
            texts          = pruned,
            combined_text  = "\n\n".join(pruned),
            module_index   = 1,
            evidence_score = gate.evidence_score if gate.proceed else 0.5,
            word_count     = sum(len(c.split()) for c in pruned),
            query          = query,
        )

        return RetrievedEvidence(
            doc_id             = chunked.doc_id,
            evidence_by_module = {1: evidence},
            health             = health,
        )
    except ContractViolation:
        return None
    except Exception as e:
        print(f"[S4] Retrieval failed: {e}")
        return None


def _build_generation_request(
    raw: RawFile,
    evidence: RetrievedEvidence,
    max_questions: int,
    health: PipelineHealth,
) -> GenerationRequest:
    from .paper_template import PaperTemplateBuilder
    builder  = PaperTemplateBuilder()
    n_mods = max(5, len(evidence.evidence_by_module)) if raw.exam_type in (ExamType.IA, ExamType.SEE) else max(1, len(evidence.evidence_by_module))
    template = builder.build(
        exam_type = raw.exam_type.value,
        n_modules = n_mods,
        subject   = raw.subject,
    )

    specs = []
    for q_slot in template.question_slots[:max_questions]:
        ev = evidence.get(q_slot.module_index) or list(evidence.evidence_by_module.values())[0]
        for sub in q_slot.sub_slots:
            specs.append(QuestionSpec(
                spec_id      = sub.slot_id,
                module_index = q_slot.module_index,
                q_number     = q_slot.q_number,
                part_letter  = sub.letter,
                marks        = sub.marks,
                bloom_level  = sub.bloom_level,
                bloom_verb   = sub.verb,
                co           = sub.co,
                is_or        = q_slot.is_or,
                evidence     = ev,
                exam_type    = raw.exam_type,
            ))

    return GenerationRequest(
        doc_id      = raw.doc_id,
        specs       = specs,
        exam_type   = raw.exam_type,
        subject     = raw.subject,
        total_marks = template.attemptable_marks,
        health      = health,
    )


def _generate(
    req: GenerationRequest, model: str
) -> tuple[list[GeneratedQuestion], int]:
    require_contract(req, GenerationRequest, "S6_GENERATE")
    from .slot_filler import fill_slot
    from .vre import VREEngine, VRERequest, FigureInput, VREDecisionState

    generated   = []
    n_fallbacks = 0

    for spec in req.specs:
        ctx  = spec.evidence.combined_text[:1000]
        text = ""
        v_output = None

        # Execute VRE check
        vre_req = VRERequest(
            request_id=spec.spec_id,
            subject=req.subject,
            department="",
            module=f"module_{spec.module_index}",
            topic=spec.evidence.query,
            bloom_level=f"L{spec.bloom_level}",
            marks=spec.marks,
            figure_candidates=[FigureInput(image_path=req.doc_id)],
        )

        try:
            v_output = VREEngine.execute(vre_req)
            if v_output.success and v_output.decision_state == VREDecisionState.IMAGE_NEEDED_AND_VALID:
                text = v_output.text
        except Exception:
            text = ""

        if not text:
            text = fill_slot(spec, ctx, req.subject)

        try:
            q = GeneratedQuestion(
                spec_id       = spec.spec_id,
                question_text = text,
                spec          = spec,
            )
            if v_output and v_output.figure_svg:
                setattr(q, "figure_svg", v_output.figure_svg)
                setattr(q, "provenance", v_output.provenance)
            generated.append(q)
        except ContractViolation:
            n_fallbacks += 1

    return generated, n_fallbacks


def _critic(
    questions: list[GeneratedQuestion],
    chunked:   Any,
) -> list[ValidatedQuestion]:
    from .critic import review_extended
    from core.validators.evidence_validator import EvidenceValidator
    from .module_alignment import ModuleAlignmentValidator
    from .question_completeness import QuestionCompletenessValidator

    chunks_list = getattr(chunked, "chunks", [])
    all_texts = [getattr(c, "text", getattr(c, "content", "")) for c in chunks_list]
    retrieved_dict_list = [
        {"chunk_id": getattr(c, "chunk_id", f"chk_{i}"), "text": getattr(c, "text", getattr(c, "content", "")), "page": getattr(c, "page", 1)}
        for i, c in enumerate(chunks_list)
    ]

    validated = []

    for q in questions:
        verdict_obj = review_extended(
            question        = q.question_text,
            evidence_chunks = all_texts,
            bloom_level     = q.spec.bloom_level,
        )

        comp_valid, comp_errors = QuestionCompletenessValidator.validate(q.question_text)
        ev_res = EvidenceValidator.validate(
            question_text=q.question_text,
            retrieved_chunks=retrieved_dict_list,
            target_module=q.spec.module_index,
            target_bloom=q.spec.bloom_level,
        )
        mod_res = ModuleAlignmentValidator.validate(
            question_text=q.question_text,
            target_module=q.spec.module_index,
        )

        passed = verdict_obj.passed and comp_valid and ev_res.passed and mod_res.passed

        vq = ValidatedQuestion(
            question = q,
            verdict  = ValidationVerdict.PASS if passed else ValidationVerdict.FAIL,
            score    = verdict_obj.score if passed else 0.40,
            issues   = [verdict_obj.reason] if not passed else [],
        )
        validated.append(vq)

    return validated


def _assemble(
    raw: RawFile,
    validated: list[ValidatedQuestion],
    health: PipelineHealth,
) -> FinalPaper:
    modules_dict: dict[int, dict] = {}
    for vq in validated:
        spec = vq.question.spec
        mi   = spec.module_index
        if mi not in modules_dict:
            modules_dict[mi] = {
                "module_index": mi,
                "module_title": f"Module {mi}",
                "questions":    [],
            }

        q_num   = spec.q_number
        q_entry = next(
            (q for q in modules_dict[mi]["questions"]
             if q["mqIndex"] == q_num and q["isOr"] == spec.is_or),
            None
        )
        if q_entry is None:
            q_entry = {
                "mqIndex":      q_num,
                "totalMarks":   10 if raw.exam_type == ExamType.IA else 20,
                "bloomLevel":   spec.bloom_level,
                "bloomName":    "Understand",
                "isOr":         spec.is_or,
                "subQuestions": [],
            }
            modules_dict[mi]["questions"].append(q_entry)

        sub_item = {
            "letter": spec.part_letter,
            "text":   vq.question.question_text,
            "marks":  spec.marks,
            "co":     spec.co,
            "bloom":  spec.bloom_level,
        }

        # Include VRE figure SVG & provenance if attached
        if hasattr(vq.question, "figure_svg"):
            sub_item["figure_svg"] = getattr(vq.question, "figure_svg")
        if hasattr(vq.question, "provenance"):
            prov = getattr(vq.question, "provenance")
            if prov and hasattr(prov, "__dict__"):
                sub_item["provenance"] = prov.__dict__

        q_entry["subQuestions"].append(sub_item)

    # Check for total attemptable mark completeness (taking max per OR pair per module)
    total_paper_marks = 0
    for m in modules_dict.values():
        pairs = {}
        for idx, q in enumerate(m.get("questions", [])):
            m_idx = q.get("mqIndex") or q.get("mq_index") or (idx + 1)
            p_key = (m_idx - 1) // 2
            pairs.setdefault(p_key, []).append(q)
        for p_qs in pairs.values():
            total_paper_marks += max(q.get("totalMarks", 10) for q in p_qs)

    expected_total = 50 if raw.exam_type == ExamType.IA else 100
    if total_paper_marks < expected_total:
        health.deduct(25, f"Attemptable marks mismatch: {total_paper_marks}/{expected_total}")

    qa_score = int(
        sum(vq.score for vq in validated) / max(1, len(validated)) * 100
    )

    return FinalPaper(
        doc_id      = raw.doc_id,
        modules     = list(modules_dict.values()),
        exam_type   = raw.exam_type.value,
        subject     = raw.subject,
        total_marks = expected_total,
        qa_score    = qa_score,
        health      = health,
    )
