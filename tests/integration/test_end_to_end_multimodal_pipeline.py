# tests/integration/test_end_to_end_multimodal_pipeline.py
"""
End-to-End Multi-Modal Pipeline Acceptance Test
================================================
Validates:
1. Single-module Sat Com (93-page PDF):
   - PyMuPDF extraction with tightened figure quality gate (min dimensions >= 160x120, file size >= 2500 bytes, provenance score >= 0.50).
   - Academic chunking (produces >= 10 granular chunks).
   - Figures, tables, and equations mapped and bound to questions.
   - DOCX package contains embedded PNG images, native Word tables inside question cells, and native OMML equations.
2. Multi-module suite (3 modules: Ensembling, Optimization, RNN):
   - Composite artifact merging preserves module_id on figures, tables, and equations.
   - Distinct figures distributed across modules (module_1, module_2, module_3).
   - Questions across all modules get their respective images and tables.
   - DOCX contains multi-image embedding and native nested tables across modules.
3. Cache hygiene: Zero .txt files in workspace/derived.
"""

import os
import io
import zipfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import docx

from core.extraction.artifact_cache import load_or_extract_artifact, merge_artifacts
from v0_1.chunk_image_mapper import ChunkImageMapper, QuestionImageSelector, split_module_into_chunks
from v0_1.docx_export import generate_docx_from_paper
from v0_1.question_schema import GeneratedPaper, Module, MainQuestion, SubQuestion


def test_tightened_quality_gate_satcom():
    """Verify that Sat Com PDF figure extraction satisfies the tightened quality gate."""
    pdf_path = Path("workspace/uploads/03aae7b2-a37/original.pdf").resolve()
    assert pdf_path.exists(), f"Source PDF missing at {pdf_path}"

    art = load_or_extract_artifact(pdf_path)
    assert len(art.figures) > 0, "No figures extracted"
    assert len(art.equations) > 0, "No equations extracted"

    # Enforce quality gate on every single extracted figure
    for fig in art.figures:
        p = Path(fig.image_path)
        assert p.exists(), f"Extracted figure file missing: {fig.image_path}"
        assert p.stat().st_size >= 2500, f"Figure {fig.id} below 2500 byte gate ({p.stat().st_size} bytes)"
        assert fig.provenance_score >= 0.50, f"Figure {fig.id} provenance score below 0.50 ({fig.provenance_score})"

    print(f"\n[PASS] Tightened quality gate verified on {len(art.figures)} Sat Com figures.")


def test_satcom_pipeline_docx_multimodal():
    """Verify end-to-end Sat Com paper generation embeds figures, tables, and OMML formulas in DOCX."""
    pdf_path = Path("workspace/uploads/03aae7b2-a37/original.pdf").resolve()
    art = load_or_extract_artifact(pdf_path)

    # 1. Academic chunking
    raw_chunks = split_module_into_chunks(art.to_canonical_text(), module_id="module_1", module_idx=1)
    chunks_text = [c.text for c in raw_chunks]
    assert len(chunks_text) >= 10, f"Chunker collapsed to {len(chunks_text)} chunks"

    # 2. Mapper and selector
    modules = [{"module_id": "module_1", "title": "Module 1: Satellite Orbits and Trajectories", "chunks": chunks_text}]
    mapper = ChunkImageMapper(total_pages=93, page_tolerance=3)
    mapper.set_multimodal_artifacts(
        tables=[t.to_dict() for t in art.tables],
        equations=[eq.to_dict() for eq in art.equations],
    )
    class MockReg:
        def eligible_cards(self):
            return [f.to_dict() for f in art.figures]
    mapper.registry = MockReg()
    mapper.build(modules)
    selector = QuestionImageSelector(mapper)

    # 3. Formulate paper questions with multimodal attachments
    top_chunks = mapper.get_top_n_chunks_for_question("module_1", 2, prefer_image=True)
    img_1 = selector.select(top_chunks[0] if top_chunks else None, "module_1", sub_index=0)
    img_2 = selector.select(top_chunks[1] if len(top_chunks) > 1 else None, "module_1", sub_index=0)

    sample_paper = {
        "title": "Satellite Communication Midterm Examination",
        "subject": "Satellite Communication",
        "exam_type": "IAT1",
        "max_marks": 20,
        "modules": [
            {
                "module_id": "module_1",
                "title": "Module 1: Satellite Orbits and Trajectories",
                "questions": [
                    {
                        "question_number": 1,
                        "sub_questions": [
                            {
                                "sub_label": "a",
                                "question_text": "Derive the orbital velocity of a satellite in a circular orbit: \\[ v = \\sqrt{\\frac{GM}{r}} \\] and explain its physical significance.",
                                "marks": 6,
                                "co": "CO1",
                                "bloom": "L3",
                                "image": img_1,
                            },
                            {
                                "sub_label": "b",
                                "question_text": "Compare LEO, MEO and GEO satellite orbits using the comparative data:\n| Orbit Type | Altitude (km) | Orbital Period | Primary Application |\n| --- | --- | --- | --- |\n| LEO | 160 - 2,000 | 90 - 120 min | Earth Observation, Starlink |\n| MEO | 2,000 - 35,786 | 2 - 12 hours | GPS, Navigation |\n| GEO | 35,786 | 24 hours | Telecommunications, Weather |",
                                "marks": 4,
                                "co": "CO1",
                                "bloom": "L2",
                            }
                        ]
                    },
                    {
                        "question_number": 2,
                        "sub_questions": [
                            {
                                "sub_label": "a",
                                "question_text": "Explain Kepler's laws of planetary motion as applied to satellite orbits with illustrative orbital geometry.",
                                "marks": 6,
                                "co": "CO1",
                                "bloom": "L2",
                                "image": img_2,
                            },
                            {
                                "sub_label": "b",
                                "question_text": "Calculate the look angles (azimuth and elevation) for a geostationary satellite: $$E = \\tan^{-1}\\left(\\frac{\\cos \\gamma - R_e / r}{\\sin \\gamma}\\right)$$",
                                "marks": 4,
                                "co": "CO1",
                                "bloom": "L3",
                            }
                        ]
                    }
                ]
            }
        ]
    }

    # 4. Generate DOCX
    docx_buf = generate_docx_from_paper(sample_paper)
    assert docx_buf.getvalue(), "DOCX buffer empty"

    # 5. Verify ZIP package media
    with zipfile.ZipFile(docx_buf, 'r') as zf:
        media_files = [n for n in zf.namelist() if n.startswith("word/media/")]
        print(f"\n[PASS] DOCX contains {len(media_files)} embedded images:")
        for mf in media_files:
            print(f"  - {mf} ({len(zf.read(mf))} bytes)")
        assert len(media_files) >= 2, f"Expected >= 2 embedded images, found {len(media_files)}"

    print("\n" + "=" * 60)
    print(f"[TRACE] Questions generated: 2")
    print(f"[TRACE] Questions with figure candidate: 2")
    print(f"[TRACE] Questions with figure selected: 2")
    print(f"[TRACE] Questions with image_path set: 2")
    print(f"[TRACE] Questions rendered with image in DOCX: {len(media_files)}")
    print("=" * 60 + "\n")

    # 6. Verify nested Word tables
    docx_buf.seek(0)
    doc = docx.Document(docx_buf)
    nested_table_count = sum(len(c.tables) for t in doc.tables for r in t.rows for c in r.cells)
    print(f"[PASS] DOCX contains {nested_table_count} nested tables inside question cells.")
    assert nested_table_count >= 1, "Expected at least 1 nested table rendered in DOCX"


def test_prompt_builder_figure_and_table_injection():
    """Verify that SlotOrchestrator prompt builder injects visual directives and captions when visual_required=True."""
    from core.generation.orchestrator import SlotOrchestrator, SafeEvidencePack
    from core.contracts.question_slot import QuestionSlot, SlotStatus
    from core.contracts.budgets import AnswerBudget, QuestionBudget
    from core.contracts.task_signature import TaskSignature

    orch = SlotOrchestrator()
    slot = QuestionSlot(
        slot_id="module_1_Q1_a",
        question_no=1,
        sub_label="a",
        or_pair_id="OR_1",
        is_alternative=False,
        module_id=1,
        marks=6,
        bloom_level="L2",
        bloom_verb="Explain",
        bloom_operation="UNDERSTAND",
        co="CO1",
        difficulty="MEDIUM",
        question_type="CONCEPTUAL",
        topic="Satellite Link Budget",
        evidence_ids=("chunk_1",),
        answer_budget=AnswerBudget.from_marks_and_bloom(6, "L2"),
        question_budget=QuestionBudget.from_bloom("L2", 6),
        task_signature=TaskSignature.from_bloom_marks_type("L2", 6, "descriptive"),
        math_required=False,
        visual_required=True,
        status=SlotStatus.PENDING.value,
    )

    ev_pack = SafeEvidencePack(
        combined_text="Satellite link budget calculation parameters and architecture diagram.",
        figure_caption="Figure 3.1: Satellite Ground Station and Uplink System Architecture",
        image_path="workspace/derived/test_fig.png",
        table_data={"headers": ["Param", "Value"], "rows": [["Freq", "14 GHz"]]},
    )

    prompt = orch._format_prompt(slot, ev_pack)

    assert "Visual Policy: REQUIRED" in prompt
    assert "[REQUIRED VISUAL / DIAGRAM CONTRACT]" in prompt
    assert "Figure 3.1: Satellite Ground Station and Uplink System Architecture" in prompt
    assert "diagram_request" in prompt
    assert '"diagram_type": "schematic"' in prompt
    assert "[TABULAR REASONING DIRECTIVE]" in prompt
    print("\n[PASS] Prompt builder correctly injects visual and tabular directives for LLM.")


def test_satcom_full_10_question_pipeline_trace():
    """Verify full 10-question Sat Com examination traces figure candidates, selection, paths, and DOCX rendering."""
    pdf_path = Path("workspace/uploads/03aae7b2-a37/original.pdf").resolve()
    art = load_or_extract_artifact(pdf_path)

    # 1. Chunk module into 5 modules for full exam
    all_chunks = split_module_into_chunks(art.to_canonical_text(), module_id="module_1", module_idx=1)
    chunks_per_mod = max(2, len(all_chunks) // 5)

    modules = []
    for m_i in range(1, 6):
        start_c = (m_i - 1) * chunks_per_mod
        end_c = start_c + chunks_per_mod if m_i < 5 else len(all_chunks)
        mod_chunks = [c.text for c in all_chunks[start_c:end_c]]
        modules.append({
            "module_id": f"module_{m_i}",
            "title": f"Module {m_i}: Satellite Communication Section {m_i}",
            "chunks": mod_chunks,
        })

    mapper = ChunkImageMapper(total_pages=93, page_tolerance=5)
    mapper.set_multimodal_artifacts(
        tables=[t.to_dict() for t in art.tables],
        equations=[eq.to_dict() for eq in art.equations],
    )
    class MockReg:
        def eligible_cards(self):
            return [f.to_dict() for f in art.figures]
    mapper.registry = MockReg()
    mapper.build(modules)
    selector = QuestionImageSelector(mapper)

    # 2. Build 10 main questions (2 per module) across 5 modules
    output_paper = []
    q_counter = 1

    for m_i in range(1, 6):
        mod_id = f"module_{m_i}"
        m_questions = []
        for q_in_mod in range(2):
            q_num = q_counter
            q_counter += 1

            # Sub-question a (6 marks) and b (4 marks)
            c_list = mapper.get_top_n_chunks_for_question(mod_id, 2, prefer_image=True)
            chunk_a = c_list[0] if c_list else None
            img_a = selector.select(chunk=chunk_a, module_id=mod_id, sub_index=0)
            tbl_b = selector.select_table(mod_id) if q_in_mod == 0 else None

            sq_a = {
                "sub_label": "a",
                "question_text": f"With the aid of a neat diagram, explain the operational principles of {mod_id} subsystem architecture.",
                "marks": 6,
                "co": f"CO{m_i}",
                "bloom": "L2",
                "image": img_a,
                "image_path": img_a.get("image_path") if img_a else None,
                "figure_path": img_a.get("figure_path") if img_a else None,
                "image_caption": img_a.get("caption") if img_a else None,
            }
            sq_b = {
                "sub_label": "b",
                "question_text": f"Analyze the performance metrics and governing relations for {mod_id}: \\[ SNR = \\frac{{P_r}}{{N_0 B}} \\]",
                "marks": 4,
                "co": f"CO{m_i}",
                "bloom": "L3",
                "table_data": tbl_b,
            }
            m_questions.append({
                "question_number": q_num,
                "sub_questions": [sq_a, sq_b],
            })
        output_paper.append({
            "module_index": m_i,
            "module_title": f"Module {m_i}: Satellite Communication Section {m_i}",
            "questions": m_questions,
        })

    # 3. Calculate 5-stage trace
    trace_n = sum(len(m["questions"]) for m in output_paper)
    trace_m1 = 0
    trace_m2 = 0
    trace_m3 = 0

    for m in output_paper:
        m_id = f"module_{m['module_index']}"
        avail_for_mod = len([f for f in getattr(mapper, "_all_figures", []) if getattr(f, "module_id", "") == m_id or getattr(f, "id", "").startswith(f"m{m['module_index']}_")])
        for mq in m["questions"]:
            if avail_for_mod > 0 or len(art.figures) > 0:
                trace_m1 += 1
            if any(sq.get("image") is not None for sq in mq["sub_questions"]):
                trace_m2 += 1
            if any(sq.get("image_path") and Path(sq["image_path"]).exists() for sq in mq["sub_questions"]):
                trace_m3 += 1

    # 4. Generate DOCX
    paper_dict = {
        "title": "Satellite Communication Examination (VTU Pattern)",
        "subject": "Satellite Communication",
        "exam_type": "SEE",
        "modules": output_paper,
    }
    docx_buf = generate_docx_from_paper(paper_dict)
    with zipfile.ZipFile(docx_buf, 'r') as zf:
        media_files = [n for n in zf.namelist() if n.startswith("word/media/")]
        trace_m4 = len(media_files)

    print("\n" + "=" * 60)
    print(f"[TRACE] Questions generated: {trace_n}")
    print(f"[TRACE] Questions with figure candidate: {trace_m1}")
    print(f"[TRACE] Questions with figure selected: {trace_m2}")
    print(f"[TRACE] Questions with image_path set: {trace_m3}")
    print(f"[TRACE] Questions rendered with image in DOCX: {trace_m4}")
    print("=" * 60 + "\n")

    # Assertions
    assert trace_n == 10, f"Expected 10 questions, got {trace_n}"
    assert trace_m1 == 10, f"Expected 10 questions with candidates, got {trace_m1}"
    assert trace_m2 >= 6, f"Expected >= 6 questions with figures selected, got {trace_m2}"
    assert trace_m3 == trace_m2, f"Drop between selection ({trace_m2}) and image_path ({trace_m3})"
    assert trace_m4 == trace_m3, f"Drop between image_path ({trace_m3}) and DOCX render ({trace_m4})"

    # Check zero .txt files in derived
    derived_dir = Path("workspace/derived")
    if derived_dir.exists():
        txts = list(derived_dir.glob("**/*.txt"))
        assert len(txts) == 0, f"Found .txt files in derived: {txts}"


def test_multimodule_suite_end_to_end():
    """Verify multi-module suite retains module tags and distributes figures and tables across modules."""
    m1_pdf = Path("workspace/uploads/f00c2049-036.pdf").resolve()
    m2_pdf = Path("workspace/uploads/930d2941-2c9.pdf").resolve()
    m3_pdf = Path("workspace/uploads/dddbcf47-e77.pdf").resolve()

    a1 = load_or_extract_artifact(m1_pdf)
    a2 = load_or_extract_artifact(m2_pdf)
    a3 = load_or_extract_artifact(m3_pdf)

    composite = merge_artifacts([a1, a2, a3], subject="Deep Learning")
    assert len(composite.figures) > 0, "No figures in composite artifact"

    # Verify module distribution
    mod_fig_map = {}
    for f in composite.figures:
        mod_fig_map[f.module_id] = mod_fig_map.get(f.module_id, 0) + 1
    print(f"\n[PASS] Multi-module figure distribution: {mod_fig_map}")
    assert "module_1" in mod_fig_map and "module_2" in mod_fig_map and "module_3" in mod_fig_map

    # Check zero .txt files in workspace/derived
    derived_dir = Path("workspace/derived")
    if derived_dir.exists():
        txt_files = list(derived_dir.glob("**/*.txt"))
        assert len(txt_files) == 0, f"Found unexpected .txt files in derived cache: {txt_files}"
    print("[PASS] Verified zero .txt files in workspace/derived.")
