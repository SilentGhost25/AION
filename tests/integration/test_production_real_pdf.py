# tests/integration/test_production_real_pdf.py

import os
import uuid
import pytest
from reportlab.pdfgen import canvas
from core.extraction.gateway import ExtractionGateway
from v0_1.main import run_pipeline
from core.validation.export_gate import ExportGate


@pytest.fixture
def real_syllabus_pdf() -> str:
    """Generates a clean, simple syllabus PDF file with sufficient chunks to pass extraction gates."""
    pdf_path = f"real_syllabus_test_{uuid.uuid4().hex[:8]}.pdf"
    c = canvas.Canvas(pdf_path)
    
    # We generate 75 distinct blocks to satisfy the 50-chunk floor.
    y = 750
    for i in range(75):
        if y < 80:
            c.showPage()
            y = 750
        
        if i % 2 == 0:
            text = f"Syllabus details for unit {i}: Transmission Control Protocol is connection-oriented."
        else:
            text = f"Syllabus details for unit {i}: Ohm's Law states that voltage is current times resistance."
        
        c.drawString(100, y, text)
        y -= 45
        
    c.save()
    yield pdf_path
    if os.path.exists(pdf_path):
        os.remove(pdf_path)


def test_production_real_pdf_generation(real_syllabus_pdf):
    """
    Production Reality Test.
    Runs the entire pipeline using:
    - Real PDF source file
    - Real ExtractionGateway (PyMuPDF backend)
    - Real segmentation
    - Real local LLM (manually overridden to qwen2.5:7b for instruction-following quality)
    - Real ExportGate validation
    """
    # Force use of qwen2.5:1.5b for fast local E2E testing
    os.environ["AION_MODEL"] = "qwen2.5:1.5b"

    # 1. Verify Extraction Gateway handles the PDF successfully
    artifact = ExtractionGateway.extract(real_syllabus_pdf)
    assert artifact is not None
    assert artifact.page_count > 1
    assert len(artifact.chunks) >= 50
    
    # 2. Run the E2E generation pipeline
    paper, qa_report = run_pipeline(
        file_path=real_syllabus_pdf,
        exam_type="ia",
        difficulty="easy",
        include_visual=False,
        max_concepts=10,
        sub_question_count=2
    )
    
    # 3. Assertions on the generated paper structure and Export Gate verdict
    assert len(paper) >= 2  # modules extracted and processed
    
    # Check authoritative ExportGate status
    assert qa_report.get("export_gate_passed") is True
    
    # Verify VTU IA structure: exactly 4 main questions per module (Q1/Q2 alternative pair, Q3/Q4 alternative pair)
    for module in paper:
        assert "module_index" in module
        questions = module["questions"]
        assert len(questions) == 4
        
        # Check subquestion structure and marks
        for mq in questions:
            assert "sub_questions" in mq
            sub_qs = mq["sub_questions"]
            assert len(sub_qs) == 2  # sub_question_count locked to 2
            
            # Check VTU marks compliance (6 + 4 = 10 marks per main question)
            total_marks = sum(sub["marks"] for sub in sub_qs)
            assert total_marks == 10
            
            # Verify no answer leakage, no unicode corruption, and valid start verbs
            for sub in sub_qs:
                text = sub["text"]
                assert "\ufffd" not in text
                assert "answer:" not in text.lower()
                assert "solution:" not in text.lower()
                
                # Check for pedagogy question type mapping
                assert "question_type" in sub
                assert sub["question_type"] in ["CONCEPTUAL", "NUMERICAL", "APPLICATION"]

                # Check for starting verb compliance
                first_word = text.strip().split()[0].lower().rstrip(".,;:")
                # Verb should be alphabetical
                assert first_word.isalpha()
