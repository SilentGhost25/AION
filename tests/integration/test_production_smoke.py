# tests/integration/test_production_smoke.py

import json
import pytest
from unittest.mock import patch
from aion_api import app
from v0_1.main import run_pipeline
from core.extraction.gateway import DocumentArtifact
from core.extraction.contracts import EvidenceChunk, ChunkStatus, ContentType, ExtractionAdapterID

from concurrent.futures import Future

class SyncExecutor:
    def __init__(self, max_workers=None):
        pass
    def submit(self, fn, *args, **kwargs):
        f = Future()
        try:
            res = fn(*args, **kwargs)
            f.set_result(res)
        except Exception as e:
            f.set_exception(e)
        return f
    def shutdown(self, wait=True):
        pass


# Flag to verify orchestrator is indeed invoked
ORCHESTRATOR_CALLED = False


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def mock_robust_llm_call(self, prompt: str, max_tokens: int = 512, stream_fn=None) -> str:
    global ORCHESTRATOR_CALLED
    if "Generate ONE examination sub-question" in prompt or "SLOT SPECIFICATION" in prompt:
        ORCHESTRATOR_CALLED = True

    # Parse expected Bloom verb, Operation & math requirements from prompt
    math_required = "Math policy    : REQUIRED" in prompt
    bloom_verb = "Explain"
    for line in prompt.splitlines():
        if "Bloom Verb:" in line:
            bloom_verb = line.split(":")[1].split("(")[0].strip()
            if "primary operation:" in line:
                bloom_op = line.split("primary operation:")[1].split(")")[0].strip().upper()
        elif "Allowed Operations:" in line:
            bloom_op = line.split(":")[1].split("and")[0].strip().upper()

    # Standard Qwen V5 Output JSON mock
    if math_required:
        return json.dumps({
            "instruction": f"{bloom_verb} [MATH:eq1] in detail, solve formulas, and calculate valid questions.",
            "question_text": f"{bloom_verb} [MATH:eq1] in detail for syllabus content and valid questions.",
            "math_blocks": [
                {
                    "block_id": "eq1",
                    "latex": "R = \\frac{V}{I}",
                    "display_mode": False
                }
            ]
        })
    elif bloom_op == "REMEMBER":
        return json.dumps({
            "instruction": f"{bloom_verb} syllabus content for Module 1 questions, define principles, and state valid questions.",
            "question_text": f"{bloom_verb} syllabus content for Module 1 questions, define principles, and state valid questions.",
            "math_blocks": []
        })
    elif bloom_op in ("APPLY", "CALCULATE"):
        return json.dumps({
            "instruction": f"{bloom_verb} syllabus content for Module 1 questions, solve formulas, and calculate valid questions.",
            "question_text": f"{bloom_verb} syllabus content for Module 1 questions, solve formulas, and calculate valid questions.",
            "math_blocks": []
        })
    elif bloom_op in ("ANALYSE", "ANALYZE", "COMPARE"):
        return json.dumps({
            "instruction": f"{bloom_verb} syllabus content for Module 1 questions, compare features, and differentiate valid questions.",
            "question_text": f"{bloom_verb} syllabus content for Module 1 questions, compare features, and differentiate valid questions.",
            "math_blocks": []
        })
    elif bloom_op in ("EVALUATE", "JUSTIFY"):
        return json.dumps({
            "instruction": f"{bloom_verb} syllabus content for Module 1 questions, assess principles, and justify valid questions.",
            "question_text": f"{bloom_verb} syllabus content for Module 1 questions, assess principles, and justify valid questions.",
            "math_blocks": []
        })
    elif bloom_op == "CREATE":
        return json.dumps({
            "instruction": f"{bloom_verb} syllabus content for Module 1 questions, assess framework, and design valid questions.",
            "question_text": f"{bloom_verb} syllabus content for Module 1 questions, assess framework, and design valid questions.",
            "math_blocks": []
        })
    else:
        # Default L2 / Understand
        return json.dumps({
            "instruction": f"{bloom_verb} syllabus content for Module 1 questions, describe principles, and outline valid questions.",
            "question_text": f"{bloom_verb} syllabus content for Module 1 questions, describe principles, and outline valid questions.",
            "math_blocks": []
        })


def mock_extract(source_path: str, document_id: str = "doc_001", store=None) -> DocumentArtifact:
    # Build mock evidence chunks for Module 1 and Module 2
    chunks = [
        EvidenceChunk(
            chunk_id="chk_mod1_1",
            document_id=document_id,
            source_path=source_path,
            adapter_id=ExtractionAdapterID.PYMUPDF,
            page_start=1,
            page_end=1,
            content_type=ContentType.TEXT,
            text="TCP is connection-oriented, UDP is connectionless. TCP guarantees packet delivery.",
            module_id="1"
        ),
        EvidenceChunk(
            chunk_id="chk_mod2_1",
            document_id=document_id,
            source_path=source_path,
            adapter_id=ExtractionAdapterID.PYMUPDF,
            page_start=2,
            page_end=2,
            content_type=ContentType.TEXT,
            text="Ohm's Law states that V = I * R where V is voltage, I is current, and R is resistance.",
            module_id="2"
        )
    ]
    
    # Force status to VALID
    for c in chunks:
        c.status = ChunkStatus.VALID
        
    return DocumentArtifact(
        document_id=document_id,
        source_path=source_path,
        mime_type="application/pdf",
        page_count=2,
        chunks=chunks,
        backends=["PyMuPDF"]
    )


from v0_1.segmenter import SegmentResult, ModuleSegment

def mock_segment_document(text: str, file_path: str = "") -> SegmentResult:
    return SegmentResult(segments=[
        ModuleSegment(
            title="Module 1",
            content="This is syllabus content for Module 1. It contains principles, formulas, assess, justify, compare, and solve options for valid questions.",
            word_count=30
        ),
        ModuleSegment(
            title="Module 2",
            content="This is syllabus content for Module 2. It contains principles, formulas, assess, justify, compare, and solve options for valid questions.",
            word_count=30
        )
    ])


@patch("v0_1.llm.RobustLLMCaller.call", mock_robust_llm_call)
@patch("core.extraction.gateway.ExtractionGateway.extract", mock_extract)
@patch("v0_1.main.upload", lambda x: x)
@patch("v0_1.main.segment_document", mock_segment_document)
def test_production_smoke_pipeline_and_api(client):
    """
    E2E Production Smoke Test.
    Verifies that the entire pipeline:
      upload -> segment -> evidence -> slot-orchestrator -> linter -> OR pair -> ExportGate -> API Response
    runs correctly, preserving metadata truth and passing all gates.
    """
    global ORCHESTRATOR_CALLED
    ORCHESTRATOR_CALLED = False

    payload = {
        "subject": "Computer Networks",
        "department": "AIML",
        "semester": 5,
        "exam_type": "IAT-1",
        "selected_modules": [1, 2],
        "bloom_levels": ["L2", "L3"],
        "difficulty": "MIXED",
        "model": "qwen2.5:14b",
        "notes_text": "Computer networks concepts: TCP and UDP protocols, Ohm's law formulas.",
    }

    # 1. Call Flask endpoint /api/generate/stream
    # We must patch mock_extract, RobustLLMCaller, segment_document and ThreadPoolExecutor inside Flask's request handler context too
    with patch("core.extraction.gateway.ExtractionGateway.extract", mock_extract), \
         patch("v0_1.main.upload", lambda x: x), \
         patch("v0_1.llm.RobustLLMCaller.call", mock_robust_llm_call), \
         patch("v0_1.main.ThreadPoolExecutor", SyncExecutor), \
         patch("v0_1.main.segment_document", mock_segment_document):
        response = client.post(
            "/api/generate/stream",
            json=payload
        )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["Content-Type"]
    stream_data = response.get_data()
    assert len(stream_data) > 0

    # 2. Directly run main.py run_pipeline E2E with a dummy PDF path
    import uuid
    dummy_pdf_path = f"dummy_syllabus_file_{uuid.uuid4().hex[:8]}.pdf"
    
    import os
    with open(dummy_pdf_path, "w") as f:
        f.write("%PDF-1.4 dummy header")
        
    try:
        with patch("core.extraction.gateway.ExtractionGateway.extract", mock_extract), \
             patch("v0_1.main.upload", lambda x: x), \
             patch("v0_1.llm.RobustLLMCaller.call", mock_robust_llm_call), \
             patch("v0_1.main.ThreadPoolExecutor", SyncExecutor), \
             patch("v0_1.main.segment_document", mock_segment_document):
            paper, qa_report = run_pipeline(
                file_path=dummy_pdf_path,
                exam_type="ia",
                difficulty="mixed",
                include_visual=False,
                max_concepts=10,
                sub_question_count=2,
            )
    finally:
        if os.path.exists(dummy_pdf_path):
            os.remove(dummy_pdf_path)

    # 3. Assertions on metadata preservation and linter gates
    assert ORCHESTRATOR_CALLED is True
    assert len(paper) == 2  # 2 modules selected

    for module in paper:
        assert "questions" in module
        questions = module["questions"]
        assert len(questions) in (2, 4)  # 2 questions for IA, 4 for SEE (VTU standard)

        for mq in questions:
            assert "sub_questions" in mq
            sub_qs = mq["sub_questions"]
            
            # Verify each subquestion preserves metadata and satisfies rules
            for idx, sq in enumerate(sub_qs):
                text = sq["text"]
                assert text.strip()
                
                # Check for unicode corruption
                assert "\ufffd" not in text
                assert "\x00" not in text
                
                # Check for answer leakages or meta-language
                assert "answer:" not in text.lower()
                assert "solution:" not in text.lower()
                assert "provided notes" not in text.lower()

                # Check for multi-slot labels contamination (a), (b), Q1, etc.
                assert not text.startswith("(a)")
                assert not text.startswith("a)")
                assert not text.startswith("Q")

    # Verify that the final qa_report was successfully generated and passed
    assert qa_report.get("status") == "PASS"
    assert qa_report.get("export_gate_passed") is True
