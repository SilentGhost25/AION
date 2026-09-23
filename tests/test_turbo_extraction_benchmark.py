"""
Tests for Phase 3: Turbo-Extraction Engine & MinerU Pre-Flight Triage.
Verifies:
  - SHA-256 file hashing
  - 3-Tier Pre-Flight Page Triage (Tier A text, Tier B math/drawings, Tier C scanned)
  - Persistent SHA-256 cache hits (<50ms / 0.00s effective)
"""

import os
import json
import time
import shutil
import tempfile
from pathlib import Path
import pytest

from v0_1.document_parser import (
    get_file_sha256,
    preflight_page_triage,
    parse_document,
    ParsedDocument,
)
from v0_1.table_validator import ValidatedTable


def test_get_file_sha256():
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as f:
        f.write("AION Universal Exam Generator Turbo Extraction Test")
        f_path = f.name

    try:
        digest1 = get_file_sha256(f_path)
        digest2 = get_file_sha256(f_path)
        assert len(digest1) == 64
        assert digest1 == digest2
    finally:
        os.remove(f_path)


def test_preflight_page_triage_synthetic():
    fitz = pytest.importorskip("fitz")

    # Create a synthetic 3-page PDF:
    # Page 1: Pure digital text (Tier A)
    # Page 2: Text with math formula tokens (Tier B)
    # Page 3: Scanned bitmap with <25 words (Tier C)
    doc = fitz.open()

    # Page 1: Pure text
    p1 = doc.new_page()
    p1.insert_text((50, 50), "This is a digital chapter on software design patterns and modular architectures with sufficient words.")

    # Page 2: Complex math page with LaTeX token
    p2 = doc.new_page()
    p2.insert_text((50, 50), "Calculate the field using formula \\int_{0}^{\\infty} f(x) dx + \\frac{a}{b} = \\sigma")

    # Page 3: Scanned/empty page (no text)
    p3 = doc.new_page()
    # Insert a small pixmap image so images > 0 and word_count < 25
    pix = fitz.Pixmap(fitz.csRGB, (0, 0, 10, 10), False)
    p3.insert_image(fitz.Rect(50, 50, 60, 60), pixmap=pix)

    tmp_pdf = tempfile.NamedTemporaryFile("wb", delete=False, suffix=".pdf")
    pdf_path = tmp_pdf.name
    tmp_pdf.close()
    doc.save(pdf_path)
    doc.close()

    try:
        triage = preflight_page_triage(pdf_path)
        assert triage["total_pages"] == 3
        assert 1 in triage["tier_a_pure_text"]
        assert 2 in triage["tier_b_complex"]
        assert 3 in triage["tier_c_scanned"]
    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)


def test_sha256_cache_hit():
    fitz = pytest.importorskip("fitz")

    doc = fitz.open()
    p = doc.new_page()
    syllabus_text = (
        "Database Management Systems and Relational Algebra Course Syllabus. "
        "Module 1 introduces entity-relationship modeling, relational data structures, and functional dependencies. "
        "Students will analyze relational query optimization, B-plus tree indexing algorithms, and transaction ACID properties. "
        "Module 2 covers query processing, concurrency control protocols including two-phase locking, and write-ahead logging. "
        "Practical laboratory exercises will cover SQL schema definitions, foreign key constraints, and relational calculus expressions."
    )
    p.insert_textbox(fitz.Rect(50, 50, 500, 500), syllabus_text)

    tmp_pdf = tempfile.NamedTemporaryFile("wb", delete=False, suffix=".pdf")
    pdf_path = tmp_pdf.name
    tmp_pdf.close()
    doc.save(pdf_path)
    doc.close()

    file_hash = get_file_sha256(pdf_path)
    cache_dir = Path(".aion_cache/extraction") / file_hash
    cache_file = cache_dir / "cached_doc.json"

    # Clean any preexisting cache for this test file
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    try:
        # Pass 1: Fresh parse -> creates cache
        t0 = time.perf_counter()
        doc1 = parse_document(pdf_path, use_docling=False, use_ocr=False)
        t_fresh = time.perf_counter() - t0

        assert doc1.word_count > 0
        assert cache_file.exists()

        # Pass 2: Cached parse -> instant return in < 0.05s
        t1 = time.perf_counter()
        doc2 = parse_document(pdf_path, use_docling=False, use_ocr=False)
        t_cached = time.perf_counter() - t1

        assert doc2.method == "sha256_cached"
        assert doc2.word_count == doc1.word_count
        assert t_cached < 0.05
    finally:
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        if os.path.exists(pdf_path):
            os.remove(pdf_path)


def test_dual_mode_dispatcher_and_cloud_api():
    from unittest.mock import patch, MagicMock
    from v0_1.document_parser import parse_with_mineru_cloud_api

    # 1. Without API key, returns None immediately
    if "MINERU_API_KEY" in os.environ:
        del os.environ["MINERU_API_KEY"]
    res = parse_with_mineru_cloud_api("dummy.pdf")
    assert res is None

    # 2. With mock API response, returns ParsedDocument with method="mineru_cloud_api"
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".pdf") as tmp:
        tmp.write("dummy")
        p_path = tmp.name

    try:
        mock_post = MagicMock()
        mock_post.status_code = 200
        mock_post.json.return_value = {"code": 0, "data": {"task_id": "mock_task_123"}}

        mock_get = MagicMock()
        mock_get.status_code = 200
        mock_get.json.return_value = {
            "code": 0,
            "data": {
                "state": "done",
                "markdown": "# Module 1: Thermodynamics\nEntropy formula is $dS = \\frac{dQ}{T}$.",
                "pages_total": 2,
            }
        }

        with patch("requests.post", return_value=mock_post), patch("requests.get", return_value=mock_get):
            doc = parse_with_mineru_cloud_api(p_path, api_key="mock_key_abc")
            assert doc is not None
            assert doc.method == "mineru_cloud_api"
            assert doc.word_count > 0
            assert "Entropy" in doc.text
    finally:
        if os.path.exists(p_path):
            os.remove(p_path)

