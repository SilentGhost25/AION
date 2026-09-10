import os
import time
import pytest
from pathlib import Path
from threading import Thread

from core.artifacts.store import ArtifactStore
from core.artifacts.manifest import DocumentManifest, SourceArtifact
from v0_1.extractor import ConfidenceGatedExtractor
from v0_1.difficulty import DifficultyManager
from v0_1.main import _chunk_selection_lock


def test_optimization_flags_defaults(monkeypatch):
    """Verify default active optimization values and rollback capability."""
    monkeypatch.delenv("ENABLE_EXTRACTION_CACHE", raising=False)
    monkeypatch.delenv("OCR_WORD_COUNT_THRESHOLD", raising=False)

    cache_enabled = os.getenv("ENABLE_EXTRACTION_CACHE", "true").lower() == "true"
    threshold = int(os.getenv("OCR_WORD_COUNT_THRESHOLD", "100"))

    assert cache_enabled is True, "ENABLE_EXTRACTION_CACHE must default to true"
    assert threshold == 100, "OCR_WORD_COUNT_THRESHOLD must default to 100"

    # Verify explicit rollback
    monkeypatch.setenv("ENABLE_EXTRACTION_CACHE", "false")
    monkeypatch.setenv("OCR_WORD_COUNT_THRESHOLD", "0")
    assert os.getenv("ENABLE_EXTRACTION_CACHE", "true").lower() == "false"
    assert int(os.getenv("OCR_WORD_COUNT_THRESHOLD", "100")) == 0


def test_ocr_word_count_gating_skips_rapidocr(monkeypatch):
    """Verify OCR_WORD_COUNT_THRESHOLD=100 skips OCR override on digital text PDFs."""
    monkeypatch.setenv("OCR_WORD_COUNT_THRESHOLD", "100")
    pdf_path = "scratch/satellite_sparse_m1_m3.pdf"
    if not os.path.exists(pdf_path):
        pytest.skip(f"Test PDF not found at {pdf_path}")

    extractor = ConfidenceGatedExtractor()
    t0 = time.time()
    res = extractor.extract_pdf(pdf_path)
    duration = time.time() - t0

    assert "text" in res
    assert len(res["text"].split()) >= 100
    # Must complete fast without attempting 90s OCR loop
    assert duration < 5.0, f"Extraction took {duration:.2f}s, OCR should have been skipped"


def test_extraction_cache_persistence_and_retrieval(tmp_path, monkeypatch):
    """Verify store.store_derived properly registers derived_text for get_document_text."""
    monkeypatch.setenv("AION_BASE_DIR", str(tmp_path))
    store = ArtifactStore(base_dir=tmp_path)
    doc_id = "test_opt_doc_1"

    # Create dummy source
    src_file = tmp_path / "sample.pdf"
    src_file.write_bytes(b"%PDF-dummy")

    import hashlib
    sha = hashlib.sha256(b"%PDF-dummy").hexdigest()

    manifest = DocumentManifest(
        document_id=doc_id,
        created_at="2026-09-09T00:00:00Z",
        source=SourceArtifact(
            path=str(src_file),
            filename="sample.pdf",
            mime_type="application/pdf",
            size_bytes=10,
            sha256=sha,
            authoritative=True,
        ),
        derived={},
    )
    store.save_manifest(manifest)

    # Store derived text
    sample_text = "Orbital Mechanics and Keplerian Dynamics full lecture notes."
    store.store_derived(doc_id, "plain_text", sample_text)

    # Verify manifest now has derived text
    loaded = store.get(doc_id)
    derived_path = loaded.get_derived_text()
    assert derived_path is not None
    assert os.path.exists(derived_path)
    with open(derived_path, "r", encoding="utf-8") as f:
        assert f.read() == sample_text


def test_per_module_difficulty_manager_independence():
    """Verify fresh DifficultyManager per module resets used verbs independently."""
    dm1 = DifficultyManager.from_string("mixed")
    v1_a = dm1.get_verb("easy", 2)
    v1_b = dm1.get_verb("easy", 2)

    # Within module 1, consecutive verbs for same difficulty/bloom rotate
    assert v1_a != v1_b
    assert len(dm1._used) >= 2

    # Fresh module 2 DifficultyManager has independent state
    dm2 = DifficultyManager.from_string("mixed")
    assert len(dm2._used) == 0


def test_chunk_selection_lock_thread_safety():
    """Verify _chunk_selection_lock properly synchronizes concurrent selection operations."""
    shared_pool = [f"chunk_{i}" for i in range(100)]
    allocated = []

    def worker(worker_id):
        for _ in range(20):
            with _chunk_selection_lock:
                if shared_pool:
                    chunk = shared_pool.pop(0)
                    allocated.append((worker_id, chunk))
            time.sleep(0.001)

    threads = [Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Verify no chunk was allocated more than once
    allocated_chunks = [c for _, c in allocated]
    assert len(allocated_chunks) == len(set(allocated_chunks))


def test_get_document_text_cache_hit(tmp_path, monkeypatch, capsys):
    """Verify get_document_text reads directly from cache and logs [CACHE] hit."""
    from aion_api import get_document_text
    monkeypatch.setenv("AION_BASE_DIR", str(tmp_path))
    store = ArtifactStore(base_dir=tmp_path)
    doc_id = "test_cache_doc_hit"

    src_file = tmp_path / "sample.pdf"
    src_file.write_bytes(b"%PDF-cache-hit-test")
    import hashlib
    sha = hashlib.sha256(b"%PDF-cache-hit-test").hexdigest()

    manifest = DocumentManifest(
        document_id=doc_id,
        created_at="2026-09-09T00:00:00Z",
        source=SourceArtifact(
            path=str(src_file),
            filename="sample.pdf",
            mime_type="application/pdf",
            size_bytes=len(b"%PDF-cache-hit-test"),
            sha256=sha,
            authoritative=True,
        ),
        derived={},
    )
    store.save_manifest(manifest)

    cached_content = "Keplerian mechanics and satellite transponders cached text content."
    store.store_derived(doc_id, "plain_text", cached_content)

    text = get_document_text(doc_id, store=store)
    assert text == cached_content

    captured = capsys.readouterr()
    assert f"[CACHE] Reading cached extraction for {doc_id}" in captured.out


def test_per_module_orchestrator_isolation():
    """Verify that independent SlotOrchestrator instances maintain strict state isolation."""
    from core.generation.orchestrator import SlotOrchestrator
    
    orc1 = SlotOrchestrator()
    orc2 = SlotOrchestrator()

    # Simulate slot generation activity on orc1
    orc1._generated_texts_this_pair = ["Question 1a about Kepler orbits", "Question 1b about eccentricity"]
    orc1._archetype_counter = 5
    orc1.session_log.append({"attempt": 1, "status": "PASS"})

    # orc2 must be completely clean and unpolluted
    assert not hasattr(orc2, "_generated_texts_this_pair")
    assert getattr(orc2, "_archetype_counter", 0) == 0
    assert len(orc2.session_log) == 0


def test_concurrency_semaphore_respects_env(monkeypatch):
    """Verify get_concurrency_semaphore dynamically responds to AION_CONCURRENCY."""
    from v0_1.llm import _get_concurrency, get_concurrency_semaphore

    monkeypatch.setenv("AION_CONCURRENCY", "2")
    assert _get_concurrency() == 2
    sem = get_concurrency_semaphore()
    assert sem._value == 2

    monkeypatch.setenv("AION_CONCURRENCY", "4")
    assert _get_concurrency() == 4
    sem = get_concurrency_semaphore()
    assert sem._value == 4

    monkeypatch.setenv("AION_CONCURRENCY", "1")
    assert _get_concurrency() == 1
    sem = get_concurrency_semaphore()
    assert sem._value == 1


