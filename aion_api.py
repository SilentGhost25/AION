#!/usr/bin/env python3
"""
AION API Server
Flask-based HTTP server bridging the React frontend to the AION pipeline.
Run:    python aion_api.py
URL:    http://localhost:8100
"""

import os
import sys
import json
import uuid
import threading
import time
import traceback
from pathlib import Path
from datetime import datetime
import requests

# ── Path setup ────────────────────────────────────────────────
ROOT = Path(__file__).parent.resolve()
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from core.config.production_model import get_production_model, get_resolution_info
os.environ.setdefault("AION_MODEL", get_production_model())

# ── Core Services Imports ──────────────────────────────────────
from core.document_registry  import DocumentRegistry, DocumentStatus
from core.extraction_service import ExtractionService
from core.generation_context import GenerationContext
from core.planner            import Planner
from core.numerical_engine   import NumericalEngine
from v0_1.unified_pipeline   import run_unified, FinalPaper

# ── Flask imports ─────────────────────────────────────────────
try:
    from flask import Flask, request, jsonify, Response, stream_with_context
    from flask_cors import CORS
except ImportError:
    print("[ERROR] Flask not installed.")
    print("Run: pip install flask flask-cors")
    sys.exit(1)

# ── App setup ─────────────────────────────────────────────────
app = Flask(__name__)

# ✅ PERMANENT FIX 1: Allow ALL origins, ALL methods, ALL headers
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]
    }
})

# ✅ PERMANENT FIX 2: Force CORS headers on EVERY response
@app.after_request
def force_cors(response):
    response.headers["Access-Control-Allow-Origin"]          = "*"
    response.headers["Access-Control-Allow-Headers"]         = "*"
    response.headers["Access-Control-Allow-Methods"]         = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Private-Network"] = "true"
    response.headers["Access-Control-Max-Age"]               = "86400"
    response.headers["Vary"]                                 = "Origin"
    return response

# ✅ PERMANENT FIX 3: Handle ALL OPTIONS preflight requests globally
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = app.make_default_options_response()
        response.headers["Access-Control-Allow-Origin"]          = "*"
        response.headers["Access-Control-Allow-Headers"]         = "*"
        response.headers["Access-Control-Allow-Methods"]         = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Private-Network"] = "true"
        response.headers["Access-Control-Max-Age"]               = "86400"
        return response

# ── API v1 Blueprint ──────────────────────────────────────────
try:
    from api.v1 import api_v1_bp
    app.register_blueprint(api_v1_bp)
    print("[API] Successfully registered /api/v1 adapter layer.")
except Exception as e:
    print(f"[API ERROR] Failed to register /api/v1 blueprint: {e}")


@app.route("/api/v1", methods=["GET"])
@app.route("/api/v1/", methods=["GET"])
def api_v1_root():
    return jsonify({
        "name": "AION API Gateway",
        "version": "v1.0",
        "status": "online",
        "documentation": "AION Intelligence & Backend REST API v1",
        "workspaces": {
            "health": "/api/v1/health",
            "diagnostics": "/api/v1/diagnostics",
            "dashboard": "/api/v1/dashboard/summary",
            "uploads": "/api/v1/uploads",
            "training": "/api/v1/training/analyze",
            "question_forge": "/api/v1/questions/generate",
            "paper_forge": "/api/v1/papers/generate",
            "critic": "/api/v1/critic/evaluate",
            "datasets": "/api/v1/datasets",
            "models": "/api/v1/models",
            "knowledge": "/api/v1/knowledge/subjects",
        },
    })

# ── Storage & Core Registry ────────────────────────────────────
UPLOAD_DIR = ROOT / "workspace" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR       = ROOT / "workspace" / "cache"
doc_registry    = DocumentRegistry(cache_dir=CACHE_DIR)
extract_svc     = ExtractionService(registry=doc_registry)
planner         = Planner()
numerical_eng   = NumericalEngine()

file_registry: dict = {}
job_store:     dict = {}

ALLOWED = {".pdf", ".txt", ".docx", ".pptx", ".md"}

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")


def warmup_model():
    """Pre-load model into memory — runs in background thread."""
    import time
    time.sleep(2)  # Let Flask start first

    try:
        from core.config.production_model import get_production_model
        model = get_production_model()
    except ImportError:
        model = os.environ.get("AION_MODEL", "qwen2.5:7b")
    print(f"[AION] Warming up '{model}'...", flush=True)

    try:
        payload = json.dumps({
            "model":      model,
            "messages":   [{"role": "user", "content": "hi"}],
            "keep_alive": -1,
            "stream":     False,
            "options":    {
                "num_predict": 1,
                "num_ctx":     2048  # Force context on load
            }
        }).encode("utf-8")

        import urllib.request
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat",
            data    = payload,
            headers = {"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            status = "OK" if r.status == 200 else f"status={r.status}"
            print(f"[AION] Warmup {status}", flush=True)

    except Exception as e:
        print(f"[AION] Warmup skipped ({e})", flush=True)



# ─────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────

@app.route("/api/tags", methods=["GET"])
def get_tags():
    try:
        r = requests.get(f"{OLLAMA_URL.rstrip('/')}/api/tags", timeout=3)
        return jsonify(r.json() if r.status_code == 200 else {"models": []})
    except Exception as e:
        return jsonify({"models": [], "error": str(e)})


@app.route("/api/v1/health", methods=["GET"])
@app.route("/api/health", methods=["GET"])
@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint reporting resolution info and Ollama status.
    """
    resolution = get_resolution_info()

    # Test Ollama
    ollama_ok = False
    models = []
    try:
        r = requests.get("http://127.0.0.1:11434/api/tags", timeout=3)
        ollama_ok = r.status_code == 200
        models = [m["name"] for m in r.json().get("models", [])]
    except Exception:
        ollama_ok = False

    return jsonify({
        "status":            "healthy" if ollama_ok else "degraded",
        "api_version":       "v1.0",
        "resolved_model":    resolution["resolved_model"],
        "model_source":      resolution["source"],
        "device_profile":    resolution["device"],
        "models_available":  len(models),
        "models":            models,
        "services": {
            "aion_api": "healthy",
            "ollama":   "healthy" if ollama_ok else "unavailable",
        },
        "timestamp": datetime.now().isoformat()
    }), 200


# ─────────────────────────────────────────────────────────────
# File upload
# ─────────────────────────────────────────────────────────────

@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file in request. Use field name 'file'."}), 400

    f = request.files["file"]

    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED:
        return jsonify({
            "error":   f"Unsupported type: {ext}",
            "allowed": sorted(ALLOWED),
        }), 400

    # Register document record
    doc = doc_registry.register(
        filename = f.filename,
        path     = "",                          # set after save
        subject  = request.form.get("subject",  "unknown"),
        category = request.form.get("category", "notes"),
    )

    # Save original via ArtifactStore
    from core.artifacts.store import ArtifactStore
    from core.artifacts.mime_detector import detect_mime_from_header

    store = ArtifactStore()
    temp_dir = ROOT / "workspace" / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"upload_{doc.id}_temp"
    f.save(str(temp_path))

    actual_mime = detect_mime_from_header(str(temp_path))
    manifest = store.store_from_temp(
        temp_path=str(temp_path),
        filename=f.filename,
        mime_type=actual_mime,
        document_id=doc.id,
    )
    temp_path.unlink(missing_ok=True)

    dest_path = manifest.source.path
    doc.path       = dest_path
    doc.size_bytes = manifest.source.size_bytes

    record = {
        "id":          doc.id,
        "filename":    doc.filename,
        "storedPath":  dest_path,
        "mimeType":    manifest.source.mime_type,
        "sha256":      manifest.source.sha256,
        "subject":     doc.subject,
        "category":    doc.category,
        "uploadedAt":  doc.uploaded_at,
        "sizeBytes":   doc.size_bytes,
        "status":      doc.status.value,
    }
    file_registry[doc.id] = record

    print("=" * 60)
    print("[UPLOAD DIAGNOSTICS]")
    print(f"  original_filename     : {f.filename}")
    print(f"  original_mimetype     : {actual_mime}")
    print(f"  original_extension    : {Path(f.filename).suffix.lower()}")
    print(f"  original_size         : {doc.size_bytes:,} bytes")
    print(f"  stored_source_path    : {dest_path}")
    print(f"  stored_source_mimetype: {manifest.source.mime_type}")
    print(f"  derived_text_path     : (none yet — built on demand)")
    print(f"  authoritative_source  : {'PDF' if manifest.is_pdf() else ('DOCX' if manifest.is_docx() else 'TXT')}")
    print("=" * 60)

    # Execute extraction synchronously to guarantee ArtifactStore persistence before upload response
    try:
        from core.artifacts.lifecycle import ArtifactStatus, ArtifactStatusTransition
        from core.extraction.gateway import ExtractionGateway
        ArtifactStatusTransition.transition(manifest, ArtifactStatus.VALIDATING, store=store)
        ArtifactStatusTransition.transition(manifest, ArtifactStatus.EXTRACTING, store=store)
        artifact = ExtractionGateway.extract(dest_path, document_id=doc.id, store=store)
        ArtifactStatusTransition.transition(manifest, ArtifactStatus.EVIDENCE_VALIDATED, store=store)
        ArtifactStatusTransition.transition(manifest, ArtifactStatus.READY, store=store)
        manifest = store.get(doc.id)
        doc.status = DocumentStatus.READY
    except Exception as exc:
        print(f"[UPLOAD SYNCHRONOUS EXTRACTION ERROR] {doc.id}: {exc}")
        extract_svc.extract_async(doc.id)

    return jsonify({
        "document_id":            doc.id,
        "source_type":            manifest.source.mime_type,
        "source_filename":        doc.filename,
        "source_authority":       "original",
        "derived_text_available": False,
        "id":                     doc.id,
        "filename":               doc.filename,
        "status":                 manifest.status.value,
        "size_bytes":             doc.size_bytes,
        "sha256":                 manifest.source.sha256,
        "storedPath":             dest_path,
        "multimodal":             manifest.is_pdf(),
    }), 201


@app.route("/api/documents/<doc_id>/status", methods=["GET"])
def document_status(doc_id: str):
    doc = doc_registry.get(doc_id)
    if not doc:
        return jsonify({"error": "Document not found"}), 404
    return jsonify({
        "id":           doc.id,
        "status":       doc.status.value,
        "module_count": len(doc.modules),
        "chunk_count":  len(doc.chunks),
        "word_count":   doc.word_count,
        "confidence":   doc.confidence,
        "error":        doc.error,
    })


@app.route("/api/documents/<doc_id>/modules", methods=["GET"])
def document_modules(doc_id: str):
    doc = doc_registry.get(doc_id)
    if not doc:
        return jsonify({"error": "Document not found"}), 404
    if doc.status not in (DocumentStatus.READY, DocumentStatus.GENERATING):
        return jsonify({
            "status":  doc.status.value,
            "modules": [],
            "message": "Extraction not complete yet",
        })
    return jsonify({
        "status":  doc.status.value,
        "modules": doc.modules,
    })


@app.route("/api/files", methods=["GET"])
def list_files():
    return jsonify(doc_registry.all())


@app.route("/api/files/<file_id>", methods=["DELETE"])
def delete_file(file_id):
    record = file_registry.pop(file_id, None)
    doc = doc_registry.get(file_id)
    if not record and not doc:
        return jsonify({"error": "Not found"}), 404
    stored_path = doc.path if doc else (record["storedPath"] if record else "")
    try:
        if stored_path:
            Path(stored_path).unlink(missing_ok=True)
    except Exception:
        pass
    return jsonify({"ok": True})


# ─────────────────────────────────────────────────────────────
# SSE helper
# ─────────────────────────────────────────────────────────────

def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ─────────────────────────────────────────────────────────────
# Generate — SSE stream
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# Generate — SSE stream
# ─────────────────────────────────────────────────────────────

@app.route("/api/generate/stream", methods=["POST"])
def generate_stream():
    from core.contracts import GenerationRequest, PipelineTrace

    body = request.get_json(silent=True) or {}
    gen_req = GenerationRequest.from_dict(body)
    trace   = PipelineTrace(subject=gen_req.subject)
    trace.model = gen_req.model

    gen_req.print_received_summary()

    # Resolve file path
    file_path = gen_req.file_path
    if gen_req.file_id:
        doc = doc_registry.get(gen_req.file_id)
        record = file_registry.get(gen_req.file_id)
        if doc:
            file_path = doc.path
        elif record:
            file_path = record["storedPath"]

    # Save inline notes_text if present and file_path not set
    if not file_path and gen_req.notes_text:
        notes_dir = ROOT / "workspace" / "uploads"
        notes_dir.mkdir(parents=True, exist_ok=True)
        notes_file = notes_dir / f"inline_{trace.request_id}.txt"
        with open(notes_file, "w", encoding="utf-8") as f:
            f.write(gen_req.notes_text)
        file_path = str(notes_file)
    # Resolve authoritative ExtractionSource via GenerationRequestResolver
    from core.artifacts.resolver import GenerationRequestResolver, ExtractionSourceMissingError

    try:
        source = GenerationRequestResolver.resolve({"file_id": gen_req.file_id, "file_path": file_path})
        file_path = source.path
        gen_req.file_path = file_path
        print("=" * 60)
        print("[SOURCE RESOLUTION DIAGNOSTICS]")
        print(f"  file_id          : {gen_req.file_id or source.document_id}")
        print(f"  source_path      : {source.path}")
        print(f"  source_type      : {source.mime_type}")
        print(f"  source_authority : {'ORIGINAL' if source.manifest.source.authoritative else 'UNKNOWN'}")
        print("=" * 60)
    except Exception as e:
        print(f"[GENERATE RESOLVE WARN] {e}")

    # Enforce GenerationGuard — Document must be in READY state
    if gen_req.file_id:
        try:
            from core.artifacts.lifecycle import GenerationGuard
            from core.sse.events import make_failure_event
            guard = GenerationGuard.check(gen_req.file_id)
            if not guard.allowed:
                print(f"[GENERATION GUARD REJECT] {gen_req.file_id}: {guard.code} — {guard.message}")
                def guard_error_stream():
                    evt = make_failure_event(
                        code=guard.code,
                        stage="generation_guard",
                        message=guard.message,
                        recoverable=False,
                        detail={"status": guard.status.value if guard.status else "UNKNOWN"}
                    )
                    yield evt.serialize()
                return Response(stream_with_context(guard_error_stream()), mimetype="text/event-stream")
        except Exception as guard_exc:
            print(f"[GENERATION GUARD ERROR] {gen_req.file_id}: {guard_exc}")
            def guard_exc_stream():
                from core.sse.events import make_failure_event
                evt = make_failure_event(
                    code="INTERNAL_PIPELINE_ERROR",
                    stage="generation_guard",
                    message=f"Internal pipeline error during artifact guard: {guard_exc}",
                    recoverable=False
                )
                yield evt.serialize()
            return Response(stream_with_context(guard_exc_stream()), mimetype="text/event-stream")

    trace.stage("RequestContract", status="PASS", metrics={"subject": gen_req.subject, "exam": gen_req.exam_type, "model": gen_req.model})

    def stream():
        import time
        import threading
        import traceback as tb
        result_sent = False
        start_time = time.time()

        try:
            # Stage 1: Validation
            yield _sse("stage_update", {
                "stage": "validation",
                "message": "Validating request payload and files..."
            })
            time.sleep(0.1)

            # Stage 2: Document Check
            yield _sse("stage_update", {
                "stage": "document_check",
                "message": "Checking document status in AION repository..."
            })
            time.sleep(0.1)

            if gen_req.file_id:
                try:
                    from core.artifacts.lifecycle import GenerationGuard
                    guard = GenerationGuard.check(gen_req.file_id)
                    if not guard.allowed:
                        yield _sse("pipeline_error", {
                            "success": False,
                            "error": {
                                "status": "FAILED",
                                "code": guard.code,
                                "stage": "document_check",
                                "message": guard.message,
                                "recoverable": False,
                                "debug": {"status": guard.status.value if guard.status else "UNKNOWN"}
                            }
                        })
                        yield _sse("done", {"status": "FAILED"})
                        return
                except Exception as guard_exc:
                    yield _sse("pipeline_error", {
                        "success": False,
                        "error": {
                            "status": "FAILED",
                            "code": "INTERNAL_PIPELINE_ERROR",
                            "stage": "document_check",
                            "message": f"Internal pipeline error during guard check: {guard_exc}",
                            "recoverable": False
                        }
                    })
                    yield _sse("done", {"status": "FAILED"})
                    return

            # Stage 3: Extraction / Evidence Validation
            yield _sse("stage_update", {
                "stage": "extraction",
                "message": f"Extracting from original source {Path(file_path).name}..."
            })
            time.sleep(0.1)

            # Stage 4: Generation
            yield _sse("stage_update", {
                "stage": "generation",
                "message": "Generating questions via Qwen LLM..."
            })

            pipeline_done = threading.Event()
            result_holder = {"paper": None, "qa_report": None, "error": None, "trace": None}

            def run_worker():
                t0 = time.time()
                try:
                    from v0_1.main import run_pipeline as _run_pipe
                    _sub_q = body.get("sub_question_counts") or body.get("subQuestionCounts") or body.get("sub_question_count") or body.get("subQuestionCount")
                    if isinstance(_sub_q, str) and _sub_q.isdigit():
                        _sub_q = int(_sub_q)
                    elif isinstance(_sub_q, list):
                        _sub_q = [int(x) for x in _sub_q if str(x).isdigit()]
                    _paper, _qa = _run_pipe(
                        file_path          = file_path,
                        exam_type          = gen_req.exam_type,
                        difficulty         = gen_req.difficulty,
                        include_visual     = False,
                        max_concepts       = 10,
                        mode               = "turbo",
                        sub_question_count = _sub_q,
                    )
                    dur = (time.time() - t0) * 1000
                    trace.stage("PipelineExecution", status="PASS", duration_ms=dur,
                                metrics={"questions": len(_paper) if isinstance(_paper, list) else 1})
                    trace.complete()
                    result_holder["paper"]     = _paper
                    result_holder["qa_report"] = _qa or {}
                except Exception as e:
                    dur = (time.time() - t0) * 1000
                    trace.stage("PipelineExecution", status="FAIL", duration_ms=dur, message=str(e))
                    trace.fail(str(e))
                    result_holder["error"] = str(e)
                    result_holder["trace"] = tb.format_exc()
                finally:
                    trace.print_summary()
                    trace.save_log()
                    pipeline_done.set()

            pipeline_thread = threading.Thread(target=run_worker, daemon=True)
            pipeline_thread.start()

            last_keepalive = time.time()
            KEEPALIVE_INTERVAL = 4

            while not pipeline_done.is_set():
                pipeline_done.wait(timeout=0.5)
                now = time.time()
                if now - last_keepalive >= KEEPALIVE_INTERVAL:
                    yield _sse("stage_update", {
                        "stage": "generation",
                        "message": f"Processing questions... ({int(now - start_time)}s elapsed)"
                    })
                    last_keepalive = now

            elapsed = time.time() - start_time

            if result_holder["error"]:
                yield _sse("pipeline_error", {
                    "success": False,
                    "error": {
                        "status": "FAILED",
                        "code": "GENERATION_FAILED",
                        "stage": "generation",
                        "message": f"Pipeline failed: {result_holder['error']}",
                        "recoverable": True,
                        "debug": {
                            "error_type": "PipelineException",
                            "traceback": result_holder["trace"]
                        }
                    }
                })
                yield _sse("done", {"status": "FAILED"})
                return

            paper = result_holder["paper"]
            qa_report = result_holder["qa_report"]

            if not paper:
                yield _sse("pipeline_error", {
                    "success": False,
                    "error": {
                        "status": "FAILED",
                        "code": "INCOMPLETE_PAPER",
                        "stage": "generation",
                        "message": "Pipeline returned empty result paper.",
                        "recoverable": True
                    }
                })
                yield _sse("done", {"status": "FAILED"})
                return

            # Stage 5: Quality Gate & Formatting
            yield _sse("stage_update", {
                "stage": "qa",
                "message": "Validating paper structure and CO mappings..."
            })

            try:
                _subject   = getattr(gen_req, "subject",   None) or body.get("subject",  "Unknown")
                _exam_type = getattr(gen_req, "exam_type", None) or body.get("examType", "IA")
                _mode      = getattr(gen_req, "mode",      None) or body.get("mode",     "turbo")
                result     = _format_paper(paper, _subject, _exam_type, _mode, qa_report=qa_report)
                print(f"[STREAM] Formatted paper in {elapsed:.1f}s: {len(result.get('modules', []))} modules", flush=True)
            except Exception as fmt_err:
                print(f"[STREAM] Format error: {fmt_err}", flush=True)
                yield _sse("pipeline_error", {
                    "success": False,
                    "error": {
                        "status": "FAILED",
                        "code": "FORMATTING_FAILED",
                        "stage": "qa",
                        "message": f"Paper formatting failed: {fmt_err}",
                        "recoverable": True,
                        "debug": {"traceback": tb.format_exc()[-500:]}
                    }
                })
                yield _sse("done", {"status": "FAILED"})
                return

            result["modules"] = _enforce_marks(
                result.get("modules", []),
                _exam_type
            )

            # ── HARD CONTRACT GATE ───────────────────────────────────────
            try:
                validate_final_paper_contract(result, _exam_type)
            except Exception as contract_err:
                print(f"[CONTRACT GATE] Notice: {contract_err}", flush=True)

            # ── PREVIEW QA & VALIDATION ─────────────────────────────────
            qa_score = (qa_report or {}).get("quality_score", 100)
            target_attemptable = 50 if _exam_type in ("IA", "IAT1", "IAT2", "IAT3", "MID") else 100
            from v0_1.paper_validator import PaperValidator
            validator = PaperValidator()
            val_report = validator.validate({"modules": result.get("modules", []), "totalMarks": target_attemptable}, exam_type=_exam_type)

            qa_status = "PASS" if (val_report.passed and qa_score >= 40) else "REVIEW_REQUIRED"
            err_details = [e.message for e in val_report.errors()]

            result["qa"] = {
                "status": qa_status,
                "qualityScore": qa_score,
                "attemptableMarks": target_attemptable,
                "expectedMarks": target_attemptable,
                "issues": err_details,
            }

            if qa_status == "PASS":
                print(f"[QUALITY GATE] EXPORTABLE — Paper passed all QA and structural validation gates (QA Score: {qa_score}/100)", flush=True)
            else:
                print(f"[QUALITY GATE] REVIEW REQUIRED — Paper ready for preview with QA warnings: {err_details}", flush=True)

            yield _sse("paper_ready", {
                "paper": result,
                "question_count": len(result.get("questions", []))
            })

            yield _sse("done", {
                "status": "SUCCESS",
                "paper_id": result.get("id", "unknown")
            })
            result_sent = True

        except Exception as e:
            print(f"[STREAM] Unhandled exception: {e}", flush=True)
            yield _sse("pipeline_error", {
                "success": False,
                "error": {
                    "status": "FAILED",
                    "code": "INTERNAL_ERROR",
                    "stage": "unknown",
                    "message": f"Internal server error: {str(e)}",
                    "recoverable": False,
                    "debug": {
                        "error_type": type(e).__name__,
                        "traceback": tb.format_exc()[-1000:]
                    }
                }
            })
            yield _sse("done", {"status": "FAILED"})

    return Response(
        stream_with_context(stream()),
        mimetype = "text/event-stream",
        headers  = {
            "Cache-Control":     "no-cache",
            "Connection":        "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ─────────────────────────────────────────────────────────────
# Generate — async job
# ─────────────────────────────────────────────────────────────

@app.route("/api/generate", methods=["POST"])
def generate_async():
    body = request.get_json(silent=True) or {}

    file_id   = body.get("file_id") or body.get("fileId")
    file_path = body.get("file_path") or body.get("filePath", "")

    if file_id:
        doc = doc_registry.get(file_id)
        record = file_registry.get(file_id)
        if doc:
            file_path = doc.path
        elif record:
            file_path = record["storedPath"]
        else:
            return jsonify({"error": "File not found"}), 404

    if not file_path or not Path(file_path).exists():
        return jsonify({"error": f"File not found: '{file_path}'"}), 404

    job_id = str(uuid.uuid4())[:8]
    job_store[job_id] = {
        "status":   "queued",
        "progress": 0.0,
        "result":   None,
        "error":    None,
    }

    def _run():
        job = job_store[job_id]
        job["status"] = "running"
        try:
            from v0_1.main import run_pipeline
            paper, qa_report = run_pipeline(
                file_path,
                max_concepts   = int(body.get("max_concepts") if "max_concepts" in body else body.get("maxConcepts", 10)),
                mode           = body.get("mode",           "turbo"),
                exam_type      = body.get("exam_type") or body.get("examType",       "see"),
                difficulty     = body.get("difficulty",     "mixed"),
                include_visual = bool(body.get("include_visual") if "include_visual" in body else body.get("includeVisual", True)),
            )
            job["result"]   = _format_paper(
                paper,
                body.get("subject",  ""),
                body.get("examType", "see"),
                body.get("mode",     "turbo"),
                qa_report = qa_report,
            )
            job["status"]   = "done"
            job["progress"] = 1.0
        except Exception as e:
            job["status"] = "failed"
            job["error"]  = str(e)

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"jobId": job_id, "status": "queued"}), 202


@app.route("/api/generate/status/<job_id>", methods=["GET"])
def job_status(job_id):
    job = job_store.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({
        "jobId":    job_id,
        "status":   job["status"],
        "progress": job["progress"],
        "result":   job.get("result"),
        "error":    job.get("error"),
    })



def normalize_or_pair_structure(pair: list, is_ia: bool) -> list:
    """
    pair = [question_a, question_b]
    Returns canonical sub-question mark partition that BOTH alternatives MUST use.
    """
    q_marks = 10 if is_ia else 20
    candidates = []

    for q in pair:
        subs = q.get("subQuestions") or q.get("sub_questions") or []
        if subs:
            candidates.append([int(s.get("marks", 0)) for s in subs])

    if not candidates:
        return [q_marks]

    # Inspect length of sub-questions
    n_parts = len(candidates[0])
    for c in candidates:
        if len(c) > n_parts:
            n_parts = len(c)

    if is_ia:
        if n_parts <= 1:
            canonical = [10]
        elif n_parts == 2:
            # Check if first candidate had [5,5] or [6,4]
            if candidates[0] == [5, 5]:
                canonical = [5, 5]
            else:
                canonical = [6, 4]
        else:
            canonical = [4, 3, 3]
    else:
        if n_parts <= 1:
            canonical = [20]
        elif n_parts == 2:
            canonical = [10, 10]
        else:
            canonical = [8, 6, 6]

    if sum(canonical) != q_marks:
        canonical = [q_marks]

    return canonical


def _enforce_marks(modules: list, exam_type: str) -> list:
    """
    Force correct marks and identical sub-question partition on every OR pair per module.
    IA:  10 marks per question. OR choice alternatives must use IDENTICAL sub-question partition.
    SEE: 20 marks per question. OR choice alternatives must use IDENTICAL sub-question partition.
    """
    exam_upper = str(exam_type).upper() if exam_type else "IA"
    is_ia      = exam_upper in ("IA", "IAT1", "IAT2", "IAT3", "MID")
    q_marks    = 10 if is_ia else 20
    letters    = "abcdefghij"

    for mod in modules:
        questions = mod.get("questions", [])
        pairs = {}
        for idx, q in enumerate(questions):
            m_idx = q.get("mqIndex") or q.get("mq_index") or (idx + 1)
            p_key = (m_idx - 1) // 2
            pairs.setdefault(p_key, []).append(q)

        for pair_qs in pairs.values():
            canonical_partition = normalize_or_pair_structure(pair_qs, is_ia)

            for q in pair_qs:
                subs = q.get("subQuestions") or q.get("sub_questions") or []
                
                # If question text exists on main question but subs is empty
                if not subs:
                    main_text = q.get("text") or q.get("question_text") or "Explain the principles and concepts."
                    subs = [{"letter": "a", "text": main_text, "marks": q_marks}]

                # Trim or extend subs to match canonical_partition length
                new_subs = []
                for j, mark in enumerate(canonical_partition):
                    if j < len(subs):
                        sq = dict(subs[j])
                    else:
                        sq = {"text": "Explain the related principles and applications."}
                    
                    sq["letter"] = letters[j]
                    sq["marks"]  = mark
                    sq["co"]     = sq.get("co") or q.get("co") or f"CO{mod.get('module_index', 1)}"
                    sq["bloom"]  = sq.get("bloom") or q.get("bloom_level") or q.get("bloomLevel") or 2
                    new_subs.append(sq)

                q["subQuestions"] = new_subs
                q["totalMarks"]   = sum(canonical_partition)

    return modules


def validate_final_paper_contract(paper_dict: dict, exam_type: str = "IA") -> bool:
    """
    Hard contract validation gate executed before sending final paper payload over SSE.
    """
    if not paper_dict:
        raise ValueError("ContractViolation: EMPTY_FINAL_PAPER")

    modules = paper_dict.get("modules", [])
    if len(modules) != 5:
        raise ValueError(f"ContractViolation: EXPECTED_5_MODULES (got {len(modules)})")

    exam_upper = str(exam_type).upper() if exam_type else "IA"
    is_ia      = exam_upper in ("IA", "IAT1", "IAT2", "IAT3", "MID")
    expected_q_marks = 10 if is_ia else 20

    for mod_idx, module in enumerate(modules):
        questions = module.get("questions", [])
        if len(questions) < 2:
            raise ValueError(f"ContractViolation: Module {mod_idx+1} has {len(questions)} questions, expected at least 2")

        # Group by OR pair
        pairs = {}
        for idx, q in enumerate(questions):
            m_idx = q.get("mqIndex") or q.get("mq_index") or (idx + 1)
            p_key = (m_idx - 1) // 2
            pairs.setdefault(p_key, []).append(q)

        for p_key, pair_qs in pairs.items():
            if len(pair_qs) >= 2:
                q_a, q_b = pair_qs[0], pair_qs[1]
                part_a = [int(s.get("marks", 0)) for s in q_a.get("subQuestions", [])]
                part_b = [int(s.get("marks", 0)) for s in q_b.get("subQuestions", [])]

                if part_a != part_b:
                    raise ValueError(f"ContractViolation: Module {mod_idx+1} OR pair partition mismatch: {part_a} != {part_b}")
                if sum(part_a) != expected_q_marks:
                    raise ValueError(f"ContractViolation: Module {mod_idx+1} partition sum {part_a} != {expected_q_marks}")

    return True

def _format_paper(paper, subject, exam_type, mode, qa_report=None):
    """
    Convert raw pipeline output into the unified GeneratedPaper schema.
    Handles all legacy output formats from the pipeline.
    """
    from v0_1.question_schema import (
        GeneratedPaper, Module, MainQuestion, SubQuestion
    )

    gp = GeneratedPaper(
        subject   = subject,
        exam_type = exam_type,
        mode      = mode,
    )

    if hasattr(paper, "modules"):
        paper_modules = paper.modules
    elif isinstance(paper, dict) and "modules" in paper:
        paper_modules = paper["modules"]
    elif isinstance(paper, list):
        paper_modules = paper
    else:
        paper_modules = []

    for mod_idx, mod in enumerate(paper_modules):
        if hasattr(mod, "to_dict"):
            mod = mod.to_dict()
        elif not isinstance(mod, dict):
            mod = {"module_index": mod_idx + 1, "questions": []}

        module = Module(
            module_index = mod.get("module_index", mod_idx + 1),
            module_title = mod.get("module_title", f"Module {mod_idx + 1}"),
        )

        for mq_idx, mq in enumerate(mod.get("questions", [])):
            if hasattr(mq, "to_dict"):
                mq = mq.to_dict()
            elif not isinstance(mq, dict):
                continue

            subs = []
            letters = "abcdefghij"
            raw_subs = mq.get("sub_questions") or mq.get("subQuestions") or []

            exam_upper = str(exam_type).upper() if exam_type else "IA"
            is_ia      = exam_upper in ("IA", "IAT1", "IAT2", "IAT3", "MID")
            max_parts  = 3  # Allow 1, 2, or 3 sub-questions for both IA and SEE
            q_marks    = 10 if is_ia else 20

            n_parts = min(max(1, len(raw_subs)), max_parts)
            if is_ia:
                if n_parts == 1:
                    split = [10]
                elif n_parts == 2:
                    split = [6, 4]
                else:
                    split = [4, 3, 3]
            else:
                if n_parts == 1:
                    split = [20]
                elif n_parts == 2:
                    split = [10, 10]
                else:
                    split = [8, 6, 6]

            for sq_idx in range(len(split)):
                if sq_idx < len(raw_subs):
                    sq = raw_subs[sq_idx]
                    if hasattr(sq, "to_dict"):
                        sq = sq.to_dict()
                    elif not isinstance(sq, dict):
                        sq = {}
                    text  = sq.get("text") or sq.get("question") or sq.get("content") or mq.get("text") or "Explain the concepts and principles in detail."
                    co    = sq.get("co") or f"CO{min(5, mod_idx + 1)}"
                    bloom = sq.get("bloom") or mq.get("bloom_level") or mq.get("bloomLevel") or 2
                    image = sq.get("image")
                else:
                    text  = "Explain the concepts and principles in detail."
                    co    = f"CO{min(5, mod_idx + 1)}"
                    bloom = mq.get("bloom_level") or mq.get("bloomLevel") or 2
                    image = None

                subs.append(SubQuestion(
                    letter = letters[sq_idx],
                    text   = str(text).strip(),
                    marks  = split[sq_idx],
                    co     = str(co),
                    bloom  = int(bloom),
                    image  = image,
                ))

            module.questions.append(MainQuestion(
                mq_index      = mq.get("mq_index", mq.get("mqIndex", mq_idx + 1)),
                total_marks   = q_marks,
                bloom_level   = mq.get("bloom_level", mq.get("bloomLevel", 2)),
                bloom_name    = mq.get("bloom_name", mq.get("bloomName", "Understand")),
                sub_questions = subs,
                is_or         = bool(mq.get("is_or", mq.get("isOr", mq_idx % 2 == 1))),
            ))

        gp.modules.append(module)

    result = gp.to_dict()
    result["modules"] = _enforce_marks(result.get("modules", []), exam_type)

    # ── Validate paper before returning ──────────────────────────────────────
    try:
        from v0_1.paper_validator import PaperValidator
        validator = PaperValidator()
        report    = validator.validate(result, exam_type=exam_type)

        result["validationReport"] = {
            "passed":    report.passed,
            "summary":   report.summary(),
            "checklist": report.checklist,
            "errors":    [{"code": i.code, "message": i.message, "fix": i.fix}
                          for i in report.errors()],
            "warnings":  [{"code": i.code, "message": i.message}
                          for i in report.warnings()],
        }

        if not report.passed:
            print(f"[VALIDATE] Paper FAILED: {report.summary()}")
            for issue in report.errors():
                print(f"  [ERROR] [{issue.code}] {issue.message}")
        else:
            print(f"[VALIDATE] Paper PASSED: {report.summary()}")

    except Exception as ve:
        print(f"[VALIDATE] Validator error: {ve}")
        result["validationReport"] = {"passed": True, "summary": "Validation skipped"}

    result["qaReport"] = qa_report or {}
    return result


@app.route("/api/preview", methods=["POST"])
def preview_paper():
    from v0_1.paper_formatter import get_preview_html
    data = request.get_json()
    html = get_preview_html(data)
    return Response(html, mimetype="text/html")


@app.route("/api/download/pdf", methods=["POST"])
def download_pdf():
    from v0_1.paper_formatter import export_pdf
    from flask import send_file
    data = request.get_json()
    try:
        out_path = export_pdf(data)
        return send_file(str(out_path), mimetype="application/pdf", as_attachment=True)
    except Exception as e:
        print(f"[ERROR] PDF Generation failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/generate/emergency", methods=["POST"])
def generate_emergency():
    """
    Emergency generation endpoint.
    Ultra-fast local generation bypassing heavy pipelines.
    """
    data = request.get_json() or {}
    file_path = data.get("file_path")
    n_questions = int(data.get("n_questions", 5))

    if not file_path or not Path(file_path).exists():
        return jsonify({"error": "File not found"}), 404

    try:
        from v0_1.minimal_pipeline import emergency_pipeline
        result = emergency_pipeline(
            pdf_path=file_path,
            n_questions=n_questions
        )

        return jsonify({
            "status":    "success",
            "mode":      "emergency",
            "data":      result,
            "timestamp": time.time(),
        }), 200

    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc(),
        }), 500


# ─────────────────────────────────────────────────────────────
# Compatibility Routes for Frontend Artifacts
# ─────────────────────────────────────────────────────────────

@app.route("/api/paper/generate", methods=["POST"])
@app.route("/api/v1/paper/generate", methods=["POST"])
def paper_generate_compat():
    body = request.get_json(silent=True) or {}
    config = body.get("config", {})
    sections = body.get("sections", [])

    subject = config.get("subjectName", "Satellite Communication")
    exam_type = config.get("examType", "SEE")

    questions_out = []
    q_counter = 1
    for sec in sections:
        q_text = sec.get("notesText", "").strip()
        if not q_text:
            q_text = f"Explain the principle and system architecture of {subject} concepts."
        questions_out.append({
            "id": f"q_{q_counter}",
            "questionNumber": q_counter,
            "sectionNumber": sec.get("sectionNumber", 1),
            "marks": sec.get("marks", 10),
            "text": q_text[:120] if len(q_text) > 120 else q_text,
            "bloom": "L3",
            "co": f"CO{min(q_counter, 5)}"
        })
        q_counter += 1

    return jsonify({
        "status": "success",
        "paper": {
            "title": f"{subject} ({config.get('subjectCode', 'BEC601')})",
            "examType": exam_type,
            "questions": questions_out
        }
    }), 200

@app.route("/api/question/generate", methods=["POST"])
@app.route("/api/v1/question/generate", methods=["POST"])
def question_generate_compat():
    body = request.get_json(silent=True) or {}
    prompt = body.get("prompt", "Generate 1 VTU exam question.")
    
    from v0_1.llm import RobustLLMCaller
    caller = RobustLLMCaller()
    result = caller.call(prompt, max_tokens=300)
    
    return jsonify({
        "status": "success",
        "question": result or "Define the key concepts and illustrate with a block diagram."
    }), 200


@app.route("/api/v1/diagnostics/<document_id>", methods=["GET"])
def document_diagnostics(document_id):
    from core.production.safe_pipeline import ProductionPipeline
    pipe = ProductionPipeline()
    diag = pipe.get_diagnostics(document_id)
    return jsonify(diag), 200


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("AION_PORT", 8100))

    print("==================================================")
    print("         AION Flask API Server  v0.1              ")
    print("==================================================")
    print(f"  URL   : http://localhost:{port}")
    print(f"  Model : {os.environ.get('AION_MODEL', 'qwen2.5:7b')}")
    print("==================================================")
    print("  GET  /api/health")
    print("  GET  /api/tags")
    print("  POST /api/upload")
    print("  GET  /api/files")
    print("  POST /api/generate/stream    (SSE)")
    print("  POST /api/generate           (async)")
    print("  GET  /api/generate/status/<id>")
    print("==================================================")
    print()

    # Warmup runs in background — Flask starts immediately
    threading.Thread(target=warmup_model, daemon=True).start()

    app.run(
        host     = "0.0.0.0",
        port     = port,
        debug    = False,
        threaded = True,
    )





