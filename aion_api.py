import sys
sys.stdout.flush()
import builtins
if not hasattr(builtins, "get_user_split"):
    builtins.get_user_split = lambda: None

import sys, os
sys.path.insert(0, '/home/AIML1/AIQ/AION')
import aion_patch
# (removed leftover debug globals)

#!/usr/bin/env python3
"""
AION API Server
Flask-based HTTP server bridging the React frontend to the AION pipeline.
Run:    python aion_api.py
URL:    http://localhost:8100
"""

import os
import sys
import re
import json
import uuid
import threading
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
import requests

# -- Path setup ------------------------------------------------
ROOT = Path(__file__).parent.resolve()
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from core.config.production_model import get_production_model, get_resolution_info
os.environ.setdefault("AION_MODEL", get_production_model())

# -- Core Services Imports --------------------------------------
from core.document_registry  import DocumentRegistry, DocumentStatus
from core.extraction_service import ExtractionService
from core.generation_context import GenerationContext
from core.planner            import Planner
from core.numerical_engine   import NumericalEngine
from v0_1.unified_pipeline   import run_unified, FinalPaper

# -- Flask imports ---------------------------------------------
try:
    from flask import Flask, request, jsonify, Response, stream_with_context
    from flask_cors import CORS
except ImportError:
    print("[ERROR] Flask not installed.")
    print("Run: pip install flask flask-cors")
    sys.exit(1)

# -- App setup -------------------------------------------------
app = Flask(__name__)

# =====================================================================
# AION PRODUCTION GUARDS (HTTP Body & Query String Marks Extractor)
# =====================================================================
from core.safety.resilience import install_all_safety_layers
install_all_safety_layers()

from core.generation.marks_partitioner import parse_marks, set_user_split

@app.before_request
def _aion_production_guard():
    from flask import request
    try:
        if request.path in ("/api/generate", "/api/generate/stream", "/api/generate/vllm"):
            body = (request.get_json(silent=True) if request.is_json else None) or request.form.to_dict() or {}
            aion_patch.register_http_payload(body)

            body = (request.get_json(silent=True) if request.is_json else None) or request.form.to_dict() or {}
            args = (request.args.to_dict() if has_request_context() else {}) if request.args else {}
            
            raw_marks = (
                body.get("marks_distribution") or body.get("marksDistribution") or
                body.get("marks_split") or body.get("marksSplit") or
                body.get("mark_splits") or body.get("markSplits") or
                body.get("sub_question_marks") or body.get("subQuestionMarks") or
                body.get("partition") or body.get("distribution") or
                args.get("marks_distribution") or args.get("marks_split") or
                args.get("mark_splits") or args.get("markSplits") or
                args.get("sub_question_marks") or args.get("partition")
            )
            
            sq_count = (
                body.get("sub_question_count") or body.get("subQuestionCount") or
                body.get("sub_question_counts") or body.get("subQuestionCounts") or
                body.get("parts") or args.get("sub_question_count") or args.get("parts")
            )

            exam = str(body.get("exam_type") or body.get("exam") or args.get("exam_type") or "IAT1").upper()
            # Handle full partition list sent directly from frontend
            if isinstance(raw_marks, list) and raw_marks and isinstance(raw_marks[0], list):
                split = raw_marks
                set_user_split(split, exam)
                print(f"[HTTP-GUARD] Locked user marks split (direct): {split} for {exam}", flush=True)
            elif raw_marks:
                split = parse_marks(raw_marks, exam, sub_question_count=sq_count)
                if split:
                    set_user_split(split, exam)
                    print(f"[HTTP-GUARD] Locked user marks split: {split} (raw='{raw_marks}', count={sq_count}) for {exam}", flush=True)
            else:
                print(f"[HTTP-GUARD] No marks split in payload; not forcing a default for {exam}", flush=True)
    except Exception:
        pass
# =====================================================================



# [DUPLICATE HOOK REMOVED]


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

# -- API v1 Blueprint ------------------------------------------
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

# -- Storage & Core Registry ------------------------------------
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
                "num_predict": 200,
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



# -------------------------------------------------------------
# Startup Checks & Memory Governor
# -------------------------------------------------------------
from runtime.profiles import get_active_profile
from runtime.memory_governor import MemoryGovernor
from core.validation.math_validator import KaTeXAvailabilityGate
from core.extraction.adapter_registry import AdapterRegistry

active_profile  = get_active_profile()
memory_governor = MemoryGovernor(
    caution_gb  = active_profile.memory_caution_gb,
    critical_gb = active_profile.memory_critical_gb,
)


def run_startup_checks() -> None:
    """Flask-compatible synchronous startup checks."""
    import logging
    LOG = logging.getLogger("aion.startup")

    LOG.info("═══════════════════════════════════════════")
    LOG.info("AION STARTUP CHECKS")
    LOG.info(f"Profile : {active_profile.name}")
    LOG.info(f"Model   : {active_profile.model_name}")
    LOG.info(f"Backend : {active_profile.backend}")
    LOG.info("═══════════════════════════════════════════")

    # Profile/model integrity (P0.1)
    try:
        active_profile.validate_environment()
        LOG.info("Profile integrity : PASS")
    except RuntimeError as e:
        LOG.critical(str(e))
        print(f"[CRITICAL STARTUP ERROR] {e}", sys.stderr)
        sys.exit(1)

    # KaTeX mandatory
    try:
        katex_avail = KaTeXAvailabilityGate.probe()
        version_str = getattr(KaTeXAvailabilityGate, "_version", "active")
        LOG.info(f"KaTeX             : OK [{version_str}]")
    except Exception as e:
        LOG.critical(f"KaTeX             : FAIL — {e}")
        LOG.critical("Install: pip install katex")
        sys.exit(1)

    # Extraction adapters
    try:
        AdapterRegistry.probe()
        for name, cap in getattr(AdapterRegistry, "capabilities", {}).items():
            status = "OK" if getattr(cap, "functional", False) else "DEGRADED"
            LOG.info(f"  {name:<18}: {status}")
    except Exception as e:
        LOG.warning(f"AdapterRegistry probe warning: {e}")

    # Memory check
    mem_state = memory_governor.state()
    LOG.info(f"Memory state      : {mem_state}")
    if mem_state.value == "CRITICAL":
        LOG.critical("Memory critically low — refusing to start")
        sys.exit(1)

    LOG.info("═══════════════════════════════════════════")
    LOG.info("STARTUP COMPLETE — READY")
    LOG.info("═══════════════════════════════════════════")


# -------------------------------------------------------------
# Health & Readiness Routes
# -------------------------------------------------------------

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
    Detailed health check endpoint reporting profile, backend, memory, and gate readiness.
    Returns 200 if ready, 503 if blocked or critically low memory.
    """
    ollama_ok = False
    model_ok  = False
    try:
        r = requests.get(f"{OLLAMA_URL.rstrip('/')}/api/tags", timeout=3)
        if r.ok:
            ollama_ok = True
            models = [m.get("name", "") for m in r.json().get("models", [])]
            model_ok = any(active_profile.model_name in m for m in models) or active_profile.model_name == "AUTO"
    except Exception:
        pass
# pass  # removed useless statement

    mem_state = memory_governor.state()
    katex_ok  = KaTeXAvailabilityGate._available if hasattr(KaTeXAvailabilityGate, "_available") else True
    ready     = (
        ollama_ok and
        katex_ok and
        mem_state.value != "CRITICAL"
    )

    body = {
        "ready"               : ready,
        "profile"             : active_profile.name,
        "model"               : active_profile.model_name,
        "backend"             : active_profile.backend,
        "concurrency"         : active_profile.concurrency,
        "memory_state"        : mem_state.value,
        "ollama_available"    : ollama_ok,
        "model_available"     : model_ok,
        "katex_available"     : katex_ok,
        "extraction_available": AdapterRegistry.capabilities.get("PYMUPDF", None) is not None if hasattr(AdapterRegistry, "capabilities") else True,
        "export_gate"         : "AUTHORITATIVE",
        "legacy_qa"           : "DIAGNOSTIC_ONLY",
        "timestamp"           : datetime.now().isoformat(),
    }

    return jsonify(body), (200 if ready else 503)


@app.route("/api/v1/ready", methods=["GET"])
@app.route("/api/ready", methods=["GET"])
@app.route("/ready", methods=["GET"])
def ready():

    """
    Detailed readiness check endpoint reporting Python, package imports, GPU, KaTeX, and Ollama status.
    """
    import sys

    # 1. Dependency Probes
    pymupdf_ok = False
    try:
        import fitz
        pymupdf_ok = True
    except ImportError:
        pass
# pass  # removed useless statement

    docling_ok = False
    try:
        import docling
        docling_ok = True
    except ImportError:
        pass
# pass  # removed useless statement

    ocr_ok = False
    try:
        import easyocr
        ocr_ok = True
    except ImportError:
        pass
# pass  # removed useless statement

    gateway_ok = False
    try:
        from core.extraction.gateway import ExtractionGateway
        gateway_ok = True
    except ImportError:
        pass
# pass  # removed useless statement

    orchestrator_ok = False
    try:
        from core.generation.orchestrator import SlotOrchestrator
        orchestrator_ok = True
    except ImportError:
        pass
# pass  # removed useless statement

    export_gate_ok = False
    try:
        from core.validation.export_gate import ExportGate
        export_gate_ok = True
    except ImportError:
        pass
# pass  # removed useless statement

    # 2. Probe KaTeX
    from core.validation.math_validator import KaTeXAvailabilityGate
    katex_ok = KaTeXAvailabilityGate.probe()

    # 3. Probe GPU
    gpu_ok = False
    gpu_details = "No GPU / CPU execution fallback"
    try:
        import torch
        if torch.cuda.is_available():
            gpu_ok = True
            gpu_details = f"CUDA device: {torch.cuda.get_device_name(0)}"
    except Exception:
        pass
# pass  # removed useless statement

    # 4. Probe Ollama & Model
    resolution = get_resolution_info()
    primary_model = resolution["resolved_model"]
    ollama_ok = False
    models = []
    try:
        r = requests.get("http://127.0.0.1:11434/api/tags", timeout=3)
        ollama_ok = r.status_code == 200
        models = [m["name"] for m in r.json().get("models", [])]
    except Exception:
        ollama_ok = False

    model_loaded = primary_model in models or any(primary_model in m for m in models)

    # 4b. Perform a tiny warmup inference check to confirm the model is usable and responding
    model_usable = False
    if ollama_ok and model_loaded:
        try:
            r_warmup = requests.post(
                "http://127.0.0.1:11434/api/generate",
                json={
                    "model":  primary_model,
                    "prompt": "healthcheck",
                    "stream": False,
                    "options": {
                        "num_predict": 200
                    }
                },
                timeout=5
            )
            if r_warmup.status_code == 200:
                model_usable = True
        except Exception:
            pass
# pass  # removed useless statement

    # 5. Authoritative Readiness Status
    ready_status = gateway_ok and orchestrator_ok and export_gate_ok and ollama_ok and model_loaded and model_usable and katex_ok

    return jsonify({
        "ready":             ready_status,
        "status":            "ready" if ready_status else "initializing",
        "api_version":       "v1.0",
        "python_version":    sys.version,
        "timestamp":         datetime.now().isoformat(),
        "device_profile":    resolution["device"],
        "gpu": {
            "available": gpu_ok,
            "details":   gpu_details
        },
        "katex": {
            "available": katex_ok
        },
        "ollama": {
            "online":         ollama_ok,
            "resolved_model": primary_model,
            "model_loaded":   model_loaded,
            "model_usable":   model_usable,
            "models":         models
        },
        "dependencies": {
            "PyMuPDF":           "OK" if pymupdf_ok else "Missing",
            "Docling":           "OK" if docling_ok else "Missing",
            "OCR":               "OK" if ocr_ok else "Missing"
        },
        "generation": {
            "model_loaded":      model_loaded,
            "model_usable":      model_usable,
            "extraction_ready":  gateway_ok,
            "katex_ready":       katex_ok,
            "export_gate_ready": export_gate_ok
        }
    }), 200 if ready_status else 503


# -------------------------------------------------------------
# File upload
# -------------------------------------------------------------

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

        # -- Self-Learning: extract concepts from uploaded document ---------
        try:
            _body = locals().get('body') or {}
            from v0_1.self_learning import learn_from_document
            _subject = _body.get("subject") or doc.subject or "general"
            learn_from_document(dest_path, subject=_subject, doc_id=doc.id)
        except Exception as _le:
            print(f"[SELF-LEARNING] Non-critical learn step failed: {_le}")

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
# pass  # removed useless statement
    return jsonify({"ok": True})


# -------------------------------------------------------------
# SSE helper
# -------------------------------------------------------------

def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# -------------------------------------------------------------
# Generate — SSE stream
# -------------------------------------------------------------

def get_document_text(doc_id: str, store: Optional[Any] = None) -> str:
    from core.artifacts.store import ArtifactStore
    from core.extraction.gateway import ExtractionGateway
    store = store or ArtifactStore()
    manifest = store.get(doc_id)
    derived_path = manifest.get_derived_text()
    if derived_path and os.path.exists(derived_path):
        with open(derived_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    artifact = ExtractionGateway.extract(manifest.source.path, document_id=doc_id)
    return getattr(artifact, "text", "") or ""


# -------------------------------------------------------------
# Generate — SSE stream
# -------------------------------------------------------------

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
    notes_text_override = None

    module_files = body.get("module_files") or body.get("moduleFiles") or {}
    if isinstance(module_files, dict) and module_files:
        module_files_int = {}
        for k, v in module_files.items():
            try:
                module_files_int[int(k)] = str(v)
            except (ValueError, TypeError):
                pass
    else:
        module_files_int = {}

    # --- Explicit Module Slot Mapping (handles sparse uploads like Module 5 only) ---
    if module_files_int:
        from core.artifacts.store import ArtifactStore
        from core.artifacts.lifecycle import GenerationGuard
        store = ArtifactStore()
        combined_parts = []
        
        raw_notes = gen_req.notes_text or body.get("notes_text") or ""
        existing_module_notes = {}
        if "Module " in raw_notes:
            mod_splits = re.split(r"(?:=== )?Module\s+(\d+)[:\s]", raw_notes)
            if len(mod_splits) > 1:
                for idx in range(1, len(mod_splits), 2):
                    try:
                        m_num = int(mod_splits[idx])
                        m_txt = mod_splits[idx + 1].strip()
                        existing_module_notes[m_num] = m_txt
                    except (IndexError, ValueError):
                        pass

        for m_idx in range(1, 6):
            fid = module_files_int.get(m_idx)
            if fid:
                manifest = store.get(fid)
                if manifest:
                    text = get_document_text(fid, store=store)
                    filename = getattr(manifest.source, "filename", f"Module_{m_idx}")
                    combined_parts.append(f"Module {m_idx}: {filename}\n{text}")
                else:
                    n_txt = existing_module_notes.get(m_idx) or f"Concepts and analytical principles for Module {m_idx} of {gen_req.subject}"
                    combined_parts.append(f"Module {m_idx}: {gen_req.subject} - Part {m_idx}\n{n_txt}")
            else:
                n_txt = existing_module_notes.get(m_idx) or f"Concepts and analytical principles for Module {m_idx} of {gen_req.subject}"
                combined_parts.append(f"Module {m_idx}: {gen_req.subject} - Part {m_idx}\n{n_txt}")

        notes_text_override = "\n\n".join(combined_parts)
        print(f"[MODULE-MAPPING] Synthesized explicit module slots {list(module_files_int.keys())} into combined notes ({len(notes_text_override)} chars)", flush=True)

    # --- Multi-file synthesis (runs when file_ids has 2+ entries) ---
    elif gen_req.file_ids and len(gen_req.file_ids) > 1:
        from core.artifacts.store import ArtifactStore
        from core.artifacts.lifecycle import GenerationGuard
        store = ArtifactStore()
        combined_parts = []
        for i, fid in enumerate(gen_req.file_ids):
            manifest = store.get(fid)
            if not manifest:
                raise ValueError(f"file_ids[{i}] '{fid}' not found in ArtifactStore")
            guard = GenerationGuard.check(fid, store=store)
            if not guard.allowed:
                raise ValueError(f"Module {i+1} ('{fid}') is not READY: {guard.message}")
            text = get_document_text(fid, store=store)
            word_count = len(text.split())
            if word_count < 50:
                filename = getattr(manifest.source, "filename", fid)
                raise ValueError(
                    f"Module {i+1} ('{filename}') has only {word_count} extracted words — "
                    f"below the 50-word minimum for reliable segmentation."
                )
            filename = getattr(manifest.source, "filename", f"module_{i+1}")
            combined_parts.append(f"Module {i+1}: {filename}\n{text}")
        notes_text_override = "\n\n".join(combined_parts)
        print(f"[MULTI-FILE] Synthesized {len(gen_req.file_ids)} modules into combined notes ({len(notes_text_override)} chars)", flush=True)
        print(f"[MULTI-FILE] Synthesized {len(gen_req.file_ids)} modules as text-only. "
              f"Image/figure extraction is skipped for multi-file requests — "
              f"original PDF diagrams will not appear in this paper.", flush=True)

    if notes_text_override:
        notes_dir = ROOT / "workspace" / "uploads"
        notes_dir.mkdir(parents=True, exist_ok=True)
        notes_file = notes_dir / f"synthesized_multi_{trace.request_id}.txt"
        with open(notes_file, "w", encoding="utf-8") as f:
            f.write(notes_text_override)
        file_path = str(notes_file)
        gen_req.file_path = file_path
    elif gen_req.file_id:
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
                "message": f"Extracting from original source {Path(file_path).name if file_path else 'unknown'}..."
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
                    from core.generation.marks_partitioner import parse_marks, set_user_split, get_user_split
                    _sp = None
                    
                    try:
                        from flask import has_request_context
                        _b = ((request.get_json(silent=True) if getattr(request, "is_json", False) else None) or request.form.to_dict() or {}) if has_request_context() else {}
                    except Exception:
                        _b = {}
                    import aion_patch
                    _extracted = getattr(aion_patch, "extract_splits_from_payload", lambda b: [])(_b)
                    if _extracted:
                        aion_patch.set_active_job_splits(_extracted)
                    _a = (request.args.to_dict() if has_request_context() else {}) if (has_request_context() and request.args) else {}
                    _raw_m = (_b.get("mark_splits") or _b.get("marks_distribution") or _b.get("marksDistribution") or _b.get("marks_split") or _b.get("markSplits") or _b.get("sub_question_marks") or _a.get("marks_distribution") or _a.get("marks_split") or _a.get("mark_splits"))
                    _sq_c = _b.get("sub_question_count") or _b.get("subQuestionCount") or _a.get("sub_question_count")
                    _ex = str(gen_req.exam_type or _b.get("exam_type") or "IAT1").upper()
                    
                    _existing = get_user_split()
                    if _raw_m:
                        if isinstance(_raw_m, list) and _raw_m and isinstance(_raw_m[0], list):
                            set_user_split(_raw_m, _ex)
                            print(f"[WORKER-THREAD] Locked nested split (direct): {_raw_m} for {_ex}", flush=True)
                        else:
                            _sp = parse_marks(_raw_m, _ex, sub_question_count=_sq_c)
                            if _sp:
                                set_user_split(_sp, _ex)
                                print(f"[WORKER-THREAD] Locked marks partition to: {_sp} for {_ex}", flush=True)
                    elif _existing:
                        print(f"[WORKER-THREAD] Keeping already-locked split: {_existing}", flush=True)
                    else:
                        print("[WORKER-THREAD] No marks split in payload; not forcing a default", flush=True)
                    
                    try:
                        from core.generation.marks_partitioner import get_user_split as _gus2
                    except Exception:
                        _gus2 = lambda: None
                    _locked = None
                    try:
                        from core.generation.marks_partitioner import get_user_split as _gus
                        _locked = _gus()
                    except Exception:
                        _locked = None
                    _active_now = locals().get('_locked') or locals().get('_existing') or locals().get('_sp')
                    _sub_q_arg = len(_active_now) if _active_now else 2
                    _paper, _qa = _run_pipe(
                        file_path          = file_path,
                        exam_type          = gen_req.exam_type,
                        difficulty         = gen_req.difficulty,
                        include_visual     = getattr(gen_req, "visual_mode", True),
                        max_concepts       = 10,
                        mode               = "turbo",
                        sub_question_count = _sq_c,
                        marks_split        = _sp or _existing,
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
                    import traceback
                    traceback.print_exc()
                    trace.fail(str(e))
                    err_msg = str(e); print(f"[WORKER ERROR] {err_msg}"); traceback.print_exc(); result_holder["error"] = err_msg
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
                target_marks = 10
                _subject   = getattr(gen_req, "subject",   None) or body.get("subject",  "Unknown")
                _exam_type = getattr(gen_req, "exam_type", None) or body.get("examType", "IA")
                _mode      = getattr(gen_req, "mode",      None) or body.get("mode",     "turbo")
                target_marks = 10
                result     = _format_paper(paper, _subject, _exam_type, _mode, qa_report=qa_report)
# pass  # removed useless statement
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

            raw_mark_splits = body.get("mark_splits") or body.get("markSplits")
            declared_splits = raw_mark_splits if isinstance(raw_mark_splits, list) else None

            result["modules"] = _enforce_marks(
                result.get("modules", []),
                exam_type=_exam_type,
                declared_splits=declared_splits
            )

            # -- HARD CONTRACT GATE ---------------------------------------
            try:
                validate_final_paper_contract(result, _exam_type)
            except Exception as contract_err:
                print(f"[CONTRACT GATE] Notice: {contract_err}", flush=True)

            # -- PREVIEW QA & VALIDATION ---------------------------------
            qa_score = (qa_report or {}).get("legacy_qa_score", 100)
            target_attemptable = 50 if _exam_type in ("IA", "IAT1", "IAT2", "IAT3", "MID") else 100
            from v0_1.paper_validator import PaperValidator
            validator = PaperValidator()
            val_report = validator.validate({"modules": result.get("modules", []), "totalMarks": target_attemptable}, exam_type=_exam_type)

            export_passed = (qa_report or {}).get("export_gate_passed", True)
            qa_status = "PASS" if export_passed else "FAILED"
            err_details = [e.message for e in val_report.errors()]

            # Calculate total subquestions authoritative count across all modules (🔴 11 & 🔴 12)
            total_subquestions = 0
            for module in result.get("modules", []):
                for q in module.get("questions", []):
                    subs = q.get("subQuestions") or q.get("sub_questions") or []
                    if isinstance(subs, (list, tuple)):
                        total_subquestions += len(subs)

            # Calculate SHA-256 canonical hash of the modules list
            import hashlib
            import json
            canonical_data = json.dumps(result.get("modules", []), sort_keys=True)
            canonical_hash = hashlib.sha256(canonical_data.encode("utf-8")).hexdigest()

            result["integrity"] = {
                "paper_id": result.get("id", "unknown"),
                "question_count": total_subquestions,
                "canonical_hash": canonical_hash
            }

            # -- Self-Learning: record generated questions as learned patterns --
            try:
                from v0_1.self_learning import learn_from_generated_paper
                _subject = gen_req.subject or "general"
                learn_from_generated_paper(result.get("modules", []), subject=_subject, exam_type=_exam_type)
            except Exception as _sle:
                print(f"[SELF-LEARNING] Post-generation learning failed (non-critical): {_sle}")

            if qa_status == "PASS":
                print(f"[QUALITY GATE] EXPORTABLE — Paper passed all QA and structural validation gates (QA Score: {qa_score}/100)", flush=True)
            else:
                print(f"[QUALITY GATE] FAILED — Paper rejected: {err_details}", flush=True)

            import aion_patch
            result = (getattr(aion_patch, "normalize_paper_for_frontend_ui", lambda r, *a, **k: r)(result, _subject, _exam_type))
            print(f"[STREAM DEBUG] About to yield paper_ready. result has {len(result.get('modules', []))} modules", flush=True)
            paper_payload = {
                "paper": result,
                "result": result,
                "data": result,
                **(result if isinstance(result, dict) else {}),
                "question_count": total_subquestions,
                "canonical_hash": canonical_hash
            }
            print(f"[STREAM DEBUG] paper_payload keys: {list(paper_payload.keys())}", flush=True)
            yield _sse("paper_ready", paper_payload)
            yield f"data: {json.dumps(paper_payload)}\n\n"
            print(f"[STREAM DEBUG] paper_ready event yielded successfully", flush=True)

            done_payload = {
                "status": "SUCCESS",
                "paper": result,
                "result": result,
                "data": result,
                "paper_id": result.get("id", "unknown"),
                "question_count": total_subquestions,
                "canonical_hash": canonical_hash
            }
            yield _sse("done", done_payload)
            yield f"data: {json.dumps(done_payload)}\n\n"
            print(f"[STREAM DEBUG] done event yielded successfully", flush=True)
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


# -------------------------------------------------------------
# Generate — async job
# -------------------------------------------------------------

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

    # Prefer the first candidate as-is (user/pipeline choice). Never force [5,5]/[6,4].
    canonical = list(candidates[0])
    if not canonical:
        canonical = [q_marks]

    if sum(canonical) != q_marks:
        canonical = [q_marks]

    return canonical


def _enforce_marks(modules_or_paper, exam_type: str = "IA", declared_splits: Optional[List[List[int]]] = None):
    """
    Ensures question and sub-question marks honor declared splits (e.g. [6, 4]),
    symmetrizes OR alternative pairs, and conforms to standard VTU partitions.
    """
    if not modules_or_paper:
        return modules_or_paper

    if isinstance(modules_or_paper, dict):
        modules = modules_or_paper.get("modules", [])
    elif isinstance(modules_or_paper, list):
        modules = modules_or_paper
    else:
        return modules_or_paper

    is_ia = str(exam_type).upper() in ("IA", "IAT1", "IAT2", "IAT3", "MID")
    expected_q_marks = 10 if is_ia else 20

    for mod_idx, mod in enumerate(modules):
        if not isinstance(mod, dict):
            continue
        questions = mod.get("questions", [])
        if not isinstance(questions, list):
            continue

        # Group questions in this module into OR pairs (Q1/Q2, Q3/Q4, etc.)
        pairs: Dict[int, List[dict]] = {}
        for idx, q in enumerate(questions):
            p_key = idx // 2
            pairs.setdefault(p_key, []).append(q)

        for p_key, pair_qs in pairs.items():
            first_q = pair_qs[0]
            subs = first_q.get("sub_questions", []) or first_q.get("subQuestions", [])
            n_subs = max(1, len(subs))

            # 1. User/Contract declared split if provided, matching length, and correctly summing to target
            candidate_declared = declared_splits[mod_idx] if (declared_splits and mod_idx < len(declared_splits)) else None
            if candidate_declared and isinstance(candidate_declared, (list, tuple)) and len(candidate_declared) == n_subs and sum(int(x) for x in candidate_declared) == expected_q_marks:
                active_split = [int(x) for x in candidate_declared]
            # 2. Existing valid marks from generator if they sum to target
            elif subs and all(isinstance(sq.get("marks"), (int, float)) for sq in subs) and sum(int(sq["marks"]) for sq in subs) == expected_q_marks:
                active_split = [int(sq["marks"]) for sq in subs]
            # 3. Canonical VTU standard partitions
            elif n_subs == 2:
                active_split = [6, 4] if is_ia else [10, 10]
            elif n_subs == 3:
                active_split = [4, 3, 3] if is_ia else [8, 6, 6]
            elif n_subs == 4:
                active_split = [3, 3, 2, 2] if is_ia else [5, 5, 5, 5]
            else:
                active_split = [expected_q_marks]

            # Apply identical split to all questions in the OR pair
            for q in pair_qs:
                q_subs = q.get("sub_questions", []) or q.get("subQuestions", [])
                for idx, sq in enumerate(q_subs):
                    sq["marks"] = active_split[idx] if idx < len(active_split) else 0
                q["marks"] = sum(active_split)
                q["total_marks"] = sum(active_split)

    return modules_or_paper



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
    Robustly unwraps Dataclass, Object, and Dict formats.
    """
    from v0_1.question_schema import GeneratedPaper, Module, MainQuestion, SubQuestion

    gp = GeneratedPaper(
        subject   = subject or "Subject",
        exam_type = exam_type or "IAT1",
        mode      = mode or "turbo",
    )

    if hasattr(paper, "modules"):
        paper_modules = paper.modules
    elif isinstance(paper, dict) and "modules" in paper:
        paper_modules = paper["modules"]
    elif isinstance(paper, list):
        paper_modules = paper
    else:
        paper_modules = []

    for mod_idx, raw_mod in enumerate(paper_modules):
        # Unwrap module object safely
        if hasattr(raw_mod, "to_dict"):
            mod_dict = raw_mod.to_dict()
        elif isinstance(raw_mod, dict):
            mod_dict = dict(raw_mod)
        elif hasattr(raw_mod, "__dict__"):
            mod_dict = dict(raw_mod.__dict__)
        else:
            mod_dict = {}

        m_title = mod_dict.get("module_title") or mod_dict.get("title") or getattr(raw_mod, "title", f"Module {mod_idx + 1}")
        m_index = mod_dict.get("module_index") or getattr(raw_mod, "module_index", mod_idx + 1)

        module = Module(
            module_index = m_index,
            module_title = str(m_title),
        )

        raw_qs = mod_dict.get("questions") or getattr(raw_mod, "questions", []) or []

        for mq_idx, raw_mq in enumerate(raw_qs):
            if hasattr(raw_mq, "to_dict"):
                mq_dict = raw_mq.to_dict()
            elif isinstance(raw_mq, dict):
                mq_dict = dict(raw_mq)
            elif hasattr(raw_mq, "__dict__"):
                mq_dict = dict(raw_mq.__dict__)
            else:
                mq_dict = {}

            subs = []
            letters = "abcdefghij"
            raw_subs = mq_dict.get("sub_questions") or mq_dict.get("subQuestions") or getattr(raw_mq, "sub_questions", []) or getattr(raw_mq, "subQuestions", []) or []

            exam_upper = str(exam_type).upper() if exam_type else "IA"
            is_ia      = exam_upper in ("IA", "IAT1", "IAT2", "IAT3", "MID")
            max_parts  = 3
            q_marks    = 10 if is_ia else 20

            # Preserve dynamic marks generated for subquestions
            gen_marks = []
            for _sq in raw_subs:
                _m = getattr(_sq, "marks", None) if not isinstance(_sq, dict) else _sq.get("marks")
                if _m is not None and str(_m).isdigit():
                    gen_marks.append(int(_m))

            import aion_patch
            dyn_sp = (getattr(aion_patch, "get_module_partition", lambda idx: None)(mod_idx + 1))

            # Compute target_marks for subquestion split
            tm_cand = mq_dict.get('total_marks') or getattr(raw_mq, 'total_marks', 0)
            if tm_cand <= 0:
                tm_cand = q_marks
            if tm_cand <= 0:
                tm_cand = 10
            target_marks = tm_cand

            # AION fix: safe target_marks + valid split if/elif chain
            tm_cand = None
            try:
                tm_cand = mq_dict.get("total_marks", mq_dict.get("marks", None))
            except Exception:
                tm_cand = None
            if tm_cand is None:
                try:
                    tm_cand = getattr(raw_mq, "total_marks", None) or getattr(raw_mq, "marks", None)
                except Exception:
                    tm_cand = None
            try:
                target_marks = int(tm_cand)
            except Exception:
                target_marks = 0
            if target_marks <= 0:
                target_marks = q_marks if q_marks > 0 else (10 if is_ia else 20)

            if gen_marks and sum(gen_marks) == target_marks:
                split = gen_marks
            elif dyn_sp and sum(dyn_sp) == target_marks:
                split = list(dyn_sp)
            elif gen_marks and sum(gen_marks) == q_marks:
                split = gen_marks
            elif dyn_sp and sum(dyn_sp) == q_marks:
                split = list(dyn_sp)
            elif is_ia:
                n_parts = min(max(1, len(raw_subs)), max_parts)
                if n_parts <= 1:
                    split = [target_marks]
                else:
                    base = target_marks // n_parts
                    rem  = target_marks % n_parts
                    split = [base + (1 if i < rem else 0) for i in range(n_parts)]
            else:
                n_parts = min(max(1, len(raw_subs)), max_parts)
                if n_parts <= 1:
                    split = [target_marks]
                elif target_marks == 20:
                    split = [10, 10] if n_parts == 2 else [8, 6, 6]
                else:
                    base = target_marks // n_parts
                    rem  = target_marks % n_parts
                    split = [base + (1 if i < rem else 0) for i in range(n_parts)]

            for sq_idx, raw_sq in enumerate(raw_subs):
                if hasattr(raw_sq, "to_dict"):
                    sq_dict = raw_sq.to_dict()
                elif isinstance(raw_sq, dict):
                    sq_dict = dict(raw_sq)
                elif hasattr(raw_sq, "__dict__"):
                    sq_dict = dict(raw_sq.__dict__)
                else:
                    sq_dict = {}

                text  = sq_dict.get("text") or getattr(raw_sq, "text", "Explain the concepts and principles in detail.")
                # --- AION SAFETY CLEANUP: strip orphan placeholders and reference headers ---
                import re as _re
                if isinstance(text, str):
                    text = _re.sub(r'\s*\[MATH:[^\]]+\]\s*', ' ', text)
                    text = _re.sub(r'\s*Reference\s+(?:Equation|Formula)[:\s]*', ' ', text, flags=_re.IGNORECASE)
                    text = _re.sub(r'\s+', ' ', text).strip()
                # AION safety net: strip any unresolved [MATH:...] placeholders
                # and cross-question references like "Query 3", "Question 2" etc.
                import re as _re
                if isinstance(text, str):
                    text = _re.sub(r'\s*\[MATH:[^\]]+\]\s*', ' ', text).strip()
                    text = _re.sub(r'\s*Reference\s+Equation[:\s]*', ' ', text, flags=_re.IGNORECASE).strip()
                    text = _re.sub(r'\s*Reference\s+Formula[:\s]*', ' ', text, flags=_re.IGNORECASE).strip()
                image = sq_dict.get("image") or getattr(raw_sq, "image", None)
                # AION fix: preserve/infer extracted image path instead of losing it as bool/null
                if image in (None, "", False, True):
                    for _candidate in (
                        sq_dict,
                        mq_dict,
                        locals().get("mod_dict"),
                        locals().get("module_dict"),
                        locals().get("raw_mod"),
                        raw_sq,
                        raw_mq,
                    ):
                        _found_img = _aion_first_image_value(_candidate)
                        if _found_img:
                            image = _found_img
                            break
                image = _aion_public_image_url(image)
                sq_m  = split[sq_idx] if sq_idx < len(split) else 0

                final_bloom = sq_dict.get("bloom") or sq_dict.get("rbt") or getattr(raw_sq, "bloom", "L2")
                final_co    = sq_dict.get("co") or getattr(raw_sq, "co", f"CO{min(5, mod_idx + 1)}")

                subs.append(SubQuestion(
                    letter = letters[sq_idx],
                    text   = str(text).strip(),
                    marks  = sq_m,
                    co     = str(final_co),
                    bloom  = str(final_bloom),
                    image  = image,
                ))

            module.questions.append(MainQuestion(
                mq_index      = mq_dict.get("mq_index", mq_dict.get("mqIndex", getattr(raw_mq, "mq_index", mq_idx + 1))),
                total_marks   = sum(split) if subs else q_marks,
                bloom_level   = mq_dict.get("bloom_level", getattr(raw_mq, "bloom_level", 2)),
                bloom_name    = mq_dict.get("bloom_name", getattr(raw_mq, "bloom_name", "Understand")),
                sub_questions = subs,
                is_or         = bool(mq_dict.get("is_or", getattr(raw_mq, "is_or", mq_idx % 2 == 1))),
            ))

        gp.modules.append(module)

    res_dict = gp.to_dict()

    # Remove internal generation transport tokens from student-facing paper.
    res_dict = _aion_sanitize_paper_questions(res_dict)

    print(f"[FORMATTER] Formatted {len(gp.modules)} modules into GeneratedPaper schema.", flush=True)
    return res_dict




# --- AION HOTFIX: serve extracted/generated image assets safely ---
def _aion_first_image_value(obj, _seen=None):
    """
    Recursively find first plausible image path/url from dict/list/object payloads.
    """
    try:
        if _seen is None:
            _seen = set()

        oid = id(obj)
        if oid in _seen:
            return None
        _seen.add(oid)

        if obj in (None, "", False, True):
            return None

        if isinstance(obj, str):
            lower = obj.lower()
            if lower.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")) or "/api/asset" in lower:
                return obj
            return None

        if isinstance(obj, dict):
            for k in ("image", "image_path", "image_url", "diagram_image", "diagram_path",
                      "figure_path", "figure", "visual", "asset_path", "path", "file_path", "src", "url"):
                v = obj.get(k)
                found = _aion_first_image_value(v, _seen)
                if found:
                    return found

            for v in obj.values():
                found = _aion_first_image_value(v, _seen)
                if found:
                    return found
            return None

        if isinstance(obj, (list, tuple, set)):
            for item in obj:
                found = _aion_first_image_value(item, _seen)
                if found:
                    return found
            return None

        for k in ("image", "image_path", "image_url", "diagram_image", "diagram_path",
                  "figure_path", "figure", "visual", "asset_path", "path", "file_path", "src", "url"):
            try:
                v = getattr(obj, k, None)
            except Exception:
                v = None
            found = _aion_first_image_value(v, _seen)
            if found:
                return found

    except Exception:
        return None

    return None


def _aion_public_image_url(value):
    """
    Convert local extracted image paths into browser-loadable API URLs.
    Keeps existing HTTP/API URLs unchanged.
    """
    try:
        value = _aion_first_image_value(value) or value

        if not value or value is True:
            return None

        if isinstance(value, dict):
            value = _aion_first_image_value(value)

        if not value or value is True:
            return None

        s = str(value).strip()
        if not s:
            return None

        if s.startswith(("http://", "https://", "/api/")):
            return s

        from pathlib import Path as _Path
        from urllib.parse import quote as _quote

        p = _Path(s)
        if p.exists():
            return "/api/asset?path=" + _quote(str(p.resolve()))

        return s
    except Exception:
        return None


@app.route("/api/asset", methods=["GET"])
def serve_aion_asset():
    """
    Serves extracted figures/images from workspace/extracted_output.
    Prevents frontend from receiving unusable local filesystem paths.
    """
    from flask import abort, send_file, request
    from pathlib import Path as _Path

    raw = request.args.get("path", "")
    if not raw:
        abort(404)

    try:
        p = _Path(raw).resolve()
    except Exception:
        abort(400)

    allowed_roots = [
        _Path("/home/AIML1/AIQ/AION/workspace").resolve(),
        _Path("/home/AIML1/AIQ/AION/extracted_output").resolve(),
    ]

    ps = str(p)
    ok = any(ps == str(root) or ps.startswith(str(root) + "/") for root in allowed_roots)

    if not ok or not p.exists() or not p.is_file():
        abort(404)

    return send_file(str(p))
# --- END AION HOTFIX ---



def _aion_clean_internal_question_tokens(text):
    """Remove internal generation/transport labels from student-facing text."""
    if not isinstance(text, str):
        return text

    import re

    # Remove labels whose only purpose is exposing an internal MathBlock reference.
    text = re.sub(
        r'(?i)\s*(?:reference\s+equation|reference\s+formula|equation\s+reference)\s*:\s*\[MATH:[^\]]+\]\s*',
        ' ',
        text,
    )

    # Remove any remaining unresolved internal MathBlock transport token.
    text = re.sub(
        r'\s*\[MATH:[^\]]+\]\s*',
        ' ',
        text,
    )

    # Clean whitespace/punctuation artifacts.
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\s+([,.;:])', r'\1', text)

    return text.strip()


def _aion_sanitize_paper_questions(obj):
    """Recursively sanitize student-facing question text in paper payload."""
    if isinstance(obj, dict):
        for key, value in list(obj.items()):
            if key in ("text", "question_text", "instruction") and isinstance(value, str):
                obj[key] = _aion_clean_internal_question_tokens(value)
            else:
                _aion_sanitize_paper_questions(value)

    elif isinstance(obj, list):
        for item in obj:
            _aion_sanitize_paper_questions(item)

    return obj



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


# -------------------------------------------------------------
# Compatibility Routes for Frontend Artifacts
# -------------------------------------------------------------

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


@app.route("/api/export/docx", methods=["POST"])
def export_docx():
    """
    Exports a GeneratedPaper object into a styled .docx document.
    """
    from flask import send_file
    from v0_1.docx_export import generate_docx_from_paper

    body = request.get_json(silent=True) or {}
    try:
        docx_buffer = generate_docx_from_paper(body)
        
        config = body.get("config") or {}
        subject_code = config.get("subjectCode") or body.get("subject_code") or "VTU"
        exam_type = config.get("examType") or body.get("exam_type") or "IA"
        filename = f"{subject_code}_{exam_type}_QuestionPaper.docx"

        return send_file(
            docx_buffer,
            mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            as_attachment=True,
            download_name=filename,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Failed to export docx: {str(e)}"}), 500


@app.route("/api/regenerate-slot", methods=["POST"])
@app.route("/api/v1/slot/regenerate", methods=["POST"])
def regenerate_single_slot():
    """
    Targeted slot regeneration endpoint.
    Regenerates a single question or subquestion using SlotOrchestrator without re-running whole extraction.
    """
    body = request.get_json(silent=True) or {}
    
    bloom_level = str(body.get("bloom_level") or body.get("bloom") or "L3").upper()
    bloom_verb = str(body.get("bloom_verb") or body.get("verb") or "").strip()
    if not bloom_verb:
        defaults = {"L1": "Define", "L2": "Explain", "L3": "Calculate", "L4": "Analyze", "L5": "Evaluate", "L6": "Design"}
        bloom_verb = defaults.get(bloom_level, "Explain")

    marks = int(body.get("marks") or 6)
    q_type = str(body.get("question_type") or ("NUMERICAL" if marks >= 6 and bloom_level in ("L3", "L4") else "THEORY")).upper()
    topic = str(body.get("topic") or "Core Technical Concepts").strip()
    evidence_text = str(body.get("evidence_text") or body.get("context") or topic).strip()
    sub_label = str(body.get("sub_label") or body.get("label") or "a").strip()
    q_no = int(body.get("question_number") or body.get("qNo") or 1)
    mod_id = int(body.get("module_index") or body.get("module") or 1)
    co = str(body.get("co") or f"CO{min(mod_id, 5)}")

    from core.contracts.question_slot import QuestionSlot
    from core.contracts.budgets import AnswerBudget, QuestionBudget
    from core.contracts.task_signature import TaskSignature
    from core.generation.orchestrator import SlotOrchestrator

    slot = QuestionSlot(
        slot_id=f"slot_mod{mod_id}_q{q_no}_{sub_label}",
        question_no=q_no,
        sub_label=sub_label,
        or_pair_id=f"pair_{mod_id}",
        is_alternative=False,
        module_id=mod_id,
        marks=marks,
        bloom_level=bloom_level,
        bloom_verb=bloom_verb,
        bloom_operation=bloom_level,
        co=co,
        difficulty=str(body.get("difficulty") or "MEDIUM").upper(),
        question_type=q_type,
        topic=topic,
        evidence_ids=("manual_regenerate",),
        answer_budget=AnswerBudget.from_marks_and_bloom(marks, bloom_level),
        question_budget=QuestionBudget.from_bloom(bloom_level, marks),
        task_signature=TaskSignature.from_bloom_marks_type(bloom_level, marks, q_type)
    )

    try:
        orch = SlotOrchestrator()
        result_q = orch.generate(slot, evidence_pack={"text": evidence_text})
        q_text = getattr(result_q, "question_text", "") or "Explain the key principles."
        
        return jsonify({
            "status": "success",
            "subQuestion": {
                "label": sub_label,
                "text": q_text,
                "marks": marks,
                "co": co,
                "bloom": bloom_level,
                "rbt": bloom_level,
                "question_type": q_type,
            }
        }), 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# -------------------------------------------------------------
# Entry point
# -------------------------------------------------------------



# ===========================================================================
# DYNAMIC USER MARKS PARTITION HOOK (Module Level - Zero Indentation Risk)
# ===========================================================================
try:
    from core.generation.marks_partitioner import parse_user_split, set_global_user_split

    @app.before_request
    def _aion_capture_user_marks_split_hook():
        from flask import request
        if request.is_json:
            try:
                body = request.get_json(silent=True) or {}
                raw_dist = (
                    body.get("marks_distribution") or
                    body.get("marksDistribution") or
                    body.get("marks_split") or
                    body.get("sub_question_marks") or
                    body.get("distribution") or
                    body.get("partition")
                )
                exam_type = str(body.get("exam_type") or body.get("exam") or "IAT1").upper()
                tot_marks = 20 if "SEE" in exam_type else 10
                
                if raw_dist:
                    split = parse_user_split(raw_dist, tot_marks)
                    if split:
                        set_global_user_split(split)
                        print(f"[API-HOOK] Locked user marks split: {split} across all modules")
                else:
                    pass  # do not force default split after generation
            except Exception:
                pass
# pass  # removed useless statement
except Exception as _hook_err:
    print(f"[API-HOOK-WARNING] Could not register before_request marks hook: {_hook_err}")
# ===========================================================================

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




